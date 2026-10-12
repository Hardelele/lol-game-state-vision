"""Live Client Data API: разбор образцов Riot, состояния, события без повторов.

Сети нет: ответы подаются подставным транспортом, порт 2999 не трогается.
Фикстуры — tests/fixtures/live_client/: образцы из документации Riot
(liveclientdata_sample.json, liveclientdata_events.json) и обезличенный
журнал своей игры (own_game.jsonl — начало и смерти, own_game_end.jsonl —
конец, разрыв и таймауты; ники заменены на Player1…Player10).
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from lolstream.live_client import (
    HttpTransport,
    LiveClient,
    LiveStatus,
    Response,
    State,
    Tracker,
    api_observation,
    champion_id,
    champion_index,
    journal_transport,
    parse_allgamedata,
    parse_events,
    read_journal,
    replay,
    status_observation,
)

FIX = Path(__file__).parent / "fixtures" / "live_client"
OWN = FIX / "own_game.jsonl"
# Id Data Dragon, нужные фикстурам (в тестах кэша Data Dragon нет)
IDS = ["Annie", "Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux",
       "Twitch", "Yuumi", "Fiddlesticks", "MonkeyKing"]


def sample() -> dict:
    return json.loads((FIX / "liveclientdata_sample.json").read_text(encoding="utf-8"))


def ok(d: dict) -> Response:
    return Response(200, json.dumps(d))


def with_events(d: dict, events: list[dict], game_time: float | None = None) -> dict:
    d = json.loads(json.dumps(d))
    d["events"] = {"Events": events}
    if game_time is not None:
        d["gameData"]["gameTime"] = game_time
    return d


def ev(i: int, name: str, t: float, **rest) -> dict:
    return {"EventID": i, "EventName": name, "EventTime": t, **rest}


# --- разбор образцов Riot ---

def test_riot_sample_parses():
    snap = parse_allgamedata(sample(), champion_index(IDS))
    assert snap.game_mode == "CLASSIC" and snap.map_number == 11 and snap.game_time == 0.0
    (p,) = snap.players
    assert (p.champion, p.team, p.level, p.is_dead, p.respawn_timer) == ("Annie", "ORDER", 1,
                                                                         False, 0.0)
    assert p.riot_id == "Riot Tuxedo"          # в образце нет riotId — только summonerName
    assert p.scores.kills == 0 and p.items == ()
    assert snap.me == 0 and snap.my_player is p and snap.me_reason is None
    assert snap.active.stats["resourceType"] == "MANA"
    assert [e.name for e in snap.events] == ["GameStart"]
    d = snap.to_dict()
    assert d["source"] == "api" and d["players"][0]["champion"] == "Annie"
    json.dumps(d)                              # сериализуется как есть


def test_riot_events_sample_parses_all_types():
    raw = json.loads((FIX / "liveclientdata_events.json").read_text(encoding="utf-8"))
    events = parse_events(raw)
    assert len(events) == len(raw["Events"])
    names = {e.name for e in events}
    assert {"GameStart", "ChampionKill", "DragonKill", "BaronKill", "TurretKilled",
            "InhibKilled", "Multikill", "Ace"} <= names
    kill = next(e for e in events if e.name == "ChampionKill")
    assert kill.data["VictimName"] == "Riot Gene" and kill.data["Assisters"] == ["Player 1"]
    dragon = next(e for e in events if e.name == "DragonKill")
    assert dragon.data["DragonType"] == "Earth"


def test_champion_is_ddragon_id_not_localized_name():
    index = champion_index(IDS)
    assert champion_id("game_character_displayname_FiddleSticks", index) == ("Fiddlesticks",
                                                                             None)
    assert champion_id("game_character_displayname_MonkeyKing", index) == ("MonkeyKing", None)
    cid, why = champion_id("game_character_displayname_Locke", index)
    assert cid is None and "Locke" in why
    cid, why = champion_id("Зилеан", index)
    assert cid is None and "префикс" in why
    cid, why = champion_id("game_character_displayname_Annie", None)
    assert cid is None and "Data Dragon" in why


def test_active_player_missing_is_a_reason():
    d = sample()
    d["activePlayer"] = {"error": "Spectator mode doesn't currently support this feature"}
    snap = parse_allgamedata(d, champion_index(IDS))
    assert snap.active is None and snap.me is None and "Spectator" in snap.me_reason


# --- состояния ---

def test_closed_port_is_no_game_without_exception():
    # настоящий закрытый порт на loopback: занять и сразу освободить
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    client = LiveClient(HttpTransport(f"https://127.0.0.1:{port}/liveclientdata/",
                                      timeout=0.5))
    st = client.poll_once()
    assert st.state is State.NO_GAME and "соединения" in st.reason and st.game is None
    assert client.events() == []


def test_transport_only_for_loopback():
    with pytest.raises(ValueError):
        HttpTransport("https://example.com:2999/liveclientdata/")


def test_transport_exception_becomes_error_state():
    def boom(_):
        raise RuntimeError("сломался")
    st = LiveClient(boom).poll_once()
    assert st.state is State.ERROR and "сломался" in st.reason


@pytest.mark.parametrize("resp, word", [
    (Response(200, "<html>не json"), "не JSON"),
    (Response(200, "[1, 2]"), "ожидался объект"),
    (Response(200, json.dumps({"gameData": {"gameTime": 1.0}})), "allPlayers"),
    (Response(500, "Internal error"), "HTTP 500"),
])
def test_garbage_is_error_with_reason(resp, word):
    st, new = Tracker(IDS).observe(1.0, resp)
    assert st.state is State.ERROR and word in st.reason and st.game is None and new == []


def test_bad_field_type_is_error_with_path():
    d = sample()
    d["allPlayers"][0]["level"] = "one"
    st, _ = Tracker(IDS).observe(1.0, ok(d))
    assert st.state is State.ERROR and "allPlayers[0].level" in st.reason


def test_loading_before_game():
    riot_404 = json.dumps({"errorCode": "RESOURCE_NOT_FOUND", "httpStatus": 404,
                           "message": "Invalid URI format"})
    tr = Tracker(IDS)
    st, _ = tr.observe(1.0, Response(404, riot_404))
    assert st.state is State.LOADING and "RESOURCE_NOT_FOUND" in st.reason
    empty = sample()
    empty["allPlayers"] = []
    st, _ = tr.observe(2.0, ok(empty))
    assert st.state is State.LOADING
    st, _ = tr.observe(3.0, ok(sample()))
    assert st.state is State.IN_GAME and st.game.players[0].champion == "Annie"
    # та же ошибка посреди игры — уже не загрузка
    st, _ = tr.observe(4.0, Response(404, riot_404))
    assert st.state is State.ERROR and "посреди игры" in st.reason


def test_roster_without_events_is_loading_with_snapshot():
    tr = Tracker(IDS)
    st, new = tr.observe(1.0, ok(with_events(sample(), [], 0.07)))
    assert st.state is State.LOADING and new == []
    assert st.game.players[0].champion == "Annie"      # состав известен до старта
    # пропал до старта — игры не было, а не «конец»
    st, _ = tr.observe(2.0, Response(None, None, "нет соединения: refused"))
    assert st.state is State.NO_GAME and st.game is None


def test_events_are_not_repeated():
    base = sample()
    tr = Tracker(IDS)
    _, new = tr.observe(1.0, ok(with_events(base, [ev(0, "GameStart", 0.0)], 5.0)))
    assert [e.event_id for e in new] == [0]
    _, new = tr.observe(1.5, ok(with_events(base, [ev(0, "GameStart", 0.0)], 5.5)))
    assert new == []
    _, new = tr.observe(2.0, ok(with_events(base, [ev(0, "GameStart", 0.0),
                                                   ev(1, "MinionsSpawning", 65.0)], 66.0)))
    assert [e.name for e in new] == ["MinionsSpawning"]
    # новая игра: время назад — счёт событий заново
    _, new = tr.observe(3.0, ok(with_events(base, [ev(0, "GameStart", 0.0)], 0.5)))
    assert [e.event_id for e in new] == [0]


def test_client_queue_drains_events():
    base = sample()
    answers = iter([ok(with_events(base, [ev(0, "GameStart", 0.0)], 1.0)),
                    ok(with_events(base, [ev(0, "GameStart", 0.0)], 1.5))])
    client = LiveClient(lambda _: next(answers))
    client.poll_once()
    client.poll_once()
    assert [e.name for e in client.events()] == ["GameStart"]
    assert client.events() == []
    assert client.status().state is State.IN_GAME


def test_game_end_event():
    base = sample()
    tr = Tracker(IDS)
    tr.observe(1.0, ok(with_events(base, [ev(0, "GameStart", 0.0)], 100.0)))
    st, new = tr.observe(2.0, ok(with_events(base, [ev(0, "GameStart", 0.0),
                                                    ev(7, "GameEnd", 101.0, Result="Win")],
                                             101.0)))
    assert st.state is State.ENDED and "Win" in st.reason and st.game.game_time == 101.0
    assert [e.name for e in new] == ["GameEnd"]
    st, _ = tr.observe(3.0, Response(None, None, "нет соединения: refused"))
    assert st.state is State.ENDED and st.game is not None


def test_api_lost_after_game_is_end_after_grace():
    start = [ev(0, "GameStart", 0.0)]
    tr = Tracker(IDS, lost_after=5.0)
    tr.observe(10.0, ok(with_events(sample(), start, 300.0)))
    gone = Response(None, None, "нет соединения: refused")
    st, _ = tr.observe(11.0, gone)
    assert st.state is State.ERROR and st.game is None          # сбой, ещё не конец
    st, _ = tr.observe(13.0, gone)
    assert st.state is State.ERROR
    # вернулся — игра продолжается, события не повторяются
    st, new = tr.observe(14.0, ok(with_events(sample(), start, 304.0)))
    assert st.state is State.IN_GAME and new == []
    tr.observe(15.0, gone)
    st, _ = tr.observe(21.0, gone)
    assert st.state is State.ENDED and "пропал" in st.reason and st.game.game_time == 304.0
    assert st.game_t == 14.0


def test_thread_start_stop_without_api():
    client = LiveClient(lambda _: Response(None, None, "нет соединения: refused"),
                        period=0.01).start()
    try:
        for _ in range(200):
            if client.status().reason != "ещё не опрошен":
                break
            client._stop.wait(0.01)
    finally:
        client.stop()
    assert client.status().state is State.NO_GAME


# --- журнал ---

def test_journal_rows_and_partial_tail(tmp_path):
    body = json.dumps(sample())
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"t": 1.0, "code": "ConnectionRefusedError", "body": None}) + "\n"
                 + json.dumps({"t": 2.0, "code": 200, "body": body}) + "\n"
                 + '{"t": 3.0, "code": 2', encoding="utf-8")
    rows = read_journal(p)
    assert [r.t for r in rows] == [1.0, 2.0]
    assert rows[0].response.code is None and "Refused" in rows[0].response.error
    states = [st.state for st, _ in replay(rows, IDS)]
    assert states == [State.NO_GAME, State.IN_GAME]
    tr = journal_transport(rows)
    assert tr("allgamedata").code is None and tr("allgamedata").code == 200
    assert tr("allgamedata").error == "журнал кончился"


def test_journal_broken_middle_line_fails(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text("мусор\n" + json.dumps({"t": 1.0, "code": 200, "body": None}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        read_journal(p)


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_own_game_journal():
    rows = read_journal(OWN)
    out = list(replay(rows, IDS))
    states = [st.state for st, _ in out]
    # начало журнала — хвост загрузки: состав есть, событий нет, время стоит
    n_load = states.index(State.IN_GAME)
    assert n_load > 0 and set(states[:n_load]) == {State.LOADING}
    assert {st.game.game_time for st, _ in out[:n_load]} == {out[0][0].game.game_time}
    assert State.ERROR not in states
    games = [st.game for st, _ in out if st.game is not None]
    first = games[0]
    assert len(first.players) == 10
    assert all(p.champion is not None for p in first.players)
    assert [p.team for p in first.players] == ["ORDER"] * 5 + ["CHAOS"] * 5
    assert first.me is not None and first.my_player.riot_id.startswith("Player")
    # время матча не идёт назад, события — без повторов и по порядку
    times = [g.game_time for g in games]
    assert times == sorted(times)
    ids = [e.event_id for _, new in out for e in new]
    assert ids == sorted(set(ids)) and ids[:1] == [0]
    # смерти: у мёртвого есть таймер возрождения, счёт смертей не убывает
    for g in games:
        for p in g.players:
            assert not p.is_dead or p.respawn_timer > 0
    deaths = [sum(p.scores.deaths for p in g.players) for g in games]
    assert deaths == sorted(deaths)
    assert any(p.is_dead for g in games for p in g.players)
    kills = [e for _, new in out for e in new if e.name == "ChampionKill"]
    assert len(kills) == deaths[-1]                # каждое убийство — одна смерть в счёте


# --- наблюдение для контекста матча (#798) ---

def own_statuses():
    return [st for st, _ in replay(read_journal(OWN), IDS)]


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_api_observation_of_own_game():
    st = next(s for s in own_statuses()
              if s.game is not None and any(p.is_dead for p in s.game.players))
    obs, why = api_observation(st.game)
    assert why is None and obs.me_team == "ORDER" and obs.game_time == st.game.game_time
    assert [(p.name, p.team) for p in obs.players] == \
        [(p.champion, p.team) for p in st.game.players]
    dead = {p.champion: p.respawn_timer for p in st.game.players if p.is_dead}
    assert {p.name: p.respawn_in for p in obs.players if p.dead} == dead
    # Снимку 2 с: время матча вперёд, таймеры назад (не ниже нуля).
    old, _ = api_observation(st.game, age=2.0)
    assert old.game_time == st.game.game_time + 2.0
    assert {p.name: p.respawn_in for p in old.players if p.dead} == \
        {n: max(0.0, v - 2.0) for n, v in dead.items()}


def test_api_observation_refuses_unknown_champion_and_sorts_teams():
    d = json.loads(json.dumps(sample()))
    me = d["allPlayers"][0]
    chaos = dict(me, team="CHAOS", riotId="X#1", summonerName="X")
    d["allPlayers"] = [chaos] + [me]
    snap = parse_allgamedata(d, champion_index(IDS))
    obs, why = api_observation(snap)
    assert obs is None and "игроков 2" in why       # не десять — отказ, не полсостава
    snap = parse_allgamedata(d, champion_index(["Zilean"]))   # Annie не опознана
    obs, why = api_observation(snap)
    assert obs is None and "чемпион не опознан" in why and "Annie" in why
    # Десять разных: CHAOS в ответе первыми, в наблюдении — после ORDER.
    names = IDS[:10]
    d["allPlayers"] = [dict(me, team="CHAOS" if k < 5 else "ORDER", riotId=f"P{k}#1",
                            rawChampionName=f"game_character_displayname_{n}")
                       for k, n in enumerate(names)]
    snap = parse_allgamedata(d, champion_index(IDS))
    obs, why = api_observation(snap)
    assert why is None and obs.me_team is None      # своего игрока среди десяти нет
    assert obs.roster == tuple(names[5:] + names[:5])


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_status_observation_states():
    sts = own_statuses()
    load = next(s for s in sts if s.state is State.LOADING)
    obs, _ = status_observation(load, load.t + 3.0)
    assert obs.game_time == load.game.game_time      # на загрузке время стоит
    game = next(s for s in sts if s.state is State.IN_GAME)
    obs, _ = status_observation(game, game.t + 1.5)
    assert obs.game_time == game.game.game_time + 1.5
    ended = LiveStatus(State.ENDED, "GameEnd Win", game.t, game.game, game.t)
    obs, why = status_observation(ended, game.t)
    assert obs is None and why == "API: ended (GameEnd Win)"
    obs, why = status_observation(LiveStatus(State.NO_GAME, "нет соединения", 0.0), 0.0)
    assert obs is None and why.startswith("API: no_game")


OWN_END = FIX / "own_game_end.jsonl"


def test_own_game_end_stays_ended_on_timeouts():
    # хвост своей игры: последние снимки, GameEnd (Lose), разрыв соединения и
    # таймауты — после GameEnd API не отказывает в соединении, а молчит
    rows = read_journal(OWN_END)
    out = list(replay(rows, IDS))
    states = [st.state for st, _ in out]
    n_game = states.index(State.ENDED)
    assert n_game > 0 and set(states[:n_game]) == {State.IN_GAME}
    assert set(states[n_game:]) == {State.ENDED}            # и на таймаутах тоже
    end, new = out[n_game]
    assert "GameEnd" in end.reason and "Lose" in end.reason
    assert [e.name for e in new] == ["GameEnd"]
    errs = [(row, st) for row, (st, _) in zip(rows, out) if row.response.code is None]
    assert len(errs) >= 5
    assert "RemoteDisconnected" in errs[0][0].response.error
    assert all("timed out" in row.response.error for row, _ in errs[1:])
    last_snap_t = max(row.t for row in rows if row.response.code == 200)
    for _, st in errs:
        assert st.game is not None and st.game_t == last_snap_t
        assert "после конца игры" in st.reason
        assert status_observation(st, st.t)[0] is None     # конец — не «сейчас»
    ids = [e.event_id for _, new in out for e in new]
    assert ids == sorted(set(ids)) and ids[0] == 0          # первый снимок — все события
