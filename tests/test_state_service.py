"""Сервис состояния для coach (#796): снимок по контракту, отказы, хаб, HTTP и SSE.

Без игры и видеокарты: API — воспроизведение обезличенного журнала своей
игры (tests/fixtures/live_client/own_game.jsonl), контекст матча — настоящий
MatchState, накормленный этим API и подставной миникартой, паспорт —
настоящий Passport, сервер — на свободном порту 127.0.0.1 с подставным
провайдером.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from lolstream import state_service as ss
from lolstream.live_client import LiveStatus, State, read_journal, replay, status_observation
from lolstream.state_service import (FrameStatus, StateHub, StateInputs, StateServer,
                                     VisionFrame, build_snapshot, check_snapshot, match_view)
from lolvision.match_state import API_SLACK, MatchState
from lolvision.passport import Passport
from lolvision.records import UNSET, DeathEvent, MinimapChamp, TrackRec, UnitRec
from lolvision.sides import WHY_SPECTATOR, SideContext, assign

OWN = Path(__file__).parent / "fixtures" / "live_client" / "own_game.jsonl"
IDS = ["Annie", "Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux",
       "Twitch", "Yuumi", "Fiddlesticks", "MonkeyKing"]
ORDER = ["Zilean", "Amumu", "Yone", "Caitlyn", "Nami"]
CHAOS = ["Darius", "Diana", "Lux", "Twitch", "Yuumi"]


def statuses() -> list[LiveStatus]:
    return [s for s, _ in replay(read_journal(OWN), IDS)]


def api_last() -> LiveStatus:
    """Конец журнала: 5:37, свой — Zilean (ORDER), мёртв с таймером ~9.9 с."""
    return statuses()[-1]


def minimap(dead=("Zilean", "Yone"), **over) -> list[MinimapChamp]:
    """Миникарта: все живые видны, мёртвых нет; панелей HUD нет (alive None)."""
    out = []
    for k, name in enumerate(ORDER + CHAOS):
        team = "ORDER" if name in ORDER else "CHAOS"
        slot = f"{'LR'[k // 5]}{k % 5 + 1}"
        vis = name not in dead
        m = MinimapChamp(name, team, slot, vis, 0.1 * k if vis else None,
                         0.5 if vis else None, None, None)
        out.append(over.get(name, m))
    return out


def fed_match(upto: int | None = None, champs=None, units=None, cam=(0.4, 0.6)) -> MatchState:
    """MatchState после журнала API (до `upto`) и одного кадра с миникартой."""
    ms = MatchState()
    st = statuses()[:upto] if upto else statuses()
    for s in st:
        obs, _ = status_observation(s, s.t)
        ms.update(s.t, "none", api=obs)
    if champs is not None:
        ms.update(st[-1].t, "game", champs, units, cam)
    return ms


def passport(view: str = "spectator", roster: str | None = "api") -> Passport:
    p = Passport.build({"kind": "screen", "size": (1920, 1080)}, t=0.0,
                       assets={"unit_net": True, "names_bank": True, "minimap_model": True})
    p.set_view(view, 0.0, how="тест")
    p.set_api("in_game", True, 0.0)
    if roster is not None:
        p.set_roster(roster, 0.0)
    return p


def vision(at: float, **kw) -> VisionFrame:
    base = dict(i=7, t=123.0, at=at, scene="game", camera=(0.4, 0.6), spread=0.012,
                units=[], tracks=[])
    base.update(kw)
    return VisionFrame(**base)


def frames(state: str = "ok", reason: str | None = None, last_ok: float | None = None):
    return FrameStatus("screen", state, reason, "dxcam", None, 2.0, (1920, 1080), last_ok)


DARIUS = UnitRec("champion", "CHAOS", 0.5, 0.4, 0.62, 0.9, "Darius")


def full_inputs(view: str = "spectator", lag: float = 0.2) -> tuple[StateInputs, float]:
    """Всё подключено: API, контекст матча, кадр, паспорт. → (входы, now)."""
    api = api_last()
    units = assign([DARIUS, UnitRec("minion", "ORDER", 0.1, 0.2, 1.0, 0.8)],
                   SideContext(view, "ORDER"))
    ms = fed_match(champs=minimap(), units=units)
    at = api.t
    tracks = [TrackRec("top-minion-1", "minion", "ORDER", "seen", 0.2, 0.3, 0.1, 0.2, 1.0, 0.0)]
    inp = StateInputs("идёт", frames(last_ok=at), vision(at, units=units, tracks=tracks),
                      match_view(ms, at), api, passport(view), list(ms.events))
    return inp, at + lag


def facts(x, path="$"):
    """Все факты снимка: (путь, факт)."""
    if isinstance(x, dict):
        if set(x) == {"value", "source", "age", "reason"}:
            yield path, x
            if isinstance(x["value"], (dict, list)):
                yield from facts(x["value"], path + ".value")
            return
        for k, v in x.items():
            yield from facts(v, f"{path}.{k}")
    elif isinstance(x, list):
        for n, v in enumerate(x):
            yield from facts(v, f"{path}[{n}]")


def by_champ(snap) -> dict:
    return {e["champion"]["value"]: e for e in snap["champions"]["value"]}


# ---------------------------------------------------------------- снимок

def test_full_snapshot_matches_schema_and_reads_match_state():
    inp, now = full_inputs()
    snap = build_snapshot(inp, now, seq=5)

    assert check_snapshot(snap) == []
    assert snap["schema"] == 1 and snap["seq"] == 5 and snap["error"] is None
    m = snap["match"]
    # время матча — MatchState.game_clock() (источник api), давность — задержка
    assert m["time"]["source"] == "api" and m["time"]["value"] == pytest.approx(337.35, abs=0.01)
    assert m["time"]["age"] == 0.2
    assert m["side"]["value"] == "ORDER" and m["state"]["value"] == "in_game"
    assert m["mode"]["value"] == "CLASSIC" and m["map"]["value"] == 11
    me = snap["me"]
    assert me["champion"]["value"] == "Zilean" and me["alive"]["value"] is False
    assert me["alive"]["source"] == "api"
    assert me["respawn_in"]["value"] == pytest.approx(9.9, abs=0.05)
    assert me["gold"]["source"] == "api" and isinstance(me["stats"]["value"], dict)

    ch = by_champ(snap)
    assert len(ch) == 10 and snap["champions"]["source"] == "api"
    assert list(ch) == ORDER + CHAOS
    assert ch["Zilean"]["relation"]["value"] == "self"
    assert ch["Amumu"]["relation"]["value"] == "ally"
    assert ch["Darius"]["relation"]["value"] == "enemy"
    assert ch["Darius"]["level"]["value"] == 6 and ch["Darius"]["items"]["source"] == "api"
    # место — с миникарты через контекст матча
    assert ch["Lux"]["position"]["value"] == {"u": 0.7, "v": 0.5}
    assert ch["Lux"]["position"]["source"] == "minimap" and ch["Lux"]["position"]["age"] == 0.2
    assert ch["Lux"]["map_state"]["value"] == "visible"
    assert ch["Darius"]["on_screen"]["value"] is True
    assert ch["Darius"]["hp"] == {"value": 0.62, "source": "screen", "age": 0.2, "reason": None}
    assert ch["Lux"]["on_screen"]["value"] is None and ch["Lux"]["on_screen"]["reason"]
    assert ch["Zilean"]["alive"]["source"] == "api"
    assert ch["Zilean"]["respawn_in"]["source"] == "api"
    assert ch["Amumu"]["respawn_in"]["reason"] == "жив (api)"
    assert [e["champion"]["value"] for e in snap["enemies"]["value"]] == CHAOS

    assert snap["camera"]["center"]["value"] == {"u": 0.4, "v": 0.6}
    assert snap["scene"]["value"] == "game"
    u = snap["units"]["value"]
    assert u[0]["name"]["value"] == "Darius" and u[0]["team"] == "CHAOS"
    # свои / враги / сторона — поля записи (sides.py) как есть
    assert u[0]["relation"] == {"value": None, "source": None, "age": None,
                                "reason": WHY_SPECTATOR}
    assert u[0]["side"] == {"value": "CHAOS", "source": "screen", "age": 0.2, "reason": None}
    assert "relation_reason" not in u[0] and "side_reason" not in u[0]
    tr = snap["tracks"]["value"][0]
    assert tr["relation"]["reason"] == UNSET and tr["side"]["reason"] == UNSET
    assert snap["sources"]["match"]["roster"] == "api"
    assert snap["sources"]["match"]["api_live"] is True
    pp = snap["passport"]["value"]
    assert set(pp) == {"items", "pending", "disabled", "enabled"} and "names" in pp["enabled"]
    assert snap["conflicts"]["value"] == {"total": 0, "kinds": {}}
    json.dumps(snap)


def test_every_fact_is_value_with_source_or_null_with_reason():
    inp, now = full_inputs()
    seen = list(facts(build_snapshot(inp, now)))
    assert len(seen) > 100
    for path, f in seen:
        if f["value"] is None:
            assert f["reason"] and f["source"] is None, path
        else:
            assert f["reason"] is None and f["source"] in ss.SOURCES, path


def test_no_api_roster_from_hud_and_api_fields_refused():
    ms = MatchState()
    champs = minimap(dead=())
    for t in (1.0, 2.0):
        ms.update(t, "game", champs)
    inp = StateInputs("идёт", frames(), vision(2.0), match_view(ms, 2.0), None,
                      passport(roster="hud"))
    snap = build_snapshot(inp, 3.0)
    assert check_snapshot(snap) == []
    for k in ("mode", "map", "side", "state"):
        assert snap["match"][k]["reason"] == ss.WHY_NO_API
    assert "без времени по API" in snap["match"]["time"]["reason"]
    assert all(f["value"] is None for f in snap["me"].values())
    assert snap["sources"]["api"]["state"] == "absent"
    assert snap["champions"]["source"] == "hud"
    ch = by_champ(snap)
    assert ch["Lux"]["team"]["source"] == "hud"
    assert ch["Lux"]["relation"]["value"] is None and ch["Lux"]["level"]["reason"] == ss.WHY_NO_API
    assert snap["enemies"]["value"] is None and "сторона" in snap["enemies"]["reason"]


def test_api_without_game_says_its_state():
    now = 50.0
    st = LiveStatus(State.NO_GAME, "нет соединения: refused", now - 0.4)
    snap = build_snapshot(StateInputs(api=st, match=match_view(MatchState(), now)), now)
    assert check_snapshot(snap) == []
    assert snap["match"]["state"] == {"value": "no_game", "source": "api", "age": 0.4,
                                      "reason": None}
    assert "refused" in snap["match"]["mode"]["reason"]
    assert snap["champions"]["value"] is None and "состава нет" in snap["champions"]["reason"]
    assert snap["sources"]["api"]["state"] == "no_game"


def test_loading_snapshot_has_roster_and_state():
    ms = fed_match(upto=1)
    st = statuses()[0]
    assert st.state is State.LOADING
    snap = build_snapshot(StateInputs(api=st, match=match_view(ms, st.t)), st.t)
    assert check_snapshot(snap) == []
    assert snap["match"]["state"]["value"] == "loading"
    assert len(snap["champions"]["value"]) == 10


def test_no_frame_refuses_vision_fields():
    inp, now = full_inputs()
    fr = frames("paused", "пауза: окно игры не в фокусе", last_ok=now - 12.0)
    snap = build_snapshot(StateInputs("идёт", fr, None, inp.match, inp.api, inp.passport), now)
    assert check_snapshot(snap) == []
    assert snap["sources"]["frames"]["state"] == "paused"
    assert snap["sources"]["frames"]["last_frame_age"] == 12.0
    assert snap["sources"]["vision"]["state"] == "absent"
    for f in (snap["camera"]["center"], snap["scene"], snap["units"], snap["tracks"]):
        assert f["value"] is None and f["reason"] == ss.WHY_NO_VISION
    ch = by_champ(snap)
    assert ch["Lux"]["on_screen"]["reason"] == ss.WHY_NO_VISION
    assert ch["Lux"]["alive"]["source"] == "api"           # контекст матча на месте


def test_nothing_connected_is_all_refusals():
    snap = build_snapshot(StateInputs(), 10.0)
    assert check_snapshot(snap) == []
    assert all(f["value"] is None for _, f in facts(snap))
    assert snap["passport"]["reason"] == ss.WHY_NO_PASSPORT
    assert snap["champions"]["reason"] == ss.WHY_NO_MATCH
    assert snap["conflicts"]["reason"] == ss.WHY_NO_MATCH
    assert {s["state"] for s in snap["sources"].values()} == {"absent"}


def test_provider_error_is_the_reason_everywhere():
    snap = build_snapshot(StateInputs(error="RuntimeError: boom"), 10.0)
    assert check_snapshot(snap) == []
    assert snap["error"] == "RuntimeError: boom"
    assert "boom" in snap["match"]["mode"]["reason"]
    assert "boom" in snap["camera"]["center"]["reason"]
    assert "boom" in snap["champions"]["reason"]


# ---------------------------------------------------------------- паспорт

def test_player_view_gates_names_by_passport():
    inp, now = full_inputs(view="player")
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    why = Passport.verdict(inp.passport, "names").reason
    assert why and "наблюдателя" in why
    # имена над плашками на виде игрока не читаются: и on_screen, и name — отказ
    assert by_champ(snap)["Darius"]["on_screen"]["reason"] == why
    assert snap["units"]["value"][0]["name"]["reason"] == why
    assert {"reader": "names", "title": "имена над плашками", "reason": why} in \
        snap["passport"]["value"]["disabled"]


def test_unit_relation_and_side_are_taken_from_the_record():
    inp, now = full_inputs(view="player")
    me = replace(DARIUS, kind="minion", name=None, color="green", relation="self",
                 relation_reason=None, side="ORDER", side_reason=None)
    odd = replace(DARIUS, kind="minion", name=None, color="yellow", relation=None,
                  relation_reason="цвет вне ожидаемого: yellow", side=None,
                  side_reason="цвет вне ожидаемого: yellow")
    vis = replace(inp.vision, units=[me, odd])
    snap = build_snapshot(replace(inp, vision=vis), now)
    assert check_snapshot(snap) == []
    u = snap["units"]["value"]
    assert u[0]["relation"] == {"value": "self", "source": "screen", "age": 0.2, "reason": None}
    assert u[0]["side"]["value"] == "ORDER" and u[0]["color"] == "green"
    assert u[1]["relation"]["reason"] == "цвет вне ожидаемого: yellow"
    assert u[1]["side"]["reason"] == "цвет вне ожидаемого: yellow"


def test_without_passport_readers_are_not_trusted():
    inp, now = full_inputs()
    snap = build_snapshot(StateInputs(inp.session, inp.frames, inp.vision, inp.match, inp.api),
                          now)
    assert check_snapshot(snap) == []
    assert snap["units"]["value"] is None and ss.WHY_NO_PASSPORT in snap["units"]["reason"]
    assert ss.WHY_NO_PASSPORT in by_champ(snap)["Darius"]["on_screen"]["reason"]
    assert snap["passport"]["reason"] == ss.WHY_NO_PASSPORT
    assert snap["camera"]["center"]["value"] is not None       # камера паспорта не ждёт


# ---------------------------------------------------------------- конфликты

def test_conflicts_come_from_match_state():
    st = api_last()
    ms = fed_match()
    # панель HUD «жив» у мёртвого по API Zilean (после окна API_SLACK) и иконка
    # мёртвого Yone на миникарте
    t = st.t + API_SLACK + 0.1
    hud = minimap(dead=(), Zilean=MinimapChamp("Zilean", "ORDER", "L1", False, None, None,
                                               True, None))
    ms.update(t, "game", hud)
    snap = build_snapshot(StateInputs(api=st, match=match_view(ms, t)), t)
    assert check_snapshot(snap) == []
    c = snap["conflicts"]
    assert c["source"] == "inferred"
    assert c["value"]["kinds"] == dict(ms.conflicts)
    assert c["value"]["total"] == sum(ms.conflicts.values()) > 0
    assert c["value"]["kinds"].get("alive") == 1
    ch = by_champ(snap)
    assert ch["Zilean"]["conflicts"] == ms.champs["Zilean"].conflicts == 1
    assert ch["Zilean"]["alive"]["value"] is False and ch["Zilean"]["alive"]["source"] == "api"


# ---------------------------------------------------------------- схема

def test_checker_catches_contract_breaks():
    snap = build_snapshot(StateInputs(), 10.0)
    bad = json.loads(json.dumps(snap))
    bad["match"]["time"] = {"value": None, "source": None, "age": None, "reason": None}
    bad["scene"]["extra"] = 1
    bad["camera"]["center"] = {"value": {"u": 0.1, "v": 0.2}, "source": "guess", "age": 0,
                               "reason": None}
    bad["conflicts"] = {"value": {"total": 1, "kinds": {"alive": "one"}}, "source": "inferred",
                        "age": 0, "reason": None}
    del bad["me"]
    errs = check_snapshot(bad)
    assert any(e.startswith("$.match.time") for e in errs)
    assert any("лишнее поле extra" in e for e in errs)
    assert any(e.startswith("$.camera.center") for e in errs)
    assert any(e.startswith("$.conflicts") for e in errs)
    assert any("нет поля me" in e for e in errs)


def test_schema_is_valid_json_schema_and_agrees():
    jsonschema = pytest.importorskip("jsonschema")
    schema = ss.snapshot_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    inp, now = full_inputs()
    for i in (StateInputs(), StateInputs(api=inp.api), inp):
        jsonschema.validate(build_snapshot(i, now), schema, cls=jsonschema.Draft202012Validator)


# ---------------------------------------------------------------- без сеанса live

class FakeClient:
    def __init__(self, sts):
        self.sts, self.n = sts, 0

    def status(self) -> LiveStatus:
        s = self.sts[min(self.n, len(self.sts) - 1)]
        self.n += 1
        return s


def test_api_only_provider_feeds_its_own_match_state():
    sts = statuses()
    clock = iter(s.t for s in sts)
    provide = ss.api_only_provider(FakeClient(sts), clock=lambda: next(clock))
    for _ in range(len(sts) - 1):
        provide()
    inp = provide()
    snap = build_snapshot(inp, sts[-1].t)
    assert check_snapshot(snap) == []
    ch = by_champ(snap)
    assert len(ch) == 10 and ch["Zilean"]["alive"]["value"] is False
    assert snap["match"]["time"]["source"] == "api"
    assert snap["units"]["reason"] == ss.WHY_NO_VISION


# ---------------------------------------------------------------- хаб

class Feed:
    """Подставной провайдер: отдаёт заранее заданные входы по очереди."""

    def __init__(self, items):
        self.items, self.n = list(items), 0

    def __call__(self) -> StateInputs:
        item = self.items[min(self.n, len(self.items) - 1)]
        self.n += 1
        if isinstance(item, Exception):
            raise item
        return item


def drain(sub) -> list[tuple[int, str, dict]]:
    out = []
    while not sub.q.empty():
        out.append(sub.q.get_nowait())
    return out


def test_hub_publishes_new_events_once():
    st = statuses()
    clock = iter(float(x) for x in range(100, 200))
    death = DeathEvent(110.0, "Yone", "ORDER", 0.3, 0.4, 20.0)
    feed = Feed([StateInputs(api=st[20]),
                 StateInputs(api=st[20], vision_events=[death]),
                 StateInputs(api=st[-1], vision_events=[death]),
                 RuntimeError("провайдер упал")])
    hub = StateHub(feed, state_every=10.0, clock=lambda: next(clock))
    sub, hello = hub.subscribe()                 # первый такт — внутри subscribe
    assert hello["seq"] == 1
    hub.tick()
    hub.tick()
    snap = hub.tick()
    kinds = [k for _, k, _ in drain(sub)]
    game_before = {e.event_id for e in st[20].game.events}
    game_after = {e.event_id for e in st[-1].game.events}
    assert kinds.count("vision") == 1
    assert kinds.count("game") == len(game_after - game_before)
    assert kinds.count("source") == 1            # провайдер упал: api in_game → absent
    assert snap["error"] == "RuntimeError: провайдер упал"
    assert check_snapshot(snap) == []


def test_hub_restarts_event_ids_on_new_game():
    st = statuses()
    feed = Feed([StateInputs(api=st[-1]), StateInputs(api=st[5])])
    hub = StateHub(feed, state_every=100.0, clock=iter([1.0, 2.0]).__next__)
    sub, _ = hub.subscribe()
    first = [k for _, k, _ in drain(sub)]        # hello-такт уже прошёл
    assert first == []
    hub.tick()                                   # время матча ушло назад — новая игра
    got = [d["data"]["event_id"] for _, k, d in drain(sub) if k == "game"]
    assert got == [e.event_id for e in st[5].game.events]


def test_hub_drops_slow_subscriber():
    hub = StateHub(lambda: StateInputs(), state_every=0.0, queue_size=2,
                   clock=iter(float(x) for x in range(100)).__next__)
    sub, _ = hub.subscribe()
    for _ in range(3):
        hub.tick()
    assert sub.dropped and hub.health()["subscribers"] == 0


# ---------------------------------------------------------------- HTTP

def get(url: str) -> tuple[int, dict]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def read_sse(resp, want: set[str], limit: int = 50) -> list[tuple[str, dict]]:
    """Сообщения SSE до первого события каждого вида из `want`."""
    got, kind, data = [], None, []
    while want - {k for k, _ in got} and len(got) < limit:
        line = resp.readline().decode("utf-8").rstrip("\n")
        if line.startswith("event: "):
            kind = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif line == "" and kind is not None:
            got.append((kind, json.loads("".join(data))))
            kind, data = None, []
    return got


def test_server_state_health_and_events():
    feed = Feed([full_inputs()[0]])
    srv = StateServer(feed, port=0, period=0.05, state_every=0.05, keepalive=0.2).start()
    try:
        assert srv.url.startswith("http://127.0.0.1:")
        code, snap = get(srv.url + "/state")
        assert code == 200 and check_snapshot(snap) == []
        assert snap["match"]["side"]["value"] == "ORDER"
        code, health = get(srv.url + "/health")
        assert code == 200 and health["ok"] and health["schema"] == 1
        code, err = get(srv.url + "/nope")
        assert code == 404 and "/state" in err["paths"]

        host, port = srv.httpd.server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/event-stream")
        got = read_sse(resp, {"hello", "state"})
        kinds = [k for k, _ in got]
        assert kinds[0] == "hello" and "state" in kinds
        hello = got[0][1]
        assert hello["schema"] == 1 and check_snapshot(hello["data"]) == []
        state = next(d for k, d in got if k == "state")
        assert check_snapshot(state["data"]) == []
        conn.close()
    finally:
        srv.close()


def load_example():
    path = Path(__file__).resolve().parent.parent / "tools" / "state_client_example.py"
    spec = importlib.util.spec_from_file_location("state_client_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_client_reads_state_and_stream():
    ex = load_example()
    feed = Feed([full_inputs()[0]])
    srv = StateServer(feed, port=0, period=0.05, keepalive=0.2).start()
    try:
        snap = ex.get_state(srv.url)
        got = ex.to_source_facts(snap)
        ids = {f["id"] for f in got}
        assert {"match.time", "match.side", "me.champion", "me.respawn"} <= ids
        assert sum(i.startswith("enemy.") for i in ids) == 5
        assert all(f["source"] in ss.SOURCES for f in got)
        kind, msg = next(ex.events(srv.url, timeout=5))
        assert kind == "hello" and msg["data"]["schema"] == 1
    finally:
        srv.close()
    # без API фактов о матче нет — отказ не становится утверждением
    assert ex.to_source_facts(build_snapshot(StateInputs(), 1.0)) == []


def test_server_refuses_non_loopback():
    with pytest.raises(ValueError):
        StateServer(lambda: StateInputs(), host="0.0.0.0", port=0)
