"""Сервис состояния для coach (#796): снимок по контракту, отказы, хаб, HTTP и SSE.

Без игры и видеокарты: API — воспроизведение обезличенного журнала своей
игры (tests/fixtures/live_client/own_game.jsonl), зрение — записи records.py
руками, сервер — на свободном порту 127.0.0.1 с подставным провайдером.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lolstream import state_service as ss
from lolstream.live_client import LiveStatus, State, read_journal, replay
from lolstream.state_service import (FrameStatus, StateHub, StateInputs, StateServer,
                                     VisionFrame, build_snapshot, check_snapshot)
from lolvision.records import ChampCtx, DeathEvent, TrackRec, UnitRec

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


def ctx(**over) -> list[ChampCtx]:
    """Контекст матча по зрению: все живы и где-то видны, кроме переопределённых."""
    out = []
    for k, name in enumerate(ORDER + CHAOS):
        team = "ORDER" if name in ORDER else "CHAOS"
        out.append(over.get(name, ChampCtx(name, team, "remembered", 0.1 * k, 0.5, 4.0,
                                           "minimap", None, None)))
    return out


def vision(at: float, view: str = "spectator", **kw) -> VisionFrame:
    base = dict(i=7, t=123.0, at=at, view=view, scene="game", camera=(0.4, 0.6),
                spread=0.012, units=[], tracks=[], ctx=ctx())
    base.update(kw)
    return VisionFrame(**base)


def frames(state: str = "ok", reason: str | None = None, last_ok: float | None = None):
    return FrameStatus("screen", state, reason, "dxcam", None, 2.0, (1920, 1080), last_ok)


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

def test_full_snapshot_matches_schema_and_prefers_api():
    api = api_last()
    now = api.game_t + 0.5
    darius = UnitRec("champion", "CHAOS", 0.5, 0.4, 0.62, 0.9, "Darius")
    c = ctx(Darius=ChampCtx("Darius", "CHAOS", "visible", 0.7, 0.3, 0.0, "minimap", 0.62, None),
            Zilean=ChampCtx("Zilean", "ORDER", "dead", 0.2, 0.8, 9.0, "minimap", None, 9.6))
    vis = vision(now - 0.2, units=[darius, UnitRec("minion", "ORDER", 0.1, 0.2, 1.0, 0.8)],
                 tracks=[TrackRec("top-minion-1", "minion", "ORDER", "seen", 0.2, 0.3,
                                  0.1, 0.2, 1.0, 0.0)], ctx=c)
    snap = build_snapshot(StateInputs("идёт", frames(last_ok=now - 0.3), vis, api), now, seq=5)

    assert check_snapshot(snap) == []
    assert snap["schema"] == 1 and snap["seq"] == 5 and snap["error"] is None
    m = snap["match"]
    assert m["time"]["source"] == "api" and m["time"]["value"] == pytest.approx(337.35, abs=0.01)
    assert m["time"]["age"] == 0.5
    assert m["side"]["value"] == "ORDER" and m["state"]["value"] == "in_game"
    me = snap["me"]
    assert me["champion"]["value"] == "Zilean" and me["alive"]["value"] is False
    assert me["respawn_in"]["value"] == pytest.approx(9.85, abs=0.01)
    assert me["gold"]["source"] == "api" and isinstance(me["stats"]["value"], dict)

    ch = by_champ(snap)
    assert len(ch) == 10 and snap["champions"]["source"] == "api"
    assert ch["Zilean"]["relation"]["value"] == "self"
    assert ch["Amumu"]["relation"]["value"] == "ally"
    assert ch["Darius"]["relation"]["value"] == "enemy"
    # место — со зрения, давность = задержка кадра + возраст места
    pos = ch["Lux"]["position"]
    assert pos["source"] == "minimap" and pos["age"] == pytest.approx(4.2)
    assert ch["Darius"]["on_screen"]["value"] is True and ch["Darius"]["hp"]["value"] == 0.62
    assert ch["Lux"]["on_screen"]["value"] is None and ch["Lux"]["on_screen"]["reason"]
    # жив ли и таймер — из API, хотя HUD тоже знает
    assert ch["Zilean"]["alive"]["source"] == "api"
    assert ch["Zilean"]["respawn_in"]["source"] == "api"
    assert ch["Amumu"]["respawn_in"]["value"] is None
    assert [e["champion"]["value"] for e in snap["enemies"]["value"]] == CHAOS

    assert snap["camera"]["center"]["value"] == {"u": 0.4, "v": 0.6}
    assert snap["camera"]["center"]["source"] == "screen"
    assert snap["scene"]["value"] == "game"
    u = snap["units"]["value"]
    assert u[0]["team"]["value"] == "CHAOS" and u[0]["relation"]["value"] == "enemy"
    assert u[0]["relation"]["source"] == "inferred"
    assert u[1]["name"]["value"] is None and u[1]["name"]["reason"]
    assert snap["tracks"]["value"][0]["relation"]["value"] == "ally"
    assert snap["sources"]["frames"]["state"] == "ok"
    assert snap["sources"]["frames"]["last_frame_age"] == 0.3
    assert snap["sources"]["vision"]["view"] == "spectator"
    assert snap["sources"]["api"]["state"] == "in_game"
    assert snap["passport"]["value"] is None and "2C" in snap["passport"]["reason"]
    assert snap["conflicts"] == {"count": 0, "items": []}
    json.dumps(snap)                         # уходит в JSON как есть


def test_every_fact_is_value_with_source_or_null_with_reason():
    api = api_last()
    snap = build_snapshot(StateInputs("идёт", frames(), vision(api.game_t), api),
                          api.game_t + 1.0)
    seen = list(facts(snap))
    assert len(seen) > 100
    for path, f in seen:
        if f["value"] is None:
            assert f["reason"] and f["source"] is None, path
        else:
            assert f["reason"] is None and f["source"] in ss.SOURCES, path


def test_no_api_vision_only_refuses_api_fields():
    now = 1000.0
    snap = build_snapshot(StateInputs("идёт", frames(), vision(now - 1.0)), now)
    assert check_snapshot(snap) == []
    for k in ("time", "mode", "map", "side", "state"):
        assert snap["match"][k]["value"] is None
        assert snap["match"][k]["reason"] == ss.WHY_NO_API
    assert all(f["value"] is None for f in snap["me"].values())
    assert snap["sources"]["api"] == {"state": "absent", "reason": ss.WHY_NO_API,
                                      "age": None, "game_age": None}
    # состав — с HUD, а свои/враги без своей стороны не выдумываются
    assert snap["champions"]["source"] == "hud"
    ch = by_champ(snap)
    assert ch["Lux"]["team"] == {"value": "CHAOS", "source": "hud", "age": 1.0, "reason": None}
    assert ch["Lux"]["relation"]["value"] is None
    assert ch["Lux"]["level"]["reason"] == ss.WHY_NO_API
    assert snap["enemies"]["value"] is None and "сторона" in snap["enemies"]["reason"]


def test_api_without_game_says_its_state():
    now = 50.0
    st = LiveStatus(State.NO_GAME, "нет соединения: refused", now - 0.4)
    snap = build_snapshot(StateInputs(api=st), now)
    assert check_snapshot(snap) == []
    assert snap["match"]["state"] == {"value": "no_game", "source": "api", "age": 0.4,
                                      "reason": None}
    assert "no_game" in snap["match"]["time"]["reason"]
    assert "refused" in snap["match"]["time"]["reason"]
    assert snap["champions"]["value"] is None
    assert snap["sources"]["api"]["state"] == "no_game"


def test_loading_snapshot_has_roster_and_state():
    st = statuses()[0]
    assert st.state is State.LOADING and st.game is not None
    snap = build_snapshot(StateInputs(api=st), st.t)
    assert check_snapshot(snap) == []
    assert snap["match"]["state"]["value"] == "loading"
    assert len(snap["champions"]["value"]) == 10


def test_no_frame_refuses_vision_fields():
    api = api_last()
    now = api.game_t
    fr = frames("paused", "пауза: окно игры не в фокусе", last_ok=now - 12.0)
    snap = build_snapshot(StateInputs("идёт", fr, None, api), now)
    assert check_snapshot(snap) == []
    assert snap["sources"]["frames"]["state"] == "paused"
    assert snap["sources"]["frames"]["reason"] == "пауза: окно игры не в фокусе"
    assert snap["sources"]["frames"]["last_frame_age"] == 12.0
    assert snap["sources"]["vision"]["state"] == "absent"
    for f in (snap["camera"]["center"], snap["scene"], snap["units"], snap["tracks"]):
        assert f["value"] is None and f["reason"] == ss.WHY_NO_VISION
    ch = by_champ(snap)
    assert ch["Lux"]["position"]["reason"] == ss.WHY_NO_VISION
    assert ch["Lux"]["alive"]["source"] == "api"           # API на месте


def test_nothing_connected_is_all_refusals():
    snap = build_snapshot(StateInputs(), 10.0)
    assert check_snapshot(snap) == []
    assert all(f["value"] is None for _, f in facts(snap))
    assert snap["passport"]["reason"] == ss.WHY_NO_PASSPORT
    assert snap["sources"]["frames"]["state"] == "absent"


def test_passport_passes_through():
    snap = build_snapshot(StateInputs(passport={"view": "player", "api": True}), 10.0)
    assert check_snapshot(snap) == []
    assert snap["passport"]["value"] == {"view": "player", "api": True}


def test_provider_error_is_the_reason_everywhere():
    snap = build_snapshot(StateInputs(error="RuntimeError: boom"), 10.0)
    assert check_snapshot(snap) == []
    assert snap["error"] == "RuntimeError: boom"
    assert "boom" in snap["match"]["time"]["reason"]
    assert "boom" in snap["camera"]["center"]["reason"]


def test_conflicts_counted_and_api_used():
    api = api_last()
    now = api.game_t
    c = ctx(Zilean=ChampCtx("Zilean", "ORDER", "visible", 0.2, 0.8, 0.0, "minimap", 1.0, None),
            Lux=ChampCtx("Lux", "ORDER", "remembered", 0.5, 0.5, 2.0, "minimap", None, None),
            Yone=ChampCtx("Yone", "ORDER", "dead", None, None, None, None, None, 30.0))
    c.append(ChampCtx("Annie", "CHAOS", "unknown", None, None, None, None, None, None))
    snap = build_snapshot(StateInputs(vision=vision(now, ctx=c), api=api), now)
    assert check_snapshot(snap) == []
    got = {(x["field"], x["champion"]) for x in snap["conflicts"]["items"]}
    # Zilean мёртв по API, а зрение видит его; Lux в чужой команде; таймер Yone
    # расходится больше допуска; Annie нет в составе API
    assert got == {("alive", "Zilean"), ("team", "Lux"), ("respawn_in", "Yone"),
                   ("roster", None)}
    assert snap["conflicts"]["count"] == 4
    assert all(x["used"] == "api" for x in snap["conflicts"]["items"])
    ch = by_champ(snap)
    assert ch["Zilean"]["alive"] == {"value": False, "source": "api", "age": 0.0, "reason": None}
    assert ch["Lux"]["team"]["value"] == "CHAOS"
    assert "Annie" not in ch


def test_player_view_colors_are_relative():
    api = api_last()
    now = api.game_t
    units = [UnitRec("champion", "ORDER", 0.5, 0.5, 1.0, 0.9),     # цвет по раскладке зрителя
             UnitRec("champion", "ENEMY", 0.6, 0.5, 0.4, 0.9),     # цвет вида игрока
             UnitRec("minion", "SELF", 0.5, 0.6, 1.0, 0.9),
             UnitRec("minion", "PURPLE", 0.5, 0.6, 1.0, 0.9)]
    snap = build_snapshot(StateInputs(vision=vision(now, "player", units=units), api=api), now)
    assert check_snapshot(snap) == []
    u = snap["units"]["value"]
    assert u[0]["team"]["reason"] == ss.WHY_PLAYER_COLORS
    assert u[0]["relation"]["reason"] == ss.WHY_NO_RELATION
    assert u[1]["relation"] == {"value": "enemy", "source": "screen", "age": 0.0, "reason": None}
    assert u[1]["team"]["value"] == "CHAOS" and u[1]["team"]["source"] == "inferred"
    assert u[2]["relation"]["value"] == "self" and u[2]["team"]["value"] == "ORDER"
    assert u[3]["team"]["value"] is None and "вне ожидаемого" in u[3]["team"]["reason"]
    # имена на виде игрока не читаются — «на экране» не выдумывается
    assert "виде игрока" in by_champ(snap)["Lux"]["on_screen"]["reason"]


def test_unknown_view_refuses_team_by_color():
    now = 5.0
    units = [UnitRec("minion", "ORDER", 0.5, 0.5, 1.0, 0.9)]
    snap = build_snapshot(StateInputs(vision=vision(now, None, units=units)), now)
    assert check_snapshot(snap) == []
    assert "неизвестен" in snap["units"]["value"][0]["team"]["reason"]


# ---------------------------------------------------------------- схема

def test_checker_catches_contract_breaks():
    snap = build_snapshot(StateInputs(), 10.0)
    bad = json.loads(json.dumps(snap))
    bad["match"]["time"] = {"value": None, "source": None, "age": None, "reason": None}
    bad["scene"]["extra"] = 1
    bad["camera"]["center"] = {"value": {"u": 0.1, "v": 0.2}, "source": "guess", "age": 0,
                               "reason": None}
    del bad["me"]
    errs = check_snapshot(bad)
    assert any(e.startswith("$.match.time") for e in errs)
    assert any("лишнее поле extra" in e for e in errs)
    assert any(e.startswith("$.camera.center") for e in errs)
    assert any("нет поля me" in e for e in errs)


def test_schema_is_valid_json_schema_and_agrees():
    jsonschema = pytest.importorskip("jsonschema")
    schema = ss.snapshot_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    api = api_last()
    for inp in (StateInputs(), StateInputs(api=api),
                StateInputs("идёт", frames(), vision(api.game_t), api)):
        jsonschema.validate(build_snapshot(inp, api.game_t), schema,
                            cls=jsonschema.Draft202012Validator)


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
    api = api_last()
    feed = Feed([StateInputs("идёт", frames(), vision(api.game_t), api)])
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
    api = api_last()
    feed = Feed([StateInputs("идёт", frames(), vision(api.game_t), api)])
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
