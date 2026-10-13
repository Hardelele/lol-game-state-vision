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
import itertools
import json
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from lolstream import state_service as ss
from lolstream.live_client import (EventRecord, LiveStatus, RuneRecord, SpellRecord, State,
                                   read_journal, replay, status_observation)
from lolstream.state_service import (FrameStatus, StateHub, StateInputs, StateServer,
                                     VisionFrame, build_snapshot, check_snapshot, match_view)
from lolvision.match_state import API_SLACK, MatchState
from lolvision.passport import Passport
from lolvision.records import UNSET, DeathEvent, MinimapChamp, TrackRec, UnitRec
from lolvision.sides import WHY_SPECTATOR, SideContext, assign, decide

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
    assert ch["Lux"]["spells"]["reason"] == ss.WHY_NO_API
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


def test_off_rift_camera_is_from_minimap_and_says_why_when_absent():
    """Не Ущелье (#803): центр камеры — рамка на миникарте, источник minimap;
    нет её — отказ со своей причиной, треки — тоже с причиной."""
    inp, now = full_inputs()
    vis = vision(inp.vision.at, camera=(0.55, 0.45), spread=None, tracks=None,
                 camera_src="minimap", tracks_why="карта 12 — не Ущелье: треки в единицах Ущелья")
    snap = build_snapshot(replace(inp, vision=vis), now)
    assert check_snapshot(snap) == []
    assert snap["camera"]["center"]["value"] == {"u": 0.55, "v": 0.45}
    assert snap["camera"]["center"]["source"] == "minimap"
    assert "только у модели камеры" in snap["camera"]["spread"]["reason"]
    assert snap["tracks"]["reason"].startswith("карта 12")
    vis = replace(vis, camera=None, camera_why="карта 12 — не Ущелье: рамки нет")
    snap = build_snapshot(replace(inp, vision=vis), now)
    assert snap["camera"]["center"]["reason"] == "карта 12 — не Ущелье: рамки нет"


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


def self_inputs(units, alive_me: bool = True) -> tuple[StateInputs, float]:
    """Вид игрока, свой (Zilean, ORDER) — `alive_me`: журнал до момента, где
    он жив (`upto=20`), или весь (в конце мёртв); юниты кадра — после
    sides.assign и в контекст матча, и в кадр."""
    api = statuses()[19] if alive_me else api_last()
    ctx = SideContext("player", "ORDER")
    # цвет полоски уже «пробит» (поле color), как сделал бы assign по кадру
    units = [replace(u, **{k: getattr(d, k) for k in ("relation", "relation_reason", "side",
                                                      "side_reason", "side_src")})
             for u in units for d in [decide(ctx, u.kind, u.team, u.color)]]
    ms = fed_match(upto=20 if alive_me else None, champs=minimap(dead=()), units=units)
    if alive_me:
        assert ms.champs["Zilean"].alive is None or ms.champs["Zilean"].alive.value is True
    inp = StateInputs("идёт", frames(last_ok=api.t), vision(api.t, units=units),
                      match_view(ms, api.t), api, passport("player"), list(ms.events))
    return inp, api.t + 0.2


ME_BAR = UnitRec("champion", "ORDER", 0.5, 0.5, 0.8, 0.9, color="green")
ALLY_BAR = UnitRec("champion", "ORDER", 0.3, 0.5, 0.7, 0.9, color="blue")


def test_me_position_is_the_own_champion_record():
    """#802: место своего в `me` — то же, что у его записи в `champions`."""
    for inp, now in (full_inputs(), self_inputs([ME_BAR, ALLY_BAR])):
        snap = build_snapshot(inp, now)
        assert check_snapshot(snap) == []
        me = snap["me"]["champion"]["value"]
        assert snap["me"]["position"] == by_champ(snap)[me]["position"]
    # на виде наблюдателя Zilean мёртв: места нет, причина записи
    snap = build_snapshot(*full_inputs())
    assert snap["me"]["position"]["reason"] == "место не наблюдалось"
    # без контекста матча — его причина, как у alive / respawn_in
    inp, now = full_inputs()
    snap = build_snapshot(replace(inp, match=None), now)
    assert snap["me"]["position"]["reason"] == ss.WHY_NO_MATCH == snap["me"]["alive"]["reason"]


def test_own_champion_on_screen_by_self_bar_on_player_view():
    """#802: на виде игрока имена не читаются, но своя полоска (relation
    self) видна — свой на экране; остальным по-прежнему отказ читателя names."""
    inp, now = self_inputs([ME_BAR, ALLY_BAR])
    assert [u.relation for u in inp.vision.units] == ["self", "ally"]
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    ch = by_champ(snap)
    assert ch["Zilean"]["relation"]["value"] == "self"
    assert ch["Zilean"]["on_screen"] == {"value": True, "source": "screen", "age": 0.2,
                                        "reason": None}
    why = Passport.verdict(inp.passport, "names").reason
    assert ch["Amumu"]["on_screen"]["reason"] == why
    # своей полоски нет — отказ с причиной, false не выдаётся
    inp, now = self_inputs([ALLY_BAR])
    zil = by_champ(build_snapshot(inp, now))["Zilean"]["on_screen"]
    assert zil["value"] is None and "своей полоски (self) на кадре нет" in zil["reason"]
    assert why in zil["reason"]


def test_own_champion_on_screen_refuses_many_self_and_dead():
    # две своих полоски — какая своя, неизвестно: отказ, как self_many в контексте
    other = replace(ME_BAR, x=0.7)
    inp, now = self_inputs([ME_BAR, other])
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    zil = by_champ(snap)["Zilean"]["on_screen"]
    assert zil["value"] is None and "self_many" in zil["reason"]
    assert snap["conflicts"]["value"]["kinds"].get("self_many") == 1
    # мёртвый по API свой: полоска self на кадре — не его
    inp, now = self_inputs([ME_BAR], alive_me=False)
    snap = build_snapshot(inp, now)
    zil = by_champ(snap)["Zilean"]
    assert zil["alive"]["value"] is False
    assert zil["on_screen"]["value"] is None and "мёртв" in zil["on_screen"]["reason"]
    # детектор юнитов выключен паспортом — по полоске не решаем
    inp, now = self_inputs([ME_BAR])
    no_units = replace(inp, passport=None)
    assert ss.WHY_NO_PASSPORT in by_champ(build_snapshot(no_units, now))["Zilean"][
        "on_screen"]["reason"]


def test_schema_takes_self_conflict_kinds():
    inp, now = full_inputs()
    snap = build_snapshot(inp, now)
    snap["conflicts"]["value"] = {"total": 3, "kinds": {"self_many": 1, "self_place": 2}}
    assert check_snapshot(snap) == []


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


def test_example_client_main_prints_contract_fields(monkeypatch, capsys):
    """main() примера против живого сервера: снимок в потоке читается по
    контракту (`conflicts.value.total`), а не по выдуманному полю (#796)."""
    ex = load_example()
    inp, _ = full_inputs()
    srv = StateServer(Feed([inp]), port=0, period=0.05, state_every=0.05, keepalive=0.2).start()
    real = ex.events
    streams = []

    def two(url, **kw):                          # hello и первый state — и хватит
        g = real(url, timeout=5)
        streams.append(g)
        return itertools.islice(g, 2)
    monkeypatch.setattr(ex, "events", two)
    try:
        ex.main(["--url", srv.url])
    finally:
        for g in streams:
            g.close()
        srv.close()
    lines = capsys.readouterr().out.splitlines()
    got = [json.loads(s) for s in lines if s.startswith("{")]
    assert {"match.time", "me.champion"} <= {f["id"] for f in got}
    stream = [s for s in lines if not s.startswith("{")]
    assert len(stream) == 2 and stream[0].startswith("hello seq ")
    assert stream[1].startswith("state seq ")
    assert all(s.endswith(" conflicts 0") for s in stream)


def test_example_client_event_line_follows_contract():
    ex = load_example()
    inp, now = full_inputs()
    snap = build_snapshot(inp, now, seq=3)
    assert check_snapshot(snap) == []
    snap["conflicts"]["value"] = {"total": 4, "kinds": {"alive": 3, "self_place": 1}}
    assert ex.event_line("state", {"data": snap}) == "state seq 3 conflicts 4"
    # контекста матча нет — conflicts отказ: печатается причина, а не 0
    bare = build_snapshot(StateInputs(), now, seq=1)
    assert ex.event_line("hello", {"data": bare}) == \
        f"hello seq 1 conflicts нет ({ss.WHY_NO_MATCH})"
    assert ex.event_line("game", {"data": {"event_id": 1}}) == 'game {"event_id": 1}'


def test_server_refuses_non_loopback():
    with pytest.raises(ValueError):
        StateServer(lambda: StateInputs(), host="0.0.0.0", port=0)


def test_refused_api_snapshot_is_the_reason():
    """Practice Tool: в снимке API один игрок — status_observation его
    отклоняет, в MatchState он не доходит. Причина в /state — этот отказ, а
    не «API пропал» и не «контекст матча ещё не знает состава» (#796)."""
    st = api_last()
    me = st.game.my_player
    solo = replace(st, game=replace(st.game, players=(me,)))
    obs, why = status_observation(solo, solo.t)
    assert obs is None and why == "игроков 1, а не 10"
    snap = build_snapshot(StateInputs(api=solo, match=match_view(MatchState(), solo.t, why)),
                          solo.t)
    assert check_snapshot(snap) == []
    want = "снимок API не принят: игроков 1, а не 10"
    assert snap["match"]["time"]["value"] is None and snap["match"]["time"]["reason"] == want
    assert snap["champions"]["reason"] == want and snap["enemies"]["reason"] == want
    assert snap["me"]["alive"]["reason"] == want and snap["me"]["respawn_in"]["reason"] == want
    assert snap["me"]["champion"]["value"] == me.champion       # статика API — своя правда
    assert snap["sources"]["match"]["api_refused"] == why
    # Без отказа причина прежняя: API жив, состава в контексте нет.
    plain = build_snapshot(StateInputs(api=solo, match=match_view(MatchState(), solo.t)),
                           solo.t)
    assert "ещё не знает состава" in plain["champions"]["reason"]


# ---------------------------------------------------------------- события API (#803)

OWN_END = OWN.parent / "own_game_end.jsonl"
# Выдуманные ники: свой — «Лис#EUW» (Zilean), остальные по составу журнала.
NICKS = ["Лис#EUW", "Бобр#RU1", "Ёж#RU1", "Сова#RU2", "Волк#RU1",
         "Кабан#RU1", "Рысь#RU1", "Сом#RU1", "Ёж#RU2", "Барсук#RU1"]


def ev(i: int, name: str, t: float, **data) -> EventRecord:
    return EventRecord(i, name, t, data)


def nick_game(*events: EventRecord, game_time: float = 600.0):
    """Снимок API своей игры с выдуманными никами и своими событиями."""
    st = api_last()
    g = st.game
    players = tuple(replace(p, riot_id=n) for p, n in zip(g.players, NICKS))
    active = replace(g.active, riot_id=NICKS[0])
    game = replace(g, players=players, active=active, game_time=game_time,
                   events=(ev(0, "GameStart", 0.07), ev(1, "MinionsSpawning", 30.0), *events))
    return replace(st, game=game)


def events_of(st: LiveStatus, clock: float | None = 600.0) -> dict:
    """Поле `events` снимка из одного API; `clock` — время матча в контексте
    (None — контекст без времени)."""
    mv = match_view(MatchState(), st.t)
    if clock is not None:
        mv = replace(mv, clock=ss.Observed(clock, "api", mv.t))
    snap = build_snapshot(StateInputs(api=st, match=mv), st.t)
    assert check_snapshot(snap) == []
    return snap["events"]


def who(champion, relation, other=None) -> dict:
    return {"champion": champion, "relation": relation, "other": other}


def test_events_in_full_snapshot_are_champions_not_nicknames():
    """Журнал своей игры, 5:37: три убийства за последние 180 с (первое
    убийство и FirstBlood на 1:43 — старше окна), ники → чемпионы."""
    inp, now = full_inputs()
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    e = snap["events"]
    assert e["source"] == "api" and e["age"] == 0.2
    assert [x["kind"] for x in e["value"]] == ["ChampionKill"] * 3
    last = e["value"][-1]
    assert last["killer"] == who("Darius", "enemy") and last["victim"] == who("Zilean", "self")
    assert last["assisters"] == [] and last["detail"] is None
    assert last["time"] == pytest.approx(333.2, abs=0.05)
    assert last["ago"] == pytest.approx(337.35 - 333.2, abs=0.1)
    first = e["value"][0]
    assert first["killer"] == who("Amumu", "ally") and first["victim"] == who("Lux", "enemy")
    assert first["assisters"] == [who("Yone", "ally")]
    assert "Player" not in json.dumps(e)                     # ники обезличенного журнала


def test_event_names_match_by_full_id_or_name_before_tag():
    st = nick_game(
        ev(2, "ChampionKill", 500.0, KillerName="Кабан#RU1", VictimName="Лис",
           Assisters=["Рысь", "Сом#RU1"]),
        ev(3, "ChampionKill", 510.0, KillerName="Незнакомец", VictimName="Ёж",
           Assisters=["Бобр"]),
        ev(4, "FirstBlood", 500.0, Recipient="Кабан"),
        ev(5, "Ace", 520.0, Acer="Сова", AcingTeam="ORDER"),
    )
    got = events_of(st)["value"]
    assert [x["kind"] for x in got] == ["ChampionKill", "FirstBlood", "ChampionKill", "Ace"]
    kill = got[0]
    assert kill["killer"] == who("Darius", "enemy") and kill["victim"] == who("Zilean", "self")
    assert kill["assisters"] == [who("Diana", "enemy"), who("Lux", "enemy")]
    assert got[1]["killer"] == who("Darius", "enemy") and got[1]["victim"] is None
    # ник не нашёлся — unknown; «Ёж» без тега у двух игроков — не угадываем
    assert got[2]["killer"] == who(None, None, "unknown")
    assert got[2]["victim"] == who(None, None, "unknown")
    assert got[2]["assisters"] == [who("Amumu", "ally")]
    assert got[3]["killer"] == who("Caitlyn", "ally") and got[3]["detail"] == "ORDER"
    assert [x["ago"] for x in got] == [100.0, 100.0, 90.0, 80.0]
    text = json.dumps(got, ensure_ascii=False)
    assert not any(n.split("#")[0] in text for n in NICKS + ["Незнакомец"])


def test_event_objects_are_labels_with_team_relation():
    st = nick_game(
        ev(2, "ChampionKill", 500.0, KillerName="Turret_T2_C_05_A", VictimName="Бобр",
           Assisters=[]),
        ev(3, "ChampionKill", 501.0, KillerName="Minion_T100L0S37N0230", VictimName="Рысь"),
        ev(4, "ChampionKill", 502.0, KillerName="SRU_Baron12.1.1", VictimName="Сом"),
        ev(5, "TurretKilled", 503.0, TurretKilled="Turret_TOrder_L1_P3_1242677625_0",
           KillerName="Minion_T200L2S29N0180", Assisters=["Кабан"]),
        ev(6, "InhibKilled", 504.0, InhibKilled="Inhib_TChaos_L1_P1_1931666598_0",
           KillerName="Лис", Assisters=[]),
        ev(7, "DragonKill", 505.0, DragonType="Fire", Stolen="True", KillerName="Рысь",
           Assisters=["Сом"]),
        ev(8, "BaronKill", 506.0, Stolen="False", KillerName="Бобр", Assisters=[]),
        ev(9, "Multikill", 507.0, KillerName="Лис", KillStreak=2),
        ev(10, "HordeKill", 508.0, Stolen="False", KillerName="Сова"),
    )
    got = {x["kind"] + str(n): x for n, x in enumerate(events_of(st)["value"])}
    assert got["ChampionKill0"]["killer"] == who(None, "enemy", "turret")
    assert got["ChampionKill0"]["victim"] == who("Amumu", "ally")
    assert got["ChampionKill1"]["killer"] == who(None, "ally", "minion")
    assert got["ChampionKill2"]["killer"] == who(None, None, "monster")
    tk = got["TurretKilled3"]
    assert tk["killer"] == who(None, "enemy", "minion")
    assert tk["victim"] == who(None, "ally", "turret")     # своя башня
    assert tk["assisters"] == [who("Darius", "enemy")]
    assert tk["detail"] == "Turret_TOrder_L1_P3_1242677625_0"
    inhib = got["InhibKilled4"]
    assert inhib["killer"] == who("Zilean", "self") and inhib["victim"] is None
    assert inhib["detail"] == "Inhib_TChaos_L1_P1_1931666598_0"
    drag = got["DragonKill5"]
    assert drag["detail"] == "Fire, Stolen" and drag["victim"] == who(None, None, "monster")
    assert drag["killer"] == who("Diana", "enemy") and drag["assisters"] == [who("Lux", "enemy")]
    assert got["BaronKill6"]["detail"] is None
    assert got["Multikill7"]["detail"] == "2" and got["Multikill7"]["victim"] is None
    assert got["HordeKill8"]["killer"] == who("Caitlyn", "ally")
    out = json.dumps(got)
    assert "Turret_T2" not in out and "SRU_" not in out and "Minion_" not in out


def test_events_window_limit_and_unknown_clock():
    many = [ev(10 + k, "ChampionKill", 300.0 + 10 * k, KillerName="Лис", VictimName="Кабан")
            for k in range(30)]                                # 300 … 590
    st = nick_game(*many)
    got = events_of(st)["value"]
    # окно 180 с от 600: события с 420 — их 18, все влезают в 20
    assert [x["time"] for x in got] == [420.0 + 10 * k for k in range(18)]
    assert got[0]["ago"] == 180.0 and got[-1]["ago"] == 10.0
    burst = [ev(100 + k, "ChampionKill", 590.0, KillerName="Лис", VictimName="Кабан")
             for k in range(25)]
    got = events_of(nick_game(*burst))["value"]
    assert len(got) == ss.EVENTS_MAX == 20                     # последние 20 по EventID
    # время матча неизвестно: окно — от времени снимка API, ago — null
    got = events_of(st, clock=None)["value"]
    assert len(got) == 18 and all(x["ago"] is None for x in got)


def test_events_refused_without_api_or_game():
    snap = build_snapshot(StateInputs(), 1.0)
    assert snap["events"] == ss.refuse(ss.WHY_NO_API)
    now = 50.0
    st = LiveStatus(State.NO_GAME, "нет соединения: refused", now)
    snap = build_snapshot(StateInputs(api=st), now)
    assert snap["events"]["value"] is None and "no_game" in snap["events"]["reason"]
    loading = statuses()[0]
    assert loading.state is State.LOADING and loading.game is not None
    snap = build_snapshot(StateInputs(api=loading), loading.t)
    assert snap["events"]["reason"].startswith("не в игре: Live Client loading")


def test_events_after_game_end_are_kept():
    st = [s for s, _ in replay(read_journal(OWN_END), IDS)][-1]
    assert st.state is State.ENDED
    e = events_of(st, clock=None)
    assert e["source"] == "api" and 0 < len(e["value"]) <= ss.EVENTS_MAX
    assert e["value"][-1]["kind"] == "GameEnd" and e["value"][-1]["detail"] in ("Win", "Lose")
    kills = [x for x in e["value"] if x["kind"] == "ChampionKill"]
    assert kills and all(x["killer"]["champion"] and x["victim"]["champion"] for x in kills)


def test_schema_closes_event_items():
    """Лишнее поле (сырой ник) и метка не из списка — ошибки схемы."""
    inp, now = full_inputs()
    for spoil in (lambda v: v[0].update(killer_name="Лис"),
                  lambda v: v[1]["killer"].update(other="tower"),
                  lambda v: v[2].update(assisters=["Лис"])):
        snap = build_snapshot(inp, now)
        spoil(snap["events"]["value"])
        assert check_snapshot(snap) == ["$.events.value: не подходит ни под один вариант anyOf"]


def test_example_client_prints_recent_events():
    ex = load_example()
    inp, now = full_inputs()
    got = {f["id"]: f for f in ex.to_source_facts(build_snapshot(inp, now))}
    line = got["events.recent"]
    assert line["source"] == "api"
    assert line["text"].endswith("ChampionKill Darius (враг) → Zilean (вы), 4 с назад")
    assert ex.event_text({"kind": "TurretKilled", "killer": who(None, "enemy", "minion"),
                          "victim": who(None, "ally", "turret"), "assisters": [],
                          "detail": "Turret_TOrder_L1_P3", "ago": None}) == \
        "TurretKilled миньоны (враг) → башня (союзник) [Turret_TOrder_L1_P3]"
    # без API строки нет: отказ не становится утверждением
    assert "events.recent" not in {f["id"] for f in
                                   ex.to_source_facts(build_snapshot(StateInputs(), 1.0))}


# ---------------------------------------------------------------- заклинания (#804)

def spells(*variants: str) -> tuple[SpellRecord, ...]:
    fam = {"SummonerTeleportUpgrade": "SummonerTeleport",
           "SummonerSmiteAvatarOffensive": "SummonerSmite"}
    return tuple(SpellRecord(n, fam.get(v, v), v, f"GeneratedTip_SummonerSpell_{v}_DisplayName",
                             "") for n, v in enumerate(variants, 1))


def with_spells(st: LiveStatus, by_champ: dict) -> LiveStatus:
    g = st.game
    players = tuple(replace(p, spells=by_champ[p.champion], spells_reason=None)
                    if p.champion in by_champ else p for p in g.players)
    return replace(st, game=replace(g, players=players))


def test_spells_of_me_and_champions_from_api():
    inp, now = full_inputs()
    api = with_spells(inp.api, {"Zilean": spells("SummonerTeleportUpgrade", "SummonerFlash"),
                                "Diana": spells("SummonerFlash", "SummonerSmiteAvatarOffensive")})
    snap = build_snapshot(replace(inp, api=api), now)
    assert check_snapshot(snap) == []
    me = snap["me"]
    assert me["spells"] == {"value": ["SummonerTeleport", "SummonerFlash"], "source": "api",
                            "age": 0.2, "reason": None}
    assert me["spell_variants"]["value"] == ["SummonerTeleportUpgrade", "SummonerFlash"]
    ch = by_champ(snap)
    assert ch["Diana"]["spells"]["value"] == ["SummonerFlash", "SummonerSmite"]
    assert ch["Diana"]["spell_variants"]["value"] == ["SummonerFlash",
                                                      "SummonerSmiteAvatarOffensive"]
    # у остальных API заклинаний не дал (обезличенный журнал) — причина разбора
    assert ch["Lux"]["spells"]["value"] is None
    assert "summonerSpells" in ch["Lux"]["spells"]["reason"]


def test_spells_refused_with_parse_reason_and_schema_closes_them():
    inp, now = full_inputs()
    snap = build_snapshot(inp, now)
    assert "summonerSpells" in snap["me"]["spells"]["reason"]
    assert snap["me"]["spell_variants"]["reason"] == snap["me"]["spells"]["reason"]
    bad = json.loads(json.dumps(snap))
    bad["me"]["spells"] = {"value": ["SummonerFlash"], "source": "api", "age": 0.0,
                           "reason": None}
    assert any("me.spells" in e for e in check_snapshot(bad))


# ---------------------------------------------------------------- роль и руны (#807)

def test_role_and_runes_of_me_and_champions_from_api():
    inp, now = full_inputs()
    g = inp.api.game
    comet = RuneRecord(8229, "ArcaneComet", 8200, "Sorcery", 8300, "Inspiration")
    players = tuple(replace(p, runes=comet, runes_reason=None) if p.champion == "Zilean" else p
                    for p in g.players)
    api = replace(inp.api, game=replace(g, players=players))
    snap = build_snapshot(replace(inp, api=api), now)
    assert check_snapshot(snap) == []
    me = snap["me"]
    # обезличенный журнал ранкеда: position заполнен у всех
    assert me["role"] == {"value": "TOP", "source": "api", "age": 0.2, "reason": None}
    assert me["runes"]["value"] == {"keystone": {"id": 8229, "name": "ArcaneComet"},
                                    "primary": {"id": 8200, "name": "Sorcery"},
                                    "secondary": {"id": 8300, "name": "Inspiration"}}
    ch = by_champ(snap)
    assert ch["Diana"]["role"]["value"] == "JUNGLE"
    assert ch["Lux"]["runes"]["value"] is None and "runes" in ch["Lux"]["runes"]["reason"]


def test_role_refused_without_position_and_schema_closes_it():
    inp, now = full_inputs()
    g = inp.api.game
    players = tuple(replace(p, position="NONE") for p in g.players)
    snap = build_snapshot(replace(inp, api=replace(inp.api, game=replace(g, players=players))),
                          now)
    assert check_snapshot(snap) == []
    assert "режим без ролей" in snap["me"]["role"]["reason"]
    assert all(e["role"]["value"] is None for e in snap["champions"]["value"])
    bad = json.loads(json.dumps(build_snapshot(inp, now)))
    bad["me"]["role"]["value"] = "SUPPORT"
    assert any("me.role" in e for e in check_snapshot(bad))
    no_api = build_snapshot(StateInputs(), 1.0)
    assert no_api["me"]["runes"]["reason"] and no_api["me"]["role"]["value"] is None
# ---------------------------------------------------------------- готовность D/F по HUD (#806)

def slot(k: str, state: str, value, reason=None) -> dict:
    return {"slot": k, "state": state, "value": value, "reason": reason, "sim": None,
            "vrel": None}


def spell_inputs(reads, panel: bool = True, view: str = "player"):
    """Свой — Zilean с Телепортом (D) и Скачком (F); кадр с чтениями слотов."""
    inp, now = full_inputs(view)
    api = with_spells(inp.api, {"Zilean": spells("SummonerTeleportUpgrade", "SummonerFlash")})
    p = passport(view)
    if panel:
        p.set_spell_panel(True, 0.0, "тест", scale=1.0, score=0.6)
    vis = replace(inp.vision, spells=reads, spells_why=None if reads is not None else "нет")
    return replace(inp, api=api, passport=p, vision=vis), now


def test_spells_ready_by_family_from_hud():
    inp, now = spell_inputs([slot("D", "cooldown", False), slot("F", "ready", True)])
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    sr = snap["me"]["spells_ready"]
    assert sr["source"] == "hud" and sr["reason"] is None
    assert sr["value"] == {"SummonerTeleport": False, "SummonerFlash": True}
    hud = snap["me"]["spells_hud"]["value"]
    assert [h["state"] for h in hud] == ["cooldown", "ready"]
    assert hud[0]["spell"] == "SummonerTeleport"


def test_spells_ready_null_slot_keeps_other_and_reason_in_hud():
    why = "заклинание под запретом: красный крестик"
    inp, now = spell_inputs([slot("D", "blocked", None, why), slot("F", "ready", True)])
    snap = build_snapshot(inp, now)
    assert check_snapshot(snap) == []
    assert snap["me"]["spells_ready"]["value"] == {"SummonerTeleport": None,
                                                   "SummonerFlash": True}
    assert snap["me"]["spells_hud"]["value"][0]["reason"] == why


def test_spells_ready_refused_when_both_unknown_or_passport_says_no():
    inp, now = spell_inputs([slot("D", "dead", None, "свой мёртв"),
                             slot("F", "hidden", None, "панели нет")])
    snap = build_snapshot(inp, now)
    sr = snap["me"]["spells_ready"]
    assert sr["value"] is None and "свой мёртв" in sr["reason"] and "панели нет" in sr["reason"]
    assert check_snapshot(snap) == []
    # вид наблюдателя — читатель выключен паспортом
    inp, now = spell_inputs([slot("D", "ready", True), slot("F", "ready", True)],
                            view="spectator")
    snap = build_snapshot(inp, now)
    assert "нужен вид игрока" in snap["me"]["spells_ready"]["reason"]
    # панель не найдена
    inp, now = spell_inputs([slot("D", "ready", True), slot("F", "ready", True)], panel=False)
    snap = build_snapshot(inp, now)
    assert "панель умений не найдена" in snap["me"]["spells_ready"]["reason"]
    assert check_snapshot(snap) == []


def test_spells_ready_needs_spells_from_api():
    inp, now = full_inputs("player")
    p = passport("player")
    p.set_spell_panel(True, 0.0, "тест", scale=1.0, score=0.6)
    vis = replace(inp.vision, spells=[slot("D", "ready", True), slot("F", "ready", True)])
    snap = build_snapshot(replace(inp, passport=p, vision=vis), now)
    assert snap["me"]["spells_ready"]["value"] is None
    assert "заклинания из API неизвестны" in snap["me"]["spells_ready"]["reason"]
    bad = json.loads(json.dumps(snap))
    bad["me"]["spells_ready"] = {"value": {"SummonerFlash": "yes"}, "source": "hud",
                                 "age": 0.1, "reason": None}
    assert any("spells_ready" in e for e in check_snapshot(bad))
