"""Строка кадра live: истина камеры только там, где миникарта настоящая (#788);
склейка Session._emit — юниты, треки, контекст и события именованными полями
(records.py, schema 1, #797)."""

import json
import threading

import numpy as np

from lolstream import live
from lolvision.geometry import Projection
from lolvision.passport import Passport
from lolvision.records import SCHEMA, UNSET, MinimapChamp, UnitRec
from lolvision.unit_tracker import UnitTracker


class _Model:
    def predict(self, prep):
        return {"px": 0.4, "py": 0.6, "qx": 0.4, "qy": 0.6, "spread": 0.01}, np.zeros((4, 4)), {}


class _Out:
    def __init__(self):
        self.rows = []

    def submit(self, fn, row, *rest):
        self.rows.append(row)


ASSETS = {"unit_net": True, "names_bank": True, "minimap_model": True}


def _passport(found: bool, view: str, assets=None) -> Passport:
    """Паспорт ролика 1920×1080: вид, находка миникарты, состав по HUD."""
    p = Passport.build({"kind": "youtube", "size": (1920, 1080)}, 0.0,
                       assets=ASSETS if assets is None else assets)
    if view != "unknown":
        p.set_view(view, 0.0, how="задан тестом")
    if found:
        p.set_minimap((0.85, 0.75, 1.0, 1.0), 1.1, 0.7, 0.0)
    p.set_roster("hud", 0.0)
    return p


def _session(found: bool, view: str, assets=None) -> live.Session:
    s = object.__new__(live.Session)
    s.start, s.fps, s._n_emit = 0.0, 3.0, 0
    s.match, s.tracker, s.proj, s._match_at, s.vision = None, None, None, None, None
    s.hud_clock, s.last_hits, s.last_hits_recent = live.hud_stats.HudClock(), live.lh.LastHits(), []
    s.self_place = live.lh.SelfPlace()
    s.lane = live.LaneWatch()
    s.minimap = {"box": [0.85, 0.75, 1.0, 1.0], "scale": None, "found": found}
    s.view = view
    s.passport = _passport(found, view, assets)
    s.passport_events, s._pnote = [], []
    s.api, s.api_reason = None, "нет — источник YouTube"
    s.state_url, s.state_reason = None, "у этого сеанса не поднимается"
    s._lock = threading.Lock()
    s._out = _Out()
    return s


def _emit(s: live.Session, ms: dict | None = None) -> dict:
    mask = np.zeros((live.MM, live.MM), np.float32)
    mask[100:140, 60:130] = 1                    # белая рамка камеры
    s._emit(mask, None, None, None, 70, 40, _Model(), ms or {}, 0.0)
    return s._out.rows[-1]


def test_no_minimap_in_player_view_gives_no_truth():
    row = _emit(_session(found=False, view="player"))
    assert row["err"] is None and row["cx"] is None and row["cy"] is None and row["q"] is None
    assert row["truth"] == "no_minimap"
    # Причина — паспорта (camera_truth), она же в отказах кадра.
    assert row["truth_reason"].startswith("миникарта не найдена") and "вид игрока" in \
        row["truth_reason"]
    assert row["off"]["camera_truth"] == row["truth_reason"]
    assert (row["px"], row["py"]) == (0.4, 0.6)


def test_unknown_view_without_minimap_gives_no_truth():
    row = _emit(_session(found=False, view="unknown"))
    assert row["truth"] == "no_minimap" and "вид неизвестен" in row["truth_reason"]
    assert row["view"] == "unknown"


def test_spectator_layout_and_found_minimap_keep_truth():
    for found, view in ((False, "spectator"), (True, "player"), (True, "unknown")):
        row = _emit(_session(found, view))
        assert "truth" not in row and "camera_truth" not in row["off"]
        assert row["err"] is not None and row["cx"] is not None and row["q"] is not None


def test_roster_error_is_in_session_status():
    import time
    from types import SimpleNamespace
    s = _session(found=True, view="spectator")
    s.sid, s.run, s.kind, s.model_label, s.trained_on = "x", "r", "", "", []
    s.state, s.note, s.title, s.duration = live.State.RUNNING, "1920x1080", "", 0
    s.rows, s.roster, s.started = [], None, time.time()
    s.source = SimpleNamespace(kind="youtube", describe=dict)
    s.source_state, s.source_gaps = {"state": "ok"}, {}
    s.api, s.api_reason, s.api_events = None, "нет — источник YouTube", []
    s._pipe = SimpleNamespace(
        champs=SimpleNamespace(error={"text": "KeyError: 'Hwei'", "time": time.time()}))
    st = s.status()
    assert st["champs_error"]["text"] == "KeyError: 'Hwei'"
    assert "KeyError: 'Hwei'" in st["note"]


# ---------- склейка кадра на синтетике (#797) ----------

NAMES = ["Irelia", "Evelynn", "TwistedFate", "Jhin", "Zyra",
         "Volibear", "Hecarim", "Locke", "Jinx", "Lulu"]
# Кадр 1000×1000 — сцена 100×100, пиксель сцены — 0.001 доли карты.
PROJ = Projection(np.array([[0.001, 0, -0.05], [0, 0.001, -0.05], [0, 0, 1.0]]),
                  (100, 100), (0, 0, 1000, 1000), (1000, 1000))


def _minimap(**over) -> list[MinimapChamp]:
    out = []
    for k, n in enumerate(NAMES):
        d = {"visible": False, "u": None, "v": None, "alive": True, "timer": None,
             **over.get(n, {})}
        out.append(MinimapChamp(n, "ORDER" if k < 5 else "CHAOS",
                                f"{'LR'[k // 5]}{k % 5 + 1}", **d))
    return out


def _game_ms(units, champs) -> dict:
    kind_p = np.zeros(len(live.KIND_CLASSES), np.float32)
    kind_p[live.KIND_CLASSES.index("game")] = 0.9
    return {"kind_p": kind_p, "unit_recs": units, "champs": champs}


def _glued_rows() -> list[dict]:
    """Три игровых кадра: Volibear на экране (с именем) и миньон рядом,
    Jinx на миникарте, на третьем кадре Jinx умирает (таймер 12 с)."""
    s = _session(found=True, view="spectator")
    s.proj = PROJ
    s.tracker = UnitTracker(PROJ)
    mask = np.zeros((live.MM, live.MM), np.float32)
    mask[100:140, 60:130] = 1
    for k in range(3):
        units = [UnitRec("champion", "CHAOS", 0.6, 0.5, 0.7, 0.95, "Volibear"),
                 UnitRec("minion", "ORDER", 0.3 + 0.001 * k, 0.4, 1.0, 0.8),
                 UnitRec("ward", "ORDER", 0.2, 0.7, 1.0, 0.6)]
        jinx = (dict(visible=True, u=0.62, v=0.31) if k < 2 else
                dict(alive=False, timer=12))
        s._emit(mask, None, None, None, 70, 40, _Model(), _game_ms(units, _minimap(Jinx=jinx)),
                k / s.fps)
    return s._out.rows


def _no_positional_lists(x, path="row"):
    """В строке кадра нет списков-записей: список — только записей-словарей
    (или имён состава в событии roster)."""
    if isinstance(x, dict):
        for k, v in x.items():
            _no_positional_lists(v, f"{path}.{k}")
    elif isinstance(x, list):
        if path.endswith(".roster"):
            assert all(isinstance(v, str) for v in x), path
            return
        for j, v in enumerate(x):
            assert isinstance(v, dict), f"{path}[{j}] — {v!r}"
            _no_positional_lists(v, f"{path}[{j}]")


def test_emit_glues_units_tracks_context_and_events_by_name():
    rows = _glued_rows()
    for row in rows:
        assert row["schema"] == SCHEMA == 1
        assert row["kind"] == "game"
        json.dumps(row)                                # уходит в JSON как есть
        _no_positional_lists(row)
    last = rows[-1]
    # Юниты: доли кадра, имя у чемпиона, None у прочих.
    assert [(u["kind"], u["name"]) for u in last["units"]] == [
        ("champion", "Volibear"), ("minion", None), ("ward", None)]
    # relation/side (sides.py, #790): пока live.py не зовёт sides.assign —
    # None с причиной UNSET, а не выдуманное значение.
    assert last["units"][0] == {"kind": "champion", "team": "CHAOS", "x": 0.6, "y": 0.5,
                                "hp": 0.7, "score": 0.95, "name": "Volibear", "color": None,
                                "relation": None, "relation_reason": UNSET,
                                "side": None, "side_reason": UNSET, "side_src": None}
    # Чемпионы миникарты — по составу, именованные поля.
    jinx = next(c for c in last["champs"] if c["name"] == "Jinx")
    assert jinx == {"name": "Jinx", "team": "CHAOS", "slot": "R4", "visible": False,
                    "u": None, "v": None, "alive": False, "timer": 12}
    # Треки: подтверждены со второй встречи; у видимого место в кадре то же,
    # что у юнита (по нему смотрелка сводит id с юнитом).
    tracks = {t["id"]: t for t in last["tracks"]}
    assert "champ-CHAOS-Volibear" in tracks
    vb = tracks["champ-CHAOS-Volibear"]
    assert vb["state"] == "seen" and (vb["x"], vb["y"]) == (0.6, 0.5)
    # Камера (0.4, 0.6) + смещение по проекции: (600, 500) px → (+0.01, 0).
    assert (vb["u"], vb["v"]) == (0.41, 0.6)
    assert {t["kind"] for t in last["tracks"]} == {"champion", "minion", "ward"}
    assert set(tracks[next(iter(tracks))]) == {"id", "kind", "team", "state", "u", "v",
                                               "x", "y", "hp", "hp_rate", "age", "relation",
                                               "relation_reason", "side", "side_reason",
                                               "side_src"}
    # События трекера — на втором кадре (подтверждение треков).
    assert sorted(e["type"] for e in rows[1]["uev"]) == ["unit_new"] * 3
    assert all({"t", "type", "id", "kind", "team"} <= set(e) for e in rows[1]["uev"])
    # Контекст: Volibear виден на экране, место — через ту же проекцию; Jinx
    # мёртв, таймер из панели.
    ctx = {c["name"]: c for c in last["ctx"]}
    assert ctx["Volibear"]["state"] == "visible" and ctx["Volibear"]["src"] == "screen"
    assert (ctx["Volibear"]["u"], ctx["Volibear"]["v"]) == (0.41, 0.6)
    assert ctx["Jinx"]["state"] == "dead" and ctx["Jinx"]["respawn_in"] == 12
    assert (ctx["Jinx"]["u"], ctx["Jinx"]["v"], ctx["Jinx"]["src"]) == (0.62, 0.31, "minimap")
    assert ctx["Lulu"] == {"name": "Lulu", "team": "CHAOS", "state": "unknown", "u": None,
                           "v": None, "age": None, "src": None, "hp": None,
                           "respawn_in": None, "alive": True, "alive_src": "hud",
                           "alive_age": 0.0, "hp_src": None, "hp_age": None,
                           "respawn_src": None, "conflicts": 0}
    # Событие смерти — на третьем кадре, с последним местом.
    assert last["ev"] == [{"t": last["t"], "type": "death", "name": "Jinx", "team": "CHAOS",
                           "u": 0.62, "v": 0.31, "respawn_in": 12, "src": "hud"}]
    assert "ev" not in rows[0] or all(e["type"] != "death" for e in rows[0]["ev"])


def test_emit_without_units_or_roster_has_schema_and_no_records():
    row = _emit(_session(found=True, view="spectator"))
    assert row["schema"] == 1
    assert not {"units", "champs", "ctx", "tracks", "ev", "uev"} & set(row)
    # Без API: время матча — null с причиной, контекст без конфликтов.
    assert row["game_time"] == {"value": None, "src": None,
                                "reason": "Live Client API: нет — источник YouTube"}
    assert row["conflicts"] == {} and row["api_refused"] is None


def test_refused_readers_are_null_with_reason_in_row():
    # Детектора нет и вид неизвестен: юниты и иконки миникарты — null, а не
    # пропуск; причина — в off (как её дал паспорт на кадре).
    s = _session(found=False, view="unknown", assets={**ASSETS, "unit_net": False})
    off = s.passport.disabled()
    row = _emit(s, {"off": off})
    assert row["units"] is None and row["off"]["units"] == "нет: детектор юнитов"
    assert row["champs"] is None and row["off"]["minimap_icons"].startswith("миникарта не найдена")
    assert row["off"]["names"].startswith("нужен вид наблюдателя, а вид неизвестен")
    json.dumps(row)


# ---------- другая карта по API (#803) ----------

class _NotRift(_Model):
    """Патчевая модель: «это не места Ущелья»."""
    def predict(self, prep):
        out, hm, mt = super().predict(prep)
        return {**out, "is_game": False}, hm, mt


def _arena_rows(map_number):
    s = _session(found=True, view="player")
    s.map_number = map_number
    s.proj, s.tracker = PROJ, UnitTracker(PROJ)
    mask = np.zeros((live.MM, live.MM), np.float32)
    mask[100:140, 60:130] = 1                    # рамка вьюпорта на миникарте
    units = [UnitRec("champion", "CHAOS", 0.6, 0.5, 0.7, 0.95, "Volibear")]
    s._emit(mask, None, None, None, 70, 40, _NotRift(),
            _game_ms(units, _minimap(Jinx=dict(visible=True, u=0.9, v=0.15))), 0.0)
    return s, s._out.rows[-1]


def test_off_rift_camera_from_minimap_and_no_rift_only_readers():
    s, row = _arena_rows(12)
    # «не Ущелье» патчевой модели на ARAM — не довод против «игры».
    assert row["kind"] == "game"
    vis = s.vision
    assert vis.camera_src == "minimap" and vis.camera == (row["cx"], row["cy"])
    assert vis.spread is None and vis.tracks is None and "не Ущелье" in vis.tracks_why
    assert "tracks" not in row
    # Чемпионы миникарты приняты, место — с миникарты (доли годятся для любой карты).
    ctx = {c["name"]: c for c in row["ctx"]}
    assert (ctx["Jinx"]["u"], ctx["Jinx"]["v"], ctx["Jinx"]["src"]) == (0.9, 0.15, "minimap")
    assert ctx["Volibear"]["u"] is None                  # с экрана — нет: проекция Ущелья


def test_off_rift_without_visible_viewport_refuses_camera():
    """Рамки вьюпорта не видно (окно аугментов ARAM) — камеры нет, с причиной."""
    s = _session(found=True, view="player")
    s.map_number = 12
    s._emit(np.zeros((live.MM, live.MM), np.float32), None, None, None, 70, 40, _NotRift(),
            _game_ms([], None), 0.0)
    assert s.vision.camera is None and "не видна" in s.vision.camera_why


def test_rift_keeps_model_camera_and_not_game_veto():
    for map_number in (None, 11):
        s, row = _arena_rows(map_number)
        assert row["kind"] == "other"                   # как раньше: не места Ущелья
        assert s.vision.camera_src == "screen" and s.vision.camera == (0.4, 0.6)


# ---------- добивания в сеансе (#816) ----------

def test_last_hits_step_pairs_cs_gain_with_minion_death():
    """Смерть вражеского миньона линии на кадре роста CS → last_hit в строке
    кадра и в итоге для /state; без разрешения last_hits — только CS."""
    s = _session(found=True, view="player")
    me = {"kind": "champion", "color": "green", "x": 0.5, "y": 0.5}
    me_tr = {"id": "c1", "kind": "champion", "x": 0.5, "y": 0.5, "u": 0.30, "v": 0.70}
    death = {"type": "unit_death", "id": "m7", "kind": "minion", "relation": "enemy",
             "u": 0.31, "v": 0.70}
    out = None
    for k in range(12):
        t = 0.25 * k
        row = {"t": t, "units": [me], "tracks": [me_tr],
               "uev": [death] if k == 4 else []}
        out = s._last_hits_step(row, {"present": True, "cs": 30 if k < 4 else 31, "time": 90}, {})
        if k == 4:
            assert "last_hits" not in row          # событие — после подтверждения роста
        if row.get("last_hits"):
            assert [e["type"] for e in row["last_hits"]] == ["last_hit"]
    assert out["cs"] == 31 and out["last_hits"]["counts"]["last_hit"] == 1
    assert out["last_hits"]["recent"][0]["id"] == "m7"

    s2 = _session(found=True, view="player")
    out2 = s2._last_hits_step({"t": 0.0}, {"present": True, "cs": 5, "time": 10},
                              {"last_hits": "нет: детектор юнитов"})
    out2 = s2._last_hits_step({"t": 1.0}, {"present": True, "cs": 5, "time": 11},
                              {"last_hits": "нет: детектор юнитов"})
    assert out2["cs"] == 5 and out2["last_hits"] is None
    assert out2["last_hits_why"] == "нет: детектор юнитов"
