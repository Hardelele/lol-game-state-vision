"""Строка кадра live: истина камеры только там, где миникарта настоящая (#788);
склейка Session._emit — юниты, треки, контекст и события именованными полями
(records.py, schema 1, #797)."""

import json

import numpy as np

from lolstream import live
from lolvision.geometry import Projection
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


def _session(found: bool, view: str) -> live.Session:
    s = object.__new__(live.Session)
    s.start, s.fps, s._n_emit = 0.0, 3.0, 0
    s.match, s.tracker, s.proj = None, None, None
    s.minimap = {"box": [0.85, 0.75, 1.0, 1.0], "scale": None, "found": found}
    s.view = view
    s._out = _Out()
    return s


def _emit(s: live.Session) -> dict:
    mask = np.zeros((live.MM, live.MM), np.float32)
    mask[100:140, 60:130] = 1                    # белая рамка камеры
    s._emit(mask, None, None, None, 70, 40, _Model(), {}, 0.0)
    return s._out.rows[-1]


def test_no_minimap_in_player_view_gives_no_truth():
    row = _emit(_session(found=False, view="player"))
    assert row["err"] is None and row["cx"] is None and row["cy"] is None and row["q"] is None
    assert row["truth"] == "no_minimap"
    assert (row["px"], row["py"]) == (0.4, 0.6)


def test_spectator_layout_and_found_minimap_keep_truth():
    for found, view in ((False, "spectator"), (True, "player")):
        row = _emit(_session(found, view))
        assert "truth" not in row
        assert row["err"] is not None and row["cx"] is not None and row["q"] is not None


def test_roster_error_is_in_session_status():
    import threading
    import time
    from types import SimpleNamespace
    s = _session(found=True, view="spectator")
    s.sid, s.run, s.kind, s.model_label, s.trained_on = "x", "r", "", "", []
    s.state, s.note, s.title, s.duration = live.State.RUNNING, "1920x1080", "", 0
    s.rows, s.roster, s.started, s._lock = [], None, time.time(), threading.Lock()
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
                                               "x", "y", "hp", "age", "relation",
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
