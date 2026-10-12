"""Строка кадра live: истина камеры только там, где миникарта настоящая (#788)."""

import numpy as np

from lolstream import live


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
    s._emit(mask, None, None, None, 70, 40, _Model(), {})
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
    s.vid, s.run, s.kind, s.model_label, s.trained_on = "x", "r", "", "", []
    s.state, s.note, s.title, s.duration = live.State.RUNNING, "1920x1080", "", 0
    s.rows, s.roster, s.started, s._lock = [], None, time.time(), threading.Lock()
    s._champs = SimpleNamespace(error={"text": "KeyError: 'Hwei'", "time": time.time()})
    st = s.status()
    assert st["champs_error"]["text"] == "KeyError: 'Hwei'"
    assert "KeyError: 'Hwei'" in st["note"]
