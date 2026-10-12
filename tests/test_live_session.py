"""Сеанс live на подставном источнике кадров и подставном Live Client (#794, #795).

Без видеокарты и без захвата: источник отдаёт заранее заданные ответы
(кадр, пауза, окна нет, конец), путь кадра собирается настоящим
`build_pipeline` на процессоре, но вместо модели камеры — заглушка, а
модели юнитов, чемпионов и типа сцены «не найдены на диске». Live Client
читает журнал своей игры через подставной транспорт — порт 2999 не трогается.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lolstream import live
from lolstream.frame_source import Frame, Status, gap
from lolstream.live_client import LiveClient, Response, journal_transport, read_journal

OWN = Path(__file__).parent / "fixtures" / "live_client" / "own_game.jsonl"
NOWHERE = Path(__file__).parent / "fixtures" / "нет-такого-файла"


class _Runner:
    """Модель камеры-заглушка: всегда одна точка, сеть не нужна."""

    kind, label, trained_on = "stub", "заглушка", []

    def describe(self, scene_size) -> str:
        return f"заглушка {scene_size[0]}x{scene_size[1]}"

    def prepare(self, scene):
        return None

    def frame_prep(self, frame_size, crop, boxes, store_size):
        return None

    def predict(self, prep):
        return ({"px": 0.4, "py": 0.6, "qx": 0.4, "qy": 0.6, "spread": 0.01},
                np.zeros((4, 4), np.float32), {"net": 0.0, "agg": 0.0})


@pytest.fixture
def cpu_models(monkeypatch, tmp_path):
    """build_pipeline без тяжёлых моделей: чекпоинт есть (пустой файл),
    грузится заглушка; детектора юнитов, чемпионов и типа сцены «нет»."""
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"")
    monkeypatch.setattr(live, "run_checkpoint", lambda run: ck)
    monkeypatch.setattr(live, "load_runner", lambda path, device: _Runner())
    monkeypatch.setattr(live, "MINI_RUN", NOWHERE)
    monkeypatch.setattr(live, "KIND_NET", NOWHERE / "net.pt")
    monkeypatch.setattr(live, "KIND_HEAD", NOWHERE / "head.pt")
    monkeypatch.setattr(live, "live_net", lambda: (None, None))
    built: list[tuple[int, int]] = []
    real = live.build_pipeline

    def build(size, run, device, fps):
        built.append(size)
        return real(size, run, device, fps)
    return build, built


def _img(w: int = 640, h: int = 360) -> np.ndarray:
    return np.full((h, w, 3), 40, np.uint8)


class _Source:
    """Источник по сценарию: список ответов, затем ENDED (или пауза без
    конца при `forever`). Запоминает потоки, в которых его открыли и читали."""

    def __init__(self, script: list[Frame], kind: str = "youtube", size=(640, 360),
                 forever: Frame | None = None, fps: float = 2.0):
        self.script, self.kind, self._size, self.forever = list(script), kind, size, forever
        self.fps, self.backend, self.backend_note = fps, "подставной", "задан тестом"
        self.start = 0.0
        self.threads: dict[str, set[int]] = {"open": set(), "read": set()}
        self.closed = False

    @property
    def size(self):
        return self._size

    def open(self) -> None:
        self.threads["open"].add(threading.get_ident())

    def read(self) -> Frame:
        self.threads["read"].add(threading.get_ident())
        if self.script:
            return self.script.pop(0)
        if self.forever is not None:
            time.sleep(0.005)
            return self.forever
        return gap(Status.ENDED, 99.0, "ролик кончился")

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "fps": self.fps,
                "size": self._size, "title": "сценарий", "duration": 0.0}


def _ok(t: float, w: int = 640, h: int = 360) -> Frame:
    return Frame(Status.OK, t, _img(w, h))


def _wait(s: live.Session, cond, timeout: float = 20.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        st = s.status()
        if cond(st):
            return st
        time.sleep(0.01)
    raise AssertionError(f"не дождались: {s.status()['state']} · {s.status()['note']}")


def _finish(s: live.Session) -> dict:
    s.join(20.0)
    assert not s._th.is_alive()
    return s.status()


# ---------- путь кадра и состояния источника ----------

def test_rows_take_frame_time_and_gaps_write_no_rows(cpu_models):
    build, built = cpu_models
    src = _Source([_ok(0.0), gap(Status.PAUSED, 0.5), _ok(1.0),
                   gap(Status.NO_WINDOW, 1.5, "окна нет"), _ok(2.25)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert st["state"] == live.State.ENDED.label and st["active"] is False
    # Строки — только на кадры, время — Frame.t, а не номер / fps.
    assert [r["t"] for r in st["frames"]] == [0.0, 1.0, 2.25]
    assert [r["i"] for r in st["frames"]] == [0, 1, 2]
    assert st["source_gaps"] == {"paused": 1, "no_window": 1}
    assert st["source_state"]["state"] == "ended"
    assert st["source_state"]["reason"] == "ролик кончился"
    assert built == [(640, 360)]               # размер известен заранее — один путь
    assert st["kind"] == "stub" and "без юнитов" in st["note"]
    # У ролика API нет — явно, с причиной.
    assert st["api"] is None and st["api_reason"] == "нет — источник YouTube"
    assert src.closed
    json.dumps(st)


def test_open_and_read_happen_in_session_thread(cpu_models):
    build, _ = cpu_models
    src = _Source([_ok(0.0)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    _finish(s)
    assert src.threads["open"] == src.threads["read"] == {s._th.ident}
    assert threading.get_ident() not in src.threads["read"]


def test_pause_keeps_session_running_with_reason(cpu_models):
    build, _ = cpu_models
    paused = gap(Status.PAUSED, 3.0)
    src = _Source([_ok(0.0)], kind="screen", forever=paused)
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _wait(s, lambda st: st["source_state"]["state"] == "paused")
    assert st["state"] == live.State.RUNNING.label and st["active"] is True
    assert st["source_state"]["reason"] == Status.PAUSED.label
    assert st["speed"] is None                 # экран — в реальном времени
    s.stop()
    st = _finish(s)
    assert st["state"] == live.State.STOPPED.label
    assert st["source_gaps"]["paused"] >= 1
    # Остановленный сеанс прогрев не досчитывает: кадр в строки не попал.
    assert st["count"] == 0


def test_unknown_size_builds_pipeline_on_first_frame(cpu_models):
    build, built = cpu_models
    src = _Source([gap(Status.NO_WINDOW, 0.0, "окна нет"), _ok(0.5, 800, 450)],
                  kind="screen", size=None)
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert built == [(800, 450)]
    assert st["count"] == 1 and st["note"].startswith("свой экран 800x450")


def test_resize_rebuilds_pipeline_and_says_so(cpu_models):
    build, built = cpu_models
    src = _Source([_ok(0.0), _ok(0.5), _ok(1.0, 800, 450), _ok(1.5, 800, 450)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert built == [(640, 360), (800, 450)]
    assert "640x360 → 800x450: путь кадра пересобран" in st["note"]
    # Кадры прогрева старого размера отданы до пересборки, ничего не потеряно.
    assert [r["t"] for r in st["frames"]] == [0.0, 0.5, 1.0, 1.5]
    assert st["source_state"]["state"] == "ended"


def test_missing_model_fails_with_reason(monkeypatch):
    monkeypatch.setattr(live, "run_checkpoint", lambda run: NOWHERE / "model.pt")
    src = _Source([_ok(0.0)])
    s = live.Session("vid", "нет/такого", src, device="cpu")
    st = _finish(s)
    assert st["state"] == live.State.FAILED.label
    assert st["note"] == "нет модели runs/нет/такого/model.pt"
    assert st["count"] == 0 and src.closed


def test_source_open_failure_fails_session(cpu_models):
    build, _ = cpu_models

    class _Broken(_Source):
        def open(self):
            raise RuntimeError("ни один бэкенд захвата не открылся")
    src = _Broken([], kind="screen")
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert st["state"] == live.State.FAILED.label
    assert "ни один бэкенд захвата не открылся" in st["note"]


# ---------- Live Client в сеансе ----------

def _client(transport) -> LiveClient:
    return LiveClient(transport=transport, period=0.01, champion_ids=None)


def test_live_client_runs_with_screen_session(cpu_models):
    build, _ = cpu_models
    api = _client(journal_transport(read_journal(OWN)))
    src = _Source([], kind="screen", forever=gap(Status.PAUSED, 0.0))
    s = live.Session("screen", "r", src, device="cpu", build=build, api=api)
    st = _wait(s, lambda st: st["api"] is not None and st["api"]["state"] == "in_game"
               and st["api"]["events"] > 0)
    a = st["api"]
    assert st["api_reason"] is None
    # В status — сводка без полного снимка игры.
    assert "game" not in a and a["source"] == "api"
    assert a["brief"]["players"] == 10 and a["brief"]["game_time"] > 0
    assert a["reason"] is None or isinstance(a["reason"], str)
    full = s.api_full()
    assert full["api"]["game"] is not None and len(full["api"]["game"]["players"]) == 10
    assert len(full["events"]) == len(s.api_events) >= a["events"]
    ids = [e["event_id"] for e in full["events"]]
    assert len(ids) == len(set(ids))           # события без повторов
    json.dumps(s.status())
    s.stop()
    _finish(s)
    assert api._thread is None                 # опрос остановлен вместе с сеансом


def test_no_api_does_not_break_session(cpu_models):
    build, _ = cpu_models
    api = _client(lambda endpoint: Response(None, None, "нет соединения: refused"))
    src = _Source([_ok(0.0), _ok(0.5)], kind="screen", forever=gap(Status.PAUSED, 1.0))
    s = live.Session("screen", "r", src, device="cpu", build=build, api=api)
    st = _wait(s, lambda st: st["api"]["state"] == "no_game"
               and st["api"]["reason"] != "ещё не опрошен")
    assert st["active"] is True and st["api"]["brief"] is None
    assert "нет соединения" in st["api"]["reason"]
    s.stop()
    st = _finish(s)
    assert st["state"] == live.State.STOPPED.label


def test_start_screen_opens_screen_session_with_api(monkeypatch, cpu_models):
    build, _ = cpu_models
    made = {}

    def screen(fps, backend="auto"):
        made["fps"], made["backend"] = fps, backend
        return _Source([], kind="screen", forever=gap(Status.NO_WINDOW, 0.0, "окна нет"))
    monkeypatch.setattr(live, "ScreenSource", screen)
    monkeypatch.setattr(live, "build_pipeline", build)
    monkeypatch.setattr(live, "LiveClient", lambda champion_ids=None: _client(
        lambda endpoint: Response(None, None, "нет соединения")))
    monkeypatch.setattr(live, "_sessions", {})
    with pytest.raises(ValueError):
        live.start_screen("r", 2.0, backend="gdi")
    st = live.start_screen("r", 3.0, backend="mss")
    try:
        assert st["id"] == live.SCREEN_ID == "screen"
        assert made == {"fps": 3.0, "backend": "mss"}
        s = live.get("screen")
        assert s.api is not None
        st = _wait(s, lambda st: st["source_state"]["state"] == "no_window")
        assert st["source_state"]["reason"] == "окна нет" and st["count"] == 0
        assert live.listing()[0]["source"] == "screen"
    finally:
        live.stop("screen")
        _finish(live.get("screen"))


def test_start_youtube_reads_video_source_without_api(monkeypatch, cpu_models):
    build, _ = cpu_models
    made = {}

    def video(src, fps, start=0.0):
        made.update(src=src, fps=fps, start=start)
        return _Source([_ok(start)])
    monkeypatch.setattr(live, "VideoSource", video)
    monkeypatch.setattr(live, "build_pipeline", build)
    monkeypatch.setattr(live, "_sessions", {})
    live.start("dQw4w9WgXcQ", "r", 1.0, 30.0)
    s = live.get("dQw4w9WgXcQ")
    st = _finish(s)
    assert made == {"src": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "fps": 1.0,
                    "start": 30.0}
    assert s.api is None and st["api_reason"] == "нет — источник YouTube"
    assert [r["t"] for r in st["frames"]] == [30.0]


def test_serve_start_route_picks_source(monkeypatch):
    from types import SimpleNamespace

    from lolstream import serve_inspect
    calls = []
    monkeypatch.setattr(live, "start_screen", lambda *a: calls.append(("screen", a)) or {})
    monkeypatch.setattr(live, "start", lambda *a: calls.append(("youtube", a)) or {})
    out = []
    h = SimpleNamespace(server=SimpleNamespace(live_run="def"), _json=out.append)
    for body in ({"source": "screen", "fps": 3, "backend": "mss"},
                 {"id": "abc", "run": "x", "fps": 2, "start": "5"}):
        h.body = body
        serve_inspect.live_start(h)
    assert calls == [("screen", ("def", 3, "mss")), ("youtube", ("abc", "x", 2, 5.0))]
    h.body = {"source": "twitch"}
    with pytest.raises(ValueError):
        serve_inspect.live_start(h)
