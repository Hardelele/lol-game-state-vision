"""tools/live_screen.py (#801): сеанс «свой экран» одной командой.

Без экрана и видеокарты: start_screen / get / stop модуля live подменены
подставным сеансом, сон — счётчиком; Ctrl+C — KeyboardInterrupt из сна.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lolstream import live

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import live_screen as ls  # noqa: E402


class FakeSession:
    def __init__(self, url="http://127.0.0.1:8733", reason=None):
        self.url, self.reason = url, reason
        self.active, self.stopped, self.joined = True, False, None
        self.count = 0

    def status(self, since=0):
        self.count += 1
        return {"state": "идёт" if self.active else "остановлено", "active": self.active,
                "state_url": self.url, "state_reason": self.reason,
                "source_state": {"state": "paused", "reason": "пауза: окно игры не в фокусе"},
                "api": {"state": "in_game", "reason": None}, "api_reason": None,
                "view": "player", "count": self.count, "note": ""}

    def join(self, timeout):
        self.joined = timeout


@pytest.fixture
def fake(monkeypatch):
    reg: dict = {"sess": None, "started": [], "make": FakeSession}

    def start_screen(run, fps, backend):
        reg["started"].append((run, fps, backend))
        reg["sess"] = reg["make"]()
        return reg["sess"].status()

    def stop(sid):
        assert sid == live.SCREEN_ID
        if reg["sess"] is not None:
            reg["sess"].active, reg["sess"].stopped = False, True

    monkeypatch.setattr(live, "start_screen", start_screen)
    monkeypatch.setattr(live, "get", lambda sid: reg["sess"])
    monkeypatch.setattr(live, "stop", stop)
    return reg


def interrupt_after(n):
    calls = []

    def sleep(sec):
        calls.append(sec)
        if len(calls) >= n:
            raise KeyboardInterrupt
    return sleep, calls


def test_prints_url_and_stops_on_ctrl_c(fake, capsys):
    sleep, calls = interrupt_after(2)
    code = ls.run(2.0, "auto", "patches/cnn", sleep=sleep, busy=lambda: False)
    out = capsys.readouterr().out
    assert code == 0 and fake["started"] == [("patches/cnn", 2.0, "auto")]
    assert "сервис состояния: http://127.0.0.1:8733/state" in out
    assert "сеанс: идёт · источник: paused (пауза: окно игры не в фокусе) · API: in_game" in out
    assert "вид: player" in out and "кадров:" in out
    assert calls == [ls.PERIOD, ls.PERIOD]
    s = fake["sess"]
    assert s.stopped and s.joined == ls.STOP_WAIT        # остановлен и дождался
    assert "остановлено" in out


def test_session_end_exits_loop(fake, capsys):
    def sleep(sec):
        fake["sess"].active = False                    # источник кончился сам
    assert ls.run(2.0, "auto", "r", sleep=sleep, busy=lambda: False) == 0
    assert "сеанс: остановлено" in capsys.readouterr().out


def test_busy_port_does_not_start(fake, capsys):
    code = ls.run(2.0, "auto", "r", sleep=lambda s: None, busy=lambda: True)
    err = capsys.readouterr().err
    assert code == 2 and fake["started"] == []
    assert "смотрелка держит сеанс «свой экран»" in err


def test_active_screen_session_in_process_is_busy(fake, capsys):
    fake["sess"] = FakeSession()
    code = ls.run(2.0, "auto", "r", sleep=lambda s: None, busy=lambda: False)
    assert code == 2 and fake["started"] == [] and not fake["sess"].stopped
    assert "уже занят" in capsys.readouterr().err


def test_state_port_taken_after_start_stops_session(fake, capsys):
    fake["make"] = lambda: FakeSession(url=None, reason="не поднят: OSError: [WinError 10048]")
    code = ls.run(2.0, "auto", "r", sleep=lambda s: None, busy=lambda: False)
    cap = capsys.readouterr()
    assert code == 2 and fake["sess"].stopped
    assert "не поднят: OSError" in cap.out and "смотрелка" in cap.err


def test_bad_backend_is_a_message(fake, monkeypatch, capsys):
    def bad(run, fps, backend):
        raise ValueError(f"неизвестный бэкенд захвата {backend!r}")
    monkeypatch.setattr(live, "start_screen", bad)
    assert ls.run(2.0, "gdi", "r", sleep=lambda s: None, busy=lambda: False) == 2
    assert "сеанс не открыт" in capsys.readouterr().err


def test_defaults_match_viewer(monkeypatch):
    seen = {}
    monkeypatch.setattr(ls, "run", lambda fps, backend, run, minimap_scale: seen.update(
        fps=fps, backend=backend, run=run, minimap_scale=minimap_scale) or 0)
    assert ls.main([]) == 0
    assert seen == {"fps": 2.0, "backend": "auto", "run": live.DEFAULT_RUN,
                    "minimap_scale": None}


def test_minimap_scale_goes_to_session(monkeypatch):
    """--minimap-scale доходит до start_screen; неверный масштаб — отказ с кодом 2."""
    seen = {}

    def start_screen(run, fps, backend, minimap_scale=None):
        seen["scale"] = minimap_scale
        raise ValueError("стоп после старта")

    monkeypatch.setattr(live, "start_screen", start_screen)
    monkeypatch.setattr(live, "get", lambda sid: None)
    assert ls.run(2.0, "auto", "r", sleep=lambda s: None, busy=lambda: False,
                  minimap_scale=1.2) == 2
    assert seen == {"scale": 1.2}


def test_pending_service_is_waited_for_not_taken_for_busy(fake, capsys):
    """Сразу после старта причина «ещё не поднят» — ждать, а не считать порт занятым."""
    class Late(FakeSession):
        def status(self, since=0):
            st = super().status(since)
            if self.count < 3:
                st.update(state_url=None, state_reason=live.STATE_PENDING)
            return st
    fake["make"] = Late
    sleep, _ = interrupt_after(5)
    assert ls.run(2.0, "auto", "r", sleep=sleep, busy=lambda: False) == 0
    assert "http://127.0.0.1:8733/state" in capsys.readouterr().out
    assert fake["sess"].stopped
