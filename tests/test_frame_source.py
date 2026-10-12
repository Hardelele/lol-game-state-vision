"""Источник кадров (#794) на синтетике: без экрана, окна игры и ffmpeg."""

import numpy as np
import pytest

from lolstream import frame_source as fs
from lolstream.frame_source import Frame, ScreenSource, Status, VideoSource, WindowState


class _Clock:
    """Часы, которые идут только во сне и при снимке."""

    def __init__(self):
        self.now, self.slept = 100.0, []

    def __call__(self):
        return self.now

    def sleep(self, dt):
        self.slept.append(dt)
        self.now += dt


class _Windows:
    def __init__(self, state: WindowState | None, hwnd: int | None = 7):
        self.st, self.hwnd, self.finds = state, hwnd, 0

    def find(self, title):
        self.finds += 1
        return self.hwnd

    def state(self, hwnd):
        return self.st if hwnd == self.hwnd else None


class _Grabber:
    name = "fake"

    def __init__(self, bounds=(0, 0, 1920, 1080), cost=0.0, clock=None, img=None, error=None):
        self.bounds, self.cost, self.clock = bounds, cost, clock
        self.img, self.error = img, error
        self.rects, self.closed = [], 0

    def grab(self, rect):
        if self.error is not None:
            raise self.error
        self.rects.append(rect)
        if self.clock is not None:
            self.clock.now += self.cost
        if self.img is not None:
            return self.img
        x0, y0, x1, y1 = rect
        return np.full((y1 - y0, x1 - x0, 3), 9, np.uint8)

    def close(self):
        self.closed += 1


GAME = WindowState(rect=(0, 0, 1920, 1080), foreground=True, minimized=False)


def _screen(state=GAME, fps=2.0, grabber=None, windows=None, **kw):
    clock = _Clock()
    g = grabber or _Grabber(clock=clock)
    if g.clock is None:
        g.clock = clock
    src = ScreenSource(fps=fps, windows=windows or _Windows(state), grabber=g,
                       clock=clock, sleep=clock.sleep, **kw)
    return src, g, clock


# ---------- экран: кадр и отказы ----------

def test_screen_frame_is_window_client_rect_rgb():
    src, g, _ = _screen(WindowState((100, 50, 1380, 770), True, False))
    fr = src.read()
    assert fr.ok and fr.reason is None
    assert fr.size == (1280, 720)            # не 1080p — как есть, без масштаба
    assert fr.image.dtype == np.uint8 and fr.image.shape == (720, 1280, 3)
    assert g.rects == [(100, 50, 1380, 770)]
    assert src.size == (1280, 720)


def test_screen_paused_when_window_not_in_focus():
    src, g, _ = _screen(WindowState((0, 0, 1920, 1080), False, False))
    fr = src.read()
    assert fr.status is Status.PAUSED and not fr.ok
    assert fr.image is None and fr.size is None and "фокус" in fr.reason
    assert g.rects == []                     # чужой экран не снимается


def test_screen_minimized_window():
    src, g, _ = _screen(WindowState((-32000, -32000, -31840, -31972), False, True))
    fr = src.read()
    assert fr.status is Status.MINIMIZED and fr.image is None and fr.reason
    assert g.rects == []


def test_screen_no_window_then_window_appears():
    win = _Windows(None, hwnd=None)
    src, g, _ = _screen(windows=win)
    fr = src.read()
    assert fr.status is Status.NO_WINDOW and fs.WINDOW_TITLE in fr.reason
    win.hwnd, win.st = 7, GAME               # игра запустилась
    assert src.read().ok


def test_screen_window_outside_capture_bounds_is_refused():
    # Окно на втором мониторе, а бэкенд видит только основной.
    src, g, _ = _screen(WindowState((1920, 0, 3840, 1080), True, False))
    fr = src.read()
    assert fr.status is Status.NO_FRAME and "выходит за область захвата" in fr.reason
    assert g.rects == []


def test_screen_grab_error_is_a_gap_not_an_exception():
    src, _, _ = _screen(grabber=_Grabber(error=OSError("DXGI access lost")))
    fr = src.read()
    assert fr.status is Status.NO_FRAME and "DXGI access lost" in fr.reason
    assert src.backend == "fake"             # бэкенд на ходу не подменяется


def test_screen_no_new_frame_and_bad_shape_are_gaps():
    class _None(_Grabber):
        def grab(self, rect):
            return None
    src, _, _ = _screen(grabber=_None())
    assert src.read().status is Status.NO_FRAME
    src, _, _ = _screen(grabber=_Grabber(img=np.zeros((4, 4, 4), np.uint8)))
    fr = src.read()
    assert fr.status is Status.NO_FRAME and "(H, W, 3)" in fr.reason


def test_screen_whole_monitor_without_title():
    clock = _Clock()
    g = _Grabber(bounds=(0, 0, 2560, 1440), clock=clock)
    src = ScreenSource(fps=1, title=None, grabber=g, clock=clock, sleep=clock.sleep)
    src.open()
    assert src.size == (2560, 1440)
    assert src.read().size == (2560, 1440)


# ---------- экран: ограничение кадр/с ----------

def test_screen_fps_limit_spaces_frames():
    src, _, clock = _screen(fps=4.0)
    ts = [src.read().t for _ in range(5)]
    assert np.allclose(np.diff(ts), 0.25)
    assert clock.slept and all(dt <= 0.25 + 1e-9 for dt in clock.slept)


def test_screen_fps_limit_holds_for_gaps_too():
    # Пауза не крутит цикл вхолостую: отказы идут тем же тактом.
    src, _, clock = _screen(WindowState((0, 0, 1920, 1080), False, False), fps=2.0)
    ts = [src.read().t for _ in range(4)]
    assert np.allclose(np.diff(ts), 0.5)


def test_screen_slow_capture_is_not_caught_up_in_a_burst():
    clock = _Clock()
    g = _Grabber(clock=clock, cost=1.0)      # снимок дольше такта 0.25 с
    src = ScreenSource(fps=4.0, windows=_Windows(GAME), grabber=g,
                       clock=clock, sleep=clock.sleep)
    ts = [src.read().t for _ in range(4)]
    assert np.allclose(np.diff(ts), 1.0)     # каждые 1 с, а не пачка
    assert all(dt <= 0.25 + 1e-9 for dt in clock.slept)


def test_screen_close_ends_source():
    src, g, _ = _screen()
    src.read()
    src.close()
    src.close()
    assert g.closed == 1
    assert src.read().status is Status.ENDED


def test_screen_rejects_bad_fps():
    with pytest.raises(ValueError):
        ScreenSource(fps=0)


# ---------- выбор бэкенда: не молча ----------

def _fake_backends(monkeypatch, fail: set[str]):
    def make(name):
        def ctor():
            if name in fail:
                raise ImportError(f"No module named '{name}'")
            g = _Grabber()
            g.name = name
            return g
        return ctor
    monkeypatch.setattr(fs, "GRABBERS", {n: make(n) for n in fs.BACKENDS})


def test_backend_auto_prefers_dxcam(monkeypatch):
    _fake_backends(monkeypatch, fail=set())
    g, note = fs.open_grabber("auto")
    assert g.name == "dxcam" and "основной" in note


def test_backend_fallback_to_mss_says_why(monkeypatch):
    _fake_backends(monkeypatch, fail={"dxcam"})
    g, note = fs.open_grabber("auto")
    assert g.name == "mss"
    assert "запасной" in note and "dxcam" in note and "No module named" in note


def test_backend_named_explicitly_is_not_replaced(monkeypatch):
    _fake_backends(monkeypatch, fail={"dxcam"})
    with pytest.raises(ImportError):
        fs.open_grabber("dxcam")
    with pytest.raises(ValueError):
        fs.open_grabber("bitblt")


def test_backend_none_available_raises_with_both_reasons(monkeypatch):
    _fake_backends(monkeypatch, fail={"dxcam", "mss"})
    with pytest.raises(RuntimeError, match="dxcam.*mss"):
        fs.open_grabber("auto")


def test_screen_source_reports_its_backend(monkeypatch):
    _fake_backends(monkeypatch, fail={"dxcam"})
    clock = _Clock()
    src = ScreenSource(fps=2, windows=_Windows(GAME), clock=clock, sleep=clock.sleep)
    assert src.read().ok
    d = src.describe()
    assert d["kind"] == "screen" and d["backend"] == "mss"
    assert "запасной" in d["backend_note"] and d["size"] == (1920, 1080)


# ---------- ролик ----------

def _video(monkeypatch, n: int, dur: float, fps: float = 2.0, start: float = 0.0):
    calls = {}

    def probe_url(url):
        calls["probe"] = url
        return 1280, 720, dur, "vid123", {"title": "Ролик"}

    def frames(video, w, h, fps_, from_url=False, decode="auto", skip=0.0):
        calls["frames"] = (video, w, h, fps_, from_url, skip)
        try:
            for i in range(n):
                yield np.full((h, w, 3), i, np.uint8)
        finally:
            calls["closed"] = True

    monkeypatch.setattr(fs.video_stream, "probe_url", probe_url)
    monkeypatch.setattr(fs.video_stream, "frames", frames)
    return VideoSource("https://youtu.be/x", fps, start), calls


def test_video_source_wraps_frames_with_time(monkeypatch):
    src, calls = _video(monkeypatch, n=3, dur=1.5, fps=2.0, start=10.0)
    src.open()
    assert src.size == (1280, 720) and src.title == "Ролик"
    out = [src.read() for _ in range(3)]
    assert all(f.ok for f in out)
    assert [f.t for f in out] == [10.0, 10.5, 11.0]
    assert [int(f.image[0, 0, 0]) for f in out] == [0, 1, 2]
    assert calls["frames"] == ("https://youtu.be/x", 1280, 720, 2.0, True, 10.0)


def test_video_source_end_is_a_state(monkeypatch):
    src, calls = _video(monkeypatch, n=4, dur=2.0)
    fr = [f for f in fs.iter_frames(src)]
    assert len(fr) == 4
    end = src.read()
    assert end.status is Status.ENDED and end.reason == "ролик кончился"
    assert src.read() is end                 # и дальше — конец, без исключения
    assert calls["closed"]


def test_video_source_short_stream_is_named(monkeypatch):
    src, _ = _video(monkeypatch, n=2, dur=60.0)
    list(fs.iter_frames(src))
    end = src.read()
    assert end.status is Status.ENDED and "поток оборван" in end.reason


def test_video_source_close_stops_the_stream(monkeypatch):
    src, calls = _video(monkeypatch, n=100, dur=50.0)
    assert src.read().ok
    src.close()
    assert calls.get("closed")


def test_frame_without_image_has_no_size():
    f = Frame(Status.PAUSED, 1.0, None, "x")
    assert not f.ok and f.size is None


# ---------- настоящий Win32: только поиск, без захвата ----------

@pytest.mark.skipif(not fs.sys.platform.startswith("win"), reason="только Windows")
def test_win32_windows_missing_window_is_none():
    w = fs.Win32Windows()
    assert w.find("lol-game-state-vision: такого окна нет 7f3a") is None
    assert w.state(0) is None
    x0, y0, x1, y1 = w.screen()
    assert (x0, y0) == (0, 0) and x1 > 0 and y1 > 0
