"""Источник кадров для live: ролик (YouTube или файл) или свой экран (#794).

Сеанс live не должен знать, откуда кадр. Источник отдаёт по запросу
`read()` запись `Frame`: картинку RGB uint8 (H, W, 3), время кадра и его
размер — или явное «кадра нет» с причиной: пауза (окно игры не в фокусе),
окно свёрнуто, окна нет, поток кончился. Это состояние, а не исключение:
пауза в игре — обычное дело, и сеанс должен её показать, а не упасть.

Две реализации:

* `VideoSource` — обёртка над `video_stream.frames` (так live читал ролик и
  раньше): размеры и длительность заранее, время кадра = start + i / fps,
  в конце — `ENDED` с причиной, полный ли был поток (`short_stream`).
* `ScreenSource` — окно `League of Legends (TM) Client` через DXGI (dxcam)
  или MSS. Прямоугольник — клиентская область окна (без рамки и заголовка),
  ищется через Win32 API на ctypes. Кадр не 1080p отдаётся как есть: к
  1080p его приводит путь кадра (#788), а не источник.

Отказ лучше выдумки: переход dxcam → mss не молчаливый — источник сообщает,
каким бэкендом он работает и почему (`backend`, `backend_note`); сбой
захвата посреди сеанса — кадр `NO_FRAME` с причиной, бэкенд на ходу не
меняется.

dxcam и mss импортируются лениво, внутри экранного источника: ролику они не
нужны, а на машине без них YouTube-путь работает как прежде.
"""

from __future__ import annotations

import enum
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from lolstream import video_stream

# Заголовок окна самой игры (не клиента-лаунчера). Совпадение точное.
WINDOW_TITLE = "League of Legends (TM) Client"

# Бэкенды захвата экрана в порядке предпочтения при backend="auto".
BACKENDS = ("dxcam", "mss")


class Status(enum.Enum):
    """Есть ли кадр, и если нет — почему (машинное имя + подпись)."""

    OK = ("ok", "кадр")
    PAUSED = ("paused", "пауза: окно игры не в фокусе")
    MINIMIZED = ("minimized", "окно игры свёрнуто")
    NO_WINDOW = ("no_window", "окна игры нет")
    NO_FRAME = ("no_frame", "кадр не снят")
    ENDED = ("ended", "поток кончился")

    def __init__(self, key: str, label: str):
        self.key, self.label = key, label


@dataclass(frozen=True)
class Frame:
    """Один ответ источника: кадр или его отсутствие с причиной.

    `image` — RGB uint8 (H, W, 3) или None; `t` — время кадра в секундах по
    шкале источника (у ролика — позиция в ролике, у экрана — от открытия
    источника); `size` — (W, H) снятого кадра или None, если кадра нет;
    `reason` — почему кадра нет (None у кадра).
    """

    status: Status
    t: float
    image: np.ndarray | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def size(self) -> tuple[int, int] | None:
        if self.image is None:
            return None
        return int(self.image.shape[1]), int(self.image.shape[0])


def gap(status: Status, t: float, reason: str | None = None) -> Frame:
    """Кадра нет: состояние и причина (по умолчанию — подпись состояния)."""
    return Frame(status, t, None, reason or status.label)


class FrameSource(Protocol):
    """Откуда сеанс берёт кадры. Сеанс видит только это.

    `kind` — "youtube" / "file" / "screen"; `backend` и `backend_note` —
    чем снимаются кадры и почему именно так (у экрана известны после
    `open()`); `fps` — сколько кадров в секунду просят; `size` — ожидаемый
    (W, H) кадра, известный до первого кадра (под него собирается путь
    кадра); у экрана он может смениться — тогда `Frame.size` его покажет.
    """

    kind: str
    fps: float
    backend: str | None
    backend_note: str | None

    @property
    def size(self) -> tuple[int, int] | None: ...

    def open(self) -> None: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...

    def describe(self) -> dict: ...


def iter_frames(source: FrameSource) -> Iterator[Frame]:
    """Все ответы источника до `ENDED` (его не включая)."""
    while True:
        fr = source.read()
        if fr.status is Status.ENDED:
            return
        yield fr


# ---------------------------------------------------------------- ролик


class VideoSource:
    """Ролик YouTube или локальный файл потоком через ffmpeg.

    То же, что делал live: `probe_url` (или `probe` для файла) даёт размеры
    и длительность, `video_stream.frames` — поток кадров, прореженный до
    `fps`, с пропуском первых `start` секунд. Время кадра i — start + i / fps.
    """

    def __init__(self, src: str | Path, fps: float, start: float = 0.0,
                 from_url: bool = True, decode: str = "auto", align: str = "near",
                 rate: str | None = None):
        self.src, self.fps, self.start = src, float(fps), float(start)
        self.from_url, self.decode, self.align, self.rate = from_url, decode, align, rate
        self.kind = "youtube" if from_url else "file"
        self.backend: str | None = "ffmpeg"
        self.backend_note: str | None = None
        self.title = ""
        self.duration = 0.0
        self.info: dict = {}
        self._size: tuple[int, int] | None = None
        self._stream: Iterator[np.ndarray] | None = None
        self._n = 0
        self._ended: Frame | None = None

    @property
    def size(self) -> tuple[int, int] | None:
        return self._size

    def open(self) -> None:
        """Размеры и длительность ролика (без скачивания)."""
        if self._size is not None:
            return
        if self.from_url:
            w, h, dur, vid, info = video_stream.probe_url(str(self.src))
            self.info, self.title = info, info.get("title") or vid
        else:
            w, h, dur = video_stream.probe(Path(self.src))
            self.title = Path(self.src).name
        self._size, self.duration = (w, h), float(dur)

    def read(self) -> Frame:
        if self._ended is not None:
            return self._ended
        self.open()
        w, h = self._size
        if self._stream is None:
            self._stream = video_stream.frames(self.src, w, h, self.fps, self.from_url,
                                               decode=self.decode, skip=self.start,
                                               **self._extra())
        t = self.start + self._n / self.fps
        fr = next(self._stream, None)
        if fr is None:
            self.close()
            short = video_stream.short_stream(self._n, self.duration, self.fps, self.start)
            why = f"{Status.ENDED.label}: {short}" if short else "ролик кончился"
            self._ended = gap(Status.ENDED, t, why)
            return self._ended
        self._n += 1
        return Frame(Status.OK, t, fr)

    def _extra(self) -> dict:
        """Выравнивание кадра и потолок полосы — только если заданы (#843)."""
        out = {}
        if self.align != "near":
            out["align"] = self.align
        if self.rate:
            out["rate"] = self.rate
        return out

    def close(self) -> None:
        """Погасить ffmpeg и yt-dlp: генератор закрывается — они завершаются."""
        if self._stream is not None:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
            self._stream = None

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "backend_note": self.backend_note,
                "fps": self.fps, "size": self._size, "title": self.title,
                "duration": self.duration, "start": self.start}


# ---------------------------------------------------------------- окно игры


@dataclass(frozen=True)
class WindowState:
    """Окно в момент опроса: клиентский прямоугольник в пикселях экрана
    (left, top, right, bottom), в фокусе ли, свёрнуто ли."""

    rect: tuple[int, int, int, int]
    foreground: bool
    minimized: bool


class Windows(Protocol):
    """Поиск окна и его состояние; в тестах — подставной."""

    def find(self, title: str) -> int | None: ...

    def state(self, hwnd: int) -> WindowState | None: ...


class Win32Windows:
    """Окна через user32 на ctypes, без pywin32.

    Процесс переводится в режим «знаю DPI каждого монитора»: иначе при
    масштабе Windows 125–150% GetClientRect отдаёт логические пиксели, и
    прямоугольник окна не совпал бы с пикселями захвата. Если перевести не
    вышло (режим уже задан иначе), это записывается в `note`.
    """

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("захват окна игры есть только под Windows")
        import ctypes
        from ctypes import wintypes

        self._ct, self._wt = ctypes, wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.note: str | None = None
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4.
        ok = self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if not ok and not self.user32.IsProcessDPIAware():
            self.note = ("процесс не знает DPI: при масштабе Windows не 100% "
                         "прямоугольник окна не совпадёт с пикселями экрана")
        self.user32.FindWindowW.restype = wintypes.HWND
        self.user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        for name in ("IsWindow", "IsIconic"):
            getattr(self.user32, name).argtypes = [wintypes.HWND]
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]

    def find(self, title: str) -> int | None:
        h = self.user32.FindWindowW(None, title)
        return int(h) if h else None

    def state(self, hwnd: int) -> WindowState | None:
        """None — окна больше нет (игра закрыта)."""
        u, ct, wt = self.user32, self._ct, self._wt
        if not u.IsWindow(hwnd):
            return None
        rc, pt = wt.RECT(), wt.POINT(0, 0)
        if not u.GetClientRect(hwnd, ct.byref(rc)) or not u.ClientToScreen(hwnd, ct.byref(pt)):
            return None
        fg = u.GetForegroundWindow()
        return WindowState(rect=(pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom),
                           foreground=bool(fg) and int(fg) == hwnd,
                           minimized=bool(u.IsIconic(hwnd)))

    def screen(self) -> tuple[int, int, int, int]:
        """Основной монитор целиком (он всегда начинается в (0, 0))."""
        return (0, 0, self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1))


# ---------------------------------------------------------------- захват


class Grabber(Protocol):
    """Снимок прямоугольника экрана. `bounds` — что этот бэкенд вообще
    может снять (left, top, right, bottom) в пикселях экрана."""

    name: str
    bounds: tuple[int, int, int, int]

    def grab(self, rect: tuple[int, int, int, int]) -> np.ndarray | None: ...

    def close(self) -> None: ...


class DxcamGrabber:
    """DXGI Desktop Duplication через dxcam: основной монитор, RGB сразу.

    dxcam видит один монитор (основной); окно на другом мониторе он не
    снимет — это отказ кадра с причиной, а не тихий переход на mss.
    """

    name = "dxcam"

    def __init__(self):
        import dxcam

        self.cam = dxcam.create(output_color="RGB")
        if self.cam is None:
            raise RuntimeError("dxcam.create вернул None (нет вывода DXGI)")
        self.bounds = (0, 0, int(self.cam.width), int(self.cam.height))

    def grab(self, rect):
        # new_frame_only=False: экран не менялся — тот же кадр, а не None.
        # None остаётся только до первого кадра вообще.
        return self.cam.grab(region=tuple(rect), copy=True, new_frame_only=False)

    def close(self) -> None:
        self.cam.release()


class MssGrabber:
    """GDI через mss: весь виртуальный экран, медленнее DXGI.

    Экземпляр mss на Windows привязан к потоку, в котором создан, поэтому
    источник открывает бэкенд лениво — в потоке, который читает кадры.
    """

    name = "mss"

    def __init__(self):
        import mss

        make = getattr(mss, "MSS", None) or mss.mss
        self.sct = make()
        m = self.sct.monitors[0]
        self.bounds = (m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"])

    def grab(self, rect):
        x0, y0, x1, y1 = rect
        shot = self.sct.grab({"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
        bgra = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(bgra[..., 2::-1])

    def close(self) -> None:
        self.sct.close()


GRABBERS: dict[str, Callable[[], Grabber]] = {"dxcam": DxcamGrabber, "mss": MssGrabber}


def open_grabber(backend: str = "auto") -> tuple[Grabber, str]:
    """Бэкенд захвата и запись, почему именно он.

    "auto" — dxcam, а если он не открылся — mss, с причиной в записи.
    Явно названный бэкенд не подменяется: не открылся — исключение.
    """
    if backend != "auto":
        if backend not in GRABBERS:
            raise ValueError(f"неизвестный бэкенд захвата {backend!r}: {', '.join(BACKENDS)}")
        return GRABBERS[backend](), f"{backend}: задан явно"
    errors = []
    for name in BACKENDS:
        try:
            g = GRABBERS[name]()
        except Exception as exc:                              # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not errors:
            return g, f"{name}: основной бэкенд"
        return g, f"{name}: запасной — " + "; ".join(errors)
    raise RuntimeError("ни один бэкенд захвата не открылся — " + "; ".join(errors))


# ---------------------------------------------------------------- экран


def _inside(rect, bounds) -> bool:
    return (rect[0] >= bounds[0] and rect[1] >= bounds[1]
            and rect[2] <= bounds[2] and rect[3] <= bounds[3])


class ScreenSource:
    """Окно игры (или весь основной монитор при title=None) с ограничением fps.

    `read()` ждёт до следующего такта 1 / fps и отдаёт кадр или отказ:
    окна нет, оно свёрнуто, не в фокусе (пауза: игрок ушёл в другое окно —
    там чужой экран), окно выходит за то, что видит бэкенд, или кадр не
    снялся. Такт выдерживается и для отказов — цикл сеанса не крутится
    вхолостую. Отставание не догоняется пачкой кадров: следующий такт
    отсчитывается от момента снимка, если тот опоздал.

    `windows`, `grabber`, `clock`, `sleep` — подставляются в тестах.
    """

    kind = "screen"

    def __init__(self, fps: float = 2.0, title: str | None = WINDOW_TITLE,
                 backend: str = "auto", windows: Windows | None = None,
                 grabber: Grabber | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if not fps > 0:
            raise ValueError(f"fps должен быть больше нуля: {fps}")
        self.fps, self.title, self.want = float(fps), title, backend
        self._windows, self._grabber = windows, grabber
        self.clock, self.sleep = clock, sleep
        self.backend: str | None = grabber.name if grabber is not None else None
        self.backend_note: str | None = "подставлен" if grabber is not None else None
        self.notes: list[str] = []
        self._hwnd: int | None = None
        self._t0: float | None = None
        self._next: float | None = None
        self._size: tuple[int, int] | None = None
        self._closed = False

    # ---------- открытие ----------

    def open(self) -> None:
        """Окна и бэкенд захвата. Звать в потоке, который будет читать."""
        if self._t0 is not None:
            return
        if self._windows is None and self.title is not None:
            self._windows = Win32Windows()
        note = getattr(self._windows, "note", None)
        if note:
            self.notes.append(note)
        if self._grabber is None:
            self._grabber, self.backend_note = open_grabber(self.want)
            self.backend = self._grabber.name
        self._t0 = self.clock()
        self._size = self._probe_size()

    def _probe_size(self) -> tuple[int, int] | None:
        """Размер кадра, который сейчас дал бы источник, или None."""
        if self.title is None:
            b = self._grabber.bounds
            return b[2] - b[0], b[3] - b[1]
        hwnd = self._windows.find(self.title)
        ws = self._windows.state(hwnd) if hwnd else None
        if ws is None:
            return None
        self._hwnd = hwnd
        x0, y0, x1, y1 = ws.rect
        return (x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None

    @property
    def size(self) -> tuple[int, int] | None:
        return self._size

    # ---------- чтение ----------

    def _tick(self) -> None:
        """Выдержать такт 1 / fps от прошлого снимка."""
        now = self.clock()
        if self._next is not None and now < self._next:
            self.sleep(self._next - now)
            now = self.clock()
        base = now if self._next is None else max(self._next, now)
        self._next = base + 1.0 / self.fps

    def _rect(self, t: float) -> tuple[int, int, int, int] | Frame:
        """Прямоугольник снимка или отказ кадра."""
        if self.title is None:
            return self._grabber.bounds
        ws = self._windows.state(self._hwnd) if self._hwnd else None
        if ws is None:                   # окна не было или его закрыли — ищем заново
            self._hwnd = self._windows.find(self.title)
            ws = self._windows.state(self._hwnd) if self._hwnd else None
        if ws is None:
            return gap(Status.NO_WINDOW, t, f"окна «{self.title}» нет")
        if ws.minimized:
            return gap(Status.MINIMIZED, t)
        if not ws.foreground:
            return gap(Status.PAUSED, t)
        x0, y0, x1, y1 = ws.rect
        if x1 <= x0 or y1 <= y0:
            return gap(Status.NO_FRAME, t, f"пустая клиентская область окна {ws.rect}")
        if not _inside(ws.rect, self._grabber.bounds):
            return gap(Status.NO_FRAME, t,
                       f"окно {ws.rect} выходит за область захвата {self.backend} "
                       f"{self._grabber.bounds}")
        return ws.rect

    def read(self) -> Frame:
        if self._closed:
            return gap(Status.ENDED, 0.0, "источник закрыт")
        self.open()
        self._tick()
        t = self.clock() - self._t0
        rect = self._rect(t)
        if isinstance(rect, Frame):
            return rect
        try:
            img = self._grabber.grab(rect)
        except Exception as exc:                              # noqa: BLE001
            return gap(Status.NO_FRAME, t, f"{self.backend}: {type(exc).__name__}: {exc}")
        if img is None:
            return gap(Status.NO_FRAME, t, f"{self.backend}: нового кадра ещё нет")
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            return gap(Status.NO_FRAME, t,
                       f"{self.backend}: кадр {img.shape} {img.dtype}, а не (H, W, 3) uint8")
        fr = Frame(Status.OK, t, img)
        self._size = fr.size
        return fr

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._grabber is not None:
            self._grabber.close()

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "backend_note": self.backend_note,
                "fps": self.fps, "size": self._size, "title": self.title,
                "notes": list(self.notes)}
