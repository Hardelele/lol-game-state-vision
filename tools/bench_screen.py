"""Сколько стоит снять кадр с экрана: dxcam (DXGI) против mss (GDI) (#794).

Зачем. Экранный источник live (lolstream/frame_source.py) снимает окно
игры несколько раз в секунду. Время снимка входит в путь кадра и отнимает
у игры процессор и видеокарту — его надо знать до того, как мерить просадку
FPS игры (docs/screen-source.md).

Как. Тот же бэкенд, что у источника (`open_grabber`), тот же прямоугольник:
клиентская область окна `League of Legends (TM) Client`, а если игры нет —
весь основной монитор. Прогрев, затем N снимков подряд без пауз (или с
паузой --every, ближе к живому режиму); печатаются медиана и p95, мс.

У dxcam снимок без изменений на экране — не снимок: DXGI отдаёт кадр,
только когда экран обновился. Поэтому для dxcam время меряется на
`new_frame_only=True`, и пустые ответы считаются отдельно («нового кадра не
было»); статичный рабочий стол даст их много — мерить при движении на
экране (игра, видео).

--fake — подставной бэкенд с заданным временем снимка: проверка самого
скрипта без экрана.

Пример:
    python tools/bench_screen.py --n 300
    python tools/bench_screen.py --backend mss --full --every 0.5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream.frame_source import BACKENDS, WINDOW_TITLE, Win32Windows, open_grabber


class FakeGrabber:
    """Подставной бэкенд: спит `cost` секунд и отдаёт чёрный кадр."""

    name = "fake"

    def __init__(self, cost: float, bounds=(0, 0, 1920, 1080)):
        self.cost, self.bounds, self.cam = cost, bounds, None

    def grab(self, rect):
        time.sleep(self.cost)
        x0, y0, x1, y1 = rect
        return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)

    def close(self) -> None:
        pass


def target_rect(title: str, full: bool, bounds) -> tuple[tuple[int, int, int, int], str]:
    """Прямоугольник снимка и откуда он взят."""
    if full:
        return tuple(bounds), "весь экран захвата (--full)"
    w = Win32Windows()
    hwnd = w.find(title)
    st = w.state(hwnd) if hwnd else None
    if st is None or st.minimized:
        why = "окна нет" if st is None else "окно свёрнуто"
        return tuple(bounds), f"{why} — весь экран захвата"
    note = "окно игры" + ("" if st.foreground else " (не в фокусе: снимается то, что поверх)")
    return st.rect, note


def bench(g, rect, n: int, warm: int, every: float) -> dict:
    """Медиана и p95 времени снимка, мс; пустые ответы — отдельно."""
    cam = getattr(g, "cam", None)
    if g.name == "dxcam" and cam is not None:
        grab = lambda: cam.grab(region=tuple(rect), copy=True, new_frame_only=True)
    else:
        grab = lambda: g.grab(rect)
    for _ in range(warm):
        grab()
    ts, empty, shape = [], 0, None
    for _ in range(n):
        t = time.perf_counter()
        img = grab()
        dt = time.perf_counter() - t
        if img is None:
            empty += 1
        else:
            ts.append(dt)
            shape = img.shape
        if every > 0:
            time.sleep(every)
    out = {"n": n, "frames": len(ts), "no_new_frame": empty, "shape": shape}
    if ts:
        ms = np.array(ts) * 1000
        out.update(median_ms=round(float(np.median(ms)), 2),
                   p95_ms=round(float(np.percentile(ms, 95)), 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", nargs="+", default=list(BACKENDS),
                    choices=list(BACKENDS), help="какие бэкенды мерить")
    ap.add_argument("--n", type=int, default=200, help="снимков на бэкенд")
    ap.add_argument("--warm", type=int, default=10, help="снимков прогрева")
    ap.add_argument("--every", type=float, default=0.0,
                    help="пауза между снимками, с (0 — подряд)")
    ap.add_argument("--title", default=WINDOW_TITLE, help="заголовок окна игры")
    ap.add_argument("--full", action="store_true", help="весь экран, а не окно игры")
    ap.add_argument("--fake", type=float, default=None, metavar="SEC",
                    help="подставной бэкенд с таким временем снимка (без экрана)")
    a = ap.parse_args()

    names = ["fake"] if a.fake is not None else a.backend
    for name in names:
        if name == "fake":
            g, note = FakeGrabber(a.fake), "fake: подставной (--fake)"
            rect, where = g.bounds, "подставной экран"
        else:
            try:
                g, note = open_grabber(name)
            except Exception as exc:                          # noqa: BLE001
                print(f"{name}: не открылся — {type(exc).__name__}: {exc}")
                continue
            rect, where = target_rect(a.title, a.full, g.bounds)
            b = g.bounds
            if not (rect[0] >= b[0] and rect[1] >= b[1] and rect[2] <= b[2] and rect[3] <= b[3]):
                print(f"{name}: окно {rect} выходит за область захвата {b} — не меряю")
                g.close()
                continue
        try:
            r = bench(g, rect, a.n, a.warm, a.every)
        finally:
            g.close()
        size = f"{rect[2] - rect[0]}x{rect[3] - rect[1]}"
        if "median_ms" in r:
            res = f"медиана {r['median_ms']} мс, p95 {r['p95_ms']} мс"
        else:
            res = "ни одного кадра"
        print(f"{name}: {res} · кадров {r['frames']}/{r['n']}, "
              f"нового кадра не было {r['no_new_frame']} · {size} {rect} — {where} · {note}")


if __name__ == "__main__":
    main()
