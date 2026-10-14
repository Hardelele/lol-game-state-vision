"""Полоса статистики HUD в отдельных процессах (офлайн-прогон, #843).

`hud_stats.StatsReader.read_strip` — чистая функция выреза полосы, но
дорогая: ~20 мс на кадр (ближайший сосед по 3183 образцам цифр на каждую
цифру CS и часов). В потоках её держит GIL, поэтому пачка кадров читается в
процессах. Модуль нарочно лёгкий: процесс-работник импортирует только его и
lolvision.hud_stats (numpy, OpenCV), а не torch и весь путь кадра.

Один поток BLAS / OpenMP / OpenCV на работника: работников и так несколько.
"""

from __future__ import annotations

import os

_reader = None


def init() -> None:
    """Инициализатор работника: один поток у numpy и OpenCV."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import cv2
    cv2.setNumThreads(1)


def read(strip):
    """Вырез полосы (BGR, как у Session._stats_step) → hud_stats.Stats."""
    global _reader
    if _reader is None:
        from lolvision.hud_stats import StatsReader
        _reader = StatsReader()
    return _reader.read_strip(strip)
