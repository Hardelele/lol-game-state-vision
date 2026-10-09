"""Геометрия карты, общая для моделей, проверок и смотрелок.

Координаты камеры везде — доли карты в [0, 1], в игровые единицы их
переводит MAP_UNITS. Связь «пиксель хранимой сцены → смещение от центра
камеры» — гомография из dataset/layouts/projection.json (подбирает её
tools/calibrate_projection.py). Модуль без torch: его берут и лёгкие
скрипты.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paths import PROJECTION

MAP_UNITS = 14800      # примерный размер карты League of Legends в игровых единицах


def load_projection(path: Path = PROJECTION) -> tuple[np.ndarray, tuple[int, int]]:
    """Гомография «пиксель хранимой сцены → смещение от центра камеры (доли
    карты)» и размер хранимой сцены, под который она подобрана."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(d["homography"], dtype=np.float64), tuple(d["scene_size"])


def apply_h(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить гомографию к точкам (N, 2)."""
    v = np.c_[pts, np.ones(len(pts))] @ m.T
    return v[:, :2] / v[:, 2:3]
