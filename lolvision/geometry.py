"""Геометрия карты, общая для моделей, проверок и смотрелок.

Координаты камеры везде — доли карты в [0, 1], в игровые единицы их
переводит MAP_UNITS. Связь «пиксель хранимой сцены → смещение от центра
камеры» — гомография из dataset/layouts/projection.json (подбирает её
tools/calibrate_projection.py). Перевод точки кадра в точку карты — только
через `Projection`: трекер юнитов и контекст матча берут одну и ту же. Модуль
без torch: его берут и лёгкие скрипты.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lolvision.paths import PROJECTION

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


class Projection:
    """Пиксель кадра ↔ точка карты (доли) при данном центре камеры.

    Камера наклонена: масштаб по высоте кадра меняется на ~28%, и «центр
    камеры плюс доля вида» неверен. Точка кадра переводится в пиксель
    хранимой сцены (обрезка `crop` кадра, уменьшенная до `scene_size`),
    гомография `h` даёт смещение от центра камеры, центр камеры — ответ
    модели камеры. `frame_size` — (ширина, высота) кадра: в долях кадра
    записаны юниты строки live (records.UnitRec)."""

    def __init__(self, h: np.ndarray, scene_size: tuple[int, int],
                 crop: tuple[int, int, int, int], frame_size: tuple[int, int]):
        self.h, self.hi = h, np.linalg.inv(h)
        self.sw, self.sh = scene_size
        self.crop = crop
        self.frame_size = tuple(frame_size)

    @classmethod
    def load(cls, crop: tuple[int, int, int, int], frame_size: tuple[int, int],
             path: Path = PROJECTION) -> Projection:
        """Проекция по гомографии из файла (по умолчанию projection.json)."""
        h, scene_size = load_projection(path)
        return cls(h, scene_size, crop, frame_size)

    def to_map(self, xy: np.ndarray, cam: tuple[float, float]) -> np.ndarray:
        """Пиксели кадра (N, 2) → точки карты (N, 2), доли."""
        cx0, cy0, cx1, cy1 = self.crop
        s = np.c_[(xy[:, 0] - cx0) / (cx1 - cx0) * self.sw,
                  (xy[:, 1] - cy0) / (cy1 - cy0) * self.sh]
        return apply_h(self.h, s) + np.asarray(cam)

    def frac_to_map(self, xy: np.ndarray, cam: tuple[float, float]) -> np.ndarray:
        """Доли кадра (N, 2) → точки карты (N, 2), доли."""
        return self.to_map(np.asarray(xy, np.float64).reshape(-1, 2) * self.frame_size, cam)

    def to_frame(self, uv: np.ndarray, cam: tuple[float, float]) -> np.ndarray:
        """Точки карты (N, 2) → пиксели кадра (N, 2)."""
        cx0, cy0, cx1, cy1 = self.crop
        s = apply_h(self.hi, uv - np.asarray(cam))
        return np.c_[s[:, 0] / self.sw * (cx1 - cx0) + cx0, s[:, 1] / self.sh * (cy1 - cy0) + cy0]
