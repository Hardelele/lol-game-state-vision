"""«Кто это» для плашки на основном виде: вырезы имени и облика.

Зачем. Плашку находит lolvision/health_bars.py (место, команда, уровень).
Чемпиона над ней определяют три источника: состав матча по HUD
(lolvision/roster.py), имя над плашкой (lolvision/name_reader.py) и облик
под плашкой (lolvision/appearance.py). Здесь — общая геометрия вырезов,
чтобы сбор данных, обучение и чтение резали одинаково.

Геометрия при 1920×1080 от левого верхнего угла шаблона плашки (x, y):
* имя — строка над плашкой, x −12..W+12, y −19..0 (147×19 px): текст
  центрирован над плашкой, длинные имена шире самой плашки;
* облик — квадрат 192×192 px с центром по горизонтали над серединой
  плашки, сверху на 12 px ниже её верха: модель стоит под плашкой (на
  опорном кадре тела Volibear, Irelia, Hecarim — от 30 до 220 px под
  плашкой). Хранится уменьшенным до 128×128.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from lolvision.health_bars import W

NAME_BOX = (-12, -19, W + 12, 0)
BODY_SIDE = 192
BODY_BOX = (W // 2 - BODY_SIDE // 2, 12, W // 2 + BODY_SIDE // 2, 12 + BODY_SIDE)
BODY_OUT = 128


def crop_pad(rgb: np.ndarray, x: int, y: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Вырез `box` (x0, y0, x1, y1 от точки x, y); за краем кадра — нули."""
    x0, y0, x1, y1 = box
    h, w = rgb.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + rgb.shape[2:], rgb.dtype)
    ax0, ay0 = max(0, x + x0), max(0, y + y0)
    ax1, ay1 = min(w, x + x1), min(h, y + y1)
    if ax1 > ax0 and ay1 > ay0:
        out[ay0 - (y + y0):ay1 - (y + y0), ax0 - (x + x0):ax1 - (x + x0)] = rgb[ay0:ay1, ax0:ax1]
    return out


def name_crop(rgb: np.ndarray, x: int, y: int) -> np.ndarray:
    return crop_pad(rgb, x, y, NAME_BOX)


def body_crop(rgb: np.ndarray, x: int, y: int) -> np.ndarray:
    c = crop_pad(rgb, x, y, BODY_BOX)
    return cv2.resize(c, (BODY_OUT, BODY_OUT), interpolation=cv2.INTER_AREA)


# --- объединение: кадр и соседние кадры ---------------------------------

def assign_distinct(scores: np.ndarray) -> np.ndarray:
    """Плашки одной команды на одном кадре — разные чемпионы.

    scores (плашки, кандидаты) — чем больше, тем вероятнее. Возвращает
    номер кандидата для каждой плашки (венгерский алгоритм; плашек не
    больше пяти, кандидатов пять)."""
    if len(scores) == 0:
        return np.zeros(0, int)
    rows, cols = linear_sum_assignment(-scores)
    out = np.full(len(scores), -1, int)
    out[rows] = cols
    left = out < 0                      # плашек больше кандидатов — быть не должно
    out[left] = scores[left].argmax(1)
    return out


def neighbours(frame: np.ndarray, x: np.ndarray, y: np.ndarray, team: np.ndarray,
               level: np.ndarray, gap: int = 1, max_px: float = 350.0) -> list[list[int]]:
    """Для каждой плашки — плашки соседних кадров (±gap) «с той же плашкой»:
    та же команда, уровень отличается не больше чем на 1, расстояние не
    больше max_px. Из нескольких подходящих в кадре берётся ближайшая.

    Это замена трекеру при редких кадрах (0.5 кадра/с): камера наблюдателя
    сама ходит за боем, так что плашка за 2 с смещается на сотни пикселей,
    и сопоставление по месту ненадёжно — поэтому только как голос, не как
    жёсткая связь."""
    by_frame: dict[int, list[int]] = {}
    for i, f in enumerate(frame):
        by_frame.setdefault(int(f), []).append(i)
    out = []
    for i, f in enumerate(frame):
        near = []
        for g in range(-gap, gap + 1):
            if g == 0:
                continue
            best, bd = -1, max_px
            for j in by_frame.get(int(f) + g, []):
                if team[j] != team[i] or abs(int(level[j]) - int(level[i])) > 1:
                    continue
                d = float(np.hypot(x[j] - x[i], y[j] - y[i]))
                if d <= bd:
                    best, bd = j, d
            if best >= 0:
                near.append(best)
        out.append(near)
    return out
