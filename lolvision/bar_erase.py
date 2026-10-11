"""Стирание, перенос и перекраска полосок ХП на кадре — для проверки и
обучения детектора юнитов без опоры на полоски (#772, #773).

Зачем. Детектор юнитов (unit_net) учился на псевдо-метках полосок, и тело
у него всегда стоит под полоской. Если сеть выучила «полоска = юнит», она
пропустит юнита с перекрытой полоской (крип за плашкой чемпиона) и найдёт
юнита там, где полоска без тела. Здесь — операции над пикселями, общие для
замера (tools/eval_units_bars.py) и аугментаций (tools/train_units.py):

* `erase` — закрасить полоску окружающим фоном (cv2.inpaint по вырезу);
* `paste` — перенести вырез полоски в другое место кадра (перекрытие,
  полоска без тела);
* `recolor` — повернуть оттенок заливки (свои цвета полосок, режим для
  дальтоников).

Координаты — пиксели кадра; `scale` — во сколько раз кадр меньше 1080p.
"""

from __future__ import annotations

import cv2
import numpy as np

# Расширение прямоугольника метки `bar` до всего, что рисует интерфейс над
# юнитом (px при 1080p): влево, вверх, вправо, вниз. У чемпиона метка —
# плашка 123×23, над ней имя (~14 px) и надписи состояния; у миньона и варда
# метка — заливка 53×3, вокруг тёмная рамка 1–2 px, у варда сверху белая
# линия таймера.
PAD = {"champion": (10, 20, 10, 4), "minion": (3, 3, 4, 4), "ward": (3, 7, 4, 4)}


def bar_rect(u: dict, scale: float = 1.0, shape: tuple[int, int] | None = None
             ) -> tuple[int, int, int, int]:
    """Прямоугольник (x0, y0, x1, y1) всего, что интерфейс рисует над юнитом."""
    l, t, r, b = PAD[u["kind"]]
    x0, y0, x1, y1 = u["bar"]
    out = [int((x0 - l) * scale), int((y0 - t) * scale),
           int(np.ceil((x1 + r) * scale)), int(np.ceil((y1 + b) * scale))]
    if shape is not None:
        h, w = shape[:2]
        out = [max(0, out[0]), max(0, out[1]), min(w, out[2]), min(h, out[3])]
    return tuple(out)


def erase(img: np.ndarray, rects: list[tuple[int, int, int, int]], radius: int = 3,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """Закрасить прямоугольники фоном вокруг них (на месте, возвращает img).

    cv2.inpaint по вырезу с полем 8 px; поверх — слабый шум уровня соседних
    пикселей, чтобы не оставлять гладкое пятно, по которому сеть узнала бы
    место стёртой полоски."""
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in rects:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        m = 8
        cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
        roi = np.ascontiguousarray(img[cy0:cy1, cx0:cx1])
        mask = np.zeros(roi.shape[:2], np.uint8)
        mask[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = 255
        out = cv2.inpaint(roi, mask, radius, cv2.INPAINT_TELEA)
        if rng is not None:
            ring = roi[mask == 0].astype(np.float32)
            sd = float(ring.std(0).mean()) if len(ring) else 0.0
            sub = out[mask > 0].astype(np.float32)
            sub += rng.standard_normal(sub.shape).astype(np.float32) * min(sd, 12.0) * 0.5
            out[mask > 0] = np.clip(sub, 0, 255).astype(np.uint8)
        img[cy0:cy1, cx0:cx1] = out
    return img


def paste(img: np.ndarray, src: np.ndarray, x: int, y: int) -> tuple[int, int, int, int] | None:
    """Наложить вырез `src` левым верхним углом в (x, y), с обрезкой по
    краю кадра. Возвращает занятый прямоугольник или None."""
    h, w = img.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + sw), min(h, y + sh)
    if x1 <= x0 or y1 <= y0:
        return None
    img[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]
    return x0, y0, x1, y1


def recolor(img: np.ndarray, rect: tuple[int, int, int, int], hue_shift: int,
            min_sat: int = 80) -> np.ndarray:
    """Повернуть оттенок насыщенных пикселей в прямоугольнике на `hue_shift`
    (шкала OpenCV 0–179): заливка полоски меняет цвет, тёмная рамка и пустая
    часть — нет."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, rect[0]), max(0, rect[1]), min(w, rect[2]), min(h, rect[3])
    if x1 <= x0 or y1 <= y0:
        return img
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1] >= min_sat
    hsv[..., 0][sat] = ((hsv[..., 0][sat].astype(np.int32) + hue_shift) % 180).astype(np.uint8)
    img[y0:y1, x0:x1] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def free_spot(rng: np.random.Generator, size: tuple[int, int], valid: np.ndarray,
              avoid: list[tuple[float, float]], min_dist: float, tries: int = 30
              ) -> tuple[int, int] | None:
    """Случайный левый верхний угол прямоугольника `size` (w, h), у которого
    центр в `valid` (H, W) и дальше `min_dist` от всех точек `avoid`."""
    h, w = valid.shape
    sw, sh = size
    for _ in range(tries):
        x = int(rng.integers(0, max(1, w - sw)))
        y = int(rng.integers(0, max(1, h - sh)))
        cx, cy = x + sw // 2, y + sh // 2
        if not valid[min(h - 1, cy), min(w - 1, cx)]:
            continue
        if all((cx - ax) ** 2 + (cy - ay) ** 2 >= min_dist ** 2 for ax, ay in avoid):
            return x, y
    return None
