"""Плашки чемпионов на основном виде spectator-ролика: поиск и чтение.

Зачем. Классический читатель плашек — этап 1 второй модели
(docs/entity-model-plan.md): базовая точка и источник псевдо-меток. Над
каждым видимым чемпионом канал рисует плашку: квадрат уровня слева,
полоса ХП цвета команды, под ней полоса ресурса; над плашкой имя
чемпиона.

Геометрия при 1920×1080 (кадр 58w57eJ5Qks_t00300, плашка Irelia):
рамка 123×23 px с полем; внутри — квадрат уровня x 3..22, полоса ХП
x 25..118 и y 4..13 (10 строк), ресурс y 16..18. Координаты ниже — от
левого верхнего угла шаблона.

Поиск: нормированная корреляция с шаблоном рамки, где заливка ХП,
ресурс и цифра замаскированы — от ХП и команды отклик не зависит.
Шаблон — медиана уверенных находок по кадрам dataset/videos
(dataset/health_bar_template.npz).

Уровень: ближайший сосед по квадрату уровня целиком среди проверенных
вручную образцов (dataset/health_bar_levels.npz, 197 шт., уровни 1–18).
Деление на отдельные цифры по порогу ненадёжно: две цифры сливаются, а
тусклая цифра на красном фоне пропадает.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lolvision.paths import DATASET

TEMPLATE = DATASET / "health_bar_template.npz"
LEVELS = DATASET / "health_bar_levels.npz"
W, H = 123, 23
HP_X0, HP_X1 = 25, 119
HP_Y0, HP_Y1 = 4, 14
RES_Y0, RES_Y1 = 16, 19
LEVEL_BOX = (3, 2, 23, 21)          # x0, y0, x1, y1
NAME_BOX = (-10, -17, W + 10, 0)    # над плашкой, с запасом по ширине
MIN_SCORE = 0.45      # кандидат; дальше проверка квадрата уровня (verify)
SURE_SCORE = 0.75     # выше — плашка без проверки


@dataclass
class Bar:
    x: int                  # левый верхний угол шаблона в кадре
    y: int
    score: float
    team: str = ""          # "ORDER" (синие) / "CHAOS" (красные)
    hp: float = float("nan")         # доля заливки ХП 0..1
    resource: float = float("nan")   # доля ресурса 0..1 (нет ресурса — nan)
    level: int = 0                   # 0 — не прочитан
    level_sim: float = float("nan")  # сходство с ближайшим образцом


def load_template(path: Path = TEMPLATE) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["template"].astype(np.float32), z["mask"].astype(np.float32)


def response(gray: np.ndarray, tmpl: np.ndarray, mask: np.ndarray,
             keep: np.ndarray | None = None) -> np.ndarray:
    """Карта отклика: значение в (y, x) — шаблон с левым верхним углом там.
    `keep` — маска сцены кадра (1 — можно искать): плашка целиком в сцене."""
    r = cv2.matchTemplate(gray.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    if keep is not None:
        h, w = r.shape
        ok = keep[:h, :w] & keep[H - 1:H - 1 + h, W - 1:W - 1 + w]
        r = np.where(ok > 0, r, -1.0)
    return r


def peaks(r: np.ndarray, thr: float = MIN_SCORE) -> list[tuple[int, int, float]]:
    """Локальные максимумы выше порога. Подавление — на размер плашки: шаблон
    второй раз совпадает со сдвигом ~16 px вниз (рамка ХП ложится на рамку
    ресурса), такой дубль не должен считаться второй плашкой."""
    r = r.copy()
    out = []
    while True:
        y, x = np.unravel_index(int(np.argmax(r)), r.shape)
        if r[y, x] < thr:
            return out
        out.append((int(x), int(y), float(r[y, x])))
        r[max(0, y - H):y + H + 1, max(0, x - W // 2):x + W // 2 + 1] = -1.0


def verify(rgb: np.ndarray, x: int, y: int) -> bool:
    """Квадрат уровня: тёмный фон и яркая цифра. Отсекает полосы строений и
    монстров (золотая рамка, нет квадрата) и тёмные пустые участки сцены.
    Пороги — по находкам на dataset/videos: у плашек 14–31 ярких пикселей
    цифры и 75–90% тёмного фона, у ложных — 0 ярких или светлый фон."""
    box = rgb[y + 4:y + 19, x + 6:x + 21].astype(np.int16)
    if box.shape[:2] != (15, 15):
        return False
    bright = int((box.min(2) > 160).sum())
    dark = float((box.max(2) < 110).mean())
    return bright >= 6 and dark >= 0.6


def fill_fraction(band: np.ndarray, color: str) -> float:
    """Доля заливки полосы (строки × столбцы × RGB) слева направо.

    Столбец залит, если цвет команды заметно ярче фона. Засечки (тёмные
    вертикальные чёрточки внутри заливки) не обрывают заливку: берётся
    крайний правый залитый столбец, за которым не меньше 4 пустых подряд."""
    b = band.astype(np.int16)
    r, bl = b[..., 0], b[..., 2]
    if color == "red":
        on = (r > 110) & (r - bl > 50)
    elif color == "blue":
        on = (bl > 120) & (bl - r > 40)
    else:                                   # ресурс: любой яркий цвет
        on = b.max(2) > 90
    col = on.mean(0) > 0.5
    n = len(col)
    last = -1
    gap = 0
    for i, c in enumerate(col):
        if c:
            last, gap = i, 0
        else:
            gap += 1
            if gap >= 4 and last >= 0:
                break
    return (last + 1) / n


def team_of(rgb: np.ndarray, x: int, y: int) -> str:
    """Команда по фону квадрата уровня: тёмно-синий у синих, тёмно-красный у красных
    (виден и при пустой полосе ХП)."""
    x0, y0, x1, y1 = LEVEL_BOX
    box = rgb[y + y0 + 1:y + y1 - 1, x + x0 + 1:x + x1 - 1].astype(np.int16)
    dark = box[box.max(2) < 120]           # без белой цифры
    if not len(dark):
        return ""
    r, _, b = dark.mean(0)
    return "CHAOS" if r > b else "ORDER"


def read(rgb: np.ndarray, x: int, y: int, score: float) -> Bar:
    bar = Bar(x, y, score, team=team_of(rgb, x, y))
    color = "red" if bar.team == "CHAOS" else "blue"
    bar.hp = fill_fraction(rgb[y + HP_Y0:y + HP_Y1, x + HP_X0:x + HP_X1], color)
    res = rgb[y + RES_Y0:y + RES_Y1, x + HP_X0:x + HP_X1]
    bar.resource = fill_fraction(res, "any")
    return bar


def crop(rgb: np.ndarray, bar: Bar, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = rgb.shape[:2]
    return rgb[max(0, bar.y + y0):min(h, bar.y + y1), max(0, bar.x + x0):min(w, bar.x + x1)]


# Квадрат уровня в координатах шаблона и запас на сдвиг при сравнении.
LV_X0, LV_Y0, LV_W, LV_H = 5, 3, 16, 17
LV_PAD = 2


def _norm(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return a / (np.linalg.norm(a, axis=(-2, -1), keepdims=True) + 1e-6)


class LevelReader:
    """Уровень по образцам: сходство квадрата (минимум по каналам — белая
    цифра ярка во всех, фон команды нет) со сдвигом ±2 px."""

    def __init__(self, path: Path = LEVELS):
        z = np.load(path)
        self.bank = _norm(z["boxes"].astype(np.float32))
        self.levels = z["levels"].astype(int)

    def read(self, rgb: np.ndarray, x: int, y: int) -> tuple[int, float]:
        p = LV_PAD
        big = rgb[y + LV_Y0 - p:y + LV_Y0 + LV_H + p,
                  x + LV_X0 - p:x + LV_X0 + LV_W + p].astype(np.float32).min(2)
        if big.shape != (LV_H + 2 * p, LV_W + 2 * p):
            return 0, float("nan")
        wins = np.stack([big[dy:dy + LV_H, dx:dx + LV_W]
                         for dy in range(2 * p + 1) for dx in range(2 * p + 1)])
        sims = np.einsum("shw,bhw->sb", _norm(wins), self.bank).max(0)
        j = int(np.argmax(sims))
        return int(self.levels[j]), float(sims[j])


def find_bars(rgb: np.ndarray, tmpl: np.ndarray, mask: np.ndarray,
              keep: np.ndarray | None = None, thr: float = MIN_SCORE,
              levels: LevelReader | None = None) -> list[Bar]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bars = [read(rgb, x, y, s) for x, y, s in peaks(response(gray, tmpl, mask, keep), thr)
            if s >= SURE_SCORE or verify(rgb, x, y)]
    if levels is not None:
        for b in bars:
            b.level, b.level_sim = levels.read(rgb, b.x, b.y)
    return bars
