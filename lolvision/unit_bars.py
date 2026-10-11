"""Тонкие полоски ХП миньонов и вардов на основном виде spectator-ролика.

Зачем. Путь B плана второй модели (docs/entity-model-plan.md, 2.3): HUD
кадра — учитель. Плашки чемпионов читает lolvision/health_bars.py; здесь —
полоски юнитов без квадрата уровня. От полоски к телу (unit_boxes) — по
постоянным смещениям, измеренным на кадрах канала (docs/unit-detection.md).
Псевдо-метки детектора юнитов собираются из обоих читателей
(tools/build_unit_labels.py).

Геометрия при 1920×1080 (кадры dataset/videos, 58w57eJ5Qks_t00300):

* миньон — полоска 53×4 px заливки в тёмной рамке толщиной 1 px; заливка
  цветная слева, пустая часть тёмная. Над полоской ни имени, ни уровня.
  Ширина заливки при полном ХП — пик 51–53 px по 646 кадрам;
* вард — та же ширина, но заливка из 3 (у контрольного варда — 4)
  сегментов ~16 px с просветами, над полоской белая линия таймера.

Поиск — по форме, а не по оттенку. Заливка — насыщенный яркий цвет любого
оттенка; пиксели делятся на корзины оттенка (COLORS), в каждой — компоненты
связности высотой 1–5 строк, сверху и снизу которых темнее (рамка).
Компоненты одной строки в пределах ширины полоски склеиваются (сегменты
варда, перекрытие эффектом). Полоска принимается, если справа от заливки до
конца полоски — ровная тёмная пустая часть, а за её концом светлее (фон):
доля ХП — граница заливки и пустой части, а не порог по оттенку.

Цвет ≠ команда. Найденная полоска несёт корзину цвета, а команду даёт
отдельное решение team_of_color по режиму HUD: у наблюдателя синяя —
ORDER, красная — CHAOS; у вида игрока синие — союзники, красные — враги,
зелёная — своя; режим для дальтоников и свои цвета полосок меняют красный
на жёлтый и т. п. Проверен только режим наблюдателя канала (синие/красные);
остальные корзины находятся, но команда для них не назначается.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BAR_W = 53            # ширина заливки при полном ХП, px при 1080p
SEG_W = (12, 19)      # ширина сегмента варда
MAX_FILL_H = 5        # толще — не тонкая полоска (у чемпиона ХП 10 строк)
EDGE = 25             # рамка темнее заливки хотя бы на столько по яркости
MIN_FILL = 5          # заливка уже — неотличима от бликов травы и кромок карточек


@dataclass
class UnitBar:
    kind: str           # "minion" / "ward"
    team: str           # "ORDER" / "CHAOS" / "" — по team_of_color
    x: int              # левый край заливки
    y: int              # верхняя строка заливки
    h: int              # высота заливки, строк
    fill: int           # ширина заливки, px
    segments: int = 1
    color: str = ""     # корзина оттенка заливки (COLORS)

    @property
    def hp(self) -> float:
        return min(1.0, self.fill / BAR_W)

    @property
    def cx(self) -> float:
        """Середина полоски (не заливки): от ХП не зависит."""
        return self.x + BAR_W / 2


# Корзины оттенка (OpenCV: H 0..180). Красный переходит через 0.
COLORS = {"red": ((0, 12), (165, 180)), "yellow": ((13, 34),), "green": ((35, 85),),
          "blue": ((86, 130),), "purple": ((131, 164),)}
MIN_SAT, MIN_VAL = 80, 105     # заливка: насыщенность и яркость (0..255)

# Цвет → команда по режиму HUD. Проверен только "spectator".
TEAM_BY_COLOR = {
    "spectator": {"blue": "ORDER", "red": "CHAOS"},
    "player": {"green": "SELF", "blue": "ALLY", "red": "ENEMY"},
}


def color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Насыщенные яркие пиксели по корзинам оттенка."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (sat >= MIN_SAT) & (val >= MIN_VAL)
    out = {}
    for name, ranges in COLORS.items():
        m = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h <= hi)
        out[name] = vivid & m
    return out


def team_of_color(color: str, mode: str = "spectator") -> str:
    return TEAM_BY_COLOR.get(mode, {}).get(color, "")


def _components(mask: np.ndarray, lum: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Тонкие компоненты маски с тёмной рамкой сверху, снизу и слева:
    (x, y, ширина, высота)."""
    h, w = lum.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    out = []
    for x, y, ww, hh, _ in st[1:]:
        if not (1 <= hh <= MAX_FILL_H and 2 <= ww <= BAR_W + 6):
            continue
        if y < 3 or y + hh + 2 >= h or x < 3 or x + ww + 2 >= w:
            continue
        mid = lum[y:y + hh, x:x + ww].mean()
        above = lum[y - 2:y, x:x + ww].min(0).mean()
        below = lum[y + hh:y + hh + 2, x:x + ww].min(0).mean()
        if above < mid - EDGE and below < mid - EDGE:
            out.append((int(x), int(y), int(ww), int(hh)))
    return out


def _group(comps: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Склейка компонент одной полоски: та же строка (±1), левый край группы
    не дальше ширины полоски."""
    comps = sorted(comps, key=lambda c: (c[0], c[1]))
    groups: list[list[tuple[int, int, int, int]]] = []
    for c in comps:
        for g in groups:
            x0 = g[0][0]
            if abs(g[0][1] - c[1]) <= 1 and c[0] - x0 <= BAR_W - 4 and \
                    c[0] - (g[-1][0] + g[-1][2]) <= 4:
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


def tail_ok(lum: np.ndarray, x: int, y: int, hh: int, fill: int, mid: float) -> bool:
    """Пустая часть неполной полоски: ровная тёмная лента до конца полоски,
    а за концом — светлее (фон). Пороги — по 646 кадрам dataset/videos:
    у полосок с ХП 30–95% разброс яркости ленты (медиана) 6, ступень на
    конце +30; у ложных находок с заливкой в 2–4 px — 16 и 0."""
    x0, x1 = x + fill + 1, x + BAR_W - 1
    if x1 - x0 < 3:
        return True
    if x + BAR_W + 3 > lum.shape[1]:
        return False
    tail = lum[y:y + hh, x0:x1]
    col = tail.mean(0)
    if float((col < min(mid - 30, 90)).mean()) < 0.75:
        return False
    if tail.std() > 12 or tail.mean() > 55:
        return False
    end = lum[y:y + hh, x + BAR_W + 1:x + BAR_W + 3].mean() - lum[y:y + hh, x1 - 2:x1].mean()
    return end > 12


def _is_ward(g: list[tuple[int, int, int, int]], rgb: np.ndarray) -> bool:
    """Вард: ≥2 сегмента ширины SEG_W с просветами и белая линия таймера над
    полоской."""
    segs = [c for c in g if SEG_W[0] <= c[2] <= SEG_W[1]]
    if len(segs) < 2 or len(segs) < len(g) - 1:
        return False
    x0, y = g[0][0], g[0][1]
    x1 = g[-1][0] + g[-1][2]
    line = rgb[max(0, y - 4):y - 1, x0:x1].astype(np.int16)
    return bool(len(line)) and float((line.min(2) > 140).any(0).mean()) > 0.3


def find_unit_bars(rgb: np.ndarray, keep: np.ndarray | None = None,
                   exclude: list[tuple[int, int, int, int]] | None = None,
                   mode: str = "spectator") -> list[UnitBar]:
    """Полоски миньонов и вардов на кадре 1080p.

    `keep` — маска сцены (1 — можно искать), `exclude` — прямоугольники
    (x0, y0, x1, y1), где полоски не ищутся: плашки чемпионов (полоса ресурса
    чемпиона по форме — та же тонкая полоска)."""
    lum = rgb.astype(np.float32).mean(2)
    masks = color_masks(rgb)
    out = []
    for color, m in masks.items():
        if keep is not None:
            m = m & (keep > 0)
        if exclude:
            m = m.copy()
            for x0, y0, x1, y1 in exclude:
                m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        for g in _group(_components(m, lum)):
            x0, y = g[0][0], min(c[1] for c in g)
            hh = max(c[1] + c[3] for c in g) - y
            fill = max(c[0] + c[2] for c in g) - x0
            if fill > BAR_W + 4:
                continue
            ward = _is_ward(g, rgb)
            mid = float(np.mean([lum[c[1]:c[1] + c[3], c[0]:c[0] + c[2]].mean() for c in g]))
            if not ward and (fill < MIN_FILL or not tail_ok(lum, x0, y, hh, fill, mid)):
                continue
            # заливка выше полоски цвета команды — это ресурс под ХП чемпиона
            above = m[max(0, y - 9):max(0, y - 3), x0:x0 + fill]
            if above.size and above.mean() > 0.4:
                continue
            out.append(UnitBar("ward" if ward else "minion", team_of_color(color, mode),
                               x0, y, hh, fill, len(g) if ward else 1, color))
    return out


# --- от полоски к телу -------------------------------------------------------
# Смещения тела относительно полоски при 1080p: (dx центра от середины
# полоски, dy верха тела от верха полоски, ширина, высота). Оценены по
# средней карте краёв (Собель) в вырезах, выровненных по полоске: 618
# полных полосок миньонов и 1010 плашек чемпионов dataset/videos
# (docs/unit-detection.md); варды — кольцо под полоской, по листам.
BODY = {
    "minion": (0, 10, 64, 62),
    "ward": (0, 6, 40, 34),
    "champion": (0, 30, 120, 150),
}


def body_box(kind: str, bar_cx: float, bar_y: float) -> tuple[float, float, float, float]:
    """Рамка тела (x0, y0, x1, y1) по середине и верху полоски."""
    dx, dy, w, h = BODY[kind]
    cx = bar_cx + dx
    return cx - w / 2, bar_y + dy, cx + w / 2, bar_y + dy + h
