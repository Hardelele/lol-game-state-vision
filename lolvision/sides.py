"""Свои, враги, сам — и сторона ORDER/CHAOS отдельно (#790).

Зачем. Детектор юнитов (unit_net.py) отдаёт `team` ORDER/CHAOS по логиту,
выученному на наблюдателе канала, где синяя полоска — ORDER, красная —
CHAOS. Значит, `team` детектора — это на деле «синий или красный по выучке
наблюдателя», а не сторона. На виде игрока цвета относительны: союзники
синие, враги красные, своя плашка зелёная (у дальтоников — жёлтая), и
сторона из цвета не следует вовсе: за CHAOS враги тоже красные. Здесь —
отношение юнита к игроку (`relation`: ally / enemy / self) и сторона
(`side`: ORDER / CHAOS), каждое — значение или None с причиной.

Правило (таблица случаев — docs/sides.md):

* вид наблюдателя: relation = None («вид наблюдателя»); side = `team`
  детектора (проверено на всей оценке наблюдателя, 0.998);
* вид игрока: цвет — проба пикселей полоски над юнитом (`probe_color`),
  а не `team` детектора; relation = unit_bars.TEAM_BY_COLOR["player"][цвет];
  цвет вне таблицы (жёлтый, фиолетовый) или не прочитан → None с причиной;
  side = команда своего игрока из Live Client API для self/ally и другая —
  для enemy; нет API → side None («нет команды своего игрока»);
* вид неизвестен → оба None.

Почему не `team` детектора на виде игрока: на 30 кадрах стримов
(runs/units/eval/player) зелёные плашки (14) получили ORDER 12 раз и CHAOS
2 раза, жёлтые (3) — CHAOS: детектор не отличает «свой» и уверенно
называет его союзником или врагом. Цвет полоски в пикселях
различает все три (docs/sides.md, «Проба цвета»).

Цвета и их смысл — только unit_bars.COLORS и unit_bars.TEAM_BY_COLOR;
здесь своих таблиц цветов нет.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from lolvision.unit_bars import BAR_W, BODY, EDGE, TEAM_BY_COLOR, color_masks

# Стороны — значения таблицы наблюдателя (синий ORDER, красный CHAOS).
SIDES = tuple(TEAM_BY_COLOR["spectator"].values())
# `team` детектора → цвет, которому он выучен (обратная таблица наблюдателя).
DET_COLOR = {side: color for color, side in TEAM_BY_COLOR["spectator"].items()}
VIEWS = ("spectator", "player")

# Причины отказа — постоянные строки: по ним считают доли отказов (2D).
WHY_SPECTATOR = "вид наблюдателя: своего игрока нет"
WHY_VIEW = "вид неизвестен"
WHY_NO_FRAME = "нет кадра для пробы цвета"
WHY_NO_BAR = "полоска не прочитана"
WHY_AMBIGUOUS = "цвет полоски неоднозначен"
WHY_COLOR = "цвет вне ожидаемого"          # + ": <корзина>"
WHY_NO_API = "нет команды своего игрока"   # + ": <причина API>"
WHY_DET_TEAM = "детектор не дал команду"
WHY_DISAGREE = "цвет полоски и ответ детектора расходятся"


@dataclass(frozen=True)
class SideContext:
    """Что известно о виде и своём игроке на этот кадр (live.py).

    view — "spectator" / "player" / None (не решён: live_champions.view);
    my_team — команда своего игрока из Live Client API (ORDER / CHAOS) или
    None, тогда my_team_reason — почему (нет API, нет activePlayer…)."""
    view: str | None
    my_team: str | None = None
    my_team_reason: str | None = None


@dataclass(frozen=True)
class Sides:
    relation: str | None           # ally / enemy / self
    relation_reason: str | None
    side: str | None               # ORDER / CHAOS
    side_reason: str | None


def other_side(side: str) -> str:
    a, b = SIDES
    return b if side == a else a


def decide(ctx: SideContext, kind: str, det_team: str, color: str | None,
           color_reason: str | None = None) -> Sides:
    """Отношение и сторона одного юнита.

    `det_team` — `team` детектора (на наблюдателе — сторона), `color` —
    корзина цвета полоски по пикселям (probe_color) или None с
    `color_reason`. Цвет используется только на виде игрока; там `team`
    детектора — лишь проверка: синий/красный пробы против синего/красного
    детектора, расхождение — отказ."""
    if ctx.view not in VIEWS:
        return Sides(None, WHY_VIEW, None, WHY_VIEW)
    if ctx.view == "spectator":
        if det_team in SIDES:
            return Sides(None, WHY_SPECTATOR, det_team, None)
        return Sides(None, WHY_SPECTATOR, None, WHY_DET_TEAM)
    # Вид игрока: отношение — по цвету полоски, сторона — от своего игрока.
    if color is None:
        why = color_reason or WHY_NO_BAR
        return Sides(None, why, None, why)
    role = TEAM_BY_COLOR["player"].get(color)
    if role is None:
        why = f"{WHY_COLOR}: {color}"
        return Sides(None, why, None, why)
    relation = role.lower()
    if relation == "self" and kind != "champion":
        # Своя полоска — только у своего чемпиона; зелёная у миньона —
        # трава, эффект или чужой интерфейс, а не «сам».
        why = f"{WHY_COLOR}: {color} у {kind}"
        return Sides(None, why, None, why)
    seen = DET_COLOR.get(det_team)
    if color in DET_COLOR.values() and seen is not None and seen != color:
        return Sides(None, WHY_DISAGREE, None, WHY_DISAGREE)
    if ctx.my_team not in SIDES:
        why = WHY_NO_API + (f": {ctx.my_team_reason}" if ctx.my_team_reason else "")
        return Sides(relation, None, None, why)
    side = other_side(ctx.my_team) if relation == "enemy" else ctx.my_team
    return Sides(relation, None, side, None)


# ---------- проба цвета полоски по пикселям ----------
# Заливка полоски — вытянутая по горизонтали плотная полоса насыщенного
# цвета; тело юнита, эффекты и трава — пятна другой формы. Цвет тела не
# годится: тела миньонов окрашены по стороне (ORDER синие, CHAOS красные),
# а не по отношению к игроку.
#
# Окна — px при 1080p от точки тела: (x0, y0, x1, y1), высота заливки
# (строк) и наименьшая ширина.
# * Чемпион: плашка вида игрока (30 кадров runs/units/eval/player) —
#   заливка ХП высотой 9–13 строк, заполненная на 0.7–0.95, над точкой тела
#   на 60–170 px; под ней полоска ресурса в 4–5 строк (синяя у маны — её
#   отсекает высота, иначе свой стал бы «союзником»).
# * Миньон и вард: полоска 53×4 (unit_bars), её место — по unit_bars.BODY.
CH_WIN = (-90, -170, 90, -50)
CH_H, CH_MIN_W = (6, 16), 12
BAR_H, BAR_MIN_W = (2, 7), 8
BAR_PAD = (12, 14)                  # запас окна полоски по x и y, px при 1080p
FILL = 0.6                          # доля закрашенного в рамке компоненты
DOMINANCE = 2.0                     # первая корзина шире второй во столько раз


def _band_color(rgb: np.ndarray, box: tuple[float, float, float, float],
                hr: tuple[float, float], min_w: float,
                framed: bool) -> tuple[str | None, str | None]:
    """Корзина цвета с самой широкой горизонтальной заливкой в окне `box`
    (пиксели кадра): компоненты высотой в `hr`, шириной не меньше `min_w`
    и не меньше высоты, плотные; `framed` — ещё и в тёмной рамке сверху и
    снизу. Две корзины сопоставимой ширины — отказ."""
    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(w, box[2])), int(min(h, box[3]))
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None, WHY_NO_BAR
    crop = rgb[y0:y1, x0:x1]
    lum = crop.astype(np.float32).mean(2)
    score = {}
    for name, m in color_masks(crop).items():
        _, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
        tot = 0
        for cx, cy, ww, hh, area in st[1:]:
            if not (hr[0] <= hh <= hr[1] and ww >= max(min_w, hh) and area >= FILL * ww * hh):
                continue
            if not framed:
                tot += int(ww)
                continue
            if cy < 2 or cy + hh + 2 > lum.shape[0]:
                continue
            # Рамка полоски: строки над и под заливкой темнее неё (как в
            # unit_bars._components); у травы и бликов рамки нет. Плашке
            # чемпиона не нужна: её отличает высота, а красная заливка
            # тёмная, и рамка с ней почти не контрастирует.
            mid = lum[cy:cy + hh, cx:cx + ww].mean()
            if lum[cy - 2:cy, cx:cx + ww].min(0).mean() < mid - EDGE and \
                    lum[cy + hh:cy + hh + 2, cx:cx + ww].min(0).mean() < mid - EDGE:
                tot += int(ww)
        if tot:
            score[name] = tot
    if not score:
        return None, WHY_NO_BAR
    best = sorted(score.items(), key=lambda kv: -kv[1])
    if len(best) > 1 and best[0][1] < DOMINANCE * best[1][1]:
        return None, WHY_AMBIGUOUS
    return best[0][0], None


def probe_color(rgb: np.ndarray, kind: str, x: float, y: float) -> tuple[str | None, str | None]:
    """Корзина цвета (unit_bars.COLORS) полоски над юнитом в точке тела
    (x, y — доли кадра) или (None, причина). Кадр любого размера: геометрия
    1080p масштабируется по высоте."""
    h, w = rgb.shape[:2]
    s = h / 1080
    px, py = x * w, y * h
    if kind == "champion":
        a, b, c, d = CH_WIN
        return _band_color(rgb, (px + a * s, py + b * s, px + c * s, py + d * s),
                           (CH_H[0] * s, CH_H[1] * s), CH_MIN_W * s, framed=False)
    if kind not in BODY:
        return None, WHY_NO_BAR
    dx, dy, _, bh = BODY[kind]
    # Середина и верх полоски: тело (x, y) — центр рамки unit_bars.body_box.
    bx, by = px - dx * s, py - (dy + bh / 2) * s
    half = (BAR_W / 2 + BAR_PAD[0]) * s
    return _band_color(rgb, (bx - half, by - BAR_PAD[1] * s, bx + half, by + BAR_PAD[1] * s),
                       (BAR_H[0] * s, BAR_H[1] * s), BAR_MIN_W * s, framed=True)


def assign(units: list, ctx: SideContext, rgb: np.ndarray | None = None) -> list:
    """Записи UnitRec → те же записи с relation/side и причинами.

    Проба цвета идёт только на виде игрока: у наблюдателя сторона — ответ
    детектора. Без кадра на виде игрока — отказ, а не `team` детектора."""
    out = []
    for u in units:
        color, why = None, None
        if ctx.view == "player":
            if rgb is None:
                why = WHY_NO_FRAME
            else:
                color, why = probe_color(rgb, u.kind, u.x, u.y)
        d = decide(ctx, u.kind, u.team, color, why)
        out.append(replace(u, color=color, relation=d.relation, relation_reason=d.relation_reason,
                           side=d.side, side_reason=d.side_reason))
    return out
