"""Цели и сверка для детектора юнитов: тепловые карты, область без меток,
сопоставление ответов с метками.

Общее для обучения (tools/train_units.py) и проверки (tools/eval_units.py),
вынесено сюда, чтобы тесты шли без кадров.

Область без меток. Псевдо-метки ставятся только там, где полоска юнита
целиком в сцене (не под HUD). Тело стоит под полоской, поэтому у юнита,
чья полоска ушла под HUD или за край кадра, тело видно, а метки нет. Если
считать такие места фоном, сеть учится не видеть юнитов у края HUD.
Поэтому закрытое HUD расширяется: вниз — на расстояние от полоски до
центра тела чемпиона (~105 px), в стороны — на полширины плашки, вверх —
чуть-чуть; там потерь нет, и при проверке там ничего не считается.
"""

from __future__ import annotations

import numpy as np

from lolvision.unit_net import CLASSES

# Сигма гауссианы центра в клетках выхода (шаг 4 на входе 960): у чемпиона
# точка тела по плашке известна грубо (±40 px при 1080p), у миньона и варда —
# точнее.
SIGMA = {"champion": 3.0, "minion": 1.6, "ward": 1.3}
# Радиус совпадения при сверке, px при 1080p.
RADIUS = {"champion": 60.0, "minion": 28.0, "ward": 24.0}
# Расширение закрытого HUD (px при 1080p): влево, вправо, вверх, вниз.
GROW = (65, 65, 10, 115)


def ignore_map(keep: np.ndarray, grow: tuple[int, int, int, int] = GROW) -> np.ndarray:
    """1 — без меток (HUD, край кадра и полосы у них), 0 — метки полные."""
    h, w = keep.shape
    bad = np.zeros((h + 2, w + 2), bool)
    bad[1:-1, 1:-1] = keep == 0
    bad[0, :] = bad[-1, :] = True                   # край кадра — как HUD
    bad[:, 0] = bad[:, -1] = True
    left, right, up, down = grow
    out = bad.copy()
    # Расширение прямоугольным окном по сторонам, через накопленные суммы.
    c =np.pad(bad.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = bad.shape
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    # точка помечается, если в окне [y-down, y+up] × [x-right, x+left] есть HUD:
    # HUD выше точки на down закрывает её полоску, сбоку — на ширину плашки
    y0 = np.clip(y - down, 0, H)
    y1 = np.clip(y + up + 1, 0, H)
    x0 = np.clip(x - right, 0, W)
    x1 = np.clip(x + left + 1, 0, W)
    s = c[y1, x1] - c[y0, x1] - c[y1, x0] + c[y0, x0]
    out |= s > 0
    return out[1:-1, 1:-1].astype(np.uint8)


def draw_gaussian(heat: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    """Максимум из гауссианы и того, что уже есть (как в CenterNet)."""
    h, w = heat.shape
    r = int(3 * sigma) + 1
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 2)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    gx = np.arange(x0, x1) - cx
    gy = np.arange(y0, y1) - cy
    g = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * sigma ** 2))
    np.maximum(heat[y0:y1, x0:x1], g, out=heat[y0:y1, x0:x1])


def build_targets(units: list[dict], out_hw: tuple[int, int], stride: int, scale: float
                  ) -> dict[str, np.ndarray]:
    """Цели одного кадра. `units` — центры в пикселях входа сети: kind, team,
    x, y, hp. `scale` — масштаб кадра относительно 1080p/2 (сигмы растут с ним).

    heat (K, H, W); team, hp, hp_m, team_m (H, W); off (2, H, W), off_m (H, W).
    Команда и ХП учатся в клетке центра и её 3×3 соседях, сдвиг — в клетке."""
    h, w = out_hw
    k = len(CLASSES)
    t = {"heat": np.zeros((k, h, w), np.float32),
         "team": np.zeros((h, w), np.float32), "team_m": np.zeros((h, w), np.float32),
         "hp": np.zeros((h, w), np.float32), "hp_m": np.zeros((h, w), np.float32),
         "off": np.zeros((2, h, w), np.float32), "off_m": np.zeros((h, w), np.float32)}
    for u in units:
        cx, cy = u["x"] / stride, u["y"] / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        c = CLASSES.index(u["kind"])
        draw_gaussian(t["heat"][c], cx, cy, max(0.8, SIGMA[u["kind"]] * scale))
        t["heat"][c, iy, ix] = 1.0
        t["off"][:, iy, ix] = (cx - ix, cy - iy)
        t["off_m"][iy, ix] = 1.0
        ys, xs = slice(max(0, iy - 1), iy + 2), slice(max(0, ix - 1), ix + 2)
        if u.get("team") in ("ORDER", "CHAOS"):
            t["team"][ys, xs] = float(u["team"] == "CHAOS")
            t["team_m"][ys, xs] = 1.0
        hp = u.get("hp")
        if hp is not None and np.isfinite(hp) and u["kind"] != "ward":
            t["hp"][ys, xs] = float(np.clip(hp, 0, 1))
            t["hp_m"][ys, xs] = 1.0
    return t


def match(preds: list[dict], gts: list[dict], radius: dict[str, float] = RADIUS
          ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Жадное сопоставление по убыванию оценки внутри класса.

    preds: kind, x, y, score; gts: kind, x, y. Возвращает пары (i_pred, i_gt),
    несопоставленные ответы и несопоставленные метки."""
    pairs, used = [], set()
    order = sorted(range(len(preds)), key=lambda i: -preds[i].get("score", 1.0))
    fp = []
    for i in order:
        p = preds[i]
        best, bd = -1, radius[p["kind"]]
        for j, g in enumerate(gts):
            if j in used or g["kind"] != p["kind"]:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d <= bd:
                best, bd = j, d
        if best >= 0:
            used.add(best)
            pairs.append((i, best))
        else:
            fp.append(i)
    fn = [j for j in range(len(gts)) if j not in used]
    return pairs, fp, fn


def box_center(u: dict) -> tuple[float, float]:
    b = u["box"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
