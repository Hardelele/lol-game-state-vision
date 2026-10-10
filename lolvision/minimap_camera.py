"""Чтение положения камеры с миникарты: белая рамка → координаты на карте.

Зачем. Метка «что показывает камера» уже есть в каждом кадре — это рамка
вьюпорта на миникарте. Если читать её программно, разметка перестаёт стоить
человеко-часов: из одного ролика получаются тысячи кадров с координатами.

Утечки нет: вся нижняя полоса с миникартой закрыта маской и во вход модели
не попадает. Рамка — источник цели, а не признак.

Как ищется. Рамка — тонкий светлый несатурированный прямоугольник
постоянного размера (вьюпорт камеры не меняется). Миникарта приводится к
256x256, по ней строится маска «светлое и несатурированное», и перебором
положений ищется прямоугольник, по периметру которого таких пикселей больше
всего. Размер оценивается по самому ролику.

Пример:
    python tools/minimap_camera.py dataset/videos/58w57eJ5Qks \
        --layout dataset/layouts/spectator-volibear-challenger.json \
        --out runs/minimap/camera_coords.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

MM = 256                  # сторона приведённой миникарты
WHITE_V, WHITE_S = 150, 28  # порог яркости и максимальный разброс каналов
# Доля стороны миникарты. Вьюпорт 16:9, ширина около 27% карты — уточняется по ролику.
BOX_W, BOX_H = 0.272, 0.153


def minimap(img: Image.Image, layout: dict) -> np.ndarray:
    x0, y0, x1, y1 = layout["labelling"]["minimap"]
    w, h = img.size
    crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return np.asarray(crop.convert("RGB").resize((MM, MM), Image.LANCZOS),
                      dtype=np.float32)


def white_mask(mm: np.ndarray) -> np.ndarray:
    """Рамка белая; рельеф и иконки чемпионов — цветные либо тёмные."""
    hi, lo = mm.max(2), mm.min(2)
    return ((hi > WHITE_V) & ((hi - lo) < WHITE_S)).astype(np.float32)


def find_box(mask: np.ndarray, bw: int, bh: int) -> tuple[int, int, float]:
    """Положение прямоугольника bw x bh с максимальной «белизной» периметра.

    Суммы по строкам и столбцам считаются один раз, поэтому перебор всех
    положений дешёвый и точный — приближения вроде поиска пиков не нужны.
    """
    n = mask.shape[0]
    cx = np.cumsum(np.pad(mask, ((0, 0), (1, 0))), axis=1)  # суммы вдоль x
    cy = np.cumsum(np.pad(mask, ((1, 0), (0, 0))), axis=0)  # суммы вдоль y
    # Нижнее/правое ребро лежит на y+bh и x+bw, поэтому крайнее положение — n-1-b.
    xs = n - bw - 1
    ys = n - bh - 1
    # Горизонтальные рёбра: отрезок длины bw в строке y, начиная с x.
    hseg = cx[:, bw:bw + xs + 1] - cx[:, :xs + 1]           # (n, xs+1)
    # Вертикальные рёбра: отрезок длины bh в столбце x, начиная с y.
    vseg = cy[bh:bh + ys + 1, :] - cy[:ys + 1, :]           # (ys+1, n)
    score = (hseg[:ys + 1, :] + hseg[bh:bh + ys + 1, :]
             + vseg[:, :xs + 1] + vseg[:, bw:bw + xs + 1])
    y, x = np.unravel_index(int(score.argmax()), score.shape)
    perimeter = 2 * (bw + bh)
    return int(x), int(y), float(score[y, x]) / perimeter


# Где на миникарте 256 (обрезка раскладки) игра вообще рисует рамку: слева
# и сверху обрезка захватывает тёмный край интерфейса. Замер по Replay API
# (#754): рамка обрезается по внешнему краю миникарты — 8.8 и 12.7 пикселя
# сетки 256 от левого и верхнего края обрезки. Справа и снизу обрезка сама
# срезает край карты, там рамка уходит за кадр обрезки.
DRAW_X0, DRAW_Y0 = 9, 13
MISS_WEIGHT = 0.3     # цена пропущенного пикселя рамки против найденного белого
MIN_VISIBLE = 0.15    # минимальная видимая доля периметра


def find_box_clipped(mask: np.ndarray, bw: int, bh: int) -> tuple[int, int, float, bool]:
    """Положение рамки bw x bh, которая может выходить за край миникарты.

    У края карты игра рисует рамку обрезанной: от неё остаются одна-две
    линии и обрывки. find_box перебирает только положения целиком внутри и
    такую рамку прижимает к краю или переносит на соседние светлые линии —
    ошибка в сотни и тысячи единиц (docs/label-noise.md, docs/replay-api.md).
    Здесь положения перебираются и за краем, а оценка — сумма по видимой
    части периметра: белый пиксель +1, пропущенный −MISS_WEIGHT. Так длинная
    целая линия весит больше короткого обрывка, а рамка, у которой видна
    только нижняя линия, не проигрывает целой рамке, сдвинутой на неё.
    Проверено против эталона Replay API: у края 0 ошибок больше 200 ед.
    против 62% у find_box (на положениях камеры, достижимых в роликах).

    Ответ — (x, y, качество, обрезана ли): x, y — левый верхний угол в сетке
    миникарты, могут быть отрицательными или выходить за неё; качество, как
    у find_box, — доля белых пикселей от полного периметра.
    """
    px, py = bw // 2, bh // 2
    vis_area = np.zeros_like(mask)
    vis_area[DRAW_Y0:, DRAW_X0:] = 1
    pm = np.pad(mask * vis_area, ((py, py + 1), (px, px + 1)))
    inside = np.pad(vis_area, ((py, py + 1), (px, px + 1)))

    def perimeter_sums(m: np.ndarray) -> np.ndarray:
        n_y, n_x = m.shape
        cx = np.cumsum(np.pad(m, ((0, 0), (1, 0))), axis=1)
        cy = np.cumsum(np.pad(m, ((1, 0), (0, 0))), axis=0)
        xs, ys = n_x - bw - 1, n_y - bh - 1
        hseg = cx[:, bw:bw + xs + 1] - cx[:, :xs + 1]
        vseg = cy[bh:bh + ys + 1, :] - cy[:ys + 1, :]
        return (hseg[:ys + 1, :] + hseg[bh:bh + ys + 1, :]
                + vseg[:, :xs + 1] + vseg[:, bw:bw + xs + 1])

    white, vis = perimeter_sums(pm), perimeter_sums(inside)
    per = 2 * (bw + bh)
    score = np.where(vis >= MIN_VISIBLE * per, white - MISS_WEIGHT * (vis - white), -np.inf)
    y, x = np.unravel_index(int(score.argmax()), score.shape)
    return int(x) - px, int(y) - py, float(white[y, x]) / per, bool(vis[y, x] < per)


def estimate_box(masks: list[np.ndarray]) -> tuple[int, int]:
    """Подобрать размер рамки по ролику: берём размер с лучшим средним качеством."""
    best, size = -1.0, (int(BOX_W * MM), int(BOX_H * MM))
    for bw in range(int(0.22 * MM), int(0.32 * MM)):
        bh = int(round(bw * 9 / 16))
        s = float(np.mean([find_box(m, bw, bh)[2] for m in masks]))
        if s > best:
            best, size = s, (bw, bh)
    return size


def read_video(video_dir: Path, layout: dict, sample: int = 12) -> list[dict]:
    rows = list(csv.DictReader((video_dir / "frames.csv").open(encoding="utf-8")))
    masks = []
    for r in rows:
        with Image.open(video_dir / r["frame"]) as img:
            masks.append(white_mask(minimap(img, layout)))
    step = max(1, len(masks) // sample)
    bw, bh = estimate_box(masks[::step])

    out = []
    for r, m in zip(rows, masks):
        x, y, q, _ = find_box_clipped(m, bw, bh)
        out.append({
            "frame": Path(r["frame"]).stem,
            "t_sec": int(r["t_sec"]),
            # Центр вьюпорта в долях миникарты: (0,0) верхний левый угол карты.
            "cx": round((x + bw / 2) / MM, 4),
            "cy": round((y + bh / 2) / MM, 4),
            "quality": round(q, 3),
        })
    return out, (bw / MM, bh / MM)
