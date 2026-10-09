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

import argparse
import csv
import json
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
        x, y, q = find_box(m, bw, bh)
        out.append({
            "frame": Path(r["frame"]).stem,
            "t_sec": int(r["t_sec"]),
            # Центр вьюпорта в долях миникарты: (0,0) верхний левый угол карты.
            "cx": round((x + bw / 2) / MM, 4),
            "cy": round((y + bh / 2) / MM, 4),
            "quality": round(q, 3),
        })
    return out, (bw / MM, bh / MM)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dirs", type=Path, nargs="+")
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="CSV с координатами")
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    everything = []
    for vd in args.video_dirs:
        rows, box = read_video(vd, layout)
        for r in rows:
            r["video_id"] = vd.name
        everything += rows
        q = np.array([r["quality"] for r in rows])
        print(f"{vd.name}: кадров {len(rows)}, размер рамки "
              f"{box[0]:.3f}x{box[1]:.3f} доли карты, качество "
              f"медиана {np.median(q):.2f}, минимум {q.min():.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "frame", "t_sec",
                                              "cx", "cy", "quality"])
            w.writeheader()
            w.writerows(everything)
        print(f"{len(everything)} строк → {args.out}")


if __name__ == "__main__":
    main()
