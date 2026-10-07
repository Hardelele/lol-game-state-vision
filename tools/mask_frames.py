"""Закраска HUD и миникарты по раскладке с сохранением кадров в PNG.

Пример:
    python tools/mask_frames.py dataset/layouts/spectator-volibear-challenger.json \
        dataset/videos/58w57eJ5Qks/frames/58w57eJ5Qks_t00300.jpg --out runs/checks/masks/58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def mask_boxes(size: tuple[int, int], layout: dict) -> list[tuple[int, int, int, int]]:
    """Пиксельные границы: включаем левый/верхний край, исключаем правый/нижний."""
    w, h = size
    reference = layout.get("reference_size")
    if reference:
        rw, rh = reference
        if abs(w / h - rw / rh) > 0.001:
            raise ValueError(
                f"Раскладка рассчитана на {rw}:{rh}, получен кадр {w}×{h}. "
                "Выберите раскладку для этого формата."
            )

    boxes = []
    for name, coords in layout["mask"].items():
        if len(coords) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in coords
        ):
            raise ValueError(f"Некорректный прямоугольник {name}: {coords}")
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Прямоугольник {name} должен быть в пределах [0, 1]")
        boxes.append((math.floor(x0 * w), math.floor(y0 * h),
                      math.ceil(x1 * w), math.ceil(y1 * h)))
    return boxes


def apply_mask(img: Image.Image, layout: dict, fill=(0, 0, 0)) -> Image.Image:
    out = img.convert("RGB").copy()
    for box in mask_boxes(out.size, layout):
        out.paste(fill, box)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("layout", type=Path)
    ap.add_argument("frames", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        with Image.open(frame) as img:
            # PNG сохраняет чёрные области точно, без артефактов JPEG на границах.
            apply_mask(img, layout).save(args.out / f"{frame.stem}.png")
    print(f"{len(args.frames)} кадров → {args.out}")


if __name__ == "__main__":
    main()
