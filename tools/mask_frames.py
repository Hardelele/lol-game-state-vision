"""Закраска HUD и миникарты по раскладке — вход модели видит только сцену.

Пример:
    python tools/mask_frames.py dataset/layouts/spectator-volibear-challenger.json \
        dataset/videos/58w57eJ5Qks/frames/58w57eJ5Qks_t00300.jpg --out runs/mask-preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def apply_mask(img: Image.Image, layout: dict, fill=(0, 0, 0)) -> Image.Image:
    out = img.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in layout["mask"].values():
        draw.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), fill=fill)
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
        apply_mask(Image.open(frame), layout).save(args.out / frame.name, quality=90)
    print(f"{len(args.frames)} кадров → {args.out}")


if __name__ == "__main__":
    main()
