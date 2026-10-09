"""Закраска HUD и миникарты по раскладке с сохранением кадров в PNG.

Командная строка над lolvision/mask_frames.py.

Пример:
    python tools/mask_frames.py dataset/layouts/spectator-volibear-challenger.json \
        dataset/videos/58w57eJ5Qks/frames/58w57eJ5Qks_t00300.jpg --out runs/checks/masks/58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.mask_frames import apply_mask


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
