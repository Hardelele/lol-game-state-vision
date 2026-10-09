"""Чтение положения камеры с миникарты: белая рамка → координаты на карте.

Командная строка над lolvision/minimap_camera.py: координаты камеры для
опорных кадров dataset/videos/<ролик>/.

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

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.minimap_camera import read_video


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
