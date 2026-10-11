"""Чистые фоны миникарты для синтетики: медиана по кадрам отрезка ролика.

Зачем. Синтетике (lolvision/minimap_synth.py) нужна настоящая миникарта
канала без иконок чемпионов. Иконки, рамка камеры и точки миньонов
двигаются, а карта, башни и надписи за пару минут почти не меняются,
поэтому медиана по кадрам отрезка их убирает. Отрезки берутся только по
обучающим роликам (отложенные — для проверки, их фон в синтетику не идёт).

Кадры «не игры» отсекаются по качеству рамки камеры из data/coords/<id>.csv
(рамка не нашлась — это не миникарта).

Пример:
    python tools/minimap_backgrounds.py           # → data/minimap/backgrounds.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision.paths import COORDS_DATA, DATA

HELD_OUT = ("58w57eJ5Qks", "olmTXkkUv58", "5nBGNokOUDA", "4AIX8QtRid4")
OUT = DATA / "minimap" / "backgrounds.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", type=int, default=150, help="длина отрезка, кадров (~с)")
    ap.add_argument("--per", type=int, default=31, help="кадров на медиану")
    ap.add_argument("--min-quality", type=float, default=0.45)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    bgs, meta = [], []
    for f in sorted(COORDS_DATA.glob("*.csv")):
        vid = f.stem
        if vid in HELD_OUT:
            continue
        rows = [r for r in csv.DictReader(f.open(encoding="utf-8"))
                if float(r["quality"]) >= args.min_quality]
        for s in range(0, len(rows) - args.window // 2, args.window):
            seg = rows[s:s + args.window]
            if len(seg) < args.per:
                continue
            pick = np.linspace(0, len(seg) - 1, args.per).astype(int)
            stack = np.stack([np.asarray(Image.open(COORDS_DATA / seg[i]["mini"]).convert("RGB"))
                              for i in pick])
            bgs.append(np.median(stack, 0).astype(np.uint8))
            meta.append(f"{vid}:{seg[0]['idx']}")
        print(vid, len(bgs), flush=True)
    out = np.stack(bgs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, bg=out, src=np.array(meta))
    print(f"{len(out)} фонов → {args.out}")


if __name__ == "__main__":
    main()
