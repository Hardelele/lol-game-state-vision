"""Ответ патчевой модели по одному кадру — проверка на сохранённых сценах.

Командная строка над lolvision/patch_infer.py: медиана ошибки по первым
кадрам ролика из data/coords/ и время последнего кадра по стадиям.

Пример:
    python tools/patch_infer.py runs/patches/cnn-split16/model.pt 58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.checkpoints import load_checkpoint
from lolvision.geometry import MAP_UNITS
from lolvision.patch_infer import PatchInfer
from lolvision.paths import COORDS_DATA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PatchInfer(*load_checkpoint(args.model, dev, "patch"), dev)
    rows = [r for r in csv.DictReader((COORDS_DATA / f"{args.video}.csv").open(encoding="utf-8"))
            if float(r["quality"]) >= 0.4][:args.n]
    err = []
    for r in rows:
        with Image.open(COORDS_DATA / r["scene"]) as s:
            a = np.asarray(s.convert("RGB"))
        out, _, ms = m.predict(m.prepare(a))
        err.append(np.hypot(out["px"] - float(r["cx"]), out["py"] - float(r["cy"])) * MAP_UNITS)
    print(f"{m.describe(m.store)}: {len(err)} кадров, медиана {np.median(err):.0f} ед.,"
          f" последний кадр {ms}")


if __name__ == "__main__":
    main()
