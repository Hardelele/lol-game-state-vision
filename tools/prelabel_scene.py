"""Предварительная разметка типа сцены для нового ролика + листы для проверки.

Зачем. Размечать часовой стрим глазами кадр за кадром долго. Здесь
кадры ролика (data/not_game/<id>, собраны build_not_game.py) получают
ответ DINO-головы (scene_kind_dino.py), соседние кадры с одним ответом
склеиваются в интервалы, и на лист выводятся границы интервалов и по
несколько кадров изнутри. Проверенные интервалы правятся руками и
сохраняются в dataset/not_game/<id>/labels.csv (+ overlays.csv для Tab и
магазина).

Пример:
    python tools/prelabel_scene.py tXxnABxitiM
    # → runs/scene_kind/prelabel/<id>.csv (интервалы) и <id>_NN.jpg (листы)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from lolvision.paths import NOT_GAME_DATA, RUNS
from scene_kind_dino import Head, cached, load_dino
from train_scene_kind import CLASSES

OUT = RUNS / "scene_kind" / "prelabel"


def segments(t: list[float], c: list[int], min_len: int = 2) -> list[list]:
    """Интервалы одинакового ответа; одиночные выбросы короче min_len кадров
    приклеиваются к соседу."""
    c = list(c)
    for i in range(1, len(c) - 1):
        if c[i - 1] == c[i + 1] != c[i]:
            c[i] = c[i - 1]
    segs = []
    for ti, ci in zip(t, c):
        if segs and segs[-1][2] == ci:
            segs[-1][1] = ti
        else:
            segs.append([ti, ti, ci])
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--head", type=Path, default=RUNS / "scene_kind" / "dino-s" / "head.pt")
    ap.add_argument("--per-seg", type=int, default=3, help="кадров изнутри интервала на листе")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [{"video_id": args.video, "t_sec": float(r["t_sec"]), "full": NOT_GAME_DATA / r["full"]}
            for r in csv.DictReader((NOT_GAME_DATA / f"{args.video}.csv").open(encoding="utf-8"))]
    rows.sort(key=lambda r: r["t_sec"])
    ck = torch.load(args.head, map_location=dev)
    head = Head(ck["cin"], len(ck["classes"])).to(dev)
    head.load_state_dict(ck["state"])
    head.eval()
    x = cached(load_dino(dev=dev), rows, dev)
    with torch.no_grad():
        p = torch.softmax(head(torch.from_numpy(x).to(dev)), 1).cpu().numpy()
    t = [r["t_sec"] for r in rows]
    segs = segments(t, p.argmax(1).tolist())
    step = t[1] - t[0] if len(t) > 1 else 5.0

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.video}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_sec", "end_sec", "class", "frames"])
        for a, b, c in segs:
            w.writerow([a, b + step, CLASSES[c], int(round((b - a) / step)) + 1])

    # Лист: для каждого интервала — первый, середина…, последний кадр.
    by_t = {r["t_sec"]: r["full"] for r in rows}
    cells = []
    for k, (a, b, c) in enumerate(segs):
        inside = [ti for ti in t if a <= ti <= b]
        pick = sorted({inside[0], inside[-1],
                       *[inside[int(j)] for j in np.linspace(0, len(inside) - 1,
                                                             args.per_seg + 2)[1:-1]]})
        for ti in pick:
            cells.append((k, ti, CLASSES[c]))
    per = 30
    for n in range(0, len(cells), per):
        sheet = Image.new("RGB", (5 * 288, 6 * 172))
        for j, (k, ti, name) in enumerate(cells[n:n + per]):
            im = Image.open(by_t[ti]).convert("RGB").resize((288, 162))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, 200, 14), fill=(0, 0, 0))
            d.text((3, 1), f"#{k} {ti:.0f}s {name}", fill=(255, 255, 0))
            sheet.paste(im, ((j % 5) * 288, (j // 5) * 172))
        sheet.save(OUT / f"{args.video}_{n // per:02d}.jpg", quality=80)
    print(f"{args.video}: кадров {len(rows)}, интервалов {len(segs)}, листов "
          f"{(len(cells) + per - 1) // per} → {OUT}")
    for a, b, c in segs:
        print(f"  {a:7.0f}-{b + step:7.0f} {CLASSES[c]}")


if __name__ == "__main__":
    main()
