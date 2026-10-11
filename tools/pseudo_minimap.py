"""Псевдо-метки иконок на реальных миникартах обучающих роликов (этап 2).

Модель, обученная на синтетике, прогоняется по сохранённым миникартам
data/coords/<id>/mini (только обучающие ролики), по ролику голосованием
набирается состав (10 чемпионов; если есть полные кадры в dataset/videos —
по портретам HUD), затем в каждом кадре детекции
назначаются чемпионам состава. Метки:

* уверенная детекция (score ≥ --pos) — положительная, команда — команда
  назначенного чемпиона (состав надёжнее цвета кольца одной детекции);
* сомнительная (--ign ≤ score < --pos) — «не учитывать»;
* чемпион для классификатора — только если назначен и p ≥ --cls-p.

Результат — data/minimap/pseudo_mini.npy (N×256×256×3, читается через mmap
процессами обучения) и pseudo_labels.json, плюс составы по роликам.

Пример:
    python tools/pseudo_minimap.py --run runs/minimap/icons-syn --step 3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision import ddragon
from lolvision.minimap_champions import Composition, assign, roster_from_hud
from lolvision.minimap_icons import IconModel
from lolvision.paths import COORDS_DATA, DATA, DATASET_VIDEOS

HELD_OUT = ("58w57eJ5Qks", "olmTXkkUv58", "5nBGNokOUDA", "4AIX8QtRid4")
OUT = DATA / "minimap"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, default=3, help="каждый N-й кадр ролика")
    ap.add_argument("--pos", type=float, default=0.55)
    ap.add_argument("--ign", type=float, default=0.15)
    ap.add_argument("--cls-p", type=float, default=0.6)
    ap.add_argument("--min-quality", type=float, default=0.45)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = IconModel(Path(args.run), args.device)
    names = model.champions
    icons = ddragon.load_icons()
    minis, labels, comps = [], [], {}
    for f in sorted(COORDS_DATA.glob("*.csv")):
        vid = f.stem
        if vid in HELD_OUT:
            continue
        rows = [r for r in csv.DictReader(f.open(encoding="utf-8"))
                if float(r["quality"]) >= args.min_quality][::args.step]
        imgs = np.stack([np.asarray(Image.open(COORDS_DATA / r["mini"]).convert("RGB"))
                         for r in rows])
        dets = []
        for k in range(0, len(imgs), 64):
            dets += model(imgs[k:k + 64])
        comp = Composition(len(names))
        for ds in dets:
            comp.add(ds)
        teams = comp.teams()
        fdir = DATASET_VIDEOS / vid / "frames"
        if fdir.is_dir():
            # Есть полные кадры — состав по портретам HUD надёжнее голосования
            # по миникарте (zJvTSjEnKNE: голосование спутало Aphelios и Braum).
            fs = sorted(fdir.glob("*.jpg"))[10::10][:8]
            roster, _ = roster_from_hud([cv2.imread(str(f)) for f in fs], icons, names)
            idx = {n: i for i, n in enumerate(names)}
            teams = [[idx[n] for n in roster[:5]], [idx[n] for n in roster[5:]]]
        champ_team = {c: t for t in (0, 1) for c in teams[t]}
        comps[vid] = [[names[c] for c in t] for t in teams]
        npos = 0
        for img, ds in zip(imgs, dets):
            who = assign(ds, teams)
            lab = []
            for d, c in zip(ds, who):
                if d["score"] < args.ign:
                    continue
                ign = d["score"] < args.pos
                team = champ_team.get(c, d["team"])
                ch = c if (c >= 0 and not ign and np.exp(d["logp"][c]) >= args.cls_p) else -1
                lab.append([round(d["x"], 2), round(d["y"], 2), team, int(ign), int(ch)])
                npos += int(not ign)
            minis.append(img)
            labels.append(lab)
        print(vid, len(rows), "кадров,", npos, "иконок;", comps[vid], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "pseudo_mini.npy", np.stack(minis))
    (OUT / "pseudo_labels.json").write_text(json.dumps(
        {"labels": labels, "compositions": comps, "args": vars(args)}, ensure_ascii=False),
        encoding="utf-8")
    print(len(minis), "кадров →", OUT)


if __name__ == "__main__":
    main()
