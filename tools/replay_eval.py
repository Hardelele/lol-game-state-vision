"""Метка с миникарты и модель против эталона Replay API.

Зачем. tools/replay_capture.py снимает кадры реплея, для которых известно
точное положение камеры. Здесь по ним меряется:
* как рамка камеры на миникарте связана с настоящими координатами
  (масштаб, начало, остаток — должен быть на уровне долей пикселя);
* ошибка метки, которую даёт нынешний поиск рамки (minimap_camera.find_box
  по миникарте 256, как в build_coords), и поиска с учётом обрезки краем
  (label_noise.find_box_clipped);
* ошибка патчевой модели по сцене (как во входе при обучении).

Эталон в пространстве меток: где был бы центр рамки, если читать её без
округления и обрезки. Он получается из линейной связи «положение камеры →
центр рамки в пикселях экрана», подогнанной по кадрам с целой рамкой.
Ошибки — в настоящих игровых единицах.

Пример:
    python tools/replay_eval.py data/replay/rand300 data/replay/grid7
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision.dataset_io import STORE_W, stored_keep
from lolvision.geometry import load_projection
from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.minimap_camera import MM, estimate_box, find_box, white_mask
from lolvision.paths import DATASET, RUNS

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
MODEL = RUNS / "patches" / "cnn-split16-notgame" / "model.pt"
OUT = RUNS / "replay_eval"
# Область поиска линий рамки на экране реплея 1920×1080 (миникарта с запасом).
REGION = (1650, 804, 1914, 1072)


def line_centers(prof: np.ndarray, need: int) -> list[float]:
    """Центры групп подряд идущих строк/столбцов, где белых пикселей ≥ need."""
    idx = np.nonzero(prof >= need)[0]
    if not len(idx):
        return []
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [float(g.mean()) + 0.5 for g in groups]  # +0.5: центр пикселя


def native_box(img: Image.Image) -> tuple[list[float], list[float]]:
    """Линии рамки в непрерывных координатах экрана (центры пикселей)."""
    a = np.asarray(img.crop(REGION)).astype(int)
    w = (a.min(2) > 185) & (a.max(2) - a.min(2) < 40)
    cols = [c + REGION[0] for c in line_centers(w.sum(0), 18)]
    rows = [r + REGION[1] for r in line_centers(w.sum(1), 30)]
    return cols, rows


def load(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        for r in csv.DictReader((d / "frames.csv").open(encoding="utf-8")):
            r["path"] = d / r["file"]
            r["set"] = d.name
            out.append(r)
    return out


def fit_axis(g: np.ndarray, c: np.ndarray) -> tuple[float, float, np.ndarray]:
    """c = a + b·g по МНК; остатки в пикселях."""
    A = np.stack([np.ones_like(g), g], 1)
    (a, b), *_ = np.linalg.lstsq(A, c, rcond=None)
    return float(a), float(b), c - (a + b * g)


def pct(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 1),
            "p90": round(float(np.percentile(v, 90)), 1),
            "p99": round(float(np.percentile(v, 99)), 1),
            "max": round(float(v.max()), 1),
            "gt200": round(float((v > 200).mean() * 100), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()

    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    rows = load(args.dirs)
    imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
    gx = np.array([float(r["ground_x"]) for r in rows])
    gz = np.array([float(r["ground_z"]) for r in rows])

    # 1. Рамка на экране ↔ камера: по кадрам, где видны все четыре линии.
    cx = np.full(len(rows), np.nan)
    cy = np.full(len(rows), np.nan)
    for i, im in enumerate(imgs):
        cols, rws = native_box(im)
        if len(cols) == 2 and len(rws) == 2:
            cx[i], cy[i] = np.mean(cols), np.mean(rws)
    full = np.isfinite(cx)
    ax, bx, rx = fit_axis(gx[full], cx[full])
    ay, by, ry = fit_axis(gz[full], cy[full])
    # Эталонный центр рамки на экране для любого кадра, в том числе у края.
    tx, ty = ax + bx * gx, ay + by * gz
    upx = 1 / abs(bx)                      # игровых единиц на пиксель экрана
    link = {"frames_full_box": int(full.sum()), "units_per_px_x": round(1 / bx, 3),
            "units_per_px_y": round(1 / by, 3),
            "screen_x_at_0": round(ax, 2), "screen_y_at_0": round(ay, 2),
            "screen_x_at_14800": round(ax + bx * 14800, 2),
            "screen_y_at_14800": round(ay + by * 14800, 2),
            "resid_px_rms": [round(float(np.sqrt(np.mean(rx ** 2))), 3),
                             round(float(np.sqrt(np.mean(ry ** 2))), 3)],
            "resid_px_max": [round(float(np.abs(rx).max()), 3),
                             round(float(np.abs(ry).max()), 3)]}

    # 2. Пространство меток: миникарта раскладки → 256, центр = (x + bw/2)/256.
    w, h = imgs[0].size
    m0, n0, m1, n1 = layout["labelling"]["minimap"]
    mb = (int(m0 * w), int(n0 * h), int(m1 * w), int(n1 * h))
    sx, sy = (mb[2] - mb[0]) / MM, (mb[3] - mb[1]) / MM   # пикселей экрана на пиксель 256

    def to_label(px, py):
        # Непрерывная координата экрана → доля миникарты в условности меток:
        # ребро в столбце k сетки 256 имеет центр k + 0.5, центр рамки x + bw/2.
        return ((px - mb[0]) / sx - 0.5) / MM, ((py - mb[1]) / sy - 0.5) / MM

    TL = np.stack(to_label(tx, ty), 1)
    lab_units = sx * MM * upx              # игровых единиц на единицу доли метки
    # Где рамка выходит за миникарту (край поля): половина рамки от границы.
    bw_px = np.nanmedian([c[1] - c[0] for c in (native_box(im)[0] for im in imgs)
                          if len(c) == 2])
    bh_px = np.nanmedian([r[1] - r[0] for r in (native_box(im)[1] for im in imgs)
                          if len(r) == 2])
    lo_x, hi_x = ax, ax + bx * 14800
    hi_y, lo_y = ay, ay + by * 14800       # ось y экрана смотрит вниз
    edge = ((tx - bw_px / 2 < lo_x) | (tx + bw_px / 2 > hi_x)
            | (ty - bh_px / 2 < lo_y) | (ty + bh_px / 2 > hi_y))

    minis = [np.asarray(im.crop(mb).resize((MM, MM), Image.LANCZOS), dtype=np.float32)
             for im in imgs]
    masks = [white_mask(m) for m in minis]
    bw, bh = estimate_box(masks)
    L = np.array([[(x + bw / 2) / MM, (y + bh / 2) / MM]
                  for x, y, _ in (find_box(m, bw, bh) for m in masks)])
    from label_noise import find_box_clipped
    K = np.array([[(x + bw / 2) / MM, (y + bh / 2) / MM]
                  for x, y, *_ in (find_box_clipped(m, bw, bh) for m in masks)])

    res = {"link": link, "box_native_px": [round(float(bw_px), 2), round(float(bh_px), 2)],
           "box_256": [bw, bh], "label_unit_in_game_units": round(lab_units, 1),
           "frames": len(rows), "edge_frames": int(edge.sum())}

    def errs(P):
        return np.hypot(*(P - TL).T) * lab_units

    E = {"label": errs(L), "clip": errs(K)}

    # 3. Модель по сцене, как при обучении: маска HUD, плотная обрезка, 720 px.
    if not args.no_model:
        import torch  # noqa: F401
        from lolvision.checkpoints import load_checkpoint
        from lolvision.patch_model import infer_images
        cx0, cy0, cx1, cy1 = crop_box((w, h), layout, "dense")
        store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
        boxes = mask_boxes((w, h), layout)
        scenes = []
        for im in imgs:
            s = im.copy()
            for b in boxes:
                s.paste((0, 0, 0), b)
            scenes.append(np.asarray(s.crop((cx0, cy0, cx1, cy1))
                                     .resize((STORE_W, store_h), Image.LANCZOS)))
        keep = stored_keep(layout) > 127
        h0, _ = load_projection()
        model, _ = load_checkpoint(args.model, args.device, "patch")
        cams, agree, _ = infer_images(model, scenes, [np.eye(3)] * len(scenes),
                                      [keep] * len(scenes), h0, args.device, batch=8)
        M = np.array([[float(c[0]), float(c[1])] for c in cams])
        E["model"] = errs(M)
        E["model_vs_label"] = np.hypot(*(M - L).T) * lab_units
        res["model"] = str(args.model)

    for name, e in E.items():
        res[name] = {"all": pct(e), "interior": pct(e[~edge]), "edge": pct(e[edge])}
    # Смещение метки по осям внутри карты: постоянный сдвиг или шум.
    for name, P in (("label", L), ("clip", K)):
        d = (P - TL)[~edge] * lab_units
        res[name]["interior_bias_xy"] = np.round(np.median(d, 0), 1).tolist()
        res[name]["interior_sd_xy"] = np.round(d.std(0), 1).tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "+".join(d.name for d in args.dirs)
    with (OUT / f"{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["set", "file", "ground_x", "ground_z", "edge", "truth_lx", "truth_ly",
                     "label_lx", "label_ly", "clip_lx", "clip_ly"] + list(E))
        for i, r in enumerate(rows):
            wr.writerow([r["set"], r["file"], r["ground_x"], r["ground_z"], int(edge[i]),
                         *np.round(TL[i], 5), *np.round(L[i], 5), *np.round(K[i], 5),
                         *[round(float(E[k][i]), 1) for k in E]])
    (OUT / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
