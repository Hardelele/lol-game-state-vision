"""Проверка чемпионов миникарты по ручной разметке dataset/minimap_champions.

Разметка (tools/label_minimap.py → правка по листам): на кадр — иконки
{x, y, champion|null, team, unsure, added} и мёртвые слоты HUD с таймером.
Считается:

* детекция иконок: точность и полнота (совпадение — центр ближе MATCH_PX
  пикселей сетки 256);
* положение: у разметки нет независимых координат (точки — черновик модели
  или поставлены глазами), поэтому ошибка меряется против независимой
  классической оценки — подгонки иконки Data Dragon нормированной
  корреляцией вокруг размеченной точки (согласие двух методов, не истина);
  истинная ошибка — на синтетике (train_minimap_icons.py);
* состав по портретам HUD против состава из разметки;
* кто это: доля совпавших детекций, где назначенный чемпион верен;
* 10 записей кадра: видимость (чемпион размечен на кадре ↔ запись visible);
* жив/мёртв и таймер против HUD из разметки;
* время на кадр.

Пример:
    python tools/eval_minimap.py --run runs/minimap/icons
    # → runs/minimap/icons/eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import _bootstrap  # noqa: F401
from lolvision import ddragon
from lolvision.minimap_champions import FrameReader, grid_px_to_units, roster_from_hud
from lolvision.minimap_icons import DEFAULT_RUN, IconModel
from lolvision.paths import DATASET, DATASET_VIDEOS, DEFAULT_LAYOUT

LABELS = DATASET / "minimap_champions"
MATCH_PX = 5.0


def load_frame(vid: str, frame: str) -> np.ndarray:
    with Image.open(DATASET_VIDEOS / vid / "frames" / f"{frame}.jpg") as im:
        return np.asarray(im.convert("RGB"))


def fit_icon(mm: np.ndarray, icon: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Классическая оценка центра: иконка в кружке ~15.5 px, NCC в окне ±6 px,
    субпиксель — парабола по соседям максимума."""
    best = (-2.0, x, y)
    for d in (14.5, 15.5, 16.5):
        n = int(np.ceil(d)) + 1
        a, b = int(64 - 0.84 * 64), int(64 + 0.84 * 64)
        t = cv2.resize(icon[a:b, a:b], (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
        yy, xx = np.mgrid[:n, :n]
        m = (np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) <= d / 2 - 0.5).astype(np.float32)
        x0, y0 = int(round(x)) - 6 - n // 2, int(round(y)) - 6 - n // 2
        if x0 < 0 or y0 < 0 or x0 + n + 12 > mm.shape[1] or y0 + n + 12 > mm.shape[0]:
            continue
        win = mm[y0:y0 + n + 12, x0:x0 + n + 12].astype(np.float32)
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED, mask=np.repeat(m[..., None], 3, 2))
        r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
        j, i = np.unravel_index(int(r.argmax()), r.shape)
        if r[j, i] <= best[0]:
            continue
        dx = dy = 0.0
        if 0 < i < r.shape[1] - 1:
            den = r[j, i - 1] - 2 * r[j, i] + r[j, i + 1]
            dx = 0.5 * (r[j, i - 1] - r[j, i + 1]) / den if den < 0 else 0.0
        if 0 < j < r.shape[0] - 1:
            den = r[j - 1, i] - 2 * r[j, i] + r[j + 1, i]
            dy = 0.5 * (r[j - 1, i] - r[j + 1, i]) / den if den < 0 else 0.0
        best = (float(r[j, i]), x0 + i + dx + n / 2, y0 + j + dy + n / 2)
    return best[1], best[2]


def match(dets: list[dict], labs: list[dict]) -> list[tuple[int, int]]:
    if not dets or not labs:
        return []
    D = np.array([[np.hypot(d["x"] - lb["x"], d["y"] - lb["y"]) for lb in labs] for d in dets])
    r, c = linear_sum_assignment(D)
    return [(i, j) for i, j in zip(r, c) if D[i, j] <= MATCH_PX]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--device", default=None)
    ap.add_argument("--labels", default=str(LABELS))
    args = ap.parse_args()

    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (Path(args.labels) / "labels.jsonl").open(encoding="utf-8")]
    comps = json.loads((Path(args.labels) / "compositions.json").read_text(encoding="utf-8"))
    model = IconModel(Path(args.run), args.device)
    icons = ddragon.load_icons()
    upx = grid_px_to_units(layout)

    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video"]].append(r)

    rep: dict = {"frames": len(rows), "run": args.run, "device": model.device}
    tp = fp = fn = 0
    pos_err = []
    who_ok = who_n = who_ok_sure = who_n_sure = team_ok = 0
    vis = defaultdict(int)
    alive = defaultdict(int)
    timer_ok = timer_n = 0
    misses = defaultdict(list)          # разбор ошибок: первые случаи каждого вида
    roster_rep = {}
    t_frame = []
    for vid, rs in by_vid.items():
        frames = [load_frame(vid, r["frame"]) for r in rs]
        roster, margin = roster_from_hud([f[..., ::-1] for f in frames[::5][:8]], icons,
                                         model.champions)
        truth = comps[vid]["ORDER"] + comps[vid]["CHAOS"]
        roster_rep[vid] = {"roster": roster, "margin": [round(float(m), 3) for m in margin],
                           "set_correct": sorted(roster[:5]) == sorted(comps[vid]["ORDER"])
                           and sorted(roster[5:]) == sorted(comps[vid]["CHAOS"]),
                           "set_overlap": len(set(roster) & set(truth))}
        reader = FrameReader(model, roster, layout, icons)
        slot_of = {n: f"{'LR'[i // 5]}{i % 5 + 1}" for i, n in enumerate(roster)}
        for r, f in zip(rs, frames):
            t0 = time.perf_counter()
            mm = reader.minimap(f)
            dets = model(mm[None])[0]
            recs = reader.records(f, dets)
            t_frame.append(time.perf_counter() - t0)
            labs = r["icons"]
            pairs = match(dets, labs)
            tp += len(pairs)
            fp += len(dets) - len(pairs)
            fn += len(labs) - len(pairs)
            # Кто это: чемпион, назначенный детекции в записях кадра.
            det_name = {}
            for rec in recs:
                if rec["visible"]:
                    det_name[tuple(rec["mm"])] = rec["champion"]
            for i, j in pairs:
                lb, d = labs[j], dets[i]
                name = det_name.get((round(d["x"], 2), round(d["y"], 2)))
                if lb["champion"]:
                    who_n += 1
                    who_ok += int(name == lb["champion"])
                    if name != lb["champion"]:
                        misses["identity"].append(f"{r['frame']} {lb['champion']} -> {name}")
                    if not lb.get("unsure"):
                        who_n_sure += 1
                        who_ok_sure += int(name == lb["champion"])
                    team_ok += int(("ORDER", "CHAOS")[d["team"]] == lb["team"])
                    if not lb.get("added"):
                        fx, fy = fit_icon(mm, icons[lb["champion"]], lb["x"], lb["y"])
                        pos_err.append(float(np.hypot(fx - d["x"], fy - d["y"])))
            # Видимость в 10 записях: размеченный чемпион ↔ visible и рядом с меткой.
            lab_pos = {lb["champion"]: lb for lb in labs if lb["champion"] and not lb.get("unsure")}
            for rec in recs:
                lb = lab_pos.get(rec["champion"])
                if lb is not None:
                    ok = rec["visible"] and np.hypot(rec["mm"][0] - lb["x"],
                                                     rec["mm"][1] - lb["y"]) <= MATCH_PX
                    vis["labelled"] += 1
                    vis["found_right_place"] += int(ok)
                elif rec["visible"]:
                    vis["visible_not_labelled"] += 1
            # Жив / мёртв против HUD разметки.
            hud = r["hud"]
            for rec in recs:
                slot = slot_of[rec["champion"]]
                if slot.startswith("L") and not hud["left_visible"]:
                    alive["skipped_hidden_hud"] += 1
                    continue
                if slot in hud.get("unknown", []):
                    alive["skipped_unknown_slot"] += 1
                    continue
                dead_true = slot in hud["dead"]
                if rec["alive"] is None:
                    alive["no_answer"] += 1
                    continue
                dead_pred = not rec["alive"]
                kind = ("tp" if dead_true else "fp") if dead_pred else ("fn" if dead_true else "tn")
                alive[kind] += 1
                if kind in ("fp", "fn"):
                    misses[f"alive_{kind}"].append(f"{r['frame']} {slot}")
                if dead_true and dead_pred and hud["dead"][slot] is not None:
                    timer_n += 1
                    timer_ok += int(rec["respawn"] == hud["dead"][slot])
                    if rec["respawn"] != hud["dead"][slot]:
                        misses["timer"].append(f"{r['frame']} {slot} {rec['respawn']}"
                                               f" != {hud['dead'][slot]}")
    pe = np.array(pos_err)
    rep.update({
        "rosters": roster_rep,
        "detection": {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(1, tp + fp), 4),
                      "recall": round(tp / max(1, tp + fn), 4), "match_px": MATCH_PX},
        "position_vs_template_fit_px": {
            "n": int(len(pe)), "median": round(float(np.median(pe)), 3),
            "p90": round(float(np.percentile(pe, 90)), 3),
            "median_units": round(float(np.median(pe)) * upx, 1),
            "p90_units": round(float(np.percentile(pe, 90)) * upx, 1)} if len(pe) else None,
        "units_per_grid_px": round(upx, 2),
        "identity": {"n": who_n, "acc": round(who_ok / max(1, who_n), 4),
                     "n_sure": who_n_sure, "acc_sure": round(who_ok_sure / max(1, who_n_sure), 4),
                     "team_acc_detector": round(team_ok / max(1, who_n), 4)},
        "records_visibility": dict(vis) | {
            "recall": round(vis["found_right_place"] / max(1, vis["labelled"]), 4)},
        "alive": dict(alive) | {
            "dead_precision": round(alive["tp"] / max(1, alive["tp"] + alive["fp"]), 4),
            "dead_recall": round(alive["tp"] / max(1, alive["tp"] + alive["fn"]), 4),
            "accuracy": round((alive["tp"] + alive["tn"]) / max(1, alive["tp"] + alive["tn"]
                                                              + alive["fp"] + alive["fn"]), 4)},
        "timer": {"n": timer_n, "exact": round(timer_ok / max(1, timer_n), 4)},
        "misses": {k: v[:40] for k, v in misses.items()},
        "ms_per_frame": {"median": round(1000 * float(np.median(t_frame)), 2),
                         "mean": round(1000 * float(np.mean(t_frame)), 2)},
    })
    out = Path(args.run) / "eval.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
