"""Замер зависимости детектора юнитов от полосок ХП (#772).

Зачем. Псевдо-метки детектора — от полосок, тело всегда под полоской.
Вопрос: видит ли сеть тело или только полоску? Проверка на отложенных
роликах в четырёх условиях, кадры одни и те же для всех сетей:

* `orig` — кадр как есть;
* `erased` — полоски всех размеченных юнитов стёрты (lolvision/bar_erase),
  тела на месте: полнота против тех же меток;
* `occluded` — на половину миньонов наложена чужая плашка чемпиона так,
  что закрывает полоску (крип за плашкой — случай со скриншота): полнота
  этих миньонов; заодно — сколько раз сеть увидела «чемпиона» под
  наложенной плашкой, где тела чемпиона нет;
* `phantom` — в пустые места сцены вставлены полоски миньонов и плашки
  чемпионов без тел: доля таких полосок, под которыми сеть нашла юнита.

Пример:
    python tools/eval_units_bars.py --labels 58w57eJ5Qks olmTXkkUv58 5nBGNokOUDA 4AIX8QtRid4 \\
        --nets runs/units/net/model.pt runs/units/net-nobars/model.pt --review 12 --name heldout
    # → runs/units/eval/bars/<имя>/{metrics.json, review/<сеть>/*.jpg}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from build_unit_labels import LABELS
from eval_units import dim, load_frame, review_image
from lolvision.bar_erase import PAD, bar_rect, erase, free_spot, paste
from lolvision.paths import RUNS
from lolvision.unit_bars import BODY
from lolvision.unit_net import CLASSES, UnitNetRunner
from lolvision.unit_targets import RADIUS, box_center, ignore_map, match, unit_keep

OUT = RUNS / "units" / "eval" / "bars"
CONDS = ("orig", "erased", "occluded", "phantom")
PHANTOMS = {"minion": 3, "champion": 1}


def body_center(kind: str, rect: tuple[int, int, int, int], u: dict) -> tuple[float, float]:
    """Где было бы тело под полоской, вставленной прямоугольником `rect`
    (rect — bar_rect юнита u, перенесённый)."""
    x0, y0, x1, y1 = u["bar"]
    rx0, ry0, _, _ = bar_rect(u)
    dx, dy = rect[0] - rx0, rect[1] - ry0
    _, oy, _, bh = BODY[kind]
    return (x0 + x1) / 2 + dx, y0 + dy + oy + bh / 2


def build_bank(rows: list[dict], rng: np.random.Generator, n: int = 150) -> dict[str, list]:
    """Вырезы полосок (всё, что рисует интерфейс) с отложенных кадров:
    {kind: [(вырез RGB, юнит)]}."""
    bank = {"minion": [], "champion": []}
    for i in rng.permutation(len(rows)):
        r = rows[i]
        if all(len(v) >= n for v in bank.values()):
            break
        rgb = None
        for u in r["units"]:
            k = u["kind"]
            if k not in bank or len(bank[k]) >= n:
                continue
            x0, y0, x1, y1 = bar_rect(u)
            if x0 < 0 or y0 < 0 or x1 > 1920 or y1 > 1080:
                continue
            rgb = load_frame(r["frame"]) if rgb is None else rgb
            bank[k].append((rgb[y0:y1, x0:x1].copy(), u))
    return bank


def make_conditions(rgb: np.ndarray, gts_all: list[dict], valid: np.ndarray,
                    bank: dict[str, list], rng: np.random.Generator) -> dict:
    """Кадры условий и что в каждом проверять."""
    out = {"orig": {"rgb": rgb}}
    rects = [bar_rect(u, shape=rgb.shape) for u in gts_all]
    out["erased"] = {"rgb": erase(rgb.copy(), rects, rng=rng)}

    occ = rgb.copy()
    occluded, plates = [], []
    for j, u in enumerate(gts_all):
        if u["kind"] != "minion" or not u["in"] or rng.random() >= 0.5:
            continue
        crop, cu = bank["champion"][int(rng.integers(len(bank["champion"])))]
        ch, cw = crop.shape[:2]
        x0, y0, x1, _ = u["bar"]
        # полоска миньона — под строками самой плашки (в вырезе они ниже имени)
        px = int((x0 + x1) / 2 - cw / 2 + rng.integers(-25, 26))
        py = int(y0 - PAD["champion"][1] - rng.integers(3, 12))
        if paste(occ, crop, px, py) is not None:
            occluded.append(j)
            plates.append(body_center("champion", (px, py, px + cw, py + ch), cu))
    out["occluded"] = {"rgb": occ, "occluded": occluded, "plates": plates}

    ph = rgb.copy()
    avoid = [(u["x"], u["y"]) for u in gts_all] + \
            [((u["bar"][0] + u["bar"][2]) / 2, u["bar"][1]) for u in gts_all]
    phantoms = []
    for k, cnt in PHANTOMS.items():
        for _ in range(cnt):
            crop, cu = bank[k][int(rng.integers(len(bank[k])))]
            ch, cw = crop.shape[:2]
            spot = free_spot(rng, (cw, ch + 120), valid, avoid, 130)
            if spot is None:
                continue
            px, py = spot
            paste(ph, crop, px, py)
            bx, by = body_center(k, (px, py, px + cw, py + ch), cu)
            if not valid[min(1079, int(by)), min(1919, int(bx))]:
                continue
            phantoms.append({"kind": k, "x": bx, "y": by})
            avoid += [(bx, by), (px + cw / 2, py)]
    out["phantom"] = {"rgb": ph, "phantoms": phantoms}
    return out


def score(preds: list[dict], gts: list[dict], st: dict) -> None:
    pairs, fp, fn = match(preds, gts)
    for _, b in pairs:
        st[gts[b]["kind"]][0] += 1
    for a in fp:
        st[preds[a]["kind"]][1] += 1
    for b in fn:
        st[gts[b]["kind"]][2] += 1
    return pairs


def near(preds: list[dict], x: float, y: float, kind: str | None) -> bool:
    r = RADIUS[kind] if kind else RADIUS["champion"]
    return any(np.hypot(p["x"] - x, p["y"] - y) <= r and (kind is None or p["kind"] == kind)
               for p in preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--nets", nargs="+", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--review", type=int, default=0, help="листов на сеть и условие")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--name", default="heldout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep = unit_keep()
    ign = ignore_map(keep)
    valid = 1 - ign
    runners = {str(n): UnitNetRunner(n, keep=keep, valid=valid, thr=args.thr) for n in args.nets}
    rows = [json.loads(s) for n in args.labels for s in (LABELS / f"{n}.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if "error" not in r]
    rng = np.random.default_rng(args.seed)
    if args.sample:
        rows = [rows[i] for i in sorted(rng.permutation(len(rows))[:args.sample])]
    bank = build_bank(rows, np.random.default_rng(args.seed + 1))
    review = set(np.random.default_rng(args.seed + 2).permutation(len(rows))[:args.review].tolist())
    out = OUT / args.name
    acc = {n: {c: {k: [0, 0, 0] for k in CLASSES} for c in CONDS} for n in runners}
    extra = {n: {"occ_n": 0, "occ_hit": 0, "occ_hit_orig": 0, "plate_champ": 0, "plates": 0,
                 "ph": {k: [0, 0, 0] for k in PHANTOMS}} for n in runners}
    for i, r in enumerate(rows):
        rgb = load_frame(r["frame"])
        gts_all = []
        for u in r["units"]:
            cx, cy = box_center(u)
            inside = ign[min(1079, max(0, int(cy))), min(1919, max(0, int(cx)))] == 0
            gts_all.append({**u, "x": cx, "y": cy, "in": inside})
        gts = [g for g in gts_all if g["in"]]
        conds = make_conditions(rgb, gts_all, valid.astype(bool), bank,
                                np.random.default_rng([args.seed, i]))
        for n, run in runners.items():
            res = {}
            for c, d in conds.items():
                units = run.units(d["rgb"])
                preds = [{"kind": u.kind, "team": u.team, "score": u.score, "x": u.x, "y": u.y,
                          "hp": u.hp} for u in units
                         if ign[min(1079, int(u.y)), min(1919, int(u.x))] == 0]
                res[c] = preds
                pairs = score(preds, gts, acc[n][c])
                if c == "occluded":
                    gi = {id(gts_all[j]) for j in d["occluded"]}
                    hit = {id(gts[b]) for _, b in pairs}
                    e = extra[n]
                    occ = [g for g in gts if id(g) in gi]
                    e["occ_n"] += len(occ)
                    e["occ_hit"] += sum(id(g) in hit for g in occ)
                    e["occ_hit_orig"] += sum(near(res["orig"], g["x"], g["y"], "minion") for g in occ)
                    e["plates"] += len(d["plates"])
                    e["plate_champ"] += sum(near(preds, x, y, "champion") for x, y in d["plates"])
                if c == "phantom":
                    for p in d["phantoms"]:
                        s = extra[n]["ph"][p["kind"]]
                        s[0] += 1
                        s[1] += near(preds, p["x"], p["y"], p["kind"])
                        s[2] += near(preds, p["x"], p["y"], None)
                if i in review:
                    marks = gts + [{"kind": p["kind"], "team": "", "box": [p["x"] - 8, p["y"] - 8,
                                    p["x"] + 8, p["y"] + 8]} for p in d.get("phantoms", [])]
                    review_image(dim(d["rgb"], valid), marks, preds,
                                 out / "review" / Path(n).parent.name / f"{i:04d}_{c}.jpg")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    report = {"frames": len(rows), "labels": args.labels, "thr": args.thr, "nets": {}}
    for n in runners:
        rep = {}
        for c in CONDS:
            rep[c] = {}
            for k, (tp, fp, fn) in acc[n][c].items():
                rep[c][k] = {"tp": tp, "fp": fp, "fn": fn,
                             "precision": round(tp / max(1, tp + fp), 4),
                             "recall": round(tp / max(1, tp + fn), 4)}
        e = extra[n]
        rep["occluded_minions"] = {"n": e["occ_n"],
                                   "recall": round(e["occ_hit"] / max(1, e["occ_n"]), 4),
                                   "recall_same_minions_orig": round(e["occ_hit_orig"] / max(1, e["occ_n"]), 4),
                                   "plates": e["plates"],
                                   "champion_under_pasted_plate": round(e["plate_champ"] / max(1, e["plates"]), 4)}
        rep["phantom_bars"] = {k: {"n": v[0], "same_class_found": round(v[1] / max(1, v[0]), 4),
                                   "any_unit_found": round(v[2] / max(1, v[0]), 4)}
                               for k, v in e["ph"].items()}
        report["nets"][n] = rep
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
