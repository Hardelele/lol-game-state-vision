"""Проверить детектор юнитов: сверка с псевдо-метками, листы для проверки
глазами, кадры вида игрока, скорость.

Зачем. Этап 3 плана второй модели (docs/unit-detection.md): точность и
полнота детектора на отложенных роликах — против псевдо-меток (много кадров,
но учитель с ошибками) и против проверенной вручную выборки (листы
`--review`: метки учителя — рамки, ответы сети — крестики с номерами).

Пример:
    python tools/eval_units.py --labels 58w57eJ5Qks olmTXkkUv58 --review 40
    python tools/eval_units.py --frames "data/not_game/*/full/*.jpg" --sample 24 --name player
    python tools/eval_units.py --bench
    python tools/eval_units.py --gold --nets runs/units/net/model.pt runs/units/net-nobars2/model.pt \
        --thrs 0.2 0.3 0.4 0.5     # ручной эталон dataset/units_gold
    # → runs/units/eval/<имя>/{metrics.json, preds.jsonl, review/*.jpg}
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from build_unit_labels import COLORS, LABELS, label_keep
from lolvision.paths import ROOT, RUNS
from lolvision.unit_net import CLASSES, DEFAULT_NET, UnitNetRunner
from lolvision.unit_targets import box_center, ignore_map, match

OUT = RUNS / "units" / "eval"
GOLD = ROOT / "dataset" / "units_gold" / "labels.jsonl"


def load_frame(path: str) -> np.ndarray:
    """Кадр как 1080p RGB: меньшие кадры (стримы 960×540) растягиваются."""
    rgb = cv2.imread(str(ROOT / path))[..., ::-1]
    if rgb.shape[:2] != (1080, 1920):
        rgb = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb)


def review_image(rgb: np.ndarray, gts: list[dict], preds: list[dict], out: Path,
                 crop: tuple[int, int, int, int] = (0, 0, 1920, 1080)) -> None:
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for g in gts:
        c = COLORS.get(g["team"], (255, 255, 0))
        d.rectangle(g["box"], outline=c, width=1)
    for i, p in enumerate(preds):
        c = COLORS.get(p["team"], (255, 255, 0))
        x, y = p["x"], p["y"]
        r = 10 if p["kind"] == "champion" else 6
        d.line((x - r, y - r, x + r, y + r), fill=c, width=3)
        d.line((x - r, y + r, x + r, y - r), fill=c, width=3)
        d.text((x + r + 2, y - 6), f"{i}{p['kind'][0]}{int(p['hp'] * 100)}", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.crop(crop).save(out, quality=88)


def dim(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Затемнить то, что не проверяется (HUD и полосы у его краёв)."""
    out = rgb.copy()
    out[valid == 0] = out[valid == 0] // 3
    return out


def bench(runner: UnitNetRunner, n: int = 300) -> dict:
    """мс на кадр: только граф (кадр уже в видеопамяти) и с загрузкой кадра."""
    rgb = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    pinned = torch.from_numpy(rgb).pin_memory()
    dev = torch.empty_like(pinned, device="cuda")
    for _ in range(30):
        runner.raw(dev)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        runner.raw(dev)
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t) / n * 1000
    t = time.perf_counter()
    for _ in range(n):
        dev.copy_(pinned, non_blocking=True)
        runner.raw(dev)[0].cpu()
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t) / n * 1000
    return {"graph_ms": round(graph_ms, 3), "upload_graph_readback_ms": round(full_ms, 3),
            "max_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)}


def gold_rows(path: Path) -> list[dict]:
    """Готовые кадры ручного эталона: юниты — точки центров тел (x, y при
    1080p). Классы вне CLASSES (structure, monster, other) идут отдельно:
    ответ сети на них — не ошибка счёта, а отдельная строка отчёта."""
    rows = []
    for s in path.open(encoding="utf-8"):
        r = json.loads(s)
        if r.get("status") != "done":
            continue
        units = [u for u in r["units"] if u["kind"] in CLASSES]
        dup = [(a, b) for i, a in enumerate(units) for b in units[i + 1:]
               if a["kind"] == b["kind"] and np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 8]
        for a, b in dup:
            print(f"! эталон: дубль {a['kind']} ({a['x']:.0f}, {a['y']:.0f}) / "
                  f"({b['x']:.0f}, {b['y']:.0f}) в {r['frame']}", file=sys.stderr)
        rows.append({"frame": r["frame"], "units": units, "duplicates": len(dup),
                     "other": [u for u in r["units"] if u["kind"] not in CLASSES]})
    return rows


def gold_eval(args, keep: np.ndarray, ign: np.ndarray) -> None:
    """Сети против ручного эталона в двух зонах: `valid` — где live сейчас
    выдаёт ответы, `scene` — вся сцена вне HUD (с полосами у его краёв)."""
    rows = gold_rows(args.gold)
    nets = args.nets or [args.net]
    thrs = sorted(args.thrs or [args.thr])
    zones = {"valid": ign == 0, "scene": keep == 1}
    res = {"gold": str(args.gold), "frames": len(rows),
           "duplicate_points": sum(r["duplicates"] for r in rows), "nets": {}}
    for n in nets:
        run = UnitNetRunner(n, keep=keep, valid=keep, thr=thrs[0])
        acc = {(z, t): {"cls": {k: [0, 0, 0] for k in CLASSES}, "on_other": 0, "team": [0, 0]}
               for z in zones for t in thrs}
        for r in rows:
            units = run.units(load_frame(r["frame"]))
            for z, m in zones.items():
                inside = lambda x, y: m[min(1079, max(0, int(y))), min(1919, max(0, int(x)))]
                gts = [u for u in r["units"] if inside(u["x"], u["y"])]
                oth = [u for u in r["other"] if inside(u["x"], u["y"])]
                for t in thrs:
                    a = acc[z, t]
                    preds = [{"kind": u.kind, "team": u.team, "score": u.score, "x": u.x,
                              "y": u.y} for u in units if u.score >= t and inside(u.x, u.y)]
                    pairs, fp, fn = match(preds, gts)
                    for i, j in pairs:
                        a["cls"][gts[j]["kind"]][0] += 1
                        if gts[j].get("team") in ("ORDER", "CHAOS") and gts[j]["kind"] == "minion":
                            a["team"][0] += preds[i]["team"] == gts[j]["team"]
                            a["team"][1] += 1
                    for i in fp:
                        p = preds[i]
                        if any(np.hypot(p["x"] - o["x"], p["y"] - o["y"]) <= 60 for o in oth):
                            a["on_other"] += 1         # ответ на строении / монстре
                        else:
                            a["cls"][p["kind"]][1] += 1
                    for j in fn:
                        a["cls"][gts[j]["kind"]][2] += 1
        out = {}
        for (z, t), a in acc.items():
            d = {k: {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(1, tp + fp), 4),
                     "recall": round(tp / max(1, tp + fn), 4)} for k, (tp, fp, fn) in a["cls"].items()}
            d["answers_on_structure_monster_other"] = a["on_other"]
            d["minion_team_acc"] = round(a["team"][0] / max(1, a["team"][1]), 4)
            out.setdefault(z, {})[str(t)] = d
        res["nets"][str(n)] = out
    o = OUT / (args.name or "gold")
    o.mkdir(parents=True, exist_ok=True)
    (o / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--labels", nargs="*", default=[], help="наборы data/units/labels/<имя>.jsonl")
    ap.add_argument("--frames", default=None, help="glob кадров без меток (качественно)")
    ap.add_argument("--list", type=Path, default=None, help="файл со списком кадров без меток")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--review", type=int, default=0, help="сколько листов для проверки глазами")
    ap.add_argument("--no-mask", action="store_true", help="без маски HUD (другой HUD)")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gold", nargs="?", const=GOLD, type=Path, default=None,
                    help="ручной эталон (точки центров тел), по умолчанию " + str(GOLD))
    ap.add_argument("--nets", nargs="+", type=Path, default=None,
                    help="с --gold: несколько сетей сразу")
    ap.add_argument("--thrs", type=float, nargs="+", default=None,
                    help="с --gold: несколько порогов сразу")
    args = ap.parse_args()

    keep = label_keep()
    ign = ignore_map(keep)
    if args.gold:
        gold_eval(args, keep, ign)
        return
    valid = None if args.no_mask else 1 - ign
    runner = UnitNetRunner(args.net, keep=None if args.no_mask else keep, valid=valid,
                           thr=args.thr)
    if args.bench:
        res = bench(runner)
        res["fp32_graph"] = bench(UnitNetRunner(args.net, keep=keep, fp16=False))
        res["no_graph_fp16"] = bench(UnitNetRunner(args.net, keep=keep, graph=False))
        res["gpu"] = torch.cuda.get_device_name()
        print(json.dumps(res, ensure_ascii=False, indent=1))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "bench.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        return

    rng = np.random.default_rng(args.seed)
    if args.labels:
        rows = [json.loads(s) for n in args.labels
                for s in (LABELS / f"{n}.jsonl").open(encoding="utf-8")]
    else:
        paths = (args.list.read_text(encoding="utf-8").split() if args.list
                 else sorted(glob.glob(args.frames)))
        rows = [{"frame": Path(p).as_posix(), "units": None} for p in paths]
    if args.sample:
        rows = [rows[i] for i in sorted(rng.permutation(len(rows))[:args.sample])]
    name = args.name or "+".join(args.labels) or "frames"
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    stats = {k: [0, 0, 0] for k in CLASSES}
    team_ok = team_n = 0
    hp_err = {k: [] for k in CLASSES}
    review = set(rng.permutation(len(rows))[:args.review].tolist()) if args.review else set()
    if not args.labels:
        review = set(range(len(rows)))
    with (out / "preds.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rgb = load_frame(r["frame"])
            units = runner.units(rgb)
            preds = [{"kind": u.kind, "team": u.team, "score": round(u.score, 3),
                      "x": round(u.x, 1), "y": round(u.y, 1), "hp": round(u.hp, 3)}
                     for u in units]
            fh.write(json.dumps({"frame": r["frame"], "units": preds}) + "\n")
            if r["units"] is not None:
                p_in = [p for p in preds if ign[min(1079, int(p["y"])), min(1919, int(p["x"]))] == 0]
                gts = []
                for u in r["units"]:
                    cx, cy = box_center(u)
                    if ign[min(1079, max(0, int(cy))), min(1919, max(0, int(cx)))] == 0:
                        gts.append({**u, "x": cx, "y": cy})
                pairs, fp, fn = match(p_in, gts)
                for a, b in pairs:
                    stats[gts[b]["kind"]][0] += 1
                    team_ok += p_in[a]["team"] == gts[b]["team"]
                    team_n += 1
                    if gts[b].get("hp") is not None and gts[b]["kind"] != "ward":
                        hp_err[gts[b]["kind"]].append(abs(p_in[a]["hp"] - gts[b]["hp"]))
                for a in fp:
                    stats[p_in[a]["kind"]][1] += 1
                for b in fn:
                    stats[gts[b]["kind"]][2] += 1
            if i in review:
                review_image(rgb if valid is None else dim(rgb, valid), r["units"] or [], preds,
                             out / "review" / f"{i:03d}_{Path(r['frame']).stem}.jpg")
    res = {"frames": len(rows), "thr": args.thr, "net": str(args.net)}
    if args.labels:
        for c, (tp, fp, fn) in stats.items():
            res[c] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision_vs_pseudo": round(tp / max(1, tp + fp), 4),
                      "recall_vs_pseudo": round(tp / max(1, tp + fn), 4),
                      "hp_mae_vs_pseudo": round(float(np.mean(hp_err[c])), 4) if hp_err[c] else None}
        res["team_acc_vs_pseudo"] = round(team_ok / max(1, team_n), 4)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
