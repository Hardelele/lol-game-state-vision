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
    # → runs/units/eval/<имя>/{metrics.json, preds.jsonl, review/*.jpg}
"""

from __future__ import annotations

import argparse
import glob
import json
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
    args = ap.parse_args()

    keep = label_keep()
    ign = ignore_map(keep)
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
