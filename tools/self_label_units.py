"""Дополнить псевдо-метки юнитов ответами самих сетей (#773).

Зачем. Учитель (читатели полосок) пропускает юнитов, чья полоска перекрыта
или смазана, — у миньонов ~30%. Пока сеть смотрела на полоски, это почти не
мешало. Со стиранием полосок (BarAug) тело без метки учится как фон: сеть
становится неуверенной и теряет полноту и на кадрах с полосками.

Как. По обучающим кадрам прогоняются сети (`--nets`) на кадре как есть и на
кадре со стёртыми полосками всех размеченных юнитов; ответы с оценкой ≥ 0.2
сливаются по радиусу совпадения (берётся максимум оценки). Ответы рядом с
уже размеченным юнитом отбрасываются. Остальные:

* оценка ≥ `--add` — новая метка: класс и команда от сети, ХП нет, полоски
  нет (`"bar": null`, `"source": "self"`);
* меньше — точка игнорирования (`"ignore"`): вокруг неё нет потерь
  тепловой карты (tools/train_units.py).

Пример:
    python tools/self_label_units.py ibUVbSX7ARU zJvTSjEnKNE ... \\
        --nets runs/units/net/model.pt runs/units/net-nobars/model.pt
    # → data/units/labels_self/<набор>.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from build_unit_labels import LABELS
from eval_units import load_frame
from lolvision.bar_erase import bar_rect, erase
from lolvision.unit_bars import BODY
from lolvision.unit_net import UnitNetRunner
from lolvision.unit_targets import RADIUS, box_center, ignore_map, unit_keep
from train_units import load_rows

OUT = LABELS.parent / "labels_self"


def merge(preds: list[dict]) -> list[dict]:
    """Слить ответы нескольких прогонов: жадно по убыванию оценки, ответ
    ближе радиуса своего класса к уже взятому — тот же юнит."""
    out = []
    for p in sorted(preds, key=lambda p: -p["score"]):
        if all(np.hypot(p["x"] - q["x"], p["y"] - q["y"]) > RADIUS[p["kind"]] for q in out):
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+", help="наборы меток (как --train у train_units.py)")
    ap.add_argument("--nets", nargs="+", type=Path, required=True)
    ap.add_argument("--add", type=float, default=0.6)
    ap.add_argument("--low", type=float, default=0.2)
    args = ap.parse_args()

    keep = unit_keep()
    ign = ignore_map(keep)
    runners = [UnitNetRunner(n, keep=keep, valid=1 - ign, thr=args.low) for n in args.nets]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        rows = load_rows([name])
        added = ignored = 0
        out_name = name.replace(":", "_")
        with (OUT / f"{out_name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                rgb = load_frame(r["frame"])
                erased = erase(rgb.copy(), [bar_rect(u, shape=rgb.shape) for u in r["units"]],
                               rng=np.random.default_rng(0))
                preds = []
                for run in runners:
                    for img in (rgb, erased):
                        preds += [{"kind": u.kind, "team": u.team, "score": u.score,
                                   "x": u.x, "y": u.y} for u in run.units(img)]
                labeled = [(*box_center(u), u["kind"]) for u in r["units"]]
                extra, ignore = [], []
                for p in merge(preds):
                    rad = max(RADIUS[p["kind"]], 28.0)
                    if any(np.hypot(p["x"] - x, p["y"] - y) <= max(rad, RADIUS[k])
                           for x, y, k in labeled):
                        continue
                    if p["score"] >= args.add:
                        _, _, w, h = BODY[p["kind"]]
                        extra.append({"kind": p["kind"], "team": p["team"], "bar": None,
                                      "box": [round(p["x"] - w / 2, 1), round(p["y"] - h / 2, 1),
                                              round(p["x"] + w / 2, 1), round(p["y"] + h / 2, 1)],
                                      "hp": None, "score": round(p["score"], 3),
                                      "source": "self"})
                    else:
                        ignore.append([round(p["x"], 1), round(p["y"], 1), p["kind"]])
                added += len(extra)
                ignored += len(ignore)
                fh.write(json.dumps({**r, "units": r["units"] + extra, "ignore": ignore}) + "\n")
        print(f"{name}: кадров {len(rows)}, добавлено {added}, игнорировать {ignored}", flush=True)


if __name__ == "__main__":
    main()
