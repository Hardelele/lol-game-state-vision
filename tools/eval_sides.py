"""Проверить свои / враги / сам и сторону на кадрах (#790, lolvision/sides.py).

Зачем. Готовность 2A — на записанных кадрах своей игры за обе стороны
отношение и сторона верны, а за красных сторона не перевёрнута. Здесь —
прогон sides.assign по ответам детектора на кадрах (preds.jsonl от
tools/eval_units.py) с командой своего игрока из Live Client API и счёт:
отказы по причинам, «сам» больше одного на кадре, расхождения цвета полоски
с детектором, а с ручной разметкой (--gold) — доля уверенных ошибок. Лист
плашек чемпионов с ответом (--sheet) — для проверки глазами.

Видеокарта не нужна: детектор уже отработал в eval_units.py.

Пример (запись своей игры за CHAOS, команда — из журнала Live Client API):
    python tools/eval_units.py --frames "data/own/<запись>/full/*.jpg" --no-mask --name own-chaos
    python tools/eval_sides.py --preds runs/units/eval/own-chaos/preds.jsonl --my-team CHAOS \
        --sheet runs/units/eval/own-chaos/sides.jpg
    python tools/eval_sides.py --preds ... --my-team CHAOS --gold data/own/<запись>/sides_gold.jsonl

Формат --gold (jsonl, по строке на кадр; x, y — пиксели 1080p, как в preds):
    {"frame": "data/own/.../00012000.jpg",
     "units": [{"kind": "champion", "x": 957, "y": 484, "relation": "self"}, ...]}
relation в разметке: ally / enemy / self или null — «здесь должен быть отказ».
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from lolvision.paths import ROOT
from lolvision.records import UnitRec
from lolvision.sides import SideContext, assign

REF_W, REF_H = 1920, 1080          # preds.jsonl — пиксели кадра 1080p
GOLD_DIST = 40                     # px: ответ дальше от метки — не она


def load_frame(path: str, root: Path = ROOT) -> np.ndarray | None:
    p = root / path
    bgr = cv2.imread(str(p)) if p.is_file() else None
    if bgr is None:
        return None
    if bgr.shape[:2] != (REF_H, REF_W):
        bgr = cv2.resize(bgr, (REF_W, REF_H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr[..., ::-1])


def recs_of(units: list[dict]) -> list[UnitRec]:
    return [UnitRec(u["kind"], u["team"], u["x"] / REF_W, u["y"] / REF_H,
                    float(u.get("hp", 0.0)), float(u.get("score", 0.0))) for u in units]


def gold_index(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["frame"]] = d["units"]
    return out


def tile(rgb: np.ndarray, r: UnitRec) -> np.ndarray:
    x, y = int(r.x * REF_W), int(r.y * REF_H)
    t = rgb[max(0, y - 170):max(1, y - 40), max(0, x - 90):x + 90][..., ::-1]
    t = cv2.resize(np.ascontiguousarray(t), (270, 195))
    text = f"{r.team[:2]} {r.color or '-'} {r.relation or 'null'} {r.side or 'null'}"
    for c, w in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(t, text, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, w)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, required=True, help="preds.jsonl от tools/eval_units.py")
    ap.add_argument("--view", choices=("player", "spectator"), default="player")
    ap.add_argument("--my-team", choices=("ORDER", "CHAOS", "none"), required=True,
                    help="команда своего игрока из Live Client API; none — API не было")
    ap.add_argument("--gold", type=Path, default=None, help="ручная разметка relation (см. выше)")
    ap.add_argument("--sheet", type=Path, default=None, help="лист плашек чемпионов с ответами")
    ap.add_argument("--root", type=Path, default=ROOT, help="откуда пути кадров в preds.jsonl")
    ap.add_argument("--min-score", type=float, default=0.0, help="ответы детектора слабее — мимо")
    a = ap.parse_args()

    my = None if a.my_team == "none" else a.my_team
    ctx = SideContext(a.view, my, None if my else "--my-team none")
    gold = gold_index(a.gold)
    stat = collections.Counter()
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tiles, multi_self, frames = [], [], 0
    for line in a.preds.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        rgb = load_frame(d["frame"], a.root)
        if rgb is None:
            stat["frame_missing"] += 1
            continue
        frames += 1
        recs = assign(recs_of([u for u in d["units"] if u["score"] >= a.min_score]), ctx, rgb)
        if sum(r.relation == "self" for r in recs) > 1:
            multi_self.append(d["frame"])
        for r in recs:
            c = by_kind[r.kind]
            c["n"] += 1
            c[f"relation={r.relation}" if r.relation else f"refused: {r.relation_reason}"] += 1
            c[f"side={r.side}" if r.side else f"side refused: {r.side_reason}"] += 1
            if r.kind == "champion" and a.sheet is not None:
                tiles.append(tile(rgb, r))
        for g in gold.get(d["frame"], []):
            near = [r for r in recs if r.kind == g["kind"]
                    and np.hypot(r.x * REF_W - g["x"], r.y * REF_H - g["y"]) <= GOLD_DIST]
            if not near:
                stat["gold_missed"] += 1
                continue
            r = min(near, key=lambda r: np.hypot(r.x * REF_W - g["x"], r.y * REF_H - g["y"]))
            want = g.get("relation")
            if r.relation is None:
                stat["gold_refused"] += 1
            elif r.relation == want:
                stat["gold_right"] += 1
            else:
                stat["gold_wrong"] += 1          # уверенная ошибка (или ответ там, где нужен отказ)
    out = {"frames": frames, "view": a.view, "my_team": my, **stat,
           "frames_with_several_self": len(multi_self),
           "kinds": {k: dict(sorted(v.items())) for k, v in sorted(by_kind.items())}}
    judged = stat["gold_right"] + stat["gold_wrong"] + stat["gold_refused"]
    if judged:
        out["gold_wrong_share"] = round(stat["gold_wrong"] / judged, 4)
        out["gold_refused_share"] = round(stat["gold_refused"] / judged, 4)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()
    if a.sheet is not None and tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        a.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.sheet), sheet)
        print(f"лист: {a.sheet} ({len(tiles)} плашек)")


if __name__ == "__main__":
    main()
