"""Разметка Tab и магазина поверх игровых кадров стримеров (overlays.csv).

Зачем. Таблица счёта (Tab) и магазин открываются на секунды, стримеры
делают это постоянно — размечать вручную тысячи игровых кадров долго.
Здесь — классификатор по признакам DINOv2 (кэш scene_kind_dino.py):

1. образцы — проверенные глазами кадры Tab и магазина (SEEDS ниже);
2. «игра» — кадры, далёкие от обоих образцов (косинус < 0.25);
3. линейный классификатор игра / Tab / магазин; уверенные (≥ 0.97)
   ответы добавляются к образцам, уверенная игра (≥ 0.995) — к игре;
4. второй классификатор; Tab или магазин — при вероятности ≥ 0.9.

Проверка глазами (docs/not-game.md): среди ответов ≥ 0.9 ложных Tab 1 из
46, в полосе 0.3–0.9 почти всё — обычная игра. Результат —
dataset/not_game/<id>/overlays.csv для роликов с labels.csv.

Пример:
    python tools/label_overlays.py
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision.dataset_io import label_at, load_labels
from lolvision.paths import NOT_GAME_LABELS, RUNS

CACHE = RUNS / "scene_kind" / "cache"
# (ролик, секунда) — проверено глазами по листам кандидатов.
SEEDS = {
    "tab": [("Pa8fH6tfZB0", t) for t in (310, 313, 157, 219, 217, 921, 682, 909, 1048, 804,
                                         914, 194, 317, 1088, 1086, 218, 629, 1380, 1051, 691,
                                         591, 1049, 513, 536, 92, 630, 1038, 1201, 997, 823,
                                         690, 155, 689, 447, 1199, 1170)]
           + [("ruTlP0EyRYM", t) for t in (24, 43, 41, 42)],
    "shop": [("Pa8fH6tfZB0", t) for t in (692, 1164, 922, 905, 517, 518, 915, 917, 916, 648,
                                          1035, 518, 824, 1280, 298)]
            + [("LPQwQCkfnkg", t) for t in (666, 774, 776)],
}


def frames() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Игровые кадры роликов с разметкой: ключ (путь), ролик, секунда, признаки."""
    keys, vids, ts, feats = [], [], [], []
    for f in sorted(CACHE.glob("*.npz")):
        vid = f.stem
        if not (NOT_GAME_LABELS / vid / "labels.csv").exists():
            continue
        labels = load_labels(vid)
        z = np.load(f, allow_pickle=True)
        for k, v in zip(z["k"], z["f"]):
            k = str(k)
            if "not_game" not in k:
                continue
            t = int(Path(k).stem) / 1000
            if label_at(labels, t)[0] != "game":
                continue
            keys.append(k)
            vids.append(vid)
            ts.append(t)
            feats.append(v)
    return keys, vids, np.array(ts), np.stack(feats)


def probe(x: np.ndarray, y: np.ndarray, balance: bool) -> np.ndarray:
    m = y >= 0
    xt = torch.tensor(x[m], dtype=torch.float32)
    yt = torch.tensor(y[m])
    w = torch.zeros(x.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    cnt = np.bincount(y[m], minlength=3).astype(np.float32)
    cw = torch.tensor(cnt[0] / np.maximum(cnt, 1)) if balance else None
    opt = torch.optim.Adam([w, b], lr=1e-3)
    for _ in range(300):
        loss = F.cross_entropy(xt @ w + b, yt, weight=cw) + 1e-3 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.softmax(torch.tensor(x, dtype=torch.float32) @ w + b, 1).numpy()


def main() -> None:
    keys, vids, ts, feats = frames()
    x = (feats - feats.mean(0)) / (feats.std(0) + 1e-3)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    where = {(v, round(t, 1)): i for i, (v, t) in enumerate(zip(vids, ts))}
    y = np.full(len(keys), -1)
    for c, name in ((1, "tab"), (2, "shop")):
        for v, t in SEEDS[name]:
            i = where.get((v, float(t)))
            if i is not None:
                y[i] = c
    pt, ps = xn[y == 1].mean(0), xn[y == 2].mean(0)
    far = np.nonzero((xn @ pt < 0.25) & (xn @ ps < 0.25) & (y == -1))[0]
    rng = np.random.default_rng(0)
    y[rng.choice(far, size=min(1500, len(far)), replace=False)] = 0
    print(f"образцы: Tab {(y == 1).sum()}, магазин {(y == 2).sum()}, игра {(y == 0).sum()}; "
          f"игровых кадров {len(y)}")
    p = probe(x, y, balance=True)
    y[(y == -1) & (p[:, 1] >= 0.97)] = 1
    y[(y == -1) & (p[:, 2] >= 0.97)] = 2
    y[(y == -1) & (p[:, 0] >= 0.995)] = 0
    p = probe(x, y, balance=False)

    by = collections.defaultdict(list)
    for v, t, q in zip(vids, ts, p):
        by[v].append((float(t), q))
    for v, items in sorted(by.items()):
        items.sort()
        tt = np.array([t for t, _ in items])
        step = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0
        rows: list[list] = []
        for t, q in items:
            k = "tab" if q[1] >= 0.9 else "shop" if q[2] >= 0.9 else None
            if k is None:
                continue
            if rows and rows[-1][2] == k and abs(rows[-1][1] - t) < 1e-6:
                rows[-1][1] = t + step
            else:
                rows.append([t, t + step, k])
        out = NOT_GAME_LABELS / v / "overlays.csv"
        if not rows:
            out.unlink(missing_ok=True)
            continue
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["start_sec", "end_sec", "kind", "note"])
            for a, b, k in rows:
                w.writerow([f"{a:g}", f"{b:g}", k, "tools/label_overlays.py, p >= 0.9"])
        n = collections.Counter()
        for a, b, k in rows:
            n[k] += int(round((b - a) / step))
        print(f"  {v}: Tab {n['tab']} кадров, магазин {n['shop']}")


if __name__ == "__main__":
    main()
