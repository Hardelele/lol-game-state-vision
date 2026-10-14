"""Проверка детектора изменений на добиваниях (#818).

События — добивания отложенных роликов (#816): кадр смерти `k` при 60 кадр/с
(по CS полосы HUD) и место крипа на экране `xy` (доли кадра). Файл событий —
`runs/change/events_<id>.json`: [{"k": 6123, "xy": [0.61, 0.54]}, ...].

Для каждого события — кадры 15 кадр/с (шаг 4 при 60), так что один из
предсказываемых кадров — ровно `k`. Три карты на сетке 136×240:

* `surprise` — ошибка предсказания, делённая на выученный разброс;
* `copy` — |признаки сейчас − прошлый кадр, сдвинутый по камере|²: та же
  разность без обучения, в признаках;
* `pixels` — то же в картинке.

Два вопроса:

1. **Где** — ранг участка с крипом среди всех участков 8×8 клеток (510 на
   кадр) в окне ±2 кадра 15 кадр/с вокруг смерти, и то же место за 1,5 с до
   смерти (контроль: крипа бьют, но он жив).
2. **Когда** — в участках крипа за 1,5 с до смерти и 0,5 с после: на каком
   кадре карта сильнее всего; доля событий, где это кадр смерти ±1.

    python tools/eval_change.py --model runs/change/base/model.pt D37cbVNs0EU InEwM0UBAbc
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision import change_net as cn
from lolvision.paths import RUNS
from train_change import Prep, gray_small, keep_mask, read_frames

STEP = 4                        # кадров 60 кадр/с между кадрами модели
T = 8                           # участок — T×T клеток сетки
BEFORE, AFTER = 24, 8           # кадров модели до и после смерти (1,6 с и 0,5 с)
CTL = 22                        # контроль — на столько кадров модели раньше (~1,5 с)


def maps(prep: Prep, net: cn.ChangeNet, frs: list[np.ndarray]) -> dict[str, torch.Tensor]:
    """Карты для кадров 2…n-1 (каждый предсказан по двум прошлым)."""
    hud = keep_mask()[..., 0]
    import cv2
    hud_s = cv2.resize(hud, (hud.shape[1] // 4, hud.shape[0] // 4))
    gs = [gray_small(f) * hud_s for f in frs]
    f, r = [], []
    for i in range(0, len(frs), 16):
        a, b = prep(frs[i:i + 16])
        f.append(a)
        r.append(b)
    f, r = torch.cat(f), torch.cat(r)
    out = {"surprise": [], "copy": [], "pixels": []}
    for i in range(2, len(frs)):
        d13 = torch.tensor([cn.camera_shift(gs[i - 2], gs[i])], device=f.device)
        d23 = torch.tensor([cn.camera_shift(gs[i - 1], gs[i])], device=f.device)
        f1, ok1 = cn.shift_grid(f[i - 2:i - 1], d13)
        r1, _ = cn.shift_grid(r[i - 2:i - 1], d13)
        f2, ok2 = cn.shift_grid(f[i - 1:i], d23)
        r2, _ = cn.shift_grid(r[i - 1:i], d23)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pf, pr, lf, lr_ = net(f1, f2, r1, r2)
        ef, er = cn.errors(pf.float(), pr.float(), f[i:i + 1], r[i:i + 1])
        ok = ok1 * ok2
        out["surprise"].append(cn.surprise(ef, er, lf.float(), lr_.float()) * ok)
        out["copy"].append(((f2 - f[i:i + 1]) ** 2).mean(1, keepdim=True) * ok2)
        out["pixels"].append(((r2 - r[i:i + 1]) ** 2).mean(1, keepdim=True) * ok2)
    keep = (F.avg_pool2d(prep.keep, 4) > 0.99).float()
    keep = F.pad(keep, (0, 0, 0, f.shape[-2] - keep.shape[-2]))
    return {k: (torch.cat(v) * keep)[:, 0] for k, v in out.items()}


def tile_sum(m: torch.Tensor) -> torch.Tensor:
    """(N, 136, 240) → (N, 17, 30)."""
    return F.avg_pool2d(m[:, None], T)[:, 0] * T * T


def cells(xy) -> list[tuple[int, int]]:
    """Участки вокруг крипа: тело и над ним (полоска, всплывающее золото)."""
    cx, cy = int(xy[0] * 30), int(xy[1] * 17)
    return [(i, j) for i in range(cx - 1, cx + 2) for j in range(cy - 2, cy + 2)
            if 0 <= i < 30 and 0 <= j < 17]


def rank(score: torch.Tensor, cs) -> int:
    best = max(float(score[j, i]) for i, j in cs)
    return int((score > best).sum()) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default=str(RUNS / "change" / "base" / "model.pt"))
    args = ap.parse_args()
    ck = torch.load(args.model, map_location="cuda", weights_only=False)
    net = cn.ChangeNet(ck["width"]).cuda().eval()
    net.load_state_dict(ck["state"])
    prep = Prep("cuda")
    prep.scale = ck["scale"].cuda()
    rep = {}
    for vid in args.videos:
        evs = json.load(open(RUNS / "change" / f"events_{vid}.json", encoding="utf-8"))
        res: dict = {}
        for e in evs:
            n0 = BEFORE + CTL + 2                    # кадров модели до смерти
            first = e["k"] - STEP * n0
            frs = list(read_frames(vid, first / 60, (n0 + AFTER + 1) * STEP, fps=60))[::STEP]
            if len(frs) < n0 + AFTER + 1:
                continue
            ms = maps(prep, net, frs)               # индекс i ↔ кадр модели i+2
            kd = n0 - 2                              # индекс кадра смерти
            cs = cells(e["xy"])
            for name, m in ms.items():
                tm = tile_sum(m)
                win = tm[kd - 2:kd + 3].sum(0)
                ctl = tm[kd - CTL - 2:kd - CTL + 3].sum(0)
                series = torch.stack([max(tm[t, j, i] for i, j in cs) for t in
                                      range(kd - BEFORE, kd + AFTER + 1)])
                peak = int(series.argmax()) - BEFORE
                r = res.setdefault(name, {"rank": [], "ctl": [], "peak": []})
                r["rank"].append(rank(win, cs))
                r["ctl"].append(rank(ctl, cs))
                r["peak"].append(peak)
        print(f"{vid}: событий {len(res.get('copy', {}).get('rank', []))}")
        for name, r in res.items():
            a, c, p = (np.array(r[k]) for k in ("rank", "ctl", "peak"))
            print(f"  {name:8s} где: топ-5 {np.mean(a <= 5):.2f} (контроль {np.mean(c <= 5):.2f}), "
                  f"топ-20 {np.mean(a <= 20):.2f} ({np.mean(c <= 20):.2f}); "
                  f"когда: пик на смерти ±1 {np.mean(np.abs(p) <= 1):.2f}")
        rep[vid] = res
    out = RUNS / "change" / (args.model.replace("\\", "/").split("/")[-2] + "_eval.json")
    out.write_text(json.dumps(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
