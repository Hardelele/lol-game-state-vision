"""Обучить детектор и классификатор иконок миникарты (lolvision/minimap_icons.py).

Этап 1 — только синтетика: иконки Data Dragon на чистых фонах
(tools/minimap_backgrounds.py), генерация на лету в пуле процессов.
Этап 2 (`--real`) — к синтетике подмешиваются реальные миникарты обучающих
роликов с псевдо-метками (tools/pseudo_minimap.py): уверенные детекции —
положительные, сомнительные — «не учитывать»; вырезы для классификатора —
только там, где чемпион определён с ограничением на состав матча и с запасом.

Пример:
    python tools/train_minimap_icons.py --out runs/minimap/icons-syn
    python tools/train_minimap_icons.py --init runs/minimap/icons-syn \
        --real data/minimap --out runs/minimap/icons
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision import ddragon
from lolvision.minimap_icons import CROP, STRIDE, IconCls, IconDet, crops, decode, to_tensor
from lolvision.minimap_synth import MM, synth
from lolvision.paths import DATA

BG = DATA / "minimap" / "backgrounds.npz"
G = MM // STRIDE
SIGMA = 0.8                  # σ гауссианы цели, клеток сетки шага 2
MAX_CROPS = 12               # вырезов классификатора с одной миникарты


def targets(icons: list[tuple[float, float, int, bool]]) -> tuple[np.ndarray, ...]:
    """(x, y, команда, не учитывать) → тепловая карта 2×G×G, сдвиг 2×G×G,
    маска сдвига G×G, вес тепловой карты G×G (0 вокруг «не учитывать»)."""
    heat = np.zeros((2, G, G), np.float32)
    off = np.zeros((2, G, G), np.float32)
    om = np.zeros((G, G), np.float32)
    w = np.ones((G, G), np.float32)
    yy, xx = np.mgrid[:G, :G].astype(np.float32) + 0.5
    for x, y, team, ign in icons:
        gx, gy = x / STRIDE, y / STRIDE
        if ign:
            w[(xx - gx) ** 2 + (yy - gy) ** 2 <= 9] = 0
            continue
        g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * SIGMA ** 2))
        heat[team] = np.maximum(heat[team], g)
        j, i = int(gx), int(gy)
        if 0 <= j < G and 0 <= i < G:
            heat[team, i, j] = 1.0
            off[:, i, j] = (gx - j, gy - i)
            om[i, j] = 1
    return heat, off, om, w


_W: dict = {}


def _init_worker(real: str | None) -> None:
    cv2.setNumThreads(1)          # иначе процессы пула душат друг друга потоками
    _W["icons"] = list(ddragon.load_icons().values())
    _W["bgs"] = np.load(BG)["bg"]
    if real:
        # Миникарты — через mmap: каждому процессу пула не нужна своя копия.
        _W["real"] = {"mini": np.load(Path(real) / "pseudo_mini.npy", mmap_mode="r"),
                      "labels": json.loads((Path(real) / "pseudo_labels.json")
                                           .read_text(encoding="utf-8"))["labels"]}
    else:
        _W["real"] = None


def make_chunk(seed: int, n: int, real_frac: float) -> list[tuple]:
    """n примеров: (миникарта, [(x, y, команда, не учитывать)], вырезы, чемпионы, команды).

    Синтетика или (с вероятностью real_frac) реальная миникарта с
    псевдо-метками [(x, y, команда, не учитывать, чемпион или -1)].
    """
    rng = np.random.default_rng(seed)
    real = _W["real"]
    out = []
    for _ in range(n):
        if real is not None and rng.random() < real_frac:
            i = int(rng.integers(len(real["mini"])))
            img = np.array(real["mini"][i])
            labs = real["labels"][i]
            ics = [(float(a[0]), float(a[1]), int(a[2]), bool(a[3])) for a in labs]
            cl = [(float(a[0]), float(a[1]), int(a[4]), int(a[2])) for a in labs
                  if not a[3] and a[4] >= 0]
        else:
            s = synth(_W["bgs"][rng.integers(len(_W["bgs"]))], _W["icons"], rng)
            img = s.image
            ics = [(i.x, i.y, i.team, i.ignore) for i in s.icons]
            cl = [(i.x, i.y, i.champ, i.team) for i in s.icons
                  if not i.ignore and i.visible >= 0.55]
        rng.shuffle(cl)
        cl = cl[:MAX_CROPS]
        pts = [(x + rng.normal(0, 0.8), y + rng.normal(0, 0.8)) for x, y, _, _ in cl]
        cr = crops(img, pts) if cl else np.zeros((0, CROP, CROP, 3), np.uint8)
        out.append((img, ics, cr, np.array([c[2] for c in cl], np.int64),
                    np.array([c[3] for c in cl], np.int64)))
    return out


def stream(workers: int, batch: int, seed: int, real: str | None, real_frac: float):
    """Бесконечный поток пачек из пула процессов; заданий в полёте — ограниченно,
    чтобы готовые пачки не копились в памяти."""
    chunk = 8
    with mp.get_context("spawn").Pool(workers, _init_worker, (real,)) as pool:
        pending: deque = deque()
        k = 0
        buf: list = []
        while True:
            while len(pending) < workers * 3:
                pending.append(pool.apply_async(
                    make_chunk, (seed * 1_000_003 + k, chunk, real_frac)))
                k += 1
            buf += pending.popleft().get()
            while len(buf) >= batch:
                part, buf = buf[:batch], buf[batch:]
                img, ics, cr, ch, tm = zip(*part)
                tg = [targets(i) for i in ics]
                yield (torch.from_numpy(np.stack(img)),
                       *(torch.from_numpy(np.stack([t[j] for t in tg])) for j in range(4)),
                       torch.from_numpy(np.concatenate(cr)), torch.from_numpy(np.concatenate(ch)),
                       torch.from_numpy(np.concatenate(tm)))


def focal(logit: torch.Tensor, gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Фокальная потеря CenterNet; `w` — вес клетки (0 — не учитывать)."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = (1 - pos) * (1 - gt) ** 4
    w = w[:, None]
    lp = -(torch.log(p) * (1 - p) ** 2 * pos * w).sum()
    ln = -(torch.log(1 - p) * p ** 2 * neg * w).sum()
    return (lp + ln) / pos.sum().clamp(min=1)


def synth_val(n: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    icons = list(ddragon.load_icons().values())
    bgs = np.load(BG)["bg"]
    return [synth(bgs[rng.integers(len(bgs))], icons, rng) for _ in range(n)]


@torch.no_grad()
def evaluate(det, cls, val, dev) -> dict:
    det.eval()
    cls.eval()
    tp = fp = fn = 0
    errs, ok, tot = [], 0, 0
    for k in range(0, len(val), 32):
        part = val[k:k + 32]
        heat, off = det(to_tensor(np.stack([s.image for s in part]), dev))
        for s, ds in zip(part, decode(heat, off, 0.3)):
            gts = [i for i in s.icons if not i.ignore]
            ign = [i for i in s.icons if i.ignore]
            used = set()
            for d in ds:
                best, bj = 4.0, -1
                for j, g in enumerate(gts):
                    dd = np.hypot(d.x - g.x, d.y - g.y)
                    if j not in used and dd < best:
                        best, bj = dd, j
                if bj >= 0:
                    used.add(bj)
                    tp += 1
                    errs.append(best)
                elif not any(np.hypot(d.x - g.x, d.y - g.y) < 4 for g in ign):
                    fp += 1
            fn += len(gts) - len(used)
            vis = [g for g in gts if g.visible >= 0.55]
            if vis:
                lc, _ = cls(to_tensor(crops(s.image, [(g.x, g.y) for g in vis]), dev))
                ok += int((lc.argmax(1).cpu().numpy() == np.array([g.champ for g in vis])).sum())
                tot += len(vis)
    det.train()
    cls.train()
    return {"precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4),
            "pos_err_px_median": round(float(np.median(errs)), 3) if errs else None,
            "cls_top1_all": round(ok / max(1, tot), 4)}


def save(out: Path, det, cls, champs, args, rep) -> None:
    tmp = out / "model.pt.tmp"
    torch.save({"det": det.state_dict(), "cls": cls.state_dict(), "champions": champs,
                "ddragon": ddragon.DEFAULT_VERSION, "det_w": 32, "cls_w": 48, "thr": 0.3,
                "args": vars(args), "synth_val": rep}, tmp)
    tmp.replace(out / "model.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--init", default=None, help="прогон, с которого продолжить")
    ap.add_argument("--real", default=None,
                    help="каталог псевдо-меток (pseudo_mini.npy, pseudo_labels.json)")
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    champs = ddragon.champions()
    det, cls = IconDet(), IconCls(len(champs))
    if args.init:
        ck = torch.load(Path(args.init) / "model.pt", map_location="cpu", weights_only=False)
        assert ck["champions"] == champs
        det.load_state_dict(ck["det"])
        cls.load_state_dict(ck["cls"])
    det.to(dev)
    cls.to(dev)
    opt = torch.optim.AdamW(list(det.parameters()) + list(cls.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters,
                                                pct_start=0.05)
    dl = stream(args.workers, args.batch, args.seed, args.real, args.real_frac)
    val = synth_val(400)
    log = (out / "train.log").open("a", encoding="utf-8")
    t0 = time.time()
    it = 0
    for img, heat, off, om, w, cr, ch, tm in dl:
        x = to_tensor(img.numpy(), dev)
        heat, off, om, w = heat.to(dev), off.to(dev), om.to(dev), w.to(dev)
        ph, po = det(x)
        l_heat = focal(ph, heat, w)
        l_off = (F.l1_loss(po, off, reduction="none").sum(1) * om).sum() / om.sum().clamp(min=1)
        loss = l_heat + l_off
        if len(cr):
            lc, lt = cls(to_tensor(cr.numpy(), dev))
            l_cls = F.cross_entropy(lc, ch.to(dev), label_smoothing=0.05)
            l_team = F.cross_entropy(lt, tm.to(dev))
            loss = loss + l_cls + 0.2 * l_team
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        it += 1
        if it % 200 == 0:
            msg = (f"it {it} loss {loss.item():.3f} heat {l_heat.item():.3f} off {l_off.item():.3f}"
                   f" cls {l_cls.item() if len(cr) else 0:.3f} {time.time() - t0:.0f}s")
            if it % 2000 == 0 or it == args.iters:
                rep = evaluate(det, cls, val, dev)
                msg += " " + json.dumps(rep)
                save(out, det, cls, champs, args, rep)      # прерванный прогон не пропадает
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if it >= args.iters:
            break
    rep = evaluate(det, cls, val, dev)
    save(out, det, cls, champs, args, rep)
    (out / "report.json").write_text(json.dumps({"synth_val": rep, "args": vars(args),
                                                 "seconds": round(time.time() - t0)},
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    print("готово", rep, flush=True)


if __name__ == "__main__":
    main()
