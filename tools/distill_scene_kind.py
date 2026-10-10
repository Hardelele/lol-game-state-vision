"""Дистилляция типа сцены: DINOv2 + голова (учитель) → своя маленькая CNN.

Зачем. Учитель (lolvision/scene_kind.py) стоит ~10 мс на кадр; ученик
(lolvision/scene_net.py) — около миллисекунды. Размеченных кадров мало
(экранов загрузки — десятки), поэтому ученик учится по ответам учителя на
всём пуле кадров, в том числе неразмеченных, а там, где есть ручная
разметка, — ещё и по ней.

Пул — только обучающая часть: ролики split["train"] с разметкой, кадры
стримов без разметки (data/not_game/<id>, кроме split["test"]), кадры
наблюдателя dataset/videos и data/scene_kind/spectator (кроме coords_test).
Проверка — те же отложенные ролики и то же объединение с головой «игра /
не игра», что у учителя (scene_kind_dino.py).

Пример:
    python tools/distill_scene_kind.py --out runs/scene_kind/net
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision.paths import DATASET, DATA, NOT_GAME_DATA, RUNS
from lolvision.scene_kind import CLASSES, DEFAULT_HEAD, Head, fuse, load_dino
from lolvision.scene_net import IN_H, IN_W, SceneNet, small
from runlog import tee_stdout
from scene_kind_dino import cached, patch_p_game, spectator_rows, stream_rows
from train_not_game import SPLIT
from train_scene_kind import confusion, show

SMALL_CACHE = RUNS / "scene_kind" / "small"


def pool_rows(split: dict) -> list[dict]:
    """Кадры обучающей части: размеченные (cls ≥ 0) и нет (cls = -1)."""
    test = set(split["test"]) | set(split["coords_test"])
    rows = stream_rows(split["train"])
    have = {str(r["full"]) for r in rows}
    for man in sorted(NOT_GAME_DATA.glob("*.csv")):
        vid = man.stem
        if vid in test or vid in split["train"]:
            continue
        for r in csv.DictReader(man.open(encoding="utf-8")):
            p = NOT_GAME_DATA / r["full"]
            if str(p) not in have:
                rows.append({"video_id": vid, "t_sec": float(r["t_sec"]), "full": p, "cls": -1,
                             "source": "unlabeled", "kind": ""})
    spec = sorted(p.name for p in (DATASET / "videos").iterdir() if p.is_dir())
    for v in spec:
        if v not in test:
            rows += spectator_rows(v)
    for d in sorted((DATA / "scene_kind" / "spectator").glob("*")):
        if d.name in test:
            continue
        for p in sorted(d.glob("*.jpg")):
            rows.append({"video_id": f"spec-{d.name}", "t_sec": float(p.stem), "full": p,
                         "cls": 0, "source": "spectator", "kind": "lol_game_spectator"})
    return rows


def small_images(rows: list[dict], tag: str) -> np.ndarray:
    """Все кадры 256×144 uint8 (кэш по набору: пересобирается при смене пула)."""
    SMALL_CACHE.mkdir(parents=True, exist_ok=True)
    keys = [str(r["full"]) for r in rows]
    cp = SMALL_CACHE / f"{tag}-v2.npz"        # v2: прореживание + INTER_AREA (scene_net.small)
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        if z["k"].tolist() == keys:
            return z["x"]
    x = np.stack([small(cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)) for k in keys])
    np.savez(cp, k=np.array(keys, dtype=object), x=x)
    return x


def teacher_probs(rows: list[dict], dev: str) -> np.ndarray:
    ck = torch.load(DEFAULT_HEAD, map_location=dev)
    head = Head(ck["cin"], len(ck["classes"])).to(dev).eval()
    head.load_state_dict(ck["state"])
    x = cached(load_dino(dev=dev), rows, dev)
    with torch.no_grad():
        return torch.softmax(head(torch.from_numpy(x).to(dev)), 1).cpu().numpy()


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 3, H, W) в [0, 1]. Небольшой сдвиг/масштаб, яркость, контраст,
    насыщенность, шум сжатия — разные стримы, разное качество."""
    b = x.shape[0]
    s = torch.empty(b, device=x.device).uniform_(0.9, 1.0)
    tx = torch.empty(b, device=x.device).uniform_(-1, 1) * (1 - s)
    ty = torch.empty(b, device=x.device).uniform_(-1, 1) * (1 - s)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                      align_corners=False, padding_mode="border")
    br = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    ct = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * ct + m * br
    gray = x.mean(1, keepdim=True)
    sat = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    x = gray + (x - gray) * sat
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--hard", type=float, default=0.5, help="вес ручных меток рядом с учителем")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="ансамбль: по ученику на зерно, ответы усредняются")
    ap.add_argument("--patch-model", type=Path,
                    default=RUNS / "patches" / "cnn-split16-v2-notgame" / "model.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with tee_stdout(args.out / "distill.log"):
        run(args)


def run(args) -> None:
    dev = "cuda"
    t0 = time.perf_counter()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    tr = pool_rows(split)
    spec = sorted(p.name for p in (DATASET / "videos").iterdir() if p.is_dir())
    te = stream_rows(split["test"]) + [r for v in spec if v in split["coords_test"]
                                       for r in spectator_rows(v)]
    te = [r for r in te if r["full"] is not None]
    lab = np.array([r["cls"] for r in tr])
    print(f"пул: {len(tr)} кадров, размечено {(lab >= 0).sum()}, без разметки {(lab < 0).sum()}; "
          f"проверка {len(te)}")

    p_t = teacher_probs(tr, dev)
    print("учитель по пулу: " + ", ".join(f"{n} {v}" for n, v in
                                          zip(CLASSES, np.bincount(p_t.argmax(1), minlength=6))))
    x_tr = small_images(tr, "pool")
    x_te = small_images(te, "test")
    print(f"кадры 256×144 готовы ({time.perf_counter() - t0:.0f} с)")

    # Вес кадра: редкие по учителю классы чаще (иначе ~70% пула — обычная игра).
    cls_t = p_t.argmax(1)
    freq = np.bincount(cls_t, minlength=len(CLASSES)).astype(np.float64)
    w = 1.0 / np.sqrt(freq[cls_t])
    w /= w.sum()

    xg = torch.from_numpy(x_tr)                 # на CPU, батчи — на карту
    pt_g = torch.from_numpy(p_t).float()
    lab_g = torch.from_numpy(lab).long()
    steps_per_epoch = max(1, len(tr) // 128)

    def train_one(seed: int) -> SceneNet:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = SceneNet().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3,
                                                    total_steps=args.epochs * steps_per_epoch)
        for ep in range(args.epochs):
            net.train()
            tot = 0.0
            for _ in range(steps_per_epoch):
                idx = torch.from_numpy(rng.choice(len(tr), 128, p=w))
                xb = augment(xg[idx].to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255)
                logit = net(xb)
                tgt = pt_g[idx].to(dev)
                soft = F.kl_div(F.log_softmax(logit / args.temp, 1),
                                (tgt.clamp_min(1e-6).log() / args.temp).softmax(1),
                                reduction="batchmean") * args.temp ** 2
                yb = lab_g[idx].to(dev)
                m = yb >= 0
                hard = F.cross_entropy(logit[m], yb[m]) if m.any() else logit.sum() * 0
                loss = soft + args.hard * hard
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach())
            if ep == args.epochs - 1:
                print(f"зерно {seed}: потеря последней эпохи {tot / steps_per_epoch:.4f}",
                      flush=True)
        return net.eval()

    @torch.no_grad()
    def test_probs(net: SceneNet) -> np.ndarray:
        return torch.cat([torch.softmax(net(torch.from_numpy(x_te[j:j + 256]).to(dev)
                                            .permute(0, 3, 1, 2).float() / 255), 1).cpu()
                          for j in range(0, len(x_te), 256)]).numpy()

    y_te = np.array([r["cls"] for r in te])
    pgd = patch_p_game(args.patch_model, te, dev)
    thr = pgd.pop("threshold")
    out_kinds = set(split.get("out_of_scope_kinds", []))
    ins = np.array([r["kind"] not in out_kinds for r in te])

    def score(p: np.ndarray) -> dict:
        fused = np.array([fuse(int(c), pgd.get(i, 1.0), thr) for i, c in enumerate(p.argmax(1))])
        return confusion(y_te[ins], fused[ins]), fused

    nets, probs = [], []
    for seed in args.seeds:
        net = train_one(seed)
        nets.append(net)
        probs.append(test_probs(net))
        print(f"  зерно {seed}: верно {score(probs[-1])[0]['accuracy']:.4f}")
    p = np.mean(probs, 0)
    c, fused = score(p)
    vids = np.array([r["video_id"] for r in te])
    c["by_video"] = {v: confusion(y_te[ins & (vids == v)], fused[ins & (vids == v)])
                     for v in sorted(set(vids))}
    c["seeds"] = {str(sd): score(q)[0]["accuracy"] for sd, q in zip(args.seeds, probs)}
    show(f"отложенные ролики, ансамбль из {len(nets)} учеников + голова «игра / не игра» "
         "(без ARAM)", c)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "scene_net", "classes": CLASSES, "widths": [24, 48, 96, 160, 192],
                "input": [IN_W, IN_H], "states": [n.state_dict() for n in nets],
                "trained": {"pool": len(tr), "epochs": args.epochs, "temp": args.temp,
                            "hard": args.hard, "seeds": args.seeds}}, args.out / "model.pt")
    (args.out / "report.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    print(f"\n→ {args.out} ({time.perf_counter() - t0:.0f} с)")


if __name__ == "__main__":
    main()
