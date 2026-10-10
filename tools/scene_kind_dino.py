"""Тип сцены по всему кадру: замороженный DINOv2 + маленькая голова.

Зачем. Голова на признаках патчевой модели (train_scene_kind.py) путает
клиент с игрой: эти признаки учились искать место на карте, а не узнавать
интерфейс. Здесь — общий энкодер изображений по всему кадру без маски:
таблица Tab, магазин, клиент и экран загрузки — крупные элементы с
постоянным расположением, а DINOv2 их хорошо различает.

Признаки кадра: кадр 16:9 → 392×224 (сетка патчей 28×16), токен CLS,
среднее патчей и средние по сетке 3×3 (расположение элементов): 11×384.
Классы и разметка — как в train_scene_kind.py (labels.csv + overlays.csv);
игра наблюдателем — полные кадры dataset/videos (обучение и проверка — по
роликам, как у модели камеры: coords_test из split.json).

Итоговый ответ — вместе с головой «игра / не игра» патчевой модели
(--patch-model): DINO говорит, какой интерфейс на экране, а голова
патчевой модели — узнаются ли места Ущелья. Ответ «игра» при P(игра) ниже
её порога становится «other»: так отсекаются чужие игры с видом сверху
(Dota 2, StarCraft), которые DINO принимает за League. Подклассы из
out_of_scope_kinds (ARAM) в метрики не входят.

Пример:
    python tools/scene_kind_dino.py --out runs/scene_kind/dino-s
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision.paths import DATASET, DEFAULT_LAYOUT, RUNS
from runlog import tee_stdout
from train_not_game import SPLIT, not_game_rows
from train_scene_kind import CLASSES, confusion, load_overlays, scene_class, show, weights

IN_W, IN_H = 392, 224
CACHE = RUNS / "scene_kind" / "cache"
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def load_dino(arch: str = "dinov2_vits14", dev: str = "cuda"):
    repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    vit = (torch.hub.load(str(repo), arch, source="local") if repo.exists()
           else torch.hub.load("facebookresearch/dinov2", arch))
    return vit.to(dev).eval()


@torch.no_grad()
def dino_features(vit, paths: list[Path], dev: str, batch: int = 32) -> np.ndarray:
    out = []
    for j in range(0, len(paths), batch):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((IN_W, IN_H), Image.BILINEAR))
               for p in paths[j:j + batch]]
        x = torch.from_numpy(np.stack(ims)).to(dev).permute(0, 3, 1, 2).float() / 255
        x = (x - MEAN.to(dev)) / STD.to(dev)
        f = vit.forward_features(x)
        cls = f["x_norm_clstoken"]
        pt = f["x_norm_patchtokens"].transpose(1, 2).reshape(len(ims), -1, IN_H // 14, IN_W // 14)
        grid = F.adaptive_avg_pool2d(pt, 3).flatten(2).transpose(1, 2)      # (B, 9, C)
        feat = torch.cat([cls[:, None], pt.mean((2, 3))[:, None], grid], 1).flatten(1)
        out.append(feat.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 11 * 384), np.float32)


def cached(vit, rows: list[dict], dev: str) -> np.ndarray:
    """Признаки по ролику с кэшем (ключ — путь кадра)."""
    out = np.zeros((len(rows), 11 * vit.embed_dim), np.float32)
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["video_id"], []).append(i)
    for vid, idx in by.items():
        cp = CACHE / f"{vid}.npz"
        have = {}
        if cp.exists():
            z = np.load(cp, allow_pickle=True)
            have = dict(zip(z["k"].tolist(), z["f"]))
        need = [i for i in idx if str(rows[i]["full"]) not in have]
        if need:
            f = dino_features(vit, [rows[i]["full"] for i in need], dev)
            for i, v in zip(need, f):
                have[str(rows[i]["full"])] = v
            cp.parent.mkdir(parents=True, exist_ok=True)
            ks = sorted(have)
            np.savez(cp, k=np.array(ks, dtype=object), f=np.stack([have[k] for k in ks]))
        for i in idx:
            out[i] = have[str(rows[i]["full"])]
    return out


def spectator_rows(vid: str) -> list[dict]:
    """Полные кадры ролика наблюдателя: игра, кроме кадров с пометкой «чёрный»/«победа»."""
    d = DATASET / "videos" / vid
    out = []
    for r in csv.DictReader((d / "frames.csv").open(encoding="utf-8")):
        note = r.get("note", "")
        if any(w in note for w in ("чёрный", "победы", "заставк")):
            continue
        p = d / r["frame"]
        if p.exists():
            out.append({"video_id": vid, "t_sec": float(r["t_sec"]), "full": p, "y": 1,
                        "kind": "lol_game_spectator", "source": "spectator", "cls": 0})
    return out


def stream_rows(vids: list[str]) -> list[dict]:
    rows = []
    for v in vids:
        ov = load_overlays(v)
        for r in not_game_rows(v):
            r["cls"] = CLASSES.index(scene_class(r, ov))
            rows.append(r)
    return rows


class Head(nn.Module):
    def __init__(self, cin: int, k: int, hidden: int = 256):
        super().__init__()
        self.register_buffer("mu", torch.zeros(cin))
        self.register_buffer("sd", torch.ones(cin))
        self.net = nn.Sequential(nn.Dropout(0.3), nn.Linear(cin, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(0.3), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def fit(x, y, w, dev, epochs=200, seed=0) -> Head:
    torch.manual_seed(seed)
    h = Head(x.shape[1], len(CLASSES)).to(dev)
    xt = torch.from_numpy(x).to(dev)
    h.mu.copy_(xt.mean(0))
    h.sd.copy_(xt.std(0).clamp(min=1e-3))
    yt = torch.from_numpy(y).long().to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=5e-2)
    h.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for j in range(0, len(perm), 512):
            k = perm[j:j + 512]
            loss = (F.cross_entropy(h(xt[k]), yt[k], reduction="none") * wt[k]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return h.eval()


def patch_p_game(path: Path, rows: list[dict], dev: str) -> dict:
    """P(игра) головы патчевой модели на входе как в live (кадры стримеров;
    у кадров наблюдателя входа «scene» нет — для них ответа нет)."""
    from lolvision.checkpoints import load_checkpoint
    from lolvision.patch_model import load_projection
    from train_not_game import Inputs, features
    model, _ = load_checkpoint(path, dev, "patch")
    if model.frame is None:
        raise SystemExit(f"{path}: нет головы «игра / не игра»")
    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    h0, store = load_projection()
    idx = [i for i, r in enumerate(rows) if r["source"] == "streams"]
    x = features(model, [rows[i] for i in idx], "scene", Inputs(layout, store), h0, dev,
                 path.parent.name.removesuffix("-notgame"))
    with torch.no_grad():
        p = torch.sigmoid(model.frame.to(dev)(torch.from_numpy(x).to(dev))).cpu().numpy()
    out = {i: float(v) for i, v in zip(idx, p)}
    out["threshold"] = float(model.frame_threshold)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patch-model", type=Path,
                    default=RUNS / "patches" / "cnn-split16-v2-notgame" / "model.pt",
                    help="чекпоинт с головой «игра / не игра» для итогового ответа")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with tee_stdout(args.out / "scene_kind_dino.log"):
        run(args)


def run(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    spec = sorted(p.name for p in (DATASET / "videos").iterdir() if p.is_dir())
    spec_te = [v for v in spec if v in split["coords_test"]]
    spec_tr = [v for v in spec if v not in spec_te]
    tr = stream_rows(split["train"]) + [r for v in spec_tr for r in spectator_rows(v)]
    te = stream_rows(split["test"]) + [r for v in spec_te for r in spectator_rows(v)]
    tr = [r for r in tr if r["full"] is not None]
    te = [r for r in te if r["full"] is not None]
    for name, rows in (("обучение", tr), ("проверка", te)):
        c = np.bincount([r["cls"] for r in rows], minlength=len(CLASSES))
        print(f"{name}: {len(rows)} кадров, " + ", ".join(f"{n} {v}" for n, v in zip(CLASSES, c)))

    vit = load_dino(dev=dev)
    x_tr, x_te = cached(vit, tr, dev), cached(vit, te, dev)
    y_tr = np.array([r["cls"] for r in tr])
    y_te = np.array([r["cls"] for r in te])
    head = fit(x_tr, y_tr, weights(y_tr, [r["video_id"] for r in tr]), dev, args.epochs)
    with torch.no_grad():
        p = torch.softmax(head(torch.from_numpy(x_te).to(dev)), 1).cpu().numpy()
    pred = p.argmax(1)
    p_game = patch_p_game(args.patch_model, te, dev)
    thr = p_game.pop("threshold")
    pg = np.array([p_game.get(i, 1.0) for i in range(len(te))])
    fused = np.where((pred == 0) & (pg < thr), CLASSES.index("other"), pred)
    out_kinds = set(split.get("out_of_scope_kinds", []))
    ins = np.array([r["kind"] not in out_kinds for r in te])
    c = confusion(y_te[ins], pred[ins])
    show("отложенные ролики, только DINO (без ARAM)", c)
    cf = confusion(y_te[ins], fused[ins])
    vids_te = np.array([r["video_id"] for r in te])
    cf["by_video"] = {v: confusion(y_te[ins & (vids_te == v)], fused[ins & (vids_te == v)])
                      for v in sorted(set(vids_te))}
    show("отложенные ролики, DINO + голова «игра / не игра» (без ARAM)", cf)
    c = {"dino": c, "fused": cf, "patch_threshold": thr}

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "scene_kind_dino", "arch": "dinov2_vits14", "input": [IN_W, IN_H],
                "classes": CLASSES, "state": head.state_dict(), "cin": x_tr.shape[1]},
               args.out / "head.pt")
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_id", "t_sec", "kind", "full", "truth", "pred", "fused", "p_game"]
                   + [f"p_{n}" for n in CLASSES])
        for r, t, q, fq, g, pp in zip(te, y_te, pred, fused, pg, p):
            w.writerow([r["video_id"], r["t_sec"], r["kind"], r["full"], CLASSES[t], CLASSES[q],
                        CLASSES[fq], f"{g:.4f}", *[f"{v:.3f}" for v in pp]])
    (args.out / "report.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")
    print(f"\n→ {args.out} ({time.perf_counter() - t0:.0f} с)")


if __name__ == "__main__":
    main()
