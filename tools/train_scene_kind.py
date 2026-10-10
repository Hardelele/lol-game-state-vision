"""Тип сцены поверх замороженной патчевой модели: игра, Tab, магазин,
клиент, загрузка, не League.

Зачем. Голова «игра / не игра» (train_not_game.py) отвечает только «есть
карта или нет». Нужно знать, что именно на экране: клиент League, экран
загрузки, открытая таблица счёта (Tab) или магазин поверх игры. На Tab и
в магазине карта почти закрыта, модель камеры там ошибается чаще.

Как. Те же признаки кадра (frame_features), что у головы «игра / не
игра», и тот же кэш. Классы:

* game — игровая сцена (label game в labels.csv);
* tab, shop — игровая сцена с открытой таблицей счёта / магазином:
  интервалы в dataset/not_game/<id>/overlays.csv поверх labels.csv;
* client, loading — kind lol_client / lol_loading;
* other — всё прочее «не игра» (вебкамера, заставки, другие игры…).

Варианты входа — как в live и без маски; случайный кусок кадра не
берётся: он может вырезать таблицу или магазин и испортить метку.
Итог — новый чекпоинт: исходный плюс поле scene_head.

Пример:
    python tools/train_scene_kind.py runs/patches/cnn-split16-v2-notgame/model.pt \\
        --out runs/patches/cnn-split16-v2-scene
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.checkpoints import load_checkpoint
from lolvision.paths import DEFAULT_LAYOUT, NOT_GAME_LABELS
from lolvision.patch_model import SceneHead, load_projection
from runlog import tee_stdout
from train_not_game import SPLIT, Inputs, coords_rows, features, not_game_rows

CLASSES = ["game", "tab", "shop", "client", "loading", "other"]
VARIANTS = ("scene", "nomask")
KIND_CLASS = {"lol_client": "client", "lol_loading": "loading"}


def load_overlays(vid: str) -> list[tuple[float, float, str]]:
    p = NOT_GAME_LABELS / vid / "overlays.csv"
    if not p.exists():
        return []
    return [(float(r["start_sec"]), float(r["end_sec"]), r["kind"])
            for r in csv.DictReader(p.open(encoding="utf-8"))]


def scene_class(r: dict, overlays: list[tuple[float, float, str]]) -> str:
    if r["y"] == 1:
        for a, b, k in overlays:
            if a <= r["t_sec"] < b:
                return k
        return "game"
    return KIND_CLASS.get(r["kind"], "other")


def rows_for(vids: list[str], coords: list[str], step: int, min_q: float) -> list[dict]:
    rows = []
    for v in vids:
        ov = load_overlays(v)
        for r in not_game_rows(v):
            r["cls"] = CLASSES.index(scene_class(r, ov))
            rows.append(r)
    for v in coords:
        for r in coords_rows(v, step, min_q):
            r["cls"] = 0
            rows.append(r)
    return rows


def weights(y: np.ndarray, groups: list[str]) -> np.ndarray:
    """Вес ∝ 1/√(кадров этого класса в ролике); классы уравнены по сумме."""
    key = [f"{g}|{c}" for g, c in zip(groups, y)]
    cnt: dict[str, int] = {}
    for k in key:
        cnt[k] = cnt.get(k, 0) + 1
    w = np.array([1 / math.sqrt(cnt[k]) for k in key])
    present = [c for c in range(len(CLASSES)) if (y == c).any()]
    for c in present:
        w[y == c] /= w[y == c].sum()
    return w * len(w) / len(present)


def fit(x, y, w, dev, epochs=400, seed=0) -> SceneHead:
    torch.manual_seed(seed)
    head = SceneHead(x.shape[1], CLASSES).to(dev)
    xt = torch.from_numpy(x).to(dev)
    head.mu.copy_(xt.mean(0))
    head.sd.copy_(xt.std(0).clamp(min=1e-4))
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for j in range(0, len(perm), 1024):
            k = perm[j:j + 1024]
            xb = xt[k] + torch.randn_like(xt[k]) * 0.1 * head.sd
            loss = (F.cross_entropy(head(xb), yt[k], reduction="none") * wt[k]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def predict(head: SceneHead, x: np.ndarray, dev: str) -> np.ndarray:
    return torch.softmax(head.to(dev)(torch.from_numpy(x).to(dev)), 1).cpu().numpy()


def confusion(y: np.ndarray, p: np.ndarray) -> dict:
    k = len(CLASSES)
    m = np.zeros((k, k), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    per = {}
    for c in range(k):
        n = m[c].sum()
        got = m[:, c].sum()
        per[CLASSES[c]] = {"n": int(n),
                           "recall": round(m[c, c] / n, 4) if n else None,
                           "precision": round(m[c, c] / got, 4) if got else None}
    return {"matrix": m.tolist(), "per_class": per,
            "accuracy": round(float(np.trace(m) / m.sum()), 4) if m.sum() else None}


def show(name: str, c: dict) -> None:
    print(f"\n{name}: точность {c['accuracy']}")
    print("истина \\ ответ " + " ".join(f"{n:>8s}" for n in CLASSES))
    for n, row in zip(CLASSES, c["matrix"]):
        print(f"{n:>14s} " + " ".join(f"{v:8d}" for v in row))
    for n, d in c["per_class"].items():
        if d["n"]:
            print(f"  {n:8s} n={d['n']:5d} полнота {d['recall']}  точность {d['precision']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="чекпоинт патчевой модели")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--coords-train-step", type=int, default=10)
    ap.add_argument("--coords-test-step", type=int, default=5)
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--cache-tag", help="кэш признаков (по умолчанию — папка модели без -notgame:"
                    " энкодер у них один)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with tee_stdout(args.out / "train_scene_kind.log"):
        run(args)


def run(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model, base = load_checkpoint(args.model, dev, "patch")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    coords_train = split.get("coords_train") or base.get("trained_on", [])
    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    h0, store = load_projection()
    inputs = Inputs(layout, store)
    tag = args.cache_tag or args.model.parent.name.removesuffix("-notgame")

    tr = rows_for(split["train"], coords_train, args.coords_train_step, args.min_quality)
    te = rows_for(split["test"], split["coords_test"], args.coords_test_step, args.min_quality)
    for name, rows in (("обучение", tr), ("проверка", te)):
        c = np.bincount([r["cls"] for r in rows], minlength=len(CLASSES))
        print(f"{name}: {len(rows)} кадров, " + ", ".join(f"{n} {v}" for n, v in zip(CLASSES, c)))

    def feats(rows):
        out = {}
        for k in VARIANTS:
            idx = [i for i, r in enumerate(rows) if k != "nomask" or r["full"] is not None]
            out[k] = (np.array(idx), features(model, [rows[i] for i in idx], k, inputs, h0,
                                              dev, tag))
        return out

    ftr, fte = feats(tr), feats(te)
    xs, ys, gs = [], [], []
    for idx, x in ftr.values():
        xs.append(x)
        ys.append(np.array([tr[i]["cls"] for i in idx]))
        gs += [tr[i]["video_id"] for i in idx]
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    head = fit(x_tr, y_tr, weights(y_tr, gs), dev, args.epochs)

    report = {"model": str(args.model), "classes": CLASSES, "split": split, "variants": {}}
    for k, (idx, x) in fte.items():
        rows = [te[i] for i in idx]
        y = np.array([r["cls"] for r in rows])
        p = predict(head, x, dev).argmax(1)
        c = confusion(y, p)
        c["by_video"] = {v: confusion(y[m], p[m]) for v in sorted({r["video_id"] for r in rows})
                         for m in [np.array([r["video_id"] == v for r in rows])]}
        report["variants"][k] = c
        show(f"отложенные ролики, вход «{k}»", c)

    model.scene_kind = head.cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    ck = dict(base)
    ck.update(model.state())
    ck["scene_kind_trained"] = {"split": str(args.split), "classes": CLASSES,
                                "sec": round(time.perf_counter() - t0, 1)}
    torch.save(ck, args.out / "model.pt")
    (args.out / "scene_kind.json").write_text(json.dumps(report, ensure_ascii=False, indent=1)
                                              + "\n", encoding="utf-8")
    print(f"\n→ {args.out / 'model.pt'} ({time.perf_counter() - t0:.0f} с)")


if __name__ == "__main__":
    main()
