"""Голова «игра / не игра» поверх замороженной патчевой модели.

Зачем. Патчевая модель (patch_model.py) отвечает «где камера» на любой
кадр, даже если на экране вебкамера, клиент с выбором чемпионов или другая
игра: голова «сцена» на панели во весь кадр даёт P(сцена) = 0.76, а голоса
патчей всё равно сводятся в какую-то точку. Нужен ответ «тут не лига».

Как. Энкодер и голова патча заморожены, поэтому координаты остаются ровно
такими же, как у исходного чекпоинта, а обучение стоит секунды. Кадр
проходит обычный путь (маска раскладки → плотная обрезка → вид сверху →
патчи), из ответов патчей собираются признаки кадра (frame_features:
средний эмбеддинг патчей, средняя P(сцена), уверенность патчей в своём
месте, доля веса в согласии), и маленькая голова FrameHead учится по ним
отличать игру от не игры.

Данные (dataset/not_game/split.json):

* не игра и игра чужим HUD — ролики стримеров из dataset/not_game/<id>/
  (кадры собирает build_not_game.py, метки — интервалы labels.csv);
* игра наблюдателем — data/coords: обучающие ролики исходной модели
  (каждый N-й кадр) и её же 4 отложенных ролика для проверки (все кадры).

Каждый кадр берётся в нескольких вариантах входа: как в live (маска и
обрезка), без маски (только для роликов, где сохранён весь кадр) и
случайный кусок 30-70% площади. Порог подбирается по ответам вне выборки
(перекрёстная проверка по роликам на обучающей части): не больше 1%
ложных отбрасываний игровых кадров. Отложенные ролики в выбор порога не
входят. Подклассы из out_of_scope_kinds (ARAM — другая карта) в общие
цифры не входят и видны только в разбивке по подклассам.

Итог — новый чекпоинт: веса исходной модели без изменений плюс поле
frame_head (load_patchnet подхватывает его сам, старый код его не читает).

Пример:
    python tools/train_not_game.py runs/patches/cnn-split16/model.pt \
        --out runs/patches/cnn-split16-notgame
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from paths import COORDS_DATA, RUNS, DATASET
from runlog import tee_stdout
from mask_frames import crop_box
from patch_model import (load_patchnet, load_projection, scene_from_box,
                         infer_images, FrameHead, FRAME_SCALARS)
from build_not_game import NOT_GAME_DATA, NOT_GAME_LABELS, load_labels, label_at

SPLIT = NOT_GAME_LABELS / "split.json"
LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
CACHE = RUNS / "not_game" / "cache"
VARIANTS = ("scene", "nomask", "piece")
FULL = (1920, 1080)


# ------------------------------------------------------------- выборка кадров

def not_game_rows(vid: str) -> list[dict]:
    """Кадры ролика стримера с метками; без метки и «skip» — не берутся."""
    labels = load_labels(vid)
    out = []
    for r in csv.DictReader((NOT_GAME_DATA / f"{vid}.csv").open(encoding="utf-8")):
        t = float(r["t_sec"])
        lab, kind = label_at(labels, t)
        if lab not in ("game", "not_game"):
            continue
        out.append({"video_id": vid, "t_sec": t, "scene": NOT_GAME_DATA / r["scene"],
                    "full": NOT_GAME_DATA / r["full"], "y": int(lab == "game"),
                    "kind": kind, "source": "streams"})
    return out


def coords_rows(vid: str, step: int, min_quality: float) -> list[dict]:
    """Игровые кадры наблюдателя из data/coords (полного кадра там нет)."""
    rows = [r for r in csv.DictReader((COORDS_DATA / f"{vid}.csv").open(encoding="utf-8"))
            if float(r["quality"]) >= min_quality]
    return [{"video_id": vid, "t_sec": float(r["t_sec"]), "scene": COORDS_DATA / r["scene"],
             "full": None, "y": 1, "kind": "lol_game_spectator", "source": "coords"}
            for r in rows[::step]]


# ------------------------------------------------------------- признаки

class Inputs:
    """Варианты входа одного кадра: картинка и её геометрия A (см. scene_from_box)."""

    def __init__(self, layout: dict, store: tuple[int, int]):
        self.store = store
        self.crop = crop_box(FULL, layout, "dense")

    def make(self, key: str, r: dict, rng: np.random.Generator):
        if key == "scene":
            return cv2.cvtColor(cv2.imread(str(r["scene"])), cv2.COLOR_BGR2RGB), np.eye(3)
        if key == "nomask":
            full = cv2.cvtColor(cv2.imread(str(r["full"])), cv2.COLOR_BGR2RGB)
            f = FULL[0] / full.shape[1]
            x0, y0, x1, y1 = (int(round(v / f)) for v in self.crop)
            img = full[y0:y1, x0:x1]
            box = (x0 * f, y0 * f, x1 * f, y1 * f)
            return img, scene_from_box(box, (img.shape[1], img.shape[0]), self.crop, self.store)
        if key == "piece":
            img = cv2.cvtColor(cv2.imread(str(r["scene"])), cv2.COLOR_BGR2RGB)
            sh, sw = img.shape[:2]
            area = rng.uniform(0.30, 0.70) * sw * sh
            ar = math.exp(rng.uniform(math.log(0.5), math.log(3.0)))
            cw = int(min(sw, math.sqrt(area * ar)))
            ch = int(min(sh, area / cw))
            x0, y0 = int(rng.integers(0, sw - cw + 1)), int(rng.integers(0, sh - ch + 1))
            return (img[y0:y0 + ch, x0:x0 + cw],
                    np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float))
        raise KeyError(key)


def features(model, rows: list[dict], key: str, inputs: Inputs, h0, dev: str,
             cache_tag: str) -> np.ndarray:
    """Признаки кадров (N, D + 5) для варианта входа; кэш по ролику."""
    out = np.zeros((len(rows), model.frame_dim), np.float32)
    by_vid: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_vid.setdefault(r["video_id"], []).append(i)
    for vid, idx in by_vid.items():
        cpath = CACHE / cache_tag / f"{vid}-{key}.npz"
        have = {}
        if cpath.exists():
            z = np.load(cpath)
            have = dict(zip(z["t"].round(3).tolist(), z["f"]))
        need = [i for i in idx if round(rows[i]["t_sec"], 3) not in have]
        for j in range(0, len(need), 256):
            part = need[j:j + 256]
            imgs, mats = [], []
            for i in part:
                rng = np.random.default_rng(int(rows[i]["t_sec"] * 1000) + 7)
                im, a = inputs.make(key, rows[i], rng)
                imgs.append(im)
                mats.append(a)
            _, _, pt = infer_images(model, imgs, mats, [None] * len(imgs), h0, dev, batch=32)
            for i, p in zip(part, pt):
                have[round(rows[i]["t_sec"], 3)] = p["frame_feat"]
        if need:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            ts = np.array(sorted(have))
            np.savez(cpath, t=ts, f=np.stack([have[t] for t in ts.round(3).tolist()]))
        for i in idx:
            out[i] = have[round(rows[i]["t_sec"], 3)]
    return out


# ------------------------------------------------------------- голова

def sample_weights(y: np.ndarray, groups: list[str]) -> np.ndarray:
    """Вес примера: ролик с тысячей похожих кадров не должен перевешивать
    ролик с сотней — вес ∝ 1/√(кадров этой метки в ролике); затем классы
    уравниваются по сумме веса."""
    key = [f"{g}|{v}" for g, v in zip(groups, y)]
    cnt: dict[str, int] = {}
    for k in key:
        cnt[k] = cnt.get(k, 0) + 1
    w = np.array([1 / math.sqrt(cnt[k]) for k in key])
    for c in (0, 1):
        w[y == c] /= w[y == c].sum()
    return w * len(w) / 2


def fit_head(x: np.ndarray, y: np.ndarray, w: np.ndarray, dev: str,
             epochs: int = 400, seed: int = 0) -> FrameHead:
    torch.manual_seed(seed)
    head = FrameHead(x.shape[1]).to(dev)
    xt = torch.from_numpy(x).to(dev)
    head.mu.copy_(xt.mean(0))
    head.sd.copy_(xt.std(0).clamp(min=1e-4))
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-2)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for j in range(0, len(perm), 1024):
            k = perm[j:j + 1024]
            # Шум во входе — дешёвая защита от того, чтобы голова не
            # запоминала мелкие особенности нескольких роликов.
            xb = xt[k] + torch.randn_like(xt[k]) * 0.1 * head.sd
            loss = (F.binary_cross_entropy_with_logits(head(xb), yt[k], reduction="none")
                    * wt[k]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def predict(head: FrameHead, x: np.ndarray, dev: str) -> np.ndarray:
    return torch.sigmoid(head(torch.from_numpy(x).to(dev))).cpu().numpy()


def pick_threshold(p: np.ndarray, y: np.ndarray, max_frr: float = 0.01) -> float:
    """Порог P(игра): кадр ниже порога — «не игра».

    Ограничение — доля ложно отброшенных игровых кадров не больше max_frr;
    внутри него берётся порог с лучшей сбалансированной точностью.
    """
    cap = float(np.quantile(p[y == 1], max_frr))
    best, bt = -1.0, cap
    for t in np.linspace(0.01, 0.99, 197):
        if t > cap:
            break
        ba = ((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2
        if ba > best:
            best, bt = ba, float(t)
    return bt


def binary(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    """«Не игра» — положительный класс: точность и полнота отбрасывания."""
    rej = p < t
    neg = y == 0
    tp = int((rej & neg).sum())
    out = {"n": int(len(y)), "n_game": int((~neg).sum()), "n_not_game": int(neg.sum()),
           "frr_game": float(rej[~neg].mean()) if (~neg).any() else None,
           "recall_not_game": float(rej[neg].mean()) if neg.any() else None,
           "precision_not_game": float(tp / rej.sum()) if rej.sum() else None}
    return out


def smooth3(p: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Медиана по трём соседним кадрам одного ролика (в live — задержка 1 кадр)."""
    out = p.copy()
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["video_id"], []).append(i)
    for idx in by.values():
        idx = sorted(idx, key=lambda i: rows[i]["t_sec"])
        v = p[idx]
        if len(v) >= 3:
            m = np.median(np.stack([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]]), 0)
            out[idx] = m
    return out


def boundary_dist(rows: list[dict]) -> np.ndarray:
    """Секунд до ближайшей смены метки игра/не игра в разметке ролика."""
    d = np.full(len(rows), np.inf)
    cache: dict[str, list[float]] = {}
    for i, r in enumerate(rows):
        if r["source"] != "streams":
            continue
        vid = r["video_id"]
        if vid not in cache:
            lab = load_labels(vid)
            b = []
            for a, c in zip(lab, lab[1:]):
                if a["label"] != c["label"]:
                    b.append(c["start_sec"])
            cache[vid] = b
        if cache[vid]:
            d[i] = min(abs(r["t_sec"] - b) for b in cache[vid])
    return d


# ------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="чекпоинт патчевой модели")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--coords-train-step", type=int, default=10)
    ap.add_argument("--coords-test-step", type=int, default=1)
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--max-frr", type=float, default=0.01,
                    help="доля игровых кадров, которую можно ложно отбросить (на выборе порога)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with tee_stdout(args.out / "train_not_game.log"):
        run(args)


def run(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    model = load_patchnet(args.model, dev)
    model.frame_dim = model.head.embed.out_channels + len(FRAME_SCALARS)
    base = torch.load(args.model, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    coords_train = split.get("coords_train") or base.get("trained_on", [])
    coords_test = split["coords_test"]
    out_kinds = set(split.get("out_of_scope_kinds", []))
    seen = set(coords_test) & set(base.get("trained_on", []))
    if seen:
        raise SystemExit(f"отложенные ролики data/coords видела исходная модель: {seen}")
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    h0, store = load_projection()
    inputs = Inputs(layout, store)
    tag = f"{args.model.parent.name}"

    def gather(vids_streams, vids_coords, step):
        rows = []
        for v in vids_streams:
            rows += not_game_rows(v)
        for v in vids_coords:
            rows += coords_rows(v, step, args.min_quality)
        return rows

    tr = gather(split["train"], coords_train, args.coords_train_step)
    te = gather(split["test"], coords_test, args.coords_test_step)
    for name, rows in (("обучение", tr), ("проверка", te)):
        y = np.array([r["y"] for r in rows])
        print(f"{name}: {len(rows)} кадров, игра {int(y.sum())}, не игра {int((y == 0).sum())}, "
              f"роликов {len({r['video_id'] for r in rows})}")

    def feats(rows, keys):
        """Признаки по вариантам: {вариант: (индексы строк, матрица)}."""
        out = {}
        for k in keys:
            idx = [i for i, r in enumerate(rows) if k != "nomask" or r["full"] is not None]
            sub = [rows[i] for i in idx]
            t0 = time.perf_counter()
            out[k] = (np.array(idx), features(model, sub, k, inputs, h0, dev, tag))
            print(f"  признаки «{k}»: {len(sub)} кадров, {time.perf_counter() - t0:.0f} с",
                  flush=True)
        return out

    print("признаки обучающей части:")
    ftr = feats(tr, VARIANTS)
    print("признаки отложенной части:")
    fte = feats(te, VARIANTS)

    # Обучающая матрица: все варианты всех кадров.
    xs, ys, gs = [], [], []
    for k, (idx, x) in ftr.items():
        xs.append(x)
        ys.append(np.array([tr[i]["y"] for i in idx]))
        gs += [tr[i]["video_id"] for i in idx]
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    w_tr = sample_weights(y_tr, gs)

    # Ответы вне выборки: перекрёстная проверка по роликам → порог.
    vids = sorted(set(gs))
    fold_of = {v: i % args.folds for i, v in enumerate(vids)}
    g_fold = np.array([fold_of[g] for g in gs])
    oof = np.zeros(len(y_tr))
    for f in range(args.folds):
        m = g_fold == f
        h = fit_head(x_tr[~m], y_tr[~m], sample_weights(y_tr[~m], [g for g, k in zip(gs, m) if not k]),
                     dev, args.epochs, seed=f)
        oof[m] = predict(h, x_tr[m], dev)
    thr = pick_threshold(oof, y_tr, args.max_frr)
    cv = binary(oof, y_tr, thr)
    print(f"\nперекрёстная проверка по роликам ({args.folds} частей): порог P(игра) = {thr:.3f}; "
          f"ложно отброшено игры {cv['frr_game'] * 100:.2f}%, найдено не игры "
          f"{cv['recall_not_game'] * 100:.1f}%, точность «не игра» "
          f"{cv['precision_not_game'] * 100:.1f}%")

    head = fit_head(x_tr, y_tr, w_tr, dev, args.epochs, seed=0)
    model.frame = head.cpu()
    model.frame_threshold = thr

    # Проверка на отложенных роликах: по вариантам входа и по источникам.
    report = {"model": str(args.model), "threshold": thr, "cv": cv,
              "split": split, "coords_train": coords_train, "variants": {}}
    preds = {}
    print(f"\nотложенные ролики, порог {thr:.3f} (кадр с P(игра) ниже — «не игра»):")
    for k, (idx, x) in fte.items():
        p = predict(head.to(dev), x, dev)
        rows = [te[i] for i in idx]
        y = np.array([r["y"] for r in rows])
        preds[k] = (rows, p)
        # Кадры вне задачи (ARAM — другая карта) — только в разбивке по подклассам.
        ins = np.array([r["kind"] not in out_kinds for r in rows])
        ps = smooth3(p, rows)
        res = {"all": binary(p[ins], y[ins], thr), "smooth3": binary(ps[ins], y[ins], thr),
               "by_video": {}, "by_kind": {}}
        for v in sorted({r["video_id"] for r in rows}):
            m = np.array([r["video_id"] == v for r in rows]) & ins
            if m.any():
                res["by_video"][v] = binary(p[m], y[m], thr)
        for kd in sorted({r["kind"] for r in rows}):
            m = np.array([r["kind"] == kd for r in rows])
            res["by_kind"][kd] = binary(p[m], y[m], thr)
        bd = boundary_dist(rows)
        for name, m in (("near_boundary_3s", bd <= 3), ("far_from_boundary", (bd > 3) & np.isfinite(bd))):
            m = m & ins
            if m.any():
                res[name] = binary(p[m], y[m], thr)
                res[name + "_smooth3"] = binary(ps[m], y[m], thr)
        report["variants"][k] = res
        a = res["all"]
        print(f"  {k:7s} кадров {a['n']:5d}: ложно отброшено игры "
              f"{a['frr_game'] * 100:5.2f}%, найдено не игры "
              f"{(a['recall_not_game'] or 0) * 100:5.1f}%, точность «не игра» "
              f"{(a['precision_not_game'] or 0) * 100:5.1f}%")
        for v, b in res["by_video"].items():
            s = (f"ложно отброшено {b['frr_game'] * 100:5.2f}% из {b['n_game']}"
                 if b["n_game"] else " " * 24)
            s2 = (f"найдено не игры {b['recall_not_game'] * 100:5.1f}% из {b['n_not_game']}"
                  if b["n_not_game"] else "")
            print(f"      {v:12s} {s}  {s2}")
        for kd, b in res["by_kind"].items():
            share = b["frr_game"] if b["n_game"] else b["recall_not_game"]
            what = "ложно отброшено" if b["n_game"] else "найдено не игры"
            print(f"      [{kd}] {what} {share * 100:5.1f}% из {b['n']}"
                  + ("  (вне задачи)" if kd in out_kinds else ""))
        s3 = res["smooth3"]
        print(f"      медиана по 3 кадрам: ложно отброшено {s3['frr_game'] * 100:.2f}%, "
              f"найдено не игры {(s3['recall_not_game'] or 0) * 100:.1f}%")
        for name in ("near_boundary_3s", "far_from_boundary"):
            if name in res:
                b = res[name]
                print(f"      {name}: кадров {b['n']}, ложно отброшено "
                      f"{(b['frr_game'] or 0) * 100:.1f}%, найдено не игры "
                      f"{(b['recall_not_game'] or 0) * 100:.1f}%")

    # Простые пороги без обучения — для сравнения в отчёте.
    rows, _ = preds["scene"]
    idx, x = fte["scene"]
    ins = np.array([r["kind"] not in out_kinds for r in rows])
    rows = [r for r, k in zip(rows, ins) if k]
    x = x[ins]
    y = np.array([r["y"] for r in rows])
    d = model.head.embed.out_channels
    base_cmp = {}
    for j, name in enumerate(FRAME_SCALARS):
        v = x[:, d + j]
        base_cmp[name] = {"game_median": float(np.median(v[y == 1])),
                          "not_game_median": float(np.median(v[y == 0])),
                          "auc": auc(v, y)}
    base_cmp["frame_head"] = {"auc": auc(preds["scene"][1][ins], y)}
    report["single_feature_auc_scene"] = base_cmp
    print("\nразделимость по одному признаку (вариант «scene», AUC):")
    for k, v in base_cmp.items():
        print(f"  {k:12s} AUC {v['auc']:.3f}"
              + (f"  медиана: игра {v['game_median']:.3f}, не игра {v['not_game_median']:.3f}"
                 if "game_median" in v else ""))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "not_game_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "video_id", "t_sec", "y", "kind", "p_game"])
        for k, (rows, p) in preds.items():
            for r, q in zip(rows, p):
                wr.writerow([k, r["video_id"], r["t_sec"], r["y"], r["kind"], f"{q:.4f}"])
    model.frame.cpu()
    ck = model.state() | {k: v for k, v in base.items()
                          if k not in ("state", "frame_head", "kind", "encoder")}
    ck["frame_head"]["trained_on"] = {"streams": split["train"], "coords": coords_train}
    torch.save(ck, args.out / "model.pt")
    report["minutes"] = (time.perf_counter() - t_start) / 60
    (args.out / "not_game_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nчекпоинт (исходные веса + frame_head) и отчёт → {args.out}")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """AUC «игра выше не игры» через ранги."""
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    main()
