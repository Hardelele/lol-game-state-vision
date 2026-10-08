"""Обучение модели «кадр → координаты камеры» на плотном датасете.

Разметка здесь не человеческая, а снятая с миникарты (tools/build_coords.py),
поэтому примеров на два порядка больше, чем в ручном наборе. Сплит идёт по
роликам: кадры одного матча похожи, и смешивать их между обучением и
проверкой нельзя.

Кроме ошибки в координатах считается производная метрика: предсказанную
точку прогоняем через L-коридор верхней линии и сравниваем с РУЧНЫМИ
метками top/not_top. Это прямое сравнение с прежним классификатором,
который на отложенных роликах давал 0.634.

Пример:
    python tools/train_coords.py --train 58w57eJ5Qks ibUVbSX7ARU \
        --test olmTXkkUv58 zJvTSjEnKNE --epochs 12
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

from paths import COORDS_DATA, COORDS_RUNS, DATASET_VIDEOS, TRAIN_LOG
from runlog import tee_stdout
from coord_model import (CoordNet, GRID, soft_target, expected_point, peak_point,
                         spread, count_params)

MAP_UNITS = 14800
# L-коридор верхней линии, подобранный по ручной разметке (согласие 0.958).
TOP_CX, TOP_CY = 0.19, 0.22


def in_top(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    return (cx < TOP_CX) | (cy < TOP_CY)


def decode_all(rows: list[dict], root: Path, size, workers: int = 16
               ) -> torch.Tensor:
    """Распаковать все кадры один раз и сложить в память как uint8.

    Иначе каждую эпоху одни и те же JPEG декодируются заново, процессор не
    успевает подавать данные, и видеокарта простаивает: при замере загрузка
    GPU была 0-8%, а эпоха шла 12-15 с вместо ожидаемых двух-трёх. В uint8
    кадр занимает 230 КБ, то есть пять тысяч кадров — чуть больше гигабайта.
    """
    out = torch.empty((len(rows), 3, size[1], size[0]), dtype=torch.uint8)

    def one(i_row):
        i, r = i_row
        a = cv2.imread(str(root / r["scene"]), cv2.IMREAD_COLOR)
        a = cv2.cvtColor(cv2.resize(a, size, interpolation=cv2.INTER_LINEAR),
                         cv2.COLOR_BGR2RGB)
        out[i] = torch.from_numpy(a.transpose(2, 0, 1).copy())

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, enumerate(rows)))
    return out


def to_input(batch_u8: torch.Tensor) -> torch.Tensor:
    """uint8 → float в [-1, 1]. Делается на том устройстве, где лежит батч."""
    return batch_u8.float().div_(127.5).sub_(1.0)


def augment_gpu(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Яркость и контраст прямо на видеокарте, без возврата на процессор."""
    n = x.shape[0]
    dev = x.device
    c = torch.rand(n, 1, 1, 1, device=dev, generator=gen) * 0.3 + 0.85
    b = torch.rand(n, 1, 1, 1, device=dev, generator=gen) * 0.3 - 0.15
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m + b).clamp_(-1.0, 1.0)


class Scenes(Dataset):
    def __init__(self, rows: list[dict], root: Path, size: tuple[int, int],
                 augment: bool = False):
        self.rows, self.root, self.size, self.augment = rows, root, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r["scene"]) as im:
            a = np.asarray(im.convert("RGB").resize(self.size, Image.BILINEAR),
                           dtype=np.float32) / 255.0
        x = a.transpose(2, 0, 1) * 2 - 1
        if self.augment:
            # Только яркость и контраст. Сдвиг кадра сдвигает и настоящее
            # положение камеры, поэтому без пересчёта метки он её ломает.
            g = np.random.default_rng()
            m = x.mean()
            x = np.clip((x - m) * g.uniform(0.85, 1.15) + m + g.uniform(-0.15, 0.15),
                        -1, 1)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.tensor(float(r["cx"])), torch.tensor(float(r["cy"])))


def load_rows(ids: list[str], root: Path, min_q: float) -> list[dict]:
    rows = []
    for vid in ids:
        f = root / f"{vid}.csv"
        if not f.exists():
            raise SystemExit(f"нет манифеста {f} — сначала tools/build_coords.py")
        all_r = list(csv.DictReader(f.open(encoding="utf-8")))
        keep = [r for r in all_r if float(r["quality"]) >= min_q]
        print(f"  {vid}: {len(keep)} из {len(all_r)} кадров "
              f"(отброшено по качеству детекции {len(all_r) - len(keep)})")
        rows += keep
    return rows


def hand_labels(ids: list[str]) -> dict:
    """Ручные метки top/not_top по роликам: нужны только для сверки."""
    out = {}
    for vid in ids:
        p = DATASET_VIDEOS / vid / "frames.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r["label"] in ("top", "not_top"):
                out[(vid, int(r["t_sec"]))] = r["label"]
    return out


@torch.no_grad()
def evaluate(model, cache, rows, dev, hand, batch=96) -> dict:
    model.eval()
    exp, pk, spr = [], [], []
    for i in range(0, len(cache), batch):
        x = to_input(cache[i : i + batch].to(dev, non_blocking=True))
        lo = model(x).float()
        exp.append(expected_point(lo, GRID).cpu())
        pk.append(peak_point(lo, GRID).cpu())
        spr.append(spread(lo, GRID).cpu())
    exp = torch.cat(exp).numpy()
    pk = torch.cat(pk).numpy()
    spr = torch.cat(spr).numpy()
    true = np.array([[float(r["cx"]), float(r["cy"])] for r in rows])
    d_exp = np.linalg.norm(exp - true, axis=1)
    d_pk = np.linalg.norm(pk - true, axis=1)
    res = {"n": len(rows),
           "err_mean": float(d_exp.mean()), "err_median": float(np.median(d_exp)),
           "err_peak_median": float(np.median(d_pk)),
           "err_units": float(np.median(d_exp) * MAP_UNITS),
           "spread_median": float(np.median(spr))}

    # Сверка с ручной разметкой через L-коридор.
    idx = [i for i, r in enumerate(rows)
           if (r["video_id"], int(round(float(r["t_sec"])))) in hand]
    if idx:
        y = np.array([hand[(rows[i]["video_id"],
                            int(round(float(rows[i]["t_sec"]))))] == "top"
                      for i in idx])
        p = in_top(pk[idx, 0], pk[idx, 1])
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        rec = [float(((p == y) & (y == c)).sum() / max((y == c).sum(), 1))
               for c in (False, True)]
        res["vs_hand"] = {
            "n": len(idx), "accuracy": float((p == y).mean()),
            "balanced_accuracy": float(np.mean(rec)),
            "top_precision": tp / max(tp + fp, 1), "top_recall": tp / max(tp + fn, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--root", type=Path, default=COORDS_DATA)
    ap.add_argument("--size", default="384x200")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--region", default=None,
                    help="x0,y0,x1,y1 в долях карты. Разделить не по роликам, "
                         "а по месту: учить вне участка, проверять внутри. "
                         "Отвечает на вопрос, умеет ли модель определять место, "
                         "которого не видела, или просто помнит карту")
    ap.add_argument("--subsample", type=int, default=0,
                    help="оставить N обучающих кадров (0 — все). Нужно, чтобы "
                         "отделить пользу от числа матчей от пользы от числа кадров")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", choices=("gpu", "ram"), default="gpu",
                    help="где держать распакованные кадры")
    ap.add_argument("--out", type=Path, default=COORDS_RUNS / "heatmap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="остановиться по времени, не доходя до конца эпох "
                         "(0 — без ограничения). Чекпоинт и отчёт всё равно "
                         "сохраняются")
    args = ap.parse_args()
    with tee_stdout(args.out / TRAIN_LOG):
        run(args)


def run(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = tuple(int(v) for v in args.size.lower().split("x"))
    print(f"устройство: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    if args.region:
        x0, y0, x1, y1 = (float(v) for v in args.region.split(","))
        print(f"сплит по месту: проверка внутри участка "
              f"x {x0}-{x1}, y {y0}-{y1}; обучение вне его")
        pool = load_rows(sorted(set(args.train + args.test)), args.root,
                         args.min_quality)
        inside = lambda r: (x0 <= float(r["cx"]) <= x1
                            and y0 <= float(r["cy"]) <= y1)
        tr = [r for r in pool if not inside(r)]
        te = [r for r in pool if inside(r)]
        print(f"  вне участка {len(tr)} кадров, внутри {len(te)}")
    else:
        print("обучающие ролики:")
        tr = load_rows(args.train, args.root, args.min_quality)
        print("отложенные ролики:")
        te = load_rows(args.test, args.root, args.min_quality)
    if args.subsample and args.subsample < len(tr):
        # Прореживаем равномерно по времени внутри каждого ролика, а не
        # случайно: случайная выборка оставила бы почти соседние кадры.
        keep, per = [], args.subsample / len(args.train)
        for vid in args.train:
            sub = [r for r in tr if r["video_id"] == vid]
            step = max(1, round(len(sub) / per))
            keep += sub[::step][:round(per)]
        print(f"прорежено до {len(keep)} кадров из {len(tr)}")
        tr = keep

    hand = hand_labels(args.train + args.test)
    print(f"всего: обучение {len(tr)}, проверка {len(te)}; "
          f"ручных меток для сверки {len(hand)}")

    t_dec = time.perf_counter()
    cache_tr = decode_all(tr, args.root, size)
    cache_te = decode_all(te, args.root, size)
    mb = (cache_tr.nbytes + cache_te.nbytes) / 1e6
    where = "в памяти процессора"
    if args.cache == "gpu" and dev == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1e6
        if mb < free * 0.5:
            cache_tr, cache_te = cache_tr.to(dev), cache_te.to(dev)
            where = "в видеопамяти"
        else:
            where = f"в памяти процессора (в видеопамяти свободно {free:.0f} МБ)"
    print(f"кадры распакованы за {time.perf_counter() - t_dec:.0f} с, "
          f"{mb:.0f} МБ {where}")

    model = CoordNet().to(dev)
    print(f"параметров {count_params(model):,}, вход {size[0]}x{size[1]}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    # Тривиальный ориентир: всегда предсказывать среднюю точку обучения.
    mean_pt = np.array([[np.mean([float(r["cx"]) for r in tr]),
                         np.mean([float(r["cy"]) for r in tr])]])
    te_true = np.array([[float(r["cx"]), float(r["cy"])] for r in te])
    base = float(np.median(np.linalg.norm(te_true - mean_pt, axis=1)))
    print(f"базовый уровень (всегда средняя точка): медианная ошибка {base:.3f} "
          f"≈ {base * MAP_UNITS:.0f} игровых единиц\n")

    cx_tr = torch.tensor([float(r["cx"]) for r in tr], device=dev)
    cy_tr = torch.tensor([float(r["cy"]) for r in tr], device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    hist = []
    t_start = time.perf_counter()
    stopped_early = False
    for ep in range(args.epochs):
        if args.max_minutes and (time.perf_counter() - t_start) / 60 > args.max_minutes:
            print(f"остановка по времени: бюджет {args.max_minutes:.0f} мин "
                  f"исчерпан на эпохе {ep} из {args.epochs}")
            stopped_early = True
            break
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            sel = perm[i : i + args.batch]
            x = augment_gpu(to_input(cache_tr[sel].to(dev, non_blocking=True)), gen)
            tgt = soft_target(cx_tr[sel], cy_tr[sel], GRID, args.sigma)
            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                logp = F.log_softmax(model(x).float(), dim=1)
                loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(x)
            n += len(x)
        ev = evaluate(model, cache_te, te, dev, hand)
        row = {"epoch": ep + 1, "loss": tot / n, "sec": time.perf_counter() - t0, **ev}
        hist.append(row)
        vh = ev.get("vs_hand", {})
        print(f"эпоха {ep + 1:2d}  loss {row['loss']:.4f}  {row['sec']:.0f} с  |  "
              f"отложенные: медиана {ev['err_median']:.3f} "
              f"(~{ev['err_units']:.0f} ед.), среднее {ev['err_mean']:.3f}"
              + (f"  |  через L-коридор acc {vh['accuracy']:.3f} "
                 f"bal {vh['balanced_accuracy']:.3f}" if vh else ""), flush=True)

    tr_ev = evaluate(model, cache_tr, tr, dev, hand)
    print(f"\nна обучающих роликах: медиана {tr_ev['err_median']:.3f} "
          f"(~{tr_ev['err_units']:.0f} ед.)")
    final = hist[-1]
    print(f"на отложенных:        медиана {final['err_median']:.3f} "
          f"(~{final['err_units']:.0f} ед.) против базового {base:.3f} "
          f"(~{base * MAP_UNITS:.0f} ед.)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(), "head": model.head.state_dict(),
                "grid": GRID, "input_size": list(size), "crop": "dense",
                "trained_on": args.train, "map_units": MAP_UNITS},
               args.out / "model.pt")
    (args.out / "report.json").write_text(json.dumps({
        "train": args.train, "test": args.test, "input_size": list(size),
        "params": count_params(model), "baseline_median": base,
        "train_eval": tr_ev, "holdout": final, "history": hist,
        "stopped_early": stopped_early, "epochs_done": len(hist),
        "hyperparams": {k: str(v) for k, v in vars(args).items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"чекпоинт и отчёт → {args.out}")


if __name__ == "__main__":
    main()
