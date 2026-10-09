"""Обучение патчевой модели (patch_model.py): каждый кусок сцены — точка карты.

Зачем. Нужна модель, которая узнаёт место по самой земле, а не по тому,
где на экране лежит признак и как выглядит маска. Поэтому разметка не на
кадр, а на патч: точка патча на карте = центр камеры (с миникарты) +
гомография(центр патча). Метка патча не зависит от того, какой кусок кадра
вырезан и где он оказался на входе, — значит, можно резать кадр как угодно,
не ломая разметку.

Как готовится пример (на процессоре, в загрузчике):

* хранимая сцена 720×372 (уже замаскированная) + маска «здесь сцена»
  той же раскладки;
* 0-4 случайных прямоугольника поверх: чёрные или синтетический UI
  (панели, текст, полоски, значки — см. synth_ui). Настоящий HUD
  трансляции сюда намеренно не берётся: его полосы и есть то, чем
  probe_hud.py проверяет устойчивость, и учить на них — подглядывать;
* случайный вырез любых пропорций (0.4-4) и площади (15-80% кадра;
  в 30% примеров кадр целиком), иногда уменьшенный в 1-2.5 раза —
  имитация другого разрешения;
* перевод в вид сверху с UNITS_PER_PX ± 12% (погрешность калибровки и
  зума) в случайное место холста фиксированного размера.

На видеокарте — яркость, контраст и баланс каналов, затем потеря:
мягкая кросс-энтропия по сетке карты для патчей, которые больше чем на 60%
закрыты сценой, и BCE «это сцена» для всех патчей.

Пример:
    python tools/train_patches.py --encoder cnn --out runs/patches/cnn \
        --train 0Jijr5p5gAg ... --test 58w57eJ5Qks olmTXkkUv58 --max-minutes 60
"""

from __future__ import annotations

import argparse
import json
import math
import string
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from paths import COORDS_DATA, RUNS, TRAIN_LOG, DATASET
from runlog import tee_stdout
from mask_frames import crop_box, keep_grid
from train_coords import load_rows
from patch_model import (PatchNet, MAP_UNITS, UNITS_PER_PX, load_projection,
                         to_world, footprint, cell_offsets, soft_target_sep,
                         infer_images, count_params, decode_points)

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
PATCH_RUNS = RUNS / "patches"


def stored_keep(layout: dict, full=(1920, 1080), store=(720, 372)) -> np.ndarray:
    """Маска «здесь сцена» в пикселях хранимой сцены (bool, H×W)."""
    k = keep_grid(full, layout)
    x0, y0, x1, y1 = crop_box(full, layout, "dense")
    k = k[y0:y1, x0:x1].astype(np.uint8) * 255
    return cv2.resize(k, store, interpolation=cv2.INTER_NEAREST) > 127


# ------------------------------------------------------------- синтетический UI

_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
          cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX,
          cv2.FONT_HERSHEY_TRIPLEX)
_CHARS = string.ascii_letters + string.digits + "  /:%+-"


def synth_ui(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная «панель интерфейса» (h, w, 3) uint8.

    Не копия чьего-то HUD, а его общие черты: тёмная подложка с градиентом,
    рамка, строки текста, полоски здоровья/маны, квадратные значки. Этого
    достаточно, чтобы модель училась отличать нарисованное поверх игры от
    самой игры, не запоминая конкретную трансляцию.
    """
    w, h = max(int(w), 2), max(int(h), 2)
    c0 = rng.integers(0, 70, 3)
    c1 = np.clip(c0 + rng.integers(-30, 60, 3), 0, 255)
    t = np.linspace(0, 1, h)[:, None, None]
    img = (c0[None, None] * (1 - t) + c1[None, None] * t).repeat(w, 1)
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(2, 14), (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    gold = tuple(int(v) for v in rng.integers(120, 230, 3))
    if rng.random() < 0.7:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), gold, int(rng.integers(1, 4)))
    for _ in range(int(rng.integers(0, 6))):            # значки
        s = int(rng.integers(6, max(7, min(h, w) // 2 + 1)))
        x, y = int(rng.integers(0, max(1, w - s))), int(rng.integers(0, max(1, h - s)))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + s), col, -1)
            cv2.rectangle(img, (x, y), (x + s, y + s), gold, 1)
        else:
            cv2.circle(img, (x + s // 2, y + s // 2), s // 2, col, -1)
    for _ in range(int(rng.integers(0, 4))):            # полоски
        bw = int(rng.integers(max(2, w // 6), max(3, w)))
        bh = int(rng.integers(2, max(3, h // 8 + 3)))
        x, y = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        col = [(40, 200, 60), (200, 40, 40), (50, 110, 230), (230, 200, 60)][
            int(rng.integers(0, 4))]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (10, 10, 10), -1)
        cv2.rectangle(img, (x, y), (x + int(bw * rng.uniform(0.1, 1)), y + bh), col, -1)
    for _ in range(int(rng.integers(0, 7))):            # текст
        n = int(rng.integers(2, 16))
        txt = "".join(rng.choice(list(_CHARS), n))
        sc = float(rng.uniform(0.3, 1.3))
        col = tuple(int(v) for v in (rng.integers(170, 256, 3) if rng.random() < 0.7
                                     else rng.integers(0, 256, 3)))
        org = (int(rng.integers(0, max(1, w - 10))), int(rng.integers(8, max(9, h))))
        cv2.putText(img, txt, org, _FONTS[int(rng.integers(0, len(_FONTS)))], sc, col,
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    return img


# ------------------------------------------------------------- набор данных

class PatchSamples(Dataset):
    """Пример = холст вида сверху + маска «здесь сцена» + геометрия.

    Метки патчей здесь не строятся: точка патча на карте однозначно
    восстанавливается на видеокарте из (центр камеры, q0, ед./px).
    """

    def __init__(self, rows, root: Path, h0, keep, canvas, train: bool):
        self.rows, self.root, self.h0, self.keep = rows, root, h0, keep
        self.canvas, self.train = canvas, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng()
        img = cv2.cvtColor(cv2.imread(str(self.root / r["scene"])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        open_ = self.keep.astype(np.uint8) * 255
        units = UNITS_PER_PX
        a = np.eye(3)
        if self.train:
            img = img.copy()
            for _ in range(int(rng.integers(0, 5))):        # заслонки
                bw = int(rng.uniform(0.05, 0.4) * W)
                bh = int(rng.uniform(0.04, 0.35) * H)
                x, y = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
                if rng.random() < 0.45:
                    img[y:y + bh, x:x + bw] = 0
                else:
                    ui = synth_ui(bw, bh, rng).astype(np.float32)
                    al = rng.uniform(0.75, 1.0)
                    reg = img[y:y + bh, x:x + bw].astype(np.float32)
                    img[y:y + bh, x:x + bw] = (al * ui + (1 - al) * reg).astype(np.uint8)
                open_[y:y + bh, x:x + bw] = 0
            if rng.random() > 0.3:                           # вырез
                area = rng.uniform(0.15, 0.8) * W * H
                ar = math.exp(rng.uniform(math.log(0.4), math.log(4.0)))
                cw = int(min(W, max(16, math.sqrt(area * ar))))
                ch = int(min(H, max(16, area / max(cw, 1))))
                x0, y0 = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
                img = img[y0:y0 + ch, x0:x0 + cw]
                open_ = open_[y0:y0 + ch, x0:x0 + cw]
                a = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
            if rng.random() < 0.3:                           # другое разрешение
                f = rng.uniform(1.0, 2.5)
                nh, nw = max(8, int(img.shape[0] / f)), max(8, int(img.shape[1] / f))
                sx, sy = img.shape[1] / nw, img.shape[0] / nh
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                open_ = cv2.resize(open_, (nw, nh), interpolation=cv2.INTER_NEAREST)
                a = a @ np.diag([sx, sy, 1.0])
            units = UNITS_PER_PX * math.exp(rng.uniform(-0.12, 0.12))
        m = to_world(self.h0, a, units)
        fp = footprint(m, (img.shape[1], img.shape[0]))
        lo, hi = fp.min(0), fp.max(0)
        cw, ch = self.canvas
        span = hi - lo
        if self.train:
            q0 = -lo + np.array([rng.uniform(min(0, cw - span[0]), max(0, cw - span[0])),
                                 rng.uniform(min(0, ch - span[1]), max(0, ch - span[1]))])
        else:
            q0 = -lo + (np.array([cw, ch]) - span) / 2
        t = np.array([[1, 0, q0[0]], [0, 1, q0[1]], [0, 0, 1]]) @ m
        out = cv2.warpPerspective(img, t, (cw, ch), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        val = cv2.warpPerspective(open_, t, (cw, ch), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (torch.from_numpy(out).permute(2, 0, 1), torch.from_numpy(val),
                torch.tensor(q0, dtype=torch.float32), torch.tensor(units, dtype=torch.float32),
                torch.tensor([float(r["cx"]), float(r["cy"])]))


def color_gpu(x: torch.Tensor) -> torch.Tensor:
    """Яркость, контраст и баланс каналов; x в [0, 1]."""
    n, dev = x.shape[0], x.device
    c = torch.rand(n, 1, 1, 1, device=dev) * 0.4 + 0.8
    b = torch.rand(n, 1, 1, 1, device=dev) * 0.16 - 0.08
    g = torch.rand(n, 3, 1, 1, device=dev) * 0.12 + 0.94
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m + b).mul_(g).clamp_(0, 1)


def patch_loss(model, x, val, q0, units, cam, sigma: float, scene_w: float,
               max_patches: int = 6000):
    """Потеря на батче холстов.

    Логиты карты считаются только для отобранных патчей (не больше
    max_patches случайных): полная карта 4096 логитов на каждый патч
    батча вдвое удлиняла шаг, а градиента от закрытых патчей всё равно нет.
    """
    e, z, sc = model.head.features(model.encoder(x))
    b, _, h, w = z.shape
    s = model.stride
    vf = F.avg_pool2d(val[:, None].float() / 255, s, s)[:, 0, :h, :w]
    off = cell_offsets(h, w, s, q0, units)
    pts = cam[:, None, None, :] + off
    inmap = ((pts >= 0) & (pts <= 1)).all(-1)
    sel = (vf > 0.6) & inmap
    lsc = F.binary_cross_entropy_with_logits(sc.float(), vf)
    if sel.sum() == 0:
        return lsc * scene_w, {"loc": 0.0, "scene": float(lsc.detach()), "n": 0}
    zs = z.permute(0, 2, 3, 1)[sel]
    ps = pts[sel]
    if len(zs) > max_patches:
        k = torch.randperm(len(zs), device=zs.device)[:max_patches]
        zs, ps = zs[k], ps[k]
    lg = model.head.loc_of(zs).float()
    tgt = soft_target_sep(ps, model.grid, sigma)
    lloc = -(tgt * F.log_softmax(lg, 1)).sum(1).mean()
    with torch.no_grad():
        p, _ = decode_points(lg.detach(), model.grid)
        perr = (p - ps).norm(dim=1).median() * MAP_UNITS
    return lloc + scene_w * lsc, {"loc": float(lloc.detach()), "scene": float(lsc.detach()),
                                  "n": int(sel.sum()), "patch_med": float(perr)}


def eval_split(model, rows, root, h0, keep, dev, step: int = 6) -> dict:
    """Камера и отдельный патч на полной сцене отложенных роликов."""
    model.eval()
    sub = rows[::step]
    imgs = [cv2.cvtColor(cv2.imread(str(root / r["scene"])), cv2.COLOR_BGR2RGB) for r in sub]
    cams, agree, patches = infer_images(model, imgs, [np.eye(3)] * len(imgs),
                                        [keep] * len(imgs), h0, dev)
    return summarize(cams, patches, np.array([[float(r["cx"]), float(r["cy"])] for r in sub]),
                     agree)


def summarize(cams, patches, true, agree=None) -> dict:
    """Метрики камеры и отдельного патча, в игровых единицах."""
    e = np.linalg.norm(cams - true, axis=1) * MAP_UNITS
    e = np.where(np.isnan(e), 1e5, e)
    pe = np.concatenate([np.linalg.norm(p["pts"] - (t + p["off"]), axis=1)
                         for p, t in zip(patches, true) if len(p["pts"])] or [np.array([])])
    pe = pe * MAP_UNITS
    out = {"n": int(len(e)), "median": float(np.median(e)), "mean": float(e.mean()),
           "miss1000": float((e > 1000).mean()),
           "patch_n": int(len(pe)),
           "patch_median": float(np.median(pe)) if len(pe) else float("nan"),
           "patch_within500": float((pe < 500).mean()) if len(pe) else float("nan")}
    if agree is not None:
        out["agree_median"] = float(np.median(agree))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--encoder", default="cnn",
                    help="cnn — свой энкодер; dinov2_vits14 / dinov2_vitb14 — замороженный DINOv2")
    ap.add_argument("--root", type=Path, default=COORDS_DATA)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=1.0, help="ширина цели, ячеек сетки карты")
    ap.add_argument("--scene-weight", type=float, default=0.5)
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--out", type=Path, default=PATCH_RUNS / "cnn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with tee_stdout(args.out / TRAIN_LOG):
        run(args)


def run(args) -> None:
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    h0, store = load_projection()
    keep = stored_keep(layout, store=store)
    print("обучающие ролики:")
    tr = load_rows(args.train, args.root, args.min_quality)
    print("отложенные ролики:")
    te = load_rows(args.test, args.root, args.min_quality)

    model = PatchNet(args.encoder).to(dev)
    s = model.stride
    # Холст вмещает след целого кадра (432×257 при 6 ед./px) с запасом.
    canvas = (448, 272) if s == 16 else (448, 266)
    print(f"энкодер {args.encoder}, шаг патча {s} px = {s * UNITS_PER_PX:.0f} ед., "
          f"холст {canvas[0]}x{canvas[1]}, обучаемых параметров "
          f"{count_params(model, True):,} из {count_params(model):,}")
    ds = PatchSamples(tr, args.root, h0, keep, canvas, train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)
    opt = torch.optim.AdamW(model.trainable(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * len(dl),
                                                pct_start=0.15)
    hist = []
    t_start = time.perf_counter()
    stopped = False
    for ep in range(args.epochs):
        if args.max_minutes and (time.perf_counter() - t_start) / 60 > args.max_minutes:
            print(f"остановка по времени на эпохе {ep} из {args.epochs}")
            stopped = True
            break
        model.train()
        t0 = time.perf_counter()
        agg = {"loss": 0.0, "loc": 0.0, "scene": 0.0, "patch_med": 0.0}
        nb = 0
        for x, val, q0, units, cam in dl:
            x = color_gpu(x.to(dev, non_blocking=True).float().div_(255))
            val, q0 = val.to(dev, non_blocking=True), q0.to(dev, non_blocking=True)
            units, cam = units.to(dev), cam.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss, st = patch_loss(model, x, val, q0, units, cam, args.sigma,
                                      args.scene_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable(), 5.0)
            opt.step()
            sched.step()
            agg["loss"] += float(loss.detach())
            for k in ("loc", "scene", "patch_med"):
                agg[k] += st.get(k, 0.0)
            nb += 1
        row = {"epoch": ep + 1, "sec": time.perf_counter() - t0,
               **{k: v / max(nb, 1) for k, v in agg.items()}}
        line = (f"эпоха {ep + 1:2d}  loss {row['loss']:.3f} (место {row['loc']:.3f}, "
                f"сцена {row['scene']:.3f})  патч на обучении ~{row['patch_med']:.0f} ед.  "
                f"{row['sec']:.0f} с")
        last = ep + 1 == args.epochs
        if (ep + 1) % args.eval_every == 0 or last:
            ev = eval_split(model, te, args.root, h0, keep, dev)
            row["holdout"] = ev
            line += (f"  |  отложенные: камера медиана {ev['median']:.0f} ед., "
                     f">1000 ед. {ev['miss1000'] * 100:.1f}%, патч медиана "
                     f"{ev['patch_median']:.0f} ед.")
        hist.append(row)
        print(line, flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state() | {"trained_on": args.train, "epoch": ep + 1},
                   args.out / "model.pt")

    final = eval_split(model, te, args.root, h0, keep, dev, step=1)
    print(f"\nотложенные, все кадры: камера медиана {final['median']:.0f} ед., "
          f"среднее {final['mean']:.0f}, >1000 ед. {final['miss1000'] * 100:.1f}%; "
          f"патч медиана {final['patch_median']:.0f} ед., "
          f"в пределах 500 ед. {final['patch_within500'] * 100:.1f}%")
    per = {}
    for vid in args.test:
        rows = [r for r in te if r["video_id"] == vid]
        per[vid] = eval_split(model, rows, args.root, h0, keep, dev, step=1)
        print(f"  {vid}: камера медиана {per[vid]['median']:.0f} ед., "
              f">1000 ед. {per[vid]['miss1000'] * 100:.1f}%, "
              f"патч медиана {per[vid]['patch_median']:.0f} ед.")
    torch.save(model.state() | {"trained_on": args.train, "epoch": len(hist)},
               args.out / "model.pt")
    (args.out / "report.json").write_text(json.dumps({
        "encoder": args.encoder, "train": args.train, "test": args.test,
        "units_per_px": UNITS_PER_PX, "canvas": canvas,
        "params_trainable": count_params(model, True), "holdout": final,
        "holdout_per_video": per, "history": hist, "stopped_early": stopped,
        "hyperparams": {k: str(v) for k, v in vars(args).items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"чекпоинт и отчёт → {args.out}")


if __name__ == "__main__":
    main()
