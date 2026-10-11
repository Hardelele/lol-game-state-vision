"""Обучить детектор юнитов (lolvision/unit_net.py) на псевдо-метках полосок.

Зачем. Этап 3 плана второй модели: псевдо-метки классических читателей
(tools/build_unit_labels.py) → свой быстрый детектор. Сплит — по роликам:
проверочные ролики в обучение не попадают.

Аугментации (раздел 3.4 плана, «размер — не признак»): масштаб кадра
×0.6–1.5 вместе с метками (лог-равномерно), случайный вырез 960×544;
яркость/контраст; перекодирование JPEG с низким качеством и шум —
полоски миньонов тонкие, сжатие видео их съедает (docs/pitfalls.md).

`--bar-aug` (#773): стирание, перекрытие, перенос и перекраска полосок
(BarAug) — без них сеть находит полоску, а не тело (замер
tools/eval_units_bars.py). Тогда же проверка идёт и на кадрах со стёртыми
полосками, лучшая эпоха — по среднему F1 обоих вариантов.

Пример:
    python tools/train_units.py --train ibUVbSX7ARU zJvTSjEnKNE ... \
        --val 58w57eJ5Qks olmTXkkUv58 --epochs 30 --name net
    # → runs/units/<имя>/model.pt, report.json, train.log; last.pt — для
    #   продолжения: повторный запуск с тем же --name продолжает обучение
    python tools/train_units.py ... --bar-aug --init runs/units/net/model.pt --lr 1e-3 \
        --epochs 10 --name net-nobars
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401
from build_unit_labels import LABELS, label_keep
from lolvision.bar_erase import PAD, bar_rect, erase, free_spot, paste, recolor
from lolvision.unit_net import CLASSES, IN_H, IN_W, OUT_H, OUT_W, STRIDE, UnitNet, decode
from lolvision.unit_targets import RADIUS, box_center, build_targets, ignore_map, match
from lolvision.paths import ROOT, RUNS, TRAIN_LOG

OUT = RUNS / "units"


def load_rows(names: list[str], self_labels: bool = False) -> list[dict]:
    """Строки JSONL меток по именам наборов (имя файла = ID ролика или
    `dataset:<ID>` — кадры этого ролика из общего dataset.jsonl).
    `self_labels` — наборы, дополненные tools/self_label_units.py."""
    rows = []
    for n in names:
        if self_labels:
            p = LABELS.parent / "labels_self" / f"{n.replace(':', '_')}.jsonl"
            rows += [json.loads(s) for s in p.open(encoding="utf-8")]
            continue
        if n.startswith("dataset:"):
            vid = n.split(":", 1)[1]
            src = [json.loads(s) for s in (LABELS / "dataset.jsonl").open(encoding="utf-8")]
            rows += [r for r in src if Path(r["frame"]).name.startswith(vid)]
        else:
            rows += [json.loads(s) for s in (LABELS / f"{n}.jsonl").open(encoding="utf-8")]
    return [r for r in rows if "error" not in r]


@dataclass
class BarAug:
    """Аугментации полосок (#773): сеть должна находить тело, а не полоску.

    erase — у юнита стёрта полоска (и имя у чемпиона), метка тела остаётся,
    ХП неизвестна; occlude — полоска миньона закрыта чужой плашкой чемпиона
    из того же кадра (крип за плашкой); phantom — копия полоски без тела в
    пустом месте, метки нет; decoy — стёртое пятно без юнита, чтобы след
    стирания не стал признаком; recolor — другой оттенок заливки (свои цвета
    полосок у игроков). Вероятности — на юнит (erase, occlude, recolor) или
    на кадр (phantom, decoy, erase_frame — стирать ли в этом кадре вообще).
    `all_erased` — проверочный вариант: стёрты все."""
    erase: float = 0.0
    occlude: float = 0.0
    phantom: float = 0.0
    decoy: float = 0.0
    recolor: float = 0.0
    erase_frame: float = 1.0
    erase_champion: float | None = None     # своя вероятность стирания у чемпионов
    all_erased: bool = False

    @property
    def on(self) -> bool:
        return self.all_erased or any((self.erase, self.occlude, self.phantom, self.decoy,
                                       self.recolor))


def bar_aug(img: np.ndarray, units: list[dict], k: float, valid: np.ndarray, aug: BarAug,
            rng: np.random.Generator) -> list[dict]:
    """Применить BarAug к кадру `img` (масштаб k к 1080p, на месте); вернуть
    метки с поправками: у юнита без видимой полоски нет ХП, у чемпиона без
    плашки — и команды (её видно только по плашке)."""
    src = img.copy()
    out = [dict(u) for u in units]
    shape = img.shape
    hidden = set()
    # метки самообучения (tools/self_label_units.py) — без полоски: их не
    # трогаем и не переносим
    erase_on = aug.all_erased or rng.random() < aug.erase_frame
    for j, u in enumerate(out):
        p = (aug.erase_champion if u["kind"] == "champion" and aug.erase_champion is not None
             else aug.erase)
        if u.get("bar") and (aug.all_erased or (erase_on and rng.random() < p)):
            hidden.add(j)
    erase(img, [bar_rect(out[j], k, shape) for j in sorted(hidden)], rng=rng)
    units = [u for u in units if u.get("bar")]
    champs = [u for u in units if u["kind"] == "champion"]
    for j, u in enumerate(out):
        if j in hidden or not u.get("bar"):
            continue
        if u["kind"] == "minion" and champs and rng.random() < aug.occlude:
            c = champs[int(rng.integers(len(champs)))]
            x0, y0, x1, y1 = bar_rect(c, k, shape)
            crop = src[y0:y1, x0:x1]
            if crop.size:
                bx0, by0, bx1, _ = u["bar"]
                px = int(((bx0 + bx1) / 2 + rng.integers(-25, 26)) * k - crop.shape[1] / 2)
                py = int((by0 - PAD["champion"][1] - rng.integers(3, 12)) * k)
                paste(img, crop, px, py)
                hidden.add(j)
        elif rng.random() < aug.recolor:
            x0, y0, x1, y1 = u["bar"]
            recolor(img, (int(x0 * k), int(y0 * k), int(x1 * k) + 1, int(y1 * k) + 1),
                    int(rng.integers(20, 160)))
            if u["kind"] == "champion":
                u["team"] = None
    for j in hidden:
        out[j]["hp"] = None
        if out[j]["kind"] == "champion":
            out[j]["team"] = None
    avoid = [(u["x"] * k, u["y"] * k) for u in out] + \
            [((u["bar"][0] + u["bar"][2]) / 2 * k, u["bar"][1] * k) for u in units]
    if units and rng.random() < aug.phantom:
        for _ in range(int(rng.integers(1, 3))):
            u = units[int(rng.integers(len(units)))]
            x0, y0, x1, y1 = bar_rect(u, k, shape)
            crop = src[y0:y1, x0:x1]
            if not crop.size:
                continue
            spot = free_spot(rng, (crop.shape[1], crop.shape[0] + int(120 * k)), valid, avoid,
                             130 * k)
            if spot is not None:
                paste(img, crop, *spot)
                avoid.append((spot[0] + crop.shape[1] / 2, spot[1] + 60 * k))
    if rng.random() < aug.decoy:
        rects = []
        for _ in range(int(rng.integers(1, 4))):
            w, h = (int(146 * k), int(47 * k)) if rng.random() < 0.4 else (int(60 * k), int(10 * k))
            spot = free_spot(rng, (w, h + int(120 * k)), valid, avoid, 130 * k)
            if spot is not None:
                rects.append((spot[0], spot[1], spot[0] + w, spot[1] + h))
        erase(img, rects, rng=rng)
    return out


class UnitFrames(Dataset):
    def __init__(self, rows: list[dict], keep: np.ndarray, ign: np.ndarray, train: bool,
                 scale=(0.6, 1.5), jpeg_p: float = 0.5, noise_p: float = 0.3,
                 aug: BarAug | None = None):
        self.rows, self.keep, self.ign, self.train = rows, keep, ign, train
        self.scale, self.jpeg_p, self.noise_p = scale, jpeg_p, noise_p
        self.aug = aug or BarAug()
        self.valid = ign == 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        if self.train:
            lo, hi = self.scale
            s = math.exp(random.uniform(math.log(lo), math.log(hi)))
        else:
            s = 1.0
        f = 0.5 * s
        nw, nh = round(1920 * f), round(1080 * f)
        half = self.train and f <= 0.5
        if half:
            # распаковка JPEG сразу в половинном размере — вчетверо дешевле;
            # на проверке — полный кадр, как в UnitNetRunner
            img = cv2.imread(str(ROOT / r["frame"]), cv2.IMREAD_REDUCED_COLOR_2)[..., ::-1]
        else:
            img = cv2.imread(str(ROOT / r["frame"]))[..., ::-1]
        img = np.ascontiguousarray(img)
        labels = [{**u, "x": box_center(u)[0], "y": box_center(u)[1]} for u in r["units"]]
        for u in labels:
            # команду чемпиона видно только по плашке — у самометки она догадка сети
            if u.get("source") == "self" and u["kind"] == "champion":
                u["team"] = None
        if self.aug.on:
            # до маски HUD: стирание берёт фон и у края сцены
            rng = (np.random.default_rng(random.getrandbits(32)) if self.train
                   else np.random.default_rng(i))
            labels = bar_aug(img, labels, img.shape[1] / 1920,
                             self.valid[::2, ::2] if half else self.valid, self.aug, rng)
        img = img * (self.keep[::2, ::2, None] if half else self.keep[..., None])
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA
                        if nw < img.shape[1] else cv2.INTER_LINEAR)
        ig = cv2.resize(self.ign, (nw, nh), interpolation=cv2.INTER_NEAREST)
        # вырез 960×544 (или поле, если кадр меньше)
        if self.train:
            ox = random.randint(min(0, nw - IN_W), max(0, nw - IN_W))
            oy = random.randint(min(0, nh - IN_H), max(0, nh - IN_H))
        else:
            ox, oy = 0, 0
        x = np.zeros((IN_H, IN_W, 3), np.uint8)
        g = np.ones((IN_H, IN_W), np.uint8)
        sx0, sy0 = max(0, ox), max(0, oy)
        dx0, dy0 = max(0, -ox), max(0, -oy)
        cw = min(nw - sx0, IN_W - dx0)
        ch = min(nh - sy0, IN_H - dy0)
        x[dy0:dy0 + ch, dx0:dx0 + cw] = im[sy0:sy0 + ch, sx0:sx0 + cw]
        g[dy0:dy0 + ch, dx0:dx0 + cw] = ig[sy0:sy0 + ch, sx0:sx0 + cw]
        if self.train:
            x = self.photometric(x)
        units = [{"kind": u["kind"], "team": u["team"], "hp": u.get("hp"),
                  "x": u["x"] * f - ox, "y": u["y"] * f - oy} for u in labels]
        t = build_targets(units, (OUT_H, OUT_W), STRIDE, s)
        # неуверенные ответы сетей без метки (self_label_units.py): возможно,
        # юнит — там нет потерь тепловой карты
        for x0, y0, kind in r.get("ignore", ()):
            rad = max(2, int(RADIUS[kind] * f))
            cv2.circle(g, (int(x0 * f - ox), int(y0 * f - oy)), rad, 1, -1)
        # область без меток — на сетке выхода: клетка без меток, если хоть
        # один пиксель в ней без меток
        gi = g.reshape(OUT_H, STRIDE, OUT_W, STRIDE).max((1, 3)).astype(np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return xt, {k: torch.from_numpy(v) for k, v in t.items()}, torch.from_numpy(gi)

    def photometric(self, x: np.ndarray) -> np.ndarray:
        a = x.astype(np.float32)
        a = (a - 128) * random.uniform(0.85, 1.15) + 128 + random.uniform(-15, 15)
        if random.random() < self.noise_p:
            a += np.random.default_rng(random.getrandbits(32)).standard_normal(a.shape, dtype=np.float32) * random.uniform(2, 7)
        a = np.clip(a, 0, 255).astype(np.uint8)
        if random.random() < self.jpeg_p:
            q = random.randint(25, 85)
            ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])
            a = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return a


def focal(pred: torch.Tensor, gt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Фокальная потеря CenterNet (α=2, β=4); w — 0 в области без меток."""
    p = torch.sigmoid(pred.float()).clamp(1e-4, 1 - 1e-4)
    pos = (gt >= 0.999).float()
    neg = 1 - pos
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg * w
    n = pos.sum().clamp(min=1)
    return (lp.sum() + ln.sum()) / n


def losses(out, t, ign):
    heat, team, off, hp = out
    w = (1 - ign)[:, None]
    lh = focal(heat, t["heat"], w)
    m = t["team_m"]
    lt = (F.binary_cross_entropy_with_logits(team[:, 0].float(), t["team"], reduction="none")
          * m).sum() / m.sum().clamp(min=1)
    mo = t["off_m"]
    lo = ((off.float() - t["off"]).abs().sum(1) * mo).sum() / mo.sum().clamp(min=1)
    mh = t["hp_m"]
    lhp = ((torch.sigmoid(hp[:, 0].float()) - t["hp"]).abs() * mh).sum() / mh.sum().clamp(min=1)
    return lh, lt, lo, lhp


@torch.no_grad()
def evaluate(net, loader, dev, ign_full: np.ndarray, thr: float = 0.3) -> dict:
    """Точность/полнота против псевдо-меток проверочных кадров (масштаб 1)."""
    net.eval()
    # как в UnitNetRunner (live): ответы вне `valid` гасятся до поиска
    # максимумов, иначе пики в полосе у HUD занимают места topk
    v = torch.from_numpy((1 - ign_full).astype(np.float32)).to(dev)[None, None]
    v = -F.max_pool2d(-F.avg_pool2d(v, 2), STRIDE)
    valid = F.pad(v, (0, OUT_W - v.shape[-1], 0, OUT_H - v.shape[-2]))
    stats = {k: [0, 0, 0] for k in CLASSES}      # tp, fp, fn
    team_ok = team_n = 0
    hp_err = {k: [] for k in CLASSES}
    rows = loader.dataset.rows
    k = 0
    for x, _, _ in loader:
        x = x.to(dev).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
        det = decode(*out, thr=thr, valid=valid).cpu().numpy()
        for b in range(len(x)):
            r = rows[k]
            k += 1
            preds = [{"kind": CLASSES[int(c)], "score": float(s), "x": float(px * 2),
                      "y": float(py * 2), "team": "CHAOS" if tp > 0.5 else "ORDER", "hp": float(h)}
                     for c, s, tp, px, py, _, h in det[b] if s > 0]
            preds = [p for p in preds if ign_full[min(1079, int(p["y"])), min(1919, int(p["x"]))] == 0]
            gts = []
            for u in r["units"]:
                cx, cy = box_center(u)
                if ign_full[min(1079, max(0, int(cy))), min(1919, max(0, int(cx)))] == 0:
                    gts.append({"kind": u["kind"], "x": cx, "y": cy, "team": u["team"],
                                "hp": u.get("hp")})
            pairs, fp, fn = match(preds, gts)
            for i, j in pairs:
                stats[gts[j]["kind"]][0] += 1
                team_ok += preds[i]["team"] == gts[j]["team"]
                team_n += 1
                if gts[j]["hp"] is not None:
                    hp_err[gts[j]["kind"]].append(abs(preds[i]["hp"] - gts[j]["hp"]))
            for i in fp:
                stats[preds[i]["kind"]][1] += 1
            for j in fn:
                stats[gts[j]["kind"]][2] += 1
    res = {}
    for c, (tp, fp, fn) in stats.items():
        p = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        res[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(rc, 4),
                  "f1": round(2 * p * rc / max(1e-9, p + rc), 4),
                  "hp_mae": round(float(np.mean(hp_err[c])), 4) if hp_err[c] else None}
    res["team_acc"] = round(team_ok / max(1, team_n), 4)
    net.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--name", default="net")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--widths", type=int, nargs=5, default=(24, 48, 96, 160, 256))
    ap.add_argument("--depths", type=int, nargs=5, default=(1, 2, 3, 3, 2))
    ap.add_argument("--head", type=int, default=64)
    ap.add_argument("--scale", type=float, nargs=2, default=(0.6, 1.5))
    ap.add_argument("--no-codec-aug", action="store_true", help="без JPEG и шума (абляция)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=4.0, help="предел видеопамяти процесса")
    ap.add_argument("--init", type=Path, default=None, help="начальные веса (дообучение)")
    ap.add_argument("--bar-aug", action="store_true", help="аугментации полосок BarAug (#773)")
    ap.add_argument("--erase-p", type=float, default=0.5)
    ap.add_argument("--occlude-p", type=float, default=0.15)
    ap.add_argument("--phantom-p", type=float, default=0.3)
    ap.add_argument("--decoy-p", type=float, default=0.3)
    ap.add_argument("--recolor-p", type=float, default=0.15)
    ap.add_argument("--erase-frame-p", type=float, default=1.0,
                    help="доля кадров, в которых вообще стираются полоски")
    ap.add_argument("--erase-champion-p", type=float, default=None,
                    help="вероятность стирания у чемпионов (по умолчанию — --erase-p)")
    ap.add_argument("--self-labels", action="store_true",
                    help="обучающие метки из data/units/labels_self (self_label_units.py)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda"
    # карту делят несколько процессов: не больше 4 ГБ из 16
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2 ** 30 / total))
    keep = label_keep()
    ign = ignore_map(keep)
    tr_rows, va_rows = load_rows(args.train, args.self_labels), load_rows(args.val)
    jp, npz = (0.0, 0.0) if args.no_codec_aug else (0.4, 0.25)
    aug = (BarAug(args.erase_p, args.occlude_p, args.phantom_p, args.decoy_p, args.recolor_p,
                  args.erase_frame_p, args.erase_champion_p) if args.bar_aug else BarAug())
    tr = UnitFrames(tr_rows, keep, ign, True, tuple(args.scale), jp, npz, aug)
    va = UnitFrames(va_rows, keep, ign, False)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=True, pin_memory=True)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers)
    # те же проверочные кадры со стёртыми полосками всех размеченных юнитов
    vl_erased = (DataLoader(UnitFrames(va_rows, keep, ign, False, aug=BarAug(all_erased=True)),
                            args.batch, shuffle=False, num_workers=args.workers)
                 if args.bar_aug else None)

    net = UnitNet(tuple(args.widths), tuple(args.depths), args.head).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state"])
    # без channels_last: depthwise-свёртки в fp16 с ним впятеро медленнее
    # (замер: 324 против 63 мс на шаг батча 8, RTX 5060 Ti)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / TRAIN_LOG).open("a", encoding="utf-8")
    nparam = sum(p.numel() for p in net.parameters())
    print(f"train {len(tr)} кадров, val {len(va)}, параметров {nparam / 1e6:.2f} М", file=log,
          flush=True)
    best, best_res, start = -1.0, None, 0
    last = out / "last.pt"
    if last.exists():
        # продолжение после обрыва: веса, оптимизатор, расписание, лучший итог
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, best_res, start = ck["best"], ck["best_res"], ck["epoch"]
        print(f"продолжаю с эпохи {start + 1}", file=log, flush=True)
    t0 = time.time()
    for ep in range(start, args.epochs):
        acc = np.zeros(4)
        for x, t, ig in tl:
            x = x.to(dev, non_blocking=True).float().div_(255)
            t = {k: v.to(dev, non_blocking=True) for k, v in t.items()}
            ig = ig.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x)
            ls = losses(o, t, ig)
            loss = ls[0] + ls[1] + ls[2] + 2 * ls[3]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            acc += [v.item() for v in ls]
        acc /= len(tl)
        res = evaluate(net, vl, dev, ign)
        f1 = (res["champion"]["f1"] + res["minion"]["f1"]) / 2
        msg = (f"ep {ep + 1} {time.time() - t0:.0f}s loss heat {acc[0]:.3f} team {acc[1]:.3f} "
               f"off {acc[2]:.3f} hp {acc[3]:.3f} | " +
               " ".join(f"{c} P{res[c]['precision']:.3f} R{res[c]['recall']:.3f}" for c in CLASSES)
               + f" team {res['team_acc']:.3f}")
        if vl_erased is not None:
            res["erased"] = er = evaluate(net, vl_erased, dev, ign)
            f1 = (f1 + (er["champion"]["f1"] + er["minion"]["f1"]) / 2) / 2
            msg += " | без полосок " + " ".join(
                f"{c} P{er[c]['precision']:.3f} R{er[c]['recall']:.3f}" for c in CLASSES[:2])
        print(msg, flush=True)
        print(msg, file=log, flush=True)
        if f1 > best:
            best, best_res = f1, res
            torch.save({"state": net.state_dict(), "widths": args.widths, "depths": args.depths,
                        "head": args.head, "classes": CLASSES, "epoch": ep + 1,
                        "train": args.train, "val": args.val}, out / "model.pt")
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_res": best_res, "epoch": ep + 1}, last)
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "bar_aug": asdict(aug), "train_frames": len(tr), "val_frames": len(va),
              "params_m": round(nparam / 1e6, 3), "best_val_vs_pseudo": best_res,
              "minutes": round((time.time() - t0) / 60, 1)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
