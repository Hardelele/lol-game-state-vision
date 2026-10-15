"""Обучение детектора изменений (lolvision/change_net.py, #818).

Кадры — потоком из роликов вида игрока (ffmpeg → 960×540 RGB, 15 кадр/с:
шаг 4 кадра при 60 кадр/с), без записи на диск. Потоков STREAMS, каждый
читает свой ролик с случайного места и через SEG кадров прыгает на новое.
Пример — три подряд кадра одного потока: два прошлых → текущий.

Детектор юнитов заморожен (признаки шеи), учится только ChangeNet.
Отложены ролики с разметкой добиваний (#816): D37cbVNs0EU, InEwM0UBAbc.

    python tools/train_change.py --steps 20000 --name base
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import subprocess
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision import change_net as cn
from lolvision.paths import RUNS, VIDEOS
from lolvision.unit_net import IN_H, IN_W, LIVE_NET, STRIDE, load_net
from lolvision.unit_targets import unit_keep

TRAIN = ["JKZgBDX1WfY", "8lkxSQHq81k", "1Z9DJyLsXHo", "Pq2V04mhRTE",
         "tEMffCUg-Js", "htjWVBQd0gw", "LIZxk-kM-6o", "ANXVB571YkE"]
HELD = ["D37cbVNs0EU", "InEwM0UBAbc"]
FPS = 15
SEG = 150                       # кадров потока до прыжка (10 с)
OUT = RUNS / "change"
FW, FH = IN_W, 540              # кадр ffmpeg; до 544 — дополнение нулями


def duration(vid: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(VIDEOS / f"{vid}.mp4")],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def keep_mask() -> np.ndarray:
    k = unit_keep(1920, 1080, "player").astype(np.float32)
    return cv2.resize(k, (FW, FH), interpolation=cv2.INTER_AREA)[..., None]


def read_frames(vid: str, start: float, n: int, fps: float = FPS):
    """n кадров 960×540 RGB uint8 с секунды `start`."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(VIDEOS / f"{vid}.mp4"),
           "-vf", f"fps={fps:g},scale={FW}:{FH}:flags=area", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = FW * FH * 3
    try:
        for _ in range(n):
            buf = p.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def gray_small(fr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(fr, (FW // 4, FH // 4), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)


def stream(vids: list[str], durs: dict, q: queue.Queue, keep: np.ndarray, seed: int,
           stop: threading.Event) -> None:
    """Поток примеров: (кадр1, кадр2, кадр3, сдвиг 1→3, сдвиг 2→3) в клетках сетки."""
    rng = random.Random(seed)
    hud = cv2.resize(keep[..., 0], (FW // 4, FH // 4))
    while not stop.is_set():
        vid = rng.choice(vids)
        start = rng.uniform(30, durs[vid] - SEG / FPS - 5)
        win: list = []
        for fr in read_frames(vid, start, SEG):
            g = gray_small(fr) * hud
            win = (win + [(fr, g)])[-3:]
            if len(win) < 3 or rng.random() < 0.5:
                continue
            (a, ga), (b, gb), (c, gc) = win
            # сдвиг в пикселях 240×135 = клетки сетки шага 4 при входе 960
            s13 = cn.camera_shift(ga, gc)
            s23 = cn.camera_shift(gb, gc)
            q.put((a, b, c, s13, s23))
            if stop.is_set():
                return


def to_input(frs: list[np.ndarray], keep_t: torch.Tensor, dev: str) -> torch.Tensor:
    x = torch.from_numpy(np.stack(frs)).to(dev).permute(0, 3, 1, 2).float() / 255
    x = x * keep_t
    return F.pad(x, (0, 0, 0, IN_H - FH))


class Prep:
    """Кадры → нормированные признаки шеи и картинка в сетке шага 4."""

    def __init__(self, dev: str):
        self.net = load_net(LIVE_NET, dev).half()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.keep = torch.from_numpy(keep_mask()).permute(2, 0, 1)[None].to(dev)
        self.scale: torch.Tensor | None = None
        self.dev = dev

    @torch.no_grad()
    def __call__(self, frs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        x = to_input(frs, self.keep, self.dev)
        f = cn.neck(self.net, x.half()).float()
        if self.scale is None:
            self.scale = f.std((0, 2, 3), keepdim=True).clamp_min(1e-3)
        r = F.avg_pool2d(x, STRIDE) * 4            # картинка в той же сетке, шкала ~признаков
        return f / self.scale, r


def batch_tensors(prep: Prep, items: list) -> tuple:
    b = len(items)
    frs = [it[k] for k in range(3) for it in items]
    f, r = prep(frs)
    f1, f2, f3 = f[:b], f[b:2 * b], f[2 * b:]
    r1, r2, r3 = r[:b], r[b:2 * b], r[2 * b:]
    d13 = torch.tensor([it[3] for it in items], device=f.device, dtype=f.dtype)
    d23 = torch.tensor([it[4] for it in items], device=f.device, dtype=f.dtype)
    f1, ok1 = cn.shift_grid(f1, d13)
    r1, _ = cn.shift_grid(r1, d13)
    f2, ok2 = cn.shift_grid(f2, d23)
    r2, _ = cn.shift_grid(r2, d23)
    k = (F.avg_pool2d(prep.keep, STRIDE) > 0.99).float()
    k = F.pad(k, (0, 0, 0, f1.shape[-2] - k.shape[-2]))
    valid = ok1 * ok2 * k
    return f1, f2, f3, r1, r2, r3, valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--streams", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--name", default="base")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    durs = {v: duration(v) for v in TRAIN}
    keep = keep_mask()
    q: queue.Queue = queue.Queue(maxsize=256)
    stop = threading.Event()
    ths = [threading.Thread(target=stream, args=(TRAIN, durs, q, keep, s, stop), daemon=True)
           for s in range(args.streams)]
    for t in ths:
        t.start()
    prep = Prep(dev)
    net = cn.ChangeNet(args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    log = open(out / "train.log", "a", encoding="utf-8")
    t0, run = time.time(), {}
    pool: list = []
    for step in range(1, args.steps + 1):
        # Перемешивание: буфер примеров из разных потоков.
        while len(pool) < 4 * args.batch:
            pool.append(q.get())
        random.shuffle(pool)
        items, pool = pool[:args.batch], pool[args.batch:]
        f1, f2, f3, r1, r2, r3, valid = batch_tensors(prep, items)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pf, pr, lf, lr_ = net(f1, f2, r1, r2)
        ef, er = cn.errors(pf.float(), pr.float(), f3, r3)
        loss_map = cn.nll(ef, lf.float()) + cn.nll(er, lr_.float())
        loss = (loss_map * valid).sum() / valid.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            m = valid.sum().clamp_min(1)
            base = (((f2 - f3) ** 2).mean(1, keepdim=True) * valid).sum() / m
            for k, v in (("loss", loss), ("ef", (ef * valid).sum() / m), ("copy", base)):
                run[k] = 0.98 * run.get(k, float(v)) + 0.02 * float(v)
        if step % 100 == 0:
            line = (f"step {step} loss {run['loss']:.4f} err_f {run['ef']:.4f} "
                    f"copy_f {run['copy']:.4f} q {q.qsize()} {time.time() - t0:.0f}s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        if step % 2000 == 0 or step == args.steps:
            torch.save({"state": net.state_dict(), "width": args.width,
                        "scale": prep.scale.cpu(), "step": step, "fps": FPS,
                        "detector": str(LIVE_NET)}, out / "model.pt")
    stop.set()
    (out / "report.json").write_text(json.dumps({"steps": args.steps, "train": TRAIN,
                                                 "held": HELD, "final": run}, indent=1),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
