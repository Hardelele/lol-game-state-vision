"""Скорость моделей координат: CoordNet, патчевая CNN и DINOv2 + голова.

Зачем. Модель пойдёт в live (tools/live.py): кадр трансляции → место
камеры. Важна не только точность, но и сколько стоит один кадр — и самой
сети, и всего пути вокруг неё: маски, обрезки, перевода в вид сверху и
сведения голосов патчей в одну точку.

Как. Каждая модель меряется на входе, на котором она реально работает
(CoordNet — 384×200, патчевые — вид сверху целого кадра при 6 ед./px), и
на общем размере 448×224 (делится и на 14, и на 16) для честного
сравнения. GPU: batch 1 (задержка) и batch 32 (пропускная способность),
fp32 и autocast fp16; прогрев, torch.cuda.synchronize, медиана по N
прогонам; пиковая видеопамять — torch.cuda.max_memory_allocated. CPU:
batch 1 в заданном числе потоков. Отдельно — полный путь кадра в live по
стадиям. Запускать, только когда видеокарта свободна: чужая нагрузка
делает цифры бессмысленными (скрипт печатает занятость перед стартом).

Пример:
    python tools/bench_models.py --coordnet runs/patches/coordnet-split16/model.pt \
        --patch runs/patches/cnn-split16/model.pt runs/patches/dinov2s-split16/model.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from paths import CHECKS, DATA, DATASET
from mask_frames import crop_box, mask_boxes
from coord_model import CoordNet, GRID, expected_point
from patch_model import (load_patchnet, load_projection, rectify, scene_from_box,
                         decode_points, aggregate, aggregate_batch, cell_offsets,
                         count_params, UNITS_PER_PX)

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
COMMON = (448, 224)


def timeit(fn, n: int, warm: int, cuda: bool) -> float:
    """Медиана времени одного вызова, мс."""
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def load_coordnet(path: Path, dev: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CoordNet(grid=ck.get("grid", GRID))
    m.encoder.load_state_dict(ck["encoder"])
    m.head.load_state_dict(ck["head"])
    return m.to(dev).eval(), tuple(ck["input_size"])


def net_fn(model, kind: str, x: torch.Tensor, amp: bool):
    """Один прямой проход «сеть + головы» на готовом входе."""
    def f():
        with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.float16
                                             if x.device.type == "cuda" else torch.bfloat16,
                                             enabled=amp):
            if kind == "coordnet":
                model(x * 2 - 1)
            else:
                model(x)
    return f


def bench_net(name, model, kind, size, dev, n, cpu_n, threads, rows):
    w, h = size
    for amp in (False, True):
        for b in (1, 32):
            x = torch.rand(b, 3, h, w, device=dev)
            f = net_fn(model, kind, x, amp)
            # Прогрев до сброса пика: cudnn.benchmark при первом проходе
            # пробует алгоритмы с рабочими буферами в гигабайты, и они
            # попадали бы в «пиковую память» самой модели.
            for _ in range(20):
                f()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ms = timeit(f, n if b == 1 else max(50, n // 4), 5, True)
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
            rows.append({"model": name, "input": f"{w}x{h}", "device": "cuda",
                         "precision": "fp16" if amp else "fp32", "batch": b,
                         "ms": ms, "fps": b * 1000 / ms, "peak_mb": peak})
            print(f"  {name:22s} {w}x{h} cuda {'fp16' if amp else 'fp32'} b{b:<2d} "
                  f"{ms:7.2f} мс  {b * 1000 / ms:7.0f} кадр/с  пик {peak:6.0f} МБ", flush=True)
    if cpu_n:
        torch.set_num_threads(threads)
        mc = model.to("cpu")
        x = torch.rand(1, 3, h, w)
        ms = timeit(net_fn(mc, kind, x, False), cpu_n, 5, False)
        rows.append({"model": name, "input": f"{w}x{h}", "device": f"cpu x{threads}",
                     "precision": "fp32", "batch": 1, "ms": ms, "fps": 1000 / ms})
        print(f"  {name:22s} {w}x{h} cpu{threads} fp32 b1  {ms:7.2f} мс", flush=True)
        model.to(dev)


def live_path(name, model, kind, frame, layout, dev, n, size=None):
    """Полный путь: кадр 1920×1080 → маска/обрезка/масштаб → сеть → точка."""
    H, W = frame.shape[:2]
    boxes = mask_boxes((W, H), layout)
    crop = crop_box((W, H), layout, "dense")
    store = (720, 372)
    h0, _ = load_projection()
    st = {}

    def prep():
        fr = frame.copy()
        for x0, y0, x1, y1 in boxes:
            fr[y0:y1, x0:x1] = 0
        c = fr[crop[1]:crop[3], crop[0]:crop[2]]
        if kind == "coordnet":
            return cv2.resize(cv2.resize(c, store, interpolation=cv2.INTER_AREA),
                              size, interpolation=cv2.INTER_AREA)
        a = scene_from_box(crop, (c.shape[1], c.shape[0]), crop, store)
        return rectify(c, a, h0, UNITS_PER_PX, model.stride)

    st["prep"] = timeit(prep, n, 10, False)
    p = prep()

    if kind == "coordnet":
        def net():
            x = torch.from_numpy(p).to(dev).permute(2, 0, 1)[None].float().div_(127.5).sub_(1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return expected_point(model(x).float(), GRID).cpu()
        st["net"] = timeit(net, n, 20, True)
        st["decode"] = 0.0
        st["aggregate"] = 0.0
    else:
        cv, val, q0 = p
        s = model.stride

        def to_dev():
            x = torch.from_numpy(cv).to(dev).permute(2, 0, 1)[None].float().div_(255)
            v = torch.from_numpy(val).to(dev)[None].float().div_(255)
            return x, v

        def net():
            x, v = to_dev()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return model(x)
        st["net"] = timeit(net, n, 20, True)
        x, v = to_dev()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            e, lo, sc = model(x)
        g2, hh, ww = lo.shape[1:]
        q0t = torch.tensor(q0[None], device=dev, dtype=torch.float32)

        def dec():
            pts, mass = decode_points(lo[0].permute(1, 2, 0).reshape(-1, g2), model.grid)
            off = cell_offsets(hh, ww, s, q0t, torch.full((1,), UNITS_PER_PX, device=dev))
            vf = torch.nn.functional.avg_pool2d(v[:, None], s, s)[0, 0, :hh, :ww]
            wt = torch.sigmoid(sc[0].float()) * mass.reshape(hh, ww) * (vf >= 0.5)
            cand = (pts.reshape(hh, ww, 2) - off[0]).reshape(1, -1, 2)
            return cand, wt.reshape(1, -1)
        st["decode"] = timeit(dec, n, 20, True)
        cand, wt = dec()
        # Агрегация на видеокарте (так работает predict_canvases) с выгрузкой
        # ответа; для сравнения — тот же алгоритм на numpy.
        st["aggregate"] = timeit(lambda: aggregate_batch(cand, wt)[0].cpu(), n, 20, True)
        cn, wn = cand[0].cpu().numpy().astype(np.float64), wt[0].cpu().numpy().astype(np.float64)
        st["aggregate_numpy_cpu"] = timeit(lambda: aggregate(cn, wn), n, 10, False)
        st["patches"] = int((wn > 0).sum())
    st["total"] = st["prep"] + st["net"] + st["decode"] + st["aggregate"]
    print(f"  {name:22s} live: подготовка {st['prep']:.2f} мс, сеть {st['net']:.2f}, "
          f"разбор патчей {st['decode']:.2f}, агрегация {st['aggregate']:.2f} "
          f"(numpy {st.get('aggregate_numpy_cpu', 0):.2f}) "
          f"→ итого {st['total']:.2f} мс ({1000 / st['total']:.0f} кадр/с)", flush=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coordnet", type=Path, required=True)
    ap.add_argument("--patch", type=Path, nargs="+", required=True)
    ap.add_argument("--frame", type=Path,
                    default=DATASET / "videos" / "58w57eJ5Qks" / "frames" / "58w57eJ5Qks_t00300.jpg",
                    help="полный кадр 1920×1080 для замера пути live")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--cpu-runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=CHECKS / "bench_models.json")
    args = ap.parse_args()

    dev = "cuda"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"видеокарта перед замером: {smi}")
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "cpu_threads": args.threads, "smi_before": smi}
    print(f"torch {env['torch']}, CUDA {env['cuda']}, cuDNN {env['cudnn']}")
    torch.backends.cudnn.benchmark = True
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    if args.frame.exists():
        frame = cv2.cvtColor(cv2.imread(str(args.frame)), cv2.COLOR_BGR2RGB)
    else:
        # Кадры датасета хранятся только локально; без них берём 5-ю минуту ролика.
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "300", "-i",
                              str(DATA / "videos" / "58w57eJ5Qks.mp4"), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        frame = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
    h0, store = load_projection()

    models = []
    cn, cn_size = load_coordnet(args.coordnet, dev)
    models.append(("CoordNet", cn, "coordnet", cn_size))
    for p in args.patch:
        m = load_patchnet(p, dev)
        cv, _, _ = rectify(np.zeros((store[1], store[0], 3), np.uint8), np.eye(3), h0,
                           UNITS_PER_PX, m.stride)
        nm = "патч-CNN" if m.kind == "cnn" else f"{m.kind}+голова"
        models.append((nm, m, "patch", (cv.shape[1], cv.shape[0])))

    rows, live, params = [], {}, {}
    for nm, m, kind, size in models:
        params[nm] = {"total": count_params(m), "trainable_in_ckpt":
                      count_params(m) - (count_params(m.encoder) if "dino" in nm else 0)}
        print(f"\n{nm}: параметров {params[nm]['total']:,}")
        bench_net(nm, m, kind, size, dev, args.runs, args.cpu_runs, args.threads, rows)
        if size != COMMON:
            bench_net(nm, m, kind, COMMON, dev, args.runs, 0, args.threads, rows)
        live[nm] = live_path(nm, m, kind, frame, layout, dev, args.runs, size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"env": env, "params": params, "net": rows,
                                    "live": live}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\nсводка → {args.out}")


if __name__ == "__main__":
    main()
