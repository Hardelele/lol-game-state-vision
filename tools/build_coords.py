"""Плотный датасет «кадр → координаты камеры» без участия человека.

Метка берётся с миникарты: белая рамка вьюпорта даёт положение камеры на
карте. Поэтому размечать нечего — из ролика получается столько примеров,
сколько кадров мы из него возьмём.

Утечки нет: нижняя полоса с миникартой закрыта маской и во вход модели не
попадает. Рамка — источник цели, а не признак.

Кадры достаются одним проходом ffmpeg (`-vf fps=N`), а не перемоткой на
каждый кадр: перемотка с точным поиском на часовом ролике на порядок
дороже последовательного декодирования.

Сохраняется не полный кадр, а уже обрезанная сцена — она и нужна модели,
и весит в разы меньше. Разрешение хранения выше входа модели, чтобы можно
было менять размер входа без повторного декодирования.

Пример:
    python tools/build_coords.py data/videos/58w57eJ5Qks.mp4 \
        --layout dataset/layouts/spectator-volibear-challenger.json --fps 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.paths import COORDS_DATA, VIDEOS
from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.minimap_camera import MM, white_mask, find_box, estimate_box
from lolvision.dataset_io import MINI_STORE, STORE_W
from video_stream import probe_url, probe, have_nvdec, frames



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path, nargs="?", default=None,
                    help="локальный файл; либо задайте --url")
    ap.add_argument("--url", default=None,
                    help="ссылка на YouTube или идентификатор: ролик читается "
                         "потоком, на диск не сохраняется")
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=1.0, help="кадров в секунду ролика")
    ap.add_argument("--out", type=Path, default=COORDS_DATA)
    ap.add_argument("--jpeg", type=int, default=88)
    ap.add_argument("--decode", choices=("auto", "gpu", "cpu"), default="auto",
                    help="чем распаковывать видео; это узкое место сборки")
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    if not args.video and not args.url:
        raise SystemExit("укажите файл или --url")
    if args.url:
        url = args.url if "://" in args.url else             f"https://www.youtube.com/watch?v={args.url}"
        w, h, dur, vid, info = probe_url(url)
        source, from_url = url, True
        # Описание пригодится дальше (роль, сторона), сохраняем рядом.
        meta = VIDEOS / f"{vid}.info.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        if not meta.exists():
            meta.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        print(f"{vid}: читаю потоком, на диск видео не пишется")
    else:
        vid = args.video.stem
        w, h, dur = probe(args.video)
        source, from_url = args.video, False
    print(f"{vid}: {w}x{h}, {dur:.0f} с, выборка {args.fps} кадр/с "
          f"→ ожидается ~{int(dur * args.fps)} кадров")

    cx0, cy0, cx1, cy1 = crop_box((w, h), layout, "dense")
    store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
    boxes = mask_boxes((w, h), layout)
    mm0, mn0, mm1, mn1 = layout["labelling"]["minimap"]
    mmbox = (int(mm0 * w), int(mn0 * h), int(mm1 * w), int(mn1 * h))
    print(f"  сцена {cx1 - cx0}x{cy1 - cy0} → хранение {STORE_W}x{store_h}")
    use_gpu = args.decode == "gpu" or (args.decode == "auto" and have_nvdec())
    print(f"  распаковка: {'видеокарта (NVDEC)' if use_gpu else 'процессор'}")

    sdir = args.out / vid / "scene"
    mdir = args.out / vid / "mini"
    sdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    # Размер рамки постоянен внутри ролика, но оценивать его надо по кадрам
    # со всего ролика: первые секунды — это заставка и чёрный кадр, по ним
    # оценка уезжает. Маски копим целиком (на 1500 кадров это ~90 МБ).
    masks: list[np.ndarray] = []

    def handle(i: int, fr: np.ndarray) -> np.ndarray:
        """Обработка одного кадра: маска миникарты и два файла на диск."""
        img = Image.fromarray(fr)
        mini = img.crop(mmbox).resize((MM, MM), Image.LANCZOS)
        scene = img.copy()
        for bx in boxes:
            scene.paste((0, 0, 0), bx)
        scene.crop((cx0, cy0, cx1, cy1)).resize(
            (STORE_W, store_h), Image.LANCZOS).save(
            sdir / f"{i:06d}.jpg", quality=args.jpeg)
        mini.resize((MINI_STORE, MINI_STORE)).save(
            mdir / f"{i:06d}.jpg", quality=82)
        return white_mask(np.asarray(mini, dtype=np.float32)).astype(np.uint8)

    # Уменьшение и сжатие идут в потоках: после перехода на распаковку
    # видеокартой именно они стали узким местом (замер: 11.5 мс на одно
    # уменьшение LANCZOS против 42 мс на распаковку секунды видео). PIL на
    # этих операциях отпускает GIL, так что потоки работают вправду
    # параллельно. Очередь ограничена: кадр 1080p весит 6 МБ, и без предела
    # распаковщик забьёт память быстрее, чем потоки разберут.
    pend: deque = deque()
    done = 0
    with ThreadPoolExecutor(max(2, (os.cpu_count() or 8) // 4)) as pool:
        for i, fr in enumerate(frames(source, w, h, args.fps, from_url,
                                      args.decode)):
            # Копия обязательна: буфер трубы переиспользуется следующим кадром.
            pend.append(pool.submit(handle, i, fr.copy()))
            while len(pend) > 12:
                masks.append(pend.popleft().result())
                done += 1
                if done % 300 == 0:
                    print(f"  {done} кадров", flush=True)
        while pend:
            masks.append(pend.popleft().result())

    step = max(1, len(masks) // 24)
    bw, bh = estimate_box([m.astype(np.float32) for m in masks[::step]])
    print(f"  рамка камеры {bw}x{bh} из {MM} ({bw / MM:.3f} ширины карты)")

    out = []
    for i, m in enumerate(masks):
        x, y, q = find_box(m.astype(np.float32), bw, bh)
        out.append({"video_id": vid, "idx": i, "t_sec": round(i / args.fps, 2),
                    "cx": round((x + bw / 2) / MM, 4),
                    "cy": round((y + bh / 2) / MM, 4),
                    "quality": round(q, 3),
                    "scene": f"{vid}/scene/{i:06d}.jpg",
                    "mini": f"{vid}/mini/{i:06d}.jpg"})

    man = args.out / f"{vid}.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    q = np.array([r["quality"] for r in out])
    print(f"{len(out)} кадров, качество детекции: медиана {np.median(q):.2f}, "
          f"ниже 0.4 у {int((q < 0.4).sum())} ({(q < 0.4).mean() * 100:.1f}%)")
    print(f"манифест → {man}")


if __name__ == "__main__":
    main()
