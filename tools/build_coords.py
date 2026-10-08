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
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from paths import COORDS_DATA
from paths import ROOT
from scene_data import crop_box
from mask_frames import mask_boxes
from minimap_camera import MM, white_mask, find_box, estimate_box

ROOT_VIDEOS = ROOT / "data" / "videos"
STORE_W = 720   # ширина хранимой сцены; вход модели делается из неё
MINI_STORE = 256


def probe_url(url: str) -> tuple[int, int, float, str, dict]:
    """Размеры и длительность ролика без его скачивания."""
    out = subprocess.run(
        ["python", "-m", "yt_dlp", "-f", "bv*[height<=1080]", "--skip-download",
         "--dump-json", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"не удалось получить сведения о ролике: {url}")
    d = json.loads(out.stdout)
    return int(d["width"]), int(d["height"]), float(d["duration"]), d["id"], d


def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return int(s["width"]), int(s["height"]), float(d["format"]["duration"])


@lru_cache(maxsize=1)
def have_nvdec() -> bool:
    """Умеет ли ffmpeg распаковывать видео на видеокарте."""
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-hwaccels"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in out.split()


def frames(video, w: int, h: int, fps: float, from_url: bool = False,
           decode: str = "auto", skip: float = 0.0):
    """Последовательный поток кадров через ffmpeg; память не растёт.

    Из YouTube читаем через трубу от yt-dlp, а не отдаём ffmpeg прямой
    медиа-адрес: YouTube душит прямое обращение к нему, и ffmpeg на таком
    адресе просто висит (замерено: 7 минут без единого байта, при том что
    yt-dlp по той же ссылке идёт на 18 МиБ/с — он снимает ограничение,
    решая проверочный скрипт). На диск при этом не пишется ничего.

    Распаковка — узкое место всей сборки, а не модель и не сеть. Замер на
    пяти минутах матча (AV1 1080p60, одинаковый выход 1920×1080): на
    половине ядер процессора 53 с, на видеокарте 12.5 с. Поэтому по
    умолчанию берём NVDEC, а на машине без него откатываемся на процессор.
    """
    # Без ограничения ffmpeg забирает все ядра на распаковку 1080p60,
    # и машина перестаёт отзываться.
    threads = max(2, (os.cpu_count() or 8) // 2)
    gpu = decode == "gpu" or (decode == "auto" and have_nvdec())
    puller = None
    if from_url:
        puller = subprocess.Popen(
            ["python", "-m", "yt_dlp", "-f", "bv*[height<=1080]", "-o", "-",
             "--quiet", "--no-warnings", str(video)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
        src, stdin = "pipe:0", puller.stdout
    else:
        src, stdin = str(video), None
    # Кадры остаются в видеопамяти до прореживания, вниз едут только
    # отобранные: выгружать 60 кадров в секунду, чтобы взять один, — это
    # и есть та разница в четыре раза.
    head = (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if gpu
            else ["-threads", str(threads)])
    vf = f"fps={fps},hwdownload,format=nv12" if gpu else f"fps={fps}"
    # Перемотка только «пропуском»: по трубе назад не прыгнуть, а прямой
    # медиа-адрес YouTube душит (проверено: ffmpeg с -ss по нему за 100 с не
    # получил ни байта). Пропуск стоит распаковки, то есть около 25 секунд
    # на каждые десять минут ролика.
    seek = ["-ss", f"{skip:.3f}"] if skip > 0 else []
    cmd = ["ffmpeg", "-v", "error", *head,
           "-i", src, *seek, "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=stdin, bufsize=n * 2)
    if puller is not None:
        puller.stdout.close()          # трубой дальше владеет ffmpeg
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        # Сначала гасим ffmpeg, потом закрываем трубу: в обратном порядке он
        # успевает написать в закрытый канал и завалить вывод руганью про
        # broken pipe, хотя всё в порядке — просто читатель ушёл.
        if p.poll() is None:
            p.terminate()
        p.stdout.close()
        p.wait()
        if puller is not None:
            puller.terminate()
            puller.wait()


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
        meta = ROOT_VIDEOS / f"{vid}.info.json"
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
