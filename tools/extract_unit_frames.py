"""Полные кадры 1920×1080 из ролика для детектора юнитов.

Зачем. 646 опорных кадров dataset/videos мало для детектора; ролики канала
дают кадр раз в несколько секунд без ручного труда — метки потом ставит
tools/build_unit_labels.py по полоскам ХП. Локальный файл data/videos/<id>.mp4
читается напрямую, иначе ролик идёт потоком с YouTube (на диск видео не
пишется, прокси — через HTTPS_PROXY, как у mine_frames.py).

Пример:
    python tools/extract_unit_frames.py ibUVbSX7ARU --every 2
    python tools/extract_unit_frames.py IPZjdkFsPQM --every 4      # поток
    # → data/units/frames/<id>/<id>_t<секунды*10>.jpg
"""

from __future__ import annotations

import argparse

from PIL import Image

import _bootstrap  # noqa: F401
from lolstream.video_stream import frames, probe, probe_url, short_stream
from lolvision.paths import DATA, VIDEOS

OUT = DATA / "units" / "frames"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="ID ролика YouTube")
    ap.add_argument("--every", type=float, default=2.0, help="шаг, с")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    local = VIDEOS / f"{args.video}.mp4"
    if local.exists():
        w, h, dur = probe(local)
        src, url = local, False
    else:
        src = f"https://www.youtube.com/watch?v={args.video}"
        w, h, dur, _, _ = probe_url(src)
        url = True
    if (w, h) != (1920, 1080):
        raise SystemExit(f"{args.video}: {w}×{h}, нужен 1920×1080")
    out = OUT / args.video
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, fr in enumerate(frames(src, w, h, 1.0 / args.every, url, skip=args.start)):
        t = args.start + i * args.every
        Image.fromarray(fr).save(out / f"{args.video}_t{round(t * 10):05d}.jpg",
                                 quality=args.quality)
        n += 1
    print(f"{args.video}: {n} кадров за {dur:.0f} с -> {out}")
    # Оборванный поток выглядел бы готовым роликом с дырой в конце.
    short = short_stream(n, dur, 1.0 / args.every, args.start)
    if short:
        raise SystemExit(f"{args.video}: {short}")


if __name__ == "__main__":
    main()
