"""Чемпионы по кадрам ролика канала: где на карте, команда, жив ли, таймер.

Вход — кадры из dataset/videos/<id>/frames (по frames.csv), локальный
ролик или ссылка на YouTube (поток через ffmpeg, на диск не пишется).
Состав и порядок слотов берутся по портретам HUD с первых игровых кадров
(игровой — где на миникарте нашлась рамка камеры), или задаются
--roster. Выход — JSONL: на кадр 10 записей
{champion, team, slot, visible, x, y (ед. карты), alive, respawn}, и
сводка со временем на кадр.

Пример:
    python tools/minimap_champions.py 58w57eJ5Qks
    python tools/minimap_champions.py data/videos/olmTXkkUv58.mp4 --fps 2
    # → runs/minimap/champions/<имя>.jsonl и <имя>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision import ddragon
from lolvision.minimap_camera import find_box_clipped, white_mask
from lolvision.minimap_champions import FrameReader, roster_from_hud
from lolvision.minimap_icons import DEFAULT_RUN, IconModel
from lolvision.paths import DATASET_VIDEOS, DEFAULT_LAYOUT, RUNS

OUT = RUNS / "minimap" / "champions"
BOX = (68, 38)            # рамка камеры в сетке 256 (docs/replay-api.md)
IN_GAME_Q = 0.45          # качество рамки: ниже — не игра (заставка, таблица)


def in_game(frame: np.ndarray, layout: dict) -> bool:
    mm = FrameReader.minimap_of(frame, layout).astype(np.float32)
    return find_box_clipped(white_mask(mm), *BOX)[2] >= IN_GAME_Q


def source(arg: str, fps: float):
    """(имя, итератор (метка, кадр RGB))."""
    vdir = DATASET_VIDEOS / arg
    if vdir.is_dir():
        rows = list(csv.DictReader((vdir / "frames.csv").open(encoding="utf-8")))

        def it():
            for r in rows:
                with Image.open(vdir / r["frame"]) as im:
                    yield Path(r["frame"]).stem, np.asarray(im.convert("RGB"))
        return arg, it()
    from lolstream.video_stream import frames, probe, probe_url

    url = arg.startswith("http")
    if url:
        w, h, _, vid, _ = probe_url(arg)
    else:
        w, h, _ = probe(Path(arg))
        vid = Path(arg).stem

    def it():
        for i, f in enumerate(frames(arg, w, h, fps, from_url=url)):
            yield round(i / fps, 2), f
    return vid, it()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="id из dataset/videos, путь к ролику или ссылка")
    ap.add_argument("--fps", type=float, default=1.0, help="для ролика/ссылки")
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--roster", default=None, help="10 имён через запятую: L1..L5,R1..R5")
    ap.add_argument("--roster-frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    model = IconModel(Path(args.run), args.device)
    model_version = ddragon.DEFAULT_VERSION
    icons = ddragon.load_icons(model_version)
    name, frames = source(args.src, args.fps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    buf, game = [], []
    roster = args.roster.split(",") if args.roster else None
    margin = None
    if roster is None:
        # Буферизуем кадры до набора игровых — по ним состав.
        for key, f in frames:
            buf.append((key, f))
            if in_game(f, layout):
                game.append(f[..., ::-1])
            if len(game) >= args.roster_frames * 3:
                break
        pick = game[::3][:args.roster_frames] or [b[1][..., ::-1] for b in buf[:4]]
        roster, margin = roster_from_hud(pick, icons, model.champions)
    reader = FrameReader(model, roster, layout, icons)

    def all_frames():
        yield from buf
        yield from frames

    t_total, n = 0.0, 0
    fj = (out / f"{name}.jsonl").open("w", encoding="utf-8")
    batch = []

    def flush():
        nonlocal t_total, n
        t0 = time.perf_counter()
        recs = reader([f for _, f in batch])
        t_total += time.perf_counter() - t0
        for (key, _), r in zip(batch, recs):
            fj.write(json.dumps({"frame": key, "champions": r}, ensure_ascii=False) + "\n")
        n += len(batch)
        batch.clear()

    for key, f in all_frames():
        batch.append((key, f))
        if len(batch) >= args.batch:
            flush()
    if batch:
        flush()
    fj.close()
    rep = {"source": args.src, "frames": n, "roster": roster,
           "roster_margin": None if margin is None else [round(float(m), 3) for m in margin],
           "ms_per_frame": round(1000 * t_total / max(1, n), 2), "device": model.device,
           "model": args.run, "ddragon": model_version}
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
