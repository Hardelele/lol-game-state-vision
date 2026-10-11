"""Собрать с роликов наблюдателя всё для «кто это»: портреты HUD, плашки, вырезы.

Зачем. Задача #760: состав матча по HUD, имя над плашкой и облик под ней
учатся и проверяются на одних и тех же данных. Ролик читается один раз
(ffmpeg, NVDEC), с каждого кадра снимаются:
* корреляции 20 портретов HUD со всеми иконками Data Dragon (lolvision/roster.py);
* плашки (lolvision/health_bars.py): место, команда, ХП, уровень;
* вырезы над плашкой (имя) и под ней (облик) — lolvision/identity.py.

Пишет data/identity/harvest/<id>.npz. Готовый ролик пропускается.

Пример:
    python tools/identity_harvest.py 58w57eJ5Qks olmTXkkUv58 --fps 1
    python tools/identity_harvest.py --all --fps 1      # все скачанные ролики
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from lolstream.video_stream import frames as video_frames
from lolvision import ddragon, health_bars as hb, identity, roster
from lolvision.paths import DATA, VIDEOS
from read_bars import LAYOUT, scene_keep

IDENTITY = DATA / "identity"
OWN_VIDEOS = IDENTITY / "videos"
HARVEST = IDENTITY / "harvest"


def find_video(vid: str) -> Path:
    for d in (VIDEOS, OWN_VIDEOS):
        for ext in (".mp4", ".webm"):
            p = d / f"{vid}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"нет ролика {vid} ни в {VIDEOS}, ни в {OWN_VIDEOS}")


def video_title(vid: str) -> str:
    p = VIDEOS / f"{vid}.info.json"
    if p.exists():  # сведения от yt-dlp лежат у общих роликов
        return json.loads(p.read_text(encoding="utf-8")).get("title", "")
    return ""


def harvest(vid: str, fps: float, bank: roster.IconBank, ids: list[str]) -> dict:
    tmpl, mask = hb.load_template()
    levels = hb.LevelReader()
    keep = scene_keep(json.loads(LAYOUT.read_text(encoding="utf-8")))
    ts, slots, bars, names, bodies = [], [], [], [], []
    t_bars = t_slots = 0.0
    skipped = 0
    t0 = time.perf_counter()
    for i, rgb in enumerate(video_frames(find_video(vid), 1920, 1080, fps)):
        ts.append(i / fps)
        t = time.perf_counter()
        slots.append(roster.read_frame(rgb, bank).scores.astype(np.float16))
        t_slots += time.perf_counter() - t
        t = time.perf_counter()
        try:
            found = hb.find_bars(rgb, tmpl, mask, keep, levels=levels)
        except cv2.error as e:
            # Машину делят несколько процессов: кадр без памяти пропускаем,
            # а не теряем весь ролик (пропуск попадает в счётчик).
            print(f"{vid}: кадр {i} пропущен: {str(e).splitlines()[-1][:80]}", flush=True)
            skipped += 1
            found = []
        t_bars += time.perf_counter() - t
        for b in found:
            bars.append((i, b.x, b.y, b.score, roster.TEAMS.index(b.team) if b.team else -1,
                         b.hp, b.resource, b.level, b.level_sim))
            names.append(identity.name_crop(rgb, b.x, b.y))
            bodies.append(identity.body_crop(rgb, b.x, b.y))
        if i % 300 == 0:
            print(f"{vid}: кадр {i}, плашек {len(bars)}, {time.perf_counter() - t0:.0f} с",
                  flush=True)
    n = len(ts)
    dt = np.dtype([("frame", "i4"), ("x", "i2"), ("y", "i2"), ("score", "f4"), ("team", "i1"),
                   ("hp", "f4"), ("resource", "f4"), ("level", "i1"), ("level_sim", "f4")])
    HARVEST.mkdir(parents=True, exist_ok=True)
    out = HARVEST / f"{vid}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, t=np.array(ts, np.float32), slots=np.stack(slots), ids=np.array(ids),
             bars=np.array(bars, dt),
             names=np.stack(names) if names else np.zeros((0, 19, 147, 3), np.uint8),
             bodies=np.stack(bodies) if bodies else np.zeros((0, 128, 128, 3), np.uint8),
             fps=fps, title=video_title(vid))
    tmp.replace(out)
    return {"video": vid, "frames": n, "bars": len(bars), "skipped_frames": skipped,
            "ms_slots": round(t_slots / max(n, 1) * 1000, 1),
            "ms_bars": round(t_bars / max(n, 1) * 1000, 1),
            "sec_total": round(time.perf_counter() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true", help="все ролики в data/videos и data/identity/videos")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--version", default="16.19.1", help="версия Data Dragon")
    args = ap.parse_args()
    vids = list(args.videos)
    if args.all:
        vids += sorted({p.stem for d in (VIDEOS, OWN_VIDEOS) for p in d.iterdir()
                       if p.suffix in (".mp4", ".webm")})
    dd = ddragon.DDragon(args.version)
    bank = roster.IconBank.from_ddragon(dd)
    for vid in dict.fromkeys(vids):
        if (HARVEST / f"{vid}.npz").exists():
            print(f"{vid}: уже собран")
            continue
        print(json.dumps(harvest(vid, args.fps, bank, bank.ids), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
