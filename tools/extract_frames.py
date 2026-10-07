"""Нарезка ролика на кадры с фиксированным шагом и запись манифеста.

Пример:
    python tools/extract_frames.py data/videos/58w57eJ5Qks.mp4 \
        --info data/videos/58w57eJ5Qks.info.json --every 15

Кадр берётся точным поиском по времени (`-ss` перед `-i`), поэтому подпись
`t` совпадает с моментом ролика независимо от ключевых кадров.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from paths import DATASET, ROOT

FIELDS = ["frame", "video_id", "t_sec", "timecode", "label", "note"]


def duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


def timecode(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def grab(video: Path, t: int, dest: Path, quality: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video),
         "-frames:v", "1", "-q:v", str(quality), str(dest)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--info", type=Path, help="info.json от yt-dlp — источник метаданных")
    ap.add_argument("--every", type=int, default=15, help="шаг в секундах")
    ap.add_argument("--quality", type=int, default=3, help="ffmpeg -q:v для JPEG (2 — лучше)")
    ap.add_argument("--out", type=Path, default=DATASET)
    args = ap.parse_args()

    video_id = args.video.stem
    info = json.loads(args.info.read_text(encoding="utf-8")) if args.info else {}
    length = duration(args.video)

    vdir = args.out / "videos" / video_id
    fdir = vdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)

    manifest = vdir / "frames.csv"
    old = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as f:
            old = {r["frame"]: r for r in csv.DictReader(f)}

    rows = []
    for t in range(0, int(length), args.every):
        name = f"{video_id}_t{t:05d}.jpg"
        dest = fdir / name
        if not dest.exists():
            grab(args.video, t, dest, args.quality)
        prev = old.get(f"frames/{name}", {})
        rows.append({
            "frame": f"frames/{name}",
            "video_id": video_id,
            "t_sec": t,
            "timecode": timecode(t),
            "label": prev.get("label", ""),
            "note": prev.get("note", ""),
        })

    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    source = {
        "video_id": video_id,
        "url": info.get("webpage_url", f"https://www.youtube.com/watch?v={video_id}"),
        "title": info.get("title"),
        "channel": info.get("channel"),
        "channel_url": info.get("channel_url"),
        "upload_date": info.get("upload_date"),
        "duration_sec": round(length, 3),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "format_id": info.get("format_id"),
        "sampling": {"every_sec": args.every, "first_sec": 0, "count": len(rows),
                     "seek": "ffmpeg -ss before -i (exact)", "jpeg_q": args.quality},
    }
    (vdir / "source.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{video_id}: {len(rows)} кадров → {fdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
