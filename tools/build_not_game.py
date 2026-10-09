"""Кадры для классификатора «игра / не игра» из ролика на YouTube.

Зачем. Патчевая модель (patch_model.py) училась только на игровых кадрах
наблюдателя, и её голова «сцена» на кадре, где интерфейс занимает весь
экран, говорит P(сцена) ≈ 0.76: сказать «это не игра» ей нечем. Нужны
кадры, где League of Legends нет вовсе — вебкамера стримера, клиент с
выбором чемпионов, экран загрузки, заставки, другие игры, — и игровые
кадры с чужим HUD (вид игрока, а не наблюдателя).

Как. Ролик читается потоком, как в build_coords.py (на диск видео не
пишется), из каждого отобранного кадра сохраняются:

* `scene/` — то, что получила бы модель в live: раскладка
  spectator-volibear-challenger закрашивает HUD, кадр обрезается по плотной
  области и хранится шириной 720 px. Раскладка та же, что у датасета
  координат, даже если HUD в ролике другой: именно так кадр чужого
  стрима пройдёт через live;
* `full/` — весь кадр 960×540 без маски: по нему размечают глазами и из
  него делают вариант входа «без маски».

Разметка не здесь: интервалы времени с метками лежат в
dataset/not_game/<id>/labels.csv и подклеиваются при обучении
(train_not_game.py), так что правка разметки не требует пересборки кадров.
Повторный запуск с другим интервалом дописывает кадры в тот же манифест
data/not_game/<id>.csv.

Пример:
    python tools/build_not_game.py --url Pa8fH6tfZB0 --fps 1
    python tools/build_not_game.py --url IgJCPlXnArU --fps 1 --start 300 --end 600
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

from paths import DATA, DATASET
from mask_frames import crop_box, mask_boxes
from build_coords import probe_url, frames, have_nvdec, STORE_W

NOT_GAME_DATA = DATA / "not_game"
NOT_GAME_LABELS = DATASET / "not_game"
LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
FULL_SIZE = (960, 540)
FIELDS = ["video_id", "t_sec", "scene", "full"]


def to_16x9(img: Image.Image) -> Image.Image:
    """Кадр другого формата — в 16:9 с чёрными полями, как в плеере."""
    w, h = img.size
    if abs(w / h - 16 / 9) < 0.01:
        return img
    tw, th = (w, round(w * 9 / 16)) if w / h > 16 / 9 else (round(h * 16 / 9), h)
    out = Image.new("RGB", (tw, th))
    out.paste(img, ((tw - w) // 2, (th - h) // 2))
    return out


def load_manifest(path: Path) -> dict[float, dict]:
    if not path.exists():
        return {}
    return {float(r["t_sec"]): r for r in csv.DictReader(path.open(encoding="utf-8"))}


def load_labels(vid: str, root: Path = NOT_GAME_LABELS) -> list[dict]:
    """Интервалы разметки ролика: start_sec, end_sec, label, kind, note."""
    p = root / vid / "labels.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    for r in rows:
        r["start_sec"], r["end_sec"] = float(r["start_sec"]), float(r["end_sec"])
    return rows


def label_at(intervals: list[dict], t: float) -> tuple[str, str]:
    """Метка и подкласс для момента t; вне разметки — ("", "")."""
    for r in intervals:
        if r["start_sec"] <= t < r["end_sec"]:
            return r["label"], r["kind"]
    return "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True, help="ссылка или идентификатор ролика")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0, help="с какой секунды")
    ap.add_argument("--end", type=float, default=0.0, help="до какой секунды (0 — до конца)")
    ap.add_argument("--out", type=Path, default=NOT_GAME_DATA)
    ap.add_argument("--decode", choices=("auto", "gpu", "cpu"), default="auto")
    args = ap.parse_args()

    url = args.url if "://" in args.url else f"https://www.youtube.com/watch?v={args.url}"
    w, h, dur, vid, info = probe_url(url)
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    end = args.end or dur
    print(f"{vid}: {w}x{h}, {dur:.0f} с; беру {args.start:.0f}-{end:.0f} с, "
          f"{args.fps} кадр/с; читаю потоком, на диск видео не пишется")
    meta = DATA / "videos" / f"{vid}.info.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    if not meta.exists():
        meta.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    full16 = to_16x9(Image.new("RGB", (w, h))).size
    boxes = mask_boxes(full16, layout)
    cx0, cy0, cx1, cy1 = crop_box(full16, layout, "dense")
    store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
    sdir, fdir = args.out / vid / "scene", args.out / vid / "full"
    sdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    man_path = args.out / f"{vid}.csv"
    man = load_manifest(man_path)
    gpu = args.decode == "gpu" or (args.decode == "auto" and have_nvdec())
    print(f"  распаковка: {'видеокарта (NVDEC)' if gpu else 'процессор'}")

    def handle(t: float, fr: np.ndarray) -> dict:
        img = to_16x9(Image.fromarray(fr))
        name = f"{int(round(t * 1000)):08d}.jpg"
        img.resize(FULL_SIZE, Image.LANCZOS).save(fdir / name, quality=85)
        scene = img.copy()
        for b in boxes:
            scene.paste((0, 0, 0), b)
        scene.crop((cx0, cy0, cx1, cy1)).resize((STORE_W, store_h), Image.LANCZOS).save(
            sdir / name, quality=88)
        return {"video_id": vid, "t_sec": round(t, 3),
                "scene": f"{vid}/scene/{name}", "full": f"{vid}/full/{name}"}

    pend: deque = deque()
    n = 0
    with ThreadPoolExecutor(max(2, (os.cpu_count() or 8) // 4)) as pool:
        for i, fr in enumerate(frames(url, w, h, args.fps, True, args.decode,
                                      skip=args.start)):
            t = args.start + i / args.fps
            if t >= end:
                break
            pend.append(pool.submit(handle, t, fr.copy()))
            while len(pend) > 12:
                r = pend.popleft().result()
                man[r["t_sec"]] = r
                n += 1
        while pend:
            r = pend.popleft().result()
            man[r["t_sec"]] = r
            n += 1
    with man_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(man[k] for k in sorted(man))
    print(f"{n} кадров, всего в манифесте {len(man)} → {man_path}")


if __name__ == "__main__":
    main()
