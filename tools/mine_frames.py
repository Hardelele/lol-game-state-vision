"""Умный сбор кадров из потока: сеть типа сцены смотрит каждый кадр, на диск
идут только нужные.

Зачем. build_not_game.py сохраняет каждый N-й кадр, а в стриме ~90% —
обычная игра, которой у нас и так много. Здесь поток читается так же (на
диск видео не пишется), своя сеть типа сцены (scene_net.py, ~1 мс на кадр)
оценивает каждый кадр, и сохраняются:

* редкие сцены (Tab, магазин, клиент, загрузка, не League) — не чаще
  раза в `--gap` секунд для одной и той же сцены;
* смены сцены — первый кадр новой сцены всегда;
* неуверенные кадры (вероятность ответа < `--unsure`) — не чаще раза в
  `--gap` секунд: на них учителю и разметке есть чему учиться;
* обычная игра — раз в `--game-every` секунд.

Формат — как у build_not_game.py (data/not_game/<id>/scene, full,
манифест <id>.csv), поэтому prelabel_scene.py, label_overlays.py и
обучение работают с этими кадрами без изменений. Причина сохранения и
ответ сети — в data/not_game/<id>.mined.csv.

Пример:
    python tools/mine_frames.py --url IUcEyWpbv8Y --fps 2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from build_not_game import FIELDS, FULL_SIZE, load_manifest, to_16x9
from lolstream.video_stream import frames, have_nvdec, probe_url
from lolvision.dataset_io import STORE_W
from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.paths import DEFAULT_LAYOUT, NOT_GAME_DATA, VIDEOS
from lolvision.scene_net import DEFAULT_NET, SceneNetRunner

RARE = ("tab", "shop", "client", "loading", "other")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--fps", type=float, default=2.0, help="сколько кадров в секунду смотреть")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--gap", type=float, default=3.0,
                    help="не чаще, с, для одной редкой сцены и для неуверенных")
    ap.add_argument("--game-every", type=float, default=60.0)
    ap.add_argument("--unsure", type=float, default=0.8)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--out", type=Path, default=NOT_GAME_DATA)
    args = ap.parse_args()

    url = args.url if "://" in args.url else f"https://www.youtube.com/watch?v={args.url}"
    w, h, dur, vid, info = probe_url(url)
    end = args.end or dur
    meta = VIDEOS / f"{vid}.info.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    if not meta.exists():
        meta.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    full16 = to_16x9(Image.new("RGB", (w, h))).size
    boxes = mask_boxes(full16, layout)
    cx0, cy0, cx1, cy1 = crop_box(full16, layout, "dense")
    store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
    sdir, fdir = args.out / vid / "scene", args.out / vid / "full"
    sdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    man_path = args.out / f"{vid}.csv"
    man = load_manifest(man_path)
    net = SceneNetRunner(args.net, "cuda")
    print(f"{vid}: {w}x{h}, {dur:.0f} с; смотрю {args.start:.0f}-{end:.0f} с по {args.fps} кадр/с, "
          f"распаковка {'NVDEC' if have_nvdec() else 'CPU'}")

    def save(t: float, fr: np.ndarray) -> dict:
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

    last_kept: dict[str, float] = {}
    prev = None
    log, seen = [], 0
    counts: dict[str, int] = {}
    for i, fr in enumerate(frames(url, w, h, args.fps, True, skip=args.start)):
        t = args.start + i / args.fps
        if t >= end:
            break
        seen += 1
        p = net.probs(fr)
        c = net.classes[int(p.argmax())]
        conf = float(p.max())
        why = None
        if prev is not None and c != prev:
            why = "change"
        elif conf < args.unsure and t - last_kept.get("unsure", -1e9) >= args.gap:
            why = "unsure"
        elif c in RARE and t - last_kept.get(c, -1e9) >= args.gap:
            why = "rare"
        elif c == "game" and t - last_kept.get("game", -1e9) >= args.game_every:
            why = "game"
        prev = c
        if why is None:
            continue
        last_kept["unsure" if why == "unsure" else c] = t
        r = save(t, fr)
        man[r["t_sec"]] = r
        counts[c] = counts.get(c, 0) + 1
        log.append({"t_sec": round(t, 3), "why": why, "class": c, "conf": round(conf, 3),
                    **{f"p_{n}": round(float(v), 3) for n, v in zip(net.classes, p)}})
    with man_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(man[k] for k in sorted(man))
    if log:
        with (args.out / f"{vid}.mined.csv").open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0]))
            wr.writeheader()
            wr.writerows(log)
    print(f"просмотрено {seen} кадров, сохранено {len(log)} ({len(log) / max(1, seen):.1%}): "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
