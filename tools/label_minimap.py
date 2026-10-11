"""Листы для ручной разметки иконок миникарты и HUD: черновик от модели + картинка.

Зачем. Проверке (tools/eval_minimap.py) нужна разметка, сделанная глазами:
где иконки, чья команда, какой чемпион, кто мёртв по HUD. С нуля ставить
точки дорого, поэтому модель даёт черновик, а человек по листу правит его:
удаляет ложные, добавляет пропущенные, исправляет чемпиона. На листе —
миникарта ×4 с номерами детекций и сеткой через 32 px сетки 256 (чтобы
считывать координаты пропущенных иконок), справа — топ-3 чемпиона по
каждой детекции и вырезы портретов HUD (10 слотов).

Пример:
    python tools/label_minimap.py --frames runs/minimap/eval_frames.txt \
        --out runs/minimap/label_sheets
    # → <out>/<кадр>.png и <out>/draft.jsonl; правленая разметка —
    #   dataset/minimap_champions/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401
from lolvision import hud_panels
from lolvision.minimap_camera import minimap
from lolvision.minimap_icons import DEFAULT_RUN, TEAMS, IconModel
from lolvision.paths import DATASET_VIDEOS, DEFAULT_LAYOUT

K = 4
TEAM_RGB = {0: (60, 170, 255), 1: (255, 70, 70)}


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def sheet(frame: Image.Image, mm: np.ndarray, dets: list[dict], names: list[str]) -> Image.Image:
    big = Image.fromarray(mm.astype(np.uint8)).resize((256 * K, 256 * K), Image.NEAREST)
    W = Image.new("RGB", (256 * K + 520, 256 * K), (20, 20, 20))
    W.paste(big, (0, 0))
    d = ImageDraw.Draw(W)
    f, fs = font(18), font(13)
    for g in range(0, 257, 32):
        d.line([(g * K, 0), (g * K, 256 * K)], fill=(90, 90, 0))
        d.line([(0, g * K), (256 * K, g * K)], fill=(90, 90, 0))
        d.text((g * K + 2, 2), str(g), fill=(200, 200, 0), font=fs)
        d.text((2, g * K + 2), str(g), fill=(200, 200, 0), font=fs)
    for i, det in enumerate(dets):
        x, y = det["x"] * K, det["y"] * K
        col = TEAM_RGB[det["team"]]
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col, width=2)
        d.text((x + 30, y - 30), str(i), fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        top = np.argsort(-det["logp"])[:3]
        txt = (f"{i} {TEAMS[det['team']][0]} s{det['score']:.2f} "
               f"({det['x']:.0f},{det['y']:.0f}): ")
        txt += ", ".join(f"{names[c]} {np.exp(det['logp'][c]):.2f}" for c in top)
        d.text((256 * K + 8, 8 + 22 * i), txt, fill=col, font=fs)
    # Портреты HUD: слева синие, справа красные, по 5 сверху вниз.
    for s, box in enumerate(hud_panels.slot_boxes(frame.size)):
        p = frame.crop(box).resize((64, 80), Image.NEAREST)
        W.paste(p, (256 * K + 20 + (s // 5) * 120, 600 + (s % 5) * 84))
        d.text((256 * K + 88 + (s // 5) * 120, 630 + (s % 5) * 84), f"{'LR'[s // 5]}{s % 5 + 1}",
               fill=(220, 220, 220), font=fs)
    return W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True, help="файл со списком <id>/<кадр>.jpg")
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, help="cpu, когда видеокарта занята")
    args = ap.parse_args()
    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    model = IconModel(Path(args.run), args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "draft.jsonl").open("w", encoding="utf-8") as fd:
        for line in Path(args.frames).read_text(encoding="utf-8").split():
            vid, name = line.split("/")
            with Image.open(DATASET_VIDEOS / vid / "frames" / name) as im:
                frame = im.convert("RGB")
            mm = minimap(frame, layout)
            dets = model(mm.astype(np.uint8)[None])[0]
            stem = Path(name).stem
            sheet(frame, mm, dets, model.champions).save(out / f"{stem}.png")
            fd.write(json.dumps({"frame": stem, "video": vid, "icons": [
                {"x": round(d["x"], 1), "y": round(d["y"], 1), "team": TEAMS[d["team"]],
                 "champion": model.champions[int(np.argmax(d["logp"]))]} for d in dets]},
                ensure_ascii=False) + "\n")
    print("листы:", out)


if __name__ == "__main__":
    main()
