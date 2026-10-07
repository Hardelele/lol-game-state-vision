"""Контактные листы для разметки: миникарта + сцена для каждого кадра.

Метка ставится по рамке камеры на миникарте (методика из dataset/README.md),
а сцена нужна, чтобы отличить игру от заставок и чёрных кадров. Лист собирает
десяток кадров на одно изображение, поэтому размечать можно подряд и видеть
соседние моменты рядом — это заметно снижает разнобой на границах.

Миникарта берётся из поля `labelling.minimap` раскладки. В сам вход модели
она не попадает: там она закрыта маской.

Пример:
    python tools/label_sheets.py dataset/videos/58w57eJ5Qks \
        --layout dataset/layouts/spectator-volibear-challenger.json 
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paths import SHEETS


CELL_MAP = 300       # сторона миникарты в ячейке
CELL_SCENE_W = 300   # ширина превью сцены
PAD = 10
TEXT_H = 22
DARK_MEAN = 12.0     # средняя яркость 0..255, ниже которой кадр считается чёрным


def load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def minimap_box(layout: dict) -> tuple[float, float, float, float]:
    box = layout.get("labelling", {}).get("minimap")
    if not box:
        raise ValueError(
            f"В раскладке {layout['name']} нет labelling.minimap — "
            "добавьте долевые координаты миникарты")
    return tuple(box)


def is_dark(img: Image.Image) -> bool:
    """Кадр без изображения: чёрный экран между сценами, заставка перед игрой."""
    return float(np.asarray(img.convert("L"), dtype=np.float32).mean()) < DARK_MEAN


def select_rows(rows: list[dict], select: Path | None, video_id: str) -> list[dict]:
    """Оставить только кадры, перечисленные в CSV (колонки video_id и frame).

    Нужно, чтобы собрать лист из ошибок модели и разобрать их глазами, не
    пересматривая весь ролик.
    """
    if select is None:
        return rows
    wanted = {
        r["frame"] for r in csv.DictReader(select.open(encoding="utf-8"))
        if r.get("video_id", video_id) == video_id
    }
    picked = [r for r in rows if Path(r["frame"]).stem in wanted]
    if not picked:
        raise ValueError(f"в {select} нет кадров ролика {video_id}")
    return picked


def build_sheets(
    video_dir: Path, layout: dict, out: Path, per_sheet: int = 12, cols: int = 4,
    select: Path | None = None
) -> list[Path]:
    rows = list(csv.DictReader((video_dir / "frames.csv").open(encoding="utf-8")))
    total = len(rows)
    rows = select_rows(rows, select, video_dir.name)
    if len(rows) != total:
        print(f"выбрано {len(rows)} кадров из {total}")
    mx0, my0, mx1, my1 = minimap_box(layout)
    font = load_font(16)
    font_big = load_font(20)
    out.mkdir(parents=True, exist_ok=True)

    cell_h = TEXT_H + CELL_MAP + PAD + int(CELL_SCENE_W * 9 / 16)
    cell_w = max(CELL_MAP, CELL_SCENE_W)
    paths, dark = [], []

    for start in range(0, len(rows), per_sheet):
        chunk = rows[start : start + per_sheet]
        n_rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new(
            "RGB",
            (cols * (cell_w + PAD) + PAD, n_rows * (cell_h + PAD) + PAD + 28),
            (24, 24, 28))
        draw = ImageDraw.Draw(sheet)
        title = f"{video_dir.name}  кадры {start + 1}–{start + len(chunk)} из {len(rows)}"
        draw.text((PAD, 6), title, fill=(235, 235, 235), font=font_big)

        for i, row in enumerate(chunk):
            with Image.open(video_dir / row["frame"]) as img:
                w, h = img.size
                mini = img.crop((int(mx0 * w), int(my0 * h),
                                 int(mx1 * w), int(my1 * h))).resize(
                    (CELL_MAP, CELL_MAP), Image.LANCZOS)
                scene = img.resize(
                    (CELL_SCENE_W, int(CELL_SCENE_W * h / w)), Image.BILINEAR)
                if is_dark(img):
                    dark.append(row["frame"])

            cx = PAD + (i % cols) * (cell_w + PAD)
            cy = 28 + PAD + (i // cols) * (cell_h + PAD)
            mark = " ЧЁРНЫЙ" if row["frame"] in dark else ""
            if row.get("label"):
                mark += f"  [{row['label']}]"
            draw.text((cx, cy), f"#{start + i + 1}  t={row['t_sec']}с  "
                                f"{row['timecode']}{mark}",
                      fill=(255, 215, 120), font=font)
            sheet.paste(mini, (cx, cy + TEXT_H))
            sheet.paste(scene, (cx, cy + TEXT_H + CELL_MAP + PAD))
            draw.rectangle(
                [cx - 2, cy + TEXT_H - 2, cx + CELL_MAP + 1, cy + TEXT_H + CELL_MAP + 1],
                outline=(90, 90, 100))

        path = out / f"{video_dir.name}_sheet{start // per_sheet + 1:02d}.png"
        sheet.save(path)
        paths.append(path)

    if dark:
        print(f"чёрных кадров: {len(dark)} → {', '.join(Path(d).stem for d in dark)}")
        print("  они не содержат игры: ставьте им unknown")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="по умолчанию runs/sheets/<ролик>")
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--select", type=Path, default=None,
                    help="CSV с колонками video_id и frame: собрать лист только по ним")
    args = ap.parse_args()

    args.out = args.out or SHEETS / args.video_dir.name
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    paths = build_sheets(args.video_dir, layout, args.out, args.per_sheet,
                         select=args.select)
    print(f"{len(paths)} листов → {args.out}")


if __name__ == "__main__":
    main()
