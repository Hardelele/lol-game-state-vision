"""Закраска HUD и миникарты по раскладке с сохранением кадров в PNG.

Здесь же — обрезка кадра по незакрытой области раскладки (`crop_box`): её
одинаково используют сборщик датасета и live-режим.

Пример:
    python tools/mask_frames.py dataset/layouts/spectator-volibear-challenger.json \
        dataset/videos/58w57eJ5Qks/frames/58w57eJ5Qks_t00300.jpg --out runs/checks/masks/58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def mask_boxes(size: tuple[int, int], layout: dict) -> list[tuple[int, int, int, int]]:
    """Пиксельные границы: включаем левый/верхний край, исключаем правый/нижний."""
    w, h = size
    reference = layout.get("reference_size")
    if reference:
        rw, rh = reference
        if abs(w / h - rw / rh) > 0.001:
            raise ValueError(
                f"Раскладка рассчитана на {rw}:{rh}, получен кадр {w}×{h}. "
                "Выберите раскладку для этого формата."
            )

    boxes = []
    for name, coords in layout["mask"].items():
        if len(coords) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in coords
        ):
            raise ValueError(f"Некорректный прямоугольник {name}: {coords}")
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Прямоугольник {name} должен быть в пределах [0, 1]")
        boxes.append((math.floor(x0 * w), math.floor(y0 * h),
                      math.ceil(x1 * w), math.ceil(y1 * h)))
    return boxes


def apply_mask(img: Image.Image, layout: dict, fill=(0, 0, 0)) -> Image.Image:
    out = img.convert("RGB").copy()
    for box in mask_boxes(out.size, layout):
        out.paste(fill, box)
    return out


def keep_grid(size: tuple[int, int], layout: dict) -> np.ndarray:
    """Булева карта кадра: True там, где пиксель не закрыт маской."""
    w, h = size
    keep = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in mask_boxes(size, layout):
        keep[y0:y1, x0:x1] = False
    return keep


def unmasked_bbox(size: tuple[int, int], layout: dict) -> tuple[int, int, int, int]:
    """Границы области, где хоть один пиксель не закрыт маской.

    Обрезка одинакова для всех кадров ролика, поэтому не может служить
    подсказкой о фазе матча. Внутри области остаются закрытые прямоугольники
    (чат, объявления, лента убийств) — они постоянны и информации не несут.
    """
    keep = keep_grid(size, layout)
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Маска закрывает кадр целиком — нечего подавать модели")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def dense_bbox(size: tuple[int, int], layout: dict, rel: float = 0.6
               ) -> tuple[int, int, int, int]:
    """Границы плотной середины кадра: строки и столбцы, открытые не хуже,
    чем `rel` от самой открытой строки/столбца этой раскладки.

    Обрезка по `unmasked_bbox` сохраняет всю ширину кадра: достаточно пары
    открытых уголков над боковыми панелями, чтобы строка попала в границы.
    В итоге большую часть входа занимают постоянные чёрные поля, а рельеф
    сжимается в несколько раз сильнее нужного. Порог берётся относительным,
    потому что абсолютная доля открытых пикселей зависит от раскладки:
    при плотном HUD открытых строк выше 60% может не быть вообще.
    """
    keep = keep_grid(size, layout)
    rows_p, cols_p = keep.mean(axis=1), keep.mean(axis=0)
    rows = np.flatnonzero(rows_p >= rel * rows_p.max())
    cols = np.flatnonzero(cols_p >= rel * cols_p.max())
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Маска не оставляет плотной области")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_box(size: tuple[int, int], layout: dict, mode: str = "dense"
             ) -> tuple[int, int, int, int]:
    """Прямоугольник обрезки кадра: `dense`, `bbox` или `none` (весь кадр)."""
    if mode == "bbox":
        return unmasked_bbox(size, layout)
    if mode == "dense":
        return dense_bbox(size, layout)
    if mode == "none":
        return (0, 0, size[0], size[1])
    raise ValueError(f"неизвестный режим обрезки {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("layout", type=Path)
    ap.add_argument("frames", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        with Image.open(frame) as img:
            # PNG сохраняет чёрные области точно, без артефактов JPEG на границах.
            apply_mask(img, layout).save(args.out / f"{frame.stem}.png")
    print(f"{len(args.frames)} кадров → {args.out}")


if __name__ == "__main__":
    main()
