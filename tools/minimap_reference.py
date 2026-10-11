"""Эталон миникарты для поиска её в кадре (lolvision/minimap_locate.py).

Медиана чистых фонов миникарты (tools/minimap_backgrounds.py) сначала по
каждому ролику, потом по роликам: так ролик с большим числом отрезков не
перетягивает эталон на свой стиль (туман войны у игрока, открытая карта у
наблюдателя). Все фоны сняты в обрезке стандартной раскладки, и проверка
показала у всех масштаб 1.00 и сдвиг 0 — эталон в той же сетке.

Пример:
    python tools/minimap_reference.py    # → dataset/layouts/minimap_reference.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision.minimap_locate import REFERENCE
from lolvision.paths import DATA

BACKGROUNDS = DATA / "minimap" / "backgrounds.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backgrounds", type=Path, default=BACKGROUNDS)
    ap.add_argument("--out", type=Path, default=REFERENCE)
    args = ap.parse_args()

    z = np.load(args.backgrounds, allow_pickle=True)
    by: dict[str, list[np.ndarray]] = defaultdict(list)
    for bg, src in zip(z["bg"], z["src"]):
        by[str(src).rsplit(":", 1)[0]].append(bg)
    per_video = [np.median(np.stack(v), 0) for v in by.values()]
    ref = np.median(np.stack(per_video), 0).round().astype(np.uint8)
    Image.fromarray(ref).save(args.out)
    print(f"{args.out}: {len(per_video)} роликов, {sum(map(len, by.values()))} фонов")


if __name__ == "__main__":
    main()
