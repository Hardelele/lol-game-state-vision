"""Иконки чемпионов из Data Dragon: чтение из data/ddragon/<версия>/.

Зачем. Иконка чемпиона на миникарте и портрет в боковой панели HUD — базовая
квадратная иконка Data Dragon (без скинов). Из них собирается синтетика
для детектора и классификатора миникарты (lolvision/minimap_icons.py) и
узнаются портреты (lolvision/hud_panels.py).

Раскладка: data/ddragon/<версия>/champion.json и
data/ddragon/<версия>/img/champion/<Id>.png. Скачивает tools/fetch_ddragon.py
(сеть в lolvision не ходит — tests/test_boundary.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from lolvision.paths import DATA

DDRAGON = DATA / "ddragon"
# Патч 26.19 (ролики канала осени 2026) — версия Data Dragon 16.19.1.
DEFAULT_VERSION = "16.19.1"


def champions(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> list[str]:
    """Id чемпионов (как в именах файлов Data Dragon), по алфавиту."""
    d = json.loads((root / version / "champion.json").read_text(encoding="utf-8"))
    return sorted(d["data"])


def names(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, str]:
    """Id → отображаемое имя («MonkeyKing» → «Wukong»)."""
    d = json.loads((root / version / "champion.json").read_text(encoding="utf-8"))
    return {k: v["name"] for k, v in d["data"].items()}


def load_icons(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, np.ndarray]:
    """Id → иконка RGB uint8 (128×128 у Data Dragon 16.x)."""
    out = {}
    for cid in champions(version, root):
        with Image.open(root / version / "img" / "champion" / f"{cid}.png") as im:
            out[cid] = np.asarray(im.convert("RGB"))
    return out
