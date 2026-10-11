"""Иконки чемпионов из Data Dragon: загрузка в data/ddragon/<версия>/ и чтение.

Зачем. Иконка чемпиона на миникарте — базовая квадратная иконка Data Dragon
(без скинов), вписанная в кружок. Из этих иконок собирается синтетика для
детектора и классификатора миникарты (lolvision/minimap_icons.py).

Раскладка: data/ddragon/<версия>/champion.json и
data/ddragon/<версия>/img/champion/<Id>.png. Загрузка идемпотентна: файл,
который уже лежит, не перекачивается, запись — через временный файл и
переименование, поэтому параллельный загрузчик (другой агент) не увидит
недокачанную иконку.

Сеть — через прокси из HTTPS_PROXY (прямой доступ к Data Dragon рвётся).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from lolvision.paths import DATA

DDRAGON = DATA / "ddragon"
BASE_URL = "https://ddragon.leagueoflegends.com"
# Патч 26.19 (ролики канала осени 2026) — версия Data Dragon 16.19.1.
DEFAULT_VERSION = "16.19.1"


def _get(url: str, timeout: float = 30.0, tries: int = 4) -> bytes:
    """GET через прокси; соединение через прокси иногда рвётся — повторяем."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    handlers = [urllib.request.ProxyHandler({"https": proxy, "http": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    for k in range(tries):
        try:
            with opener.open(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
    raise AssertionError("недостижимо")


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def fetch(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> Path:
    """Скачать champion.json и иконки всех чемпионов; вернуть каталог версии."""
    vdir = root / version
    cj = vdir / "champion.json"
    if not cj.exists():
        _save(cj, _get(f"{BASE_URL}/cdn/{version}/data/en_US/champion.json"))
    for cid in champions(version, root):
        p = vdir / "img" / "champion" / f"{cid}.png"
        if not p.exists():
            _save(p, _get(f"{BASE_URL}/cdn/{version}/img/champion/{cid}.png"))
    return vdir


def champions(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> list[str]:
    """Id чемпионов (как в именах файлов Data Dragon), по алфавиту."""
    d = json.loads((root / version / "champion.json").read_text(encoding="utf-8"))
    return sorted(d["data"])


def names(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, str]:
    """Id → отображаемое имя («MonkeyKing» → «Wukong»)."""
    d = json.loads((root / version / "champion.json").read_text(encoding="utf-8"))
    return {k: v["name"] for k, v in d["data"].items()}


def load_icons(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, np.ndarray]:
    """Id → иконка RGB uint8 (120×120 у Data Dragon)."""
    out = {}
    for cid in champions(version, root):
        with Image.open(root / version / "img" / "champion" / f"{cid}.png") as im:
            out[cid] = np.asarray(im.convert("RGB"))
    return out
