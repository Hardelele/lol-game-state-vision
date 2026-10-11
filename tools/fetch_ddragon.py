"""Скачать champion.json и иконки чемпионов Data Dragon в data/ddragon/<версия>/.

Загрузка идемпотентна: файл, который уже лежит, не перекачивается; запись —
через временный файл и переименование, поэтому параллельный загрузчик не
увидит недокачанную иконку. Сеть — через прокси из HTTPS_PROXY (прямой
доступ к Data Dragon рвётся), соединение повторяется при обрыве.

Пример:
    HTTPS_PROXY=http://127.0.0.1:3128 python tools/fetch_ddragon.py 16.19.1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from lolvision.ddragon import DDRAGON, DEFAULT_VERSION

BASE_URL = "https://ddragon.leagueoflegends.com"


def get(url: str, timeout: float = 30.0, tries: int = 4) -> bytes:
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


def save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("version", nargs="?", default=DEFAULT_VERSION)
    args = ap.parse_args()
    vdir = DDRAGON / args.version
    cj = vdir / "champion.json"
    if not cj.exists():
        save(cj, get(f"{BASE_URL}/cdn/{args.version}/data/en_US/champion.json"))
    ids = sorted(json.loads(cj.read_text(encoding="utf-8"))["data"])
    for cid in ids:
        p = vdir / "img" / "champion" / f"{cid}.png"
        if not p.exists():
            save(p, get(f"{BASE_URL}/cdn/{args.version}/img/champion/{cid}.png"))
    print(len(ids), "чемпионов →", vdir)


if __name__ == "__main__":
    main()
