"""Докачать ассеты Data Dragon в кэш data/ddragon/ (идемпотентно).

Зачем. Иконки чемпионов нужны для состава матча по портретам HUD, список
имён — для имени над плашкой. Уже скачанное не перекачивается; качать в
тот же кэш могут несколько процессов сразу.

Пример:
    python tools/ddragon.py --patch 26.19            # champion.json + все иконки
    python tools/ddragon.py --version 16.19.1 --full # + полные записи (скины)
    python tools/ddragon.py --patch 26.19 --splash Ahri:0 Ahri:1
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from lolstream.ddragon_net import fetch
from lolvision import ddragon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patch", help="патч из названия ролика, например 26.19")
    ap.add_argument("--version", help="версия Data Dragon, например 16.19.1")
    ap.add_argument("--full", action="store_true", help="полные записи чемпионов (скины)")
    ap.add_argument("--splash", nargs="*", default=[], help="Id:номер сплэша")
    args = ap.parse_args()

    dd = ddragon.load(args.version or args.patch, fetch=fetch)
    champs = dd.champions()
    print(f"версия {dd.version}: {len(champs)} чемпионов")
    icons = dd.all_icons()
    print(f"иконок: {len(icons)} в {dd.root / dd.version / 'img' / 'champion'}")
    if args.full:
        for cid in sorted(champs):
            dd.champion_full(cid)
        print("полные записи: готово")
    for item in args.splash:
        cid, _, num = item.partition(":")
        print(dd.splash(cid, int(num or 0)))


if __name__ == "__main__":
    main()
