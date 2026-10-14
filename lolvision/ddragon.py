"""Data Dragon: версии, список чемпионов, иконки, сплэши — с кэшем на диске.

Зачем. Общий доступ к ассетам Riot для всех направлений: состав матча по
портретам HUD (иконки), имя над плашкой (закрытый словарь имён), облик
(сплэши и скины при необходимости). Кэш — data/ddragon/ (в Git не идёт):

    data/ddragon/versions.json
    data/ddragon/<версия>/champion.json
    data/ddragon/<версия>/item.json               # предметы (паспорт выбора, #807)
    data/ddragon/<версия>/champion/<Id>.json      # полные данные: скины
    data/ddragon/<версия>/img/champion/<Id>.png   # иконка 120×120
    data/ddragon/img/champion/splash/<Id>_<n>.jpg # сплэши от версии не зависят

Сеть. lolvision в сеть не ходит (tests/test_boundary.py), поэтому скачивание
— внешняя функция `fetch(url) -> bytes`: по умолчанию lolstream.ddragon_net
(через HTTPS_PROXY), в тестах — подмена. Без `fetch` модуль работает только
с тем, что уже лежит в кэше, и на пропуск отвечает FileNotFoundError с
подсказкой, чем докачать. Запись атомарная (временный файл + rename): в тот
же кэш могут писать несколько процессов сразу.

Патч в названии ролика («Patch 26.19») — это версия Data Dragon 16.19.x:
с 2025 года номер патча — год, а Data Dragon продолжил старую нумерацию.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

from lolvision.paths import DATA

CACHE = DATA / "ddragon"
BASE = "https://ddragon.leagueoflegends.com"
HINT = "докачать: python tools/ddragon.py --version <версия>"

Fetch = Callable[[str], bytes]


def patch_to_version(patch: str, versions: list[str]) -> str:
    """«26.19» → «16.19.1» (последняя сборка этого патча среди `versions`).

    Патч-год YY.N соответствует версии Data Dragon (YY-10).N. Уже версия
    Data Dragon («16.19» или «16.19.1») тоже принимается."""
    m = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", patch.strip())
    if not m:
        raise ValueError(f"не патч: {patch!r}")
    major, minor = int(m.group(1)), int(m.group(2))
    if major >= 25:
        major -= 10
    prefix = f"{major}.{minor}."
    for v in versions:                      # versions.json — от новых к старым
        if v.startswith(prefix):
            return v
    raise KeyError(f"нет версии Data Dragon для патча {patch} ({prefix}x)")


def norm_name(name: str) -> str:
    """Имя для сравнения: без регистра, пробелов, апострофов и точек.
    «Kai'Sa» → «kaisa», «Dr. Mundo» → «drmundo», «Nunu & Willump» → «nunuwillump»."""
    s = unicodedata.normalize("NFKD", name)
    return "".join(c for c in s.lower() if c.isalnum())


class DDragon:
    """Ассеты одной версии Data Dragon."""

    def __init__(self, version: str | None = None, root: Path = CACHE,
                 fetch: Fetch | None = None):
        self.root = Path(root)
        self.fetch = fetch
        self.version = version or self.versions()[0]

    # --- низкий уровень: файл из кэша или из сети ---

    def _get(self, rel: str, url: str) -> Path:
        path = self.root / rel
        if path.exists():
            return path
        if self.fetch is None:
            raise FileNotFoundError(f"{path} нет в кэше; {HINT}")
        data = self.fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Windows: соседний процесс как раз пишет или читает тот же файл.
            # Его копия та же самая — свою выбрасываем.
            tmp.unlink(missing_ok=True)
            if not path.exists():
                raise
        return path

    def _json(self, rel: str, url: str):
        return json.loads(self._get(rel, url).read_text(encoding="utf-8"))

    # --- версии и чемпионы ---

    def versions(self) -> list[str]:
        return self._json("versions.json", f"{BASE}/api/versions.json")

    def champions(self) -> dict[str, dict]:
        """Id → запись champion.json (поля id, key, name, title, tags, …)."""
        v = self.version
        return self._json(f"{v}/champion.json", f"{BASE}/cdn/{v}/data/en_US/champion.json")["data"]

    def names(self) -> dict[str, str]:
        """Id → отображаемое имя («MonkeyKing» → «Wukong»)."""
        return {cid: c["name"] for cid, c in self.champions().items()}

    def by_name(self, name: str) -> str | None:
        """Отображаемое имя или Id в любом регистре → Id; нет такого — None."""
        key = norm_name(name)
        for cid, c in self.champions().items():
            if key in (norm_name(cid), norm_name(c["name"])):
                return cid
        return None

    def champion_full(self, cid: str) -> dict:
        """Полная запись одного чемпиона: скины (num, name), умения, подсказки."""
        v = self.version
        return self._json(f"{v}/champion/{cid}.json",
                          f"{BASE}/cdn/{v}/data/en_US/champion/{cid}.json")["data"][cid]

    def items(self) -> dict[str, dict]:
        """Id предмета (строкой) → запись item.json (name, gold, into, depth,
        tags, maps, consumed, …) — для паспорта выбора (#807)."""
        v = self.version
        return self._json(f"{v}/item.json", f"{BASE}/cdn/{v}/data/en_US/item.json")["data"]

    def skins(self, cid: str) -> list[dict]:
        """[{id, num, name, chromas}, …]; num — номер сплэша."""
        return self.champion_full(cid)["skins"]

    # --- картинки ---

    def icon(self, cid: str) -> Path:
        """Квадратная иконка 120×120 (базовый облик)."""
        v = self.version
        return self._get(f"{v}/img/champion/{cid}.png", f"{BASE}/cdn/{v}/img/champion/{cid}.png")

    def splash(self, cid: str, num: int = 0) -> Path:
        """Сплэш скина `num` (0 — базовый), 1215×717."""
        return self._get(f"img/champion/splash/{cid}_{num}.jpg",
                         f"{BASE}/cdn/img/champion/splash/{cid}_{num}.jpg")

    def spell_icon(self, file: str) -> Path:
        """Иконка умения чемпиона по имени файла из `champion_full(...)["spells"]
        [i]["image"]["full"]` (64×64) — от облика не зависит (#843)."""
        v = self.version
        return self._get(f"{v}/img/spell/{file}", f"{BASE}/cdn/{v}/img/spell/{file}")

    def summoners(self) -> dict[str, dict]:
        """Id заклинания призывателя («SummonerSmite») → запись summoner.json."""
        v = self.version
        return self._json(f"{v}/summoner.json", f"{BASE}/cdn/{v}/data/en_US/summoner.json")["data"]

    def all_icons(self) -> dict[str, Path]:
        return {cid: self.icon(cid) for cid in sorted(self.champions())}


def load(patch_or_version: str | None = None, root: Path = CACHE,
         fetch: Fetch | None = None) -> DDragon:
    """DDragon для патча из названия ролика («26.19»), версии («16.19.1») или
    последней версии (None)."""
    if patch_or_version is None:
        return DDragon(None, root, fetch)
    probe = DDragon("0", root, fetch)
    return DDragon(patch_to_version(patch_or_version, probe.versions()), root, fetch)


# --- функции для миникарты (lolvision/minimap_icons.py, tools/*minimap*) ---
# Тот же кэш; версия по умолчанию — патч 26.19 роликов канала осени 2026.

DDRAGON = CACHE
DEFAULT_VERSION = "16.19.1"


def champions(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> list[str]:
    """Id чемпионов (как в именах файлов Data Dragon), по алфавиту."""
    return sorted(DDragon(version, root).champions())


def names(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, str]:
    """Id → отображаемое имя («MonkeyKing» → «Wukong»)."""
    return DDragon(version, root).names()


def items(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict[str, dict]:
    """Id предмета → запись item.json из кэша (FileNotFoundError, если нет)."""
    return DDragon(version, root).items()


def load_icons(version: str = DEFAULT_VERSION, root: Path = DDRAGON) -> dict:
    """Id → иконка RGB uint8 (квадратная, базовый облик)."""
    import numpy as np
    from PIL import Image
    dd = DDragon(version, root)
    out = {}
    for cid, path in dd.all_icons().items():
        with Image.open(path) as im:
            out[cid] = np.asarray(im.convert("RGB"))
    return out
