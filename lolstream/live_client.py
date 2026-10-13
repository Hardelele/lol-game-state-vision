"""Live Client Data API своей игры: время, состав, смерти, события — источник `api`.

Что это. Клиент League of Legends во время матча поднимает на loopback
официальный HTTPS-сервис `https://127.0.0.1:2999/liveclientdata/` с
самоподписанным сертификатом. `allgamedata` отдаёт всё сразу: время матча и
режим (`gameData`), десятерых игроков (`allPlayers`: чемпион, команда
ORDER/CHAOS, уровень, предметы, счёт, `isDead`, `respawnTimer`), своего
игрока (`activePlayer`: статы, золото, riotId) и события (`events.Events`
с `EventID`). Кулдаунов и позиций нет (docs/pitfalls.md, 0.8 и 5.1).

Состояния — явные, а не исключения (`State`):

* `no_game` — порт закрыт или не отвечает, игры не было;
* `loading` — порт отвечает, но игра ещё не идёт. Замечено в журнале своей
  игры: полный ответ 200 с составом, `gameTime` стоит на ~0,07 с, событий
  нет (ни одного, даже GameStart) — 5 с до старта; снимок с составом при этом
  отдаётся (`game`). ДОПУЩЕНИЕ (документацией Riot не описано, не поймано):
  раньше на экране загрузки — HTTP-ошибка в формате Riot (`{"errorCode",
  "httpStatus", "message"}`) или 200 с пустым `allPlayers`; оба тоже
  считаются загрузкой;
* `in_game` — разобранный снимок игры;
* `ended` — пришло событие `GameEnd` или API пропал после игры дольше
  `lost_after` секунд (короче — `error`, чтобы сбой одного опроса не
  объявлял конец матча);
* `error` — ответ есть, но не разбирается (не JSON, нет полей, HTTP-ошибка
  посреди игры): причина в `reason`, снимка нет.

События не повторяются: `EventID` запоминаются в пределах одной игры; новая
игра (время матча пошло назад) начинает счёт заново.

Чемпион — Id Data Dragon («MonkeyKing», «Fiddlesticks»), а не локализованное
`championName` (у русского клиента — «Зилеан»): берётся из
`rawChampionName` («game_character_displayname_Zilean») и сверяется со
списком Id Data Dragon без учёта регистра (внутреннее имя бывает
«FiddleSticks»). Не нашлось или списка нет — `champion = None` и причина.

Заклинания призывателя (#804) — из `summonerSpells` игрока
(`summonerSpellOne` / `summonerSpellTwo` — слоты 1 и 2, по умолчанию клавиши
D и F): id берётся из `rawDisplayName` («GeneratedTip_SummonerSpell_
SummonerTeleport_DisplayName» → «SummonerTeleport»), а не из локализованного
`displayName` («Телепорт»). Сезонная приставка срезается
(«S12_SummonerTeleportUpgrade» → «SummonerTeleportUpgrade»), семья — по
приставке id (`SPELL_FAMILIES`: улучшенный Телепорт — семья Телепорт, все
Кары — семья Кара). Поля нет (старые сборки, обезличенные журналы) или формат
незнакомый — `spells = None` и причина, разбор игры не падает.

Роль и руны (#807). `position` (TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY)
заполнен в ранкеде и драфте; в тренировке — «NONE», в других режимах бывает
пустым — тогда `PlayerRecord.role = None` и `role_reason`. `runes` —
`{keystone, primaryRuneTree, secondaryRuneTree}` с числовым `id` и
`rawDisplayName`: имя ключевой руны берётся из «perk_displayname_<Имя>»
(«ArcaneComet»), у веток ключ числовой («perkstyle_displayname_7202»),
поэтому имя ветки — по id (`RUNE_TREES`); числовое имя у ключевой руны —
тоже по id (`KEYSTONES`, Data Dragon runesReforged 16.19.1). Нет поля или
id — `runes = None` и `runes_reason`.

Чтение — `LiveClient` в своём потоке (~2 Гц), наружу последний `LiveStatus`
и очередь новых событий (и наблюдатель `on_status` на каждый опрос);
ошибки транспорта и разбора не выходят наружу.
Разбор и состояния — `Tracker` без потока и сети, на нём же воспроизведение
журнала `{"t", "code", "body"}` (`read_journal`, `replay`).
"""

from __future__ import annotations

import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lolvision.ddragon import norm_name
# Порог «новой игры» один на трекер API и контекст матча.
from lolvision.match_state import NEW_GAME_BACKSTEP, TEAMS, ApiObs, ApiPlayer

HOST = "127.0.0.1"
PORT = 2999
BASE_URL = f"https://{HOST}:{PORT}/liveclientdata/"
# Форматы rawChampionName, в которых виден внутренний Id чемпиона. Обычный —
# «game_character_displayname_Xerath»; у части чемпионов клиент отдаёт ключ
# старого вида «Character_Aatrox_Name» (журнал своей игры 2026-10-12: Aatrox
# среди девяти обычных). Любой другой формат — отказ с причиной.
RAW_CHAMPION_FORMATS = (re.compile(r"game_character_displayname_(\w+)"),
                        re.compile(r"Character_(\w+?)_Name"))
SOURCE = "api"
# Ключ заклинания призывателя: «GeneratedTip_SummonerSpell_<id>_DisplayName»
# (в rawDescription — «…_<id>_Description»). Журналы своей игры 2026-10-12.
RAW_SPELL_FORMAT = re.compile(r"GeneratedTip_SummonerSpell_(\w+?)_(?:DisplayName|Description)")
# Сезонная приставка у вариантов: «S12_SummonerTeleportUpgrade» (Телепорт после
# 10-й минуты), «S5_SummonerSmitePlayerGanker» (Кара после роста питомца).
SPELL_SEASON = re.compile(r"S\d+_(\w+)")
# Семьи вариантов: id варианта начинается с id семьи. Телепорт: SummonerTeleport →
# SummonerTeleportUpgrade; Кара: SummonerSmite → SummonerSmitePlayerGanker →
# SummonerSmiteAvatarOffensive / Defensive (и, по той же приставке, прочие
# «первобытные»). Остальные заклинания — сами себе семья.
SPELL_FAMILIES = ("SummonerTeleport", "SummonerSmite")
SPELL_SLOTS = ("summonerSpellOne", "summonerSpellTwo")
# Роли allPlayers[].position (ранкед / драфт). «NONE» в тренировке, "" — в
# режимах без ролей: роли нет.
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
# Ветки рун по id (Data Dragon runesReforged, ключи стабильны с 2018 года).
RUNE_TREES = {8000: "Precision", 8100: "Domination", 8200: "Sorcery",
              8300: "Inspiration", 8400: "Resolve"}
# Ключевые руны по id — запасное имя, если rawDisplayName числовой или его нет
# (runesReforged 16.19.1, первый ряд каждой ветки).
KEYSTONES = {8005: "PressTheAttack", 8008: "LethalTempo", 8021: "FleetFootwork",
             8010: "Conqueror", 8112: "Electrocute", 8128: "DarkHarvest",
             9923: "HailOfBlades", 8214: "SummonAery", 8229: "ArcaneComet",
             8230: "PhaseRush", 8992: "DeathfireTouch", 8351: "GlacialAugment",
             8360: "UnsealedSpellbook", 8369: "FirstStrike", 8437: "GraspOfTheUndying",
             8439: "Aftershock", 8465: "Guardian"}
RAW_KEYSTONE = re.compile(r"perk_displayname_(\w+)")
RUNE_SLOTS = (("keystone", "keystone"), ("primary", "primaryRuneTree"),
              ("secondary", "secondaryRuneTree"))


class State(str, Enum):
    NO_GAME = "no_game"
    LOADING = "loading"
    IN_GAME = "in_game"
    ENDED = "ended"
    ERROR = "error"


# --- ответ транспорта ---

@dataclass(frozen=True)
class Response:
    """Один ответ API: код HTTP и тело, или `code=None` и ошибка соединения."""
    code: int | None
    body: str | None
    error: str | None = None


Transport = Callable[[str], Response]


def _local_ssl() -> ssl.SSLContext:
    """Без проверки сертификата — только для 127.0.0.1 (самоподписанный Riot)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class HttpTransport:
    """GET к Live Client на 127.0.0.1:2999. Прокси из окружения не используется
    (HTTPS_PROXY проекта завернул бы loopback в прокси), проверка сертификата
    отключена только здесь и только для этого адреса."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 1.0):
        if not base_url.startswith(f"https://{HOST}:"):
            raise ValueError(f"Live Client только на {HOST}: {base_url}")
        self.base_url = base_url
        self.timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=_local_ssl()))

    def __call__(self, endpoint: str) -> Response:
        try:
            with self._opener.open(self.base_url + endpoint, timeout=self.timeout) as r:
                return Response(r.status, r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else None
            return Response(e.code, body)
        except urllib.error.URLError as e:
            return Response(None, None, f"нет соединения: {e.reason}")
        except OSError as e:                # таймаут чтения, сброс соединения
            return Response(None, None, f"нет соединения: {type(e).__name__}: {e}")


# --- записи ---

@dataclass(frozen=True)
class ItemRecord:
    item_id: int
    slot: int
    count: int

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "slot": self.slot, "count": self.count}


@dataclass(frozen=True)
class ScoreRecord:
    kills: int
    deaths: int
    assists: int
    creep_score: int
    ward_score: float

    def to_dict(self) -> dict:
        return {"kills": self.kills, "deaths": self.deaths, "assists": self.assists,
                "creep_score": self.creep_score, "ward_score": self.ward_score}


@dataclass(frozen=True)
class SpellRecord:
    """Заклинание призывателя в слоте 1 или 2 (`summonerSpellOne` / `Two`).

    `variant` — id без сезонной приставки («SummonerTeleportUpgrade»), `family`
    — семья («SummonerTeleport»), `raw` — rawDisplayName как есть, `name` —
    локализованное displayName (может быть пустым)."""
    slot: int
    family: str
    variant: str
    raw: str
    name: str

    def to_dict(self) -> dict:
        return {"slot": self.slot, "family": self.family, "variant": self.variant,
                "raw": self.raw, "name": self.name}


@dataclass(frozen=True)
class RuneRecord:
    """Руны игрока: ключевая руна и две ветки — числовые id и нормализованные
    имена («ArcaneComet», «Sorcery»); имя None, если его не узнать."""
    keystone_id: int
    keystone: str | None
    primary_id: int
    primary: str | None
    secondary_id: int
    secondary: str | None

    def to_dict(self) -> dict:
        return {"keystone": {"id": self.keystone_id, "name": self.keystone},
                "primary": {"id": self.primary_id, "name": self.primary},
                "secondary": {"id": self.secondary_id, "name": self.secondary}}


@dataclass(frozen=True)
class PlayerRecord:
    """Один из десяти. `champion` — Id Data Dragon или None с `champion_reason`;
    `riot_id` — «Имя#тег» (в старых сборках и образце Riot — summonerName).
    `spells` — два заклинания призывателя по слотам или None с `spells_reason`;
    `runes` — руны или None с `runes_reason`; `role` — роль из `position`."""
    riot_id: str
    champion: str | None
    champion_reason: str | None
    champion_raw: str               # rawChampionName как есть
    team: str                       # ORDER / CHAOS
    level: int
    is_dead: bool
    respawn_timer: float            # с, 0 у живого
    items: tuple[ItemRecord, ...]
    scores: ScoreRecord
    position: str                   # TOP/JUNGLE/… или "" (не во всех режимах)
    is_bot: bool
    skin_id: int
    spells: tuple[SpellRecord, ...] | None = None
    spells_reason: str | None = "заклинания не разбирались"
    runes: RuneRecord | None = None
    runes_reason: str | None = "руны не разбирались"

    @property
    def role(self) -> str | None:
        """Роль (TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY) или None."""
        return self.position if self.position in ROLES else None

    @property
    def role_reason(self) -> str | None:
        if self.role is not None:
            return None
        if not self.position or self.position == "NONE":
            return f"API не дал роль (position={self.position!r}: режим без ролей)"
        return f"position в незнакомом формате: {self.position!r}"

    def to_dict(self) -> dict:
        return {"riot_id": self.riot_id, "champion": self.champion,
                "champion_reason": self.champion_reason, "champion_raw": self.champion_raw,
                "team": self.team, "level": self.level, "is_dead": self.is_dead,
                "respawn_timer": self.respawn_timer,
                "items": [i.to_dict() for i in self.items],
                "scores": self.scores.to_dict(), "position": self.position,
                "is_bot": self.is_bot, "skin_id": self.skin_id,
                "spells": None if self.spells is None else [x.to_dict() for x in self.spells],
                "spells_reason": self.spells_reason,
                "runes": None if self.runes is None else self.runes.to_dict(),
                "runes_reason": self.runes_reason, "role": self.role,
                "role_reason": self.role_reason, "source": SOURCE}


@dataclass(frozen=True)
class ActivePlayerRecord:
    """Свой игрок: статы чемпиона (`championStats` как есть), золото, уровень."""
    riot_id: str
    level: int
    current_gold: float
    stats: Mapping[str, float | str]

    def to_dict(self) -> dict:
        return {"riot_id": self.riot_id, "level": self.level,
                "current_gold": self.current_gold, "stats": dict(self.stats),
                "source": SOURCE}


@dataclass(frozen=True)
class EventRecord:
    """Событие матча; `data` — остальные поля как в API (KillerName, VictimName,
    Assisters, DragonType, Stolen, Result, …)."""
    event_id: int
    name: str
    time: float                     # с от начала матча
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "name": self.name, "time": self.time,
                "data": dict(self.data), "source": SOURCE}


@dataclass(frozen=True)
class GameSnapshot:
    """Один разобранный `allgamedata`. `me` — индекс своего игрока в `players`
    (по riotId из activePlayer) или None с `me_reason`."""
    game_time: float
    game_mode: str
    map_number: int | None
    players: tuple[PlayerRecord, ...]
    active: ActivePlayerRecord | None
    me: int | None
    me_reason: str | None
    events: tuple[EventRecord, ...]

    @property
    def my_player(self) -> PlayerRecord | None:
        return None if self.me is None else self.players[self.me]

    @property
    def ended(self) -> EventRecord | None:
        return next((e for e in self.events if e.name == "GameEnd"), None)

    @property
    def end_reason(self) -> str | None:
        """Причина конца игры по событию GameEnd («GameEnd Win») или None."""
        end = self.ended
        return None if end is None else f"GameEnd {end.data.get('Result', '')}".strip()

    def to_dict(self) -> dict:
        return {"game_time": self.game_time, "game_mode": self.game_mode,
                "map_number": self.map_number,
                "players": [p.to_dict() for p in self.players],
                "active": None if self.active is None else self.active.to_dict(),
                "me": self.me, "me_reason": self.me_reason,
                "events": [e.to_dict() for e in self.events], "source": SOURCE}


@dataclass(frozen=True)
class LiveStatus:
    """Состояние API на момент `t` (unix-время опроса). `game` — снимок в
    `in_game`, `loading` с составом и `ended` (там — последний снимок игры);
    `game_t` — когда он получен. В `no_game` и `error` снимка нет."""
    state: State
    reason: str | None
    t: float
    game: GameSnapshot | None = None
    game_t: float | None = None

    def to_dict(self) -> dict:
        return {"state": self.state.value, "reason": self.reason, "t": self.t,
                "game": None if self.game is None else self.game.to_dict(),
                "game_t": self.game_t, "source": SOURCE}


# --- разбор ---

class ParseError(ValueError):
    """Ответ 200, который не похож на allgamedata."""


def champion_index(ids: Iterable[str]) -> dict[str, str]:
    """Id Data Dragon → словарь «нормализованное имя → Id» для `parse_allgamedata`."""
    return {norm_name(cid): cid for cid in ids}


def champion_id(raw: str, index: Mapping[str, str] | None) -> tuple[str | None, str | None]:
    """rawChampionName → (Id Data Dragon, None) или (None, причина)."""
    hit = next((m for f in RAW_CHAMPION_FORMATS if (m := f.fullmatch(raw))), None)
    if hit is None:
        return None, f"rawChampionName в незнакомом формате: {raw!r}"
    if index is None:
        return None, "нет списка чемпионов Data Dragon"
    internal = hit.group(1)
    cid = index.get(norm_name(internal))
    if cid is None:
        return None, f"нет в Data Dragon: {internal}"
    return cid, None


def spell_id(raw: str) -> tuple[str, str] | None:
    """Ключ заклинания («GeneratedTip_SummonerSpell_S12_SummonerTeleportUpgrade_
    DisplayName») → (семья, вариант): («SummonerTeleport»,
    «SummonerTeleportUpgrade»); формат незнакомый — None."""
    m = RAW_SPELL_FORMAT.fullmatch(raw)
    if m is None:
        return None
    variant = m.group(1)
    if (season := SPELL_SEASON.fullmatch(variant)) is not None:
        variant = season.group(1)
    family = next((f for f in SPELL_FAMILIES if variant.startswith(f)), variant)
    return family, variant


def parse_spells(p: Mapping) -> tuple[tuple[SpellRecord, ...] | None, str | None]:
    """`summonerSpells` игрока → (два заклинания, None) или (None, причина).
    Не бросает: нет поля в старой сборке — не повод терять всю игру."""
    ss = p.get("summonerSpells")
    if not isinstance(ss, Mapping):
        return None, "API не дал summonerSpells"
    out = []
    for slot, key in enumerate(SPELL_SLOTS, 1):
        d = ss.get(key)
        if not isinstance(d, Mapping):
            return None, f"API не дал summonerSpells.{key}"
        raws = [d.get(k) for k in ("rawDisplayName", "rawDescription")]
        hit = next(((r, ids) for r in raws if isinstance(r, str)
                    and (ids := spell_id(r)) is not None), None)
        if hit is None:
            return None, (f"summonerSpells.{key}: ключ в незнакомом формате: "
                          f"{d.get('rawDisplayName')!r}")
        raw, (family, variant) = hit
        name = d.get("displayName")
        out.append(SpellRecord(slot, family, variant, raw, name if isinstance(name, str) else ""))
    return tuple(out), None


def rune_name(raw: Any, rid: int, table: Mapping[int, str]) -> str | None:
    """Имя руны: из «perk_displayname_<Имя>», если имя не число; иначе по id."""
    if isinstance(raw, str) and (m := RAW_KEYSTONE.fullmatch(raw)) and not m.group(1).isdigit():
        return m.group(1)
    return table.get(rid)


def parse_runes(p: Mapping) -> tuple[RuneRecord | None, str | None]:
    """`runes` игрока → (руны, None) или (None, причина). Не бросает."""
    r = p.get("runes")
    if not isinstance(r, Mapping):
        return None, "API не дал runes"
    vals: list[Any] = []
    for name, key in RUNE_SLOTS:
        d = r.get(key)
        rid = d.get("id") if isinstance(d, Mapping) else None
        if not isinstance(rid, int) or isinstance(rid, bool):
            return None, f"API не дал runes.{key}.id"
        table = KEYSTONES if name == "keystone" else RUNE_TREES
        nm = (rune_name(d.get("rawDisplayName"), rid, table) if name == "keystone"
              else table.get(rid))
        vals += [rid, nm]
    return RuneRecord(*vals), None


def _get(d: Mapping, key: str, kind: type | tuple[type, ...], where: str):
    if not isinstance(d, Mapping) or key not in d:
        raise ParseError(f"{where}: нет поля {key}")
    v = d[key]
    # bool — подкласс int: в числовых полях он означал бы мусор
    if not isinstance(v, kind) or (isinstance(v, bool) and bool not in _kinds(kind)):
        raise ParseError(f"{where}.{key}: {type(v).__name__}, ожидался {kind}")
    return v


def _kinds(kind: type | tuple[type, ...]) -> tuple[type, ...]:
    return kind if isinstance(kind, tuple) else (kind,)


_NUM = (int, float)


def _riot_id(d: Mapping, where: str) -> str:
    # riotId — с 2024 года; образец Riot старше и даёт только summonerName
    if isinstance(d.get("riotId"), str) and d["riotId"]:
        return d["riotId"]
    return _get(d, "summonerName", str, where)


def _player(p: Mapping, i: int, index: Mapping[str, str] | None) -> PlayerRecord:
    w = f"allPlayers[{i}]"
    raw = _get(p, "rawChampionName", str, w)
    cid, why = champion_id(raw, index)
    team = _get(p, "team", str, w)
    if team not in TEAMS:
        raise ParseError(f"{w}.team: {team!r}")
    items = []
    for j, it in enumerate(_get(p, "items", list, w)):
        wi = f"{w}.items[{j}]"
        items.append(ItemRecord(_get(it, "itemID", int, wi), _get(it, "slot", int, wi),
                                _get(it, "count", int, wi)))
    s = _get(p, "scores", Mapping, w)
    ws = f"{w}.scores"
    scores = ScoreRecord(_get(s, "kills", int, ws), _get(s, "deaths", int, ws),
                         _get(s, "assists", int, ws), _get(s, "creepScore", int, ws),
                         float(_get(s, "wardScore", _NUM, ws)))
    spells, spells_why = parse_spells(p)
    runes, runes_why = parse_runes(p)
    pos = p.get("position")
    return PlayerRecord(
        riot_id=_riot_id(p, w), champion=cid, champion_reason=why, champion_raw=raw,
        team=team, level=_get(p, "level", int, w), is_dead=_get(p, "isDead", bool, w),
        respawn_timer=float(_get(p, "respawnTimer", _NUM, w)), items=tuple(items),
        scores=scores, position=pos if isinstance(pos, str) else "",
        is_bot=bool(p.get("isBot", False)), skin_id=int(p.get("skinID", 0)),
        spells=spells, spells_reason=spells_why, runes=runes, runes_reason=runes_why)


def _event(e: Mapping, i: int) -> EventRecord:
    w = f"events[{i}]"
    rest = {k: v for k, v in e.items() if k not in ("EventID", "EventName", "EventTime")}
    return EventRecord(_get(e, "EventID", int, w), _get(e, "EventName", str, w),
                       float(_get(e, "EventTime", _NUM, w)), rest)


def parse_events(d: Mapping) -> tuple[EventRecord, ...]:
    """`{"Events": [...]}` (ответ /eventdata или поле events) → записи."""
    return tuple(_event(e, i) for i, e in enumerate(_get(d, "Events", list, "events")))


def _active(a: Any) -> tuple[ActivePlayerRecord | None, str | None]:
    """activePlayer → запись или (None, причина): у наблюдателя его нет."""
    if not isinstance(a, Mapping) or not a:
        return None, "нет activePlayer"
    if "error" in a:                       # так API отвечает без своего игрока
        return None, f"activePlayer: {a['error']}"
    w = "activePlayer"
    stats = {k: (v if isinstance(v, str) else float(v))
             for k, v in _get(a, "championStats", Mapping, w).items()
             if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return ActivePlayerRecord(_riot_id(a, w), _get(a, "level", int, w),
                              float(_get(a, "currentGold", _NUM, w)), stats), None


def parse_allgamedata(d: Any, index: Mapping[str, str] | None) -> GameSnapshot:
    """Ответ allgamedata (уже из JSON) → снимок; не похож — ParseError."""
    if not isinstance(d, Mapping):
        raise ParseError(f"allgamedata: {type(d).__name__}, ожидался объект")
    g = _get(d, "gameData", Mapping, "allgamedata")
    raw_players = _get(d, "allPlayers", list, "allgamedata")
    players = tuple(_player(p, i, index) for i, p in enumerate(raw_players))
    events = parse_events(_get(d, "events", Mapping, "allgamedata"))
    active, active_why = _active(d.get("activePlayer"))
    me, me_why = None, active_why
    if active is not None:
        hits = [i for i, p in enumerate(players) if p.riot_id == active.riot_id]
        if len(hits) == 1:
            me, me_why = hits[0], None
        else:
            me_why = f"activePlayer среди allPlayers: {len(hits)} совпадений"
    mapn = g.get("mapNumber")
    return GameSnapshot(
        game_time=float(_get(g, "gameTime", _NUM, "gameData")),
        game_mode=_get(g, "gameMode", str, "gameData"),
        map_number=mapn if isinstance(mapn, int) and not isinstance(mapn, bool) else None,
        players=players, active=active, me=me, me_reason=me_why, events=events)


# --- состояния ---

def _riot_error(body: str | None) -> str | None:
    """Сообщение из ошибки Riot `{"errorCode", "httpStatus", "message"}` или None."""
    try:
        d = json.loads(body or "")
    except ValueError:
        return None
    if isinstance(d, Mapping) and "errorCode" in d:
        return f"{d.get('errorCode')}: {d.get('message')}"
    return None


class Tracker:
    """Ответы по порядку → `LiveStatus` и новые события. Без сети и потока.

    `champion_ids` — Id Data Dragon (None — чемпионы не сверяются и выходят
    `None` с причиной). `lost_after` — сколько секунд без ответа после игры
    считать сбоем (`error`), а не концом (`ended`)."""

    def __init__(self, champion_ids: Iterable[str] | None, lost_after: float = 5.0):
        self.index = None if champion_ids is None else champion_index(champion_ids)
        self.lost_after = lost_after
        self.status: LiveStatus | None = None
        self._seen: set[int] = set()
        self._game: GameSnapshot | None = None   # последний разобранный снимок
        self._game_t: float | None = None
        self._playing = False                     # игра идёт: был снимок с событиями
        self._ended = False                       # GameEnd или API пропал посреди игры
        self._lost_since: float | None = None

    def _status(self, state: State, reason: str | None, t: float,
                with_game: bool = False) -> LiveStatus:
        game = self._game if with_game else None
        self.status = LiveStatus(state, reason, t, game, self._game_t if with_game else None)
        return self.status

    def observe(self, t: float, r: Response) -> tuple[LiveStatus, list[EventRecord]]:
        if r.code is None:
            return self._unreachable(t, r.error or "нет ответа"), []
        self._lost_since = None
        if r.code != 200:
            msg = _riot_error(r.body)
            why = f"HTTP {r.code}" + (f" ({msg})" if msg else "")
            if self._playing:
                return self._status(State.ERROR, f"{why} посреди игры", t), []
            if msg is None:
                return self._status(State.ERROR, f"{why}: тело не в формате Riot", t), []
            return self._status(State.LOADING, why, t), []
        try:
            d = json.loads(r.body or "")
        except ValueError as e:
            return self._status(State.ERROR, f"не JSON: {e}", t), []
        try:
            snap = parse_allgamedata(d, self.index)
        except ParseError as e:
            return self._status(State.ERROR, str(e), t), []
        if not snap.players:
            return self._status(State.LOADING, "allPlayers пуст", t), []
        return self._game_seen(t, snap)

    def _game_seen(self, t: float, snap: GameSnapshot) -> tuple[LiveStatus, list[EventRecord]]:
        if self._game is not None and snap.game_time < self._game.game_time - NEW_GAME_BACKSTEP:
            self._seen.clear()                   # новая игра: время пошло с нуля
            self._playing = self._ended = False
        self._game, self._game_t = snap, t
        new = [e for e in snap.events if e.event_id not in self._seen]
        self._seen.update(e.event_id for e in new)
        if not snap.events:
            # Хвост экрана загрузки (журнал своей игры 2026-10-12): состав уже
            # есть, время стоит на ~0,07 с, золото 0, событий нет до GameStart.
            self._playing = False
            return self._status(State.LOADING, "нет событий: игра ещё не началась", t,
                                with_game=True), new
        if snap.ended is not None:
            self._playing, self._ended = False, True
            return self._status(State.ENDED, snap.end_reason, t, with_game=True), new
        self._playing, self._ended = True, False
        return self._status(State.IN_GAME, None, t, with_game=True), new

    def _unreachable(self, t: float, err: str) -> LiveStatus:
        if self._ended:
            return self._status(State.ENDED, f"API пропал после конца игры ({err})", t,
                                with_game=True)
        if not self._playing:                    # игры не было или она не началась
            return self._status(State.NO_GAME, err, t)
        if self._lost_since is None:
            self._lost_since = t
        if t - self._lost_since < self.lost_after:
            return self._status(State.ERROR, f"{err}; нет ответа "
                                f"{t - self._lost_since:.1f} с посреди игры", t)
        self._playing, self._ended = False, True
        return self._status(State.ENDED, f"API пропал посреди игры ({err})", t, with_game=True)


# --- наблюдение для контекста матча ---

def api_observation(snap: GameSnapshot, age: float = 0.0) -> tuple[ApiObs | None, str | None]:
    """Снимок API → наблюдение контекста матча (match_state.ApiObs) или
    (None, причина). `age` — сколько секунд прошло с получения снимка до
    момента, на который нужно наблюдение: время матча идёт вперёд, таймеры
    возрождения — назад. Чемпион не опознан, не десять игроков, не 5/5 —
    отказ: половина состава хуже, чем никакого."""
    for p in snap.players:
        if p.champion is None:
            return None, f"чемпион не опознан: {p.champion_reason}"
    age = max(0.0, age)
    players = sorted(snap.players, key=lambda p: TEAMS.index(p.team))
    me = snap.my_player
    obs = ApiObs(snap.game_time + age,
                 tuple(ApiPlayer(p.champion, p.team, p.is_dead,
                                 max(0.0, p.respawn_timer - age) if p.is_dead else 0.0)
                       for p in players),
                 me.team if me is not None else None,
                 me.champion if me is not None else None,
                 snap.map_number)
    why = obs.problem()
    return (None, why) if why is not None else (obs, None)


def status_observation(status: LiveStatus, now: float) -> tuple[ApiObs | None, str | None]:
    """Последний `LiveStatus` → наблюдение на момент `now` (то же время, что
    `LiveStatus.t`, — unix) или (None, причина). Наблюдение есть в `loading`
    с составом и `in_game`; после конца игры последний снимок уже не правда
    о «сейчас». На загрузке время матча стоит — его не двигаем."""
    if status.state not in (State.LOADING, State.IN_GAME) or status.game is None:
        return None, f"API: {status.state.value}" + (f" ({status.reason})"
                                                     if status.reason else "")
    age = now - (status.game_t or now) if status.state is State.IN_GAME else 0.0
    return api_observation(status.game, age)


# --- поток ---

def ddragon_champion_ids() -> list[str] | None:
    """Id чемпионов из кэша Data Dragon (lolvision.ddragon) или None, если кэша нет."""
    from lolvision import ddragon
    try:
        return ddragon.champions()
    except (FileNotFoundError, KeyError, ValueError):
        return None


class LiveClient:
    """Опрос allgamedata в своём потоке. `status()` — последний `LiveStatus`
    (до первого опроса — `no_game` «ещё не опрошен»), `events()` — новые
    события с прошлого вызова. Исключения транспорта и разбора наружу не
    выходят: они становятся состоянием `error` с причиной."""

    def __init__(self, transport: Transport | None = None, period: float = 0.5,
                 champion_ids: Iterable[str] | None = None, lost_after: float = 5.0,
                 clock: Callable[[], float] = time.time,
                 on_status: Callable[[LiveStatus], Any] | None = None):
        self.transport = transport or HttpTransport()
        # Наблюдатель каждого статуса (паспорт выбора, #807) — в потоке опроса;
        # его исключение не роняет опрос, текст — в `observer_error`.
        self.on_status = on_status
        self.observer_error: str | None = None
        self.period = period
        self.clock = clock
        self.tracker = Tracker(champion_ids, lost_after)
        self._lock = threading.Lock()
        self._status = LiveStatus(State.NO_GAME, "ещё не опрошен", clock())
        self._events: list[EventRecord] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> LiveStatus:
        t = self.clock()
        try:
            status, new = self.tracker.observe(t, self.transport("allgamedata"))
        except Exception as e:  # noqa: BLE001 — опрос не роняет вызывающего
            status, new = LiveStatus(State.ERROR, f"{type(e).__name__}: {e}", t), []
        with self._lock:
            self._status = status
            self._events.extend(new)
        if self.on_status is not None:
            try:
                self.on_status(status)
            except Exception as e:  # noqa: BLE001 — наблюдатель не роняет опрос
                self.observer_error = f"{type(e).__name__}: {e}"
        return status

    def status(self) -> LiveStatus:
        with self._lock:
            return self._status

    def events(self) -> list[EventRecord]:
        with self._lock:
            out, self._events = self._events, []
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            self.poll_once()
            self._stop.wait(max(0.0, self.period - (time.monotonic() - start)))

    def start(self) -> LiveClient:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="live-client", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None


# --- журнал и воспроизведение ---

@dataclass(frozen=True)
class JournalRow:
    """Строка журнала `{"t": unix-время, "code": 200 | ошибка, "body": текст | null}`.
    `code` не число — ошибка соединения (её текст)."""
    t: float
    response: Response


def _row(d: Any, where: str) -> JournalRow:
    if not isinstance(d, Mapping) or not isinstance(d.get("t"), (int, float)):
        raise ValueError(f"{where}: нет времени t")
    code, body = d.get("code"), d.get("body")
    if body is not None and not isinstance(body, str):
        raise ValueError(f"{where}: body не строка")
    if isinstance(code, int) and not isinstance(code, bool):
        return JournalRow(float(d["t"]), Response(code, body))
    return JournalRow(float(d["t"]), Response(None, body, str(code)))


def read_journal(path: str | Path) -> list[JournalRow]:
    """Строки журнала по порядку. Битая строка — ValueError с номером; только
    последняя строка без перевода строки (журнал ещё пишется) пропускается."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    rows = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            if n == len(lines):              # недописанный хвост растущего файла
                break
            raise ValueError(f"{path}:{n}: не JSON") from None
        rows.append(_row(d, f"{path}:{n}"))
    return rows


def journal_transport(rows: Iterable[JournalRow]) -> Transport:
    """Подставной транспорт: отдаёт ответы журнала по одному на вызов; журнал
    кончился — «нет соединения» (как закрытый порт)."""
    it = iter(rows)

    def transport(_endpoint: str) -> Response:
        row = next(it, None)
        return Response(None, None, "журнал кончился") if row is None else row.response
    return transport


def replay(rows: Iterable[JournalRow], champion_ids: Iterable[str] | None,
           lost_after: float = 5.0) -> Iterator[tuple[LiveStatus, list[EventRecord]]]:
    """Офлайн-прогон журнала через `Tracker` с временем из журнала."""
    tracker = Tracker(champion_ids, lost_after)
    for row in rows:
        yield tracker.observe(row.t, row.response)
