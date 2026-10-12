"""Сервис состояния своей игры для coach (#796, задача 1C): снимок и поток событий.

Что это. coach спрашивает языковую модель по кадру экрана — это секунды,
деньги и выдумки. Здесь — проверенные факты о собственной игре пользователя,
несколько раз в секунду, на loopback:

* `GET /state` — снимок состояния (JSON, `schema: 1`);
* `GET /events` — поток событий Server-Sent Events (почему не WebSocket —
  docs/state-api.md, «Поток событий»);
* `GET /health` — жив ли сервис и когда собирал снимок последний раз.

Контракт полей — docs/state-api.md; машинная схема — `snapshot_schema()`,
проверка снимка без зависимостей — `check_snapshot()`.

Факт. Каждое значение выхода — объект `{"value", "source", "age", "reason"}`:
либо значение с источником (`api` / `minimap` / `screen` / `hud` /
`inferred`) и давностью в секундах, либо `value = null` с причиной.
Молчаливых подстановок нет: нет входа — null и почему.

Сервис ничего не сводит сам — он только переводит в контракт:

* состояние десяти чемпионов (состав, жив / мёртв, таймер, место, ХП),
  время матча и расхождения источников (`conflicts`) — у одного хозяина,
  контекста матча (lolvision.match_state.MatchState: приоритет API > HUD >
  зрение, счётчик конфликтов по видам). Сюда приходит его снимок
  `MatchView` (`match_view` — под замком сеанса);
* статические поля API без конкуренции со зрением (ник, уровень, предметы,
  счёт, заклинания призывателя, статы и золото своего игрока, режим, карта) —
  прямо из `LiveStatus`;
* недавние события матча (`events`, #803) — события API за последние
  `EVENTS_WINDOW` с, где ники игроков заменены чемпионами и отношением к
  своему игроку (`recent_events`); ники наружу не выходят;
* можно ли читателю зрения отвечать — решает паспорт трансляции
  (lolvision.passport: `Passport.verdict`); без паспорта условия не
  проверены — отказ;
* свои / враги / сам и сторона у юнитов и треков — поля записей
  (lolvision/sides.py, #790) как есть, со своими причинами отказа.

Устройство. `build_snapshot` — чистая функция: входы (`StateInputs`) →
словарь снимка; ни сети, ни потоков. `StateHub` раз в `period` секунд зовёт
провайдера (callable → `StateInputs`), собирает снимок и рассылает новые
события подписчикам. `StateServer` — HTTP поверх хаба
(`ThreadingHTTPServer`, только 127.0.0.1). Сервис не импортирует live.py:
провайдера из сеанса live подключает интегратор.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lolstream.live_client import (EventRecord, GameSnapshot, LiveStatus, PlayerRecord, State,
                                   status_observation)
from lolvision.match_state import NEW_GAME_BACKSTEP, MatchState
from lolvision.passport import READERS, Passport
from lolvision.records import ChampCtx, Event, TrackRec, UnitRec

SCHEMA = 1
HOST = "127.0.0.1"
PORT = 8733                    # смотрелка — 8732
SOURCES = ("api", "minimap", "screen", "hud", "inferred")
RELATIONS = ("self", "ally", "enemy")
TEAMS = ("ORDER", "CHAOS")

# Причины отказов, общие для многих полей.
WHY_NO_API = "Live Client API не подключён к сервису"
WHY_NO_VISION = "нет обработанного кадра: сеанс зрения не подключён или ещё не дал кадр"
WHY_NO_MATCH = "контекст матча (MatchState) не подключён к сервису"
WHY_NO_PASSPORT = "паспорт трансляции не подключён к сервису"
WHY_NO_SIDES = "запись прошла мимо sides.assign: причина не указана"


# ---------------------------------------------------------------- входы

@dataclass(frozen=True)
class FrameStatus:
    """Источник кадров (lolstream.frame_source) на момент снимка.

    `state` — `Status.key` последнего ответа источника (ok / paused /
    minimized / no_window / no_frame / ended); `reason` — почему кадра нет;
    `last_ok` — unix-время последнего снятого кадра или None."""
    kind: str | None             # screen / youtube / file
    state: str
    reason: str | None = None
    backend: str | None = None
    backend_note: str | None = None
    fps: float | None = None
    size: tuple[int, int] | None = None
    last_ok: float | None = None


@dataclass(frozen=True)
class VisionFrame:
    """Итог зрения по последнему обработанному кадру (без чемпионов: они —
    в контексте матча).

    `at` — unix-время, когда кадр обработан (от него считается давность);
    `t` — время кадра по шкале потока. `camera` — центр камеры (доли карты)
    или None — модель камеры не ответила. Списки None — читатель не работал
    (нет модели, кадр не «игра»), пустой список — работал и ничего не нашёл.

    `camera_src` — откуда центр: screen — модель камеры по сцене (только
    Ущелье), minimap — рамка вьюпорта на миникарте (другая карта, #803);
    `camera_why` / `tracks_why` — почему значения нет, если причина не
    обычная."""
    i: int
    t: float
    at: float
    scene: str | None = None
    camera: tuple[float, float] | None = None
    spread: float | None = None
    units: Sequence[UnitRec] | None = None
    tracks: Sequence[TrackRec] | None = None
    camera_src: str = "screen"
    camera_why: str | None = None
    tracks_why: str | None = None


@dataclass(frozen=True)
class Observed:
    """Значение контекста матча: что, откуда, когда (время потока)."""
    value: Any
    src: str
    t: float


@dataclass(frozen=True)
class MatchView:
    """Неизменяемый снимок MatchState на момент `t` (время потока); `at` —
    unix-время этого момента. Собирается `match_view` под замком сеанса:
    провайдер зовётся из потока хаба, а MatchState меняет поток кадра."""
    t: float
    at: float
    # Записи контекста (MatchState.frame_view): жив / мёртв, ХП, таймер — с
    # источником и возрастом в самой записи.
    champs: tuple[ChampCtx, ...]
    roster_src: str | None       # MatchState.roster_src: api / hud / None
    clock: Observed | None       # MatchState.game_clock()
    api_live: bool
    api_refused: str | None
    conflicts: Mapping[str, int]


def _observed(f) -> Observed | None:
    return None if f is None else Observed(f.value, f.src, f.t)


def match_view(ms: MatchState, at: float, refused: str | None = None) -> MatchView:
    """MatchState → MatchView. `at` — unix-время момента `ms.t`.

    `refused` — почему последний снимок API не принят до MatchState (отказ
    status_observation: «игроков 1, а не 10» в Practice Tool). Такой снимок
    в MatchState не попадает, и его api_refused пуст — без этой причины
    снимок состояния врал бы «API пропал»."""
    return MatchView(ms.t, at, tuple(ms.frame_view()), ms.roster_src,
                     _observed(ms.game_clock()), ms.api_live(), refused or ms.api_refused,
                     dict(ms.conflicts))


@dataclass(frozen=True)
class StateInputs:
    """Всё, из чего собирается снимок. None — входа нет (не подключён).

    `session` — состояние сеанса live (подпись, например «идёт») или None;
    `match` — снимок контекста матча (`match_view`); `passport` — паспорт
    трансляции (2C); `vision_events` — события контекста матча
    (`MatchState.events`, можно накопительно: хаб сам отсеивает
    разосланные); `error` — провайдер упал, все отказы получают эту причину."""
    session: str | None = None
    frames: FrameStatus | None = None
    vision: VisionFrame | None = None
    match: MatchView | None = None
    api: LiveStatus | None = None
    passport: Passport | None = None
    vision_events: Sequence[Event] = field(default_factory=tuple)
    error: str | None = None


# ---------------------------------------------------------------- факты

def _age(a: float | None) -> float | None:
    return None if a is None else round(max(0.0, float(a)), 2)


def fact(value: Any, source: str, age: float | None = None) -> dict:
    """Значение с источником и давностью (с)."""
    if value is None:
        raise ValueError("fact без значения — это refuse с причиной")
    if source not in SOURCES:
        raise ValueError(f"неизвестный источник {source!r}")
    return {"value": value, "source": source, "age": _age(age), "reason": None}


def refuse(reason: str) -> dict:
    """Значения нет; почему — `reason`."""
    if not reason:
        raise ValueError("отказ без причины")
    return {"value": None, "source": None, "age": None, "reason": reason}


def maybe(value: Any, source: str, age: float | None, reason: str) -> dict:
    return refuse(reason) if value is None else fact(value, source, age)


def _uv(u: float, v: float) -> dict:
    return {"u": round(float(u), 4), "v": round(float(v), 4)}


# ---------------------------------------------------------------- правила

def relation_to_me(team: str, champion: str, me: PlayerRecord) -> str:
    """Отношение чемпиона к своему игроку по API: сам — его чемпион, свой —
    его команда (activePlayer среди allPlayers), иначе враг. Это факт API,
    а не цвет плашки."""
    if champion == me.champion:
        return "self"
    return "ally" if team == me.team else "enemy"


# ---------------------------------------------------------------- события API

# Окно и длина списка `events` (#803): последние EVENTS_MAX событий за
# EVENTS_WINDOW секунд игрового времени — хватает на «что сейчас произошло».
EVENTS_WINDOW = 180.0
EVENTS_MAX = 20
# Служебные события начала матча — не «что произошло».
EVENTS_SKIP = frozenset({"GameStart", "MinionsSpawning"})
# Обобщённые метки не-чемпионов (`who.other`): сырые имена объектов API наружу
# не выходят.
OTHERS = ("turret", "minion", "monster", "unknown")
# Поля события, которые становятся `detail`, по порядку. Только не-ники:
# KillerName / VictimName / Assisters / Acer / Recipient — ники, они
# переводятся в чемпионов (`who`), а сами не выводятся никогда.
EVENT_DETAIL = ("DragonType", "KillStreak", "TurretKilled", "InhibKilled", "InhibRespawned",
                "InhibRespawningSoon", "AcingTeam", "Result", "Stolen")
# Кто убил / кто убит по видам событий: поле API → роль. Нет в таблице —
# KillerName / VictimName, если они есть.
EVENT_KILLER = {"FirstBlood": "Recipient", "Ace": "Acer"}
EVENT_MONSTER = frozenset({"DragonKill", "HeraldKill", "BaronKill", "HordeKill",
                           "AtakhanKill"})
# Команда в имени объекта: Turret_TOrder_L0_P3_…, Turret_T1_C_05_A,
# Minion_T100L0S37N0230, Inhib_TChaos_L1_P1_… (журналы своей игры 2026-10-12).
_OBJ_TEAM = re.compile(r"_T(order|chaos|100|200|1|2)(?=_|l|$)", re.IGNORECASE)
_TEAM_OF = {"order": "ORDER", "100": "ORDER", "1": "ORDER",
            "chaos": "CHAOS", "200": "CHAOS", "2": "CHAOS"}


def object_team(name: str) -> str | None:
    """Команда башни / миньона / ингибитора по имени объекта API или None."""
    m = _OBJ_TEAM.search(name)
    if m is not None:
        return _TEAM_OF[m.group(1).lower()]
    low = name.lower()                     # старые имена: SRU_OrderMinionMelee
    hits = [t for t in TEAMS if t.lower() in low]
    return hits[0] if len(hits) == 1 else None


def object_kind(name: str) -> str:
    """Обобщённая метка не-чемпиона по имени объекта API (`OTHERS`)."""
    low = name.lower()
    if low.startswith("turret_"):
        return "turret"
    if "minion" in low:
        return "minion"
    if low.startswith("sru_"):             # SRU_Baron, SRU_Dragon_…, SRU_Horde, лагеря
        return "monster"
    return "unknown"


class Names:
    """Ник из события → игрок API. В событиях ник бывает полным riotId
    («Имя#тег»), только именем без тега или summonerName (`PlayerRecord.riot_id`
    уже он, если riotId нет): точное совпадение, иначе — единственный игрок с
    этой частью до '#'. Ники наружу не выходят: только чемпион и отношение."""

    def __init__(self, players: Sequence[PlayerRecord], me: PlayerRecord | None):
        self.me = me
        self.exact = {p.riot_id: p for p in players}
        self.short: dict[str, list[PlayerRecord]] = {}
        for p in players:
            self.short.setdefault(p.riot_id.split("#", 1)[0], []).append(p)

    def player(self, name: str) -> PlayerRecord | None:
        if name in self.exact:
            return self.exact[name]
        hits = self.short.get(name.split("#", 1)[0], [])
        return hits[0] if len(hits) == 1 else None

    def _rel_team(self, team: str | None) -> str | None:
        if self.me is None or team is None:
            return None
        return "ally" if team == self.me.team else "enemy"

    def who(self, name: Any) -> dict | None:
        """`{"champion", "relation", "other"}` или None — имени нет."""
        if not isinstance(name, str) or not name:
            return None
        p = self.player(name)
        if p is not None and p.champion is not None:
            rel = None if self.me is None else relation_to_me(p.team, p.champion, self.me)
            return {"champion": p.champion, "relation": rel, "other": None}
        if p is not None:                  # игрок есть, чемпион не опознан
            rel = "self" if p is self.me else self._rel_team(p.team)
            return {"champion": None, "relation": rel, "other": "unknown"}
        kind = object_kind(name)
        rel = self._rel_team(object_team(name)) if kind in ("turret", "minion") else None
        return {"champion": None, "relation": rel, "other": kind}


def _detail(e: EventRecord) -> str | None:
    parts = []
    for k in EVENT_DETAIL:
        v = e.data.get(k)
        if k == "Stolen":                  # API отдаёт строкой "True" / "False"
            if str(v).lower() == "true":
                parts.append("Stolen")
        elif v is not None and v != "":
            parts.append(str(v))
    return ", ".join(parts) or None


def api_event(e: EventRecord, names: Names, clock: float | None) -> dict:
    """Событие Live Client → элемент `events` (без ников)."""
    d = e.data
    killer = names.who(d.get(EVENT_KILLER.get(e.name, "KillerName")))
    if e.name in EVENT_MONSTER:
        victim = {"champion": None, "relation": None, "other": "monster"}
    elif e.name == "TurretKilled" and isinstance(d.get("TurretKilled"), str):
        victim = names.who(d["TurretKilled"])
    else:
        victim = names.who(d.get("VictimName"))
    raw = d.get("Assisters")
    assisters = [w for a in (raw if isinstance(raw, list) else ())
                 if (w := names.who(a)) is not None]
    return {"time": round(e.time, 1),
            "ago": None if clock is None else round(max(0.0, clock - e.time), 1),
            "kind": e.name, "killer": killer, "victim": victim, "assisters": assisters,
            "detail": _detail(e)}


def recent_events(game: GameSnapshot, clock: float | None, window: float = EVENTS_WINDOW,
                  limit: int | None = EVENTS_MAX) -> list[dict]:
    """Недавние события снимка API (старые → новые): за `window` с игрового
    времени до `clock` (время матча; неизвестно — время снимка API), не
    больше `limit`. Служебные `EVENTS_SKIP` не входят."""
    ref = game.game_time if clock is None else clock
    names = Names(game.players, game.my_player)
    evs = sorted((e for e in game.events
                  if e.name not in EVENTS_SKIP and e.time >= ref - window),
                 key=lambda e: (e.time, e.event_id))
    if limit is not None:
        evs = evs[-limit:] if limit > 0 else []
    return [api_event(e, names, clock) for e in evs]


def record_sides(rec: UnitRec | TrackRec, age: float | None) -> dict:
    """Свой / враг / сам и сторона у юнита или трека — поля записи
    (lolvision/sides.py, #790) как есть: значение или null с причиной записи.
    Источник отношения — `screen` (цвет полоски в пикселях); стороны —
    `side_src` записи: `api` — от команды своего игрока по API, `detector` —
    ответ детектора по пикселям (`screen`)."""
    side_src = "api" if rec.side_src == "api" else "screen"
    return {"relation": maybe(rec.relation, "screen", age, rec.relation_reason or WHY_NO_SIDES),
            "side": maybe(rec.side, side_src, age, rec.side_reason or WHY_NO_SIDES)}


def player_spells(p: PlayerRecord | None, why: str,
                  age: float | None) -> tuple[dict, dict]:
    """Заклинания призывателя игрока API (#804) → (`spells`, `spell_variants`):
    семьи («SummonerTeleport» и у улучшенного Телепорта) и точные варианты
    («SummonerTeleportUpgrade») по слотам 1, 2. Игрока нет — `why`; API не дал
    заклинаний — причина разбора."""
    if p is None:
        return refuse(why), refuse(why)
    if p.spells is None:
        why_s = p.spells_reason or "заклинаний нет"
        return refuse(why_s), refuse(why_s)
    return (fact([x.family for x in p.spells], "api", age),
            fact([x.variant for x in p.spells], "api", age))


def _record(rec: UnitRec | TrackRec, age: float | None) -> dict:
    """Запись records.py в выход: поля как есть, отношение и сторона — фактами."""
    d = {k: v for k, v in rec.to_dict().items() if k not in ("relation_reason", "side_reason")}
    d.update(record_sides(rec, age))
    return d


@dataclass(frozen=True)
class SelfSeen:
    """Свой чемпион на кадре по цвету полоски (#802): сколько чемпионов с
    relation self на кадре и можно ли верить детектору юнитов (паспорт)."""
    n: int
    units_ok: bool
    units_why: str | None


def self_on_screen(seen: SelfSeen, alive: bool | None, lag: float | None, names_ok: bool,
                   names_why: str | None) -> dict | None:
    """`on_screen` своего чемпиона по полоске self или None — решать, как у
    всех, по имени над плашкой (детектор выключен паспортом или полоски нет,
    а имена читаются). Правила — как у места своего в контексте матча
    (MatchState._from_units): несколько self — отказ, а не догадка; мёртвого
    на экране нет — полоска не его."""
    if not seen.units_ok:
        return None
    if seen.n == 0:
        return None if names_ok else refuse(
            f"своей полоски (self) на кадре нет, имя над плашкой не читается: {names_why}")
    if seen.n > 1:
        return refuse(f"своих полосок на кадре {seen.n}, а не одна: который свой — "
                      "неизвестно (conflicts self_many)")
    if alive is False:
        return refuse("свой мёртв по контексту матча, а полоска self на кадре — не его")
    return fact(True, "screen", lag)


def reader_ok(passport: Passport | None, reader: str,
              miss: Callable[[str], str]) -> tuple[bool, str | None]:
    """Можно ли читателю отвечать — по паспорту; без паспорта — нельзя."""
    if passport is None:
        return False, miss(f"{WHY_NO_PASSPORT}: условия читателя «{READERS[reader].title}» "
                           "не проверены")
    ok, why = passport.verdict(reader)
    return ok, why


# ---------------------------------------------------------------- сборка

def _sources(inp: StateInputs, now: float, miss: Callable[[str], str]) -> dict:
    f = inp.frames
    frames = ({"state": "absent", "reason": miss("источник кадров не подключён"),
               "kind": None, "backend": None, "backend_note": None, "fps": None,
               "size": None, "last_frame_age": None} if f is None else
              {"state": f.state, "reason": f.reason, "kind": f.kind, "backend": f.backend,
               "backend_note": f.backend_note, "fps": f.fps,
               "size": None if f.size is None else list(f.size),
               "last_frame_age": None if f.last_ok is None else _age(now - f.last_ok)})
    if f is not None and f.state != "ok" and not f.reason:
        frames["reason"] = f"кадра нет: {f.state}"
    v = inp.vision
    vision = ({"state": "absent", "reason": miss(WHY_NO_VISION), "frame": None,
               "stream_t": None, "age": None, "session": inp.session}
              if v is None else
              {"state": "ok", "reason": None, "frame": v.i, "stream_t": v.t,
               "age": _age(now - v.at), "session": inp.session})
    m = inp.match
    match = ({"state": "absent", "reason": miss(WHY_NO_MATCH), "stream_t": None, "age": None,
              "roster": None, "api_live": None, "api_refused": None}
             if m is None else
             {"state": "ok", "reason": None, "stream_t": m.t, "age": _age(now - m.at),
              "roster": m.roster_src, "api_live": m.api_live, "api_refused": m.api_refused})
    a = inp.api
    api = ({"state": "absent", "reason": miss(WHY_NO_API), "age": None, "game_age": None}
           if a is None else
           {"state": a.state.value, "reason": a.reason, "age": _age(now - a.t),
            "game_age": None if a.game_t is None else _age(now - a.game_t)})
    return {"frames": frames, "vision": vision, "match": match, "api": api}


def build_snapshot(inp: StateInputs, now: float, seq: int = 0) -> dict:
    """Входы → снимок состояния по контракту docs/state-api.md (schema 1).

    Чистая функция: `now` — unix-время снимка, от него считаются все
    давности; `seq` — номер снимка (ставит хаб)."""
    def miss(why: str) -> str:
        return why if inp.error is None else f"сбой провайдера состояния: {inp.error}"

    # --- API: статические поля ---
    st = inp.api
    game = None if st is None else st.game
    if st is None:
        api_why = miss(WHY_NO_API)
    elif game is None:
        api_why = f"Live Client: {st.state.value}" + (f" ({st.reason})" if st.reason else "")
    else:
        api_why = ""
    api_age = None if st is None or st.game_t is None else now - st.game_t
    me = None if game is None else game.my_player
    me_why = api_why or (f"свой игрок не найден: {game.me_reason}" if me is None else "")
    players = {} if game is None else {p.champion: p for p in game.players if p.champion}

    # --- контекст матча ---
    mv = inp.match
    why_match = miss(WHY_NO_MATCH)
    mlag = None if mv is None else max(0.0, now - mv.at)

    def mage(t: float) -> float:
        return mlag + max(0.0, mv.t - t)

    vis = inp.vision
    why_vis = miss(WHY_NO_VISION)
    lag = None if vis is None else max(0.0, now - vis.at)
    names_ok, names_why = reader_ok(inp.passport, "names", miss)
    units_ok, units_why = reader_ok(inp.passport, "units", miss)
    named = {} if vis is None or vis.units is None else \
        {u.name: u for u in vis.units if u.kind == "champion" and u.name}
    # Свой чемпион на кадре — по цвету полоски (relation self, #802): на виде
    # игрока имена не читаются, а свою зелёную полоску видно.
    selves = 0 if vis is None or vis.units is None else \
        sum(u.kind == "champion" and u.relation == "self" for u in vis.units)
    seen_self = SelfSeen(selves, units_ok, units_why)

    # Снимок API отклонён контекстом матча — причина для времени, состава и
    # всего, что от них зависит (свой чемпион в контексте, враги).
    refused_why = (None if mv is None or not mv.api_refused
                   else f"снимок API не принят: {mv.api_refused}")

    # --- матч ---
    if mv is None:
        clock = refuse(why_match)
    elif mv.clock is not None:
        clock = fact(mv.clock.value, mv.clock.src, mage(mv.clock.t))
    elif mv.api_refused:
        # Отказ снимка API — и есть причина: API жив, но снимок не принят.
        clock = refuse(refused_why)
    else:
        clock = refuse(f"контекст матча без времени по API: {api_why or 'API пропал'}")
    match = {"time": clock,
             "mode": refuse(api_why) if game is None else fact(game.game_mode, "api", api_age),
             "map": refuse(api_why) if game is None else
             maybe(game.map_number, "api", api_age, "API не дал mapNumber"),
             "side": refuse(me_why) if me is None else fact(me.team, "api", api_age),
             "state": refuse(api_why) if st is None else
             fact(st.state.value, "api", now - st.t)}

    # --- недавние события API (#803): ники → чемпионы ---
    if game is None:
        events = refuse(api_why)
    elif st.state not in (State.IN_GAME, State.ENDED):
        events = refuse(f"не в игре: Live Client {st.state.value}"
                        + (f" ({st.reason})" if st.reason else ""))
    else:
        events = fact(recent_events(game, clock["value"]), "api", api_age)

    # --- десять чемпионов ---
    entries: list[dict] = []
    if mv is None:
        champions = refuse(why_match)
    elif not mv.champs:
        champions = refuse(refused_why if mv.api_refused else
                           "состава нет: " + (api_why or "контекст матча ещё не знает состава"))
    else:
        for c in mv.champs:
            entries.append(_champion(c, mv.roster_src, mlag, mage, me, me_why, players, api_age,
                                     api_why, vis, why_vis, lag, named, names_ok, names_why,
                                     seen_self))
        champions = fact(entries, mv.roster_src, mlag)

    # --- свой игрок: статика из API, жив / мёртв — из контекста матча ---
    me_fields = ("riot_id", "champion", "team", "level", "gold", "alive", "respawn_in",
                 "position", "stats", "spells", "spell_variants")
    if me is None:
        me_out = {k: refuse(me_why) for k in me_fields}
    else:
        act = game.active
        mine = next((e for e in entries if e["champion"]["value"] == me.champion), None)
        spells, variants = player_spells(me, me_why, api_age)
        why_mine = (why_match if mv is None else
                    refused_why if refused_why and not entries else
                    "своего чемпиона нет в контексте матча")
        me_out = {"riot_id": fact(me.riot_id, "api", api_age),
                  "champion": maybe(me.champion, "api", api_age,
                                    me.champion_reason or "не опознан"),
                  "team": fact(me.team, "api", api_age), "level": fact(me.level, "api", api_age),
                  "gold": (fact(act.current_gold, "api", api_age) if act is not None
                           else refuse("нет activePlayer")),
                  "alive": mine["alive"] if mine else refuse(why_mine),
                  "respawn_in": mine["respawn_in"] if mine else refuse(why_mine),
                  # место своего — то же, что у его записи в champions (#802)
                  "position": mine["position"] if mine else refuse(why_mine),
                  "stats": (fact(dict(act.stats), "api", api_age) if act is not None
                            else refuse("нет activePlayer")),
                  "spells": spells, "spell_variants": variants}

    # --- враги с давностью места ---
    if champions["value"] is None:
        enemies = refuse(champions["reason"])
    elif me is None:
        enemies = refuse(f"своя сторона неизвестна: {me_why}")
    else:
        enemies = fact([{k: e[k] for k in ("champion", "position", "map_state", "alive",
                                            "respawn_in", "on_screen")}
                        for e in entries if e["relation"]["value"] == "enemy"], "api", api_age)

    # --- камера, сцена, юниты, треки ---
    if vis is None:
        camera = {"center": refuse(why_vis), "spread": refuse(why_vis)}
        scene = units = tracks = refuse(why_vis)
    else:
        cam = vis.camera
        ok = cam is not None and cam[0] is not None and cam[1] is not None
        camera = {"center": fact(_uv(*cam), vis.camera_src, lag) if ok
                  else refuse(vis.camera_why or "модель камеры не ответила"),
                  "spread": maybe(None if vis.spread is None else round(vis.spread, 4),
                                  "screen", lag,
                                  "разброс камеры не считается" if vis.camera_src == "screen"
                                  else "разброс есть только у модели камеры (Ущелье)")}
        scene = maybe(vis.scene, "screen", lag, "тип сцены не считается (нет модели)")
        if not units_ok:
            units = tracks = refuse(units_why)
        else:
            if vis.units is None:
                units = refuse("юниты на экране не ищутся (кадр не «игра»)")
            else:
                out = []
                for u in vis.units:
                    d = _record(u, lag)
                    d["name"] = (maybe(u.name, "screen", lag, "не чемпион или имя не прочитано")
                                 if names_ok else refuse(names_why))
                    out.append(d)
                units = fact(out, "screen", lag)
            if vis.tracks is None:
                tracks = refuse(vis.tracks_why or "треки юнитов не ведутся")
            else:
                out = []
                for tr in vis.tracks:
                    out.append(_record(tr, lag + tr.age))
                tracks = fact(out, "screen", lag)

    if mv is None:
        conflicts = refuse(why_match)
    else:
        conflicts = fact({"total": sum(mv.conflicts.values()), "kinds": dict(mv.conflicts)},
                         "inferred", mlag)
    passport = (refuse(miss(WHY_NO_PASSPORT)) if inp.passport is None
                else fact(inp.passport.to_dict(), "inferred"))
    return {"schema": SCHEMA, "seq": seq, "t": round(now, 3), "error": inp.error,
            "sources": _sources(inp, now, miss), "passport": passport, "match": match,
            "events": events, "me": me_out, "champions": champions, "enemies": enemies,
            "camera": camera, "scene": scene, "units": units, "tracks": tracks,
            "conflicts": conflicts}


def _champion(c: ChampCtx, roster_src: str, mlag: float, mage: Callable[[float], float],
              me: PlayerRecord | None, me_why: str, players: Mapping[str, PlayerRecord],
              api_age: float | None, api_why: str, vis: VisionFrame | None, why_vis: str,
              lag: float | None, named: Mapping[str, UnitRec], names_ok: bool,
              names_why: str | None, seen_self: SelfSeen) -> dict:
    """Один из десяти: состояние — из записи контекста матча, статика — из API.
    Свой чемпион на кадре — ещё и по полоске self (`seen_self`, #802)."""
    p = players.get(c.name)
    why_p = api_why or "нет в составе API"
    if me is None:
        relation = refuse(f"своя сторона неизвестна: {me_why}")
    elif roster_src != "api":
        relation = refuse("состав не из API: команды по HUD с своим игроком API не сверяются")
    else:
        relation = fact(relation_to_me(c.team, c.name, me), "api", api_age)

    if c.alive is None:
        alive = respawn = refuse("жив / мёртв не наблюдался")
    else:
        alive = fact(c.alive, c.alive_src, mlag + (c.alive_age or 0.0))
        if c.alive:
            respawn = refuse(f"жив ({c.alive_src})")
        else:
            respawn = maybe(None if c.respawn_in is None else round(c.respawn_in, 1),
                            c.respawn_src or c.alive_src, mlag,
                            "мёртв, таймер возрождения неизвестен")

    position = (refuse("место не наблюдалось") if c.u is None or c.v is None or c.src is None
                else fact(_uv(c.u, c.v), c.src, mlag + (c.age or 0.0)))
    hp = (refuse("ХП не наблюдалось") if c.hp is None or c.hp_src is None
          else fact(round(float(c.hp), 2), c.hp_src, mlag + (c.hp_age or 0.0)))

    is_me = me is not None and c.name == me.champion
    by_bar = (None if vis is None or vis.units is None or not is_me
              else self_on_screen(seen_self, c.alive, lag, names_ok, names_why))
    if vis is None:
        on_screen = refuse(why_vis)
    elif by_bar is not None:
        on_screen = by_bar
    elif not names_ok:
        on_screen = refuse(names_why)
    elif vis.units is None:
        on_screen = refuse("юниты на экране не ищутся (кадр не «игра»)")
    elif c.name in named:
        on_screen = fact(True, "screen", lag)
    else:
        on_screen = refuse("на экране не опознан: имени над плашкой нет в кадре "
                           "или оно не прочитано")

    spells, variants = player_spells(p, why_p, api_age)
    return {"champion": fact(c.name, roster_src, mlag), "team": fact(c.team, roster_src, mlag),
            "relation": relation,
            "riot_id": refuse(why_p) if p is None else fact(p.riot_id, "api", api_age),
            "level": refuse(why_p) if p is None else fact(p.level, "api", api_age),
            "items": refuse(why_p) if p is None else
            fact([it.item_id for it in p.items], "api", api_age),
            "scores": refuse(why_p) if p is None else fact(p.scores.to_dict(), "api", api_age),
            "spells": spells, "spell_variants": variants,
            "alive": alive, "respawn_in": respawn, "position": position,
            "map_state": fact(c.state, "inferred", mlag), "on_screen": on_screen, "hp": hp,
            "conflicts": c.conflicts}


# ---------------------------------------------------------------- схема

def _fact_of(value: dict | None = None) -> dict:
    s: dict = {"$ref": "#/$defs/fact"}
    if value is not None:
        s["properties"] = {"value": {"anyOf": [value, {"type": "null"}]}}
    return s


def _obj(props: dict, extra: bool | dict = False) -> dict:
    return {"type": "object", "required": sorted(props), "properties": props,
            "additionalProperties": extra}


def snapshot_schema() -> dict:
    """JSON Schema (draft 2020-12) снимка. Проверка без зависимостей — `check_snapshot`."""
    num, s, b, i = {"type": "number"}, {"type": "string"}, {"type": "boolean"}, \
        {"type": "integer"}
    nnum, ns, ni, nb = {"type": ["number", "null"]}, {"type": ["string", "null"]}, \
        {"type": ["integer", "null"]}, {"type": ["boolean", "null"]}
    uv = _obj({"u": num, "v": num})
    team = {"enum": list(TEAMS)}
    rel = {"enum": list(RELATIONS)}
    spells = {"type": "array", "items": s, "minItems": 2, "maxItems": 2}
    champ = _obj({"champion": _fact_of(s), "riot_id": _fact_of(s), "team": _fact_of(team),
                  "relation": _fact_of(rel), "level": _fact_of(i), "alive": _fact_of(b),
                  "respawn_in": _fact_of(num), "position": _fact_of(uv),
                  "map_state": _fact_of({"enum": ["visible", "remembered", "dead", "unknown"]}),
                  "on_screen": _fact_of(b), "hp": _fact_of(num),
                  "items": _fact_of({"type": "array", "items": i}),
                  "scores": _fact_of({"type": "object"}), "spells": _fact_of(spells),
                  "spell_variants": _fact_of(spells), "conflicts": i})
    enemy = _obj({k: champ["properties"][k] for k in ("champion", "position", "map_state",
                                                      "alive", "respawn_in", "on_screen")})
    # Юнит и трек — запись records.py как есть (to_dict) плюс факты; поля
    # записи не закрываются схемой сервиса: их версия — records.SCHEMA.
    unit = {"type": "object", "required": ["kind", "x", "y", "name", "relation", "side"],
            "properties": {"kind": s, "x": num, "y": num, "name": _fact_of(s),
                           "relation": _fact_of(rel), "side": _fact_of(team)}}
    track = {"type": "object", "required": ["id", "kind", "state", "u", "v", "relation", "side"],
             "properties": {"id": s, "kind": s, "state": {"enum": ["seen", "unseen", "lost"]},
                            "u": num, "v": num, "relation": _fact_of(rel),
                            "side": _fact_of(team)}}
    frames = _obj({"state": {"enum": ["absent", "ok", "paused", "minimized", "no_window",
                                      "no_frame", "ended"]},
                   "reason": ns, "kind": ns, "backend": ns, "backend_note": ns, "fps": nnum,
                   "size": {"anyOf": [{"type": "array", "items": i, "minItems": 2,
                                       "maxItems": 2}, {"type": "null"}]},
                   "last_frame_age": nnum})
    vision = _obj({"state": {"enum": ["absent", "ok"]}, "reason": ns, "frame": ni,
                   "stream_t": nnum, "age": nnum, "session": ns})
    match_src = _obj({"state": {"enum": ["absent", "ok"]}, "reason": ns, "stream_t": nnum,
                      "age": nnum, "roster": {"enum": ["api", "hud", None]},
                      "api_live": nb, "api_refused": ns})
    api = _obj({"state": {"enum": ["absent", "no_game", "loading", "in_game", "ended",
                                   "error"]},
                "reason": ns, "age": nnum, "game_age": nnum})
    passport = _obj({"items": {"type": "object"}, "pending": {"type": "array", "items": s},
                     "disabled": {"type": "array",
                                  "items": _obj({"reader": s, "title": s, "reason": s})},
                     "enabled": {"type": "array", "items": s}})
    conflicts = _obj({"total": i, "kinds": {"type": "object", "additionalProperties": i}})
    who = _obj({"champion": ns, "relation": {"enum": [*RELATIONS, None]},
                "other": {"enum": [*OTHERS, None]}})
    event = _obj({"time": num, "ago": nnum, "kind": s,
                  "killer": {"anyOf": [who, {"type": "null"}]},
                  "victim": {"anyOf": [who, {"type": "null"}]},
                  "assisters": {"type": "array", "items": who}, "detail": ns})
    fact_def = {
        "type": "object", "required": ["value", "source", "age", "reason"],
        "properties": {"value": {}, "source": {"enum": [*SOURCES, None]},
                       "age": nnum, "reason": ns},
        "additionalProperties": False,
        # value есть ⇔ есть источник и нет причины; value null ⇔ есть причина
        "oneOf": [
            {"properties": {"value": {"not": {"type": "null"}}, "source": s,
                            "reason": {"type": "null"}}},
            {"properties": {"value": {"type": "null"}, "source": {"type": "null"},
                            "age": {"type": "null"}, "reason": {"type": "string",
                                                                "minLength": 1}}},
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "lol-game-state-vision/state-api/1",
        "$defs": {"fact": fact_def},
        **_obj({
            "schema": {"const": SCHEMA}, "seq": i, "t": num, "error": ns,
            "sources": _obj({"frames": frames, "vision": vision, "match": match_src,
                             "api": api}),
            "passport": _fact_of(passport),
            "match": _obj({"time": _fact_of(num), "mode": _fact_of(s), "map": _fact_of(i),
                           "side": _fact_of(team),
                           "state": _fact_of({"enum": ["no_game", "loading", "in_game",
                                                       "ended", "error"]})}),
            "events": _fact_of({"type": "array", "items": event, "maxItems": EVENTS_MAX}),
            "me": _obj({"riot_id": _fact_of(s), "champion": _fact_of(s), "team": _fact_of(team),
                        "level": _fact_of(i), "gold": _fact_of(num), "alive": _fact_of(b),
                        "respawn_in": _fact_of(num), "position": _fact_of(uv),
                        "stats": _fact_of({"type": "object"}), "spells": _fact_of(spells),
                        "spell_variants": _fact_of(spells)}),
            "champions": _fact_of({"type": "array", "items": champ, "maxItems": 10}),
            "enemies": _fact_of({"type": "array", "items": enemy, "maxItems": 5}),
            "camera": _obj({"center": _fact_of(uv), "spread": _fact_of(num)}),
            "scene": _fact_of(s),
            "units": _fact_of({"type": "array", "items": unit}),
            "tracks": _fact_of({"type": "array", "items": track}),
            "conflicts": _fact_of(conflicts),
        }),
    }


_TYPES: dict[str, Callable[[Any], bool]] = {
    "null": lambda x: x is None,
    "boolean": lambda x: isinstance(x, bool),
    "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "string": lambda x: isinstance(x, str),
    "array": lambda x: isinstance(x, list),
    "object": lambda x: isinstance(x, dict),
}


def _check(x: Any, s: dict, root: dict, path: str, out: list[str]) -> None:
    """Подмножество JSON Schema, которым написана `snapshot_schema`."""
    if "$ref" in s:
        _check(x, root["$defs"][s["$ref"].rsplit("/", 1)[1]], root, path, out)
    if "type" in s:
        kinds = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_TYPES[k](x) for k in kinds):
            out.append(f"{path}: {type(x).__name__}, ожидался {s['type']}")
            return
    if "const" in s and x != s["const"]:
        out.append(f"{path}: {x!r} ≠ {s['const']!r}")
    if "enum" in s and x not in s["enum"]:
        out.append(f"{path}: {x!r} не из {s['enum']}")
    if "minLength" in s and isinstance(x, str) and len(x) < s["minLength"]:
        out.append(f"{path}: пустая строка")
    if "not" in s and not _errors(x, s["not"], root, path):
        out.append(f"{path}: не должно подходить под {s['not']}")
    if "anyOf" in s and all(_errors(x, sub, root, path) for sub in s["anyOf"]):
        out.append(f"{path}: не подходит ни под один вариант anyOf")
    if "oneOf" in s:
        n = sum(not _errors(x, sub, root, path) for sub in s["oneOf"])
        if n != 1:
            out.append(f"{path}: подходит под {n} вариантов oneOf из {len(s['oneOf'])}")
    if isinstance(x, dict):
        props = s.get("properties", {})
        for k in s.get("required", ()):
            if k not in x:
                out.append(f"{path}: нет поля {k}")
        for k, v in x.items():
            if k in props:
                _check(v, props[k], root, f"{path}.{k}", out)
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: лишнее поле {k}")
            elif isinstance(s.get("additionalProperties"), dict):
                _check(v, s["additionalProperties"], root, f"{path}.{k}", out)
    if isinstance(x, list):
        if "minItems" in s and len(x) < s["minItems"]:
            out.append(f"{path}: меньше {s['minItems']} элементов")
        if "maxItems" in s and len(x) > s["maxItems"]:
            out.append(f"{path}: больше {s['maxItems']} элементов")
        if "items" in s:
            for n, v in enumerate(x):
                _check(v, s["items"], root, f"{path}[{n}]", out)


def _errors(x: Any, s: dict, root: dict, path: str) -> list[str]:
    out: list[str] = []
    _check(x, s, root, path, out)
    return out


def check_snapshot(snap: Any) -> list[str]:
    """Ошибки снимка против `snapshot_schema()`; пустой список — снимок по контракту.
    Стандартная библиотека: jsonschema сервису не нужен."""
    root = snapshot_schema()
    return _errors(json.loads(json.dumps(snap)), root, root, "$")


# ---------------------------------------------------------------- хаб

Provider = Callable[[], StateInputs]


def _event_key(e: Event) -> str:
    return json.dumps(e.to_dict(), sort_keys=True, ensure_ascii=False)


class _Sub:
    """Подписчик потока: своя очередь; переполнилась — подписчик отключается
    (клиент переподключится и получит свежий снимок в hello)."""

    def __init__(self, size: int):
        self.q: queue.Queue = queue.Queue(size)
        self.dropped = False


class StateHub:
    """Опрашивает провайдера раз в `period` с: снимок + новые события.

    События потока (`event` SSE): `state` — снимок, не чаще раза в
    `state_every` с; `game` — новое событие Live Client (по EventID, новая
    игра — счёт заново); `vision` — новое событие контекста матча; `source`
    — сменилось состояние источника кадров, зрения, контекста или API. Провайдер упал —
    снимок из одних отказов с причиной, хаб живёт дальше."""

    def __init__(self, provider: Provider, period: float = 0.25, state_every: float = 1.0,
                 clock: Callable[[], float] = time.time, queue_size: int = 256):
        self.provider, self.period, self.state_every = provider, period, state_every
        self.clock, self.queue_size = clock, queue_size
        self._lock = threading.Lock()
        self._subs: list[_Sub] = []
        self._snap: dict | None = None
        self._seq = 0
        self._eid = 0
        self._last_state: float | None = None
        self._last_tick: float | None = None
        self._api_seen: set[int] = set()
        self._api_time: float | None = None
        self._vision_seen: set[str] = set()
        self._src: tuple | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- такт ----------

    def tick(self) -> dict:
        """Один опрос провайдера: снимок и рассылка. Возвращает снимок."""
        now = self.clock()
        try:
            inp = self.provider()
        except Exception as e:  # noqa: BLE001 — сбой провайдера становится отказом
            inp = StateInputs(error=f"{type(e).__name__}: {e}")
        with self._lock:
            self._seq += 1
            snap = build_snapshot(inp, now, self._seq)
            self._snap, self._last_tick = snap, now
            out = self._new_events(inp, snap)
            if self._last_state is None or now - self._last_state >= self.state_every:
                self._last_state = now
                out.append(("state", snap))
            for kind, data in out:
                self._publish(kind, data, now)
        return snap

    def _new_events(self, inp: StateInputs, snap: dict) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        src = tuple(snap["sources"][k]["state"] for k in ("frames", "vision", "match", "api"))
        if src != self._src:
            self._src = src
            out.append(("source", snap["sources"]))
        game = inp.api.game if inp.api is not None else None
        if game is not None:
            if self._api_time is not None and game.game_time < self._api_time - NEW_GAME_BACKSTEP:
                self._api_seen.clear()           # новая игра: EventID с нуля
            self._api_time = game.game_time
            for e in game.events:
                if e.event_id not in self._api_seen:
                    self._api_seen.add(e.event_id)
                    out.append(("game", e.to_dict()))
        for e in inp.vision_events:
            k = _event_key(e)
            if k not in self._vision_seen:
                self._vision_seen.add(k)
                out.append(("vision", {**e.to_dict(), "source": "inferred"}))
        return out

    def _publish(self, kind: str, data: dict, now: float) -> None:
        self._eid += 1
        msg = (self._eid, kind, {"schema": SCHEMA, "t": round(now, 3), "data": data})
        for sub in list(self._subs):
            try:
                sub.q.put_nowait(msg)
            except queue.Full:
                sub.dropped = True
                self._subs.remove(sub)

    # ---------- чтение ----------

    def snapshot(self) -> dict:
        """Последний снимок; до первого такта — собрать сейчас."""
        with self._lock:
            snap = self._snap
        return snap if snap is not None else self.tick()

    def subscribe(self) -> tuple[_Sub, dict]:
        """Новый подписчик и снимок для приветствия (hello)."""
        snap = self.snapshot()
        sub = _Sub(self.queue_size)
        with self._lock:
            self._subs.append(sub)
        return sub, snap

    def unsubscribe(self, sub: _Sub) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def health(self) -> dict:
        now = self.clock()
        with self._lock:
            snap, last, n = self._snap, self._last_tick, len(self._subs)
        age = None if last is None else round(now - last, 3)
        running = self._thread is not None and self._thread.is_alive()
        error = None if snap is None else snap["error"]
        fresh = age is not None and age <= max(5 * self.period, 2.0)
        return {"ok": bool(running and fresh and error is None), "schema": SCHEMA,
                "running": running, "seq": None if snap is None else snap["seq"],
                "age": age, "error": error, "subscribers": n}

    # ---------- поток ----------

    def _run(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            self.tick()
            self._stop.wait(max(0.0, self.period - (time.monotonic() - start)))

    def start(self) -> StateHub:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="state-hub", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        with self._lock:
            subs, self._subs = self._subs, []
        for sub in subs:                         # разбудить потоки /events: конец
            try:
                sub.q.put_nowait(None)
            except queue.Full:
                sub.dropped = True


# ---------------------------------------------------------------- HTTP

def sse(eid: int, kind: str, data: dict) -> bytes:
    """Одно сообщение Server-Sent Events: id, event, data одной строкой JSON."""
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {eid}\nevent: {kind}\ndata: {body}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    server_version = "lol-state/1"
    hub: StateHub                   # ставит StateServer
    keepalive: float

    def log_message(self, fmt: str, *args) -> None:   # тихо: запросы раз в доли секунды
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — имя задаёт http.server
        path = self.path.split("?", 1)[0]
        if path == "/state":
            self._json(200, self.hub.snapshot())
        elif path == "/health":
            h = self.hub.health()
            self._json(200 if h["ok"] else 503, h)
        elif path == "/events":
            self._events()
        else:
            self._json(404, {"error": f"нет такого пути: {path}",
                             "paths": ["/state", "/events", "/health"]})

    def _events(self) -> None:
        sub, snap = self.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.write(sse(0, "hello", {"schema": SCHEMA, "t": snap["t"], "data": snap}))
            self.wfile.flush()
            while True:
                try:
                    msg = sub.q.get(timeout=self.keepalive)
                except queue.Empty:
                    if sub.dropped:
                        return
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if msg is None:                 # сервис останавливается
                    return
                self.wfile.write(sse(*msg))
                self.wfile.flush()
        except OSError:                          # клиент ушёл (BrokenPipe, Reset, Aborted)
            return
        finally:
            self.hub.unsubscribe(sub)


class StateServer:
    """HTTP сервиса состояния на loopback поверх `StateHub`.

    `port=0` — свободный порт (тесты); адрес — `url`. Только loopback:
    сервис отдаёт состояние своей игры, наружу ему незачем."""

    def __init__(self, provider: Provider, host: str = HOST, port: int = PORT,
                 period: float = 0.25, state_every: float = 1.0, keepalive: float = 10.0,
                 clock: Callable[[], float] = time.time):
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError(f"сервис состояния только на loopback, а не {host}")
        self.hub = StateHub(provider, period, state_every, clock)
        handler = type("StateHandler", (_Handler,), {"hub": self.hub, "keepalive": keepalive})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> StateServer:
        self.hub.start()
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="state-http",
                                        daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.hub.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(2.0)


# ---------------------------------------------------------------- без сеанса live

def api_only_provider(client, clock: Callable[[], float] = time.time) -> Provider:
    """Провайдер только из Live Client (`live_client.LiveClient`): свой
    контекст матча кормится одним API (`status_observation`), зрения нет —
    его поля отказы с причиной. Сеанс live подключает свой провайдер.
    Время контекста — unix-время: кадров, а значит и шкалы потока, нет."""
    ms = MatchState()

    def provide() -> StateInputs:
        now = clock()
        status = client.status()
        obs, _why = status_observation(status, now)   # причина видна в sources.api
        ms.update(now, kind="none", api=obs)
        return StateInputs(api=status, match=match_view(ms, now))
    return provide


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m lolstream.state_service` — сервис только на Live Client API."""
    from lolstream.live_client import LiveClient, ddragon_champion_ids

    ap = argparse.ArgumentParser(description="Сервис состояния своей игры (Live Client API)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--period", type=float, default=0.25, help="такт снимка, с")
    a = ap.parse_args(argv)
    client = LiveClient(champion_ids=ddragon_champion_ids()).start()
    srv = StateServer(api_only_provider(client), port=a.port, period=a.period).start()
    print(f"состояние: {srv.url}/state  поток: {srv.url}/events  здоровье: {srv.url}/health",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        client.stop()


if __name__ == "__main__":
    main()
