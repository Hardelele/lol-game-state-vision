"""Записи строки кадра live: именованные поля вместо позиционных списков.

Зачем (#787, #797). Строка кадра несла юнитов, треки, чемпионов миникарты и
контекста позиционными списками (`[класс, команда, x, y, …]`): смысл поля
жил в индексе, который по отдельности знали live.py, трекер, контекст матча
и смотрелка, и новое поле в середине молча ломало всех. Здесь у каждой
записи — dataclass с именами полей; модули обмениваются самими записями,
наружу (JSON строки кадра) запись уходит словарём `to_dict`.

Формат строки версионирован: SCHEMA — поле "schema" каждой строки кадра.
Смена имени или смысла поля — новая версия.

Единицы: x, y — доли кадра (0..1); u, v — доли карты, как у рамки камеры
(u вправо, v вниз); время, возраст и таймеры — секунды потока; ХП — доля
0..1. None — значения нет; почему — сказано у поля.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

SCHEMA = 1
# Причина по умолчанию у relation/side: запись прошла мимо sides.assign.
UNSET = "не определялось"


def _r(v: float | None, n: int) -> float | None:
    return None if v is None else round(float(v), n)


def _sides(r) -> dict:
    """Поля отношения и стороны (sides.py): значение или None с причиной;
    причина — только при None."""
    return {"relation": r.relation,
            "relation_reason": None if r.relation is not None else r.relation_reason,
            "side": r.side, "side_reason": None if r.side is not None else r.side_reason,
            "side_src": r.side_src if r.side is not None else None}


@dataclass(frozen=True)
class UnitRec:
    """Юнит детектора на основном виде (unit_net.py)."""
    kind: str                  # champion / minion / ward
    team: str                  # ORDER / CHAOS
    x: float                   # центр тела, доли кадра
    y: float
    hp: float                  # доля ХП с плашки
    score: float               # уверенность детектора
    # Id Data Dragon чемпиона по имени над плашкой (live_champions.who).
    # None — не чемпион, имя не прочиталось или не читалось (вид игрока,
    # нет состава, нет банка имён: это сказано в note сеанса).
    name: str | None = None
    # Отношение к своему игроку и сторона (lolvision/sides.py, #790).
    # `team` выше — сырой ответ детектора: логит «синий = ORDER, красный =
    # CHAOS», выученный на наблюдателе; сторона он только у наблюдателя.
    color: str | None = None              # корзина цвета полоски по пикселям (вид игрока)
    relation: str | None = None           # ally / enemy / self; у наблюдателя None
    relation_reason: str | None = UNSET   # почему relation None
    side: str | None = None               # ORDER / CHAOS
    side_reason: str | None = UNSET       # почему side None
    # Откуда сторона: detector — `team` детектора (вид наблюдателя), api —
    # от команды своего игрока по Live Client API (вид игрока); None при side None.
    side_src: str | None = None

    @classmethod
    def from_unit(cls, u, frame_size: tuple[int, int], name: str | None = None) -> UnitRec:
        """unit_net.Unit (пиксели кадра размера `frame_size` = (w, h)) → запись."""
        w, h = frame_size
        return cls(u.kind, u.team, u.x / w, u.y / h, float(u.hp), float(u.score), name)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "team": self.team, "x": _r(self.x, 4), "y": _r(self.y, 4),
                "hp": _r(self.hp, 2), "score": _r(self.score, 2), "name": self.name,
                "color": self.color, **_sides(self)}


@dataclass(frozen=True)
class TrackRec:
    """Трек юнита (unit_tracker.py): постоянный id между кадрами."""
    id: str                    # top-minion-…, ward-…, champ-ORDER-Volibear
    kind: str
    team: str
    state: str                 # seen / unseen / lost
    u: float                   # место на карте (у невидимого — предсказанное)
    v: float
    x: float | None            # место в кадре; None — трек сейчас не виден
    y: float | None
    hp: float                  # ХП на последнем наблюдении
    age: float                 # с от последнего наблюдения
    # Отношение и сторона трека: первое известное значение среди его
    # наблюдений (трек не склеивает юнитов с разными известными значениями),
    # пока его нет — None с причиной последнего наблюдения.
    relation: str | None = None
    relation_reason: str | None = UNSET
    side: str | None = None
    side_reason: str | None = UNSET
    side_src: str | None = None    # как у UnitRec

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "team": self.team, "state": self.state,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "x": _r(self.x, 4), "y": _r(self.y, 4),
                "hp": _r(self.hp, 2), "age": _r(self.age, 1), **_sides(self)}


@dataclass(frozen=True)
class MinimapChamp:
    """Чемпион состава на миникарте и в панели HUD (live_champions.minimap)."""
    name: str                  # Id Data Dragon
    team: str
    slot: str                  # L1…L5, R1…R5
    visible: bool              # иконка найдена на миникарте
    u: float | None            # место на миникарте; None — не виден
    v: float | None
    alive: bool | None         # по панели HUD; None — панель не видна
    timer: float | None        # таймер возрождения; None — жив или не прочитан

    def to_dict(self) -> dict:
        return {"name": self.name, "team": self.team, "slot": self.slot,
                "visible": self.visible, "u": _r(self.u, 4), "v": _r(self.v, 4),
                "alive": self.alive, "timer": self.timer}


@dataclass(frozen=True)
class ChampCtx:
    """Чемпион в контексте матча (match_state.py) на момент кадра."""
    name: str
    team: str
    state: str                 # visible / remembered / dead / unknown
    u: float | None            # последнее известное место; None — не знаем
    v: float | None
    age: float | None          # с от наблюдения места
    src: str | None            # источник места: minimap / screen / inferred
    hp: float | None           # последняя известная доля ХП; None — не видели
    respawn_in: float | None   # с до возрождения; None — жив или таймер неизвестен
    # Жив ли (True / False), источник (api / hud / minimap / inferred) и
    # возраст наблюдения, с; None — не наблюдалось.
    alive: bool | None = None
    alive_src: str | None = None
    alive_age: float | None = None
    hp_src: str | None = None          # источник ХП: screen / inferred
    hp_age: float | None = None        # с от наблюдения ХП
    respawn_src: str | None = None     # кто дал таймер возрождения: api / hud
    conflicts: int = 0                 # расхождений источников по этому чемпиону (#781)

    def to_dict(self) -> dict:
        return {"name": self.name, "team": self.team, "state": self.state,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "age": _r(self.age, 1),
                "src": self.src, "hp": _r(self.hp, 2), "respawn_in": _r(self.respawn_in, 1),
                "alive": self.alive, "alive_src": self.alive_src,
                "alive_age": _r(self.alive_age, 1), "hp_src": self.hp_src,
                "hp_age": _r(self.hp_age, 1), "respawn_src": self.respawn_src,
                "conflicts": self.conflicts}


# ---------- события ----------

@dataclass(frozen=True)
class Event:
    """Событие кадра; `type` — вид события, в JSON первым полем после t."""
    t: float
    type: ClassVar[str] = ""

    def to_dict(self) -> dict:
        return {"t": self.t, "type": self.type}


@dataclass(frozen=True)
class DeathEvent(Event):
    """Смерть чемпиона (match_state): по панели HUD или по Live Client API."""
    name: str
    team: str
    u: float | None            # где видели последний раз; None — не видели
    v: float | None
    respawn_in: float | None   # таймер (панели или API); None — не прочитан
    src: str = "hud"           # hud / api
    type: ClassVar[str] = "death"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name, "team": self.team,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "respawn_in": self.respawn_in,
                "src": self.src}


@dataclass(frozen=True)
class RespawnEvent(Event):
    """Возрождение чемпиона (match_state)."""
    name: str
    team: str
    src: str                   # hud / minimap / inferred
    type: ClassVar[str] = "respawn"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name, "team": self.team, "src": self.src}


@dataclass(frozen=True)
class RosterEvent(Event):
    """Новый состав — новый матч (match_state)."""
    roster: tuple[str, ...]    # L1…L5, R1…R5
    type: ClassVar[str] = "roster"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "roster": list(self.roster)}


@dataclass(frozen=True)
class UnitNewEvent(Event):
    """Трек юнита подтверждён (unit_tracker)."""
    id: str
    kind: str
    team: str
    # Отношение и сторона трека на момент события; почему None — у трека
    # в tracks той же строки.
    relation: str | None = None
    side: str | None = None
    type: ClassVar[str] = "unit_new"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "id": self.id, "kind": self.kind, "team": self.team,
                "relation": self.relation, "side": self.side}


@dataclass(frozen=True)
class UnitDeathEvent(Event):
    """Юнит исчез в кадре с малым ХП — умер (unit_tracker)."""
    id: str
    kind: str
    team: str
    u: float                   # последнее место на карте
    v: float
    relation: str | None = None    # как у UnitNewEvent; трек умер — в tracks его уже нет,
    side: str | None = None        # причина отказа — в tracks прошлых строк
    type: ClassVar[str] = "unit_death"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "id": self.id, "kind": self.kind, "team": self.team,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "relation": self.relation,
                "side": self.side}


def dicts(recs) -> list[dict]:
    """Записи → список словарей для JSON строки кадра."""
    return [r.to_dict() for r in recs]
