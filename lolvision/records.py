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


def _r(v: float | None, n: int) -> float | None:
    return None if v is None else round(float(v), n)


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

    @classmethod
    def from_unit(cls, u, frame_size: tuple[int, int], name: str | None = None) -> UnitRec:
        """unit_net.Unit (пиксели кадра размера `frame_size` = (w, h)) → запись."""
        w, h = frame_size
        return cls(u.kind, u.team, u.x / w, u.y / h, float(u.hp), float(u.score), name)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "team": self.team, "x": _r(self.x, 4), "y": _r(self.y, 4),
                "hp": _r(self.hp, 2), "score": _r(self.score, 2), "name": self.name}


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

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "team": self.team, "state": self.state,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "x": _r(self.x, 4), "y": _r(self.y, 4),
                "hp": _r(self.hp, 2), "age": _r(self.age, 1)}


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

    def to_dict(self) -> dict:
        return {"name": self.name, "team": self.team, "state": self.state,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "age": _r(self.age, 1),
                "src": self.src, "hp": _r(self.hp, 2), "respawn_in": _r(self.respawn_in, 1)}


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
    """Смерть чемпиона по панели HUD (match_state)."""
    name: str
    team: str
    u: float | None            # где видели последний раз; None — не видели
    v: float | None
    respawn_in: float | None   # таймер с панели; None — не прочитан
    type: ClassVar[str] = "death"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "name": self.name, "team": self.team,
                "u": _r(self.u, 4), "v": _r(self.v, 4), "respawn_in": self.respawn_in}


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
    type: ClassVar[str] = "unit_new"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "id": self.id, "kind": self.kind, "team": self.team}


@dataclass(frozen=True)
class UnitDeathEvent(Event):
    """Юнит исчез в кадре с малым ХП — умер (unit_tracker)."""
    id: str
    kind: str
    team: str
    u: float                   # последнее место на карте
    v: float
    type: ClassVar[str] = "unit_death"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "id": self.id, "kind": self.kind, "team": self.team,
                "u": _r(self.u, 4), "v": _r(self.v, 4)}


def dicts(recs) -> list[dict]:
    """Записи → список словарей для JSON строки кадра."""
    return [r.to_dict() for r in recs]
