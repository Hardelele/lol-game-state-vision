"""Контекст матча: состояние игры во времени, а не ответ по одному кадру.

Зачем (#780). Модели кадра — тип сцены, камера, юниты, миникарта, состав,
жив ли, имена — отвечают только про текущий кадр и всё забывают. Но часть
сведений сейчас не видна, а пару секунд назад была: чемпион ушёл в туман,
панель закрыл Tab. Часть надо помнить долго (состав, смерти). Часть
выводится, хотя на экране её нет: «мёртв, таймер 18 с» → в такой-то момент
возродится у своего фонтана.

Как. MatchState принимает наблюдения кадра (тип сцены, чемпионов миникарты,
юнитов детектора — записи records.py — и центр камеры) и ведёт по каждому из
десяти чемпионов последнее известное:

* где — с миникарты (точно) или с основного вида: имя над плашкой, точка
  кадра → карта общей проекцией geometry.Projection (гомография с учётом
  наклона камеры плюс центр камеры); без проекции или без ответа камеры
  место с экрана не берётся. У каждого значения время наблюдения и
  источник, поэтому «не виден сейчас» не путается с «нет данных»;
* жив / мёртв — по панелям HUD; таймер возрождения переводится в момент
  возрождения, и если панели не видно, возрождение выводится по времени;
* ХП — с плашки на основном виде.

Правила игры — ограничения: в команде пять разных чемпионов, состав в
матче не меняется (другой состав несколько кадров подряд — новый матч),
мёртвого не видно на миникарте, возрождается он у своего фонтана, и не
раньше, чем кончится его таймер. Последнее важно на деле: панель под
объявлением или баннером на 1–3 кадра читается как «жив» посреди отсчёта
(58w57eJ5Qks, 349–351 с: Jinx и Lulu), такое «возрождение» не
принимается. Смерть без таймера (одиночная ошибка чтения) засчитывается со
второго кадра подряд.

На кадрах не «игра» (Tab, магазин, клиент…) наблюдений нет, а память
остаётся: возраст сведений растёт.

Время — время потока в секундах; часы матча с табло — #761.

Live Client API своей игры (#798, 2B). Если он есть, состав (десять
чемпионов с командами), жив / мёртв с таймером и время матча берутся из
него — источник `api`; lolvision в сеть не ходит, наблюдение приходит
записью ApiObs (перевод из ответа API — lolstream/live_client.py).
Приоритет: API > HUD > зрение. HUD и миникарта при живом API значения не
меняют, а сверяются: расхождение — счётчик `conflicts` по виду (roster /
alive / timer / minimap_dead), а не тихое перетирание. Места на карте API
не даёт — оно по-прежнему с миникарты и экрана. Память матча сбрасывается
только по сильному сигналу API — другой состав или время матча пошло назад
(новая игра); скользящее окно HUD при API её не сбрасывает. API молчит
дольше API_STALE — жив / мёртв снова по HUD и выводу по таймеру, а состав
остаётся из API до новой игры в API.

Свой чемпион на виде игрока (#802). Имена над плашками на виде игрока не
читаются, но свой юнит узнаётся по цвету полоски (UnitRec.relation = self,
sides.py), а кто из десяти свой — говорит API (ApiObs.me). Такой юнит даёт
место своего чемпиона с экрана той же проекцией; два и больше self на
кадре — не выбираем наугад: отказ и conflicts["self_many"]. Если в том же
кадре своя иконка есть на миникарте, место — с миникарты (она не зависит
от ответа камеры), а экран с ней сверяется: дальше SELF_PLACE_TOL —
conflicts["self_place"].

Неполный состав (#803). В Practice Tool API отдаёт только своего игрока:
режим (ApiObs.game_mode, PARTIAL_ROSTER_MODES) разрешает снимок с теми, кто
есть, — состав из одного, время матча, жив / мёртв своего. Иконки миникарты
назначаются только этому составу; остальные детекции (боты) — лишние. Про
отсутствующих ничего не выводится. В остальных режимах неполный снимок —
по-прежнему отказ.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lolvision.geometry import Projection
from lolvision.records import (ChampCtx, DeathEvent, Event, MinimapChamp, RespawnEvent,
                               RosterEvent, UnitRec)

TEAMS = ("ORDER", "CHAOS")
# Карты по mapNumber Live Client API.
RIFT, ABYSS = 11, 12                   # Ущелье призывателей, Воющая бездна (ARAM)
# Фонтаны в долях миникарты (u вправо, v вниз), как у рамки камеры.
FOUNTAIN = {"ORDER": (0.035, 0.965), "CHAOS": (0.965, 0.035)}
# Воющая бездна (#803): миникарта — диагональ от ORDER слева снизу к CHAOS
# справа сверху. CHAOS — замер по кадрам своего экрана (×1.2): возродившийся
# Sylas стоит в (0.941, 0.147) четыре кадра подряд (6 с), камера своего
# возродившегося — (0.90–0.93, 0.12–0.17). ORDER — отражение через центр
# симметрии карты на миникарте (0.526, 0.538): середина нексусов и середина
# пары ингибиторов совпали в пределах 0.01. Сам ORDER на кадрах не проверен.
FOUNTAIN_ABYSS = {"ORDER": (0.111, 0.929), "CHAOS": (0.941, 0.147)}
FOUNTAINS = {RIFT: FOUNTAIN, ABYSS: FOUNTAIN_ABYSS}
# Режимы (gameMode API), где Live Client честно отдаёт не весь состав (#803):
# в Practice Tool allPlayers — только свой игрок, ботов и противников API не
# знает (журнал своей игры data/own/20261012-185628: 1 игрок, Master Yi).
# Решает режим, а не число игроков: в обычной игре (CLASSIC, ARAM, …)
# неполный снимок — сбой или хвост загрузки, и он по-прежнему отвергается.
PARTIAL_ROSTER_MODES = frozenset({"PRACTICETOOL"})
WHY_NO_ENEMIES_PRACTICE = "режим тренировки: API не отдаёт ботов/противников"


def partial_roster(game_mode: str | None) -> bool:
    """Режим, где API отдаёт не весь состав (тренировка)."""
    return game_mode in PARTIAL_ROSTER_MODES


def fountain(team: str, map_number: int | None) -> tuple[float, float] | None:
    """Фонтан команды в долях миникарты на карте `map_number` (mapNumber API).
    None — карта неизвестна API (ролик, наблюдатель): это Ущелье, как в
    обучающих роликах. Другая карта без замера — None: места не выдумываем."""
    if map_number is None:
        return FOUNTAIN[team]
    f = FOUNTAINS.get(map_number)
    return None if f is None else f[team]


def rift(map_number: int | None) -> bool:
    """Ущелье (или карта неизвестна — ролики и наблюдатель только с Ущелья):
    годится всё, что в единицах Ущелья — модель камеры, проекция экрана на
    карту (geometry.MAP_UNITS)."""
    return map_number is None or map_number == RIFT
ROSTER_SWITCH = 3          # кадров с другим составом подряд — новый матч
RESPAWN_SLACK = 2.0        # с: видим «мёртвого» раньше срока — верим глазам
# Время матча по API, на которое оно может «откатиться» без новой игры:
# опросы идут не по порядку только при сбоях часов, новая игра — с нуля.
NEW_GAME_BACKSTEP = 5.0
API_STALE = 5.0            # с без наблюдения API — источник пропал, снова HUD
# с после смены жив / мёртв по API: опрос ~2 Гц, панель HUD отстаёт или
# опережает на долю секунды — расхождение в этом окне не конфликт.
API_SLACK = 1.5
TIMER_SLACK = 2.0          # с: таймер панели против API — дальше конфликт
# Доля карты: место своего чемпиона с экрана против своей иконки миникарты в
# том же кадре — дальше конфликт (#802). Обычная ошибка обоих — иконка p90
# ~150 ед. (minimap-champions.md), камера внутри карты p90 ~160 ед.
# (replay-api.md), точка тела → земля ~50 ед. (entity-model-plan.md 4.2):
# вместе ~0.025 карты (14 800 ед.). 0.05 (~740 ед.) — вдвое выше, но ниже
# грубых промахов камеры (p99 у края — тысячи ед.) и иконки не того.
SELF_PLACE_TOL = 0.05


# ---------- наблюдение Live Client API ----------

@dataclass(frozen=True)
class ApiPlayer:
    """Один из десяти по Live Client API (lolstream.live_client.PlayerRecord)."""
    name: str                  # Id Data Dragon
    team: str                  # ORDER / CHAOS
    dead: bool                 # isDead
    respawn_in: float          # с до возрождения на момент наблюдения; 0 у живого


@dataclass(frozen=True)
class ApiObs:
    """Снимок API на момент наблюдения: время матча, десять игроков (ORDER,
    затем CHAOS), своя команда и свой чемпион (Id Data Dragon) — None, если
    своего игрока нет (наблюдатель в клиенте). В режиме неполного состава
    (`partial`, тренировка) игроков столько, сколько отдал API."""
    game_time: float           # с от начала матча
    players: tuple[ApiPlayer, ...]
    me_team: str | None = None
    me: str | None = None
    map_number: int | None = None   # mapNumber: 11 — Ущелье, 12 — Воющая бездна
    game_mode: str | None = None    # gameMode: CLASSIC, ARAM, PRACTICETOOL, …

    @property
    def roster(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.players)

    @property
    def partial(self) -> bool:
        """Режим неполного состава (PARTIAL_ROSTER_MODES): в составе только
        те, кого отдал API."""
        return partial_roster(self.game_mode)

    def problem(self) -> str | None:
        """Почему снимок не годится как состав матча; None — годится.

        Обычно нужны ровно десять, 5/5: неполный снимок — сбой, а половина
        состава хуже, чем никакого (иконки миникарты назначились бы не тем).
        В режиме неполного состава (тренировка) — от одного до пяти на
        команду, повтор чемпиона только в разных командах (боты), и свой
        игрок обязан быть в составе: без него снимок ничего не говорит."""
        n = Counter(p.team for p in self.players)
        if self.partial:
            if not self.players:
                return f"игроков нет ({self.game_mode})"
            if n["ORDER"] > 5 or n["CHAOS"] > 5:
                return f"по командам {n['ORDER']}/{n['CHAOS']}: больше пяти"
            if self.me is None:
                return f"своего игрока нет ({self.game_mode})"
        elif len(self.players) != 10:
            return f"игроков {len(self.players)}, а не 10"
        bad = sorted({p.team for p in self.players} - set(TEAMS))
        if bad:
            return f"команды не ORDER/CHAOS: {bad}"
        if not self.partial and (n["ORDER"] != 5 or n["CHAOS"] != 5):
            return f"по командам {n['ORDER']}/{n['CHAOS']}, а не 5/5"
        if self.partial:
            # В тренировке боты бывают теми же чемпионами, что и в другой
            # команде (живой Practice Tool: Malphite в обеих) — это не сбой.
            if len({(p.name, p.team) for p in self.players}) != len(self.players):
                return "чемпион повторяется в одной команде"
        elif len(set(self.roster)) != len(self.players):
            return "чемпионы повторяются"
        if self.me_team is not None and self.me_team not in TEAMS:
            return f"своя команда {self.me_team!r}"
        if self.me is not None:
            same = [p for p in self.players if p.name == self.me]
            mine = next((p for p in same if p.team == self.me_team), same[0] if same else None)
            if mine is None:
                return f"своего чемпиона {self.me!r} нет в составе"
            if self.me_team is not None and mine.team != self.me_team:
                return f"свой чемпион {self.me!r} в {mine.team}, а своя команда {self.me_team}"
        return None


@dataclass
class Fact:
    """Значение с временем наблюдения и источником."""
    value: object
    t: float
    src: str                # api / minimap / screen / hud / inferred


@dataclass
class Champ:
    name: str
    team: str
    slot: str
    pos: Fact | None = None          # (u, v) в долях миникарты
    alive: Fact | None = None        # True / False
    respawn_at: float | None = None  # момент возрождения (время потока)
    respawn_src: str | None = None   # кто дал момент возрождения: api / hud
    hp: Fact | None = None           # доля ХП с плашки
    seen: float | None = None        # последний раз видели (миникарта или экран)
    dead_n: int = 0                  # кадров подряд «мёртв» без таймера
    conflicts: int = 0               # наблюдений против выведенного или API (#781)
    api_change: float | None = None  # когда API сменил жив / мёртв

    def status(self, t: float) -> str:
        if self.alive is not None and self.alive.value is False:
            return "dead"
        if self.seen is not None and self.seen >= t:
            return "visible"
        if self.pos is not None:
            return "remembered"
        return "unknown"


@dataclass
class MatchState:
    """Состояние матча; `update` на каждый кадр, `frame_view` — снимок кадра."""
    # Кадр → карта (та же, что у трекера юнитов); None — места с экрана нет.
    proj: Projection | None = None
    champs: dict[str, Champ] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    t: float = 0.0
    # Live Client API: последнее принятое наблюдение и его время (время потока).
    api: ApiObs | None = None
    api_t: float | None = None
    api_refused: str | None = None   # почему последний снимок API не принят
    game_time: Fact | None = None    # время матча из API на момент api_t
    # Расхождения источников по виду (#781): roster / alive / timer /
    # minimap_dead — против API, hud_flicker — панель против таймера,
    # self_many — несколько своих юнитов на кадре, self_place — место
    # своего с экрана против своей иконки миникарты (#802).
    conflicts: Counter = field(default_factory=Counter)
    # Чемпионы, которые в составе API в обеих командах (тренировка с ботами,
    # #803): у них ключ в `champs` / `order` — «имя/команда», у остальных — имя.
    _dup: frozenset = frozenset()
    _other: tuple | None = None
    _other_n: int = 0

    # ---------- наблюдения ----------

    @property
    def map_number(self) -> int | None:
        """Карта по последнему принятому снимку API; None — API не было."""
        return self.api.map_number if self.api is not None else None

    def api_live(self, t: float | None = None) -> bool:
        """Есть ли свежее наблюдение API (не старше API_STALE)."""
        t = self.t if t is None else t
        return self.api_t is not None and t - self.api_t <= API_STALE

    def update(self, t: float, kind: str = "game", champs: list[MinimapChamp] | None = None,
               units: list[UnitRec] | None = None,
               cam: tuple[float | None, float | None] | None = None,
               api: ApiObs | None = None) -> list[Event]:
        """Наблюдения одного кадра → новые события этого кадра. `kind` — тип
        сцены, `cam` — центр камеры (доли карты) или None — нет ответа,
        `api` — снимок Live Client API на момент `t` (None — нового нет).
        API — не экран: он принимается при любом типе сцены."""
        self.t = t
        new: list[Event] = []
        if api is not None:
            self._from_api(t, api, new)
        # Магазин и Tab — игра под оверлеем (#786): миникарта видна, а юниты
        # сцены — нет (окно закрывает их, это не «ушли из вида»).
        if kind in ("game", "shop", "tab") and champs:
            self._roster(t, champs, new)
            self._from_champs(t, champs, new)
        # Юниты — и без миникарты на кадре: состав уже известен (из API или
        # прошлых кадров), а свой чемпион узнаётся по полоске (#802).
        if kind == "game" and units and self.order:
            self._from_units(t, units, cam)
        self._infer(t, new)
        self.events += new
        return new

    def _key(self, name: str, team: str) -> str:
        """Ключ чемпиона в `champs` / `order`: имя, а у повторённого в обеих
        командах (тренировка, #803) — «имя/команда»."""
        return f"{name}/{team}" if name in self._dup else name

    def _keys(self, champs) -> tuple[str, ...]:
        return tuple(self._key(c.name, c.team) for c in champs)

    def _roster(self, t: float, champs: list[MinimapChamp], new: list[Event]) -> None:
        if self.api is not None:
            # Состав дал API: HUD его не меняет, а сверяется с ним.
            if _pairs(champs) != _pairs(self.champs.values()):
                self._conflict(None, "roster")
            return
        roster = tuple(c.name for c in champs)
        if roster == tuple(self.order):
            self._other, self._other_n = None, 0
            return
        if not self.order:
            self._reset(champs)
            return
        # Другой состав — может быть ошибка чтения на одном кадре; новый
        # матч — только если он держится несколько кадров подряд.
        self._other_n = self._other_n + 1 if roster == self._other else 1
        self._other = roster
        if self._other_n >= ROSTER_SWITCH:
            self._reset(champs)
            new.append(RosterEvent(t, roster))

    def _reset(self, champs: list[MinimapChamp]) -> None:
        self._dup = frozenset()
        self.order = [c.name for c in champs]
        self.champs = {c.name: Champ(c.name, c.team, c.slot) for c in champs}
        self._other, self._other_n = None, 0

    def _conflict(self, c: Champ | None, kind: str) -> None:
        self.conflicts[kind] += 1
        if c is not None:
            c.conflicts += 1

    # ---------- Live Client API ----------

    def _from_api(self, t: float, obs: ApiObs, new: list[Event]) -> None:
        why = obs.problem()
        if why is not None:
            self.api_refused = why          # отказ, а не половина состава
            return
        self.api_refused = None
        pairs, prev = _pairs(obs.players), self.api
        if prev is not None:
            if pairs != _pairs(prev.players) or \
                    obs.game_time < prev.game_time - NEW_GAME_BACKSTEP:
                self._reset_api(obs)        # новая игра: другой состав или время с нуля
                new.append(RosterEvent(t, obs.roster))
        elif not self.order:
            self._reset_api(obs)
        elif pairs != _pairs(self.champs.values()):
            # Состав по HUD до API — ошибка чтения или прошлый матч: верим API.
            self._conflict(None, "roster")
            self._reset_api(obs)
            new.append(RosterEvent(t, obs.roster))
        else:
            self.order = list(self._keys(obs.players))   # тот же состав: память остаётся
        self.api, self.api_t = obs, t
        self.game_time = Fact(obs.game_time, t, "api")
        for p in obs.players:
            c = self.champs[self._key(p.name, p.team)]
            was_dead = c.alive is not None and c.alive.value is False
            if p.dead:
                c.respawn_at, c.respawn_src, c.dead_n = t + p.respawn_in, "api", 0
                if not was_dead:
                    c.api_change = t
                    new.append(DeathEvent(t, c.name, c.team,
                                          c.pos.value[0] if c.pos else None,
                                          c.pos.value[1] if c.pos else None,
                                          round(p.respawn_in, 1), src="api"))
                c.alive = Fact(False, t, "api")
            elif was_dead:
                c.api_change = t
                self._respawn(c, t, "api", new)
            else:
                c.alive = Fact(True, t, "api")

    def _reset_api(self, obs: ApiObs) -> None:
        names = Counter(obs.roster)
        self._dup = frozenset(k for k, v in names.items() if v > 1)
        self.order = list(self._keys(obs.players))
        n: Counter = Counter()
        self.champs = {}
        for p in obs.players:
            n[p.team] += 1
            slot = f"{'LR'[TEAMS.index(p.team)]}{n[p.team]}"
            self.champs[self._key(p.name, p.team)] = Champ(p.name, p.team, slot)
        self._other, self._other_n = None, 0

    def _check_hud(self, c: Champ, m: MinimapChamp, t: float) -> None:
        """Панель HUD против API: только счётчик, значение остаётся из API."""
        if m.alive is None or c.alive is None:
            return
        if c.api_change is not None and t - c.api_change < API_SLACK:
            return                          # опрос API и кадр разошлись на долю секунды
        if m.alive != c.alive.value:
            self._conflict(c, "alive")
        elif m.alive is False and m.timer is not None and c.respawn_at is not None \
                and abs((c.respawn_at - t) - float(m.timer)) > TIMER_SLACK:
            self._conflict(c, "timer")

    # ---------- кадр ----------

    def _from_champs(self, t: float, champs: list[MinimapChamp], new: list[Event]) -> None:
        if self.api is not None:
            if _pairs(champs) != _pairs(self.champs.values()):
                return                  # не тот состав: сверка уже в _roster
        elif tuple(m.name for m in champs) != tuple(self.order):
            return                      # кадр с чужим составом не наблюдение
        if self.api_live(t):
            self._from_champs_api(t, champs)
            return
        for m in champs:
            c, alive, timer = self.champs[self._key(m.name, m.team)], m.alive, m.timer
            if m.visible and m.u is not None:
                c.pos, c.seen = Fact((m.u, m.v), t, "minimap"), t
                if c.alive is not None and c.alive.value is False and \
                        (c.respawn_at is None or t >= c.respawn_at - RESPAWN_SLACK):
                    self._respawn(c, t, "minimap", new)
            if alive is False:
                was_dead = c.alive is not None and c.alive.value is False
                c.dead_n = 0 if timer is not None else c.dead_n + 1
                if timer is None and not was_dead and c.dead_n < 2:
                    continue
                c.alive = Fact(False, t, "hud")
                if timer is not None:
                    c.respawn_at, c.respawn_src = t + float(timer), "hud"
                if not was_dead:
                    new.append(DeathEvent(t, c.name, c.team,
                                          c.pos.value[0] if c.pos else None,
                                          c.pos.value[1] if c.pos else None, timer, src="hud"))
            elif alive is True:
                c.dead_n = 0
                if c.alive is not None and c.alive.value is False:
                    if c.respawn_at is not None and t < c.respawn_at - RESPAWN_SLACK:
                        self._conflict(c, "hud_flicker")   # «жив» посреди отсчёта: не верим
                        continue
                    self._respawn(c, t, "hud", new)
                else:
                    c.alive = Fact(True, t, "hud")

    def _from_champs_api(self, t: float, champs: list[MinimapChamp]) -> None:
        """Миникарта и панели при живом API: жив / мёртв — из API, отсюда
        только место (иконка) и сверка."""
        for m in champs:
            c = self.champs[self._key(m.name, m.team)]
            self._check_hud(c, m, t)
            if not (m.visible and m.u is not None):
                continue
            if c.alive is not None and c.alive.value is False and \
                    (c.respawn_at is None or t < c.respawn_at - RESPAWN_SLACK):
                # Мёртвого на миникарте нет: иконка назначена не тому.
                self._conflict(c, "minimap_dead")
                continue
            c.pos, c.seen = Fact((m.u, m.v), t, "minimap"), t

    def _from_units(self, t: float, units: list[UnitRec],
                    cam: tuple[float | None, float | None] | None) -> None:
        """Чемпионы на основном виде: опознанные по имени над плашкой и свой
        по цвету полоски (relation self, #802). Виден сейчас, ХП и место на
        карте — через проекцию кадра при этом центре камеры. Нет проекции или
        ответа камеры — место остаётся прежним: выдумывать его не из чего."""
        px, py = cam if cam is not None else (None, None)
        # Проекция экрана — в единицах Ущелья (MAP_UNITS): на другой карте
        # место с экрана вышло бы в чужом масштабе — его не берём.
        placeable = (self.proj is not None and px is not None and py is not None
                     and rift(self.map_number))
        where = (px, py) if placeable else None
        mine = [u for u in units if u.kind == "champion" and u.relation == "self"]
        # Юнит с именем повторённого чемпиона (тренировка, #803) не ставим: по
        # имени не сказать, который из двух, — ключа «имя» у него в champs нет.
        for u in units:
            if u.kind != "champion" or u.relation == "self" or u.name not in self.champs:
                continue
            self._seen_on_screen(self.champs[u.name], u, t, where)
        me = (self._key(self.api.me, self.api.me_team or "")
              if self.api is not None and self.api.me is not None else None)
        if not mine or me not in self.champs:
            return                      # свой чемпион неизвестен — некого ставить
        if len(mine) > 1:
            # Своя полоска одна; несколько — проба цвета ошиблась (табло,
            # эффект): который настоящий, не знаем — отказ, а не догадка.
            self._conflict(None, "self_many")
            return
        self._seen_on_screen(self.champs[me], mine[0], t, where, check_minimap=True)

    def _seen_on_screen(self, c: Champ, u: UnitRec, t: float,
                        where: tuple[float, float] | None, check_minimap: bool = False) -> None:
        """Чемпион `c` виден на экране юнитом `u`: виден, ХП и место с экрана,
        если в этом кадре нет места с миникарты — она точнее (не зависит от
        ответа камеры). `check_minimap` — сверить место с экрана с иконкой."""
        if c.alive is not None and c.alive.value is False:
            return                      # мёртвого на экране нет: опознан не тот
        c.seen, c.hp = t, Fact(u.hp, t, "screen")
        if where is None:
            return
        uv = self.proj.frac_to_map([[u.x, u.y]], where)[0]
        uv = (float(uv[0]), float(uv[1]))
        if c.pos is not None and c.pos.src == "minimap" and c.pos.t >= t:
            off = np.hypot(uv[0] - c.pos.value[0], uv[1] - c.pos.value[1])
            if check_minimap and off > SELF_PLACE_TOL:
                self._conflict(c, "self_place")
            return
        c.pos = Fact(uv, t, "screen")

    def _infer(self, t: float, new: list[Event]) -> None:
        if self.api_live(t):
            return                      # возрождение скажет API, выводить незачем
        for c in self.champs.values():
            dead = c.alive is not None and c.alive.value is False
            if dead and c.respawn_at is not None and t >= c.respawn_at:
                self._respawn(c, t, "inferred", new)

    def _respawn(self, c: Champ, t: float, src: str, new: list[Event]) -> None:
        c.alive, c.respawn_at, c.respawn_src = Fact(True, t, src), None, None
        if src != "minimap":
            # Возрождение — у своего фонтана; место выведено, а не увидено.
            # Фонтан неизвестной карты — места нет: прежнее (где умер) неверно.
            f = fountain(c.team, self.map_number)
            c.pos = Fact(f, t, "inferred") if f is not None else None
        c.hp = Fact(1.0, t, "inferred")
        new.append(RespawnEvent(t, c.name, c.team, src))

    # ---------- выход ----------

    @property
    def partial(self) -> bool:
        """Состав из API в режиме неполного состава (тренировка, #803)."""
        return self.api is not None and self.api.partial

    @property
    def roster_src(self) -> str | None:
        """Откуда состав: api — из Live Client API, hud — по портретам HUD
        (миникарта live_champions), None — состава нет."""
        if not self.order:
            return None
        return "api" if self.api is not None else "hud"

    def game_clock(self) -> Fact | None:
        """Время матча на момент кадра по API, с; None — API нет или он пропал."""
        if self.game_time is None or not self.api_live():
            return None
        return Fact(self.game_time.value + (self.t - self.game_time.t), self.t, "api")

    def frame_view(self) -> list[ChampCtx]:
        """Снимок для строки кадра, по составу."""
        t = self.t
        out = []
        for key in self.order:
            c = self.champs[key]
            p, a, hp = c.pos, c.alive, c.hp
            # Между опросами API момент возрождения может пройти раньше, чем
            # API скажет «жив»: до возрождения тогда 0, а не минус.
            out.append(ChampCtx(c.name, c.team, c.status(t),
                                p.value[0] if p else None, p.value[1] if p else None,
                                t - p.t if p else None, p.src if p else None,
                                float(hp.value) if hp else None,
                                max(0.0, c.respawn_at - t) if c.respawn_at is not None
                                else None,
                                alive=bool(a.value) if a else None,
                                alive_src=a.src if a else None,
                                alive_age=t - a.t if a else None,
                                hp_src=hp.src if hp else None,
                                hp_age=t - hp.t if hp else None,
                                respawn_src=c.respawn_src if c.respawn_at is not None
                                else None,
                                conflicts=c.conflicts))
        return out


def _pairs(champs) -> set[tuple[str, str]]:
    """Состав как множество (чемпион, команда): порядок слотов не важен."""
    return {(c.name, c.team) for c in champs}
