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
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from lolvision.geometry import Projection
from lolvision.records import (ChampCtx, DeathEvent, Event, MinimapChamp, RespawnEvent,
                               RosterEvent, UnitRec)

TEAMS = ("ORDER", "CHAOS")
# Фонтаны в долях миникарты (u вправо, v вниз), как у рамки камеры.
FOUNTAIN = {"ORDER": (0.035, 0.965), "CHAOS": (0.965, 0.035)}
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
    затем CHAOS), своя команда — None, если своего игрока нет (наблюдатель
    в клиенте)."""
    game_time: float           # с от начала матча
    players: tuple[ApiPlayer, ...]
    me_team: str | None = None

    @property
    def roster(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.players)

    def problem(self) -> str | None:
        """Почему снимок не годится как состав матча; None — годится."""
        if len(self.players) != 10:
            return f"игроков {len(self.players)}, а не 10"
        bad = sorted({p.team for p in self.players} - set(TEAMS))
        if bad:
            return f"команды не ORDER/CHAOS: {bad}"
        n = Counter(p.team for p in self.players)
        if n["ORDER"] != 5 or n["CHAOS"] != 5:
            return f"по командам {n['ORDER']}/{n['CHAOS']}, а не 5/5"
        if len(set(self.roster)) != 10:
            return "чемпионы повторяются"
        if self.me_team is not None and self.me_team not in TEAMS:
            return f"своя команда {self.me_team!r}"
        return None


@dataclass(frozen=True)
class SourcedDeath(DeathEvent):
    """Смерть с источником; у DeathEvent источник — панель HUD."""
    src: str = "api"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "src": self.src}


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
    # minimap_dead — против API, hud_flicker — панель против таймера.
    conflicts: Counter = field(default_factory=Counter)
    _other: tuple | None = None
    _other_n: int = 0

    # ---------- наблюдения ----------

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
            if kind == "game":
                self._from_units(t, units or [], cam)
        self._infer(t, new)
        self.events += new
        return new

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
            self.order = list(obs.roster)   # тот же состав: память остаётся
        self.api, self.api_t = obs, t
        self.game_time = Fact(obs.game_time, t, "api")
        for p in obs.players:
            c = self.champs[p.name]
            was_dead = c.alive is not None and c.alive.value is False
            if p.dead:
                c.respawn_at, c.dead_n = t + p.respawn_in, 0
                if not was_dead:
                    c.api_change = t
                    new.append(SourcedDeath(t, c.name, c.team,
                                            c.pos.value[0] if c.pos else None,
                                            c.pos.value[1] if c.pos else None,
                                            round(p.respawn_in, 1), "api"))
                c.alive = Fact(False, t, "api")
            elif was_dead:
                c.api_change = t
                self._respawn(c, t, "api", new)
            else:
                c.alive = Fact(True, t, "api")

    def _reset_api(self, obs: ApiObs) -> None:
        self.order = list(obs.roster)
        n: Counter = Counter()
        self.champs = {}
        for p in obs.players:
            n[p.team] += 1
            slot = f"{'LR'[TEAMS.index(p.team)]}{n[p.team]}"
            self.champs[p.name] = Champ(p.name, p.team, slot)
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
            c, alive, timer = self.champs[m.name], m.alive, m.timer
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
                    c.respawn_at = t + float(timer)
                if not was_dead:
                    new.append(DeathEvent(t, c.name, c.team,
                                          c.pos.value[0] if c.pos else None,
                                          c.pos.value[1] if c.pos else None, timer))
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
            c = self.champs[m.name]
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
        """Чемпионы на основном виде с именем: виден сейчас, ХП и место на
        карте — через проекцию кадра при этом центре камеры. Нет проекции или
        ответа камеры — место остаётся прежним: выдумывать его не из чего."""
        px, py = cam if cam is not None else (None, None)
        placeable = self.proj is not None and px is not None and py is not None
        for u in units:
            if u.kind != "champion" or u.name not in self.champs:
                continue
            c = self.champs[u.name]
            if c.alive is not None and c.alive.value is False:
                continue                # мёртвого на экране нет: имя спутано
            c.seen, c.hp = t, Fact(u.hp, t, "screen")
            minimap_now = c.pos is not None and c.pos.src == "minimap" and c.pos.t >= t
            if placeable and not minimap_now:
                uv = self.proj.frac_to_map([[u.x, u.y]], (px, py))[0]
                c.pos = Fact((float(uv[0]), float(uv[1])), t, "screen")

    def _infer(self, t: float, new: list[Event]) -> None:
        if self.api_live(t):
            return                      # возрождение скажет API, выводить незачем
        for c in self.champs.values():
            dead = c.alive is not None and c.alive.value is False
            if dead and c.respawn_at is not None and t >= c.respawn_at:
                self._respawn(c, t, "inferred", new)

    def _respawn(self, c: Champ, t: float, src: str, new: list[Event]) -> None:
        c.alive, c.respawn_at = Fact(True, t, src), None
        if src != "minimap":
            # Возрождение — у своего фонтана; место выведено, а не увидено.
            c.pos = Fact(FOUNTAIN[c.team], t, "inferred")
        c.hp = Fact(1.0, t, "inferred")
        new.append(RespawnEvent(t, c.name, c.team, src))

    # ---------- выход ----------

    def game_clock(self) -> Fact | None:
        """Время матча на момент кадра по API, с; None — API нет или он пропал."""
        if self.game_time is None or not self.api_live():
            return None
        return Fact(self.game_time.value + (self.t - self.game_time.t), self.t, "api")

    def frame_view(self) -> list[ChampCtx]:
        """Снимок для строки кадра, по составу."""
        t = self.t
        out = []
        for name in self.order:
            c = self.champs[name]
            p = c.pos
            # Между опросами API момент возрождения может пройти раньше, чем
            # API скажет «жив»: до возрождения тогда 0, а не минус.
            out.append(ChampCtx(name, c.team, c.status(t),
                                p.value[0] if p else None, p.value[1] if p else None,
                                t - p.t if p else None, p.src if p else None,
                                float(c.hp.value) if c.hp else None,
                                max(0.0, c.respawn_at - t) if c.respawn_at is not None
                                else None))
        return out


def _pairs(champs) -> set[tuple[str, str]]:
    """Состав как множество (чемпион, команда): порядок слотов не важен."""
    return {(c.name, c.team) for c in champs}
