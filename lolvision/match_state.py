"""Контекст матча: состояние игры во времени, а не ответ по одному кадру.

Зачем (#780). Модели кадра — тип сцены, камера, юниты, миникарта, состав,
жив ли, имена — отвечают только про текущий кадр и всё забывают. Но часть
сведений сейчас не видна, а пару секунд назад была: чемпион ушёл в туман,
панель закрыл Tab. Часть надо помнить долго (состав, смерти). Часть
выводится, хотя на экране её нет: «мёртв, таймер 18 с» → в такой-то момент
возродится у своего фонтана.

Как. MatchState принимает наблюдения кадра (строку live: поля kind, champs,
units, px/py) и ведёт по каждому из десяти чемпионов последнее известное:

* где — с миникарты (точно) или с основного вида (имя над плашкой + камера,
  грубо: без учёта перспективы); у каждого значения время наблюдения и
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
"""

from __future__ import annotations

from dataclasses import dataclass, field

TEAMS = ("ORDER", "CHAOS")
# Фонтаны в долях миникарты (u вправо, v вниз), как у рамки камеры.
FOUNTAIN = {"ORDER": (0.035, 0.965), "CHAOS": (0.965, 0.035)}
ROSTER_SWITCH = 3          # кадров с другим составом подряд — новый матч
RESPAWN_SLACK = 2.0        # с: видим «мёртвого» раньше срока — верим глазам


@dataclass
class Fact:
    """Значение с временем наблюдения и источником."""
    value: object
    t: float
    src: str                # minimap / screen / hud / inferred


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
    conflicts: int = 0               # наблюдений против выведенного (для #781)

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
    view: tuple[float, float] = (0.27, 0.15)   # размер вида камеры, доли миникарты
    champs: dict[str, Champ] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    t: float = 0.0
    _other: tuple | None = None
    _other_n: int = 0

    # ---------- наблюдения ----------

    def update(self, t: float, row: dict) -> list[dict]:
        """Наблюдения одного кадра → новые события этого кадра."""
        self.t = t
        new: list[dict] = []
        kind = row.get("kind", "game")
        champs = row.get("champs")
        if kind == "game" and champs:
            self._roster(t, champs, new)
            self._from_champs(t, champs, new)
            self._from_units(t, row)
        self._infer(t, new)
        self.events += new
        return new

    def _roster(self, t: float, champs: list, new: list) -> None:
        roster = tuple(c[0] for c in champs)
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
            new.append({"t": t, "type": "roster", "roster": list(roster)})

    def _reset(self, champs: list) -> None:
        self.order = [c[0] for c in champs]
        self.champs = {c[0]: Champ(c[0], c[1], c[2]) for c in champs}
        self._other, self._other_n = None, 0

    def _from_champs(self, t: float, champs: list, new: list) -> None:
        if tuple(c[0] for c in champs) != tuple(self.order):
            return                      # кадр с чужим составом не наблюдение
        for name, team, slot, vis, u, v, alive, timer in champs:
            c = self.champs[name]
            if vis and u is not None:
                c.pos, c.seen = Fact((u, v), t, "minimap"), t
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
                    new.append({"t": t, "type": "death", "name": name, "team": team,
                                "at": list(c.pos.value) if c.pos else None,
                                "respawn_in": timer})
            elif alive is True:
                c.dead_n = 0
                if c.alive is not None and c.alive.value is False:
                    if c.respawn_at is not None and t < c.respawn_at - RESPAWN_SLACK:
                        c.conflicts += 1       # «жив» посреди отсчёта: не верим
                        continue
                    self._respawn(c, t, "hud", new)
                else:
                    c.alive = Fact(True, t, "hud")

    def _from_units(self, t: float, row: dict) -> None:
        """Чемпионы на основном виде с именем: виден сейчас, ХП, грубое место
        (центр камеры + сдвиг от центра экрана в размерах вида)."""
        px, py = row.get("px"), row.get("py")
        for u in row.get("units") or []:
            if u[0] != "champion" or len(u) < 7 or u[6] not in self.champs:
                continue
            c = self.champs[u[6]]
            if c.alive is not None and c.alive.value is False:
                continue                # мёртвого на экране нет: имя спутано
            c.seen, c.hp = t, Fact(u[4], t, "screen")
            minimap_now = c.pos is not None and c.pos.src == "minimap" and c.pos.t >= t
            if px is not None and py is not None and not minimap_now:
                c.pos = Fact((px + (u[2] - 0.5) * self.view[0],
                              py + (u[3] - 0.5) * self.view[1]), t, "screen")

    def _infer(self, t: float, new: list) -> None:
        for c in self.champs.values():
            dead = c.alive is not None and c.alive.value is False
            if dead and c.respawn_at is not None and t >= c.respawn_at:
                self._respawn(c, t, "inferred", new)

    def _respawn(self, c: Champ, t: float, src: str, new: list) -> None:
        c.alive, c.respawn_at = Fact(True, t, src), None
        if src != "minimap":
            # Возрождение — у своего фонтана; место выведено, а не увидено.
            c.pos = Fact(FOUNTAIN[c.team], t, "inferred")
        c.hp = Fact(1.0, t, "inferred")
        new.append({"t": t, "type": "respawn", "name": c.name, "team": c.team, "src": src})

    # ---------- выход ----------

    def frame_view(self) -> list[list]:
        """Снимок для строки кадра, по составу: [чемпион, команда, состояние,
        u, v, возраст места (с), источник места, ХП, до возрождения (с)].
        Состояние — visible / remembered / dead / unknown."""
        t = self.t
        out = []
        for name in self.order:
            c = self.champs[name]
            p = c.pos
            out.append([name, c.team, c.status(t),
                        round(p.value[0], 4) if p else None,
                        round(p.value[1], 4) if p else None,
                        round(t - p.t, 1) if p else None,
                        p.src if p else None,
                        round(float(c.hp.value), 2) if c.hp else None,
                        round(c.respawn_at - t, 1) if c.respawn_at is not None else None])
        return out
