"""Постоянные id сущностей на карте: один и тот же юнит на разных кадрах.

Зачем (#782). Детектор юнитов (unit_net.py) отвечает про один кадр: «миньон
в точке (x, y)». Для контекста матча (#780) нужно знать, что это тот же
крип, что секунду назад, найти его снова после короткой потери и отличить
«умер» от «ушёл из кадра». У каждой сущности — свой id, у экземпляра, а не у
места: следующая волна, новый вард на том же месте — новые id.

Как.

* Сопоставление идёт **на карте**, а не на экране: камера двигается, и
  экранные координаты одного крипа на соседних кадрах расходятся на сотни
  пикселей. Точка экрана переводится в точку карты общей проекцией
  geometry.Projection (гомография dataset/layouts/projection.json плюс
  центр камеры — ответ модели камеры).
* Предсказание места — по скорости трека; сопоставление — венгерским
  алгоритмом внутри класса со стробом по расстоянию: юнит не может
  пройти больше, чем позволяет его скорость за прошедшее время. У миньона
  ХП не растёт — рост ХП штрафуется.
* Команда в паре — по сторонам и отношению (sides.py, #790), а не по
  `team` детектора: на виде игрока `team` — «синий или красный» по выучке
  наблюдателя, и своя зелёная плашка получает то ORDER, то CHAOS. Пара
  запрещена, если у юнита и трека известны и различаются side или
  relation; неизвестное не запрещает. Записи, прошедшие мимо
  sides.assign (обе причины UNSET), — как раньше: по `team`.
* Чемпион с именем (live_champions) — трек `champ-<команда>-<имя>`: имя
  важнее геометрии.
* Трек без пары: если он должен быть виден (предсказанное место внутри
  зоны ответов детектора) и ХП было на исходе — **умер**; виден, но не
  найден несколько кадров подряд — **потерян**; вне кадра — **не виден**,
  и живёт, пока срок не выйдет (миньон 20 с, вард 3 мин). Найден снова в
  пределах строба — прежний id.

id: `<линия>-<класс>-<8 знаков>` для миньонов (линия по месту рождения:
top / mid / bot / jungle), `ward-…`, `champ-ORDER-Volibear`.

Ограничение: при 1 кадре/с миньон проходит ~325 ед. — больше шага между
соседями в волне (~100–150 ед.), и пары внутри плотной волны путаются;
для трекинга волн нужен поток 3+ кадр/с.
"""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from lolvision.geometry import MAP_UNITS, Projection
from lolvision.records import UNSET, Event, TrackRec, UnitDeathEvent, UnitNewEvent, UnitRec

# Скорость с запасом, ед./с (миньон 325, чемпион до ~500 с бонусами), и
# базовый строб — ошибка проекции и детектора (~60–100 ед.).
SPEED = {"champion": 600.0, "minion": 420.0, "ward": 0.0}
GATE0 = {"champion": 260.0, "minion": 160.0, "ward": 120.0}
EXPIRE = {"champion": 1e9, "minion": 20.0, "ward": 180.0}   # с без наблюдений
# Ушедших треков в отчёте не больше (#801): миньонов гибнут десятки в
# минуту, а трекер своего экрана живёт часами.
GONE_KEEP = 500
CONFIRM = 2              # пар подряд, чтобы трек стал настоящим (не мусор детектора)
MISS_IN_VIEW = 3         # кадров «должен быть виден, но нет» — потерян
DEATH_HP = 0.45          # ХП к кадру пропажи (по скорости, с которой били), ниже — смерть
HP_RISE_COST = 400.0     # ед. штрафа за рост ХП у миньона на всю шкалу
HP_JUMP = 0.3            # рост ХП миньона больше — заведомо другой крип
ALPHABET = string.ascii_lowercase + string.digits


# Линии как ломаные на миникарте (доли, синяя база внизу слева): верхняя —
# по левому краю и верху, нижняя — по низу и правому краю, средняя —
# диагональ между базами.
LANES = {"top": ((0.08, 0.80), (0.08, 0.08), (0.80, 0.08)),
         "bot": ((0.20, 0.92), (0.92, 0.92), (0.92, 0.20)),
         "mid": ((0.20, 0.80), (0.80, 0.20))}
LANE_DIST = 0.13          # дальше от всех линий — лес


def _seg_dist(p: np.ndarray, a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.hypot(*(p - a - t * (b - a))))


def lane_of(u: float, v: float) -> str:
    """Линия по месту на карте: ближайшая из LANES или лес."""
    p = np.array([u, v])
    d = {k: min(_seg_dist(p, a, b) for a, b in zip(pts, pts[1:])) for k, pts in LANES.items()}
    k = min(d, key=d.get)
    return k if d[k] < LANE_DIST else "jungle"


def new_id(kind: str, team: str, u: float, v: float, name: str | None = None) -> str:
    if kind == "champion" and name:
        return f"champ-{team}-{name}"
    tail = "".join(secrets.choice(ALPHABET) for _ in range(8))
    if kind == "minion":
        return f"{lane_of(u, v)}-minion-{tail}"
    return f"{kind}-{tail}"


@dataclass
class Track:
    id: str
    kind: str
    team: str
    uv: np.ndarray                 # место на карте (доли)
    vel: np.ndarray                # доли карты в секунду
    t: float                       # последнее наблюдение
    born: float
    hp: float
    name: str | None = None
    hp_rate: float = 0.0           # доля ХП в секунду (≤ 0): как быстро крипа бьют
    hits: int = 1
    miss: int = 0                  # кадров подряд «должен быть виден, но нет»
    state: str = "new"             # new / seen / unseen / lost / dead
    xy: tuple[float, float] | None = None   # место в кадре на последнем наблюдении (доли)
    # Отношение и сторона (sides.py): первое известное значение; пока его
    # нет — None и причина последнего наблюдения.
    relation: str | None = None
    relation_reason: str | None = UNSET
    side: str | None = None
    side_reason: str | None = UNSET
    side_src: str | None = None    # detector / api — откуда сторона

    def predict(self, t: float) -> np.ndarray:
        return self.uv + self.vel * (t - self.t)

    def hp_at(self, t: float) -> float:
        """ХП на момент `t` по скорости, с которой крипа били."""
        return self.hp + self.hp_rate * max(0.0, t - self.t)

    def take_sides(self, det: UnitRec) -> None:
        """Отношение и сторона из наблюдения: известное не перетирается."""
        if self.relation is None:
            self.relation, self.relation_reason = det.relation, det.relation_reason
        if self.side is None:
            self.side, self.side_reason, self.side_src = det.side, det.side_reason, det.side_src


def _unassigned(r) -> bool:
    return r.relation_reason == UNSET and r.side_reason == UNSET and \
        r.relation is None and r.side is None


def clash(det: UnitRec, tr: Track) -> bool:
    """Юнит и трек — заведомо разные стороны или отношения."""
    if _unassigned(det) and _unassigned(tr):
        return det.team != tr.team         # записи без sides.assign: как раньше
    return any(a is not None and b is not None and a != b
               for a, b in ((det.side, tr.side), (det.relation, tr.relation)))


@dataclass
class UnitTracker:
    proj: Projection                       # кадр ↔ карта; в ней и размер кадра
    valid: np.ndarray | None = None        # (H, W) 0/1: где детектор отвечает
    tracks: list[Track] = field(default_factory=list)
    gone: list[Track] = field(default_factory=list)   # умершие и истёкшие (для отчёта)

    def reset(self) -> None:
        """Новая игра: активные треки — в ушедшие, следующий кадр с нуля."""
        self.gone.extend(tr for tr in self.tracks if tr.state != "new")
        self.tracks = []
        del self.gone[:-GONE_KEEP]

    def visible(self, uv: np.ndarray, cam) -> np.ndarray:
        """Должна ли точка карты быть видна детектору при этой камере."""
        if not len(uv):
            return np.zeros(0, bool)
        xy = self.proj.to_frame(uv, cam)
        w, h = self.proj.frame_size
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        if self.valid is not None:
            ix = np.clip(xy[:, 0].astype(int), 0, w - 1)
            iy = np.clip(xy[:, 1].astype(int), 0, h - 1)
            ok &= self.valid[iy, ix] > 0
        return ok

    def update(self, t: float, cam: tuple[float, float] | None,
               units: list[UnitRec]) -> list[Event]:
        """Кадр: время (с), центр камеры (доли карты) и юниты детектора →
        события. Без камеры (нет ответа модели) кадр пропускается: на карту
        его не перевести."""
        ev: list[Event] = []
        if cam is None or not np.all(np.isfinite(cam)):
            return ev
        dets = [u for u in units if u.kind in SPEED]
        uv = (self.proj.frac_to_map([[u.x, u.y] for u in dets], cam) if dets
              else np.zeros((0, 2)))
        used = set()
        matched_tracks = set()
        live = [k for k, tr in enumerate(self.tracks) if tr.state not in ("dead",)]
        for kind in SPEED:
            di = [i for i, u in enumerate(dets) if u.kind == kind]
            ti = [k for k in live if self.tracks[k].kind == kind]
            if not di or not ti:
                continue
            cost = np.full((len(di), len(ti)), 1e9)
            for a, i in enumerate(di):
                name = dets[i].name
                for b, k in enumerate(ti):
                    tr = self.tracks[k]
                    if clash(dets[i], tr):
                        continue
                    if name and tr.name:
                        cost[a, b] = 0.0 if name == tr.name else 1e9
                        continue
                    dt = max(0.0, t - tr.t)
                    d = np.hypot(*(uv[i] - tr.predict(t))) * MAP_UNITS
                    gate = GATE0[kind] + SPEED[kind] * dt
                    if d > gate:
                        continue
                    c = d
                    if kind == "minion" and dets[i].hp > tr.hp + HP_JUMP:
                        continue
                    if kind == "minion" and dets[i].hp > tr.hp + 0.1:
                        c += HP_RISE_COST * (dets[i].hp - tr.hp)
                    cost[a, b] = c
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] >= 1e9:
                    continue
                i, k = di[a], ti[b]
                self._observe(self.tracks[k], t, uv[i], dets[i], ev)
                used.add(i)
                matched_tracks.add(k)
        # Треки без пары: умер / потерян / не виден.
        vis = self.visible(np.array([tr.predict(t) for tr in self.tracks]).reshape(-1, 2), cam)
        for k, tr in enumerate(self.tracks):
            if k in matched_tracks or tr.state == "dead":
                continue
            if vis[k]:
                tr.miss += 1
                if tr.kind == "minion" and tr.hp_at(t) <= DEATH_HP and tr.state != "new":
                    tr.state = "dead"
                    ev.append(UnitDeathEvent(t, tr.id, tr.kind, tr.team,
                                             float(tr.uv[0]), float(tr.uv[1]),
                                             tr.relation, tr.side))
                elif tr.miss >= MISS_IN_VIEW:
                    tr.state = "lost"
            else:
                tr.state = "unseen" if tr.state != "new" else "new"
        # Новые треки из детекций без пары.
        for i, u in enumerate(dets):
            if i in used:
                continue
            if u.name and any(tr.name == u.name for tr in self.tracks):
                continue                    # чемпион с этим именем уже есть
            tr = Track(new_id(u.kind, u.team, *uv[i], u.name), u.kind, u.team,
                       uv[i].copy(), np.zeros(2), t, t, float(u.hp), u.name,
                       xy=(float(u.x), float(u.y)))
            tr.take_sides(u)
            self.tracks.append(tr)
        # Сроки: умершие и давно не виденные уходят из активных.
        keep = []
        for tr in self.tracks:
            old = t - tr.t > EXPIRE[tr.kind] or (tr.state == "new" and t - tr.t > 3.0)
            if tr.state == "dead" or old:
                if tr.state != "new":
                    self.gone.append(tr)
                continue
            keep.append(tr)
        self.tracks = keep
        del self.gone[:-GONE_KEEP]
        return ev

    def _observe(self, tr: Track, t: float, uv: np.ndarray, det: UnitRec,
                 ev: list[Event]) -> None:
        dt = t - tr.t
        if dt > 0:
            v = (uv - tr.uv) / dt
            # Скорость сглаживается: одно наблюдение с ошибкой проекции не
            # должно развернуть трек.
            tr.vel = v if tr.hits == 1 else 0.5 * tr.vel + 0.5 * v
            if np.hypot(*tr.vel) * MAP_UNITS > SPEED[tr.kind] * 1.5:
                tr.vel = np.zeros(2)
        if dt > 0:
            rate = min(0.0, (float(det.hp) - tr.hp) / dt)
            tr.hp_rate = rate if tr.hits == 1 else 0.5 * tr.hp_rate + 0.5 * rate
        tr.uv, tr.t, tr.hp = uv.copy(), t, float(det.hp)
        tr.xy = (float(det.x), float(det.y))
        tr.hits += 1
        tr.miss = 0
        tr.take_sides(det)
        if det.name and not tr.name:
            # Имя прочиталось позже рождения трека: id становится именным —
            # один раз, дальше он не меняется.
            tr.name = det.name
            tr.id = new_id(tr.kind, tr.team, *tr.uv, tr.name)
        if tr.state == "new" and tr.hits >= CONFIRM:
            tr.state = "seen"
            ev.append(UnitNewEvent(t, tr.id, tr.kind, tr.team, tr.relation, tr.side))
        elif tr.state != "new":
            tr.state = "seen"

    def frame_view(self, t: float) -> list[TrackRec]:
        """Активные треки для строки кадра. Неподтверждённые («new») не
        выдаются: это может быть мусор детектора."""
        out = []
        for tr in self.tracks:
            if tr.state == "new":
                continue
            p = tr.predict(t) if tr.state != "seen" else tr.uv
            xy = tr.xy if tr.state == "seen" else None
            out.append(TrackRec(tr.id, tr.kind, tr.team, tr.state, float(p[0]), float(p[1]),
                                xy[0] if xy else None, xy[1] if xy else None, tr.hp, t - tr.t,
                                tr.relation, tr.relation_reason, tr.side, tr.side_reason,
                                tr.side_src, tr.hp_rate))
        return out
