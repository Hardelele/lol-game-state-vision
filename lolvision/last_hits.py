"""Добивания крипов по нескольким кадрам (#816).

Зачем. «Добил ли игрок крипа» на одном кадре не видно: крип просто
исчезает. Видно на двух: счётчик CS на HUD вырос (hud_stats.py) — это
метка «добил», без ручной разметки; а смерть крипа на основном виде — из
трекера (unit_tracker.py, событие unit_death). Сопоставление даёт:

* `last_hit` — вражеский миньон линии умер на том же кадре, где вырос CS
  (± TOL): своего места для этого не нужно;
* `missed_last_hit` — вражеский миньон линии умер рядом со своим
  чемпионом, а CS не вырос: добил кто-то другой (союзные миньоны, башня,
  союзник) — упущенное добивание. Нужно своё место; его нет — смерть не
  оценивается (`unjudged`);
* `cs_unseen` — CS вырос, а смерти в кадре нет: монстры леса (детектор их
  не находит), крип за краем кадра, смерть не распознана.

Время: трекер объявляет смерть на первом кадре без крипа, и HUD тот же
кадр уже показывает новый CS — по двум роликам (D37cbVNs0EU, InEwM0UBAbc,
4 кадра/с) сдвиг «смерть − рост» собран в 0 и +0,25 с при фоне ~5 пар на
шаг 0,25 с; окно ±1,5 с собирало чужие смерти (docs/last-hits.md).

Миньоны леса — монстры, которых детектор принимает за вражеских миньонов
(id трекера `jungle-minion-…`: родился вдали от линий), — не считаются.

Счётчик CS (`CsFilter`): прочтение принимается, только когда следующее
прочтение его подтверждает (не меньше и в пределах шага) — одиночная
ошибка чтения не рождает добиваний. Меньше принятого 5 прочтений подряд —
новая игра, счётчик заново.

«Рядом» — NEAR единиц карты между последним местом крипа и своим
чемпионом: дальность атаки стрелка 550–650 ед. плюс ошибка проекции
(~60–100 ед.) и шаг крипа за кадр. Свой чемпион — единственный чемпион с
зелёной или жёлтой (режим для дальтоников) плашкой на кадре.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lolvision.geometry import MAP_UNITS

NEAR = 800.0 / MAP_UNITS        # доли карты
TOL = 0.05                      # с: смерть и рост CS — один и тот же кадр
WAIT = 1.0                      # с: рост выдаётся через прочтение после появления
MAX_GAIN = 12                   # роста CS за одно прочтение больше — ошибка чтения
NEW_GAME = 5                    # прочтений подряд меньше принятого — новая игра
SELF_COLORS = ("green", "yellow")
# Новый трек миньона так близко к «умершему» и так скоро — тот же крип:
# трекер потерял его за чужой плашкой или эффектом и объявил смерть
# (D37cbVNs0EU 108,5 с: крип на следующих кадрах жив).
REVIVE_R, REVIVE_T = 200.0 / MAP_UNITS, 1.0


@dataclass
class CsGain:
    t: float                    # время прочтения, где рост принят
    t_prev: float               # прошлое принятое прочтение
    n: int
    cs: int
    left: int = 0               # не сопоставлено со смертями


class CsFilter:
    """Монотонный счётчик CS по прочтениям HUD (None — «не прочитано»)."""

    def __init__(self):
        self.cs: int | None = None
        self.t: float | None = None
        self._cand: tuple[float, int] | None = None
        self._low = 0

    def update(self, t: float, cs: int | None) -> list[CsGain]:
        if cs is None:
            return []
        if self.cs is None:
            # Первое значение — тоже после подтверждения.
            if self._cand is not None and 0 <= cs - self._cand[1] <= MAX_GAIN:
                self.cs, self.t = self._cand[1], self._cand[0]
                self._cand = None
                return self.update(t, cs)
            self._cand = (t, cs)
            return []
        if cs < self.cs:
            self._low += 1
            if self._low >= NEW_GAME:
                self.cs, self.t, self._cand, self._low = None, None, None, 0
            return []
        self._low = 0
        out = []
        if self._cand is not None:
            ct, cv = self._cand
            # Кандидат подтверждён: следующее прочтение не меньше и рядом.
            if cv <= cs <= cv + MAX_GAIN and cv > self.cs:
                out.append(CsGain(ct, self.t, cv - self.cs, cv, cv - self.cs))
                self.cs, self.t = cv, ct
            self._cand = None
        if cs > self.cs:
            if cs - self.cs <= MAX_GAIN:
                self._cand = (t, cs)
        else:
            self.t = t
        return out


@dataclass
class Death:
    t: float
    id: str
    uv: tuple[float, float]
    me: tuple[float, float] | None   # своё место на карте в этот кадр
    revived: bool = False            # рядом тут же родился трек: смерть ложная

    @property
    def dist(self) -> float | None:
        """Ед. карты до своего чемпиона; None — своё место неизвестно."""
        if self.me is None:
            return None
        return float(np.hypot(self.uv[0] - self.me[0], self.uv[1] - self.me[1])) * MAP_UNITS


@dataclass
class LastHitEvent:
    t: float
    type: str                   # last_hit / missed_last_hit / cs_unseen
    id: str | None = None       # крип (у cs_unseen — None)
    n: int = 1
    dist: float | None = None   # ед. карты до своего чемпиона

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "type": self.type, "id": self.id, "n": self.n,
                "dist": None if self.dist is None else round(self.dist)}


def lane_minion(d: dict) -> bool:
    """Вражеский миньон линии: монстры леса (id `jungle-…`) не в счёт."""
    return (d.get("kind") == "minion" and d.get("relation") == "enemy"
            and not str(d.get("id", "")).startswith("jungle-"))


@dataclass
class LastHits:
    """Сопоставление роста CS со смертями вражеских миньонов линии.

    События выдаются с задержкой WAIT: рост CS принимается только на
    следующем прочтении. `unjudged` — смертей без своего места и без
    роста CS: ни «добил», ни «упустил» о них сказать нельзя."""
    tol: float = TOL
    cs: CsFilter = field(default_factory=CsFilter)
    deaths: list[Death] = field(default_factory=list)
    gains: list[CsGain] = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"last_hit": 0, "missed_last_hit": 0,
                                                   "cs_unseen": 0, "unjudged": 0,
                                                   "revived": 0})

    def update(self, t: float, cs: int | None, deaths: list[dict],
               me_uv: tuple[float, float] | None,
               born: list[tuple[float, float]] = ()) -> list[LastHitEvent]:
        """`deaths` — unit_death трекера этого кадра (словари to_dict),
        `me_uv` — своё место на карте (доли) или None (не видно), `born` —
        места на карте новых треков вражеских миньонов этого кадра."""
        self.gains += self.cs.update(t, cs)
        for d in self.deaths:
            if t - d.t <= REVIVE_T and any(
                    np.hypot(d.uv[0] - u, d.uv[1] - v) <= REVIVE_R for u, v in born):
                d.revived = True
        for d in deaths:
            if lane_minion(d):
                self.deaths.append(Death(t, d["id"], (float(d["u"]), float(d["v"])), me_uv))
        return self._resolve(t)

    def _resolve(self, now: float, final: bool = False) -> list[LastHitEvent]:
        out = []
        # Пары «смерть — рост» по возрастанию разницы времени, при равной —
        # ближняя к себе: роста CS на все смерти рядом может не хватить, и
        # он скорее у той, что на кадре роста (D37cbVNs0EU 106,75 / 107,0 с:
        # «+20» золота — над второй). Пары — по всем ждущим смертям, не
        # только созревшим: иначе рост достаётся той, что созрела раньше.
        live = [d for d in self.deaths if not d.revived]
        pairs = sorted(((abs(g.t - d.t), d.dist if d.dist is not None else 1e9, i, j)
                        for i, d in enumerate(live) for j, g in enumerate(self.gains)
                        if abs(g.t - d.t) <= self.tol), key=lambda p: p[:2])
        hit: dict[int, int] = {}
        left = [g.left for g in self.gains]
        for _, _, i, j in pairs:
            if i not in hit and left[j] > 0:
                left[j] -= 1
                hit[i] = j
        wait = max(self.tol + WAIT, REVIVE_T)
        for i, d in enumerate(live):
            if not final and now < d.t + wait:
                continue
            self.deaths.remove(d)
            if i in hit:
                self.gains[hit[i]].left -= 1
                out.append(LastHitEvent(d.t, "last_hit", d.id, 1, d.dist))
            elif d.dist is None:
                self.counts["unjudged"] += 1
            elif d.dist <= NEAR * MAP_UNITS:
                out.append(LastHitEvent(d.t, "missed_last_hit", d.id, 1, d.dist))
        for d in [d for d in self.deaths if d.revived and (final or now >= d.t + REVIVE_T)]:
            self.deaths.remove(d)
            self.counts["revived"] += 1
        # Рост, к которому смертей уже не придёт, — остаток в cs_unseen.
        for g in [g for g in self.gains if final or now >= g.t + self.tol + 2 * WAIT]:
            self.gains.remove(g)
            if g.left > 0:
                out.append(LastHitEvent(g.t, "cs_unseen", None, g.left))
        for e in out:
            self.counts[e.type] += e.n
        out.sort(key=lambda e: e.t)
        return out

    def flush(self) -> list[LastHitEvent]:
        return self._resolve(float("inf"), final=True)


def self_unit(units: list[dict]) -> dict | None:
    """Свой чемпион кадра: единственный чемпион с зелёной или жёлтой плашкой."""
    mine = [u for u in units or [] if u.get("kind") == "champion"
            and u.get("color") in SELF_COLORS]
    return mine[0] if len(mine) == 1 else None
