"""Ситуации на линии для подсказок коуча (#816): окно размена и кнопка в крипов.

Зачем. Коучу мало «CS 87»: полезнее сказать в моменте «он сейчас пойдёт
добивать — меняйся, кнопки готовы» или «Q в крипов — теперь стой и жди».
Обе ситуации собираются из того, что уже есть на кадре вида игрока:
треки юнитов (unit_tracker: место на карте, ХП и скорость, с которой юнита
бьют, `hp_rate`; отношение свой / враг по цвету полоски), своё место
(last_hits.SelfPlace) и готовность умений по HUD (hud_spells, Q/W/E/R).

* `ability_cast` — умение было «готово» и ушло на перезарядку (между
  прочтениями не дольше GAP с). Цель — по тому, кто рядом со своим в этот
  кадр: вражеский чемпион не дальше CAST_CHAMP — `champion`; чемпионов не
  дальше CAST_CLEAR, а вражеских миньонов не дальше CAST_WAVE хотя бы
  WAVE_MIN — `wave` («кнопка в крипов»); своё место неизвестно или
  непонятно — `unknown`. Что умение попало в чемпиона, не видно: `champion`
  — только «чемпион был рядом».
* `trade_window` — союзный миньон (свой для игрока: его будет добивать
  враг) рядом с вражеским чемпионом (не дальше LH_ENEMY) станет «на
  добивание» (ХП ≤ LH_HP) не позже чем через LEAD с по скорости, с которой
  его бьют, а свой не дальше TRADE_ME от этого чемпиона. Пока враг добивает,
  он стоит у крипа и тратит удар — время для размена. Событие — на
  открытие окна (один раз на крипа), факт `window` — пока оно открыто.

Пороги — ед. карты и доли ХП; «на добивание» у миньона по удару чемпиона
ранней игры — около четверти ХП. Это эвристика: замеров «размен удался»
нет, проверка — глазами (docs/lane-watch.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lolvision.geometry import MAP_UNITS
from lolvision.hud_spells import ABILITY_SLOTS
from lolvision.last_hits import on_lane

U = MAP_UNITS
GAP = 1.5                 # с: «готово» → «перезарядка» не дальше этого — нажатие
CAST_CHAMP = 1100.0 / U   # вражеский чемпион ближе — нажатие «по чемпиону»
CAST_CLEAR = 1500.0 / U   # чемпионов дальше этого — нажатие «не по чемпиону»
CAST_WAVE = 900.0 / U     # вражеские миньоны ближе — волна рядом
WAVE_MIN = 2
LH_HP = 0.28              # доля ХП: миньон «на добивание»
LEAD = 2.5                # с: окно размена открывается за столько до добивания
LH_ENEMY = 700.0 / U      # вражеский чемпион не дальше от миньона
TRADE_ME = 1000.0 / U     # свой не дальше от вражеского чемпиона
RECENT = 10


def _d(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _seen(tracks, kind: str, relation: str) -> list[dict]:
    return [tr for tr in tracks or [] if tr["kind"] == kind and tr.get("relation") == relation
            and tr.get("x") is not None]


@dataclass
class LaneWatch:
    """Ситуации линии по кадрам. `update` — события кадра, `window` — окно
    размена, открытое сейчас (или None), `recent` — последние события."""
    last: dict = field(default_factory=dict)       # слот → (t, готово?)
    window: dict | None = None
    recent: list = field(default_factory=list)
    _opened: set = field(default_factory=set)

    def update(self, t: float, abilities: list[dict] | None, tracks: list[dict] | None,
               me: tuple[float, float] | None) -> list[dict]:
        out = self._casts(t, abilities, tracks, me)
        w = self._window(t, abilities, tracks, me)
        if w is not None and w["minion"] not in self._opened:
            self._opened.add(w["minion"])
            out.append(dict(w, type="trade_window"))
        self.window = w
        if len(self._opened) > 500:
            self._opened.clear()
        self.recent = (self.recent + out)[-RECENT:]
        return out

    def _casts(self, t, abilities, tracks, me) -> list[dict]:
        out = []
        for r in abilities or []:
            k, v = r["slot"], r["value"]
            if v is None:
                continue                    # «не знаю» не сбивает прошлое «готово»
            was = self.last.get(k)
            self.last[k] = (t, v)
            if v is False and was is not None and was[1] is True and t - was[0] <= GAP:
                out.append({"t": round(t, 2), "type": "ability_cast", "slot": k,
                            **self._target(tracks, me)})
        return out

    def _target(self, tracks, me) -> dict:
        if me is None:
            return {"target": "unknown", "enemy_dist": None, "wave": None,
                    "why": "своё место неизвестно"}
        champs = [_d(me, (tr["u"], tr["v"])) for tr in _seen(tracks, "champion", "enemy")]
        wave = sum(_d(me, (tr["u"], tr["v"])) <= CAST_WAVE
                   for tr in _seen(tracks, "minion", "enemy"))
        near = min(champs) if champs else None
        dist = None if near is None else round(near * U)
        if near is not None and near <= CAST_CHAMP:
            return {"target": "champion", "enemy_dist": dist, "wave": wave, "why": None}
        if (near is None or near > CAST_CLEAR) and wave >= WAVE_MIN:
            return {"target": "wave", "enemy_dist": dist, "wave": wave, "why": None}
        return {"target": "unknown", "enemy_dist": dist, "wave": wave,
                "why": "рядом нет ни чемпиона, ни волны" if near is None or near > CAST_CLEAR
                else "вражеский чемпион на средней дистанции"}

    def _window(self, t, abilities, tracks, me) -> dict | None:
        if me is None:
            return None
        enemies = _seen(tracks, "champion", "enemy")
        if not enemies:
            return None
        best = None
        for m in _seen(tracks, "minion", "ally"):
            # Монстр леса, принятый за миньона (#816): родился или стоит вне линии.
            if str(m["id"]).startswith("jungle-") or not on_lane(m["u"], m["v"]):
                continue
            hp, rate = float(m["hp"]), float(m.get("hp_rate") or 0.0)
            if hp <= LH_HP:
                eta = 0.0
            elif rate < 0:
                eta = (hp - LH_HP) / -rate
            else:
                continue
            if eta > LEAD:
                continue
            mp = (m["u"], m["v"])
            for e in enemies:
                ep = (e["u"], e["v"])
                if _d(mp, ep) <= LH_ENEMY and _d(me, ep) <= TRADE_ME:
                    cand = {"t": round(t, 2), "minion": m["id"], "enemy": e["id"],
                            "eta": round(eta, 1), "enemy_dist": round(_d(me, ep) * U),
                            "ready": [r["slot"] for r in abilities or []
                                      if r["slot"] in ABILITY_SLOTS and r["value"] is True]}
                    if best is None or cand["eta"] < best["eta"]:
                        best = cand
        return best
