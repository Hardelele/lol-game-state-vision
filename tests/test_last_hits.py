"""Добивания по нескольким кадрам (#816): полоса CS/времени HUD и сопоставление
роста CS со смертями крипов.

Без роликов: полоса рисуется на тёмном фоне из значка миньона
(dataset/hud_stats_icon.png) и образцов цифр (dataset/hud_stats_digits.npz),
смерти и прочтения CS — словари и числа, как их дают трекер и читатель.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lolvision import hud_stats as hs
from lolvision import last_hits as lh
from lolvision.geometry import MAP_UNITS

Z = np.load(hs.DIGITS)


def glyph(c: str) -> np.ndarray:
    """Образец цифры в размере строки (12 px), яркость 0..230."""
    g = Z["glyphs"][list(Z["labels"]).index(c)]
    return (g * 230).astype(np.uint8)


def strip_with(cs: str, time: str, icon: bool = True) -> np.ndarray:
    st = np.full((36, 420, 3), 35, np.uint8)
    if icon:
        ic = cv2.imread(str(hs.ICON), cv2.IMREAD_GRAYSCALE)
        x0 = hs.ICON_X[0] - hs.X0
        st[hs.ICON_Y[0]:hs.ICON_Y[1], x0:x0 + ic.shape[1]] = ic[..., None]

    def put(text: str, x: int) -> None:
        for c in text:
            if c == ":":
                st[12:14, x:x + 2] = 230
                st[18:20, x:x + 2] = 230
                x += 5
                continue
            g = glyph(c)
            st[9:21, x:x + g.shape[1]] = g[..., None]
            x += g.shape[1] + 2

    put(cs, hs.CS_X[0] - hs.X0 + 2)
    put(time, hs.TIME_X[0] - hs.X0 + 2)
    return st


def test_reads_cs_and_time():
    r = hs.StatsReader()
    s = r.read_strip(strip_with("147", "12:05"))
    assert s.present and s.cs == 147 and s.time == 12 * 60 + 5


def test_no_icon_means_no_strip():
    s = hs.StatsReader().read_strip(strip_with("147", "12:05", icon=False))
    assert not s.present and s.cs is None and s.time is None


def test_touching_digits_are_split():
    # Две цифры без зазора (сжатие ролика размывает его) — режутся надвое.
    st = np.full((20, 40, 3), 35, np.uint8)
    a, b = glyph("4"), glyph("7")
    st[4:16, 5:5 + a.shape[1]] = a[..., None]
    st[4:16, 5 + a.shape[1]:5 + a.shape[1] + b.shape[1]] = b[..., None]
    assert len(hs.glyphs(st)) == 2


def test_cs_filter_needs_confirmation_and_skips_misread():
    f = lh.CsFilter()
    seq = [(0, 10), (1, 10), (2, 11), (3, 11), (4, 1521), (5, 12), (6, 12)]
    gains = [g for t, cs in seq for g in f.update(t, cs)]
    assert [(g.t, g.n, g.cs) for g in gains] == [(2, 1, 11), (5, 1, 12)]


def test_cs_filter_resets_on_new_game():
    f = lh.CsFilter()
    for t, cs in [(0, 150), (1, 150)]:
        f.update(t, cs)
    for t in range(2, 2 + lh.NEW_GAME):
        f.update(t, 0)
    assert f.cs is None
    gains = [g for t, cs in [(10, 0), (11, 1), (12, 1)] for g in f.update(t, cs)]
    assert [(g.n, g.cs) for g in gains] == [(1, 1)]


def death(id_: str, u: float, v: float, relation: str = "enemy", kind: str = "minion") -> dict:
    return {"type": "unit_death", "id": id_, "kind": kind, "team": "CHAOS",
            "u": u, "v": v, "relation": relation}


def run(frames) -> list[lh.LastHitEvent]:
    m = lh.LastHits()
    out = []
    for t, cs, deaths, me in frames:
        out += m.update(t, cs, deaths, me)
    return out + m.flush()


ME = (0.20, 0.20)
NEAR_UV = (0.20 + 300 / MAP_UNITS, 0.20)
FAR_UV = (0.20 + 2000 / MAP_UNITS, 0.20)


def test_last_hit_missed_and_unseen():
    frames = [(t / 4, 10, [], ME) for t in range(4)]
    # 1.0 с: крип умер рядом, CS +1 на том же кадре — добил.
    frames += [(1.0, 11, [death("m1", *NEAR_UV)], ME), (1.25, 11, [], ME), (1.5, 11, [], ME)]
    # 5 с: крип умер рядом, CS не растёт — упущено.
    frames += [(t / 4, 11, [], ME) for t in range(7, 20)]
    frames += [(5.0, 11, [death("m2", *NEAR_UV)], ME)]
    frames += [(t / 4, 11, [], ME) for t in range(21, 40)]
    # 12 с: CS +2 без смертей в кадре — лес или за кадром.
    frames += [(12.0, 13, [], ME), (12.25, 13, [], ME)]
    # Далёкая смерть и смерть союзного миньона не считаются.
    frames += [(14.0, 13, [death("m3", *FAR_UV), death("a1", *NEAR_UV, "ally")], ME)]
    ev = run(frames)
    assert [(e.type, e.id, e.n) for e in ev] == [
        ("last_hit", "m1", 1), ("missed_last_hit", "m2", 1), ("cs_unseen", None, 2)]
    assert ev[0].dist == pytest.approx(300, abs=1)


def test_last_hit_needs_no_self_place_but_missed_does():
    m = lh.LastHits()
    frames = [(0, 5, [], None), (0.25, 5, [], None),
              # добил: смерть и рост CS на одном кадре — своё место не нужно
              (1.0, 6, [death("m1", *NEAR_UV)], None), (1.25, 6, [], None),
              # смерть без роста и без своего места — не оценивается
              (3.0, 6, [death("m2", *NEAR_UV)], None), (3.25, 6, [], None)]
    ev = [e for t, cs, d, me in frames for e in m.update(t, cs, d, me)] + m.flush()
    assert [(e.type, e.id, e.dist) for e in ev] == [("last_hit", "m1", None)]
    assert m.counts["unjudged"] == 1


def test_jungle_monsters_and_distant_gains_do_not_match():
    # «Миньон» леса (монстр под видом миньона) и смерть за 1 с до роста CS.
    frames = [(0, 5, [], ME), (0.25, 5, [], ME),
              (1.0, 5, [death("jungle-minion-ab", *NEAR_UV)], ME),
              (2.0, 5, [death("m1", *NEAR_UV)], ME), (3.0, 6, [], ME), (3.25, 6, [], ME)]
    ev = run(frames)
    assert [(e.type, e.id) for e in ev] == [("missed_last_hit", "m1"), ("cs_unseen", None)]


def test_self_unit_is_single_green_or_yellow_champion():
    a = {"kind": "champion", "color": "yellow", "x": 0.5, "y": 0.5}
    b = {"kind": "champion", "color": "red", "x": 0.6, "y": 0.5}
    assert lh.self_unit([a, b]) is a
    assert lh.self_unit([a, dict(a, color="green")]) is None
    assert lh.self_unit([dict(a, kind="minion")]) is None


def test_gain_on_the_next_frame_is_not_a_last_hit():
    # Рост CS кадром позже смерти — фон (замер в docs/last-hits.md): не пара.
    frames = [(0, 5, [], ME), (0.25, 5, [], ME), (1.0, 5, [death("m1", *NEAR_UV)], ME),
              (1.25, 6, [], ME), (1.5, 6, [], ME)]
    ev = run(frames)
    assert [(e.type, e.id) for e in ev] == [("missed_last_hit", "m1"), ("cs_unseen", None)]


def test_gain_goes_to_death_on_its_frame_not_the_earlier_one():
    # Две смерти через кадр, один рост CS на кадре второй: добил вторую.
    m = lh.LastHits()
    frames = [(0, 5, [], ME), (0.25, 5, [], ME), (1.0, 5, [death("m1", *NEAR_UV)], ME),
              (1.25, 6, [death("m2", *NEAR_UV)], ME), (1.5, 6, [], ME)]
    frames += [(1.5 + k / 4, 6, [], ME) for k in range(1, 12)]
    ev = [e for t, cs, d, me in frames for e in m.update(t, cs, d, me)] + m.flush()
    assert [(e.type, e.id) for e in ev] == [("missed_last_hit", "m1"), ("last_hit", "m2")]


def test_death_with_new_track_on_the_spot_is_revived():
    m = lh.LastHits()
    m.update(0, 5, [], ME)
    m.update(0.25, 5, [death("m1", *NEAR_UV)], ME)
    m.update(0.5, 5, [], ME, born=[(NEAR_UV[0] + 50 / MAP_UNITS, NEAR_UV[1])])
    assert m.flush() == [] and m.counts["revived"] == 1


# ---------- часы матча по полосе (#780) ----------

def test_hud_clock_tracks_game_time_between_whole_seconds():
    c = hs.HudClock()
    # Время матча = кадр + 100,4 с; полоса показывает целые секунды вниз.
    for k in range(12):
        t = 0.25 * k
        c.update(t, int(t + 100.4))
    assert abs(c.value(3.0) - 103.4) < 0.3


def test_hud_clock_ignores_single_misread_and_follows_a_pause():
    c = hs.HudClock()
    for t in range(5):
        c.update(t, 600 + t)
    c.update(5, 644)                 # 10:44 вместо 10:05 — одно прочтение
    assert abs(c.value(5) - 605.5) < 0.6
    for t in range(6, 9):            # игра продолжилась как шла
        c.update(t, 600 + t)
    assert abs(c.value(8) - 608.5) < 0.6
    # Пауза игры 20 с: время на полосе стоит, кадр идёт — новый отрезок
    # принимается со второго согласного прочтения.
    c.update(30, 609)
    assert c.value(30) is None       # старый отрезок устарел, новый не подтверждён
    c.update(31, 610)
    assert abs(c.value(31) - 610.5) < 0.6


def test_hud_clock_goes_stale_without_readings():
    c = hs.HudClock()
    c.update(0, 300)
    c.update(1, 301)
    c.update(2, None)
    assert c.value(5) is not None
    assert c.value(1 + hs.CLOCK_STALE + 1) is None


def test_self_place_and_frame_inputs():
    units = [{"kind": "champion", "color": "green", "x": 0.5, "y": 0.5},
             {"kind": "champion", "color": "red", "x": 0.2, "y": 0.3}]
    tracks = [{"id": "me", "kind": "champion", "x": 0.5, "y": 0.5, "u": 0.3, "v": 0.7},
              {"id": "m1", "kind": "minion", "x": 0.6, "y": 0.5, "u": 0.31, "v": 0.7}]
    assert lh.self_place(units, tracks) == (0.3, 0.7)
    assert lh.self_place(units[1:], tracks) is None
    uev = [{"type": "unit_death", "id": "m0", "kind": "minion", "relation": "enemy"},
           {"type": "unit_new", "id": "m1", "kind": "minion", "relation": "enemy"}]
    deaths, born = lh.frame_inputs(uev, tracks)
    assert [d["id"] for d in deaths] == ["m0"] and born == [(0.31, 0.7)]


# ---------- монстры и своё место (#816, вторая часть) ----------

def test_raptor_death_off_mid_corridor_is_a_monster_not_a_missed_last_hit():
    # D37cbVNs0EU 152,5 с: «пташка» в 0,084 от диагонали средней линии,
    # свой рядом, CS не вырос — раньше это было missed_last_hit.
    m = lh.LastHits()
    me = (0.50, 0.37)
    raptor = {"type": "unit_death", "id": "mid-minion-r", "kind": "minion", "relation": "enemy",
              "u": 0.503, "v": 0.379}
    lane = {"type": "unit_death", "id": "mid-minion-l", "kind": "minion", "relation": "enemy",
            "u": 0.518, "v": 0.536}           # 0,038 от диагонали — настоящая средняя
    evs = []
    for k in range(20):
        t = 0.25 * k
        evs += m.update(t, 40, [raptor, lane] if k == 2 else [], (0.52, 0.52) if k == 2 else me)
    evs += m.flush()
    assert m.counts["monster"] == 1
    assert [e.id for e in evs if e.type == "missed_last_hit"] == ["mid-minion-l"]


def test_on_lane_corridors():
    assert lh.on_lane(0.50, 0.50)                 # на диагонали средней
    assert not lh.on_lane(0.475, 0.393)           # лагерь пташек
    assert lh.on_lane(0.12, 0.33)                 # верхняя линия, смерть в 0,04
    assert not lh.on_lane(0.30, 0.30)             # лес между верхней и средней


def test_self_place_remembers_own_track_while_bar_color_is_unreadable():
    sp = lh.SelfPlace(keep=20.0)
    me_unit = {"kind": "champion", "color": "green", "x": 0.5, "y": 0.5}
    tr = {"id": "c1", "kind": "champion", "x": 0.5, "y": 0.5, "u": 0.3, "v": 0.7}
    assert sp.update(0.0, [me_unit], [tr]) == (0.3, 0.7) and sp.src == "color"
    # Плашка не читается (полоса храбрости Kled): цвет «неоднозначен».
    blurred = dict(me_unit, color=None)
    moved = dict(tr, x=0.55, u=0.32)
    assert sp.update(5.0, [blurred], [moved]) == (0.32, 0.7) and sp.src == "track"
    assert sp.update(6.0, [blurred], [dict(moved, x=None)]) is None   # трек не виден
    assert sp.update(30.0, [blurred], [moved]) is None                # память истекла
