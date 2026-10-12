"""Свои / враги / сам и сторона (lolvision/sides.py, #790) на синтетике.

Кадр вида игрока рисуется руками: плашка чемпиона — заливка ХП высотой
10 строк в тёмной рамке над точкой тела, под ней полоска ресурса; полоска
миньона — 53×4 в тёмной рамке на светлом фоне, на месте по unit_bars.BODY.
"""

import cv2
import numpy as np
import pytest

from lolvision.geometry import Projection
from lolvision.records import UNSET, TrackRec, UnitRec
from lolvision.sides import (WHY_DISAGREE, WHY_NO_API, WHY_NO_BAR, WHY_NO_FRAME, WHY_SPECTATOR,
                             WHY_VIEW, SideContext, assign, probe_color)
from lolvision.unit_bars import BODY
from lolvision.unit_tracker import UnitTracker

W, H = 1920, 1080
RGB = {"green": (40, 200, 40), "blue": (40, 120, 230), "red": (210, 40, 40),
       "yellow": (230, 200, 30), "mana": (60, 110, 220)}


def scene() -> np.ndarray:
    """Светлый неяркий фон: трава без насыщенных пикселей."""
    return np.full((H, W, 3), (110, 120, 105), np.uint8)


def plaque(rgb, x, y, color, fill=100, mana=True):
    """Плашка чемпиона вида игрока над точкой тела (x, y)."""
    x0, top = x - 60, y - 115
    cv2.rectangle(rgb, (x0 - 4, top - 4), (x0 + 124, top + 19), (15, 15, 20), -1)
    cv2.rectangle(rgb, (x0, top), (x0 + fill - 1, top + 9), RGB[color], -1)
    if mana:                                   # ресурс: 4 строки, синий
        cv2.rectangle(rgb, (x0, top + 12), (x0 + 110, top + 15), RGB["mana"], -1)


def bar(rgb, kind, x, y, color, fill=53):
    """Полоска миньона/варда: место — от точки тела по unit_bars.BODY."""
    dx, dy, _, bh = BODY[kind]
    bx, by = round(x - dx - 26.5), round(y - dy - bh / 2)
    cv2.rectangle(rgb, (bx - 1, by - 2), (bx + 53, by + 5), (10, 10, 10), -1)
    cv2.rectangle(rgb, (bx, by), (bx + fill - 1, by + 3), RGB[color], -1)


def rec(kind, team, x, y):
    return UnitRec(kind, team, x / W, y / H, 1.0, 0.9)


def frame_player():
    """Свой (зелёный), союзник (синий), враг (красный) и по миньону."""
    rgb = scene()
    plaque(rgb, 900, 500, "green")
    plaque(rgb, 500, 500, "blue")
    plaque(rgb, 1400, 500, "red")
    bar(rgb, "minion", 600, 800, "blue")
    bar(rgb, "minion", 1300, 800, "red")
    # Ответ детектора: своя зелёная плашка получает «какую-то» команду —
    # так на кадрах стримов (docs/sides.md), здесь нарочно CHAOS.
    units = [rec("champion", "CHAOS", 900, 500), rec("champion", "ORDER", 500, 500),
             rec("champion", "CHAOS", 1400, 500), rec("minion", "ORDER", 600, 800),
             rec("minion", "CHAOS", 1300, 800)]
    return rgb, units


def table(recs):
    return [(r.relation, r.side) for r in recs]


def test_spectator_side_is_detector_team_and_no_relation():
    out = assign([rec("minion", "ORDER", 100, 100), rec("champion", "CHAOS", 300, 300)],
                 SideContext("spectator"))
    assert table(out) == [(None, "ORDER"), (None, "CHAOS")]
    assert {r.relation_reason for r in out} == {WHY_SPECTATOR}
    assert all(r.side_reason is None for r in out)


def test_player_view_playing_order():
    rgb, units = frame_player()
    out = assign(units, SideContext("player", "ORDER"), rgb)
    assert [r.color for r in out] == ["green", "blue", "red", "blue", "red"]
    assert table(out) == [("self", "ORDER"), ("ally", "ORDER"), ("enemy", "CHAOS"),
                          ("ally", "ORDER"), ("enemy", "CHAOS")]


def test_player_view_playing_chaos_side_is_not_flipped():
    # Тот же экран за красных: союзники так же синие, но их сторона — CHAOS.
    rgb, units = frame_player()
    out = assign(units, SideContext("player", "CHAOS"), rgb)
    assert table(out) == [("self", "CHAOS"), ("ally", "CHAOS"), ("enemy", "ORDER"),
                          ("ally", "CHAOS"), ("enemy", "ORDER")]


def test_unexpected_color_is_refused():
    rgb = scene()
    plaque(rgb, 900, 500, "yellow")            # режим для дальтоников
    out = assign([rec("champion", "CHAOS", 900, 500)], SideContext("player", "ORDER"), rgb)[0]
    assert (out.relation, out.side, out.color) == (None, None, "yellow")
    assert out.relation_reason.startswith("цвет вне ожидаемого") and "yellow" in out.relation_reason


def test_green_minion_is_not_self():
    rgb = scene()
    bar(rgb, "minion", 900, 500, "green")
    out = assign([rec("minion", "ORDER", 900, 500)], SideContext("player", "ORDER"), rgb)[0]
    assert out.relation is None and out.relation_reason.startswith("цвет вне ожидаемого")


def test_bar_color_disagreeing_with_detector_is_refused():
    rgb = scene()
    bar(rgb, "minion", 900, 500, "blue")
    out = assign([rec("minion", "CHAOS", 900, 500)], SideContext("player", "ORDER"), rgb)[0]
    assert (out.relation, out.side) == (None, None) and out.relation_reason == WHY_DISAGREE


def test_unknown_view_refuses_both():
    rgb, units = frame_player()
    out = assign(units, SideContext(None, "ORDER"), rgb)
    assert table(out) == [(None, None)] * len(units)
    assert {r.relation_reason for r in out} == {r.side_reason for r in out} == {WHY_VIEW}


def test_no_api_keeps_relation_refuses_side():
    rgb, units = frame_player()
    out = assign(units, SideContext("player", None, "Live Client API не отвечает"), rgb)
    assert [r.relation for r in out] == ["self", "ally", "enemy", "ally", "enemy"]
    assert all(r.side is None for r in out)
    assert out[0].side_reason == f"{WHY_NO_API}: Live Client API не отвечает"


def test_player_view_without_frame_is_refused_not_detector_team():
    _, units = frame_player()
    out = assign(units, SideContext("player", "ORDER"))
    assert table(out) == [(None, None)] * len(units)
    assert {r.relation_reason for r in out} == {WHY_NO_FRAME}


def test_no_bar_is_refused():
    out = assign([rec("minion", "ORDER", 900, 500)], SideContext("player", "ORDER"), scene())[0]
    assert out.relation is None and out.relation_reason == WHY_NO_BAR


def test_low_hp_self_plaque_and_mana_line():
    # Малая заливка и синяя полоса маны под ней: маны не хватает высоты,
    # своя остаётся своей, а не «союзником».
    rgb = scene()
    plaque(rgb, 900, 500, "green", fill=20)
    assert probe_color(rgb, "champion", 900 / W, 500 / H) == ("green", None)


@pytest.mark.parametrize("size", [(1280, 720), (2560, 1440)])
def test_probe_scales_with_frame(size):
    rgb, units = frame_player()
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    out = assign(units, SideContext("player", "CHAOS"), small)
    assert table(out) == [("self", "CHAOS"), ("ally", "CHAOS"), ("enemy", "ORDER"),
                          ("ally", "CHAOS"), ("enemy", "ORDER")]


def test_records_reason_only_when_null():
    d = UnitRec("minion", "ORDER", 0.5, 0.5, 1.0, 0.9).to_dict()
    assert d["relation"] is None and d["relation_reason"] == UNSET
    rgb, units = frame_player()
    d = assign(units, SideContext("player", None, "нет API"), rgb)[0].to_dict()
    assert d["relation"] == "self" and d["relation_reason"] is None
    assert d["side"] is None and d["side_reason"].startswith(WHY_NO_API)
    t = TrackRec("x", "minion", "ORDER", "seen", 0.5, 0.5, None, None, 1.0, 0.0,
                 "ally", None, "CHAOS", None).to_dict()
    assert (t["relation"], t["side"], t["relation_reason"], t["side_reason"]) == \
        ("ally", "CHAOS", None, None)


# ---------- трекер: команда по сторонам, а не по team детектора ----------

PH = np.array([[0.001, 0, -0.05], [0, 0.001, -0.05], [0, 0, 1.0]])


def tracker():
    return UnitTracker(Projection(PH, (100, 100), (0, 0, 1000, 1000), (1000, 1000)))


def unit(kind, team, x, relation, side):
    return UnitRec(kind, team, x / 1000, 0.5, 1.0, 0.9, None, None, relation, None, side, None)


def test_self_keeps_track_while_detector_team_flips():
    tr = tracker()
    for k, team in enumerate(["ORDER", "CHAOS", "ORDER", "CHAOS"]):
        tr.update(float(k), (0.5, 0.5), [unit("champion", team, 500 + k, "self", "CHAOS")])
    assert len(tr.tracks) == 1
    view = tr.frame_view(3.0)
    assert [(v.relation, v.side) for v in view] == [("self", "CHAOS")]


def test_ally_and_enemy_minions_do_not_swap():
    tr = tracker()
    ev = []
    for k in range(3):
        ev += tr.update(float(k), (0.5, 0.5),
                        [unit("minion", "ORDER", 500 + 5 * k, "ally", "CHAOS"),
                         unit("minion", "CHAOS", 520 - 5 * k, "enemy", "ORDER")])
    assert len(tr.tracks) == 2
    assert {(t.relation, t.side) for t in tr.tracks} == {("ally", "CHAOS"), ("enemy", "ORDER")}
    assert {(e.relation, e.side) for e in ev if e.type == "unit_new"} == \
        {("ally", "CHAOS"), ("enemy", "ORDER")}


def test_unknown_relation_joins_track_and_known_value_sticks():
    tr = tracker()
    tr.update(0.0, (0.5, 0.5), [unit("minion", "ORDER", 500, None, None)])
    tr.update(1.0, (0.5, 0.5), [unit("minion", "ORDER", 505, "ally", "ORDER")])
    tr.update(2.0, (0.5, 0.5), [unit("minion", "ORDER", 510, None, None)])
    assert len(tr.tracks) == 1 and (tr.tracks[0].relation, tr.tracks[0].side) == ("ally", "ORDER")
