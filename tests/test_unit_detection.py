"""Юниты на основном виде: полоски миньонов и вардов, цели и сверка
детектора, форма выхода сети (без кадров роликов)."""

import numpy as np
import torch

from lolvision import unit_bars as ub
from lolvision import unit_targets as ut
from lolvision.unit_net import CLASSES, IN_H, IN_W, OUT_H, OUT_W, UnitNet, decode, to_units


def scene_with_bar(x, y, hp, fill_rgb=(105, 150, 178), segments=0, bg=95):
    """Кадр 1080p с тонкой полоской: рамка 1 px, заливка 4 строки, пустая
    часть тёмная; segments > 0 — заливка сегментами (вард) и белая линия
    таймера над полоской."""
    rgb = np.full((1080, 1920, 3), bg, np.uint8)
    rgb[y - 1:y + 5, x - 1:x + ub.BAR_W + 1] = (30, 30, 30)       # рамка
    rgb[y:y + 4, x:x + ub.BAR_W] = (25, 35, 40)                   # пустая часть
    if segments:
        seg = ub.BAR_W // segments
        for i in range(segments):
            rgb[y:y + 4, x + i * seg:x + (i + 1) * seg - 2] = fill_rgb
        rgb[y - 3, x:x + ub.BAR_W] = (230, 230, 230)
    else:
        rgb[y:y + 4, x:x + round(hp * ub.BAR_W)] = fill_rgb
    return rgb


def test_minion_bar_found_with_hp_and_team():
    for hp, rgb_fill, team in ((1.0, (105, 150, 178), "ORDER"), (0.4, (200, 50, 50), "CHAOS")):
        bars = ub.find_unit_bars(scene_with_bar(700, 400, hp, rgb_fill))
        assert len(bars) == 1
        b = bars[0]
        assert (b.kind, b.team, b.x, b.y) == ("minion", team, 700, 400)
        assert abs(b.hp - hp) <= 0.04


def test_colour_is_not_team():
    """Жёлтая полоска (режим для дальтоников) находится по форме, но команда
    у наблюдателя для неё не назначается."""
    bars = ub.find_unit_bars(scene_with_bar(700, 400, 0.7, (220, 200, 40)))
    assert len(bars) == 1
    assert bars[0].color == "yellow" and bars[0].team == ""
    assert ub.team_of_color("green", "player") == "SELF"


def test_bright_streak_without_dark_tail_is_rejected():
    rgb = np.full((1080, 1920, 3), 95, np.uint8)
    rgb[400:404, 700:720] = (105, 150, 178)                      # без рамки и пустой части
    assert ub.find_unit_bars(rgb) == []


def test_ward_segments_are_one_ward():
    bars = ub.find_unit_bars(scene_with_bar(700, 400, 1.0, (200, 50, 50), segments=3))
    assert [(b.kind, b.segments) for b in bars] == [("ward", 3)]


def test_excluded_region_is_skipped():
    rgb = scene_with_bar(700, 400, 0.6)
    assert ub.find_unit_bars(rgb, exclude=[(690, 390, 770, 410)]) == []


def test_ignore_map_grows_hud_down_and_sideways():
    keep = np.ones((200, 300), np.uint8)
    keep[:50, 100:200] = 0                                        # HUD сверху посередине
    ign = ut.ignore_map(keep, grow=(20, 20, 5, 60))
    assert ign[40, 150] and ign[100, 150]                         # под HUD — без меток
    assert not ign[150, 150]                                      # ниже полосы
    assert ign[30, 85] and not ign[100, 60]                       # сбоку на 20 px


def test_targets_and_match():
    units = [{"kind": "minion", "team": "CHAOS", "hp": 0.5, "x": 101.0, "y": 63.0},
             {"kind": "champion", "team": "ORDER", "hp": 1.0, "x": 400.0, "y": 300.0}]
    t = ut.build_targets(units, (OUT_H, OUT_W), 4, 1.0)
    c = CLASSES.index("minion")
    assert t["heat"][c, 15, 25] == 1.0 and t["team"][15, 25] == 1.0
    assert abs(t["off"][0, 15, 25] - 0.25) < 1e-6 and t["hp"][15, 25] == 0.5
    preds = [{"kind": "minion", "x": 110, "y": 60, "score": 0.9},
             {"kind": "minion", "x": 500, "y": 60, "score": 0.8}]
    gts = [{"kind": "minion", "x": 100, "y": 60}, {"kind": "champion", "x": 0, "y": 0}]
    pairs, fp, fn = ut.match(preds, gts)
    assert pairs == [(0, 0)] and fp == [1] and fn == [1]


def test_net_output_shape_and_decode():
    net = UnitNet(widths=(8, 16, 24, 32, 40), depths=(1, 1, 1, 1, 1), head=16).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, IN_H, IN_W))
    assert out[0].shape == (1, len(CLASSES), OUT_H, OUT_W)
    heat = torch.full_like(out[0], -10.0)
    heat[0, 1, 20, 30] = 5.0
    rows = decode(heat, out[1], torch.zeros_like(out[2]), out[3], thr=0.5)[0].numpy()
    units = to_units(rows, 2.0)
    assert len(units) == 1 and units[0].kind == "minion"
    assert (units[0].x, units[0].y) == (30 * 4 * 2, 20 * 4 * 2)
