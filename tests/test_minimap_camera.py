"""Поиск рамки камеры: целая рамка и рамка, обрезанная краем миникарты."""

from __future__ import annotations

import numpy as np

from lolvision.minimap_camera import DRAW_X0, DRAW_Y0, MM, find_box, find_box_clipped

BW, BH = 68, 38


def draw(x: int, y: int) -> np.ndarray:
    """Маска с рамкой, обрезанной так же, как её рисует игра."""
    m = np.zeros((MM + 2 * BW, MM + 2 * BW), np.float32)
    o = BW
    m[o + y, o + x:o + x + BW + 1] = 1
    m[o + y + BH, o + x:o + x + BW + 1] = 1
    m[o + y:o + y + BH + 1, o + x] = 1
    m[o + y:o + y + BH + 1, o + x + BW] = 1
    m = m[o:o + MM, o:o + MM]
    m[:DRAW_Y0, :] = 0
    m[:, :DRAW_X0] = 0
    return m


def test_inside_matches_find_box():
    m = draw(100, 120)
    assert find_box(m, BW, BH)[:2] == (100, 120)
    x, y, q, clipped = find_box_clipped(m, BW, BH)
    assert (x, y, clipped) == (100, 120, False)
    assert q > 0.99


def test_clipped_top_keeps_true_position():
    # Видна только нижняя линия и обрывки боковых: прежний поиск ставил сюда
    # верх рамки — сдвиг на её высоту.
    x, y, _, clipped = find_box_clipped(draw(80, -8), BW, BH)
    assert (x, y) == (80, -8) and clipped


def test_clipped_left_and_right():
    assert find_box_clipped(draw(-30, 90), BW, BH)[:2] == (-30, 90)
    assert find_box_clipped(draw(MM - 40, 90), BW, BH)[:2] == (MM - 40, 90)


def test_bright_stub_does_not_beat_full_box():
    m = draw(120, 100)
    m[DRAW_Y0, DRAW_X0:DRAW_X0 + 40] = 1          # светлая линия вдоль края карты
    assert find_box_clipped(m, BW, BH)[:2] == (120, 100)
