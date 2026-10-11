"""Маска детектора юнитов у вида игрока (#786)."""

from __future__ import annotations

from lolvision.unit_targets import ignore_map, unit_keep

W, H = 480, 270      # маска в долях кадра — проверяем на малом


def test_player_view_sees_more_than_spectator():
    # Полосы у HUD в ignore_map — в пикселях 1080p, поэтому кадр полный.
    spec = ignore_map(unit_keep(1920, 1080)).mean()
    player = ignore_map(unit_keep(1920, 1080, "player")).mean()
    assert spec > 0.6 and player < 0.45


def test_player_view_opens_sides_and_bottom_left():
    keep = unit_keep(W, H, "player")
    assert keep[int(0.3 * H), int(0.05 * W)] == 1        # левый край: у наблюдателя панель
    assert keep[int(0.3 * H), int(0.95 * W)] == 1        # правый край
    assert keep[int(0.9 * H), int(0.1 * W)] == 1         # низ слева: у игрока чат над игрой
    assert keep[int(0.95 * H), int(0.45 * W)] == 0       # HUD игрока


def test_player_view_masks_found_minimap():
    big = (0.785, 0.61, 0.99, 0.977)                     # миникарта ×1.5
    keep = unit_keep(W, H, "player", big)
    assert keep[int(0.6 * H), int(0.9 * W)] == 0
    assert unit_keep(W, H, "player")[int(0.6 * H), int(0.9 * W)] == 1
