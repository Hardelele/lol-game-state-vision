"""Плашки чемпионов: поиск по шаблону, чтение ХП и уровня (без кадров роликов)."""

import numpy as np

from lolvision import health_bars as hb


def scene_with_bar(x: int, y: int, hp: float, team: str = "ORDER", level_box=None):
    """Кадр 1080p с одной плашкой: шаблон рамки в сером + заливка ХП цвета команды."""
    tmpl, _ = hb.load_template()
    rgb = np.full((1080, 1920, 3), 70, np.uint8)
    rgb[y:y + hb.H, x:x + hb.W] = np.clip(tmpl, 0, 255).astype(np.uint8)[..., None]
    fill = (60, 160, 230) if team == "ORDER" else (200, 50, 50)
    w = hb.HP_X1 - hb.HP_X0
    n = round(hp * w)
    band = rgb[y + hb.HP_Y0:y + hb.HP_Y1, x + hb.HP_X0:x + hb.HP_X1]
    band[:] = (15, 15, 15)
    band[:, :n] = fill
    band[:, 10:n:10] = (0, 30, 60)          # засечки внутри заливки
    box = rgb[y + 3:y + 20, x + 5:x + 21]
    box[:] = (20, 35, 50) if team == "ORDER" else (60, 15, 15)
    if level_box is not None:
        box[:] = level_box[..., None]
    return rgb


def test_finds_bar_and_reads_hp_and_team():
    tmpl, mask = hb.load_template()
    for hp, team in ((0.6, "ORDER"), (0.25, "CHAOS"), (1.0, "ORDER")):
        rgb = scene_with_bar(700, 400, hp, team)
        bars = hb.find_bars(rgb, tmpl, mask, thr=0.45)
        assert len(bars) >= 1
        b = max(bars, key=lambda b: b.score)
        assert (b.x, b.y) == (700, 400)
        assert b.team == team
        assert abs(b.hp - hp) <= 0.02


def test_fill_ignores_ticks_but_stops_at_gap():
    band = np.full((10, 94, 3), 15, np.uint8)
    band[:, :47] = (60, 160, 230)
    band[:, 8:47:9] = (0, 30, 60)
    assert abs(hb.fill_fraction(band, "blue") - 0.5) < 0.02


def test_duplicate_one_row_of_frames_lower_is_suppressed():
    r = np.full((200, 400), -1.0)
    r[50, 100] = 0.9
    r[66, 100] = 0.6                       # тот же шаблон, сдвиг на рамку ресурса
    assert [(x, y) for x, y, _ in hb.peaks(r, 0.45)] == [(100, 50)]


def test_level_reader_recognises_bank_samples():
    z = np.load(hb.LEVELS)
    reader = hb.LevelReader()
    for k in (0, len(z["levels"]) // 2, len(z["levels"]) - 1):
        rgb = np.full((60, 200, 3), 40, np.uint8)
        x, y = 50, 20
        rgb[y + hb.LV_Y0:y + hb.LV_Y0 + hb.LV_H, x + hb.LV_X0:x + hb.LV_X0 + hb.LV_W] = \
            z["boxes"][k][..., None]
        level, sim = reader.read(rgb, x, y)
        assert level == int(z["levels"][k])
        assert sim > 0.99
