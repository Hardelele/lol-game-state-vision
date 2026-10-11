"""Иконки миникарты без данных: рендер, цели детектора, декодирование, состав, единицы карты."""

import numpy as np
import torch

from lolvision.minimap_champions import (MAP_UNITS, Composition, assign, frame_records,
                                         grid_px_to_units, grid_to_map)
from lolvision.minimap_icons import STRIDE, IconCls, IconDet, crops, decode
from lolvision.minimap_synth import icon_sprite, synth

LAYOUT = {"labelling": {"minimap": [0.857, 0.74, 0.993, 0.984]}}


def test_sprite_center_is_subpixel_exact():
    """Центр спрайта (по альфе) попадает в заданную дробную координату."""
    rng = np.random.default_rng(0)
    icon = np.full((128, 128, 3), 200, np.uint8)
    for x in (100.0, 100.3, 57.5, 31.9):
        rgb, al = icon_sprite(icon, rng, 0, (x % 1, x % 1))
        n = al.shape[0]
        x0 = int(np.floor(x)) - n // 2
        yy, xx = np.mgrid[:n, :n]
        cx = (al * (xx + 0.5)).sum() / al.sum() + x0
        cy = (al * (yy + 0.5)).sum() / al.sum() + x0
        assert abs(cx - x) < 0.05 and abs(cy - x) < 0.05


def test_synth_labels_and_visibility():
    rng = np.random.default_rng(1)
    icons = [np.full((128, 128, 3), v, np.uint8) for v in range(0, 250, 10)]
    bg = np.zeros((256, 256, 3), np.uint8)
    s = synth(bg, icons, rng, k=6)
    assert s.image.shape == (256, 256, 3) and s.image.dtype == np.uint8
    assert len(s.icons) == 6 and len({i.champ for i in s.icons}) == 6
    assert all(0.0 <= i.visible <= 1.0 for i in s.icons)
    assert s.icons[-1].visible > 0.4 or s.icons[-1].x > 250   # верхняя иконка не закрыта


def test_decode_reads_peak_and_offset():
    heat = torch.full((1, 2, 128, 128), -10.0)
    off = torch.zeros((1, 2, 128, 128))
    heat[0, 1, 40, 70] = 5.0          # красная иконка в клетке (x=70, y=40)
    off[0, :, 40, 70] = torch.tensor([0.25, 0.75])
    d = decode(heat, off, 0.3)[0]
    assert len(d) == 1 and d[0].team == 1
    assert abs(d[0].x - (70.25 * STRIDE)) < 1e-4 and abs(d[0].y - (40.75 * STRIDE)) < 1e-4


def test_nets_shapes():
    det, cls = IconDet().eval(), IconCls(7).eval()
    h, o = det(torch.zeros(2, 3, 256, 256))
    assert h.shape == (2, 2, 128, 128) and o.shape == (2, 2, 128, 128)
    lc, lt = cls(torch.zeros(3, 3, 32, 32))
    assert lc.shape == (3, 7) and lt.shape == (3, 2)
    c = crops(np.zeros((256, 256, 3), np.uint8), [(0.0, 0.0), (255.0, 128.0)])
    assert c.shape == (2, 32, 32, 3)


def test_grid_to_map_corners():
    """Края поля: левый край миникарты ~0, правый ~14 800; y смотрит вверх."""
    # Пиксель кадра 1653.96 по x → сетка: (1653.96 - 1645) / 261 * 256.
    u0 = (1653.96 - 1645) / 261 * 256
    u1 = (1908.42 - 1645) / 261 * 256
    v_top = (812.06 - 799) / 263 * 256
    v_bot = (1065.97 - 799) / 263 * 256
    x0, y_top = grid_to_map(u0, v_top, LAYOUT)
    x1, y_bot = grid_to_map(u1, v_bot, LAYOUT)
    assert abs(x0) < 1 and abs(x1 - MAP_UNITS) < 1
    assert abs(y_top - MAP_UNITS) < 1 and abs(y_bot) < 1
    assert 55 < grid_px_to_units(LAYOUT) < 62


def _det(team, champ, n=10, p=0.9, x=10.0, y=10.0):
    logp = np.full(n, np.log((1 - p) / (n - 1)))
    logp[champ] = np.log(p)
    return {"x": x, "y": y, "team": team, "score": 0.9, "team_p": [0.9, 0.1][::1 - 2 * team],
            "logp": logp}


def test_composition_and_assignment():
    comp = Composition(12)
    for _ in range(3):
        comp.add([_det(0, c, 12) for c in range(5)] + [_det(1, c, 12) for c in range(5, 10)])
    teams = comp.teams()
    assert sorted(teams[0]) == [0, 1, 2, 3, 4] and sorted(teams[1]) == [5, 6, 7, 8, 9]
    # Чемпион вне состава (11) не назначается; один чемпион — не больше одного раза.
    dets = [_det(0, 2, 12), _det(0, 2, 12, p=0.6), _det(1, 11, 12, p=0.95)]
    who = assign(dets, teams)
    assert who[0] == 2 and who[1] != 2 and who[2] == -1
    rec = frame_records(dets[:1], teams, [f"c{i}" for i in range(12)], LAYOUT)
    assert len(rec) == 10 and sum(r["visible"] for r in rec) == 1
    assert rec[[r["champion"] for r in rec].index("c2")]["x"] is not None


def test_hud_dead_slot_and_timer_glyphs():
    """Серый портрет с красными цифрами и пустой полосой ХП — мёртв; цифры режутся."""
    import cv2

    from lolvision import hud_panels as hp

    frame = np.zeros((1080, 1920, 3), np.uint8)
    # Живые слоты: цветной портрет и зелёная полоса ХП.
    for s, (x0, y0, x1, y1) in enumerate(hp.slot_boxes()):
        frame[y0:y1, x0:x1] = (40, 120, 200)
        hx0, hx1 = hp.HP_X_LEFT if s < 5 else hp.HP_X_RIGHT
        frame[y0 + hp.HP_Y[0]:y0 + hp.HP_Y[1], hx0:hx1] = (40, 200, 40)
    # Слот L3 мёртв: серый портрет, полоса пустая, красные «17».
    x0, y0, x1, y1 = hp.slot_boxes()[2]
    frame[y0:y1, x0:x1] = (90, 90, 90)
    frame[y0 + hp.HP_Y[0]:y0 + hp.HP_Y[1], hp.HP_X_LEFT[0]:hp.HP_X_LEFT[1]] = (20, 20, 20)
    cv2.putText(frame, "17", (x0 - 2, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.8, (30, 30, 230), 2)
    slots = hp.read_panels(frame)
    assert [s.dead for s in slots] == [False, False, True] + [False] * 7
    tx0, tx1 = hp.DIGIT_X_LEFT
    ty0 = hp.Y0 + hp.PITCH * 2 + hp.DIGIT_Y[0] - 8
    assert len(hp.glyphs(frame[ty0:ty0 + hp.DIGIT_Y[1], tx0:tx1])) == 2
