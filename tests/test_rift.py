"""Место на Ущелье и его смысл (#823): доли рамки ↔ мировые, зоны, башни.

Проверка на стыке — кадры своей игры на Экко 2026-10-14 (CHAOS, миникарта
×1.205, docs/state-api.md «Место на карте»): иконки 22 башен на миникарте
после перевода сервисом ложатся у своих башен («у башни»), а своя иконка
после шести возрождений — на своей базе.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lolvision import minimap_champions, minimap_locate, rift
from lolvision.match_state import FOUNTAIN
from lolvision.paths import RUNS
from lolvision.rift import (AT_TURRET, FOUNTAIN_WORLD, PITS, TURRETS, crop_to_world, describe,
                            describe_crop, fallen_turrets, turret_key, world_to_crop,
                            world_to_field, zone)

EKKO = Path(__file__).parent / "fixtures" / "ekko"
DEFAULT_RUN = RUNS / "minimap" / "icons-syn"           # как minimap_icons.DEFAULT_RUN


def test_one_frame_convention_and_the_border_formula():
    """Окантовка — в одном месте: та же формула, что в разборе (Replay API)."""
    assert rift.STD_BOX == minimap_locate.STD_BOX
    assert (minimap_champions.MAP_X0, minimap_champions.MAP_Y_BOTTOM) == (rift.MAP_X0,
                                                                        rift.MAP_Y_BOTTOM)
    u0, v1 = world_to_crop(0, rift.MAP_UNITS)       # левый верхний угол поля
    u1, v0 = world_to_crop(rift.MAP_UNITS, 0)
    assert u0 == pytest.approx(0.0343, abs=2e-4) and v1 == pytest.approx(0.0497, abs=2e-4)
    assert u1 - u0 == pytest.approx(0.9749, abs=2e-4) and v0 - v1 == pytest.approx(0.9654, abs=2e-4)
    for x, y in [(0, 0), (981, 10441), (7400, 7400), (14800, 14800)]:
        assert crop_to_world(*world_to_crop(x, y)) == pytest.approx((x, y), abs=1e-6)
    assert world_to_field(1480, 13320) == pytest.approx((0.1, 0.1))


def test_fountains_come_from_the_observed_respawn():
    """CHAOS — наблюдаемое возрождение (0.9775, 0.0903); ORDER — отражение."""
    assert FOUNTAIN["CHAOS"] == pytest.approx((0.9775, 0.0903), abs=1e-4)
    assert FOUNTAIN == rift.FOUNTAIN_CROP
    (ox, oy), (cx, cy) = FOUNTAIN_WORLD["ORDER"], FOUNTAIN_WORLD["CHAOS"]
    assert (ox + cx, oy + cy) == pytest.approx((2 * rift.CENTER[0], 2 * rift.CENTER[1]), abs=1)
    for team, (x, y) in FOUNTAIN_WORLD.items():
        p = describe(x, y, team)
        assert (p["zone"], p["base"], p["half"], p["deep"]) == ("base", team, "ally", False)
        q = describe(x, y, "CHAOS" if team == "ORDER" else "ORDER")
        assert (q["half"], q["deep"]) == ("enemy", True)


def test_turret_places_are_their_lanes_and_nexus_turrets_are_not_mid():
    for (team, lane, tier), (x, y) in TURRETS.items():
        assert zone(x, y) == ("base" if lane == "nexus" else lane), (team, lane, tier)
        p = describe(x, y, "ORDER", frozenset())
        t = p["turret"]
        assert (t["team"], t["lane"], t["tier"]) == (team, lane, tier)
        assert (t["dist"], t["near"]) == (0, True)
        assert t["relation"] == ("ally" if team == "ORDER" else "enemy")
        assert p["team_half"] == team
    assert {zone(*xy) for xy in PITS.values()} == {"river"}
    assert len(TURRETS) == 22 and sum(k[1] == "nexus" for k in TURRETS) == 4


@pytest.mark.parametrize("name, key", [
    ("Turret_TChaos_L1_P4_392430785_0", ("CHAOS", "nexus", 4)),
    ("Turret_TOrder_L1_P5_3625540808_0", ("ORDER", "nexus", 5)),
    ("Turret_TOrder_L2_P3_1509986696_0", ("ORDER", "top", 3)),
    ("Turret_TChaos_L0_P1_478290356_0", ("CHAOS", "bot", 1)),
    ("Turret_T2_C_05_A", ("CHAOS", "mid", 3)), ("Turret_T1_C_01_A", ("ORDER", "nexus", 4)),
    ("Turret_T1_R_03_A", ("ORDER", "bot", 3)),
    ("SRU_Baron", None), (None, None)])
def test_turret_names_of_the_api(name, key):
    assert turret_key(name) == key


def test_nearest_turret_skips_fallen_and_says_when_it_does_not_know():
    x, y = TURRETS[("ORDER", "top", 3)]
    gone = fallen_turrets(["Turret_TOrder_L2_P3_1509986696_0"])
    p = describe(x, y + 100, "CHAOS", gone, with_turrets=True)
    assert (p["turret"]["lane"], p["turret"]["tier"]) == ("top", 2)
    assert p["turret"]["standing"] is True and p["turret"]["near"] is False
    assert len(p["turrets"]) == 21 and p["turrets"] == sorted(p["turrets"], key=lambda t: t["dist"])
    unknown = describe(x, y + 100, "CHAOS", None)
    assert unknown["turret"]["tier"] == 3 and unknown["turret"]["standing"] is None
    assert "turrets" not in unknown


def test_relative_fields_need_own_team_and_sure_drops_at_a_border():
    p = describe(7400, 7450, None)
    assert (p["zone"], p["side"], p["half"], p["deep"]) == ("mid", "mid", None, None)
    assert p["turret"]["relation"] is None
    # У границы мида и реки: в круге ошибки зона меняется — не уверены.
    edge = describe(9000, 4860, "ORDER", err=250)
    assert edge["sure"] is False
    assert describe(1200, 9000, "ORDER", err=150)["sure"] is True


def test_pits_and_lanes_with_bent_corners():
    assert describe(*PITS["dragon"], "ORDER")["pit"] == "dragon"
    assert describe(5007 - 300, 10471, "ORDER")["pit"] == "baron"
    assert describe(3000, 3000 + 9000, "ORDER")["zone"] == "top"        # загиб угла топа
    assert describe(14800 - 3000, 14900 - 12000, "ORDER")["zone"] == "bot"
    assert describe(3800, 7900, "ORDER")["zone"] == "jungle"            # их синий бафф ORDER
    p = describe(11000, 8500, "ORDER")
    assert (p["zone"], p["half"], p["deep"]) == ("jungle", "enemy", True)


# ---------------------------------------------------------------- кадры Экко

def _turret_icons(path: Path) -> list[tuple[tuple, float, float]]:
    """Центроиды иконок башен на миникарте (сетка 256, доли рамки) у каждой
    башни: красные — ORDER (враги Экко), синие — CHAOS (свои)."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    X, Y = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)
    out = []
    for key, xy in TURRETS.items():
        cu, cv = world_to_crop(*xy)
        m = (b > 150) & (b > r + 40) if key[0] == "CHAOS" else \
            (r > 140) & (r > g + 60) & (r > b + 40)
        w = m & (np.hypot(X - cu * 256, Y - cv * 256) < 12)
        if w.sum() >= 8:
            out.append((key, X[w].mean() / 256, Y[w].mean() / 256))
    return out


@pytest.mark.parametrize("t, found", [(30, 22), (865, 20)])
def test_ekko_turret_icons_are_at_their_turrets(t, found):
    """t=30 — все 22 башни; 14:25 — 20 (две башни ORDER на топе уже пали)."""
    icons = _turret_icons(EKKO / f"minimap_t{t:04d}.png")
    assert len(icons) == found
    dists = []
    for key, u, v in icons:
        p = describe_crop(u, v, "minimap", "CHAOS", frozenset())
        tu = p["turret"]
        assert (tu["team"], tu["lane"], tu["tier"]) == key and tu["near"], (key, tu)
        assert tu["relation"] == ("ally" if key[0] == "CHAOS" else "enemy")
        if key[2] in (2, 3):                      # внешние и внутренние — на своей линии
            assert p["zone"] == key[1], (key, p["zone"])
        if key[1] == "nexus":
            assert p["zone"] == "base" and p["base"] == key[0]
        dists.append(tu["dist"])
    # Медиана — шум цвета (миньоны, база вокруг иконки), а не сдвиг рамки.
    assert np.median(dists) < 200 and max(dists) < 500 < AT_TURRET


def test_ekko_respawns_are_at_own_base():
    """Своя иконка (детектор IconModel) после шести возрождений — база CHAOS."""
    d = json.loads((EKKO / "respawns.json").read_text(encoding="utf-8"))
    ts = [f["t"] for f in d["frames"]]
    assert len(ts) == 12 and 1 + sum(b - a > 30 for a, b in zip(ts, ts[1:])) == 6
    for f in d["frames"]:
        p = describe_crop(f["x"] / 256, f["y"] / 256, "minimap", "CHAOS", frozenset())
        assert (p["zone"], p["base"], p["half"], p["sure"]) == ("base", "CHAOS", "ally", True)
        assert math.dist((p["x"], p["y"]), FOUNTAIN_WORLD["CHAOS"]) < 100


@pytest.mark.skipif(not (DEFAULT_RUN / "model.pt").exists(), reason="нет модели иконок")
@pytest.mark.parametrize("t", [225, 915])
def test_ekko_respawn_from_pixels(t):
    """Тот же вывод от пикселей: детектор иконок на кадре миникарты → место."""
    from lolvision.minimap_icons import IconModel
    model = IconModel(DEFAULT_RUN)
    ekko = model.champions.index("Ekko")
    mini = np.asarray(Image.open(EKKO / f"minimap_t{t:04d}.png").convert("RGB"))
    dets = model(mini[None])[0]
    best = max(dets, key=lambda d: d["logp"][ekko])
    p = describe_crop(best["x"] / 256, best["y"] / 256, "minimap", "CHAOS")
    assert (p["zone"], p["base"], p["half"]) == ("base", "CHAOS", "ally")
