"""Имя над чемпионом в потоке (lolstream/live_champions.py): по центру тела
от детектора находится и читается среди пяти кандидатов, со сдвигом тоже."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from lolvision.health_bars import W as PLATE_W
from lolvision.identity import NAME_BOX
from lolvision.name_reader import FONT, WIN_H, WIN_W, NameReader, render
from lolstream import live_champions as lc

pytestmark = pytest.mark.skipif(not FONT.exists(), reason="нет шрифта Constantia")

CANDS = ["Volibear", "Irelia", "Hecarim", "Jinx", "Lulu"]


def reader() -> lc.LiveChampions:
    """LiveChampions без моделей: только банк имён (отрисованные образцы)."""
    names = NameReader.rendered({c: c for c in CANDS + ["Evelynn"]})
    obj = object.__new__(lc.LiveChampions)
    obj.names = names
    obj.name_tm = np.ascontiguousarray(names.bank.T.reshape(-1, WIN_H, WIN_W))
    obj.reads = deque(maxlen=lc.VIEW_READS)
    obj._view = None
    obj.match = None
    return obj


def frame_with(name: str, x: float, y: float, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Кадр 1080p: тёмный фон и имя там, где оно стоит над чемпионом с
    центром тела (x, y), со сдвигом (dx, dy) — ошибка детектора."""
    f = np.full((1080, 1920, 3), 40, np.uint8)
    w = render(name)
    px = round(x - PLATE_W / 2) + NAME_BOX[0] + dx
    py = round(y - lc.PLATE_DY) + NAME_BOX[1] + dy
    f[py:py + w.shape[0], px:px + w.shape[1]] = (110 + 100 * w)[..., None].astype(np.uint8)
    return f


@pytest.mark.parametrize("dx,dy", [(0, 0), (5, -7), (-5, 7)])
def test_name_found_near_detector_center(dx, dy):
    r = reader()
    for name in ("Hecarim", "Lulu"):
        s = r.name_scores(frame_with(name, 900, 500, dx, dy), 900, 500, CANDS)
        assert s is not None and CANDS[int(np.argmax(s))] == name


def test_who_assigns_distinct_names_and_tracks():
    r = reader()
    r.prev = []
    r.match = SimpleNamespace(roster=["Irelia", "Evelynn", "A", "B", "C"] + CANDS, api=False)
    f = frame_with("Hecarim", 900, 500)
    units = [SimpleNamespace(kind="champion", team="CHAOS", x=900, y=500),
             SimpleNamespace(kind="minion", team="CHAOS", x=300, y=300)]
    assert r.who(f, units) == ["Hecarim", None]
    # Имени не видно (пустой кадр), но чемпион рядом с прошлым местом.
    empty = np.full((1080, 1920, 3), 40, np.uint8)
    moved = [SimpleNamespace(kind="champion", team="CHAOS", x=960, y=520)]
    assert r.who(empty, moved) == ["Hecarim"]


def test_who_silent_in_player_view():
    # У вида игрока над плашкой ник: имя чемпиона не выдумывается ни чтением,
    # ни переносом с прошлого кадра (#786: Brand был «Ivern» в 100% кадров).
    r = reader()
    r.prev = []
    r.match = SimpleNamespace(roster=["Irelia", "Evelynn", "A", "B", "C"] + CANDS, api=False)
    r.reads.extend([lc.PLAYER] * lc.VIEW_READS)
    units = [SimpleNamespace(kind="champion", team="CHAOS", x=900, y=500)]
    assert r.who(frame_with("Hecarim", 900, 500), units) == [None]
