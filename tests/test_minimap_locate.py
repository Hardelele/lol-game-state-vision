"""Поиск миникарты в кадре: стандартный масштаб, крупная миникарта, нет миникарты."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lolvision.minimap_locate import MIN_SCORE, REFERENCE, STD_BOX, Locator, box_px

W, H = 1920, 1080


@pytest.fixture(scope="module")
def ref() -> np.ndarray:
    return cv2.imread(str(REFERENCE))[:, :, ::-1].copy()


def frame(ref: np.ndarray, box) -> np.ndarray:
    """Кадр с миникартой в прямоугольнике `box` (доли кадра) на шуме сцены."""
    rng = np.random.default_rng(0)
    fr = rng.integers(20, 90, (H, W, 3), dtype=np.uint8)
    x0, y0, x1, y1 = box_px(box, (W, H))
    fr[y0:y1, x0:x1] = cv2.resize(ref, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    return fr


def scaled(scale: float):
    """Прямоугольник миникарты в `scale` раз крупнее, у того же правого нижнего угла."""
    x0, y0, x1, y1 = STD_BOX
    return (x1 - scale * (x1 - x0), y1 - scale * (y1 - y0), x1, y1)


def test_standard_snaps_to_layout(ref):
    loc = Locator((W, H), ref)
    fit = loc.locate(frame(ref, STD_BOX)[loc.region[1]:, loc.region[0]:])
    assert fit.box == STD_BOX and fit.scale == 1.0 and fit.score > MIN_SCORE


@pytest.mark.parametrize("scale", [1.05, 1.3, 1.5])
def test_larger_minimap(ref, scale):
    loc = Locator((W, H), ref)
    fr = frame(ref, scaled(scale))
    fit = loc.locate(fr[loc.region[1]:, loc.region[0]:])
    assert abs(fit.scale - scale) <= 0.01
    want = box_px(scaled(scale), (W, H))
    assert max(abs(a - b) for a, b in zip(box_px(fit.box, (W, H)), want)) <= 2


def test_needs_agreeing_frames(ref):
    loc = Locator((W, H), ref)
    fr = frame(ref, scaled(1.3))
    assert not loc.offer(fr) and not loc.offer(fr)
    assert loc.offer(fr) and loc.box is not None


def test_no_minimap_is_not_found(ref):
    rng = np.random.default_rng(1)
    loc = Locator((W, H), ref)
    for _ in range(4):
        assert not loc.offer(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    assert loc.box is None

