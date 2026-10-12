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



# ---------- другие карты: поиск по рамке (#803) ----------

from pathlib import Path  # noqa: E402

from lolvision.minimap_locate import (FRAME_MIN, Fit, corner_px, load_fit, save_fit,  # noqa: E402
                                      scale_box)

# Миникарта ARAM своего экрана (Воющая бездна, ×1.2) с полями 12 px: вырез
# из кадра 1920×1080, левый верх выреза — (1575, 728). Ников на ней нет.
ARAM = Path(__file__).parent / "fixtures" / "minimap" / "aram_1920x1080_x1575_y728.jpg"
ARAM_AT = (1575, 728)
# Тот же экран на Ущелье — прямоугольник, найденный эталоном (×1.205).
OWN_FIT = (0.82682, 0.6854, 0.9907, 0.97942)


def aram_frame() -> np.ndarray:
    rng = np.random.default_rng(2)
    fr = rng.integers(20, 90, (H, W, 3), dtype=np.uint8)
    cut = cv2.imread(str(ARAM))[:, :, ::-1]
    x, y = ARAM_AT
    fr[y:y + cut.shape[0], x:x + cut.shape[1]] = cut
    return fr


def corner(loc, fr):
    return fr[loc.region[1]:, loc.region[0]:]


def test_scale_box_follows_measured_minimaps():
    """Прямая угла миникарты: 1.0 — раскладка, 1.205 и 0.88 — замеры эталоном."""
    assert scale_box(1.0) == STD_BOX
    assert corner_px(scale_box(1.205), OWN_FIT, (W, H)) <= 1.0
    assert corner_px(scale_box(0.88), (0.8742, 0.77246, 0.99388, 0.98718), (W, H)) <= 1.0


def test_aram_found_by_frame_not_by_rift_reference(ref):
    loc = Locator((W, H), ref)
    fr = aram_frame()
    by_map = loc.locate_map(corner(loc, fr))
    assert by_map is None or by_map.score < MIN_SCORE           # эталон Ущелья — мимо
    fit = loc.locate(corner(loc, fr))
    assert fit.how == "frame" and fit.score >= FRAME_MIN
    assert abs(fit.scale - 1.205) <= 0.005
    assert corner_px(fit.box, OWN_FIT, (W, H)) <= 1.5
    assert not loc.offer(fr) and not loc.offer(fr) and loc.offer(fr)
    assert loc.fit.how == "frame"


def test_frame_search_is_resolution_independent(ref):
    for size in ((1280, 720), (2560, 1440)):
        fr = cv2.resize(aram_frame(), size, interpolation=cv2.INTER_AREA)
        loc = Locator(size, ref)
        fit = loc.locate_frame(corner(loc, fr))
        assert fit is not None and corner_px(fit.box, OWN_FIT, size) <= 1.5 * max(1, size[1] / H)


def test_frame_on_synthetic_minimap(ref):
    """Рамка эталона, а внутри — не Ущелье (шум): находит по рамке."""
    from lolvision.minimap_locate import FRAME_BAND
    odd = ref.copy()
    rng = np.random.default_rng(3)
    b = FRAME_BAND + 2
    odd[b:, b:] = rng.integers(0, 60, odd[b:, b:].shape, dtype=np.uint8)
    loc = Locator((W, H), ref)
    fit = loc.locate(corner(loc, frame(odd, scale_box(1.3))))
    assert fit.how == "frame" and abs(fit.scale - 1.3) <= 0.005
    assert corner_px(fit.box, scale_box(1.3), (W, H)) <= 1.5


def test_no_frame_in_noise(ref):
    loc = Locator((W, H), ref)
    rng = np.random.default_rng(4)
    reg = corner(loc, rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    assert loc.locate_frame(reg) is None


def test_saved_fit_is_taken_when_frame_is_there(ref):
    saved = Fit(OWN_FIT, 1.205, 0.61)
    loc = Locator((W, H), ref, saved=saved)
    fit = loc.locate(corner(loc, aram_frame()))
    assert fit.how == "saved" and fit.box == OWN_FIT and fit.scale == 1.205
    # Сохранённый от другого масштаба: рамки на его месте нет — поиск по рамке.
    loc = Locator((W, H), ref, saved=Fit(STD_BOX, 1.0, 0.6))
    fit = loc.locate(corner(loc, aram_frame()))
    assert fit.how == "frame" and corner_px(fit.box, OWN_FIT, (W, H)) <= 1.5


def test_manual_scale_needs_no_search(ref):
    loc = Locator((W, H), ref, manual=1.2)
    assert loc.offer(np.zeros((H, W, 3), np.uint8))            # с первого кадра
    assert loc.fit.how == "manual" and loc.box == scale_box(1.2) and loc.tried == 1
    assert Locator((W, H), ref, manual=1.0).locate(None).box == STD_BOX


def test_fit_store_roundtrip(tmp_path):
    path = tmp_path / "lolstream" / "minimap_fit.json"
    assert load_fit(path, (W, H)) is None                       # файла нет
    save_fit(path, (W, H), Fit(OWN_FIT, 1.205, 0.61))
    save_fit(path, (1280, 720), Fit(STD_BOX, 1.0, 0.7))
    got = load_fit(path, (W, H))
    assert got.box == OWN_FIT and got.scale == 1.205 and got.how == "saved"
    assert load_fit(path, (1280, 720)).box == STD_BOX           # другой размер не затёрт
    assert load_fit(path, (800, 600)) is None
    path.write_text("{битый", encoding="utf-8")
    assert load_fit(path, (W, H)) is None
