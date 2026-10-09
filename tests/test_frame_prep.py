"""FramePrep (подготовка кадра на видеокарте) против пути на процессоре."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.patch_infer import FramePrep, PatchInfer
from lolvision.patch_model import UNITS_PER_PX, load_projection
from lolvision.paths import DATASET

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
def test_frame_prep_matches_cpu_path():
    # Без чекпоинта: FramePrep берёт у раннера только устройство, геометрию
    # и prepare, а prepare — это rectify с шагом патча.
    h0, store = load_projection()
    infer = SimpleNamespace(device="cuda", h0=h0, units=UNITS_PER_PX,
                            net=SimpleNamespace(stride=16))
    infer.prepare = lambda scene: PatchInfer.prepare(infer, scene)

    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    w, h = 1920, 1080
    crop = crop_box((w, h), layout, "dense")
    boxes = mask_boxes((w, h), layout)
    cx0, cy0, cx1, cy1 = crop
    sw = 720
    sh = round(sw * (cy1 - cy0) / (cx1 - cx0))

    # Плавная картинка: на ней разные ядра уменьшения дают почти одно и то же,
    # и сравнивается геометрия, а не фильтры.
    rng = np.random.default_rng(0)
    small = rng.uniform(0, 255, (9, 16, 3)).astype(np.float32)
    frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    (cv_gpu, val_gpu, q_gpu), scene_gpu = FramePrep(infer, (w, h), crop, boxes, (sw, sh))(frame)

    a = frame[cy0:cy1, cx0:cx1].copy()
    for x0, y0, x1, y1 in boxes:
        if x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1:
            a[max(y0, cy0) - cy0:min(y1, cy1) - cy0,
              max(x0, cx0) - cx0:min(x1, cx1) - cx0] = 0
    scene_cpu = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    cv_cpu, val_cpu, q_cpu = infer.prepare(scene_cpu)

    assert scene_gpu.shape == scene_cpu.shape
    np.testing.assert_allclose(q_gpu, q_cpu)
    gpu = cv_gpu[0].permute(1, 2, 0).cpu().numpy() * 255
    assert gpu.shape == cv_cpu.shape
    inside = (val_cpu == 255) & (val_gpu[0].cpu().numpy() == 1)
    # Края масок HUD и следа разные фильтры размывают по-разному;
    # сравниваем внутренность.
    inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    diff = np.abs(gpu - cv_cpu.astype(np.float32))[inside]
    assert inside.mean() > 0.5
    assert np.median(diff) < 1.0
