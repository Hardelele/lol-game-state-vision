"""Аугментации обучения патчевой модели.

* synth_ui — синтетическая «панель интерфейса» для заслонок поверх сцены
  (train_patches.py) и для кадров, где игры нет вовсе (probe_hud.py);
* color_gpu — яркость, контраст и баланс каналов батча на видеокарте.
"""

from __future__ import annotations

import string

import cv2
import numpy as np
import torch

_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
          cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX,
          cv2.FONT_HERSHEY_TRIPLEX)
_CHARS = string.ascii_letters + string.digits + "  /:%+-"


def synth_ui(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная «панель интерфейса» (h, w, 3) uint8.

    Не копия чьего-то HUD, а его общие черты: тёмная подложка с градиентом,
    рамка, строки текста, полоски здоровья/маны, квадратные значки. Этого
    достаточно, чтобы модель училась отличать нарисованное поверх игры от
    самой игры, не запоминая конкретную трансляцию.
    """
    w, h = max(int(w), 2), max(int(h), 2)
    c0 = rng.integers(0, 70, 3)
    c1 = np.clip(c0 + rng.integers(-30, 60, 3), 0, 255)
    t = np.linspace(0, 1, h)[:, None, None]
    img = (c0[None, None] * (1 - t) + c1[None, None] * t).repeat(w, 1)
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(2, 14), (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    gold = tuple(int(v) for v in rng.integers(120, 230, 3))
    if rng.random() < 0.7:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), gold, int(rng.integers(1, 4)))
    for _ in range(int(rng.integers(0, 6))):            # значки
        s = int(rng.integers(6, max(7, min(h, w) // 2 + 1)))
        x, y = int(rng.integers(0, max(1, w - s))), int(rng.integers(0, max(1, h - s)))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + s), col, -1)
            cv2.rectangle(img, (x, y), (x + s, y + s), gold, 1)
        else:
            cv2.circle(img, (x + s // 2, y + s // 2), s // 2, col, -1)
    for _ in range(int(rng.integers(0, 4))):            # полоски
        bw = int(rng.integers(max(2, w // 6), max(3, w)))
        bh = int(rng.integers(2, max(3, h // 8 + 3)))
        x, y = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        col = [(40, 200, 60), (200, 40, 40), (50, 110, 230), (230, 200, 60)][
            int(rng.integers(0, 4))]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (10, 10, 10), -1)
        cv2.rectangle(img, (x, y), (x + int(bw * rng.uniform(0.1, 1)), y + bh), col, -1)
    for _ in range(int(rng.integers(0, 7))):            # текст
        n = int(rng.integers(2, 16))
        txt = "".join(rng.choice(list(_CHARS), n))
        sc = float(rng.uniform(0.3, 1.3))
        col = tuple(int(v) for v in (rng.integers(170, 256, 3) if rng.random() < 0.7
                                     else rng.integers(0, 256, 3)))
        org = (int(rng.integers(0, max(1, w - 10))), int(rng.integers(8, max(9, h))))
        cv2.putText(img, txt, org, _FONTS[int(rng.integers(0, len(_FONTS)))], sc, col,
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    return img


def color_gpu(x: torch.Tensor) -> torch.Tensor:
    """Яркость, контраст и баланс каналов; x в [0, 1]."""
    n, dev = x.shape[0], x.device
    c = torch.rand(n, 1, 1, 1, device=dev) * 0.4 + 0.8
    b = torch.rand(n, 1, 1, 1, device=dev) * 0.16 - 0.08
    g = torch.rand(n, 3, 1, 1, device=dev) * 0.12 + 0.94
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m + b).mul_(g).clamp_(0, 1)
