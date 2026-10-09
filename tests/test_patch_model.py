"""Геометрия, цели и сведение голосов патчевой модели — без данных и сети."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lolvision.patch_model import (MAP_GRID, aggregate, aggregate_batch, decode_points,
                         scene_from_box, soft_target_sep)


def _votes(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Голоса как у настоящего кадра: плотное согласное ядро, зеркальный
    кластер поменьше и случайный шум; часть весов нулевая."""
    core = rng.uniform(0.2, 0.8, 2)
    k1, k2 = n // 2, n // 5
    cand = np.concatenate([
        core + rng.normal(0, 0.008, (k1, 2)),
        (1 - core) + rng.normal(0, 0.008, (k2, 2)),
        rng.uniform(0, 1, (n - k1 - k2, 2)),
    ])
    w = rng.uniform(0, 1, n)
    w[rng.random(n) < 0.3] = 0.0
    return cand, w


@pytest.mark.parametrize("seed", range(20))
def test_aggregate_batch_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    cands, ws = zip(*(_votes(rng, 300) for _ in range(4)))
    cams, agree = aggregate_batch(torch.tensor(np.stack(cands)),
                                  torch.tensor(np.stack(ws)))
    for i in range(4):
        c, a = aggregate(cands[i], ws[i])
        np.testing.assert_allclose(cams[i].numpy(), c, atol=1e-6)
        assert agree[i].item() == pytest.approx(a, abs=1e-5)


def test_aggregate_without_weight_is_nan():
    rng = np.random.default_rng(0)
    cand, w = _votes(rng, 50)
    w[:] = 0.0
    c, a = aggregate(cand, w)
    assert np.isnan(c).all() and a == 0.0
    cams, agree = aggregate_batch(torch.tensor(cand[None]), torch.tensor(w[None]))
    assert torch.isnan(cams).all() and agree.item() == 0.0


def test_aggregate_empty_is_nan():
    c, a = aggregate(np.zeros((0, 2)), np.zeros(0))
    assert np.isnan(c).all() and a == 0.0


def test_decode_inverts_soft_target():
    p = torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.12, 0.88], [0.731, 0.264]])
    t = soft_target_sep(p, MAP_GRID)
    q, mass = decode_points(torch.log(t + 1e-9), MAP_GRID)
    np.testing.assert_allclose(q.numpy(), p.numpy(), atol=2e-3)
    assert (mass > 0.9).all()


def test_scene_from_box_identity():
    crop = (0, 120, 1920, 1110)
    store = (720, 372)
    a = scene_from_box(crop, store, crop, store)
    np.testing.assert_allclose(a, np.eye(3), atol=1e-12)
