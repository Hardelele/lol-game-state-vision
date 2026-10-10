"""Своя сеть типа сцены: размеры входа и выхода, ансамбль (без данных и видеокарты)."""

import numpy as np
import torch

from lolvision.scene_kind import CLASSES, fuse
from lolvision.scene_net import IN_H, IN_W, SceneEnsemble, SceneNet, small


def test_small_gives_same_size_for_1080p_and_540p():
    for h, w in ((1080, 1920), (540, 960), (720, 1280)):
        assert small(np.zeros((h, w, 3), np.uint8)).shape == (IN_H, IN_W, 3)


def test_small_keeps_layout():
    """Белая полоса в левой четверти кадра остаётся в левой четверти входа."""
    fr = np.zeros((1080, 1920, 3), np.uint8)
    fr[:, :480] = 255
    x = small(fr)
    assert x[:, : IN_W // 4 - 2].min() > 200 and x[:, IN_W // 4 + 2:].max() < 50


def test_net_and_ensemble_shapes():
    torch.manual_seed(0)
    nets = [SceneNet().eval() for _ in range(3)]
    x = torch.rand(2, 3, IN_H, IN_W)
    with torch.no_grad():
        assert nets[0](x).shape == (2, len(CLASSES))
        p = SceneEnsemble(nets).eval()(x)
    assert p.shape == (2, len(CLASSES))
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)


def test_fuse_turns_unconfirmed_game_into_other():
    game, other, tab = (CLASSES.index(c) for c in ("game", "other", "tab"))
    assert fuse(game, 0.01, 0.035) == other
    assert fuse(game, 0.9, 0.035) == game
    assert fuse(tab, 0.01, 0.035) == tab          # интерфейс поверх — не трогаем
    assert fuse(game, None, 0.035) == game        # ответа головы нет
