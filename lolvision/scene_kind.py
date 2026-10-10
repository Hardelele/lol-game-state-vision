"""Тип сцены по всему кадру: игра, Tab, магазин, клиент, загрузка, не League.

Замороженный DINOv2-S по кадру 392×224 (сетка патчей 28×16) → признаки
(токен CLS, среднее патчей, средние по сетке 3×3) → маленькая голова
(tools/scene_kind_dino.py, docs/not-game.md). Итоговый ответ — вместе с
головой «игра / не игра» патчевой модели: DINO узнаёт интерфейс, а
патчевая модель — места Ущелья; «игра» при P(игра) ниже её порога
становится «other» (чужие игры с видом сверху, ARAM на других картах).

Стоимость — ~10 мс на кадр на RTX 5060 Ti (fp32, без графов; fp16 не
быстрее), поэтому в live тип сцены считается раз в несколько кадров.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lolvision.paths import RUNS

CLASSES = ["game", "tab", "shop", "client", "loading", "other"]
IN_W, IN_H = 392, 224
FEAT_DIM = 11 * 384
DEFAULT_HEAD = RUNS / "scene_kind" / "dino-s" / "head.pt"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def load_dino(arch: str = "dinov2_vits14", dev: str = "cuda") -> nn.Module:
    repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    vit = (torch.hub.load(str(repo), arch, source="local") if repo.exists()
           else torch.hub.load("facebookresearch/dinov2", arch))
    return vit.to(dev).eval()


@torch.no_grad()
def dino_features(vit: nn.Module, x01: torch.Tensor) -> torch.Tensor:
    """x01: (B, 3, IN_H, IN_W) в [0, 1] → (B, 11·C)."""
    mean = torch.tensor(_MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x01.device).view(1, 3, 1, 1)
    f = vit.forward_features((x01 - mean) / std)
    b = x01.shape[0]
    pt = f["x_norm_patchtokens"].transpose(1, 2).reshape(b, -1, IN_H // 14, IN_W // 14)
    grid = F.adaptive_avg_pool2d(pt, 3).flatten(2).transpose(1, 2)
    return torch.cat([f["x_norm_clstoken"][:, None], pt.mean((2, 3))[:, None], grid],
                     1).flatten(1).float()


def to_input(rgb: np.ndarray, dev: str) -> torch.Tensor:
    """Кадр (H, W, 3) uint8 любого размера → вход (1, 3, IN_H, IN_W)."""
    a = rgb if rgb.flags.writeable and rgb.flags.c_contiguous else np.array(rgb)
    x = torch.from_numpy(a).to(dev).permute(2, 0, 1)[None].float() / 255
    return F.interpolate(x, size=(IN_H, IN_W), mode="bilinear", align_corners=False,
                         antialias=True)


class Head(nn.Module):
    def __init__(self, cin: int, k: int, hidden: int = 256):
        super().__init__()
        self.register_buffer("mu", torch.zeros(cin))
        self.register_buffer("sd", torch.ones(cin))
        self.net = nn.Sequential(nn.Dropout(0.3), nn.Linear(cin, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(0.3), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def fuse(cls: int, p_game: float | None, threshold: float) -> int:
    """«Игра» по DINO без подтверждения патчевой моделью — «other»."""
    if cls == 0 and p_game is not None and p_game < threshold:
        return CLASSES.index("other")
    return cls


class SceneKind:
    """Классификатор типа сцены для live: кадр → (класс, вероятности)."""

    def __init__(self, head: Path = DEFAULT_HEAD, device: str = "cuda"):
        ck = torch.load(head, map_location=device)
        self.classes = list(ck["classes"])
        self.head = Head(ck["cin"], len(self.classes)).to(device).eval()
        self.head.load_state_dict(ck["state"])
        self.vit = load_dino(ck.get("arch", "dinov2_vits14"), device)
        self.device = device

    @torch.no_grad()
    def probs(self, rgb: np.ndarray) -> np.ndarray:
        """Вероятности классов по кадру (H, W, 3) uint8, без объединения."""
        return torch.softmax(self.head(dino_features(self.vit, to_input(rgb, self.device))),
                             1)[0].cpu().numpy()

    def __call__(self, rgb: np.ndarray, p_game: float | None = None,
                 threshold: float = 0.5) -> tuple[str, np.ndarray]:
        p = self.probs(rgb)
        return self.classes[fuse(int(p.argmax()), p_game, threshold)], p
