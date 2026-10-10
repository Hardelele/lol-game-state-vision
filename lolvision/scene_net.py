"""Своя маленькая сеть типа сцены, дистиллированная из DINOv2 (scene_kind.py).

Зачем. DINOv2-S стоит ~10 мс на кадр (14 мс в live вместе с загрузкой
кадра) — больше, чем вся модель камеры. Тип сцены определяется крупными
элементами интерфейса с постоянным местом, для этого хватает маленькой
CNN по уменьшенному кадру. Учитель — DINO + голова: он размечает все кадры
пула, включая неразмеченные, а ученик учится повторять его ответы
(tools/distill_scene_kind.py).

Вход — весь кадр 256×144 (cv2 INTER_AREA), выход — логиты CLASSES.
Расположение элементов важно (таблица посередине, магазин слева,
клиент во весь кадр), поэтому, кроме среднего по карте, голова видит
средние по сетке 3×3.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lolvision.paths import RUNS
from lolvision.scene_kind import CLASSES

IN_W, IN_H = 256, 144
DEFAULT_NET = RUNS / "scene_kind" / "net" / "model.pt"


def conv_bn(cin: int, cout: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Block(nn.Module):
    """Свёртка с шагом 2 + depthwise/pointwise с остатком."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.down = conv_bn(cin, cout, 2)
        self.dw = conv_bn(cout, cout, 1, groups=cout)
        self.pw = nn.Sequential(nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        x = self.down(x)
        return F.relu(x + self.pw(self.dw(x)), inplace=True)


class SceneNet(nn.Module):
    def __init__(self, k: int = len(CLASSES), widths=(24, 48, 96, 160, 192)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers.append(Block(c, w))
            c = w
        self.body = nn.Sequential(*layers)          # 256×144 → 8×5 (шаг 32)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(c * 10, k)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        f = self.body((x01 - self.mean) / self.std)
        g = torch.cat([f.mean((2, 3))[:, :, None], F.adaptive_avg_pool2d(f, 3).flatten(2)], 2)
        return self.fc(self.drop(g.flatten(1)))


def small(rgb: np.ndarray) -> np.ndarray:
    """Кадр (H, W, 3) uint8 → вход 256×144 uint8.

    Сначала прореживание до ~480×270 (каждый 4-й пиксель у 1080p, каждый
    2-й у 540p), потом INTER_AREA: 1.1 мс для 1080p против 4.4 мс у
    INTER_AREA по всему кадру. Обучение идёт через эту же функцию, поэтому
    вход в live и при обучении один и тот же для кадров 1080p и 540p."""
    s = max(1, rgb.shape[0] // 270)
    return cv2.resize(np.ascontiguousarray(rgb[::s, ::s]), (IN_W, IN_H),
                      interpolation=cv2.INTER_AREA)


class SceneEnsemble(nn.Module):
    """Несколько учеников с разными зёрнами; ответ — средняя вероятность.

    Разброс между запусками обучения — до 0.7% верных кадров (почти весь в
    классе «клиент»); выбирать удачное зерно по проверочным роликам было бы
    подгонкой, а три ученика в одном графе стоят ~0.5 мс."""

    def __init__(self, nets: list[SceneNet]):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.softmax(n(x01), 1) for n in self.nets]).mean(0)


def load_net(path: Path = DEFAULT_NET, device: str = "cuda") -> SceneEnsemble:
    ck = torch.load(path, map_location=device)
    states = ck.get("states") or [ck["state"]]
    nets = []
    for st in states:
        n = SceneNet(len(ck["classes"]), tuple(ck["widths"]))
        n.load_state_dict(st)
        nets.append(n)
    ens = SceneEnsemble(nets).to(device).eval()
    ens.classes = list(ck["classes"])
    return ens


class SceneNetRunner:
    """Тип сцены для live: CUDA graph на батч 1 (на CPU — обычный вызов)."""

    def __init__(self, path: Path = DEFAULT_NET, device: str = "cuda"):
        self.net = load_net(path, device)
        self.classes = self.net.classes
        self.device = device
        self.graph = None
        if device.startswith("cuda"):
            self.x = torch.zeros(1, 3, IN_H, IN_W, device=device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self.y = self.net(self.x)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad():
                self.y = self.net(self.x)

    @torch.no_grad()
    def probs(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(small(rgb)).to(self.device).permute(2, 0, 1)[None].float() / 255
        if self.graph is None:
            return self.net(x)[0].cpu().numpy()
        self.x.copy_(x)
        self.graph.replay()
        return self.y[0].cpu().numpy()
