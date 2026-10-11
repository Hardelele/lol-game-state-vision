"""Детектор юнитов на основном виде: чемпионы, миньоны, варды — где и чьи.

Зачем. Этап 3 плана второй модели (docs/entity-model-plan.md): классические
читатели полосок (health_bars, unit_bars) медленные (~200 мс на кадр на
CPU) и ломаются на перекрытиях; детектор учится на их псевдо-метках
(tools/build_unit_labels.py, tools/train_units.py) и работает за несколько
миллисекунд.

Постановка в духе CenterNet, без якорей и NMS:

* вход — кадр 960×544 (1080p, уменьшенный вдвое, снизу 4 строки поля),
  HUD закрыт маской раскладки, как у модели камеры;
* выход с шагом 4 (240×136): тепловая карта центров тел по классам
  CLASSES (сигмоида), логит команды (CHAOS против ORDER), доля ХП
  (сигмоида; учитель — заливка полоски) и субпиксельный сдвиг центра;
* ответ — локальные максимумы тепловой карты (max-pool 3×3) выше порога.

Рамка тела — не выход сети: метки рамок у нас по постоянным смещениям от
полоски (unit_bars.BODY), учить по ним размер бессмысленно. Рамка
восстанавливается по классу (размер — не признак класса, раздел 3.4 плана,
поэтому и сеть его не видит как признак: обучение с масштабом кадра
×0.6–1.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lolvision.paths import RUNS

CLASSES = ("champion", "minion", "ward")
TEAMS = ("ORDER", "CHAOS")
IN_W, IN_H = 960, 544
STRIDE = 4
OUT_W, OUT_H = IN_W // STRIDE, IN_H // STRIDE
DEFAULT_NET = RUNS / "units" / "net" / "model.pt"


def conv_bn(cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res(nn.Module):
    """Depthwise 5×5 + pointwise с остатком: поле зрения растёт дёшево."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = conv_bn(c, c, 5, groups=c)
        self.pw = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.pw(self.dw(x)), inplace=True)


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int):
        super().__init__()
        self.down = conv_bn(cin, cout, 3, 2)
        self.blocks = nn.Sequential(*[Res(cout) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.down(x))


class UnitNet(nn.Module):
    """Энкодер с шагами 2…32 и FPN-декодер до шага 4."""

    def __init__(self, widths=(24, 48, 96, 160, 256), depths=(1, 2, 3, 3, 2), head: int = 64,
                 k: int = len(CLASSES)):
        super().__init__()
        self.widths, self.depths, self.head_c = tuple(widths), tuple(depths), head
        stages, c = [], 3
        for w, d in zip(widths, depths):
            stages.append(Stage(c, w, d))
            c = w
        self.stages = nn.ModuleList(stages)
        # латерали шагов 4, 8, 16, 32
        self.lat = nn.ModuleList([nn.Conv2d(w, head, 1) for w in widths[1:]])
        self.smooth = nn.ModuleList([conv_bn(head, head) for _ in widths[1:-1]])
        self.neck = nn.Sequential(conv_bn(head, head), Res(head))
        self.heat = nn.Conv2d(head, k, 1)
        self.team = nn.Conv2d(head, 1, 1)
        self.off = nn.Conv2d(head, 2, 1)
        self.hp = nn.Conv2d(head, 1, 1)
        nn.init.constant_(self.heat.bias, -4.6)     # априорная вероятность ~0.01
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = (x01 - self.mean) / self.std
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        feats = feats[1:]                            # шаги 4, 8, 16, 32
        y = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="nearest") + self.lat[i](feats[i])
            y = self.smooth[i](y)
        y = self.neck(y)
        return self.heat(y), self.team(y), self.off(y), self.hp(y)


@dataclass
class Unit:
    kind: str
    team: str
    score: float
    team_p: float        # вероятность CHAOS
    hp: float            # доля ХП 0..1
    x: float             # центр тела в пикселях исходного кадра
    y: float


def decode(heat: torch.Tensor, team: torch.Tensor, off: torch.Tensor, hp: torch.Tensor,
           thr: float = 0.3, topk: int = 100) -> torch.Tensor:
    """Локальные максимумы → тензор (N, 7): класс, оценка, P(CHAOS), x, y, батч, ХП
    в координатах входа сети. Без синхронизаций с CPU — годится в CUDA graph
    (фиксированный topk; строки с оценкой ниже порога отбрасываются потом)."""
    b, k, h, w = heat.shape
    p = torch.sigmoid(heat.float())
    peak = (F.max_pool2d(p, 3, 1, 1) == p).float() * p
    sc, idx = peak.reshape(b, -1).topk(topk)
    cls = idx // (h * w)
    pix = idx % (h * w)
    gy, gx = (pix // w).float(), (pix % w).float()
    o = off.float().reshape(b, 2, -1)
    ox = o[:, 0].gather(1, pix)
    oy = o[:, 1].gather(1, pix)
    tp = torch.sigmoid(team.float().reshape(b, -1).gather(1, pix))
    hv = torch.sigmoid(hp.float().reshape(b, -1).gather(1, pix))
    bi = torch.arange(b, device=heat.device)[:, None].expand(b, topk).float()
    x = (gx + ox) * STRIDE
    y = (gy + oy) * STRIDE
    sc = torch.where(sc >= thr, sc, torch.zeros_like(sc))
    return torch.stack([cls.float(), sc, tp, x, y, bi, hv], -1)


def to_units(rows: np.ndarray, scale: float = 2.0) -> list[Unit]:
    """Строки decode одного кадра → юниты в пикселях кадра (scale — во сколько
    раз кадр больше входа сети: 2 для 1080p)."""
    out = []
    for c, s, tp, x, y, _, hv in rows:
        if s <= 0:
            continue
        out.append(Unit(CLASSES[int(c)], TEAMS[int(tp > 0.5)], float(s), float(tp), float(hv),
                        float(x * scale), float(y * scale)))
    return out


def load_net(path: Path = DEFAULT_NET, device: str = "cuda") -> UnitNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UnitNet(tuple(ck["widths"]), tuple(ck["depths"]), ck["head"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


def prepare(rgb_u8: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Кадр 1080p (H, W, 3) uint8 на устройстве → вход (1, 3, 544, 960) 0..1.

    Уменьшение вдвое — среднее 2×2 (то же, что cv2.INTER_AREA при ровно
    половине, которым готовятся кадры обучения); `keep` — маска сцены
    (H, W) 0/1, закрытое HUD становится чёрным до уменьшения."""
    x = rgb_u8.permute(2, 0, 1)[None].float() / 255
    if keep is not None:
        x = x * keep
    x = F.avg_pool2d(x, 2)
    return F.pad(x, (0, IN_W - x.shape[-1], 0, IN_H - x.shape[-2]))


class UnitNetRunner:
    """Детектор для live: подготовка кадра, сеть и decode в одном CUDA graph
    на батч 1 (fp16). Кадр 1080p в видеопамяти → тензор decode."""

    def __init__(self, path: Path = DEFAULT_NET, device: str = "cuda",
                 keep: np.ndarray | None = None, thr: float = 0.3, fp16: bool = True,
                 graph: bool = True, size: tuple[int, int] = (1080, 1920)):
        self.net = load_net(path, device)
        self.device, self.thr = device, thr
        self.dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.net = self.net.to(memory_format=torch.channels_last)
        self.keep = (torch.from_numpy(keep.astype(np.float32)).to(device)[None, None]
                     if keep is not None else None)
        self.frame = torch.zeros(*size, 3, dtype=torch.uint8, device=device)
        self.graph = None
        if graph and device.startswith("cuda"):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self.out = self._run()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad():
                self.out = self._run()

    def _run(self) -> torch.Tensor:
        x = prepare(self.frame, self.keep).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype == torch.float16):
            h, t, o, hp = self.net(x)
        return decode(h, t, o, hp, self.thr)

    @torch.no_grad()
    def raw(self, frame: torch.Tensor) -> torch.Tensor:
        """Кадр (H, W, 3) uint8 на устройстве → строки decode (без копии на CPU)."""
        self.frame.copy_(frame)
        if self.graph is None:
            self.out = self._run()
        else:
            self.graph.replay()
        return self.out

    @torch.no_grad()
    def units(self, rgb: np.ndarray) -> list[Unit]:
        rows = self.raw(torch.from_numpy(rgb).to(self.device))[0].cpu().numpy()
        return to_units(rows, rgb.shape[1] / IN_W)
