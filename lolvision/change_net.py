"""Детектор изменений: предсказание следующего кадра, ошибка — «спайк» (#818).

Зачем. Тяжёлые модели не должны разбирать каждый кадр целиком: событие
(крип добит, способность, смерть) — это то, чего нельзя было предсказать по
прошлому. Модель по двум прошлым кадрам предсказывает следующий, и там,
где предсказание не сбылось, — спайк: туда и тогда смотреть подробно.

Что предсказывается. Признаки детектора юнитов (unit_net.UnitNet, выход
шеи: 64 канала, шаг 4 — уровень, на котором видны крипы) и картинка,
уменьшенная до той же сетки (изменения, которых детектор не знает:
всплывающее золото, эффекты). Детектор заморожен.

Камера. На виде игрока камера едет за чемпионом, и без поправки меняется
весь экран. Сдвиг между кадрами — фазовая корреляция серых кадров
(`camera_shift`), прошлые кадры сдвигаются к текущему до предсказания.

Крупное не глушит мелкое. Сеть, кроме кадра, предсказывает разброс
(логарифм дисперсии) в каждой клетке; потеря — гауссово правдоподобие.
Там, где всегда мельтешит (драка, эффекты, вода), разброс выучивается
большим, и спайк — ошибка, делённая на разброс (`surprise`): вспышка
способности поднимает порог у себя, а не у крипа рядом.

Вход сети — кадр 960×544 (вид 1080p, уменьшенный вдвое), HUD закрыт маской
вида игрока (unit_targets.unit_keep), как у детектора.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lolvision.unit_net import IN_H, IN_W, STRIDE, Res, UnitNet, conv_bn

FEAT_C = 64                     # каналов шеи детектора
GRID = (IN_H // STRIDE, IN_W // STRIDE)   # 136 × 240


def neck(net: UnitNet, x01: torch.Tensor) -> torch.Tensor:
    """Признаки шеи детектора (B, 64, 136, 240) — то, из чего головы
    детектора читают юнитов."""
    x = (x01 - net.mean) / net.std
    feats = []
    for s in net.stages:
        x = s(x)
        feats.append(x)
    feats = feats[1:]
    y = net.lat[-1](feats[-1])
    for i in range(len(feats) - 2, -1, -1):
        y = F.interpolate(y, size=feats[i].shape[-2:], mode="nearest") + net.lat[i](feats[i])
        y = net.smooth[i](y)
    return net.neck(y)


def camera_shift(a_gray: np.ndarray, b_gray: np.ndarray) -> tuple[float, float]:
    """Сдвиг картинки из `a` в `b` (пиксели этих серых кадров): фазовая
    корреляция. Вид игрока — почти чистый перенос: камера не вращается и не
    меняет масштаб."""
    (dx, dy), _ = cv2.phaseCorrelate(a_gray.astype(np.float32), b_gray.astype(np.float32))
    return float(dx), float(dy)


def shift_grid(x: torch.Tensor, dxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Сдвиг карт (B, C, H, W) на `dxy` (B, 2) клеток; вернуть и маску
    клеток, пришедших из кадра (за краем — 0)."""
    b, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    sx = xs[None] - dxy[:, 0, None, None]
    sy = ys[None] - dxy[:, 1, None, None]
    grid = torch.stack((2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1), -1)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    ok = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)).to(x.dtype)[:, None]
    return out, ok


class ChangeNet(nn.Module):
    """Два прошлых кадра (признаки + картинка, уже сдвинутые к текущему) →
    текущий кадр как поправка к последнему и логарифм разброса."""

    def __init__(self, width: int = 96, depth: int = 2):
        super().__init__()
        cin = 3 * FEAT_C + 2 * 3            # f1, f2, f2-f1, rgb1, rgb2
        self.inp = conv_bn(cin, width, 1)
        self.e1 = nn.Sequential(*[Res(width) for _ in range(depth)])
        self.down = conv_bn(width, 2 * width, 3, 2)
        self.e2 = nn.Sequential(*[Res(2 * width) for _ in range(depth + 1)])
        self.down2 = conv_bn(2 * width, 2 * width, 3, 2)
        self.e3 = nn.Sequential(*[Res(2 * width) for _ in range(depth + 1)])
        self.up2 = conv_bn(4 * width, 2 * width, 1)
        self.up = conv_bn(3 * width, width, 1)
        self.d1 = nn.Sequential(*[Res(width) for _ in range(depth)])
        self.out = nn.Conv2d(width, FEAT_C + 3 + 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f1, f2, r1, r2):
        x = self.e1(self.inp(torch.cat((f1, f2, f2 - f1, r1, r2), 1)))
        y = self.e2(self.down(x))
        z = self.e3(self.down2(y))
        y = self.up2(torch.cat((y, F.interpolate(z, size=y.shape[-2:], mode="nearest")), 1))
        x = self.d1(self.up(torch.cat((x, F.interpolate(y, size=x.shape[-2:], mode="nearest")), 1)))
        o = self.out(x)
        pf = f2 + o[:, :FEAT_C]
        pr = r2 + o[:, FEAT_C:FEAT_C + 3]
        return pf, pr, o[:, -2:-1], o[:, -1:]        # признаки, картинка, logvar f, logvar r


def errors(pf, pr, f3, r3):
    """Средний квадрат ошибки по каналам: (B, 1, H, W) для признаков и картинки."""
    return ((pf - f3) ** 2).mean(1, keepdim=True), ((pr - r3) ** 2).mean(1, keepdim=True)


def nll(err: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Гауссово правдоподобие на клетку (константы опущены)."""
    return err * torch.exp(-logvar) + logvar


def surprise(ef, er, lf, lr) -> torch.Tensor:
    """Спайк клетки: ошибки, делённые на выученный разброс, сумма двух частей."""
    return ef * torch.exp(-lf) + er * torch.exp(-lr)
