"""Модель «кадр → где камера на карте», с выходом в виде тепловой карты.

Почему тепловая карта, а не два числа. Верхняя и нижняя линии Ущелья
Призывателей почти зеркальны: карта симметрична поворотом на 180°, и по
одному кадру без миникарты место иногда определяется неоднозначно. Прямая
регрессия в такой ситуации выдаёт середину между двумя вариантами — точку,
где камера заведомо не была. Тепловая карта честно показывает два пика, и
эту неоднозначность видно, а не замазано.

Разделение то же, что в классификаторе сцены: энкодер кадр → признаки
отдельно от головы. Временной модуль, который следит за изменениями,
ставится поверх последовательности признаков от того же энкодера.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

GRID = 32  # сторона тепловой карты: шаг 1/32 карты ≈ 460 игровых единиц


def block(cin: int, cout: int, stride: int) -> nn.Sequential:
    # GroupNorm, а не BatchNorm: инференс по одному кадру должен совпадать
    # с обучением, а статистики батча на видео сильно скоррелированы.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class CoordEncoder(nn.Module):
    """Кадр (3,H,W) → карта признаков. Пулинга в конце нет: положение важно."""

    def __init__(self, width=(32, 64, 128, 256)):
        super().__init__()
        chans = (3,) + tuple(width)
        self.stages = nn.Sequential(
            *[block(chans[i], chans[i + 1], 2) for i in range(len(width))])
        self.out_ch = width[-1]

    def forward(self, x):
        return self.stages(x)


class CoordNet(nn.Module):
    """Энкодер + голова, выдающая логиты тепловой карты GRID x GRID."""

    def __init__(self, grid: int = GRID, pool: tuple[int, int] = (5, 9),
                 dropout: float = 0.2):
        super().__init__()
        self.grid = grid
        self.encoder = CoordEncoder()
        # Пул до небольшой сетки, а не до вектора: иначе теряется, в какой
        # части кадра что находится, а это и есть подсказка о месте.
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_ch * pool[0] * pool[1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, grid * grid),
        )

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)))   # (B, grid*grid)

    @torch.no_grad()
    def embed(self, x):
        """Признаки кадра для будущей временной модели."""
        self.eval()
        return self.pool(self.encoder(x)).flatten(1)


def grid_centers(grid: int, device) -> torch.Tensor:
    """Координаты центров ячеек в долях карты: (grid*grid, 2) как (cx, cy)."""
    c = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)


def soft_target(cx, cy, grid: int, sigma: float = 1.0) -> torch.Tensor:
    """Гауссова цель вокруг истинной точки.

    Мягкая, а не one-hot: координата с миникарты сама известна примерно
    (порядка 0.012 карты), и требовать от модели попадания в конкретную
    ячейку значило бы учить её на собственном шуме разметки.
    """
    centers = grid_centers(grid, cx.device)                 # (G²,2)
    pts = torch.stack([cx, cy], dim=1)[:, None, :]          # (B,1,2)
    d2 = ((centers[None] - pts) ** 2).sum(-1)               # (B,G²)
    t = torch.exp(-d2 / (2 * (sigma / grid) ** 2))
    return t / t.sum(1, keepdim=True).clamp(min=1e-9)


def expected_point(logits: torch.Tensor, grid: int) -> torch.Tensor:
    """Ожидаемая точка по распределению — для метрики средней ошибки."""
    p = F.softmax(logits, dim=1)
    return p @ grid_centers(grid, logits.device)


def peak_point(logits: torch.Tensor, grid: int) -> torch.Tensor:
    """Самая вероятная ячейка — устойчивее среднего, когда пиков два."""
    return grid_centers(grid, logits.device)[logits.argmax(1)]


def spread(logits: torch.Tensor, grid: int) -> torch.Tensor:
    """Разброс распределения: большой — модель сомневается или видит два места."""
    p = F.softmax(logits, dim=1)
    c = grid_centers(grid, logits.device)
    mean = p @ c
    var = (p[:, :, None] * (c[None] - mean[:, None, :]) ** 2).sum(1).sum(1)
    return var.sqrt()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = CoordNet()
    x = torch.zeros(2, 3, 200, 384)
    lo = net(x)
    print("логиты", tuple(lo.shape), "признаки", tuple(net.embed(x).shape))
    print(f"параметров: энкодер {count_params(net.encoder):,}, "
          f"голова {count_params(net.head):,}, всего {count_params(net):,}")
    cx = torch.tensor([0.2, 0.8]); cy = torch.tensor([0.3, 0.7])
    t = soft_target(cx, cy, GRID)
    print("цель: сумма", float(t.sum(1)[0]), "пик в", 
          [round(v, 3) for v in grid_centers(GRID, t.device)[t.argmax(1)][0].tolist()])
