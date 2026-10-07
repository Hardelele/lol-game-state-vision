"""Компактная CNN для классификации сцены по одному кадру.

Разделена на две части осознанно:

* `SceneEncoder` — кадр → вектор признаков. Не зависит от задачи и от того,
  сколько кадров обрабатывается. Именно он переиспользуется, когда поверх
  появится модель, следящая за изменениями: последовательность эмбеддингов
  от того же энкодера подаётся во временной модуль (GRU/внимание/разности),
  а энкодер можно заморозить или дообучать.
* `SceneClassifier` — текущая задача: один кадр → логиты классов.

Поэтому в чекпоинт веса энкодера и головы пишутся раздельно.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int, stride: int, groups: int = 4) -> nn.Sequential:
    """Conv → GroupNorm → ReLU.

    GroupNorm, а не BatchNorm: батчи здесь маленькие (датасет — десятки кадров),
    и статистики батча были бы шумными. GroupNorm от размера батча не зависит,
    поэтому инференс по одному кадру совпадает с обучением.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
    )


class SceneEncoder(nn.Module):
    """Кадр (3, H, W) → эмбеддинг (embed_dim,). H и W произвольные."""

    def __init__(self, embed_dim: int = 96, width: tuple[int, ...] = (16, 32, 64, 96)):
        super().__init__()
        if width[-1] != embed_dim:
            raise ValueError("последний слой должен давать embed_dim каналов")
        self.embed_dim = embed_dim
        chans = (3,) + tuple(width)
        self.stages = nn.Sequential(
            *[conv_block(chans[i], chans[i + 1], stride=2) for i in range(len(width))]
        )
        # Глобальный пулинг, а не Flatten: размер входа может меняться,
        # а эмбеддинг остаётся той же размерности.
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.stages(x)).flatten(1)


class SceneClassifier(nn.Module):
    """Энкодер + линейная голова на num_classes классов."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 96, dropout: float = 0.3):
        super().__init__()
        self.encoder = SceneEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Эмбеддинги без головы — вход для будущей временной модели."""
        self.eval()
        return self.encoder(x)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    model = SceneClassifier()
    x = torch.zeros(2, 3, 112, 256)
    print("логиты", tuple(model(x).shape), "эмбеддинг", tuple(model.embed(x).shape))
    print(f"параметров: энкодер {count_params(model.encoder):,}, "
          f"голова {count_params(model.head):,}, всего {count_params(model):,}")
    # Эмбеддинг не зависит от размера входа — проверка контракта.
    print("другой размер входа:", tuple(model.embed(torch.zeros(1, 3, 96, 320)).shape))
