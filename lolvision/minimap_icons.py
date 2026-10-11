"""Иконки чемпионов на миникарте: детектор центров и классификатор чемпиона.

Детектор (IconDet) — CenterNet в миниатюре: миникарта 256×256 → тепловые
карты центров иконок по командам (синие / красные) и субпиксельный сдвиг
на сетке шага 2 (128×128). Иконка ~20 px, иконки часто наложены друг на
друга, поэтому ищутся центры, а не рамки.

Классификатор (IconCls) — вырез 32×32 вокруг найденного центра → логиты по
всем чемпионам Data Dragon. Состав матча (10 чемпионов) накладывается уже
после сети (lolvision/minimap_champions.py), поэтому одна модель годится
для любого матча.

Обе сети учатся на синтетике (lolvision/minimap_synth.py) и дообучаются на
реальных миникартах с псевдо-метками (tools/train_minimap_icons.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lolvision.paths import RUNS

MM = 256
STRIDE = 2
CROP = 32                      # сторона выреза для классификатора
TEAMS = ("ORDER", "CHAOS")     # 0 — синие, 1 — красные (как в health_bars)
DEFAULT_RUN = RUNS / "minimap" / "icons"


def conv_bn(cin: int, cout: int, stride: int = 1, dil: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, dil, dilation=dil, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c: int, dil: int = 1):
        super().__init__()
        self.a = conv_bn(c, c, 1, dil)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, dil, dilation=dil, bias=False),
                               nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)), inplace=True)


class IconDet(nn.Module):
    """256×256 → (тепловые карты 2×128×128 в логитах, сдвиг 2×128×128)."""

    def __init__(self, w: int = 32):
        super().__init__()
        self.stem = conv_bn(3, w)                              # 256
        self.d1 = nn.Sequential(conv_bn(w, w, 2), Res(w))      # 128
        self.d2 = nn.Sequential(conv_bn(w, 2 * w, 2), Res(2 * w))          # 64
        self.d3 = nn.Sequential(conv_bn(2 * w, 3 * w, 2), Res(3 * w),
                                Res(3 * w, 2), Res(3 * w, 4))   # 32, поле зрения шире иконки
        self.u2 = conv_bn(3 * w + 2 * w, 2 * w)
        self.u1 = conv_bn(2 * w + w, w)
        self.heat = nn.Conv2d(w, 2, 1)
        self.off = nn.Conv2d(w, 2, 1)
        nn.init.constant_(self.heat.bias, -4.0)

    def forward(self, x):
        s1 = self.d1(self.stem(x))
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        y = self.u2(torch.cat([F.interpolate(s3, scale_factor=2, mode="nearest"), s2], 1))
        y = self.u1(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), s1], 1))
        return self.heat(y), self.off(y)


class IconCls(nn.Module):
    """Вырез 32×32 → (логиты чемпиона, логиты команды)."""

    def __init__(self, n_champ: int, w: int = 48):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(3, w), Res(w),
            conv_bn(w, 2 * w, 2), Res(2 * w),                  # 16
            conv_bn(2 * w, 4 * w, 2), Res(4 * w),              # 8
            conv_bn(4 * w, 4 * w, 2), Res(4 * w))              # 4
        self.drop = nn.Dropout(0.2)
        self.champ = nn.Linear(4 * w * 4, n_champ)              # учитывает расположение 2×2
        self.team = nn.Linear(4 * w * 4, 2)

    def embed(self, x):
        f = self.body(x)
        return F.adaptive_avg_pool2d(f, 2).flatten(1)

    def forward(self, x):
        e = self.drop(self.embed(x))
        return self.champ(e), self.team(e)


def to_tensor(imgs: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """uint8 N×H×W×3 RGB → float N×3×H×W, нормировано в [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(imgs)).to(device)
    return t.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def crops(img: np.ndarray, pts: list[tuple[float, float]], size: int = CROP) -> np.ndarray:
    """Вырезы size×size с центром в точках (сетка 256); снаружи — нули."""
    pad = size
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    out = np.zeros((len(pts), size, size, 3), np.uint8)
    for i, (x, y) in enumerate(pts):
        x0 = int(round(x)) - size // 2 + pad
        y0 = int(round(y)) - size // 2 + pad
        out[i] = p[y0:y0 + size, x0:x0 + size]
    return out


@dataclass
class Det:
    x: float                   # центр в сетке 256
    y: float
    team: int
    score: float


def decode(heat: torch.Tensor, off: torch.Tensor, thr: float = 0.3,
           k: int = 20) -> list[list[Det]]:
    """Пики тепловых карт (логиты) → детекции по каждому кадру пачки.

    Подавление — максимум 3×3 по обеим командам сразу: одна иконка не
    должна дать и синюю, и красную детекцию.
    """
    p = torch.sigmoid(heat)
    pm, team = p.max(1)                                   # N×H×W
    keep = (pm == F.max_pool2d(pm[:, None], 3, 1, 1)[:, 0]) & (pm >= thr)
    out = []
    for n in range(p.shape[0]):
        ys, xs = torch.nonzero(keep[n], as_tuple=True)
        sc = pm[n, ys, xs]
        order = torch.argsort(sc, descending=True)[:k]
        ys, xs, sc = ys[order], xs[order], sc[order]
        dx = off[n, 0, ys, xs]
        dy = off[n, 1, ys, xs]
        tm = team[n, ys, xs]
        dets = [Det(float((x + ox) * STRIDE), float((y + oy) * STRIDE), int(t), float(s))
                for x, y, ox, oy, t, s in zip(xs.tolist(), ys.tolist(), dx.tolist(),
                                              dy.tolist(), tm.tolist(), sc.tolist())]
        out.append(dets)
    return out


class IconModel:
    """Детектор + классификатор из каталога прогона (runs/minimap/icons)."""

    def __init__(self, run: Path = DEFAULT_RUN, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=False)
        self.champions: list[str] = ck["champions"]
        self.det = IconDet(ck.get("det_w", 32))
        self.det.load_state_dict(ck["det"])
        self.cls = IconCls(len(self.champions), ck.get("cls_w", 48))
        self.cls.load_state_dict(ck["cls"])
        self.det.to(self.device).eval()
        self.cls.to(self.device).eval()
        self.thr = float(ck.get("thr", 0.3))

    @torch.no_grad()
    def __call__(self, minis: np.ndarray) -> list[list[dict]]:
        """Пачка миникарт uint8 N×256×256×3 → по кадру список детекций:
        x, y (сетка 256), team, score, logp (лог-вероятности по чемпионам)."""
        heat, off = self.det(to_tensor(minis, self.device))
        dets = decode(heat, off, self.thr)
        flat, owner = [], []
        for n, ds in enumerate(dets):
            if ds:
                flat.append(crops(minis[n], [(d.x, d.y) for d in ds]))
                owner += [n] * len(ds)
        out: list[list[dict]] = [[] for _ in dets]
        if not flat:
            return out
        lc, lt = self.cls(to_tensor(np.concatenate(flat), self.device))
        logp = F.log_softmax(lc, 1).cpu().numpy()
        pt = F.softmax(lt, 1).cpu().numpy()
        i = 0
        for n, ds in enumerate(dets):
            for d in ds:
                out[n].append({"x": d.x, "y": d.y, "team": d.team, "score": d.score,
                               "team_p": pt[i].tolist(), "logp": logp[i]})
                i += 1
        return out
