"""Патчевая модель «кусок сцены → где это на карте» и геометрия мира для неё.

Зачем. Модель координат CoordNet (coord_model.py) смотрит на кадр целиком:
пул до сетки 5×9 → flatten → Linear. Голова знает, в какой клетке экрана
лежит каждый признак, и опыт с подменой HUD (tools/probe_hud.py) показал,
что она на это и опирается: стоит открыть временные маски, закрасить три
лишних места или обрезать кадр на 8% шире — медианная ошибка растёт с 89
до 300-950 игровых единиц. Такая модель привязана к маске, раскладке и
масштабу, а не к карте.

Как. Три решения, каждое снимает одну из этих привязок.

* Нормировка через мир. Камера смотрит на плоскую землю, связь «пиксель
  хранимой сцены → смещение от центра камеры» — гомография из
  dataset/layouts/projection.json. Вход переводится в вид сверху с
  постоянным числом игровых единиц на пиксель (UNITS_PER_PX). После этого
  размер исходного кадра, его разрешение и пропорции ничего не значат:
  одна и та же трава всегда занимает одинаковое число пикселей, а сдвиг
  на входе — это ровно сдвиг на карте.
* Полностью свёрточный энкодер без позиционной головы. Каждый выходной
  патч (шаг 16 пикселей ≈ 100 ед.) получает свой эмбеддинг и сам по нему
  говорит, в какой точке карты он лежит. Положение патча на экране в
  ответ не входит, поэтому выучить «клетку экрана» нечем.
* Камера кадра — согласие патчей. Каждый патч голосует за положение
  камеры: своя точка минус своё смещение от центра камеры (оно известно
  из геометрии). Голоса сводятся устойчиво (консенсус по радиусу и
  взвешенная медиана), так что закрытые, обрезанные и непохожие куски
  просто выпадают, а не тянут ответ.

Голова патча выдаёт распределение по сетке карты, а не два числа: одиночный
кусок травы или реки неоднозначен гораздо сильнее, чем целый кадр, и
регрессия усреднила бы зеркальные варианты в точку, где патча заведомо нет
(тот же довод, что в coord_model.py, только сильнее). Точная точка внутри
пика берётся локальным мягким argmax 5×5. Отдельный выход — «это сцена»:
чёрные поля, UI и пустота за краем кадра должны получать малый вес.

Вопрос «это вообще игра?» решает необязательная голова кадра FrameHead
поверх замороженной модели (обучает tools/train_not_game.py): по среднему
эмбеддингу патчей и тому, насколько патчи уверены в своём месте и согласны
между собой, она отличает игру от вебкамеры, клиента, заставок и других
игр. Её веса и порог лежат в чекпоинте отдельным полем frame_head; если его
нет, модель работает как раньше.

Пример (проверка формы и геометрии):
    python tools/patch_model.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from paths import ROOT

MAP_UNITS = 14800
UNITS_PER_PX = 6.0     # вид сверху: столько игровых единиц на пиксель входа
MAP_GRID = 64          # сетка карты у головы патча: ячейка ≈ 231 ед.
PROJECTION = ROOT / "dataset" / "layouts" / "projection.json"


# ---------------------------------------------------------------- геометрия

def load_projection(path: Path = PROJECTION) -> tuple[np.ndarray, tuple[int, int]]:
    """Гомография «пиксель хранимой сцены → смещение от центра камеры (доли карты)»."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(d["homography"], dtype=np.float64), tuple(d["scene_size"])


def apply_h(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = np.c_[pts, np.ones(len(pts))] @ m.T
    return v[:, :2] / v[:, 2:3]


def scene_from_box(box, out_size, crop, store_size) -> np.ndarray:
    """Аффинное A: пиксель картинки → пиксель хранимой сцены.

    `box` — где картинка лежала в полном кадре (x0, y0, x1, y1, в пикселях
    полного кадра), `out_size` — до какого размера её привели; `crop` —
    плотная обрезка той же раскладки в тех же пикселях, `store_size` —
    размер хранимой сцены. Так описывается любой вход: другое разрешение,
    другая обрезка, кусок кадра.
    """
    bx0, by0, bx1, by1 = box
    cx0, cy0, cx1, cy1 = crop
    sx = (bx1 - bx0) / out_size[0] * store_size[0] / (cx1 - cx0)
    sy = (by1 - by0) / out_size[1] * store_size[1] / (cy1 - cy0)
    tx = (bx0 - cx0) * store_size[0] / (cx1 - cx0)
    ty = (by0 - cy0) * store_size[1] / (cy1 - cy0)
    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)


def to_world(h0: np.ndarray, a: np.ndarray, units_px: float) -> np.ndarray:
    """Пиксель входа → пиксель вида сверху, где (0, 0) — центр камеры."""
    k = MAP_UNITS / units_px
    return np.diag([k, k, 1.0]) @ h0 @ a


def footprint(m: np.ndarray, size) -> np.ndarray:
    """Углы входа размером `size` в координатах вида сверху: (4, 2)."""
    w, h = size
    return apply_h(m, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))


def rectify(img: np.ndarray, a: np.ndarray, h0: np.ndarray,
            units_px: float = UNITS_PER_PX, stride: int = 16,
            canvas: tuple[int, int] | None = None, q0=None):
    """Перевести картинку (H, W, 3) uint8 в вид сверху.

    Возвращает холст, маску «пиксель пришёл из картинки» и q0 — где на
    холсте центр камеры (может лежать за его краем: у куска кадра почти
    всегда так). Размер холста по умолчанию — габарит следа картинки,
    округлённый вверх до шага сетки патчей.
    """
    m = to_world(h0, a, units_px)
    fp = footprint(m, (img.shape[1], img.shape[0]))
    lo, hi = fp.min(0), fp.max(0)
    if canvas is None:
        canvas = (int(math.ceil((hi[0] - lo[0]) / stride) * stride),
                  int(math.ceil((hi[1] - lo[1]) / stride) * stride))
    if q0 is None:
        q0 = -lo + (np.array(canvas) - (hi - lo)) / 2
    t = np.array([[1, 0, q0[0]], [0, 1, q0[1]], [0, 0, 1]], dtype=np.float64) @ m
    out = cv2.warpPerspective(img, t, canvas, flags=cv2.INTER_AREA
                              if units_px > 4 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.full(img.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(ones, t, canvas, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, valid, np.asarray(q0, dtype=np.float64)


def cell_offsets(h: int, w: int, stride: float, q0: torch.Tensor,
                 units: torch.Tensor) -> torch.Tensor:
    """Смещение центра каждого патча от центра камеры, доли карты.

    q0: (B, 2) — центр камеры на холсте, units: (B,) — ед./пиксель.
    Ответ (B, h, w, 2) в порядке (x, y).
    """
    dev = q0.device
    ys = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    q = torch.stack([gx, gy], -1)[None]                       # (1,h,w,2)
    return (q - q0[:, None, None, :]) * (units[:, None, None, None] / MAP_UNITS)


# ---------------------------------------------------------------- модель

def block(cin: int, cout: int, stride: int) -> nn.Sequential:
    # GroupNorm, как в coord_model: инференс по одному кадру = обучение.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class Dilated(nn.Module):
    """Остаточный блок с разрежённой свёрткой: расширяет обзор патча без
    потери разрешения сетки патчей."""

    def __init__(self, ch: int, d: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, d, dilation=d, bias=False),
            nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, d, dilation=d, bias=False),
            nn.GroupNorm(8, ch))

    def forward(self, x):
        return F.relu(x + self.f(x))


class CNNEncoder(nn.Module):
    """Вид сверху (3, H, W) в [-1, 1] → признаки (C, H/16, W/16).

    Четыре ступени с шагом 2 и три остаточных блока с разрежением 1, 2, 4.
    Обзор одного патча — около 500 пикселей входа, то есть порядка 3000
    ед.: этого хватает, чтобы узнать место по стенам и кустам вокруг, но
    ничего в архитектуре не говорит патчу, где он на экране.
    """

    stride = 16

    def __init__(self, width=(32, 64, 128, 192), dil=(1, 2, 4)):
        super().__init__()
        chans = (3,) + tuple(width)
        self.stages = nn.Sequential(
            *[block(chans[i], chans[i + 1], 2) for i in range(len(width))],
            *[Dilated(width[-1], d) for d in dil])
        self.out_ch = width[-1]

    def norm(self, x01: torch.Tensor) -> torch.Tensor:
        return x01 * 2 - 1

    def forward(self, x01):
        return self.stages(self.norm(x01))


class DinoEncoder(nn.Module):
    """Замороженный DINOv2: патч-токены последнего слоя как карта признаков.

    Базовая точка: те же голова и обучение, но признаки чужие и не
    обучаются. В ViT есть позиционные эмбеддинги, так что положение на
    экране здесь в признаки как раз подмешано — это часть сравнения.
    """

    stride = 14
    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)

    def __init__(self, arch: str = "dinov2_vits14"):
        super().__init__()
        repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
        if repo.exists():
            self.vit = torch.hub.load(str(repo), arch, source="local")
        else:
            self.vit = torch.hub.load("facebookresearch/dinov2", arch)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.eval()
        self.arch = arch
        self.out_ch = self.vit.embed_dim
        self.register_buffer("mean", torch.tensor(self._mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._std).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()            # заморожен всегда, и dropout тоже
        return self

    def forward(self, x01):
        b, _, h, w = x01.shape
        x = (x01 - self.mean) / self.std
        with torch.no_grad():
            t = self.vit.forward_features(x)["x_norm_patchtokens"]
        return t.transpose(1, 2).reshape(b, -1, h // 14, w // 14).float()


class PatchHead(nn.Module):
    """Признаки патча → эмбеддинг (L2) → распределение по карте + «сцена».

    Всё — свёртки 1×1, то есть одинаковая функция для каждого патча.
    """

    def __init__(self, cin: int, dim: int = 128, hidden: int = 256,
                 grid: int = MAP_GRID):
        super().__init__()
        self.grid = grid
        self.embed = nn.Conv2d(cin, dim, 1)
        self.loc = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, grid * grid, 1))
        self.scene = nn.Sequential(nn.Conv2d(dim, 64, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, 1))
        # Эмбеддинг на единичной сфере; масштаб вернёт обучаемая температура.
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, f):
        e, z, sc = self.features(f)
        return e, self.loc(z), sc

    def features(self, f):
        """Эмбеддинг, его масштабированная копия и логит «сцена» — без
        логитов карты: при обучении они нужны только для части патчей."""
        e = F.normalize(self.embed(f), dim=1)
        z = e * self.scale
        return e, z, self.scene(z)[:, 0]

    def loc_of(self, z: torch.Tensor) -> torch.Tensor:
        """Логиты карты для набора патчей (P, D) → (P, G²)."""
        # Та же функция, что свёртки 1×1 головы, но как линейные слои:
        # свёртка по тензору (P, D, 1, 1) на cuDNN выбирала алгоритм,
        # который шёл в десятки раз медленнее.
        c1, c2 = self.loc[0], self.loc[2]
        h = F.relu(F.linear(z, c1.weight[:, :, 0, 0], c1.bias))
        return F.linear(h, c2.weight[:, :, 0, 0], c2.bias)


FRAME_SCALARS = ("scene_mean", "scene_frac", "mass_mean", "weight_mean", "agree")


def frame_features(emb: torch.Tensor, scene: torch.Tensor, mass: torch.Tensor,
                   vf: torch.Tensor, agree: torch.Tensor,
                   min_valid: float = 0.5) -> torch.Tensor:
    """Признаки кадра для «игра / не игра» из ответов патчей: (B, D + 5).

    Средний эмбеддинг патчей внутри следа картинки и пять чисел: средняя
    P(сцена), доля патчей с P(сцена) > 0.5, средняя доля вероятности в окне
    пика (насколько патч уверен в своём месте), средний вес голоса и доля
    веса в согласии. Средний эмбеддинг — это то же, что усреднить по
    патчам линейную оценку «патч из игры», то есть доля патчей-игры с
    обучаемыми весами; числа добавлены потому, что именно по уверенности и
    согласию видно, что патчи «не узнают» место, даже когда голова «сцена»
    обманута (на панели во весь кадр P(сцена) = 0.76).
    """
    m = (vf >= min_valid).float()
    n = m.sum((1, 2)).clamp(min=1)
    e = (emb.float() * m[:, None]).sum((2, 3)) / n[:, None]
    sc = scene.float()
    ms = mass.float()
    sc_mean = (sc * m).sum((1, 2)) / n
    sc_frac = ((sc > 0.5).float() * m).sum((1, 2)) / n
    ms_mean = (ms * m).sum((1, 2)) / n
    w_mean = (sc * ms * m).sum((1, 2)) / n
    return torch.cat([e, torch.stack([sc_mean, sc_frac, ms_mean, w_mean,
                                      agree.float()], 1)], 1)


class FrameHead(nn.Module):
    """Признаки кадра (frame_features) → логит «это игровая сцена League».

    Маленькая голова поверх замороженной патчевой модели: энкодер и голова
    патча не меняются, поэтому координаты остаются ровно такими же. Порог
    хранится рядом, в чекпоинте (поле frame_head), и подбирается при
    обучении (tools/train_not_game.py).
    """

    def __init__(self, cin: int, hidden: int = 64):
        super().__init__()
        self.cin, self.hidden = cin, hidden
        self.register_buffer("mu", torch.zeros(cin))
        self.register_buffer("sd", torch.ones(cin))
        self.net = nn.Sequential(nn.Linear(cin, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net((feat.float() - self.mu) / self.sd)[:, 0]


class PatchNet(nn.Module):
    def __init__(self, encoder: str = "cnn", dim: int = 128):
        super().__init__()
        self.kind = encoder
        self.encoder = CNNEncoder() if encoder == "cnn" else DinoEncoder(encoder)
        self.stride = self.encoder.stride
        self.head = PatchHead(self.encoder.out_ch, dim)
        self.grid = self.head.grid
        # Голова «игра / не игра» (FrameHead) — необязательная: у старых
        # чекпоинтов её нет, и тогда модель просто не отвечает на этот вопрос.
        self.frame: FrameHead | None = None
        self.frame_threshold = 0.5

    def forward(self, x01):
        """x01: (B, 3, H, W) в [0, 1]. Ответ: эмбеддинги (B, D, h, w),
        логиты карты (B, G², h, w), логит «сцена» (B, h, w)."""
        return self.head(self.encoder(x01))

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def state(self) -> dict:
        # Веса FrameHead — отдельным полем, а не в "state": чекпоинт с
        # головой кадра грузится и старым load_patchnet (поле он не читает).
        sd = {k: v for k, v in self.state_dict().items()
              if not k.startswith("encoder.vit.") and not k.startswith("frame.")}
        out = {"kind": "patch", "encoder": self.kind, "state": sd,
               "units_per_px": UNITS_PER_PX, "grid": self.grid,
               "dim": self.head.embed.out_channels}
        if self.frame is not None:
            out["frame_head"] = {
                "cin": self.frame.cin, "hidden": self.frame.hidden,
                "state": self.frame.state_dict(),
                "threshold": float(self.frame_threshold),
                "features": list(FRAME_SCALARS)}
        return out


def load_patchnet(path: Path, dev: str) -> PatchNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PatchNet(ck["encoder"], ck.get("dim", 128))
    missing, unexpected = m.load_state_dict(ck["state"], strict=False)
    bad = [k for k in missing if not k.startswith("encoder.vit.")]
    if bad or unexpected:
        raise RuntimeError(f"чекпоинт не подходит: {bad[:5]} {unexpected[:5]}")
    fh = ck.get("frame_head")
    if fh:
        m.frame = FrameHead(fh["cin"], fh.get("hidden", 64))
        m.frame.load_state_dict(fh["state"])
        m.frame_threshold = float(fh.get("threshold", 0.5))
    return m.to(dev).eval()


# ---------------------------------------------------------------- цели и ответ

def map_centers(grid: int, dev) -> torch.Tensor:
    return (torch.arange(grid, device=dev, dtype=torch.float32) + 0.5) / grid


def soft_target_sep(pts: torch.Tensor, grid: int, sigma: float = 1.0) -> torch.Tensor:
    """Гауссова цель на сетке карты (P, G²), как soft_target в coord_model,
    но раздельная по осям — так она дёшево строится для тысяч патчей."""
    c = map_centers(grid, pts.device)
    s = sigma / grid
    tx = torch.exp(-(pts[:, 0:1] - c[None]) ** 2 / (2 * s * s))
    ty = torch.exp(-(pts[:, 1:2] - c[None]) ** 2 / (2 * s * s))
    t = (ty[:, :, None] * tx[:, None, :]).reshape(len(pts), -1)
    return t / t.sum(1, keepdim=True).clamp(min=1e-12)


def decode_points(logits: torch.Tensor, grid: int, win: int = 5):
    """Логиты (P, G²) → точка (P, 2) и уверенность (P,).

    Точка — мягкий argmax в окне win×win вокруг пика: на сетке 64 шаг
    231 ед., а нужна точность лучше ячейки (окно 3×3 смещает ответ к центру
    ячейки на ~30 ед., 5×5 — на единицы). Уверенность — доля вероятности в
    этом окне: у неоднозначного патча она размазана по нескольким пикам.
    """
    p = F.softmax(logits.float(), 1)
    k = p.argmax(1)
    iy, ix = k // grid, k % grid
    r = win // 2
    n = win * win
    d = torch.arange(-r, r + 1, device=logits.device)
    yy = (iy[:, None, None] + d[None, :, None]).clamp(0, grid - 1).expand(-1, win, win)
    xx = (ix[:, None, None] + d[None, None, :]).clamp(0, grid - 1).expand(-1, win, win)
    idx = (yy * grid + xx).reshape(len(k), n)
    pw = p.gather(1, idx)
    mass = pw.sum(1)
    w = pw / mass[:, None].clamp(min=1e-12)
    pts = torch.stack([(xx.reshape(len(k), n).float() + 0.5) / grid,
                       (yy.reshape(len(k), n).float() + 0.5) / grid], -1)
    return (w[:, :, None] * pts).sum(1), mass


def wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def aggregate(cand: np.ndarray, w: np.ndarray, radius: float = 0.025):
    """Положение камеры по голосам патчей: (2,) и доля веса в согласии.

    Сначала ищется голос, у которого больше всего веса в радиусе `radius`
    (~370 ед.) — так зеркальные и случайные ответы не смешиваются с
    правильными; затем по этим союзникам берётся взвешенная медиана по
    каждой оси. Это RANSAC с полным перебором гипотез: патчей сотни, и
    полный перебор дешевле случайного.
    """
    if len(cand) == 0 or w.sum() <= 0:
        return np.array([np.nan, np.nan]), 0.0
    d = np.linalg.norm(cand[:, None] - cand[None], axis=-1) < radius
    sup = d.astype(np.float64) @ w
    best = int(np.argmax(sup))
    inl = d[best] & (w > 0)
    c = np.array([wmedian(cand[inl, 0], w[inl]), wmedian(cand[inl, 1], w[inl])])
    return c, float(sup[best] / w.sum())


def aggregate_batch(cand: torch.Tensor, w: torch.Tensor, radius: float = 0.025):
    """То же, что aggregate, сразу для батча и на том устройстве, где голоса.

    cand (B, N, 2), w (B, N) → камеры (B, 2) и доля согласия (B,). На
    видеокарте перебор N² гипотез для сотен патчей стоит доли миллисекунды,
    а numpy на процессоре — 5-8 мс на кадр (замер tools/bench_models.py).
    """
    b, n, _ = cand.shape
    cand, w = cand.float(), w.float()
    d = (torch.cdist(cand, cand) < radius).float()               # (B,N,N)
    sup = (d @ w[..., None])[..., 0]                             # (B,N)
    best = sup.argmax(1)
    inl = d[torch.arange(b, device=cand.device), best] * (w > 0)
    wi = w * inl
    out = []
    for ax in range(2):
        v, o = cand[..., ax].sort(1)
        c = wi.gather(1, o).cumsum(1)
        k = (c < c[:, -1:] / 2).sum(1, keepdim=True).clamp(max=n - 1)
        out.append(v.gather(1, k)[:, 0])
    cams = torch.stack(out, 1)
    tot = w.sum(1)
    agree = sup.gather(1, best[:, None])[:, 0] / tot.clamp(min=1e-12)
    cams[tot <= 0] = float("nan")
    return cams, agree


@torch.no_grad()
def predict_canvases(model: PatchNet, x01: torch.Tensor, valid: torch.Tensor,
                     q0: torch.Tensor, units: float = UNITS_PER_PX,
                     radius: float = 0.025, min_valid: float = 0.5):
    """Холсты вида сверху → камера по каждому и сведения о патчах.

    valid — маска следа картинки на холсте (B, H, W) в [0, 1]: патчи за
    краем исходной картинки в голосовании не участвуют (это геометрия, а не
    подсказка модели). Ответ: камеры (B, 2), доля согласия (B,), и по
    патчам — точки, смещения, веса, доля следа.
    """
    cams, agree, info = canvases_post(model, *model(x01), valid, q0, units,
                                      radius, min_valid)
    return cams.cpu().numpy(), agree.cpu().numpy(), info


def canvases_post(model: PatchNet, emb: torch.Tensor, lo: torch.Tensor,
                  sc: torch.Tensor, valid: torch.Tensor, q0: torch.Tensor,
                  units: float = UNITS_PER_PX, radius: float = 0.025,
                  min_valid: float = 0.5):
    """Выходы сети → камеры, согласие и сведения о патчах, всё на устройстве.

    Вынесено из predict_canvases, чтобы live мог записать этот же код в
    CUDA graph (там нельзя выгружать на процессор посреди пути). Без
    синхронизаций с процессором и с постоянными формами.
    """
    b, g2, h, w = lo.shape
    s = model.stride
    vf = F.avg_pool2d(valid[:, None].float(), s, s)[:, 0, :h, :w]
    pts, mass = decode_points(lo.permute(0, 2, 3, 1).reshape(-1, g2), model.grid)
    pts = pts.reshape(b, h, w, 2)
    mass = mass.reshape(b, h, w)
    off = cell_offsets(h, w, s, q0, torch.full((b,), units, device=q0.device))
    scene = torch.sigmoid(sc.float())
    wt = scene * mass * (vf >= min_valid)
    cams, agree = aggregate_batch((pts - off).reshape(b, -1, 2), wt.reshape(b, -1), radius)
    info = {"pts": pts, "off": off, "w": wt, "valid": vf, "scene": scene, "emb": emb,
            "frame_feat": frame_features(emb, scene, mass, vf, agree, min_valid)}
    if model.frame is not None:
        # P(игра) кадра; камера считается всегда, решать, верить ли ей, —
        # вызывающему (порог в model.frame_threshold).
        # Голова крошечная — считаем её в fp32 и вне autocast.
        with torch.autocast(q0.device.type, enabled=False):
            info["game"] = torch.sigmoid(model.frame(info["frame_feat"].float()))
    return cams, agree, info


@torch.no_grad()
def infer_images(model: PatchNet, imgs, mats, opens, h0: np.ndarray, dev: str,
                 batch: int = 16, units: float = UNITS_PER_PX):
    """Картинки любого размера → камера и ответы патчей.

    imgs — список (H, W, 3) uint8 RGB; mats — для каждой аффинное A
    «пиксель картинки → пиксель хранимой сцены» (см. scene_from_box);
    opens — маска (H, W) «здесь видна сцена» или None. Маска модели не
    показывается: по ней только отбираются патчи для метрики «точность
    отдельного патча». Ответ: камеры (N, 2), доля согласия (N,) и по
    каждой картинке словарь с открытыми патчами (точки, смещения, веса),
    признаками кадра frame_feat и P(игра) game (None без FrameHead).
    """
    cams, agree, patches = [], [], []
    s = model.stride
    for i in range(0, len(imgs), batch):
        chunk = []
        for img, a, op in zip(imgs[i:i + batch], mats[i:i + batch], opens[i:i + batch]):
            cv, val, q0 = rectify(img, a, h0, units, s)
            if op is None:
                opw = val
            else:
                m = to_world(h0, a, units)
                t = np.array([[1, 0, q0[0]], [0, 1, q0[1]], [0, 0, 1]]) @ m
                opw = cv2.warpPerspective(op.astype(np.uint8) * 255, t,
                                          (cv.shape[1], cv.shape[0]),
                                          flags=cv2.INTER_LINEAR)
            chunk.append((cv, val, opw, q0))
        hh = max(c[0].shape[0] for c in chunk)
        ww = max(c[0].shape[1] for c in chunk)
        x = np.zeros((len(chunk), hh, ww, 3), np.uint8)
        v = np.zeros((len(chunk), hh, ww), np.uint8)
        o = np.zeros((len(chunk), hh, ww), np.uint8)
        for j, (cv, val, opw, _) in enumerate(chunk):
            x[j, :cv.shape[0], :cv.shape[1]] = cv
            v[j, :cv.shape[0], :cv.shape[1]] = val
            o[j, :cv.shape[0], :cv.shape[1]] = opw
        xt = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float().div_(255)
        vt = torch.from_numpy(v).to(dev).float().div_(255)
        q0t = torch.tensor(np.array([c[3] for c in chunk]), device=dev,
                           dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            c, a, info = predict_canvases(model, xt, vt, q0t, units)
        cams.append(c)
        agree.append(a)
        h, w = info["w"].shape[1:]
        of = F.avg_pool2d(torch.from_numpy(o).to(dev)[:, None].float() / 255,
                          s, s)[:, 0, :h, :w]
        feat = info["frame_feat"].cpu().numpy()
        game = info["game"].cpu().numpy() if "game" in info else None
        for j in range(len(chunk)):
            keep = (of[j] >= 0.75).reshape(-1)
            patches.append({
                "pts": info["pts"][j].reshape(-1, 2)[keep].cpu().numpy(),
                "off": info["off"][j].reshape(-1, 2)[keep].cpu().numpy(),
                "w": info["w"][j].reshape(-1)[keep].cpu().numpy(),
                "frame_feat": feat[j],
                "game": None if game is None else float(game[j])})
    return np.concatenate(cams), np.concatenate(agree), patches


def count_params(m: nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in m.parameters()
               if p.requires_grad or not trainable)


if __name__ == "__main__":
    h0, store = load_projection()
    a = np.eye(3)
    img = np.zeros((store[1], store[0], 3), np.uint8)
    cv, val, q0 = rectify(img, a, h0)
    print(f"хранимая сцена {store} → вид сверху {cv.shape[1]}x{cv.shape[0]} "
          f"при {UNITS_PER_PX} ед./px; центр камеры на холсте {q0.round(1)}")
    fp = footprint(to_world(h0, a, UNITS_PER_PX), store)
    print("след кадра, ед.:", (fp * UNITS_PER_PX).round(0).tolist())
    net = PatchNet("cnn")
    x = torch.zeros(2, 3, 272, 448)
    e, lo, sc = net(x)
    print("эмбеддинги", tuple(e.shape), "логиты", tuple(lo.shape), "сцена", tuple(sc.shape))
    print(f"параметров: энкодер {count_params(net.encoder):,}, голова {count_params(net.head):,}")
    t = soft_target_sep(torch.tensor([[0.3, 0.7]]), MAP_GRID)
    p, m = decode_points(torch.log(t + 1e-9), MAP_GRID)
    print("цель → точка", p.numpy().round(4), "масса окна", m.numpy().round(3))
