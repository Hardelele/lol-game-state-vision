"""Снятие и отрисовка промежуточных состояний сети на одном кадре.

Нужно, чтобы сеть перестала быть чёрным ящиком: можно пройти по этапам и
посмотреть, что пришло на свёртку, что из неё вышло и как выглядит каждая
карта признаков по отдельности.

Состояния снимаются обычными хуками на модулях, поэтому модель не меняется и
никакой отдельной «отладочной» копии не заводится — смотрим ровно ту сеть,
которая выдаёт ответ.

Порядок этапов соответствует порядку выполнения: вход, затем для каждого
блока свёртка и результат после нормировки и ReLU, затем пулинг и голова.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from coord_model import CoordNet, GRID

# Тёплая шкала: тёмное — слабый отклик, светлое — сильный. Серая шкала хуже
# читается на мелких картах признаков, а радуга искажает порядок величин.
_RAMP = np.array([
    (8, 6, 20), (40, 18, 70), (95, 24, 100), (150, 38, 92),
    (203, 62, 70), (240, 110, 52), (252, 170, 70), (252, 230, 160),
], dtype=np.float32)


def colorize(a: np.ndarray) -> np.ndarray:
    """(H, W) в [0,1] → (H, W, 3) uint8 по тёплой шкале."""
    x = np.clip(a, 0, 1) * (len(_RAMP) - 1)
    i = np.floor(x).astype(int)
    f = (x - i)[..., None]
    i2 = np.minimum(i + 1, len(_RAMP) - 1)
    return (_RAMP[i] * (1 - f) + _RAMP[i2] * f).astype(np.uint8)


def norm(a: np.ndarray) -> np.ndarray:
    """Растянуть по 1–99 перцентилям: одиночный выброс не должен гасить карту."""
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-9:
        lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@lru_cache(maxsize=4)
def load_model(ckpt: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CoordNet(grid=ck.get("grid", GRID))
    m.encoder.load_state_dict(ck["encoder"])
    m.head.load_state_dict(ck["head"])
    m.eval()
    return m, tuple(ck["input_size"])


def stage_plan(model: CoordNet) -> list[dict]:
    """Описание этапов в порядке выполнения.

    Для каждой свёртки заводится два этапа: её собственный выход и выход
    после нормировки с ReLU. Так видно и то, что свёртка посчитала, и то,
    что реально ушло дальше.
    """
    plan = [{"id": "input", "name": "вход", "kind": "image",
             "note": "кадр после маски, обрезки и масштаба"}]
    enc = model.encoder.stages
    for bi, block in enumerate(enc):
        convs = [i for i, m in enumerate(block) if isinstance(m, nn.Conv2d)]
        for ci, idx in enumerate(convs):
            conv: nn.Conv2d = block[idx]
            sid = f"stages.{bi}.{idx}"
            plan.append({
                "id": sid, "name": f"блок {bi + 1} · свёртка {ci + 1}",
                "kind": "conv",
                "note": f"{conv.in_channels}→{conv.out_channels}, "
                        f"ядро {conv.kernel_size[0]}×{conv.kernel_size[1]}, "
                        f"шаг {conv.stride[0]}",
                "stride": int(conv.stride[0]),
            })
            relu = idx + 2
            if relu < len(block) and isinstance(block[relu], nn.ReLU):
                plan.append({
                    "id": f"stages.{bi}.{relu}",
                    "name": f"блок {bi + 1} · после ReLU {ci + 1}",
                    "kind": "act",
                    "note": "нормировка по группам и отсечение отрицательных",
                })
    plan.append({"id": "pool", "name": "пулинг", "kind": "act",
                 "note": "усреднение до мелкой сетки; положение в кадре сохраняется"})
    plan.append({"id": "heat", "name": "ответ", "kind": "heat",
                 "note": "распределение вероятности по карте"})
    return plan


@lru_cache(maxsize=24)
def activations(ckpt: str, scene_path: str):
    """Прогнать кадр через сеть и вернуть состояния по именам модулей."""
    model, size = load_model(ckpt)
    with Image.open(scene_path) as im:
        a = np.asarray(im.convert("RGB").resize(size, Image.BILINEAR),
                       dtype=np.float32) / 255.0
    x = torch.from_numpy(a.transpose(2, 0, 1)[None] * 2 - 1)

    caught: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _i, out):
            caught[name] = out.detach()
        return fn

    handles = []
    for name, mod in model.encoder.stages.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ReLU)) and name:
            handles.append(mod.register_forward_hook(hook("stages." + name)))
    handles.append(model.pool.register_forward_hook(hook("pool")))
    with torch.no_grad():
        logits = model(x)
    for h in handles:
        h.remove()

    out = {k: v[0].numpy() for k, v in caught.items()}
    out["input"] = x[0].numpy()
    g = model.grid
    out["heat"] = torch.softmax(logits, 1)[0].reshape(g, g).numpy()[None]
    return out


def sheet_cell(channels: int) -> tuple[int, int]:
    """Сколько столбцов и какой ширины клетка для контактного листа.

    Клетка тем крупнее, чем меньше каналов: у первого блока их 32, и мельчить
    незачем, а у последнего 256, и крупные клетки в панель не влезут.
    """
    cols = int(np.ceil(np.sqrt(channels)))
    return cols, int(max(30, min(118, round(760 / cols))))


def grid_png(act: np.ndarray, cell: int | None = None, cols: int | None = None,
             order: list[int] | None = None) -> bytes:
    """Контактный лист из всех карт признаков этапа."""
    c, h, w = act.shape
    idx = order if order is not None else list(range(c))
    auto_cols, auto_cell = sheet_cell(len(idx))
    cols = cols or auto_cols
    cell = cell or auto_cell
    rows = int(np.ceil(len(idx) / cols))
    ch = max(8, round(cell * h / w))
    sheet = Image.new("RGB", (cols * (cell + 2) + 2, rows * (ch + 2) + 2), (14, 16, 20))
    for k, ci in enumerate(idx):
        tile = Image.fromarray(colorize(norm(act[ci]))).resize((cell, ch), Image.NEAREST)
        sheet.paste(tile, (2 + (k % cols) * (cell + 2), 2 + (k // cols) * (ch + 2)))
    return png(sheet)


def channel_png(act: np.ndarray, ch: int, width: int = 384) -> bytes:
    a = act[ch]
    h = max(1, round(width * a.shape[0] / a.shape[1]))
    return png(Image.fromarray(colorize(norm(a))).resize((width, h), Image.NEAREST))


def rgb_png(act: np.ndarray, width: int = 384) -> bytes:
    """Вход сети как картинка: три канала обратно в цвет."""
    a = ((act.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    h = round(width * a.shape[0] / a.shape[1])
    return png(Image.fromarray(a).resize((width, h), Image.BILINEAR))


def kernel_png(ckpt: str, stage: str, ch: int, cell: int = 18) -> bytes:
    """Ядра свёртки для одного выходного канала: по плитке на входной канал."""
    model, _ = load_model(ckpt)
    mod = model.encoder.stages.get_submodule(stage[len("stages."):])
    if not isinstance(mod, nn.Conv2d):
        raise ValueError("этап не является свёрткой")
    w = mod.weight.detach()[ch].numpy()               # (in, kh, kw)
    n, kh, kw = w.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    lim = float(np.abs(w).max()) or 1.0
    sheet = Image.new("RGB", (cols * (cell + 1) + 1, rows * (cell + 1) + 1),
                      (14, 16, 20))
    for i in range(n):
        # Знак важен: синее — вклад отрицательный, оранжевое — положительный.
        a = w[i] / lim
        rgb = np.zeros((kh, kw, 3), np.uint8)
        rgb[..., 0] = np.clip(255 * np.maximum(a, 0), 0, 255)
        rgb[..., 1] = np.clip(140 * np.abs(a), 0, 255)
        rgb[..., 2] = np.clip(255 * np.maximum(-a, 0), 0, 255)
        tile = Image.fromarray(rgb).resize((cell, cell), Image.NEAREST)
        sheet.paste(tile, (1 + (i % cols) * (cell + 1), 1 + (i // cols) * (cell + 1)))
    return png(sheet)


def channel_stats(act: np.ndarray) -> list[dict]:
    """Сводка по каналам: по ней канал можно отсортировать по активности."""
    flat = act.reshape(len(act), -1)
    return [{"c": i, "mean": round(float(m), 4), "max": round(float(x), 4),
             "live": round(float(l), 5)}
            for i, (m, x, l) in enumerate(zip(flat.mean(1), flat.max(1),
                                              (flat > 0).mean(1)))]
