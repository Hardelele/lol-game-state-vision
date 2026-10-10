"""Чекпоинты моделей: одно место, где они читаются и собираются в модель.

Тип берётся из ключей самого чекпоинта, а не из каталога прогона (CoordNet
для сравнения лежит и в runs/patches):

* патчевая модель (PatchNet.checkpoint в lolvision/patch_model.py) — "kind": "patch";
* CoordNet (tools/train_coords.py) — "encoder", "head", "input_size", "grid".

Чекпоинт — pickle, то есть код (weights_only=False), поэтому читать можно
только свои прогоны.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lolvision.coord_model import CoordNet
from lolvision.patch_model import PatchNet, FrameHead, SceneHead

COORDNET_KEYS = ("encoder", "head", "input_size", "grid")


def read_checkpoint(path: Path, mmap: bool = False) -> dict:
    """Чекпоинт словарём на процессоре. `mmap` — когда нужны только поля:
    тензоры тогда не читаются с диска целиком."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def checkpoint_kind(ck: dict) -> str | None:
    """«patch», «coordnet» или None — чужой или неизвестный чекпоинт."""
    if ck.get("kind") == "patch":
        return "patch"
    if all(k in ck for k in COORDNET_KEYS):
        return "coordnet"
    return None


def _coordnet(ck: dict) -> CoordNet:
    m = CoordNet(grid=ck["grid"])
    m.encoder.load_state_dict(ck["encoder"])
    m.head.load_state_dict(ck["head"])
    return m


def _patchnet(ck: dict) -> PatchNet:
    m = PatchNet(ck["encoder"], ck.get("dim", 128))
    # Веса DINOv2 в чекпоинт не кладутся: их даёт torch.hub.
    missing, unexpected = m.load_state_dict(ck["state"], strict=False)
    bad = [k for k in missing if not k.startswith("encoder.vit.")]
    if bad or unexpected:
        raise RuntimeError(f"чекпоинт не подходит: {bad[:5]} {unexpected[:5]}")
    fh = ck.get("frame_head")
    if fh:
        m.frame = FrameHead(fh["cin"], fh.get("hidden", 64))
        m.frame.load_state_dict(fh["state"])
        m.frame_threshold = float(fh.get("threshold", 0.5))
    sh = ck.get("scene_head")
    if sh:
        m.scene_kind = SceneHead(sh["cin"], sh["classes"], sh.get("hidden", 64))
        m.scene_kind.load_state_dict(sh["state"])
    return m


def load_checkpoint(path: Path, device: str = "cpu",
                    kind: str | None = None) -> tuple[nn.Module, dict]:
    """Модель в режиме eval на `device` и сам чекпоинт — его поля (input_size,
    units_per_px, trained_on…) нужны вызывающему. `kind` — какой тип ждём."""
    ck = read_checkpoint(path)
    got = checkpoint_kind(ck)
    if got is None:
        raise ValueError(f"неизвестный тип чекпоинта: {Path(path).name}")
    if kind and got != kind:
        raise ValueError(f"{Path(path).name}: чекпоинт {got}, а нужен {kind}")
    m = _patchnet(ck) if got == "patch" else _coordnet(ck)
    return m.to(device).eval(), ck
