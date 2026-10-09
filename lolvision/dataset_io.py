"""Чтение собранных наборов: общее для обучения, проверок и смотрелок.

* манифесты data/coords/<ролик>.csv (собирает build_coords.py) — строки с
  отбором по качеству детекции рамки и набор для DataLoader по ним;
* маска «здесь сцена» в пикселях хранимой сцены той же раскладки;
* разметка «игра / не игра» — интервалы времени в
  dataset/not_game/<ролик>/labels.csv (кадры к ней собирает
  build_not_game.py, учит по ним train_not_game.py).

Раньше это жило в обучающих скриптах и сборщиках, и другие скрипты
импортировали их ради одной функции.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lolvision.paths import NOT_GAME_LABELS
from lolvision.mask_frames import crop_box, keep_grid

STORE_W = 720       # ширина хранимой сцены; вход модели делается из неё
MINI_STORE = 256    # сторона хранимой миникарты



# ------------------------------------------------------------- data/coords

def load_rows(ids: list[str], root: Path, min_q: float) -> list[dict]:
    rows = []
    for vid in ids:
        f = root / f"{vid}.csv"
        if not f.exists():
            raise SystemExit(f"нет манифеста {f} — сначала tools/build_coords.py")
        all_r = list(csv.DictReader(f.open(encoding="utf-8")))
        keep = [r for r in all_r if float(r["quality"]) >= min_q]
        print(f"  {vid}: {len(keep)} из {len(all_r)} кадров "
              f"(отброшено по качеству детекции {len(all_r) - len(keep)})")
        rows += keep
    return rows


class Scenes(Dataset):
    def __init__(self, rows: list[dict], root: Path, size: tuple[int, int],
                 augment: bool = False):
        self.rows, self.root, self.size, self.augment = rows, root, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r["scene"]) as im:
            a = np.asarray(im.convert("RGB").resize(self.size, Image.BILINEAR),
                           dtype=np.float32) / 255.0
        x = a.transpose(2, 0, 1) * 2 - 1
        if self.augment:
            # Только яркость и контраст. Сдвиг кадра сдвигает и настоящее
            # положение камеры, поэтому без пересчёта метки он её ломает.
            g = np.random.default_rng()
            m = x.mean()
            x = np.clip((x - m) * g.uniform(0.85, 1.15) + m + g.uniform(-0.15, 0.15),
                        -1, 1)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.tensor(float(r["cx"])), torch.tensor(float(r["cy"])))


def stored_keep(layout: dict, full=(1920, 1080), store=(720, 372)) -> np.ndarray:
    """Маска «здесь сцена» в пикселях хранимой сцены (bool, H×W)."""
    k = keep_grid(full, layout)
    x0, y0, x1, y1 = crop_box(full, layout, "dense")
    k = k[y0:y1, x0:x1].astype(np.uint8) * 255
    return cv2.resize(k, store, interpolation=cv2.INTER_NEAREST) > 127


# ------------------------------------------------------------- «игра / не игра»

def load_labels(vid: str, root: Path = NOT_GAME_LABELS) -> list[dict]:
    """Интервалы разметки ролика: start_sec, end_sec, label, kind, note."""
    p = root / vid / "labels.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    for r in rows:
        r["start_sec"], r["end_sec"] = float(r["start_sec"]), float(r["end_sec"])
    return rows


def label_at(intervals: list[dict], t: float) -> tuple[str, str]:
    """Метка и подкласс для момента t; вне разметки — ("", "")."""
    for r in intervals:
        if r["start_sec"] <= t < r["end_sec"]:
            return r["label"], r["kind"]
    return "", ""
