"""Имя над плашкой → чемпион: закрытый словарь, ближайший образец.

Зачем. Над плашкой канал пишет английское имя чемпиона («Volibear»,
«Irelia») или вместо него состояние («CHANNELING», «STUNNED»). Словарь
закрыт — 173 имени Data Dragon и десяток состояний, — поэтому OCR общего
вида не нужен: как с уровнем в health_bars.py, вырез сравнивается с
образцами целиком.

Признак — «белизна»: минимум по каналам RGB, растянутый в 0..1 (текст
белый, обводка тёмная, фон сцены цветной). Сравнение — нормированная
корреляция со сдвигом по x ±4 px и по y ±2 px.

Образцы двух сортов:
* отрисованные шрифтом Windows (Georgia Bold — засечки и толщина близки к
  шрифту канала) — есть для каждого имени словаря, нужны для разметки
  сбора образцов и для чемпионов, которых в данных не было;
* настоящие вырезы — медиана автоматически размеченных вырезов чемпиона
  (разметка: состав матча по HUD + команда плашки, см. tools/train_names.py).
  Настоящий образец точнее отрисованного, и при наличии он главный.

Ограничение кандидатами: если состав известен, для плашки команды T
выбираем из пяти чемпионов T и состояний — ошибки «похожее имя чужого
чемпиона» исчезают. Известное ограничение: Neeko и другие копирующие облик
пишут над плашкой чужое имя (docs/champion-identity.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lolvision.identity import NAME_BOX

# Состояния над плашкой (английский клиент). Видели CHANNELING и STUNNED;
# остальные — из перечня эффектов контроля игры, могут не встречаться.
STATUSES = ("CHANNELING", "STUNNED", "ROOTED", "SILENCED", "TAUNTED", "FEARED", "CHARMED",
            "AIRBORNE", "ASLEEP", "SUPPRESSED", "POLYMORPHED", "BLINDED", "GROUNDED",
            "DISARMED", "NEARSIGHTED", "SLOWED", "KNOCKED UP", "DROWSY", "PULLED",
            "STASIS", "RECALLING")
STATUS = "status:"                  # префикс метки состояния
NONE = "none"                       # пусто или закрыто
CROP_H = NAME_BOX[3] - NAME_BOX[1]  # 19
CROP_W = NAME_BOX[2] - NAME_BOX[0]  # 147
DX, DY = 4, 2
WIN_H, WIN_W = CROP_H - 2 * DY, CROP_W - 2 * DX
FONT = Path("C:/Windows/Fonts/georgiab.ttf")
FONT_PX = 12                        # кегль, при котором ширина имён совпала с кадром
TRACK = 0.0                         # разрядка, px кадра
TEXT_CX = CROP_W / 2                # имя центрировано над плашкой
TEXT_Y = 4                          # верх заглавных в вырезе


def whiteness(crop: np.ndarray) -> np.ndarray:
    """Белизна выреза (H, W, 3) uint8 → (H, W) float32 в 0..1."""
    m = crop.min(2).astype(np.float32)
    return np.clip((m - 110.0) / 100.0, 0.0, 1.0)


def _norm(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return a / (np.sqrt((a * a).sum(axis=(-2, -1), keepdims=True)) + 1e-6)


def render(text: str, font: Path = FONT, px: int = FONT_PX,
           track: float = TRACK) -> np.ndarray:
    """Отрисованный образец: белизна (CROP_H, CROP_W), текст там, где он в кадре.
    track — добавка к промежутку между буквами в пикселях кадра (у канала
    разрядка заметная)."""
    big = 4                                           # рисуем крупно и сжимаем — мягкий край
    fb = ImageFont.truetype(str(font), px * big)
    img = Image.new("L", (CROP_W * big, CROP_H * big), 0)
    d = ImageDraw.Draw(img)
    adv = [d.textlength(ch, font=fb) + track * big for ch in text]
    width = sum(adv) - track * big
    _, y0, _, _ = d.textbbox((0, 0), "H", font=fb)
    x = TEXT_CX * big - width / 2
    for ch, a in zip(text, adv):
        d.text((x, TEXT_Y * big - y0), ch, fill=255, font=fb)
        x += a
    return np.asarray(img.resize((CROP_W, CROP_H), Image.BILINEAR), np.float32) / 255.0


def windows(w: np.ndarray) -> np.ndarray:
    """Все сдвиги окна (WIN_H, WIN_W) по белизне выреза → (сдвиги, WIN_H·WIN_W), нормированы."""
    v = np.lib.stride_tricks.sliding_window_view(w, (WIN_H, WIN_W))
    v = v.reshape(-1, WIN_H, WIN_W)
    return _norm(v).reshape(len(v), -1)


def center(w: np.ndarray) -> np.ndarray:
    """Окно без сдвига — так хранятся образцы."""
    return w[DY:DY + WIN_H, DX:DX + WIN_W]


class NameReader:
    """Банк образцов: labels[i] — Id чемпиона, «status:…» или «none»."""

    def __init__(self, labels: list[str], templates: np.ndarray, kinds: list[str] | None = None):
        self.labels = list(labels)
        self.kinds = list(kinds or ["real"] * len(labels))   # real / rendered
        t = np.asarray(templates, np.float32).reshape(len(labels), WIN_H, WIN_W)
        self.bank = _norm(t).reshape(len(labels), -1).T          # (признаки, образцы)

    @classmethod
    def rendered(cls, names: dict[str, str], font: Path = FONT, px: int = FONT_PX,
                 track: float = TRACK) -> NameReader:
        """Только отрисованные образцы: names — Id → отображаемое имя."""
        labels = list(names) + [STATUS + s for s in STATUSES]
        texts = list(names.values()) + list(STATUSES)
        return cls(labels, np.stack([center(render(t, font, px, track)) for t in texts]),
                   ["rendered"] * len(labels))

    @classmethod
    def load(cls, path: Path) -> NameReader:
        z = np.load(path)
        return cls([str(s) for s in z["labels"]], z["templates"], [str(s) for s in z["kinds"]])

    def save(self, path: Path) -> None:
        t = (self.bank.T.reshape(-1, WIN_H, WIN_W)).astype(np.float16)
        np.savez(path, labels=np.array(self.labels), templates=t, kinds=np.array(self.kinds))

    def scores(self, crop: np.ndarray) -> dict[str, float]:
        """Лучшее сходство с каждой меткой (по всем её образцам и сдвигам)."""
        sims = (windows(whiteness(crop)) @ self.bank).max(0)
        out: dict[str, float] = {}
        for lab, s in zip(self.labels, sims):
            if s > out.get(lab, -2.0):
                out[lab] = float(s)
        return out

    def read(self, crop: np.ndarray, candidates: list[str] | None = None,
             statuses: bool = True) -> tuple[str, float, float]:
        """(метка, сходство, отрыв от второй). candidates — Id допустимых
        чемпионов (например, пять из команды плашки); состояния добавляются,
        если statuses."""
        sc = self.scores(crop)
        if candidates is not None:
            keep = set(candidates)
            sc = {k: v for k, v in sc.items()
                  if k in keep or (statuses and k.startswith(STATUS)) or k == NONE}
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        if not ranked:
            return NONE, float("nan"), 0.0
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else -1.0
        return best, s1, s1 - s2
