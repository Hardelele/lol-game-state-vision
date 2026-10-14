"""Свой чемпион на виде игрока — по иконкам умений Q/W/E/R панели HUD (#843).

Зачем. Индекс роликов должен знать, за кого играет автор, а заявка в
названии — не факт (перезалив, «X vs Y» в любом порядке). Портрет своего
HUD для этого не годится: он рисуется по облику (скину), а иконки Data
Dragon — базовый облик. Проверено на 6k5-vQ_GQ_M (Volibear в тёмном облике):
по портрету лучший — TahmKench 0.69, Volibear не в первой пятёрке. Иконки
умений от облика не зависят: на том же кадре Volibear первый.

Как. Панель умений ищет lolvision.hud_spells.locate_panel (масштаб HUD
стримера свой). Каждый слот Q/W/E/R — вырез без рамки, 24×24, цвет, минус
среднее, единичная норма; с иконкой умения чемпиона — скалярное
произведение (NCC), лучшее по сдвигам ±SHIFT px (место панели находится с
ошибкой в пару пикселей). Оценка чемпиона на кадре — среднее NCC четырёх
слотов со своими четырьмя умениями: перезарядка (тёмная иконка) и
невыученное умение (серая) снижают все оценки одинаково, а не только
правильную. Решение — по сумме оценок нескольких кадров (`decide`).

Замер (кадры 6k5-vQ_GQ_M, 1080p): t=300 — Volibear 0.25, второй Braum 0.24
(панель найдена плохо, совпадение 0.43); t=900 — Volibear 0.65, второй
0.33. Поэтому решение требует отрыва MARGIN от второго по сумме кадров.
Чемпионы с превращением (Jayce, Nidalee, Elise) показывают вторую форму
умений, которых в Data Dragon нет отдельно: у них оценка ниже, решение —
по кадрам первой формы.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from lolvision import hud_spells

SIZE = 24                   # сторона выреза слота для сравнения
INSET = 0.08                # отступ от рамки слота, доля стороны
SHIFT = 2                   # px кадра: сдвиги выреза по x и y
MIN_SCORE = 0.40            # средняя оценка лучшего по кадрам с панелью не ниже
MARGIN = 0.08               # и отрыв от второго не меньше
SLOTS = ("Q", "W", "E", "R")


def _vec(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    i, j = int(round(h * INSET)), int(round(w * INSET))
    c = cv2.resize(np.ascontiguousarray(img[i:h - i, j:w - j]), (SIZE, SIZE),
                   interpolation=cv2.INTER_AREA).astype(np.float32)
    c -= c.mean()
    return (c / (np.linalg.norm(c) + 1e-6)).reshape(-1)


class AbilityBank:
    """Иконки умений всех чемпионов: Id → 4 иконки RGB (Q, W, E, R)."""

    def __init__(self, icons: Mapping[str, Sequence[np.ndarray]]):
        self.ids = sorted(c for c, v in icons.items() if len(v) == 4)
        # (слот, чемпион, признаки): слот k сравнивается только с k-м умением.
        self.bank = np.stack([np.stack([_vec(icons[c][k]) for c in self.ids])
                              for k in range(4)])

    @classmethod
    def from_ddragon(cls, dd) -> AbilityBank:
        """Из lolvision.ddragon.DDragon: умения из полной записи чемпиона,
        картинки — `spell_icon` (кэш или `fetch` у DDragon)."""
        icons = {}
        for cid in dd.champions():
            spells = dd.champion_full(cid)["spells"]
            icons[cid] = [cv2.cvtColor(cv2.imread(str(dd.spell_icon(s["image"]["full"]))),
                                       cv2.COLOR_BGR2RGB) for s in spells]
        return cls(icons)

    def slot_vectors(self, rgb: np.ndarray, panel: hud_spells.Panel) -> np.ndarray:
        """(слот, сдвиг, признаки) — вырезы Q/W/E/R со сдвигами."""
        h, w = rgb.shape[:2]
        out = []
        for key in SLOTS:
            x0, y0, x1, y1 = panel.box(key)
            vs = []
            for dy in range(-SHIFT, SHIFT + 1, 2):
                for dx in range(-SHIFT, SHIFT + 1, 2):
                    xa, ya = round(x0) + dx, round(y0) + dy
                    xb, yb = round(x1) + dx, round(y1) + dy
                    if xa < 0 or ya < 0 or xb > w or yb > h or xb - xa < 8 or yb - ya < 8:
                        continue
                    vs.append(_vec(rgb[ya:yb, xa:xb]))
            if not vs:
                vs = [np.zeros(SIZE * SIZE * 3, np.float32)]
            out.append(np.stack(vs))
        n = min(len(v) for v in out)
        return np.stack([v[:n] for v in out])

    def scores(self, rgb: np.ndarray, panel: hud_spells.Panel) -> np.ndarray:
        """Оценка каждого чемпиона на кадре (len(ids),): среднее по слотам
        лучшего по сдвигам NCC слота с умением чемпиона."""
        v = self.slot_vectors(rgb, panel)                    # (4, сдвиги, F)
        per = np.einsum("ksf,kcf->ksc", v, self.bank).max(1)  # (4, чемпионы)
        return per.mean(0)


@dataclass(frozen=True)
class Decision:
    champion: str | None
    score: float | None          # средняя оценка выбранного по кадрам
    second: str | None
    margin: float | None
    frames: int
    reason: str | None = None

    def to_dict(self) -> dict:
        r = (lambda x: None if x is None else round(float(x), 3))
        return {"champion": self.champion, "score": r(self.score), "second": self.second,
                "margin": r(self.margin), "frames": self.frames, "reason": self.reason}


def decide(ids: Sequence[str], per_frame: Sequence[np.ndarray],
           min_score: float = MIN_SCORE, margin: float = MARGIN) -> Decision:
    """Свой чемпион по оценкам нескольких кадров: лучший по среднему, если
    он уверен (не ниже `min_score`) и оторвался от второго на `margin`."""
    if not per_frame:
        return Decision(None, None, None, None, 0, "панель умений не найдена ни на одном кадре")
    m = np.mean(np.stack(per_frame), 0)
    order = np.argsort(-m)
    a, b = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    gap = float(m[a] - m[b])
    if m[a] < min_score:
        return Decision(None, float(m[a]), ids[b], gap, len(per_frame),
                        f"лучший {ids[a]} {m[a]:.2f} ниже {min_score}")
    if gap < margin:
        return Decision(None, float(m[a]), ids[b], gap, len(per_frame),
                        f"{ids[a]} {m[a]:.2f} и {ids[b]} {m[b]:.2f} — отрыв {gap:.2f} < {margin}")
    return Decision(ids[a], float(m[a]), ids[b], gap, len(per_frame))
