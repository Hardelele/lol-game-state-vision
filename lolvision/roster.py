"""Состав матча по портретам HUD канала наблюдателя.

Зачем. «Кто это» на основном виде проще решать внутри состава из десяти
(docs/entity-model-speed.md, п. 6): имя над плашкой и облик выбирают из
пяти чемпионов команды, а не из 173. Состав виден в HUD постоянно.

Где портреты (1920×1080, раскладка spectator-volibear-challenger, проверено
на 58w57eJ5Qks_t00300): нижняя таблица — 34×34 px, синие x 924, красные
x 976, строки y 861 + 43.5·i; боковые панели — 38×38 px, левая x 32,
правая x 1848, строки y 160 + 102.75·i. Порядок строк в таблице и в
панелях один — по ролям (верх, лес, мид, стрелок, поддержка), так что
строка i таблицы и строка i панели — один и тот же чемпион.

Что в портретах. Базовые иконки Data Dragon, а не иконки скинов: на
опорном кадре нормированная корреляция с базовой иконкой 0.95–0.98 при
скинах на основном виде. Мешают рамка (сравниваем центр без восьмой части
с краёв), цифра уровня в углу панели, «?» в таблице (чемпион в тумане) и
серый портрет погибшего — поэтому решение по ролику, а не по кадру.

Решение по кадру: для каждого из 20 портретов — корреляция со всеми
иконками со сдвигом ±3 px. По ролику: средняя корреляция по кадрам, где
HUD есть, сумма двух видов одного слота и назначение без повторов
(венгерский алгоритм: десять разных чемпионов).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

TEAMS = ("ORDER", "CHAOS")          # синие слева / красные справа, как в health_bars
ROLES = ("top", "jungle", "mid", "bot", "support")
SHIFT = 3                           # поиск портрета ±3 px
HUD_MIN = 0.6                       # медиана лучших корреляций: ниже — HUD на кадре нет


@dataclass(frozen=True)
class View:
    """Один портрет: левый верхний угол иконки и её сторона в кадре 1080p."""
    name: str
    x: int
    y: int
    size: int


def views() -> list[list[View]]:
    """views()[slot] — портреты слота в таблице и в боковой панели.
    Слот = команда·5 + роль: 0–4 синие, 5–9 красные."""
    out = []
    for t, (tx, sx) in enumerate(((924, 32), (976, 1848))):
        for i in range(5):
            out.append([View(f"table_{TEAMS[t]}_{i}", tx, round(861 + 43.5 * i), 34),
                        View(f"side_{TEAMS[t]}_{i}", sx, round(160 + 102.75 * i), 38)])
    return out


def _norm(a: np.ndarray) -> np.ndarray:
    """Вычесть среднее и привести к единичной норме по последним трём осям."""
    a = a - a.mean(axis=(-3, -2, -1), keepdims=True)
    return a / (np.sqrt((a * a).sum(axis=(-3, -2, -1), keepdims=True)) + 1e-6)


def _core(size: int) -> int:
    """Отступ рамки: центр портрета без восьмой части с каждого края."""
    return size // 8


class IconBank:
    """Иконки чемпионов, приведённые к размерам портретов HUD."""

    def __init__(self, icons: dict[str, np.ndarray]):
        self.ids = sorted(icons)
        self.by_size: dict[int, np.ndarray] = {}
        for size in {v.size for vs in views() for v in vs}:
            m = _core(size)
            arr = np.stack([cv2.resize(icons[c], (size, size), interpolation=cv2.INTER_AREA)
                            for c in self.ids]).astype(np.float32)
            self.by_size[size] = _norm(arr[:, m:size - m, m:size - m])
        # Тот же банк плоской матрицей (признаки × чемпионы): одно умножение на кадр.
        self.flat = {s: np.ascontiguousarray(a.reshape(len(a), -1).T)
                     for s, a in self.by_size.items()}

    @classmethod
    def from_ddragon(cls, dd) -> IconBank:
        """Из lolvision.ddragon.DDragon (иконки должны лежать в кэше)."""
        return cls({c: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                    for c, p in dd.all_icons().items()})

    def match(self, rgb: np.ndarray, v: View) -> np.ndarray:
        """Корреляция портрета `v` со всеми иконками (лучшая по сдвигам), (n_champs,)."""
        m, s = _core(v.size), SHIFT
        k = v.size - 2 * m
        y0, x0 = v.y + m - s, v.x + m - s
        big = rgb[y0:y0 + k + 2 * s, x0:x0 + k + 2 * s].astype(np.float32)
        if big.shape[:2] != (k + 2 * s, k + 2 * s):
            return np.full(len(self.ids), -1.0, np.float32)
        wins = np.lib.stride_tricks.sliding_window_view(big, (k, k), axis=(0, 1))
        wins = wins.reshape(-1, 3, k, k).transpose(0, 2, 3, 1)       # (сдвиги, k, k, 3)
        bank = self.flat[v.size]
        return (_norm(wins).reshape(len(wins), -1) @ bank).max(0)


@dataclass
class FrameSlots:
    """Корреляции одного кадра: scores[slot, вид, чемпион]."""
    scores: np.ndarray

    @property
    def hud(self) -> bool:
        """HUD на кадре есть: портреты хорошо совпадают хоть с какой-то иконкой."""
        return float(np.median(self.scores.max(-1))) >= HUD_MIN


def read_frame(rgb: np.ndarray, bank: IconBank) -> FrameSlots:
    return FrameSlots(np.stack([[bank.match(rgb, v) for v in vs] for vs in views()]))


@dataclass
class Roster:
    """Состав: champions[slot] — Id Data Dragon, slot = команда·5 + роль."""
    champions: list[str]
    margin: list[float] = field(default_factory=list)  # отрыв от второго кандидата
    frames: int = 0                                    # кадров с HUD в решении

    def team(self, team: str) -> list[str]:
        t = TEAMS.index(team)
        return self.champions[5 * t:5 * t + 5]

    def team_of(self, cid: str) -> str:
        return TEAMS[self.champions.index(cid) // 5]

    def to_json(self) -> dict:
        return {"ORDER": self.team("ORDER"), "CHAOS": self.team("CHAOS"),
                "margin": [round(m, 3) for m in self.margin], "frames": self.frames}


def decide(frames: list[FrameSlots], ids: list[str]) -> Roster | None:
    """Состав по кадрам ролика: средняя корреляция, два вида слота вместе,
    десять разных чемпионов. Кадров с HUD нет — None."""
    good = [f.scores for f in frames if f.hud]
    if not good:
        return None
    mean = np.mean(good, axis=0).sum(1)                 # (слоты, чемпионы)
    rows, cols = linear_sum_assignment(-mean)
    champs = [""] * len(rows)
    margin = [0.0] * len(rows)
    for r, c in zip(rows, cols):
        champs[r] = ids[c]
        rest = np.delete(mean[r], c)
        margin[r] = float(mean[r, c] - rest.max()) / 2   # в единицах одного вида
    return Roster(champs, margin, len(good))
