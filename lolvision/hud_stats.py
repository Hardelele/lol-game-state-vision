"""Полоса статистики HUD вида игрока: добитые крипы (CS) и время матча.

Зачем. Live Client API в ролике YouTube нет, а добивания (#816) видны на
HUD: в правом верхнем углу вида игрока — «счёт команд, K/D/A, CS, время».
Рост CS между кадрами — метка «добил», время матча — часы игры, не
зависящие от склеек и пауз ролика.

Геометрия при 1920×1080 (замер по 10 роликам вида игрока из
docs/hud-spells.md: HUD 0,96–1,28 от своего, полоса от масштаба HUD не
зависит — у всех десяти на месте с точностью до пикселя): значок миньона
x 1766…1774, цифры CS от x 1781 до значка часов (x ~1840), цифры времени
от x 1858; строка цифр y 9…20 (12 px). Цифры прижаты влево к значку.

Чтение: яркость (максимум по каналам — у полужирного шрифта цифры жёлтые,
у обычного белые), порог относительно фона полосы, разрез по пустым
столбцам, каждая цифра — ближайший сосед среди образцов
(dataset/hud_stats_digits.npz). Далеко от всех образцов, не та длина,
нет значка миньона — `None`: «не прочитано», не ноль.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lolvision.paths import DATASET

DIGITS = DATASET / "hud_stats_digits.npz"
ICON = DATASET / "hud_stats_icon.png"
REF_W, REF_H = 1920, 1080
# Строка цифр и поля полосы (x0, x1) в координатах 1920×1080.
ROW_Y = (5, 25)
ICON_X, ICON_Y = (1762, 1779), (5, 25)
CS_X = (1779, 1838)
TIME_X = (1856, 1918)
GLYPH = (8, 12)                 # нормированный размер цифры (ш, в)
LIT = 105                       # ярче — пиксель цифры (фон полосы 20–60)
MIN_H = 8                       # ниже — не цифра и не двоеточие (крошки)
WIDE = 1.0                      # шире высоты строки — слиплись две цифры
ICON_OK = 0.6                   # совпадение значка миньона: полоса на месте
FAR = 0.30                      # средний квадрат разности с ближайшим образцом
AMBIG = 0.5                     # ближайший класс не дальше этой доли второго — иначе спорно


def _scale(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.shape[:2] != (REF_H, REF_W):
        frame_bgr = cv2.resize(frame_bgr, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
    return frame_bgr


def strip(frame_bgr: np.ndarray) -> np.ndarray:
    """Вырез полосы y 0…36, x 1500…1920 (кадр приводится к 1920×1080) —
    в нём работают все функции ниже; инструменты хранят только его.

    Кадр другого размера масштабируется не целиком, а угол с запасом в
    пару пикселей: живой сеанс зовёт это на каждом кадре вида игрока."""
    h, w = frame_bgr.shape[:2]
    if (h, w) == (REF_H, REF_W):
        return np.ascontiguousarray(frame_bgr[0:36, 1500:1920])
    sx, sy = w / REF_W, h / REF_H
    pad = 4
    x0, y1 = max(0, int((1500 - pad) * sx)), min(h, int(np.ceil((36 + pad) * sy)))
    part = frame_bgr[0:y1, x0:w]
    rw, rh = round(part.shape[1] / sx), round(part.shape[0] / sy)
    part = cv2.resize(part, (rw, rh), interpolation=cv2.INTER_AREA)
    ox = 1500 - round(x0 / sx)
    return np.ascontiguousarray(part[0:36, ox:ox + 420])


X0 = 1500                       # левый край выреза strip() в координатах кадра


@dataclass
class Glyph:
    x0: int                     # столбцы в вырезе
    x1: int
    img: np.ndarray             # GLYPH (в, ш), float32 0..1
    width: int
    height: int


def glyphs(region: np.ndarray) -> list[Glyph]:
    """Цифры поля слева направо. `region` — BGR, строка цифр целиком."""
    g = region.max(2).astype(np.float32)
    m = g > LIT
    cols = m.any(0)
    out, x = [], 0
    while x < len(cols):
        if not cols[x]:
            x += 1
            continue
        x1 = x
        while x1 < len(cols) and cols[x1]:
            x1 += 1
        for a, b in _split(g, m, x, x1):
            gl = _glyph(g, m, a, b)
            if gl is not None:
                out.append(gl)
        x = x1
    return out


def _split(g: np.ndarray, m: np.ndarray, x: int, x1: int) -> list[tuple[int, int]]:
    """Слипшиеся цифры («44», «:3» — сжатие ролика размывает зазор) режутся
    по самому тёмному столбцу, пока кусок шире WIDE высоты строки."""
    ys = np.nonzero(m[:, x:x1].any(1))[0]
    hgt = ys[-1] - ys[0] + 1
    if x1 - x <= WIDE * hgt or x1 - x < 4:
        return [(x, x1)]
    prof = g[ys[0]:ys[-1] + 1, x + 1:x1 - 1].sum(0)
    cut = x + 1 + int(np.argmin(prof))
    return _split(g, m, x, cut) + _split(g, m, cut, x1)


def _glyph(g: np.ndarray, m: np.ndarray, x: int, x1: int) -> Glyph | None:
    ys = np.nonzero(m[:, x:x1].any(1))[0]
    if not len(ys):
        return None
    hgt = int(ys[-1] - ys[0] + 1)
    if hgt < MIN_H:
        return None
    part = g[ys[0]:ys[-1] + 1, x:x1]
    lo, hi = float(part.min()), float(part.max())
    part = (part - lo) / max(hi - lo, 1.0)
    # Ширина сохраняется: «1» вписывается в середину, а не растягивается.
    h0 = GLYPH[1]
    w0 = max(1, min(GLYPH[0], round((x1 - x) * h0 / hgt)))
    r = cv2.resize(part, (w0, h0), interpolation=cv2.INTER_AREA)
    img = np.zeros((GLYPH[1], GLYPH[0]), np.float32)
    off = (GLYPH[0] - w0) // 2
    img[:, off:off + w0] = r
    return Glyph(x, x1, img, x1 - x, hgt)


def fields(st: np.ndarray) -> dict[str, np.ndarray]:
    """Поля полосы из выреза strip(): значок, CS, время."""
    y0, y1 = ROW_Y
    return {"icon": st[ICON_Y[0]:ICON_Y[1], ICON_X[0] - X0:ICON_X[1] - X0],
            "cs": st[y0:y1, CS_X[0] - X0:CS_X[1] - X0],
            "time": st[y0:y1, TIME_X[0] - X0:TIME_X[1] - X0]}


class DigitReader:
    """Ближайший сосед по образцам цифр полосы. Классы — '0'…'9'."""

    def __init__(self, path: Path = DIGITS):
        self.ok = path.exists()
        if self.ok:
            z = np.load(path)
            self.x = z["glyphs"].reshape(len(z["glyphs"]), -1).astype(np.float32)
            self.y = z["labels"].astype(str)

    def classify(self, gl: Glyph) -> tuple[str, float]:
        """Класс и расстояние; спорная цифра — класс '?'. Цифра сравнивается
        и со сдвигом на столбец в обе стороны: засечка у полужирной «1»
        сдвигает её в рамке, и без сдвига «1» ролика ANXVB571YkE была
        одинаково далека от «1» и «4» образцов."""
        shifts = [gl.img, np.roll(gl.img, 1, 1), np.roll(gl.img, -1, 1)]
        shifts[1][:, 0] = 0
        shifts[2][:, -1] = 0
        q = np.stack([s.reshape(-1) for s in shifts])
        d = ((self.x[None] - q[:, None]) ** 2).mean(2).min(0)
        k = int(d.argmin())
        c = str(self.y[k])
        other = d[self.y != c]
        if len(other) and d[k] > AMBIG * float(other.min()):
            return "?", float(d[k])
        return c, float(d[k])

    def number(self, region: np.ndarray, max_len: int) -> int | None:
        gs = glyphs(region)
        if not self.ok or not gs or len(gs) > max_len:
            return None
        s = ""
        for gl in gs:
            c, d = self.classify(gl)
            if d > FAR or not c.isdigit():
                return None
            s += c
        return int(s)


class IconCheck:
    """Значок миньона на месте — полоса статистики видна (вид игрока)."""

    def __init__(self, path: Path = ICON):
        self.ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None

    def score(self, icon_bgr: np.ndarray) -> float | None:
        if self.ref is None:
            return None
        g = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r = self.ref.astype(np.float32)
        g, r = g - g.mean(), r - r.mean()
        den = float(np.sqrt((g * g).sum() * (r * r).sum()))
        return float((g * r).sum() / den) if den > 0 else 0.0


@dataclass
class Stats:
    present: bool               # значок миньона на месте
    icon: float | None
    cs: int | None
    time: int | None            # секунды матча


class StatsReader:
    def __init__(self, digits: Path = DIGITS, icon: Path = ICON):
        self.digits = DigitReader(digits)
        self.icon = IconCheck(icon)

    def read_strip(self, st: np.ndarray) -> Stats:
        f = fields(st)
        sc = self.icon.score(f["icon"])
        if sc is not None and sc < ICON_OK:
            return Stats(False, sc, None, None)
        cs = self.digits.number(f["cs"], 4)
        return Stats(True, sc, cs, self.read_time(f["time"]))

    def read(self, frame_bgr: np.ndarray) -> Stats:
        return self.read_strip(strip(frame_bgr))

    def read_time(self, region: np.ndarray) -> int | None:
        """MM:SS — ровно 4 цифры и двоеточие (образец ':'), иначе None."""
        gs = glyphs(region)
        if not self.digits.ok or not 4 <= len(gs) <= 5:
            return None
        ds = []
        for gl in gs:
            c, d = self.digits.classify(gl)
            if d > FAR or not (c.isdigit() or c == ":"):
                return None
            if c != ":":
                ds.append(int(c))
        if len(ds) != 4:
            return None
        mm, ss = ds[0] * 10 + ds[1], ds[2] * 10 + ds[3]
        return None if ss > 59 else mm * 60 + ss


CLOCK_STALE = 10.0              # с без прочтения — часам по полосе не верим
CLOCK_SLACK = 0.35              # с: время кадра против показа HUD (шаг кадра, сжатие)


@dataclass
class _Segment:
    """Отрезок, где время матча идёт вместе с кадром: время матча =
    t + смещение, смещение в [lo, hi]. Полоса показывает целые секунды
    (вниз), и каждое прочтение сужает промежуток."""
    lo: float
    hi: float
    t: float                    # последнее прочтение
    n: int = 1

    def fits(self, t: float, secs: int) -> bool:
        return secs - t <= self.hi + CLOCK_SLACK and secs + 1 - t >= self.lo - CLOCK_SLACK

    def add(self, t: float, secs: int) -> None:
        lo, hi = max(self.lo, secs - t), min(self.hi, secs + 1 - t)
        # Промежуток схлопнулся в пределах допуска — берём середину спора.
        if lo > hi:
            lo = hi = (lo + hi) / 2
        self.lo, self.hi, self.t, self.n = lo, hi, t, self.n + 1

    def value(self, t: float) -> float:
        return t + (self.lo + self.hi) / 2


class HudClock:
    """Часы матча по полосе статистики (#780, #816): время матча без Live
    Client API — у ролика YouTube его нет.

    Прочтения MM:SS (`StatsReader.read(...).time`) → время матча на момент
    кадра, с точностью до доли секунды. Новый отрезок (пауза игры,
    перемотка или склейка ролика, новая игра) принимается, только когда
    следующее прочтение его подтверждает: одиночная ошибка чтения часы не
    сбивает. Дольше CLOCK_STALE без прочтения — None (полоса не видна:
    Tab, смерть с затемнением, чужой вид)."""

    def __init__(self):
        self.cur: _Segment | None = None
        self._cand: _Segment | None = None

    def update(self, t: float, secs: int | None) -> None:
        if secs is None:
            return
        if self.cur is not None and self.cur.fits(t, secs):
            self.cur.add(t, secs)
            self._cand = None
            return
        if self._cand is not None and self._cand.fits(t, secs):
            self._cand.add(t, secs)
            self.cur, self._cand = self._cand, None
            return
        self._cand = _Segment(secs - t, secs + 1 - t, t)

    def value(self, t: float) -> float | None:
        """Время матча на момент `t` или None — часов по полосе нет."""
        c = self.cur
        if c is None or t - c.t > CLOCK_STALE or t < c.t - CLOCK_STALE:
            return None
        return c.value(t)

    def reset(self) -> None:
        self.cur = self._cand = None
