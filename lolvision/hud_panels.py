"""Боковые панели HUD канала наблюдателя: кто мёртв и таймер возрождения.

Зачем. Жив/мёртв нельзя брать с миникарты: иконка пропадает и в тумане,
и в невидимости, и у Юми на союзнике, и в царстве Мордекайзера. Канал
рисует смерть явно: портрет в боковой панели становится серым, поверх —
красный таймер возрождения, полосы ХП/ресурса пустые
(кадр ibUVbSX7ARU_t00510, слоты L3 «17» и L4 «1»).

Слоты: слева синие (ORDER) L1…L5, справа красные (CHAOS) R1…R5, сверху
вниз в порядке ролей канала. Геометрия при 1920×1080: внутренняя часть
портрета слева x 34…58, справа x 1857…1883, y 162…194 у первого слота,
шаг 102 px (замер по ibUVbSX7ARU_t00510 и 58w57eJ5Qks_t00300).

Признаки: медиана насыщенности портрета вне красных цифр (серый портрет —
единицы, живой — обычно больше 50), площадь красных цифр и число зелёных
пикселей полосы ХП (у мёртвого полоса пустая); мёртв — когда сходятся все
три. Таймер — цифры, разделённые по столбцам, ближайший сосед среди
образцов (dataset/hud_timer_digits.npz).

Сион (пассивка), Картус, Ког'Мао действуют после смерти, но по HUD они
мертвы — так и пишем. Ангел-хранитель, ульта Зилеана, яйцо Анивии —
«почти смерть»: портрет при этом не серый, таймера нет (не проверено на
кадрах — помечено в docs/minimap-champions.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lolvision.paths import DATASET

DIGITS = DATASET / "hud_timer_digits.npz"
REF_W, REF_H = 1920, 1080
# Справа портрет слева сверху закрывает кружок руны — берём правее.
X_LEFT, X_RIGHT = (34, 56), (1862, 1886)
Y0, H, PITCH = 162, 32, 102
# Шире внутренней части — для цифр таймера (двузначные выходят к краям).
DIGIT_X_LEFT, DIGIT_X_RIGHT = (31, 64), (1853, 1888)
DIGIT_Y = (8, 36)                   # от верха внутренней части слота
# Медиана насыщенности серого портрета ниже. Серый портрет не совсем серый:
# в область заходят цветной значок ранга и кольцо руны, у трети мёртвых
# слотов медиана 45–80 (4AIX8QtRid4_t01725, L3 и R3). Главные признаки —
# красные цифры и пустая полоса ХП, насыщенность только отсекает живых.
SAT_DEAD = 80.0
# Полоса ХП под портретом: у мёртвого пустая (зелёных пикселей почти нет).
HP_X_LEFT, HP_X_RIGHT = (32, 70), (1850, 1888)
HP_Y = (43, 50)                     # от верха внутренней части слота
HP_EMPTY = 6
RED_MIN = 12                        # пикселей красных цифр у мёртвого не меньше
GLYPH = (10, 14)                    # нормированный размер цифры (ш, в)


@dataclass
class Slot:
    side: str            # "L" / "R"
    index: int           # 0..4 сверху вниз
    sat: float
    red: int
    hp_px: int
    dead: bool
    timer: int | None = None


def slot_boxes(size: tuple[int, int] = (REF_W, REF_H)) -> list[tuple[int, int, int, int]]:
    """10 прямоугольников внутренней части портретов (x0, y0, x1, y1): L1…L5, R1…R5."""
    w, h = size
    sx, sy = w / REF_W, h / REF_H
    out = []
    for x0, x1 in (X_LEFT, X_RIGHT):
        for k in range(5):
            y0 = Y0 + PITCH * k
            out.append((round(x0 * sx), round(y0 * sy), round(x1 * sx), round((y0 + H) * sy)))
    return out


def red_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    return (r > 140) & (g < 90) & (b < 100) & (r - g > 80)


def glyphs(bgr: np.ndarray) -> list[np.ndarray]:
    """Красные цифры выреза слева направо, каждая — GLYPH, float 0..1."""
    m = red_mask(bgr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 1), np.uint8))
    cols = m.sum(0) > 0
    out, x = [], 0
    while x < len(cols):
        if not cols[x]:
            x += 1
            continue
        x1 = x
        while x1 < len(cols) and cols[x1]:
            x1 += 1
        part = m[:, x:x1]
        ys = np.nonzero(part.sum(1))[0]
        if x1 - x >= 2 and len(ys) and ys[-1] - ys[0] >= 8:   # отсечь крошки
            hgt = ys[-1] - ys[0] + 1
            # Две цифры иногда слипаются: шире ~0.8 высоты — режем по самому
            # тонкому столбцу в середине.
            if x1 - x > 0.8 * hgt:
                mid = part.sum(0)
                a, b = (x1 - x) // 3, 2 * (x1 - x) // 3
                cut = a + int(np.argmin(mid[a:b + 1]))
                pieces = [part[:, :cut], part[:, cut:]]
            else:
                pieces = [part]
            for pc in pieces:
                yy = np.nonzero(pc.sum(1))[0]
                xx = np.nonzero(pc.sum(0))[0]
                if len(yy) and len(xx):
                    g = pc[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1].astype(np.float32)
                    out.append(cv2.resize(g, GLYPH, interpolation=cv2.INTER_AREA))
        x = x1
    return out


class DigitReader:
    """Ближайший сосед по образцам цифр таймера."""

    def __init__(self, path: Path = DIGITS):
        self.ok = path.exists()
        if self.ok:
            z = np.load(path)
            self.x = z["glyphs"].reshape(len(z["glyphs"]), -1).astype(np.float32)
            self.y = z["digits"].astype(int)

    def read(self, bgr: np.ndarray) -> int | None:
        if not self.ok:
            return None
        gs = glyphs(bgr)
        if not gs or len(gs) > 2:
            return None
        val = 0
        for g in gs:
            d = ((self.x - g.reshape(1, -1)) ** 2).sum(1)
            val = val * 10 + int(self.y[int(d.argmin())])
        return val


def read_panels(frame_bgr: np.ndarray, reader: DigitReader | None = None) -> list[Slot]:
    """10 слотов кадра (BGR, как у cv2): серый ли портрет, таймер."""
    h, w = frame_bgr.shape[:2]
    sx, sy = w / REF_W, h / REF_H
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for s, (x0, y0, x1, y1) in enumerate(slot_boxes((w, h))):
        side, k = "LR"[s // 5], s % 5
        b = frame_bgr[y0:y1, x0:x1]
        red = red_mask(b)
        # JPEG размазывает цвет цифр на 2–3 px вокруг — их тоже не считаем.
        near = cv2.dilate(red.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        sat = hsv[y0:y1, x0:x1, 1][~near]
        sat_m = float(np.median(sat)) if sat.size else 0.0
        hx0, hx1 = HP_X_LEFT if side == "L" else HP_X_RIGHT
        hy = Y0 + PITCH * k
        hp = frame_bgr[round((hy + HP_Y[0]) * sy):round((hy + HP_Y[1]) * sy),
                       round(hx0 * sx):round(hx1 * sx)].astype(np.int16)
        hp_px = int(((hp[..., 1] > 110) & (hp[..., 1] > hp[..., 2] + 40)
                     & (hp[..., 1] > hp[..., 0] + 20)).sum())
        dead = sat_m < SAT_DEAD and int(red.sum()) >= RED_MIN and hp_px <= HP_EMPTY
        slot = Slot(side, k, round(sat_m, 1), int(red.sum()), hp_px, dead)
        if dead and reader is not None:
            dx0, dx1 = DIGIT_X_LEFT if side == "L" else DIGIT_X_RIGHT
            ty0 = Y0 + PITCH * k + DIGIT_Y[0] - 8
            box = frame_bgr[round(ty0 * sy):round((ty0 + DIGIT_Y[1]) * sy),
                            round(dx0 * sx):round(dx1 * sx)]
            if (w, h) != (REF_W, REF_H):
                box = cv2.resize(box, (dx1 - dx0, DIGIT_Y[1]), interpolation=cv2.INTER_AREA)
            slot.timer = reader.read(box)
        out.append(slot)
    return out


# Портреты слотов — базовые иконки Data Dragon, обрезанные по ширине (~28 px
# видимой ширины при ~40–50 px высоты). Ищем каждую иконку в области
# портрета нормированной корреляцией по нескольким масштабам; полоса
# шаблона по центру, ширина PORTRAIT_W. Замер на 6 роликах: верный
# чемпион с отрывом 0.1–0.4 от второго у 59 слотов из 60 (Kalista в
# zJvTSjEnKNE — отрыв 0, см. docs/minimap-champions.md).
PORTRAIT_X_LEFT, PORTRAIT_X_RIGHT = (27, 67), (1853, 1893)
PORTRAIT_Y0, PORTRAIT_H = 150, 60
PORTRAIT_W = 26
PORTRAIT_SCALES = tuple(range(30, 56, 3))


def portrait_templates(icons: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Иконки (RGB) → шаблоны BGR float32 по масштабам, средняя полоса."""
    out = {}
    for name, ic in icons.items():
        bgr = ic[..., ::-1].astype(np.float32)
        ts = []
        for s in PORTRAIT_SCALES:
            t = cv2.resize(bgr, (s, s), interpolation=cv2.INTER_AREA)
            w = min(s, PORTRAIT_W)
            ts.append(np.ascontiguousarray(t[:, (s - w) // 2:(s - w) // 2 + w]))
        out[name] = ts
    return out


def _portrait_regions(frame_bgr: np.ndarray) -> list[np.ndarray]:
    if frame_bgr.shape[:2] != (REF_H, REF_W):
        frame_bgr = cv2.resize(frame_bgr, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
    g = frame_bgr.astype(np.float32)
    out = []
    for s in range(10):
        x0, x1 = PORTRAIT_X_LEFT if s < 5 else PORTRAIT_X_RIGHT
        y0 = PORTRAIT_Y0 + PITCH * (s % 5)
        out.append(g[y0:y0 + PORTRAIT_H, x0:x1])
    return out


def _match(reg: np.ndarray, ts: list[np.ndarray]) -> float:
    return max(float(cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED).max()) for t in ts)


def portrait_scores(frame_bgr: np.ndarray, templates: dict[str, list[np.ndarray]],
                    names: list[str]) -> np.ndarray:
    """10×len(names): сходство портрета каждого слота (L1…R5) с каждым чемпионом.
    Кадр приводится к 1920×1080."""
    regs = _portrait_regions(frame_bgr)
    return np.array([[_match(r, templates[n]) for n in names] for r in regs], np.float32)


def portrait_own(frame_bgr: np.ndarray, templates: dict[str, list[np.ndarray]],
                 roster: list[str]) -> np.ndarray:
    """Сходство портрета каждого слота со своим чемпионом состава (10 чисел):
    низкое — панель скрыта или заслонена."""
    regs = _portrait_regions(frame_bgr)
    return np.array([_match(r, templates[n]) for r, n in zip(regs, roster)], np.float32)
