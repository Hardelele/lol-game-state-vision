"""Синтетика миникарты: иконки чемпионов Data Dragon на чистом фоне карты.

Зачем. Ручной разметки иконок нет, а иконка на миникарте — это базовая
иконка Data Dragon (без скинов) в кружке с кольцом цвета команды. Такую
картинку можно нарисовать сколько угодно раз с известными положениями,
командами и чемпионами — синтетика для детектора и классификатора
(lolvision/minimap_icons.py; AAAI-26 из docs/entity-model-speed.md: синтетика
плюс дообучение на реальных кадрах).

Фон — настоящая миникарта канала без иконок: медиана по кадрам одного
отрезка ролика (иконки и рамка камеры двигаются и в медиану не попадают,
tools/minimap_backgrounds.py). Поверх рисуется то, что на реальной
миникарте мешает: рамка камеры, точки миньонов, значки пингов и вардов,
цифры таймеров, полупрозрачные «призраки» иконок; потом размытие и двойное
JPEG-сжатие (YouTube + хранение кадров).

Геометрия иконки в сетке 256 (замер по кадру 58w57eJ5Qks_t00300, Jhin:
нормированная корреляция 0.97): внутренний круг ~15.5 px показывает
центральные ~80% иконки, кольцо ~2 px, всего ~20 px. Все размеры и цвета
рисуются со случайным разбросом вокруг этих значений.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MM = 256
SS = 4                        # суперсэмплинг при рисовании иконки
# Цвета кольца (RGB) — средние по реальным иконкам, разброс добавляется.
RING = {0: (40, 150, 215), 1: (205, 45, 60)}      # 0 — синие (ORDER), 1 — красные (CHAOS)
# Где на сетке 256 может быть центр иконки: обрезка раскладки захватывает
# край интерфейса слева и сверху (DRAW_X0/DRAW_Y0 в minimap_camera).
X_LO, Y_LO, HI = 11.0, 15.0, 254.0
MIN_VISIBLE = 0.45            # иконка закрыта сильнее — в разметке «не учитывать»
GHOST_ALPHA = 0.55            # полупрозрачнее — «призрак», тоже «не учитывать»


@dataclass
class Icon:
    x: float                  # центр в сетке 256 (пиксель i занимает [i, i+1))
    y: float
    team: int                 # 0 синие, 1 красные
    champ: int                # индекс в списке чемпионов
    visible: float = 1.0      # видимая доля круга
    ghost: bool = False

    @property
    def ignore(self) -> bool:
        return self.ghost or self.visible < MIN_VISIBLE


@dataclass
class Sample:
    image: np.ndarray                         # 256×256×3 uint8 RGB
    icons: list[Icon] = field(default_factory=list)


def icon_sprite(icon: np.ndarray, rng: np.random.Generator, team: int,
                frac: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Иконка в кружке с кольцом: (RGB float 0..255, альфа 0..1) в разрешении
    сетки 256; `frac` — дробная часть центра, чтобы попасть в субпиксель."""
    d_in = rng.uniform(14.5, 17.0)            # диаметр внутреннего круга
    zoom = rng.uniform(0.76, 0.92)            # видимая доля стороны иконки
    ring_w = rng.uniform(1.3, 2.4)
    outline = rng.uniform(0.0, 0.9)           # тёмная обводка снаружи кольца
    r_out = d_in / 2 + ring_w + outline
    n = int(np.ceil(r_out * 2)) + 3           # сторона спрайта в пикселях сетки
    N = n * SS
    c = icon.shape[0] / 2
    half = zoom * icon.shape[0] / 2
    crop = icon[int(c - half):int(np.ceil(c + half)), int(c - half):int(np.ceil(c + half))]
    D = int(round(d_in * SS))
    face = cv2.resize(crop, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Центр спрайта со сдвигом на дробную часть координаты.
    # Спрайт ложится левым верхним углом в floor(x) - n//2 (synth), поэтому
    # центр в координатах спрайта — n//2 + дробная часть.
    cx = (n // 2 + frac[0]) * SS
    cy = (n // 2 + frac[1]) * SS
    yy, xx = np.mgrid[:N, :N].astype(np.float32)
    r = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / SS
    rgb = np.zeros((N, N, 3), np.float32)
    x0, y0 = int(round(cx - D / 2)), int(round(cy - D / 2))
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(N, x0 + D), min(N, y0 + D)
    rgb[ya:yb, xa:xb] = face[ya - y0:yb - y0, xa - x0:xb - x0]
    base = np.array(RING[team], np.float32)
    ring_rgb = np.clip(base * rng.uniform(0.75, 1.2) + rng.normal(0, 12, 3), 0, 255)
    # Кольцо светлее у внутреннего края — как блик на реальных иконках.
    t = np.clip((r - d_in / 2) / ring_w, 0, 1)[..., None]
    ring_col = ring_rgb * (1.15 - 0.3 * t)
    in_ring = (r > d_in / 2) & (r <= d_in / 2 + ring_w)
    rgb[in_ring] = np.clip(ring_col, 0, 255)[in_ring]
    rgb[r > d_in / 2 + ring_w] = rng.uniform(0, 40)
    alpha = np.clip(r_out + 0.5 - r, 0, 1)    # мягкий край круга
    rgb = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (n, n), interpolation=cv2.INTER_AREA)
    return rgb, alpha


def paste(canvas: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x0: int, y0: int,
          a: float = 1.0) -> None:
    """Наложить спрайт с левым верхним углом (x0, y0) и непрозрачностью a."""
    H, W = canvas.shape[:2]
    h, w = alpha.shape
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    al = alpha[ya - y0:yb - y0, xa - x0:xb - x0] * a
    canvas[ya:yb, xa:xb] = (canvas[ya:yb, xa:xb] * (1 - al[..., None])
                            + rgb[ya - y0:yb - y0, xa - x0:xb - x0] * al[..., None])


def _positions(rng: np.random.Generator, k: int) -> list[tuple[float, float]]:
    """Центры иконок: часть — кучей (бой, общий поход), часть — поодиночке."""
    pts: list[tuple[float, float]] = []
    while len(pts) < k:
        if rng.random() < 0.5 and k - len(pts) >= 2:
            m = int(rng.integers(2, min(6, k - len(pts)) + 1))
            cx, cy = rng.uniform(X_LO + 6, HI - 6), rng.uniform(Y_LO + 6, HI - 6)
            spread = rng.uniform(3, 12)
            for _ in range(m):
                pts.append((float(np.clip(cx + rng.normal(0, spread), X_LO, HI)),
                            float(np.clip(cy + rng.normal(0, spread), Y_LO, HI))))
        else:
            pts.append((rng.uniform(X_LO, HI), rng.uniform(Y_LO, HI)))
    return pts[:k]


def _distractors(img: np.ndarray, rng: np.random.Generator) -> None:
    """То, что на реальной миникарте похоже на иконку или её закрывает."""
    # Точки миньонов: цепочки маленьких кружков цвета команды.
    for _ in range(int(rng.integers(0, 8))):
        col = RING[int(rng.integers(0, 2))]
        x, y = rng.uniform(10, 250), rng.uniform(14, 250)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(2, 8))):
            x += 4 * np.cos(ang) + rng.normal(0, 0.6)
            y += 4 * np.sin(ang) + rng.normal(0, 0.6)
            cv2.circle(img, (int(x * 4), int(y * 4)), int(rng.uniform(1.6, 2.6) * 4),
                       tuple(float(v) for v in np.clip(np.array(col) * rng.uniform(0.8, 1.3), 0, 255)),
                       -1, cv2.LINE_AA, shift=2)
    # Значки вардов, пингов, лагерей: мелкие фигуры разных цветов.
    for _ in range(int(rng.integers(0, 10))):
        col = tuple(float(v) for v in rng.choice(
            [(230, 200, 60), (240, 240, 240), (210, 50, 50), (60, 170, 230), (150, 90, 200),
             (90, 200, 90)]))
        x, y, s = rng.uniform(8, 250), rng.uniform(12, 250), rng.uniform(2.5, 7)
        kind = rng.integers(0, 4)
        if kind == 0:
            cv2.circle(img, (int(x * 4), int(y * 4)), int(s * 4), col, -1, cv2.LINE_AA, shift=2)
        elif kind == 1:
            cv2.circle(img, (int(x * 4), int(y * 4)), int(s * 4), col, 1 + int(rng.integers(0, 2)),
                       cv2.LINE_AA, shift=2)
        elif kind == 2:
            p = np.array([[x, y - s], [x - s, y + s * 0.8], [x + s, y + s * 0.8]]) * 4
            cv2.fillPoly(img, [p.astype(np.int32)], col, cv2.LINE_AA, shift=2)
        else:
            p = np.array([[x, y - s], [x - s, y], [x, y + s], [x + s, y]]) * 4
            cv2.fillPoly(img, [p.astype(np.int32)], col, cv2.LINE_AA, shift=2)
    # Таймеры лагерей: белые цифры.
    for _ in range(int(rng.integers(0, 4))):
        txt = rng.choice(["1:34", "2m+", "24", "1:09", "59", "8", "41", "1:5"])
        cv2.putText(img, str(txt), (int(rng.uniform(5, 220)), int(rng.uniform(20, 250))),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.35, 0.5),
                    (235, 235, 235) if rng.random() < 0.85 else (230, 210, 60), 1, cv2.LINE_AA)


def _camera_box(img: np.ndarray, rng: np.random.Generator) -> None:
    bw = int(rng.uniform(60, 76))
    bh = int(round(bw * 9 / 16))
    x, y = int(rng.uniform(-20, MM - bw + 20)), int(rng.uniform(-10, MM - bh + 10))
    col = float(rng.uniform(200, 255))
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (col, col, col), int(rng.integers(1, 3)))


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Размытие, лёгкое пересэмплирование и JPEG — как у кадра с YouTube."""
    out = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.8))
    if rng.random() < 0.5:
        s = rng.uniform(0.92, 1.05)
        small = cv2.resize(out, (int(MM * s), int(MM * s)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (MM, MM), interpolation=cv2.INTER_LINEAR)
    for _ in range(int(rng.integers(1, 3))):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 92))])
        out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out


def synth(bg: np.ndarray, icons: list[np.ndarray], rng: np.random.Generator,
          k: int | None = None) -> Sample:
    """Одна синтетическая миникарта на фоне `bg` (256×256×3 uint8 RGB).

    `icons` — иконки Data Dragon (RGB uint8), индекс — номер чемпиона.
    Иконки рисуются в случайном порядке поверх друг друга; видимая доля
    каждой считается по маскам тех, что легли сверху.
    """
    img = bg.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 3, 3)
    _distractors(img, rng)
    if k is None:
        k = int(rng.choice(np.arange(0, 11), p=[.04, .08, .1, .12, .12, .12, .11, .1, .08, .07, .06]))
    champs = rng.choice(len(icons), size=k, replace=False)
    out: list[Icon] = []
    shapes: list[tuple[int, int, np.ndarray]] = []    # (x0, y0, маска круга) для видимости
    for (x, y), ch in zip(_positions(rng, k), champs):
        team = int(rng.integers(0, 2))
        rgb, al = icon_sprite(icons[int(ch)], rng, team, (x % 1, y % 1))
        n = al.shape[0]
        x0, y0 = int(np.floor(x)) - n // 2, int(np.floor(y)) - n // 2
        ghost = rng.random() < 0.06
        paste(img, rgb, al, x0, y0, rng.uniform(0.15, 0.5) if ghost else 1.0)
        out.append(Icon(x, y, team, int(ch), ghost=ghost))
        shapes.append((x0, y0, al > 0.5))
    # Видимая доля: пиксели круга внутри холста, не закрытые кругами сверху;
    # делится на весь круг, поэтому иконка за краем тоже «закрыта».
    for i, (x0, y0, m) in enumerate(shapes):
        free = m.copy()
        n = m.shape[0]
        for x1, y1, m1 in shapes[i + 1:]:
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) >= n or abs(dy) >= n:
                continue
            n1 = m1.shape[0]
            ax, ay = max(0, dx), max(0, dy)
            bx, by = min(n, dx + n1), min(n, dy + n1)
            if ax < bx and ay < by:
                free[ay:by, ax:bx] &= ~m1[ay - dy:by - dy, ax - dx:bx - dx]
        yy, xx = np.mgrid[y0:y0 + n, x0:x0 + n]
        free &= (xx >= 0) & (xx < MM) & (yy >= 0) & (yy < MM)
        out[i].visible = float(free.sum()) / max(1, int(m.sum()))
    if rng.random() < 0.9:
        _camera_box(img, rng)
    return Sample(degrade(img, rng), out)
