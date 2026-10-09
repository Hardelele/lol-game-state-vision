"""Шум меток с миникарты: где правда, а где искажение, и сколько это в единицах.

Зачем. Метка камеры — центр белой рамки на миникарте того же кадра
(minimap_camera.py), найденный перебором целых положений на миникарте,
приведённой к 256×256. Шаг метки — 1/256 карты, около 58 ед., а медианная
ошибка патчевой модели на отложенных матчах — около 30 ед. Значит,
заголовочная цифра может мерить разметку, а не модель. Чтобы это разделить,
нужен сигнал о движении камеры, который миникарты не видит.

Как. Несколько независимых сигналов, а не мнение модели против метки.

* Сдвиг сцены (`register`). Кадр переводится в вид сверху с постоянным
  масштабом (rectify, 4 ед./px), и сдвиг между соседними холстами — это
  прямо приращение положения камеры. Сдвиг ищется маскированной нормированной
  корреляцией по всем сдвигам сразу (Padfield, 2012): закрытые маской зоны не
  участвуют, а площадь земли перевешивает чемпиона, за которым едет камера
  (он стоит в центре и тянет к нулевому сдвигу — поиск по особым точкам на
  этом ломался). Субпиксель — парабола по NCC вокруг пика на полном
  разрешении. Каждая пара меряется дважды: i→i+1 и i→i+2, и треугольник
  i→i+1→i+2 обязан замкнуться — так отсеиваются прыжки камеры и ложные
  совпадения без всякой оглядки на метку.
* Траектория (`analyze`). Надёжные приращения связывают кадры в цепочки;
  положения кадров цепочки подбираются наименьшими квадратами по двум видам
  уравнений: «разность соседей = приращение по сцене» (точно, но с
  неизвестным началом) и «положение = метка» (грубо, но без дрейфа), с
  устойчивыми весами меток. Остаток метки берётся «без себя» (leave-one-out
  через диагональ матрицы влияния), так что метка не подтверждает сама себя.
* Сама миникарта (`minimap`). Субпиксельные края рамки по профилям яркости
  поперёк каждого ребра — на сохранённых миникартах 256 и на полных кадрах
  1080p из dataset/videos; размер рамки по кадрам; смещение центра рамки
  относительно следа кадра по гомографии.
* Ответы модели (`predict`) — по всем кадрам отложенных роликов.

Масштаб сцены. Гомография dataset/layouts/projection.json подобрана по тем
же меткам, но, как оказалось, расходится с ними по масштабу на 10-17%
(см. docs/label-noise.md). Глобальный масштаб и анизотропию одна сцена не
определяет (в виде сверху камера только переносится), поэтому приращения
сцены переводятся в единицы меток матрицей 2×2, подобранной регрессией
приращений метки на приращения сцены (шум только в метке — оценка не
смещена).

Все выходы — в runs/label_noise/: reg/<ролик>.csv, mini/<ролик>.csv,
pred/<ролик>.csv, frames/<ролик>.csv, summary.json, native_frames.csv.

Пример (всё целиком, ~30 мин на 20 роликах; GPU нужен только для predict):
    python tools/label_noise.py register --all --workers 10
    python tools/label_noise.py minimap --all
    python tools/label_noise.py predict 58w57eJ5Qks olmTXkkUv58 5nBGNokOUDA 4AIX8QtRid4
    python tools/label_noise.py analyze --all
    python tools/label_noise_page.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from numpy.fft import irfft2, rfft2
from PIL import Image

from paths import COORDS_DATA, DATASET, DATASET_VIDEOS, RUNS
from mask_frames import crop_box, keep_grid
from minimap_camera import MM, WHITE_V, estimate_box, find_box, minimap, white_mask
from patch_model import MAP_UNITS, load_projection, rectify

OUT = RUNS / "label_noise"
LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
MODEL = RUNS / "patches" / "cnn-split16-notgame" / "model.pt"
HELD_OUT = ("58w57eJ5Qks", "olmTXkkUv58", "5nBGNokOUDA", "4AIX8QtRid4")
STORE = (720, 372)          # хранимая сцена плотного набора
FULL = (1920, 1080)

U_REG = 4.0                 # ед./px холста для регистрации
GAPS = (1, 2)               # пары i→i+1 и i→i+2: второе — для замыкания
MIN_OVERLAP = 0.25          # доля видимой области холста, общая у пары
PX = MAP_UNITS / MM         # шаг метки: один пиксель миникарты 256, ед.


def all_videos() -> list[str]:
    return sorted(p.stem for p in COORDS_DATA.glob("*.csv"))


def read_rows(vid: str) -> list[dict]:
    return list(csv.DictReader((COORDS_DATA / f"{vid}.csv").open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


# ------------------------------------------------------------ регистрация

def stored_keep(layout: dict) -> np.ndarray:
    """Маска «здесь сцена» в пикселях хранимой сцены (uint8 0/255)."""
    k = keep_grid(FULL, layout)
    x0, y0, x1, y1 = crop_box(FULL, layout, "dense")
    return cv2.resize(k[y0:y1, x0:x1].astype(np.uint8) * 255, STORE,
                      interpolation=cv2.INTER_NEAREST)


class Canvas:
    """Вид сверху для регистрации: общая геометрия всех кадров ролика.

    Видимая область эродируется: на границе маски JPEG и пересэмплинг дают
    полосу смешанных пикселей, неподвижную относительно экрана, — она тянет
    к нулевому сдвигу.
    """

    def __init__(self, layout: dict):
        self.h0, store = load_projection()
        assert tuple(store) == STORE, store
        k = cv2.erode(stored_keep(layout), np.ones((7, 7), np.uint8))
        kc, _, self.q0 = rectify(np.dstack([k] * 3), np.eye(3), self.h0, U_REG, 16)
        self.valid = cv2.erode((kc[..., 0] > 250).astype(np.uint8),
                               np.ones((5, 5), np.uint8))
        h, w = self.valid.shape
        self.valid2 = cv2.resize(self.valid, (w // 2, h // 2),
                                 interpolation=cv2.INTER_NEAREST).astype(np.float64)

    def load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Хранимая сцена → (холст float32 со сглаживанием, он же вдвое мельче)."""
        return self.from_gray(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    def from_gray(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cv, _, _ = rectify(np.dstack([g] * 3), np.eye(3), self.h0, U_REG, 16)
        c = cv2.GaussianBlur(cv[..., 0].astype(np.float32), (0, 0), 1.0)
        h, w = self.valid2.shape
        c2 = cv2.resize(cv2.GaussianBlur(c, (0, 0), 1.0), (w, h),
                        interpolation=cv2.INTER_AREA).astype(np.float64)
        return c, c2


def masked_ncc(a, ma, b, mb, min_overlap: float):
    """NCC между a(x) и b(x+t) по общей видимой области — для всех t сразу.

    Padfield, «Masked object registration in the Fourier domain» (2012): все
    суммы по пересечению масок — свёртки, поэтому считаются через FFT. Ответ
    — карта C[ty+H-1, tx+W-1]; сдвиги с малым пересечением получают -1.
    """
    H, W = a.shape
    sh = (2 * H - 1, 2 * W - 1)
    fs = [int(2 ** math.ceil(math.log2(s))) for s in sh]
    a, b = a * ma, b * mb

    def corr(x, y):
        r = irfft2(np.conj(x) * y, fs)
        return np.roll(r, (H - 1, W - 1), axis=(0, 1))[:sh[0], :sh[1]]

    fa, fma, fb, fmb = rfft2(a, fs), rfft2(ma, fs), rfft2(b, fs), rfft2(mb, fs)
    n = np.maximum(np.round(corr(fma, fmb)), 0)
    sa, sb = corr(fa, fmb), corr(fma, fb)
    sab = corr(fa, fb)
    sa2, sb2 = corr(rfft2(a * a, fs), fmb), corr(fma, rfft2(b * b, fs))
    nn = np.maximum(n, 1)
    da = np.maximum(sa2 - sa * sa / nn, 0)
    db = np.maximum(sb2 - sb * sb / nn, 0)
    c = (sab - sa * sb / nn) / np.sqrt(da * db + 1e-6)
    bad = (n < min_overlap) | (da < 1e-2 * nn) | (db < 1e-2 * nn)
    c[bad] = -1
    return np.clip(c, -1, 1), n


def ncc_shift(a, b, m, t):
    """NCC a(x) и b(x+t) для целого t по общей маске m (срезами, без warp)."""
    H, W = a.shape
    tx, ty = int(t[0]), int(t[1])
    ys, ye = max(0, -ty), min(H, H - ty)
    xs, xe = max(0, -tx), min(W, W - tx)
    if ye - ys < 8 or xe - xs < 8:
        return -1.0
    aa = a[ys:ye, xs:xe]
    bb = b[ys + ty:ye + ty, xs + tx:xe + tx]
    mm = m[ys:ye, xs:xe] & m[ys + ty:ye + ty, xs + tx:xe + tx]
    if mm.sum() < 500:
        return -1.0
    x, y = aa[mm].astype(np.float64), bb[mm].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float((x * y).sum() / (math.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def register_pair(cv: Canvas, A, B) -> dict:
    """Сдвиг камеры между двумя кадрами по сцене, ед., и его качество.

    Знак: если земля в кадре B стоит там, где в A была на t пикселей дальше
    (b(x+t) = a(x)), камера сдвинулась на −t·U.
    """
    a, a2 = A
    b, b2 = B
    v2 = cv.valid2
    c, n = masked_ncc(a2, v2, b2, v2, MIN_OVERLAP * v2.sum())
    H2, W2 = v2.shape
    k = np.unravel_index(int(c.argmax()), c.shape)
    peak = float(c[k])
    # Второй по силе пик вне окрестности первого: близкий соперник —
    # признак неоднозначного совпадения (повторяющаяся текстура, туман).
    c2 = c.copy()
    c2[max(0, k[0] - 8):k[0] + 9, max(0, k[1] - 8):k[1] + 9] = -1
    second = float(c2.max())
    zero = float(c[H2 - 1, W2 - 1])
    t0 = np.array([(k[1] - (W2 - 1)) * 2, (k[0] - (H2 - 1)) * 2])
    overlap = float(n[k] / v2.sum())
    # Уточнение на полном разрешении: сетка ±3 px вокруг грубого пика, затем
    # парабола по 3×3 вокруг лучшего целого.
    m = cv.valid > 0
    best, bt = -2.0, t0
    grid = {}
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            t = t0 + (dx, dy)
            s = ncc_shift(a, b, m, t)
            grid[(dx, dy)] = s
            if s > best:
                best, bt = s, t
    d = bt - t0
    sub = np.zeros(2)
    if abs(d[0]) < 3 and abs(d[1]) < 3:
        g = lambda x, y: grid[(int(d[0]) + x, int(d[1]) + y)]
        for ax in range(2):
            lo = g(-1, 0) if ax == 0 else g(0, -1)
            hi = g(1, 0) if ax == 0 else g(0, 1)
            den = lo - 2 * best + hi
            if den < 0:
                sub[ax] = float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))
    t = bt + sub
    return {"dx": round(float(-t[0] * U_REG), 2), "dy": round(float(-t[1] * U_REG), 2),
            "ncc": round(best, 4), "peak": round(peak, 4), "second": round(second, 4),
            "zero": round(zero, 4), "overlap": round(overlap, 3)}


def register_video(vid: str) -> str:
    """Все пары i→i+g ролика → runs/label_noise/reg/<ролик>.csv."""
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    cv = Canvas(layout)
    rows = read_rows(vid)
    cache: dict[int, tuple] = {}
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, i + 1 + max(GAPS)):
            if j >= len(rows) or (j - i) not in GAPS:
                continue
            for k in (i, j):
                if k not in cache:
                    cache[k] = cv.load(COORDS_DATA / rows[k]["scene"])
            r = register_pair(cv, cache[i], cache[j])
            out.append({"i": i, "j": j, **r})
        cache.pop(i - 1, None)
    write_csv(OUT / "reg" / f"{vid}.csv", out)
    return f"{vid}: пар {len(out)}"


# ------------------------------------------------------------ миникарта

def whiteness(rgb: np.ndarray) -> np.ndarray:
    """Насколько пиксель похож на линию рамки: светлый и несатурированный.

    В отличие от бинарной маски white_mask, непрерывна — по ней можно искать
    край с точностью лучше пикселя.
    """
    hi, lo = rgb.max(2), rgb.min(2)
    sat = np.clip(1 - (hi - lo - 10) / 40, 0, 1)
    return np.clip((lo - 90) / 120, 0, 1) * sat


def edge_profile(w: np.ndarray, along: slice, at: float, axis: int, r: int = 3):
    """Субпиксельное положение ребра рамки: профиль поперёк ребра.

    axis=1 — вертикальное ребро (ищем x), строки `along`; axis=0 —
    горизонтальное (ищем y), столбцы `along`. Строки (столбцы), где ребро не
    видно (иконка чемпиона, пинг, край карты), выбрасываются: берутся только
    те, где пик поперёк ребра светлый и лежит рядом с ожидаемым местом.
    Ответ — (положение, число строк в профиле) или (nan, 0).
    """
    c = int(round(at))
    lim = w.shape[1] if axis == 1 else w.shape[0]
    if c - r - 1 < 0 or c + r + 2 > lim:
        return float("nan"), 0
    seg = (w[along, c - r:c + r + 1] if axis == 1 else w[c - r:c + r + 1, along].T)
    pk = seg.max(1)
    am = seg.argmax(1)
    keep = (pk > 0.35) & (np.abs(am - r) <= 2)
    if keep.sum() < 5:
        return float("nan"), int(keep.sum())
    prof = seg[keep].mean(0)
    prof = prof - prof.min()
    k = int(prof.argmax())
    if k == 0 or k == len(prof) - 1:
        return float(c - r + k), int(keep.sum())
    lo, mid, hi = prof[k - 1], prof[k], prof[k + 1]
    den = lo - 2 * mid + hi
    off = 0.5 * (lo - hi) / den if den < 0 else 0.0
    return float(c - r + k + np.clip(off, -0.5, 0.5)), int(keep.sum())


def subpixel_box(rgb: np.ndarray, x: int, y: int, bw: int, bh: int,
                 scale=(1.0, 1.0)) -> dict:
    """Края рамки с точностью лучше пикселя.

    (x, y, bw, bh) — целочисленная рамка find_box в сетке 256; `scale` —
    во сколько раз `rgb` крупнее этой сетки по x и y (1080p: ~1.02/1.03).
    Края ищутся в пикселях `rgb` и возвращаются в индексах сетки 256 в той
    же условности, что у minimap_camera: ребро в столбце x, центр
    x + bw/2. Отсутствующее ребро (закрыто, за краем) — nan.
    """
    sx, sy = scale
    w = whiteness(rgb.astype(np.float32))
    # Ребро лежит в столбце x сетки 256, его центр в непрерывных координатах
    # x + 0.5; в пикселях rgb это (x + 0.5)·s − 0.5.
    to_px = lambda e, s: (e + 0.5) * s - 0.5
    to_mm = lambda p, s: (p + 0.5) / s - 0.5
    ys = slice(int(to_px(y + 2, sy)), int(to_px(y + bh - 2, sy)) + 1)
    xs = slice(int(to_px(x + 2, sx)), int(to_px(x + bw - 2, sx)) + 1)
    l, nl = edge_profile(w, ys, to_px(x, sx), 1)
    r_, nr = edge_profile(w, ys, to_px(x + bw, sx), 1)
    t, nt = edge_profile(w, xs, to_px(y, sy), 0)
    b, nb = edge_profile(w, xs, to_px(y + bh, sy), 0)
    return {"left": to_mm(l, sx), "right": to_mm(r_, sx),
            "top": to_mm(t, sy), "bottom": to_mm(b, sy),
            "n_left": nl, "n_right": nr, "n_top": nt, "n_bottom": nb}


def box_center(e: dict, bw_med: float, bh_med: float) -> tuple[float, float]:
    """Центр рамки по найденным рёбрам; одно пропавшее ребро — по ширине ролика."""
    def one(lo, hi, size):
        if np.isfinite(lo) and np.isfinite(hi):
            return (lo + hi) / 2
        if np.isfinite(lo):
            return lo + size / 2
        if np.isfinite(hi):
            return hi - size / 2
        return float("nan")
    return one(e["left"], e["right"], bw_med), one(e["top"], e["bottom"], bh_med)


def find_box_clipped(mask: np.ndarray, bw: int, bh: int, min_vis: float = 0.5
                     ) -> tuple[int, int, float, bool]:
    """Как find_box, но рамка может выходить за край миникарты.

    Когда камера у края карты (база, дальний край верхней или нижней линии),
    игра рисует рамку обрезанной краем миникарты. find_box перебирает только
    положения целиком внутри, поэтому такую рамку либо прижимает к краю
    (метка цензурирована: истинный центр дальше), либо, если у края есть
    другие светлые линии, переносит на несколько пикселей вглубь карты —
    грубая ошибка в сотни единиц (см. docs/label-noise.md). Здесь маска
    дополняется нулями на полрамки с каждой стороны, а «белизна» периметра
    делится на длину его видимой части (не меньше min_vis периметра).
    Ответ — (x, y, качество, обрезана ли рамка) в координатах исходной
    миникарты: x, y могут быть отрицательными или > n - bw - 1.
    """
    n = mask.shape[0]
    px, py = bw // 2, bh // 2
    pm = np.pad(mask, ((py, py + 1), (px, px + 1)))
    inside = np.pad(np.ones_like(mask), ((py, py + 1), (px, px + 1)))

    def scores(m):
        N, M = m.shape
        cx = np.cumsum(np.pad(m, ((0, 0), (1, 0))), axis=1)
        cy = np.cumsum(np.pad(m, ((1, 0), (0, 0))), axis=0)
        xs, ys = M - bw - 1, N - bh - 1
        hseg = cx[:, bw:bw + xs + 1] - cx[:, :xs + 1]
        vseg = cy[bh:bh + ys + 1, :] - cy[:ys + 1, :]
        return (hseg[:ys + 1, :] + hseg[bh:bh + ys + 1, :]
                + vseg[:, :xs + 1] + vseg[:, bw:bw + xs + 1])

    sc, vis = scores(pm), scores(inside)
    per = 2 * (bw + bh)
    norm = np.where(vis >= min_vis * per, sc / np.maximum(vis, 1), -1)
    y, x = np.unravel_index(int(norm.argmax()), norm.shape)
    x0, y0 = int(x) - px, int(y) - py
    clipped = bool(vis[y, x] < per)
    return x0, y0, float(norm[y, x]), clipped


def minimap_video(vid: str) -> str:
    """Субпиксельная рамка по сохранённым миникартам 256 плотного набора.

    Размер рамки и целые положения пересчитываются так же, как в
    build_coords (по маске той же миникарты); сохранённая mini/ — JPEG q82,
    поэтому совпадение с меткой манифеста проверяется и пишется.
    """
    rows = read_rows(vid)
    imgs, masks = [], []
    for r in rows:
        a = np.asarray(Image.open(COORDS_DATA / r["mini"]).convert("RGB"), np.float32)
        imgs.append(a)
        masks.append(white_mask(a))
    step = max(1, len(masks) // 24)
    bw, bh = estimate_box(masks[::step])
    found = []
    for r, a, m in zip(rows, imgs, masks):
        x, y, q = find_box(m, bw, bh)
        kx, ky, kq, clip = find_box_clipped(m, bw, bh)
        # Субпиксель уточняет рамку с учётом обрезки: рёбра за краем
        # миникарты не ищутся, центр — по видимому ребру и ширине ролика.
        e = subpixel_box(a, kx, ky, bw, bh)
        found.append((x, y, q, e, kx, ky, kq, clip))
    ws = [e["right"] - e["left"] for _, _, q, e, *_ in found
          if q >= 0.4 and np.isfinite(e["right"] - e["left"])]
    hs = [e["bottom"] - e["top"] for _, _, q, e, *_ in found
          if q >= 0.4 and np.isfinite(e["bottom"] - e["top"])]
    bw_med = float(np.median(ws)) if ws else float(bw)
    bh_med = float(np.median(hs)) if hs else float(bh)
    out = []
    for r, (x, y, q, e, kx, ky, kq, clip) in zip(rows, found):
        cx, cy = box_center(e, bw_med, bh_med)
        out.append({"idx": r["idx"], "bw": bw, "bh": bh,
                    "x": x, "y": y, "q": round(q, 3),
                    # прижата ли метка манифеста к краю (цензурирована)
                    "at_bound": int(x in (0, MM - bw - 1) or y in (0, MM - bh - 1)),
                    "kx": round((kx + bw / 2) / MM, 5), "ky": round((ky + bh / 2) / MM, 5),
                    "kq": round(kq, 3), "clipped": int(clip),
                    "same_as_label": int(abs((x + bw / 2) / MM - float(r["cx"])) < 1e-3
                                         and abs((y + bh / 2) / MM - float(r["cy"])) < 1e-3),
                    "sx": round(cx / MM, 5) if np.isfinite(cx) else "",
                    "sy": round(cy / MM, 5) if np.isfinite(cy) else "",
                    "w": round(e["right"] - e["left"], 3) if np.isfinite(e["right"] - e["left"]) else "",
                    "h": round(e["bottom"] - e["top"], 3) if np.isfinite(e["bottom"] - e["top"]) else "",
                    "edges": sum(np.isfinite(e[k]) for k in ("left", "right", "top", "bottom"))})
    write_csv(OUT / "mini" / f"{vid}.csv", out)
    same = np.mean([o["same_as_label"] for o in out])
    return (f"{vid}: рамка {bw}x{bh}, субпиксельная ширина {bw_med:.2f}, "
            f"высота {bh_med:.2f}; целое положение совпало с манифестом у {same:.1%}")


def native_frames(layout: dict) -> list[dict]:
    """Полные кадры 1080p dataset/videos: рамка в нативном разрешении.

    Для каждого кадра: (а) как в build_coords — LANCZOS до 256, целая рамка;
    (б) субпиксель по той же 256 без JPEG; (в) субпиксель по нативной
    миникарте (~261×263 px). Сравнение (б) и (в) показывает, что теряется
    при приведении к 256, сравнение (а) и (в) — полный шум квантования.
    Размер рамки — по кадрам того же ролика.
    """
    out = []
    for vd in sorted(p for p in DATASET_VIDEOS.iterdir() if (p / "frames").is_dir()):
        rows = list(csv.DictReader((vd / "frames.csv").open(encoding="utf-8")))
        data = []
        for r in rows:
            f = vd / r["frame"]
            if not f.exists():
                continue
            with Image.open(f) as img:
                img = img.convert("RGB")
                x0, y0, x1, y1 = layout["labelling"]["minimap"]
                W, H = img.size
                box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
                nat = np.asarray(img.crop(box), np.float32)
                mm = minimap(img, layout)
            data.append((r, nat, mm, white_mask(mm)))
        if not data:
            continue
        bw, bh = estimate_box([d[3] for d in data])
        for r, nat, mm, m in data:
            x, y, q = find_box(m, bw, bh)
            if q < 0.4:
                continue
            e256 = subpixel_box(mm, x, y, bw, bh)
            sc = (nat.shape[1] / MM, nat.shape[0] / MM)
            enat = subpixel_box(nat, x, y, bw, bh, sc)
            c256 = box_center(e256, bw, bh)
            cnat = box_center(enat, bw, bh)
            out.append({"video_id": vd.name, "frame": Path(r["frame"]).stem,
                        "q": round(q, 3), "bw": bw, "bh": bh,
                        "lx": (x + bw / 2) / MM, "ly": (y + bh / 2) / MM,
                        "sx": c256[0] / MM, "sy": c256[1] / MM,
                        "nx": cnat[0] / MM, "ny": cnat[1] / MM,
                        "w256": e256["right"] - e256["left"],
                        "wnat": enat["right"] - enat["left"],
                        "h256": e256["bottom"] - e256["top"],
                        "hnat": enat["bottom"] - enat["top"]})
    return out


# ------------------------------------------------------------ плотный кусок ролика

def video_segment(vid: str, start: float, dur: float, fps: float) -> Path:
    """Кусок локального mp4 с частым шагом: нативная рамка и сдвиг сцены.

    Плотный набор снят раз в секунду — по нему не видно, отстаёт ли рамка
    миникарты от сцены на кадр-другой и сколько шума остаётся у рамки в
    нативном разрешении. Здесь каждый кадр куска (шаг 1/fps): метка как в
    build_coords (LANCZOS до 256, целая рамка), субпиксельная рамка по
    нативной миникарте 1080p, сдвиг сцены от предыдущего кадра (сцена
    собирается так же, как хранимая: маска, плотная обрезка, 720 px).
    """
    from build_coords import frames as frame_stream
    from mask_frames import mask_boxes
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    mp4 = COORDS_DATA.parent / "videos" / f"{vid}.mp4"
    if not mp4.exists():
        raise SystemExit(f"нет {mp4}")
    m0 = read_csv(OUT / "mini" / f"{vid}.csv")[0]
    bw, bh = int(m0["bw"]), int(m0["bh"])
    cv = Canvas(layout)
    x0, y0, x1, y1 = layout["labelling"]["minimap"]
    mmbox = (int(x0 * 1920), int(y0 * 1080), int(x1 * 1920), int(y1 * 1080))
    cx0, cy0, cx1, cy1 = crop_box(FULL, layout, "dense")
    boxes = mask_boxes(FULL, layout)
    sc = ((mmbox[2] - mmbox[0]) / MM, (mmbox[3] - mmbox[1]) / MM)
    out, prev = [], None
    for k, fr in enumerate(frame_stream(mp4, 1920, 1080, fps, skip=start)):
        if k >= int(dur * fps):
            break
        img = Image.fromarray(fr)
        nat = fr[mmbox[1]:mmbox[3], mmbox[0]:mmbox[2]].astype(np.float32)
        mm = np.asarray(img.crop(mmbox).resize((MM, MM), Image.LANCZOS), np.float32)
        x, y, q = find_box(white_mask(mm), bw, bh)
        e = subpixel_box(nat, x, y, bw, bh, sc)
        ncx, ncy = box_center(e, bw, bh)
        e2 = subpixel_box(mm, x, y, bw, bh)
        scx, scy = box_center(e2, bw, bh)
        scene = img.copy()
        for b in boxes:
            scene.paste((0, 0, 0), b)
        g = np.asarray(scene.crop((cx0, cy0, cx1, cy1)).resize(STORE, Image.LANCZOS).convert("L"))
        cur = cv.from_gray(g)
        r = register_pair(cv, prev, cur) if prev is not None else {}
        prev = cur
        out.append({"k": k, "t": round(start + k / fps, 4), "q": round(q, 3),
                    "lx": (x + bw / 2) / MM * MAP_UNITS, "ly": (y + bh / 2) / MM * MAP_UNITS,
                    "nx": ncx / MM * MAP_UNITS, "ny": ncy / MM * MAP_UNITS,
                    "sx": scx / MM * MAP_UNITS, "sy": scy / MM * MAP_UNITS,
                    "dx": r.get("dx", ""), "dy": r.get("dy", ""), "ncc": r.get("ncc", ""),
                    "peak": r.get("peak", ""), "second": r.get("second", "")})
    p = OUT / "video" / f"{vid}_{int(start)}_{int(dur)}s_{fps:g}fps.csv"
    write_csv(p, out)
    return p


def lag_report(path: Path, A: np.ndarray, max_lag: int = 12) -> dict:
    """Отставание рамки от сцены и шум рамки на плотном куске.

    Скорость камеры по сцене (A·сдвиг) и по рамке сглаживаются окном и
    сравниваются со сдвигом по времени: лучший сдвиг — отставание рамки в
    кадрах куска. Затем траектория по сцене привязывается к рамке (одно
    смещение на каждый непрерывный отрезок) и считается остаток трёх меток:
    целой (как в датасете), субпиксельной по 256 и субпиксельной нативной.
    """
    rows = read_csv(path)
    n = len(rows)
    t = np.array([float(r["t"]) for r in rows])
    fps = round(1 / np.median(np.diff(t)), 3)
    win = int(round(fps))                     # ±1 с
    get = lambda a, b: np.array([[f_(r[a]), f_(r[b])] for r in rows])
    L, N, S = get("lx", "ly"), get("nx", "ny"), get("sx", "sy")
    D = get("dx", "dy")
    ncc = np.array([f_(r["ncc"]) for r in rows])
    ok = np.isfinite(D).all(1) & (ncc > 0.5) & (np.linalg.norm(D, axis=1) < 400)
    V = np.where(ok[:, None], D @ A.T, np.nan)
    # отрезки непрерывного движения: разрыв там, где сдвиг не измерен
    seg = np.cumsum(~ok)
    T = np.zeros((n, 2))
    for i in range(1, n):
        T[i] = T[i - 1] + (V[i] if ok[i] else 0)
    res = {}
    for name, M in (("integer_256", L), ("subpixel_256", S), ("subpixel_native", N)):
        best = None
        for lag in range(-max_lag, max_lag + 1):
            r_all = []
            for s in np.unique(seg):
                idx = np.where(seg == s)[0]
                idx = idx[(idx + lag >= 0) & (idx + lag < n)]
                idx = idx[seg[np.clip(idx + lag, 0, n - 1)] == s]
                if len(idx) < 20:
                    continue
                m = M[idx + lag] - T[idx]
                m = m[np.isfinite(m).all(1)]
                if len(m) < 20:
                    continue
                # Смещение — скользящая медиана за ±win кадров, а не одна на
                # отрезок: интеграл сдвигов дрейфует (случайное блуждание).
                off = np.array([np.median(m[max(0, j - win):j + win + 1], 0)
                                for j in range(len(m))])
                r_all.append(m - off)
            if not r_all:
                continue
            r = np.vstack(r_all)
            sd = robust_sd(r)
            score = float(np.mean(sd))
            if best is None or score < best[0]:
                best = (score, lag, sd, pct(np.linalg.norm(r, axis=1)), len(r))
        if best:
            res[name] = {"lag_frames": best[1], "lag_ms": round(best[1] / fps * 1000, 1),
                         "axis_sd": best[2], "radial": best[3], "n": best[4]}
    return {"file": path.name, "fps": fps, "frames": n, "registered": int(ok.sum()),
            "segments": int(len(np.unique(seg))), **res}


def f_(v) -> float:
    return float(v) if v not in ("", None) else float("nan")


# ------------------------------------------------------------ модель

def predict_video(vid: str, model_path: Path, dev: str) -> str:
    """Ответ патчевой модели по каждому кадру ролика (хранимые сцены)."""
    import torch  # noqa: F401
    from patch_model import infer_images, load_patchnet
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    keep = stored_keep(layout) > 127
    h0, _ = load_projection()
    model = load_patchnet(model_path, dev)
    rows = read_rows(vid)
    out = []
    for s in range(0, len(rows), 64):
        chunk = rows[s:s + 64]
        imgs = [cv2.cvtColor(cv2.imread(str(COORDS_DATA / r["scene"])), cv2.COLOR_BGR2RGB)
                for r in chunk]
        cams, agree, patches = infer_images(model, imgs, [np.eye(3)] * len(imgs),
                                            [keep] * len(imgs), h0, dev, batch=8)
        for r, c, a, p in zip(chunk, cams, agree, patches):
            out.append({"idx": r["idx"], "mx": round(float(c[0]), 5),
                        "my": round(float(c[1]), 5), "agree": round(float(a), 3),
                        "game": "" if p["game"] is None else round(p["game"], 3)})
    write_csv(OUT / "pred" / f"{vid}.csv", out)
    return f"{vid}: кадров {len(out)}"


# ------------------------------------------------------------ анализ

def edges_ok(reg: list[dict], n: int, ncc_min: float, margin_min: float,
             close_tol: float) -> tuple[dict, dict]:
    """Надёжные приращения сцены и учёт отсеянного.

    Пара проходит, если (а) пик NCC достаточно высок и заметно выше
    второго пика, (б) она входит хотя бы в один замкнувшийся треугольник
    i→i+1→i+2 (невязка < close_tol ед.). Замыкание не смотрит на метку:
    прыжок камеры, ложное совпадение или туман рвут треугольник.
    """
    e = {(int(r["i"]), int(r["j"])): r for r in reg}
    d = {k: np.array([float(v["dx"]), float(v["dy"])]) for k, v in e.items()}
    good = {k: float(v["ncc"]) >= ncc_min and
            float(v["peak"]) - float(v["second"]) >= margin_min for k, v in e.items()}
    closure = {}
    for i in range(n - 2):
        ks = [(i, i + 1), (i + 1, i + 2), (i, i + 2)]
        if all(k in d for k in ks):
            closure[i] = float(np.linalg.norm(d[ks[0]] + d[ks[1]] - d[ks[2]]))
    conf = {k: False for k in e}
    for i, c in closure.items():
        ks = [(i, i + 1), (i + 1, i + 2), (i, i + 2)]
        if c < close_tol and all(good[k] for k in ks):
            for k in ks:
                conf[k] = True
    stats = {"pairs": len(e),
             "gap1": sum(1 for k in e if k[1] - k[0] == 1),
             "weak_peak": sum(1 for k in e if not good[k]),
             "accepted": sum(conf.values()),
             "accepted_gap1": sum(v for k, v in conf.items() if k[1] - k[0] == 1),
             "closure": closure}
    return {k: d[k] for k, v in conf.items() if v}, stats


def fit_scale(edges: dict, lab: np.ndarray, okq: np.ndarray) -> np.ndarray:
    """2×2 A: приращение метки ≈ A · приращение сцены (устойчивый МНК).

    Регрессия метки на сцену, а не наоборот: шум (квантование) у метки, у
    сцены его на порядок меньше, поэтому наклон не занижается.
    """
    X, Y = [], []
    for (i, j), d in edges.items():
        if okq[i] and okq[j]:
            X.append(d)
            Y.append(lab[j] - lab[i])
    return fit_pairs(np.array(X), np.array(Y))


def fit_pairs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Y ≈ X·Aᵀ с весами Коши (масштаб 40 ед.): прыжки метки не тянут."""
    A = np.eye(2) * 0.86
    for _ in range(10):
        r = np.linalg.norm(Y - X @ A.T, axis=1)
        w = 1 / (1 + (r / 40) ** 2)
        Aw, *_ = np.linalg.lstsq(X * w[:, None], Y * w[:, None], rcond=None)
        A = Aw.T
    return A


def trajectory(n: int, edges: dict, A: np.ndarray, lab: np.ndarray,
               anch: np.ndarray, sig_e: float, sig_l: float, c_rob: float):
    """Согласованная траектория: МНК по приращениям сцены и меткам.

    Положения кадров каждой связной цепочки подбираются по уравнениям
    X_j − X_i = A·d_ij (σ = sig_e) и X_i = L_i для опорных кадров `anch`
    (σ = sig_l, веса Коши с масштабом c_rob — грубые ошибки метки не тянут).
    Оси независимы и решаются раздельно с одной матрицей.

    Возвращает T — траекторию; Tl — траекторию «без своей метки»: у опорного
    кадра T − влияние его собственной метки (точная формула leave-one-out
    для взвешенного МНК: остаток делится на 1 − h_ii), у остальных равна T;
    размер цепочки; σ траектории в точке по оси (из диагонали обратной
    матрицы, при принятых σ).
    """
    T = np.full((n, 2), np.nan)
    Tl = np.full((n, 2), np.nan)
    comp = np.zeros(n, int)
    tsig = np.full(n, np.nan)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in edges:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for g in groups.values():
        if len(g) < 3 or sum(anch[i] for i in g) < 3:
            continue
        pos = {f: k for k, f in enumerate(g)}
        m = len(g)
        E = [(pos[i], pos[j], A @ d) for (i, j), d in edges.items() if i in pos]
        base = np.zeros((m, m))
        rb = np.zeros((m, 2))
        w = 1 / sig_e ** 2
        for a, b, dd in E:
            base[a, a] += w; base[b, b] += w; base[a, b] -= w; base[b, a] -= w
            rb[a] -= w * dd; rb[b] += w * dd
        wl = np.array([1.0 if anch[f] else 0.0 for f in g])
        Lg = np.where(np.isfinite(lab[g]), lab[g], 0)
        for _ in range(15):
            M = base.copy()
            wa = wl / sig_l ** 2
            M[np.arange(m), np.arange(m)] += wa
            Minv = np.linalg.inv(M + 1e-9 * np.eye(m))
            X = Minv @ (rb + wa[:, None] * Lg)
            r = np.linalg.norm(Lg - X, axis=1)
            new = np.where([anch[f] for f in g], 1 / (1 + (r / c_rob) ** 2), 0.0)
            if np.allclose(new, wl, atol=1e-4):
                break
            wl = new
        h = np.diag(Minv) * wl / sig_l ** 2          # влияние метки на себя
        Xl = Lg - (Lg - X) / np.maximum(1 - h, 1e-3)[:, None]
        for f, k in pos.items():
            T[f] = X[k]
            Tl[f] = Xl[k] if wl[k] > 0 else X[k]
            comp[f] = m
            tsig[f] = math.sqrt(Minv[k, k])
    return T, Tl, comp, tsig


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(v, qs=(50, 90, 99)) -> list[float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [round(float(np.percentile(v, q)), 1) for q in qs] if len(v) else []


def robust_sd(v: np.ndarray) -> list[float]:
    v = v[np.isfinite(v).all(1)]
    if not len(v):
        return []
    return (1.4826 * np.median(np.abs(v - np.median(v, 0)), 0)).round(1).tolist()


def share(e: np.ndarray, thr: float) -> dict:
    e = e[np.isfinite(e)]
    k = int((e > thr).sum())
    lo, hi = wilson(k, len(e))
    return {"k": k, "n": int(len(e)), "p": round(k / max(len(e), 1), 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def load_mini(vid: str, n: int) -> dict:
    """Пересчитанная рамка (mini/<ролик>.csv): обрезка, субпиксель, размер."""
    out = {"S": np.full((n, 2), np.nan), "K": np.full((n, 2), np.nan),
           "at_bound": np.zeros(n, bool), "clipped": np.zeros(n, bool),
           "w": np.full(n, np.nan), "h": np.full(n, np.nan), "box": None}
    p = OUT / "mini" / f"{vid}.csv"
    if not p.exists():
        return out
    for r in read_csv(p):
        i = int(r["idx"])
        out["box"] = (int(r["bw"]), int(r["bh"]))
        if r["sx"] != "" and r["sy"] != "":
            out["S"][i] = (float(r["sx"]) * MAP_UNITS, float(r["sy"]) * MAP_UNITS)
        out["K"][i] = (float(r["kx"]) * MAP_UNITS, float(r["ky"]) * MAP_UNITS)
        out["at_bound"][i] = r["at_bound"] == "1"
        out["clipped"][i] = r["clipped"] == "1"
        out["w"][i] = float(r["w"]) if r["w"] != "" else np.nan
        out["h"][i] = float(r["h"]) if r["h"] != "" else np.nan
    return out


def analyze(vids: list[str], args) -> dict:
    """Сводка по роликам: шум меток, траектория, модель → frames/ и summary.json.

    Опорные метки траектории — с качеством ≥ min_quality и не прижатые к
    краю миникарты: прижатая метка цензурирована (рамка обрезана краем, её
    истинный центр дальше), её смещение меряется отдельно.
    """
    per_video, allf, data = {}, [], {}
    for vid in vids:
        rows = read_rows(vid)
        n = len(rows)
        lab = np.array([[float(r["cx"]), float(r["cy"])] for r in rows]) * MAP_UNITS
        q = np.array([float(r["quality"]) for r in rows])
        mini = load_mini(vid, n)
        okq = q >= args.min_quality
        anch = okq & ~mini["at_bound"]
        edges, st = edges_ok(read_csv(OUT / "reg" / f"{vid}.csv"), n,
                             args.ncc_min, args.margin_min, args.close_tol)
        data[vid] = (rows, lab, q, okq, anch, mini, edges, st)

    # 1. Масштаб сцены → меток: одна матрица на все ролики.
    X, Y = [], []
    for rows, lab, q, okq, anch, mini, edges, st in data.values():
        for (i, j), d in edges.items():
            if anch[i] and anch[j]:
                X.append(d)
                Y.append(lab[j] - lab[i])
    A = fit_pairs(np.array(X), np.array(Y))

    # 2. Шум регистрации — по невязке принятых треугольников (без меток).
    clos, acc = [], []
    for rows, lab, q, okq, anch, mini, edges, st in data.values():
        for i, c in st["closure"].items():
            clos.append(c)
            if all(k in edges for k in [(i, i + 1), (i + 1, i + 2), (i, i + 2)]):
                acc.append(c)
    acc = np.array(acc)
    # Невязка — сумма трёх ошибок рёбер: σ_оси ребра ≈ RMS(|невязка|)/√(3·2),
    # в единицах меток (×|A|).
    sig_e = float(np.sqrt(np.mean(acc ** 2)) / math.sqrt(6)
                  * np.sqrt(abs(np.linalg.det(A))))
    sig_e = max(sig_e, 0.5)

    for vid, (rows, lab, q, okq, anch, mini, edges, st) in data.items():
        n = len(rows)
        S, K = mini["S"], mini["K"]
        T, Tl, comp, tsig = trajectory(n, edges, A, lab, anch, sig_e,
                                       args.sig_label, args.c_rob)
        anchS = okq & np.isfinite(S).all(1) & ~mini["clipped"]
        TS, TSl, _, _ = trajectory(n, edges, A, S, anchS, sig_e,
                                   args.sig_label / 2, args.c_rob)
        pred = np.full((n, 2), np.nan)
        game = np.full(n, np.nan)
        pfile = OUT / "pred" / f"{vid}.csv"
        if pfile.exists():
            for r in read_csv(pfile):
                pred[int(r["idx"])] = (float(r["mx"]) * MAP_UNITS, float(r["my"]) * MAP_UNITS)
                game[int(r["idx"])] = float(r["game"]) if r["game"] != "" else np.nan
        vel = np.full((n, 2), np.nan)          # ед./с по траектории
        for i in range(1, n - 1):
            if comp[i] and comp[i - 1] == comp[i] == comp[i + 1] and \
                    np.isfinite(T[[i - 1, i + 1]]).all():
                vel[i] = (T[i + 1] - T[i - 1]) / 2
        frames = []
        for i, r in enumerate(rows):
            f = {"video_id": vid, "idx": i, "t_sec": r["t_sec"], "quality": q[i],
                 "at_bound": int(mini["at_bound"][i]), "clipped": int(mini["clipped"][i]),
                 "anchor": int(anch[i]),
                 "lx": lab[i, 0], "ly": lab[i, 1], "kx": K[i, 0], "ky": K[i, 1],
                 "sx": S[i, 0], "sy": S[i, 1],
                 "mx": pred[i, 0], "my": pred[i, 1], "game": game[i],
                 "tx": Tl[i, 0], "ty": Tl[i, 1], "t_sigma": tsig[i], "chain": comp[i],
                 "tsx": TSl[i, 0], "tsy": TSl[i, 1], "vx": vel[i, 0], "vy": vel[i, 1],
                 "box_w": mini["w"][i], "box_h": mini["h"][i]}
            frames.append({k: (round(float(v), 2) if isinstance(v, (float, np.floating))
                               and np.isfinite(v) else
                               ("" if isinstance(v, (float, np.floating)) else v))
                           for k, v in f.items()})
        write_csv(OUT / "frames" / f"{vid}.csv", frames)
        allf += frames
        # приращения по соседним кадрам: метка / субпиксель / модель против сцены
        dl, ds, dm = [], [], []
        for (i, j), d in edges.items():
            if j - i != 1 or not (anch[i] and anch[j]):
                continue
            ad = A @ d
            dl.append(lab[j] - lab[i] - ad)
            if np.isfinite(S[[i, j]]).all():
                ds.append(S[j] - S[i] - ad)
            if np.isfinite(pred[[i, j]]).all():
                dm.append(pred[j] - pred[i] - ad)
        per_video[vid] = {
            "frames": n, "quality_ok": int(okq.sum()), "at_bound": int((okq & mini["at_bound"]).sum()),
            "pairs_gap1": st["gap1"], "accepted_gap1": st["accepted_gap1"],
            "pairs": st["pairs"], "accepted": st["accepted"], "weak_peak": st["weak_peak"],
            "in_trajectory": int((okq & (comp > 0)).sum()),
            "A_video": fit_scale(edges, lab, anch).round(3).tolist(),
            "box": mini["box"],
            "box_w_sub": pct(mini["w"][okq & ~mini["clipped"]], (5, 50, 95)),
            "box_h_sub": pct(mini["h"][okq & ~mini["clipped"]], (5, 50, 95)),
            "dlabel_sd": robust_sd(np.array(dl)) if dl else None,
            "dsub_sd": robust_sd(np.array(ds)) if ds else None,
            "dmodel_sd": robust_sd(np.array(dm)) if dm else None,
            "dpairs": len(dl),
        }
        print(f"{vid}: пар i→i+1 {st['gap1']}, надёжных {st['accepted_gap1']}; "
              f"в траектории {per_video[vid]['in_trajectory']} из {okq.sum()} кадров; "
              f"A {per_video[vid]['A_video']}", flush=True)

    summary = {"A": A.round(4).tolist(), "sigma_edge": round(sig_e, 2),
               "closure_all": pct(clos, (25, 50, 75, 90)),
               "closure_accepted": pct(acc, (50, 90, 99)),
               "pairs_total": int(sum(v["pairs_gap1"] for v in per_video.values())),
               "pairs_accepted": int(sum(v["accepted_gap1"] for v in per_video.values())),
               "params": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()} | {"videos": vids},
               "videos": per_video}
    summary.update(stats_from_frames(allf))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                 default=str) + "\n", encoding="utf-8")
    return summary


def stats_from_frames(allf: list[dict]) -> dict:
    """Итоговые цифры по таблице кадров (frames/*.csv) всех роликов."""
    def col(name, rows):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])

    def xy(a, rows):
        return np.c_[col(a + "x", rows), col(a + "y", rows)]

    out = {}
    held = [r for r in allf if r["video_id"] in HELD_OUT]
    for tag, rows in (("all", allf), ("held_out", held)):
        if not rows:
            continue
        q = col("quality", rows)
        good = q >= 0.4
        atb = col("at_bound", rows) > 0
        clip = col("clipped", rows) > 0
        L, K, S, M, T, TS = (xy(a, rows) for a in ("l", "k", "s", "m", "t", "ts"))
        V = xy("v", rows)
        intr = np.isfinite(T).all(1)
        base = good & intr & ~atb                 # обычные кадры с траекторией
        eL = np.linalg.norm(L - T, axis=1)
        d = {"frames_q04": int(good.sum()), "at_bound_q04": int((good & atb).sum()),
             "clipped_q04": int((good & clip).sum()),
             "in_traj_q04": int((good & intr).sum()),
             "traj_sigma_axis": pct(col("t_sigma", rows)[good & intr], (50, 90))}
        d["label"] = {"n": int(base.sum()), "radial": pct(eL[base]),
                      "axis_sd": robust_sd((L - T)[base]),
                      "bias": np.nanmedian((L - T)[base], 0).round(1).tolist(),
                      "gt100": share(eL[base], 100), "gt200": share(eL[base], 200)}
        # Квантование: остаток метки против дробной части траектории в
        # пикселях миникарты. Чистое округление даёт пилу ±29 ед.
        ph = {}
        for ax, nm in ((0, "x"), (1, "y")):
            frac = (T[base, ax] / PX) % 1.0
            res = (L - T)[base, ax]
            bins = np.floor(frac * 8).astype(int)
            ph[nm] = [round(float(np.median(res[bins == b])), 1) if (bins == b).any() else None
                      for b in range(8)]
        d["label"]["by_phase_8"] = ph
        # Отставание: остаток против скорости камеры (ед./с) — наклон в секундах.
        ok = base & np.isfinite(V).all(1)
        lag = []
        for ax in range(2):
            v, r = V[ok, ax], (L - T)[ok, ax]
            m = np.abs(r) < 120
            lag.append(round(float(-np.sum(v[m] * r[m]) / np.sum(v[m] ** 2)), 4))
        d["label"]["lag_sec_from_velocity"] = lag
        d["label"]["speed_med"] = pct(np.linalg.norm(V[ok], axis=1), (50, 90))
        # прижатые к краю метки: смещение относительно траектории
        b2 = good & intr & atb
        d["at_bound"] = {"n": int(b2.sum()), "radial": pct(eL[b2]),
                         "gt100": share(eL[b2], 100),
                         "clip_aware_radial": pct(np.linalg.norm(K - T, axis=1)[b2]),
                         "subpixel_radial": pct(np.linalg.norm(S - T, axis=1)[b2])}
        eK = np.linalg.norm(K - T, axis=1)
        eS = np.linalg.norm(S - T, axis=1)
        allg = good & intr
        d["variants_vs_traj_all_q04"] = {
            "n": int(allg.sum()),
            "label": pct(eL[allg]), "label_gt200": share(eL[allg], 200),
            "clip_aware": pct(eK[allg]), "clip_aware_gt200": share(eK[allg], 200),
            "subpixel": pct(eS[allg]), "subpixel_gt200": share(eS[allg], 200),
            "subpixel_axis_sd": robust_sd((S - T)[allg & ~clip])}
        eSS = np.linalg.norm(S - TS, axis=1)
        okS = good & np.isfinite(TS).all(1) & ~clip
        d["subpixel_vs_own_traj"] = {"n": int(okS.sum()), "radial": pct(eSS[okS]),
                                     "axis_sd": robust_sd((S - TS)[okS])}
        if np.isfinite(M).any():
            hasm = good & np.isfinite(M).all(1)
            sub = hasm & intr
            eM = {k: np.linalg.norm(M - v, axis=1) for k, v in
                  (("label", L), ("clip_aware", K), ("subpixel", S), ("traj", T))}
            d["model"] = {"n_q04": int(hasm.sum()),
                          "vs_label_all": pct(eM["label"][hasm]),
                          "vs_clip_aware_all": pct(eM["clip_aware"][hasm]),
                          "vs_subpixel_all": pct(eM["subpixel"][hasm]),
                          "gt200_vs_label_all": share(eM["label"][hasm], 200),
                          "gt200_vs_clip_aware_all": share(eM["clip_aware"][hasm], 200),
                          "traj_n": int(sub.sum()),
                          "traj_vs_label": pct(eM["label"][sub]),
                          "traj_vs_subpixel": pct(eM["subpixel"][sub]),
                          "traj_vs_traj": pct(eM["traj"][sub]),
                          "traj_label_vs_traj": pct(eL[sub]),
                          "traj_axis_sd_model_vs_traj": robust_sd((M - T)[sub]),
                          "traj_bias_model_vs_traj": np.nanmedian((M - T)[sub], 0).round(1).tolist(),
                          "not_game_frames": int(np.nansum(col("game", rows)[hasm] < 0.5))}
            nb = sub & ~atb
            d["model"]["traj_not_bound_n"] = int(nb.sum())
            d["model"]["traj_not_bound_vs_label"] = pct(eM["label"][nb])
            d["model"]["traj_not_bound_vs_traj"] = pct(eM["traj"][nb])
        out[tag] = d
    return out


# ------------------------------------------------------------ CLI

def low_priority() -> None:
    """Понизить приоритет процесса: анализ идёт часами и не должен мешать
    интерактивной работе на той же машине (замечено — мешал игре)."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:                                         # noqa: BLE001
        pass


def _register(v):
    low_priority()
    return register_video(v)


def _minimap(v):
    low_priority()
    return minimap_video(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "minimap", "predict", "analyze"):
        p = sp.add_parser(name)
        p.add_argument("videos", nargs="*")
        p.add_argument("--all", action="store_true", help="все ролики data/coords")
        if name == "register" or name == "minimap":
            p.add_argument("--workers", type=int, default=4,
                           help="процессов; каждый понижает себе приоритет")
        if name == "predict":
            p.add_argument("--model", type=Path, default=MODEL)
        if name == "analyze":
            p.add_argument("--min-quality", type=float, default=0.4)
            p.add_argument("--ncc-min", type=float, default=0.3)
            p.add_argument("--margin-min", type=float, default=0.02)
            p.add_argument("--close-tol", type=float, default=15.0,
                           help="невязка треугольника, ед. сцены")
            p.add_argument("--sig-label", type=float, default=17.0,
                           help="σ метки по оси для весов, ед. (квантование: 57.8/√12)")
            p.add_argument("--c-rob", type=float, default=45.0,
                           help="масштаб устойчивых весов меток, ед.")
    p = sp.add_parser("video", help="плотный кусок локального mp4: отставание и нативная рамка")
    p.add_argument("video")
    p.add_argument("--start", type=float, nargs="+", default=[300.0])
    p.add_argument("--dur", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    if args.cmd == "video":
        sfile = OUT / "summary.json"
        A = (np.array(json.loads(sfile.read_text(encoding="utf-8"))["A"]) if sfile.exists()
             else np.eye(2) * 0.86)
        reps = []
        for st in args.start:
            p = video_segment(args.video, st, args.dur, args.fps)
            reps.append(lag_report(p, A))
            print(json.dumps(reps[-1], ensure_ascii=False), flush=True)
        lf = OUT / "video" / "lag.json"
        old = json.loads(lf.read_text(encoding="utf-8")) if lf.exists() else []
        old = [o for o in old if o["file"] not in {r["file"] for r in reps}] + reps
        lf.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    vids = all_videos() if args.all else args.videos
    if not vids:
        raise SystemExit("укажите ролики или --all")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "register":
        with ProcessPoolExecutor(min(args.workers, len(vids))) as ex:
            for msg in ex.map(_register, vids):
                print(msg, flush=True)
    elif args.cmd == "minimap":
        with ProcessPoolExecutor(min(args.workers, len(vids))) as ex:
            for msg in ex.map(_minimap, vids):
                print(msg, flush=True)
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        nat = native_frames(layout)
        if nat:
            write_csv(OUT / "native_frames.csv", nat)
            print(f"полные кадры dataset/videos: {len(nat)} → {OUT / 'native_frames.csv'}")
    elif args.cmd == "predict":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for v in vids:
            print(predict_video(v, args.model, dev), flush=True)
    else:
        s = analyze(vids, args)
        print(json.dumps({k: v for k, v in s.items() if k not in ("videos", "params")},
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
