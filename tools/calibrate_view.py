"""Проверка того, что экран переводится в карту линейно, и замер масштаба.

Зачем. Поячеечная разметка держится на предположении: если камера стоит в
точке (cx, cy), то ячейка картинки с относительными координатами (u, v)
показывает место карты

    mx = cx + (u - 0.5) * VIEW_W,   my = cy + (v - 0.5) * VIEW_H.

Если это неверно, поячеечные метки будут систематически кривыми, и учить по
ним бессмысленно. Проверяется так: берутся пары соседних кадров, по миникарте
известно, на сколько сдвинулась камера; сдвиг самой картинки меряется фазовой
корреляцией. Отношение одного к другому и есть масштаб, и оно обязано быть
постоянным.

Камера в League of Legends — перспективная под фиксированным углом, поэтому
точной линейности ждать нельзя. Замер показывает, насколько велико
отклонение и можно ли им пренебречь на масштабе ячейки.

Пример:
    python tools/calibrate_view.py zJvTSjEnKNE olmTXkkUv58
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def validity_mask(root: Path, rows: list[dict], n: int = 60) -> np.ndarray:
    """Пиксели, которые вообще меняются от кадра к кадру.

    Закрытые маской прямоугольники одинаковы во всех кадрах. Для фазовой
    корреляции это худший возможный сигнал: он идеально совпадает при нулевом
    сдвиге и перебивает настоящее движение сцены.
    """
    step = max(1, len(rows) // n)
    acc = []
    for r in rows[::step][:n]:
        with Image.open(root / r["scene"]) as im:
            acc.append(np.asarray(im.convert("L"), dtype=np.float32))
    return np.std(np.stack(acc), axis=0) > 3.0


def shift_by_phase(a: np.ndarray, b: np.ndarray,
                   valid: np.ndarray | None = None,
                   rng=None) -> tuple[float, float, float]:
    """Сдвиг b относительно a в пикселях через фазовую корреляцию.

    Окно Ханна гасит края: без него скачок яркости на границе кадра даёт
    ложный пик в нуле. Неизменные области заполняются независимым шумом в
    каждом кадре — так они перестают давать вклад в совпадение.
    """
    if valid is not None:
        rng = rng or np.random.default_rng(0)
        a = a.copy()
        b = b.copy()
        a[~valid] = rng.normal(128, 40, int((~valid).sum()))
        b[~valid] = rng.normal(128, 40, int((~valid).sum()))
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft2(cross, s=a.shape)
    peak = np.unravel_index(int(corr.argmax()), corr.shape)
    dy, dx = peak[0], peak[1]
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return float(dx), float(dy), float(corr.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--root", type=Path, default=Path("data/coords"))
    ap.add_argument("--min-quality", type=float, default=0.6)
    ap.add_argument("--max-step", type=float, default=0.06,
                    help="макс. сдвиг камеры между кадрами в долях карты")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    rows_all = []
    for vid in args.videos:
        rows = list(csv.DictReader((args.root / f"{vid}.csv").open(encoding="utf-8")))
        ok = {int(r["idx"]): r for r in rows if float(r["quality"]) >= args.min_quality}
        valid = validity_mask(args.root, rows)
        print(f"{vid}: меняющихся пикселей {valid.mean() * 100:.0f}% кадра")
        rng = np.random.default_rng(0)
        pairs = [(ok[i], ok[i + 1]) for i in sorted(ok) if i + 1 in ok]
        used = 0
        for a, b in pairs:
            dcx = float(b["cx"]) - float(a["cx"])
            dcy = float(b["cy"]) - float(a["cy"])
            if max(abs(dcx), abs(dcy)) > args.max_step:
                continue            # перескок камеры: фазовая корреляция не поймает
            if abs(dcx) < 0.004 and abs(dcy) < 0.004:
                continue            # камера стоит: делить будет не на что
            with Image.open(args.root / a["scene"]) as im:
                ga = np.asarray(im.convert("L"), dtype=np.float32)
            with Image.open(args.root / b["scene"]) as im:
                gb = np.asarray(im.convert("L"), dtype=np.float32)
            dx, dy, q = shift_by_phase(ga, gb, valid, rng)
            if q < 0.02:
                continue
            rows_all.append({"vid": vid, "dcx": dcx, "dcy": dcy,
                             "dx": dx, "dy": dy, "q": q,
                             "w": ga.shape[1], "h": ga.shape[0]})
            used += 1
            if used >= args.limit:
                break
        print(f"{vid}: пригодных пар {used} из {len(pairs)}")

    if not rows_all:
        raise SystemExit("не нашлось пар с движением камеры — нечего калибровать")

    dcx = np.array([r["dcx"] for r in rows_all])
    dcy = np.array([r["dcy"] for r in rows_all])
    dx = np.array([r["dx"] for r in rows_all])
    dy = np.array([r["dy"] for r in rows_all])
    W, H = rows_all[0]["w"], rows_all[0]["h"]

    # Камера едет вправо → картинка едет влево, поэтому знак отрицательный.
    mx = np.abs(dcx) > 0.006
    my = np.abs(dcy) > 0.006
    sx = -dx[mx] / dcx[mx]
    sy = -dy[my] / dcy[my]
    print(f"\nкадр хранится как {W}x{H}")
    print(f"по горизонтали: {np.median(sx):.0f} px на долю карты "
          f"(разброс {np.percentile(sx, 25):.0f}…{np.percentile(sx, 75):.0f}, "
          f"пар {mx.sum()})")
    print(f"по вертикали:   {np.median(sy):.0f} px на долю карты "
          f"(разброс {np.percentile(sy, 25):.0f}…{np.percentile(sy, 75):.0f}, "
          f"пар {my.sum()})")

    view_w = W / np.median(sx)
    view_h = H / np.median(sy)
    print(f"\nотсюда ширина вьюпорта {view_w:.3f} карты, высота {view_h:.3f}")
    print(f"  (в раскладке принято 0.266 и {0.266 * 9 / 16:.3f})")
    print(f"  соотношение сторон вьюпорта в долях карты: {view_w / view_h:.2f}")

    # Насколько хорошо линейная модель объясняет наблюдения.
    pred_dx = -dcx * np.median(sx)
    pred_dy = -dcy * np.median(sy)
    rx = dx - pred_dx
    ry = dy - pred_dy
    print(f"\nостаток линейной модели: по x медиана |{np.median(np.abs(rx)):.1f}| px, "
          f"90-й перцентиль {np.percentile(np.abs(rx), 90):.1f} px")
    print(f"                          по y медиана |{np.median(np.abs(ry)):.1f}| px, "
          f"90-й перцентиль {np.percentile(np.abs(ry), 90):.1f} px")
    cellw = W / 8
    print(f"\nячейка сетки 8x5 — это {cellw:.0f}x{H / 5:.0f} px; "
          f"типичный остаток {np.median(np.abs(rx)):.1f} px — "
          f"{np.median(np.abs(rx)) / cellw * 100:.0f}% ширины ячейки")


if __name__ == "__main__":
    main()
