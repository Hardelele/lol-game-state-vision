"""Калибровка проекции «экран → смещение на карте» по движению камеры.

Зачем. Поячеечная разметка требует знать, какому месту карты соответствует
каждая ячейка кадра. Наивная формула «центр плюс доля вьюпорта» неверна:
камера в League of Legends наклонена, и масштаб меняется по высоте кадра
примерно на 28%. Нужна настоящая проекция.

Как. Камера смотрит на плоскую землю, значит связь «экран ↔ земля» —
гомография. Берутся пары соседних кадров: смещение камеры известно с
миникарты, смещение кусочков картинки меряется поиском шаблона. Для каждого
кусочка выполняется

    p' = g⁻¹( g(p) − Δ ),

где g — искомое отображение экрана в смещение на карте, Δ — смещение камеры.
Восемь параметров гомографии подбираются так, чтобы это равенство выполнялось
на всех замерах. Проверка идёт на роликах, не участвовавших в подборе.

Результат — JSON с матрицей, который читают сборщик датасета и смотрелка.

Пример:
    python tools/calibrate_projection.py --fit 58w57eJ5Qks olmTXkkUv58 \
        --check zJvTSjEnKNE
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

PATCH = 96


def collect(vid: str, root: Path, min_q: float, max_pairs: int, rng,
            gaps=(1, 2, 3, 4, 6, 8), lo=0.010, hi=0.11, min_match=0.62,
            spread_px=9.0) -> np.ndarray:
    """Соответствия кусочков между кадрами с разным интервалом.

    Интервал больше одного кадра берётся намеренно: положение камеры
    считывается с миникарты с шагом 1/256 карты, и на коротком сдвиге эта
    ступенька составляет заметную долю самого сдвига. Чем больше база, тем
    меньше относительная погрешность метки — до предела, пока кадры ещё
    перекрываются.
    """
    rows = list(csv.DictReader((root / f"{vid}.csv").open(encoding="utf-8")))
    ok = {int(r["idx"]): r for r in rows if float(r["quality"]) >= min_q}
    acc = [cv2.imread(str(root / r["scene"]), cv2.IMREAD_GRAYSCALE)
           for r in rows[::40][:50]]
    # Закрытые маской области одинаковы во всех кадрах — шаблоны оттуда брать нельзя.
    valid = np.std(np.stack(acc).astype(np.float32), axis=0) > 3.0
    H, W = valid.shape

    recs, pairs = [], 0
    keys = sorted(ok)
    cand = [(i, i + g) for i in keys for g in gaps if i + g in ok]
    rng.shuffle(cand)
    for i, j in cand:
        if pairs >= max_pairs:
            break
        a, b = ok[i], ok[j]
        d = np.array([float(b["cx"]) - float(a["cx"]),
                      float(b["cy"]) - float(a["cy"])])
        mag = np.linalg.norm(d)
        if mag > hi or mag < lo:
            continue
        ga = cv2.imread(str(root / a["scene"]), cv2.IMREAD_GRAYSCALE)
        gb = cv2.imread(str(root / b["scene"]), cv2.IMREAD_GRAYSCALE)
        got, batch = 0, []
        for _ in range(16):
            y = int(rng.integers(0, H - PATCH))
            x = int(rng.integers(0, W - PATCH))
            if valid[y:y + PATCH, x:x + PATCH].mean() < 0.95:
                continue
            tpl = ga[y:y + PATCH, x:x + PATCH]
            if tpl.std() < 12:
                continue
            res = cv2.matchTemplate(gb, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < min_match:
                continue
            batch.append((x + PATCH / 2, y + PATCH / 2,
                          loc[0] + PATCH / 2, loc[1] + PATCH / 2, d[0], d[1]))
            got += 1
        # Кусочки одного кадра должны ехать согласованно. Если нет —
        # среди них есть ложные совпадения, и доверять паре нельзя.
        if got >= 3:
            bb = np.array(batch, float)
            off = bb[:, 2:4] - bb[:, 0:2]
            keep = np.linalg.norm(off - np.median(off, 0), axis=1) < 30
            if keep.sum() >= 3 and off[keep].std(0).max() < spread_px * 3:
                recs.extend(bb[keep].tolist())
                pairs += 1
    print(f"  {vid}: пар {pairs}, соответствий {len(recs)}, кадр {W}x{H}")
    return np.array(recs, float), (W, H)


# Калибровка закреплена условием g(центр кадра) = 0. Без него задача
# вырождена: постоянное слагаемое в g сокращается в уравнении
# p' = g⁻¹(g(p) − Δ), и подбор уезжает в произвольную точку карты.
def to_matrix(par: np.ndarray, center: np.ndarray) -> np.ndarray:
    A = np.array([[par[0], par[1]], [par[2], par[3]]])
    w = np.array([par[4], par[5]])
    t = -A @ center                      # следствие условия закрепления
    return np.array([[A[0, 0], A[0, 1], t[0]],
                     [A[1, 0], A[1, 1], t[1]],
                     [w[0], w[1], 1.0]])


def apply_h(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить гомографию к (N,2)."""
    v = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return v[:, :2] / v[:, 2:3]


def residuals(par: np.ndarray, rec: np.ndarray, center: np.ndarray) -> np.ndarray:
    Hm = to_matrix(par, center)
    try:
        Hi = np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        return np.full(2 * len(rec), 1e3)
    m = apply_h(Hm, rec[:, 0:2]) - rec[:, 4:6]    # смещение на карте после сдвига
    pred = apply_h(Hi, m)
    return (pred - rec[:, 2:4]).ravel()


def fit(rec: np.ndarray, size) -> tuple[np.ndarray, dict]:
    W, Hh = size
    center = np.array([W / 2, Hh / 2])
    # Начальное приближение — линейный масштаб из прямого замера.
    sx, sy = 5565.0, 4295.0
    p0 = np.array([1 / sx, 0, 0, 1 / sy, 0, 0])
    out = least_squares(residuals, p0, args=(rec, center), loss="soft_l1",
                        f_scale=4.0, max_nfev=2000)
    r = residuals(out.x, rec, center).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    return out.x, {"median_px": float(np.median(err)),
                   "p90_px": float(np.percentile(err, 90)), "n": len(rec)}


def report_scale(par: np.ndarray, size) -> None:
    """Во что превращается кадр: размеры видимой области и ход масштаба."""
    W, H = size
    Hm = to_matrix(par, np.array([W / 2, H / 2]))
    corners = apply_h(Hm, np.array([[0, 0], [W, 0], [W, H], [0, H]], float))
    print("  углы кадра в смещениях от центра камеры (доли карты):")
    for name, c in zip(("верх-лево", "верх-право", "низ-право", "низ-лево"), corners):
        print(f"    {name:11s} {c[0]:+.3f}, {c[1]:+.3f}")
    w = corners[:, 0].max() - corners[:, 0].min()
    h = corners[:, 1].max() - corners[:, 1].min()
    print(f"  габарит видимой области: {w:.3f} x {h:.3f} карты "
          f"(трапеция, не прямоугольник)")
    for frac in (0.1, 0.5, 0.9):
        y = frac * H
        a = apply_h(Hm, np.array([[W * 0.4, y], [W * 0.6, y]]))
        px_per_map = (W * 0.2) / abs(a[1, 0] - a[0, 0])
        print(f"  на высоте {frac:.0%} кадра: {px_per_map:.0f} px на долю карты")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--root", type=Path, default=Path("data/coords"))
    ap.add_argument("--min-quality", type=float, default=0.5)
    ap.add_argument("--max-pairs", type=int, default=900)
    ap.add_argument("--out", type=Path, default=Path("dataset/layouts/projection.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("сбор соответствий для подбора:")
    parts, size = [], None
    for v in args.fit:
        r, size = collect(v, args.root, args.min_quality, args.max_pairs, rng)
        parts.append(r)
    rec = np.vstack(parts)

    par, st = fit(rec, size)
    print(f"\nподбор на {st['n']} соответствиях: остаток медиана "
          f"{st['median_px']:.2f} px, 90-й перцентиль {st['p90_px']:.2f} px")
    report_scale(par, size)

    # Сравнение с наивной линейной моделью на тех же данных.
    W, H = size
    lin = np.array([1 / 5565, 0, 0, 1 / 4295, 0, 0])
    rl = residuals(lin, rec, np.array([W / 2, H / 2])).reshape(-1, 2)
    print(f"\nдля сравнения, линейная модель без перспективы: остаток медиана "
          f"{np.median(np.linalg.norm(rl, axis=1)):.2f} px")

    checks = {}
    for v in args.check:
        r, _ = collect(v, args.root, args.min_quality, args.max_pairs, rng)
        e = np.linalg.norm(
            residuals(par, r, np.array([size[0] / 2, size[1] / 2])
                      ).reshape(-1, 2), axis=1)
        checks[v] = {"median_px": float(np.median(e)),
                     "p90_px": float(np.percentile(e, 90)), "n": len(r)}
        print(f"  проверка на {v}: остаток медиана {np.median(e):.2f} px, "
              f"90-й перцентиль {np.percentile(e, 90):.2f} px")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "comment": "Гомография «экран (пиксели хранимой сцены) → смещение от центра "
                   "камеры в долях карты». Подобрана по движению камеры между "
                   "соседними кадрами; белая рамка миникарты для этого не "
                   "используется, она задаёт видимую область лишь приблизительно.",
        "scene_size": [size[0], size[1]],
        "homography": to_matrix(par, np.array([size[0] / 2, size[1] / 2])).tolist(),
        "gauge": "g(центр кадра) = (0, 0)",
        "fit_videos": args.fit, "fit": st, "checks": checks,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nпроекция → {args.out}")


if __name__ == "__main__":
    main()
