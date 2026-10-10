"""Пересчёт меток камеры плотного набора новым поиском рамки, без распаковки.

Зачем. После исправления поиска рамки у края миникарты (find_box_clipped,
#754) метки data/coords/<ролик>.csv нужно пересчитать. Миникарты 256 уже
лежат рядом (mini/), ролики заново распаковывать не нужно.

Сохранённая миникарта — JPEG, и на ней find_box изредка находит рамку на
пиксель иначе, чем при сборке по несжатому кадру. Поэтому метка меняется,
только если на сохранённой миникарте прежний поиск воспроизводит прежнюю
метку: тогда разница целиком от нового поиска. Иначе метка остаётся.
Прежние CSV копируются в data/coords/_labels_v1/.

Пример:
    python tools/relabel_coords.py --all
    python tools/relabel_coords.py 58w57eJ5Qks --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from lolvision.minimap_camera import MM, estimate_box, find_box, find_box_clipped, white_mask
from lolvision.paths import COORDS_DATA

BACKUP = COORDS_DATA / "_labels_v1"
TOL = 0.6 / MM        # «та же метка»: метки округлены до 1e-4, пиксель — 0.0039


def relabel(vid: str, dry: bool) -> str:
    src = COORDS_DATA / f"{vid}.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    masks = [white_mask(np.asarray(Image.open(COORDS_DATA / r["mini"]).convert("RGB"),
                                   dtype=np.float32)) for r in rows]
    step = max(1, len(masks) // 120)
    est = estimate_box(masks[::step])
    # Размер, с которым собирались прежние метки: тот, что их воспроизводит.
    # Он может отличаться от оценки по сжатым миникартам на пиксель.
    def same(m, ox, oy, bw, bh):
        x, y, _ = find_box(m, bw, bh)
        return abs((x + bw / 2) / MM - ox) <= TOL and abs((y + bh / 2) / MM - oy) <= TOL
    sample = list(zip(masks, rows))[::max(1, len(rows) // 150)]
    sizes = {est} | {(w, round(w * 9 / 16)) for w in range(66, 73)}
    hits = {s: sum(same(m, float(r["cx"]), float(r["cy"]), *s) for m, r in sample)
            for s in sizes}
    old_size = max(hits, key=hits.get)
    if hits[est] >= hits[old_size] - 2:      # равные по воспроизведению — берём оценку
        old_size = est
    # Если прежние метки почти не воспроизводятся ни с каким размером, при
    # сборке размер был выбран неверно (так у a7p5Ql2LbAg, docs/label-noise.md):
    # пересчитываем все кадры с оценкой по ролику.
    redo = hits[old_size] < 0.5 * len(sample)
    bw, bh = est if redo else old_size
    changed = kept_drift = 0
    moved = []
    for r, m in zip(rows, masks):
        ox, oy = float(r["cx"]), float(r["cy"])
        if not redo and not same(m, ox, oy, bw, bh):
            kept_drift += 1
            continue
        kx, ky, q, _ = find_box_clipped(m, bw, bh)
        nx, ny = round((kx + bw / 2) / MM, 4), round((ky + bh / 2) / MM, 4)
        if abs(nx - ox) > TOL or abs(ny - oy) > TOL:
            moved.append(float(np.hypot(nx - ox, ny - oy)) * 14800)
            r["cx"], r["cy"], r["quality"] = nx, ny, round(q, 3)
            changed += 1
    if not dry:
        BACKUP.mkdir(exist_ok=True)
        if not (BACKUP / src.name).exists():
            shutil.copy2(src, BACKUP / src.name)
        with src.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    mv = np.array(moved) if moved else np.zeros(1)
    return (f"{vid}: рамка {bw}x{bh}{' (пересчёт всех кадров)' if redo else ''}, "
            f"кадров {len(rows)}, изменено {changed} (больше 200 ед.: "
            f"{int((mv > 200).sum())}), оставлено из-за JPEG {kept_drift}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    vids = (sorted(p.stem for p in COORDS_DATA.glob("*.csv")) if args.all else args.videos)
    if not vids:
        raise SystemExit("укажите ролики или --all")
    with ProcessPoolExecutor(min(args.workers, len(vids))) as ex:
        for msg in ex.map(relabel, vids, [args.dry_run] * len(vids)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
