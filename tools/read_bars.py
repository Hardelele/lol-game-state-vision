"""Прочитать плашки чемпионов на кадрах роликов: команда, ХП, ресурс, уровень.

Зачем. Этап 1 второй модели (docs/entity-model-plan.md): классический
читатель плашек — базовая точка и псевдо-метки. Пишет JSONL «кадр →
плашки» и сводку: сколько плашек на кадр, распределение уровней и ХП,
время на кадр.

Пример:
    python tools/read_bars.py                      # все dataset/videos/*/frames
    python tools/read_bars.py 58w57eJ5Qks --sheet  # + лист вырезов для проверки глазами
    # → runs/health_bars/<имя>.jsonl, <имя>.json[, <имя>.jpg]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from lolvision import health_bars as hb
from lolvision.paths import DATASET, RUNS

LAYOUT = DATASET / "layouts" / "spectator-volibear-challenger.json"
OUT = RUNS / "health_bars"


# Части маски раскладки, которые для плашек не закрываем. Объявления
# появляются ненадолго, а их прямоугольник (y 189–297 посередине) постоянно
# закрывал бы полосу, где плашки бывают чаще всего; маска табло заходит
# ниже самого табло. Ложные находки там отсекает проверка квадрата уровня
# (на dataset/videos: +321 плашка, из них ложная одна).
OPEN_FOR_BARS = ("scoreboard_and_objective_icons", "announcements", "announcement_emblem")


def scene_keep(layout: dict, w: int = 1920, h: int = 1080) -> np.ndarray:
    """1 там, где ищем плашки, 0 под постоянным HUD раскладки."""
    k = np.ones((h, w), np.uint8)
    for name, (x0, y0, x1, y1) in layout["mask"].items():
        if name in OPEN_FOR_BARS:
            continue
        k[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", help="ID роликов из dataset/videos (по умолчанию все)")
    ap.add_argument("--sheet", action="store_true", help="лист вырезов с прочитанным")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    vids = args.videos or sorted(p.name for p in (DATASET / "videos").iterdir() if p.is_dir())
    frames = [f for v in vids for f in sorted((DATASET / "videos" / v / "frames").glob("*.jpg"))]
    if not frames:
        raise SystemExit("кадров нет: dataset/videos/<id>/frames собираются локально")
    tmpl, mask = hb.load_template()
    levels = hb.LevelReader()
    keep = scene_keep(json.loads(LAYOUT.read_text(encoding="utf-8")))

    OUT.mkdir(parents=True, exist_ok=True)
    name = args.name or ("all" if not args.videos else "+".join(vids))
    rows, times, crops = [], [], []
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for f in frames:
            rgb = np.asarray(Image.open(f).convert("RGB"))
            t = time.perf_counter()
            bars = hb.find_bars(rgb, tmpl, mask, keep, levels=levels)
            times.append(time.perf_counter() - t)
            rec = {"frame": f.name, "bars": [
                {"x": b.x, "y": b.y, "score": round(b.score, 3), "team": b.team,
                 "hp": round(b.hp, 3), "resource": round(b.resource, 3),
                 "level": b.level, "level_sim": round(b.level_sim, 3)} for b in bars]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            if args.sheet:
                for b in bars:
                    crops.append((b, rgb[max(0, b.y - 18):b.y + 26, max(0, b.x - 4):b.x + 127]))

    n = np.array([len(r["bars"]) for r in rows])
    allb = [b for r in rows for b in r["bars"]]
    summary = {
        "frames": len(rows), "bars": len(allb),
        "frames_with_bars": int((n > 0).sum()),
        "bars_per_frame": dict(zip(*[a.tolist() for a in np.unique(n, return_counts=True)])),
        "teams": {t: sum(b["team"] == t for b in allb) for t in ("ORDER", "CHAOS")},
        "levels": dict(zip(*[a.tolist() for a in np.unique([b["level"] for b in allb],
                                                           return_counts=True)])) if allb else {},
        "hp_median": round(float(np.median([b["hp"] for b in allb])), 3) if allb else None,
        "ms_per_frame": {"median": round(float(np.median(times)) * 1000, 1),
                         "p90": round(float(np.percentile(times, 90)) * 1000, 1)},
    }
    (OUT / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    if args.sheet and crops:
        rng = np.random.default_rng(0)
        sel = [crops[i] for i in rng.permutation(len(crops))[:60]]
        sheet = Image.new("RGB", (262 * 5, 110 * 12))
        for i, (b, c) in enumerate(sel):
            cell = Image.new("RGB", (262, 110))
            cell.paste(Image.fromarray(c).resize((c.shape[1] * 2, c.shape[0] * 2)), (0, 0))
            ImageDraw.Draw(cell).text(
                (4, 92), f"{b.team} L{b.level} hp {b.hp:.2f} res {b.resource:.2f}",
                fill=(255, 255, 0))
            sheet.paste(cell, ((i % 5) * 262, (i // 5) * 110))
        sheet.save(OUT / f"{name}.jpg", quality=88)


if __name__ == "__main__":
    main()
