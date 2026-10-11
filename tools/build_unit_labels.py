"""Псевдо-метки юнитов по полоскам ХП: чемпионы, миньоны, варды.

Зачем. Путь B плана второй модели (docs/entity-model-plan.md, 2.3): HUD
кадра — учитель детектора юнитов. Плашки чемпионов — lolvision/health_bars,
полоски миньонов и вардов — lolvision/unit_bars; тело — под полоской по
постоянным смещениям (unit_bars.BODY). Пишет JSONL «кадр → юниты» и, по
желанию, кадры с нарисованными метками для проверки глазами.

Пример:
    python tools/build_unit_labels.py --frames "dataset/videos/*/frames/*.jpg" --name dataset
    python tools/build_unit_labels.py --frames "data/units/frames/ibUVbSX7ARU/*.jpg" \
        --name ibUVbSX7ARU --workers 6
    python tools/build_unit_labels.py --frames "..." --name check --overlay 20
    # → data/units/labels/<имя>.jsonl[, runs/units/overlay/<имя>/*.jpg]
"""

from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from lolvision import health_bars as hb
from lolvision import unit_bars as ub
from lolvision.paths import DATA, RUNS
from lolvision.mask_frames import keep_grid
from read_bars import LAYOUT

LABELS = DATA / "units" / "labels"
OVERLAY = RUNS / "units" / "overlay"
COLORS = {"ORDER": (0, 220, 255), "CHAOS": (255, 60, 200)}

_STATE: dict = {}


def _init() -> None:
    _STATE["tmpl"], _STATE["mask"] = hb.load_template()
    _STATE["levels"] = hb.LevelReader()
    _STATE["keep"] = label_keep()


# Части маски раскладки, которые для юнитов открыты: чат и лента убийств —
# полупрозрачный текст поверх сцены, юниты под ними видны; объявления
# появляются ненадолго. Закрыть их — потерять треть сцены вместе с
# полосами «без меток» у их краёв (unit_targets.ignore_map).
OPEN_FOR_UNITS = ("chat_feed", "kill_feed", "announcements", "announcement_emblem")


def label_keep(w: int = 1920, h: int = 1080) -> np.ndarray:
    """Где метки считаются полными: сцена вне постоянного HUD (табло, боковые
    панели, таймеры, низ). Табло закрыто в отличие от read_bars: значки
    целей на нём дают ложные полоски миньонов. Тот же `keep` закрашивает
    HUD на входе детектора."""
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    layout = {**layout, "mask": {k: v for k, v in layout["mask"].items()
                                 if k not in OPEN_FOR_UNITS}}
    return keep_grid((w, h), layout).astype(np.uint8)


def label_frame(rgb: np.ndarray, tmpl, mask, keep, levels=None) -> list[dict]:
    """Юниты кадра 1080p: класс, команда, полоска, рамка тела, ХП."""
    units = []
    champs = hb.find_bars(rgb, tmpl, mask, keep, levels=levels)
    for b in champs:
        cx = b.x + hb.W / 2
        units.append({"kind": "champion", "team": b.team, "bar": [b.x, b.y, b.x + hb.W, b.y + hb.H],
                      "box": [round(v, 1) for v in ub.body_box("champion", cx, b.y)],
                      "hp": round(b.hp, 3), "level": b.level, "score": round(b.score, 3)})
    excl = [(b.x - 6, b.y - 4, b.x + hb.W + 6, b.y + hb.H + 4) for b in champs]
    for u in ub.find_unit_bars(rgb, keep, excl):
        # Полоски без команды (жёлтые, зелёные, фиолетовые) у наблюдателя —
        # почти всегда не миньоны: золотая кромка полос строений и монстров,
        # трава, карточки объявлений (лист проверки, docs/unit-detection.md).
        if not u.team:
            continue
        units.append({"kind": u.kind, "team": u.team, "color": u.color,
                      "bar": [u.x, u.y, u.x + ub.BAR_W, u.y + u.h],
                      "box": [round(v, 1) for v in ub.body_box(u.kind, u.cx, u.y)],
                      "hp": round(u.hp, 3)})
    return units


def _work(path: str) -> dict:
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except OSError as e:            # кадр, недописанный при обрыве извлечения
        return {"frame": path, "units": [], "error": str(e)}
    if rgb.shape[:2] != (1080, 1920):
        return {"frame": path, "units": [], "error": f"size {rgb.shape[1]}x{rgb.shape[0]}"}
    units = label_frame(rgb, _STATE["tmpl"], _STATE["mask"], _STATE["keep"], _STATE["levels"])
    return {"frame": Path(path).as_posix(), "units": units}


def draw(path: str, units: list[dict], out: Path, extra: list[dict] | None = None) -> None:
    """Кадр с полосками (тонко) и телами (рамка); `extra` — второй набор
    (ответы модели) пунктиром-крестиками."""
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for u in units:
        c = COLORS.get(u["team"], (255, 255, 0))
        d.rectangle(u["bar"], outline=(255, 255, 0))
        d.rectangle(u["box"], outline=c, width=2)
        d.text((u["box"][0] + 2, u["box"][1] + 2), u["kind"][0].upper(), fill=c)
    for p in extra or []:
        c = COLORS.get(p["team"], (255, 255, 0))
        x, y = p["xy"]
        d.line((x - 8, y - 8, x + 8, y + 8), fill=c, width=3)
        d.line((x - 8, y + 8, x + 8, y - 8), fill=c, width=3)
        d.text((x + 9, y - 6), f"{p['kind'][0]}{p['score']:.2f}", fill=c)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", required=True, help="glob кадров 1920×1080")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overlay", type=int, default=0, help="сколько кадров нарисовать")
    args = ap.parse_args()

    frames = sorted(glob.glob(args.frames))
    if not frames:
        raise SystemExit(f"нет кадров: {args.frames}")
    LABELS.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init) as pool:
        rows = pool.map(_work, frames, chunksize=8)
    with (LABELS / f"{args.name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        for u in r["units"]:
            kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    print(json.dumps({"frames": len(rows), "units": kinds}, ensure_ascii=False))
    if args.overlay:
        rng = np.random.default_rng(0)
        for i in rng.permutation(len(rows))[:args.overlay]:
            r = rows[i]
            draw(r["frame"], r["units"], OVERLAY / args.name / Path(r["frame"]).name)


if __name__ == "__main__":
    main()
