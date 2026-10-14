"""Ролик → снимки /state по кадрам, сети пачками на видеокарте (#843).

Пример:
    python tools/extract_states.py D:/Videos/lol-coach/game.mp4 --fps 1 --batch 16 \\
        --out D:/bc-data/states/game.jsonl.gz --roster roster.json
    python tools/extract_states.py https://www.youtube.com/watch?v=XXXX --out x.jsonl.gz

Запись на кадр — строка JSON: `video_t` (с от начала ролика), `game_t`
(время матча по HUD или null), `frame` (номер кадра), `state` — снимок той
же схемы, что /state (state_service.build_snapshot). Рядом — `<out>.meta.json`:
источник, стадии, скорость, сколько кадров посчитано мимо пачки.

Машину не душит: по умолчанию приоритет BelowNormal (наследуют ffmpeg и
yt-dlp) и не больше 4 потоков у torch, numpy и OpenCV (lolstream/limits.py).
Подробно — docs/extract-states.md.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  корень репозитория в sys.path
from lolstream import limits


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="файл ролика или ссылка (http/https — поток YouTube через yt-dlp)")
    ap.add_argument("--out", required=True, help="куда писать: .jsonl.gz или .jsonl")
    ap.add_argument("--fps", type=float, default=1.0, help="кадров в секунду ролика (1)")
    ap.add_argument("--batch", type=int, default=16,
                    help="кадров в пачке сетей (16); 1 — обычный последовательный путь live")
    ap.add_argument("--stages", default="default",
                    help="стадии через запятую: scene, camera, minimap, hud, units, names; "
                         "default = scene,camera,minimap,hud; можно «default,+units», «all,-names»")
    ap.add_argument("--roster", help="состав для вида игрока без API: файл или строка JSON "
                                     '{"ORDER": [5], "CHAOS": [5], "me": "Volibear"}')
    ap.add_argument("--start", type=float, default=0.0, help="с какой секунды ролика")
    ap.add_argument("--duration", type=float, help="сколько секунд ролика (по умолчанию — до конца)")
    ap.add_argument("--decode", choices=("auto", "gpu", "cpu"), default="auto",
                    help="распаковка: NVDEC, если ffmpeg его умеет (auto)")
    ap.add_argument("--threads", type=int, default=limits.DEFAULT_THREADS,
                    help=f"потоков у torch, numpy, OpenCV ({limits.DEFAULT_THREADS})")
    ap.add_argument("--hud-procs", type=int, default=2,
                    help="процессов чтения полосы HUD в пакетном пути (2; 0 — в потоке)")
    ap.add_argument("--normal-priority", action="store_true",
                    help="не понижать приоритет процесса")
    ap.add_argument("--keep-row", action="store_true",
                    help="писать и строку кадра live (row: камера, юниты, HUD, события)")
    ap.add_argument("--profile", action="store_true",
                    help="время по стадиям пути кадра (только с --batch 1) — в meta")
    ap.add_argument("--run", help="модель камеры: прогон под runs/ (по умолчанию — как в live)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    limits.env_limits(a.threads)                 # до импорта numpy и torch
    import json
    import time
    from pathlib import Path

    from lolstream import extract as ex
    from lolstream import live

    lim = limits.apply_limits(a.threads, priority=not a.normal_priority)
    stages = ex.parse_stages(a.stages)
    if a.profile and a.batch != 1:
        raise SystemExit("--profile меряет последовательный путь: нужен --batch 1")
    roster = ex.Roster.load(a.roster) if a.roster else None
    out = Path(a.out)
    w = ex.Writer(out)
    src = ex.open_source(a.src, a.fps, a.start, decode=a.decode)
    s = ex.OfflineSession(src, run=a.run or live.DEFAULT_RUN, stages=stages, batch=a.batch,
                          roster=roster, on_record=w, keep_row=a.keep_row,
                          hud_procs=a.hud_procs)
    prof = ex.Profiler(s) if a.profile else None
    end = None if a.duration is None else a.start + a.duration

    def frames():
        for fr, ms in ex.prefetch(src, 3 * a.batch + 2):
            if end is not None and fr.t >= end:
                return
            yield fr, ms

    t0 = time.time()
    try:
        s.process(frames())
    finally:
        w.close()
        if prof is not None:
            prof.close()
    el = time.time() - t0
    d = src.describe()
    meta = {"schema": 1, "src": a.src, "title": s.title, "duration": s.duration,
            "fps": a.fps, "start": a.start, "end": end, "batch": a.batch, "stages": list(stages),
            "roster": roster.to_dict() if roster else None, "decode": d.get("backend"),
            "frames": w.n, "wall_s": round(el, 1),
            "s_per_game_min": round(el / max(1e-9, w.n / a.fps / 60), 2) if w.n else None,
            "batch_ms": {k: round(v, 1) for k, v in s.times.items()},
            "misses": s.misses, "limits": lim, "note": s.note, "state": s.state.label}
    if prof is not None:
        meta["profile"] = prof.report(w.n)
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"{w.n} снимков за {el:.0f} с ({meta['s_per_game_min']} с на минуту ролика) → {out}")
    return 0 if s.state is live.State.ENDED else 1


if __name__ == "__main__":
    sys.exit(main())
