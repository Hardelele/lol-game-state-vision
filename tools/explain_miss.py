"""Разбор одного промаха: где именно ответ разошёлся с истиной.

Командная строка над lolstream/explain_miss.py (тот же отчёт отдаёт
смотрелка): HTML-страница в runs/inspect/.

Пример:
    python tools/explain_miss.py coords/heatmap olmTXkkUv58 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream.explain_miss import build
from lolvision.paths import INSPECT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("video")
    ap.add_argument("idx", type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    html_doc = build(args.run, args.video, args.idx)
    out = args.out or (INSPECT /
                       f"miss-{args.video}-{args.idx}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"отчёт ({out.stat().st_size / 1e6:.1f} МБ) → {out}")


if __name__ == "__main__":
    main()
