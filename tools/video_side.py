"""Сторона записанного игрока — из описания ролика, а не из догадок.

Командная строка над lolstream/video_side.py: роль, итог и сторона по
data/videos/<ролик>.info.json, покрытие «роль × сторона».

Пример:
    python tools/video_side.py                 # все ролики с info.json
    python tools/video_side.py --id 58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream.video_side import ROLES, analyse, read_info
from lolvision.paths import VIDEOS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None, help="куда сложить JSON")
    args = ap.parse_args()

    ids = args.id or sorted(p.stem.replace(".info", "")
                            for p in VIDEOS.glob("*.info.json"))
    rows = []
    for vid in ids:
        try:
            rows.append(analyse(read_info(vid)))
        except Exception as exc:                              # noqa: BLE001
            print(f"  {vid}: {exc}")
    print(f"{'ролик':14s} {'роль':9s} {'итог':5s} {'сторона':9s} основание")
    for r in rows:
        mark = "" if r["confident"] else "  ← ненадёжно"
        print(f"{r['id']:14s} {str(r['role']):9s} {r['result']:5s} "
              f"{r['side_ru']:9s} {r['why']}{mark}")

    by = {}
    for r in rows:
        if r["confident"]:
            by.setdefault((r["role"], r["side_ru"]), []).append(r["id"])
    print("\nпокрытие роль × сторона:")
    for role in ROLES:
        cells = [f"{s}: {len(by.get((role, s), []))}" for s in ("синие", "красные")]
        if any(by.get((role, s)) for s in ("синие", "красные")):
            print(f"  {role:8s} " + ",  ".join(cells))

    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
