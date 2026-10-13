"""Паспорт выбора игры (#807): журналы Live Client API → data/loadouts/<дата>.jsonl, база частот.

Подкоманды:

* `record <журнал…>` — журналы `allgamedata` (формат
  `live_client.read_journal`: `data/live_client/*.jsonl`,
  `data/own/<сеанс>/api.jsonl`) → строка на игрока каждой игры с
  пометками нетипичного (`surprises`) в `--out-dir` (по умолчанию
  data/loadouts/). Та же игра уже записана — не дублируется. Сводка в
  stdout — чемпион, роль, что нетипично; **ников нет ни в файле, ни в
  выводе**. `--dry-run` — только сводка.
* `base <файлы.jsonl…>` — строки паспортов → база частот
  (`loadouts.Base`, docs/loadouts.md) в `--out` (по умолчанию
  data/loadouts/base.json). Источник — свои записи: малая выборка, это
  помечено в базе.

Примеры:
    python tools/loadouts.py record data/live_client/*.jsonl
    python tools/loadouts.py base data/loadouts/2026-*.jsonl
    python tools/loadouts.py record data/live_client/20261012-171753.jsonl --dry-run \\
        --base data/loadouts/base.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream import surprises
from lolstream.live_client import read_journal
from lolstream.loadouts import (Base, ItemCatalog, build_base, read_records, records_from_journal,
                                write_records)
from lolvision import ddragon, paths

OUT_DIR = paths.DATA / "loadouts"
BASE = OUT_DIR / "base.json"


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def summary(recs: list[dict]) -> list[str]:
    """Строки сводки игры без ников."""
    h = recs[0]
    lines = [f"игра {h['game_id']} · {h['game_start']} · {h['mode']} карта {h['map']} · "
             f"{h['duration'] / 60:.1f} мин · "
             f"{'конец GameEnd' if h['complete'] else 'без GameEnd'}"]
    for r in recs:
        sp = r.get("spells") or {}
        ks = ((r.get("runes") or {}).get("keystone") or {}).get("name")
        done = [str(c["item"]) for c in (r["items"].get("completed") or [])]
        lines.append(f"  {r['team']:5} {r['relation'] or '?':5} {r['champion'] or '?':12} "
                     f"{r['role'] or '—':7} {'+'.join(sp.get('families') or ['?'])} · {ks} · "
                     f"законченные: {' → '.join(done) or '—'} · итог {r['result'] or '?'}")
        for s in r.get("surprises") or []:
            lines.append(f"      ! {s['kind']}: {s['what']} (ожидалось {s['expected']}; "
                         f"{s['confidence']:.2f}; {s['basis']})")
    return lines


def cmd_record(a: argparse.Namespace) -> int:
    try:
        ids = ddragon.champions(a.version, a.ddragon)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"нет списка чемпионов Data Dragon в {a.ddragon}: {e}; {ddragon.HINT}",
              file=sys.stderr)
        return 2
    catalog, why = ItemCatalog.load(a.version, a.ddragon)
    if catalog is None:
        print(f"предметы не классифицируются: {why} (докачать: python tools/ddragon.py "
              f"--version {a.version} --items)", file=sys.stderr)
    rules = surprises.load_rules(a.rules)
    base, base_why = Base.load(a.base)
    skip = () if a.all_modes else None
    total = 0
    for j in a.journals:
        kw = {} if skip is None else {"skip_modes": skip}
        rec = records_from_journal(
            read_journal(j), ids, catalog, why,
            annotate=lambda rs: surprises.annotate(rs, rules, base, base_why),
            origin=f"journal:{Path(j).name}", **kw)
        print(f"{j}: игр {len(rec.games)}" + (f"; пропущено: {'; '.join(rec.skipped)}"
                                             if rec.skipped else ""))
        for recs in rec.games:
            print("\n".join(summary(recs)))
            if not a.dry_run:
                path, n = write_records(recs, a.out_dir)
                total += n
                print(f"  → {path}: {n} строк" + ("" if n else " (игра уже записана)"))
    if not a.dry_run:
        print(f"записано строк: {total}")
    return 0


def cmd_base(a: argparse.Namespace) -> int:
    recs = read_records(a.files)
    b = build_base(recs, "own", ", ".join(Path(f).name for f in a.files), a.patch)
    b.save(a.out)
    print(f"база: {b.games} игроко-игр, {len(b.entries)} пар чемпион+роль → {a.out}"
          + (f"\n{b.note}" if b.note else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="журналы API → паспорта выбора")
    r.add_argument("journals", nargs="+", type=Path)
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r.add_argument("--dry-run", action="store_true", help="только сводка, без записи")
    r.add_argument("--rules", type=Path, default=None, help="TOML правил (по умолчанию "
                   "lolstream/surprise_rules.toml)")
    r.add_argument("--base", type=Path, default=BASE, help="база частот (нет — только правила)")
    r.add_argument("--all-modes", action="store_true", help="писать и тренировку")
    r.add_argument("--ddragon", type=Path, default=ddragon.CACHE, help="кэш Data Dragon")
    r.add_argument("--version", default=ddragon.DEFAULT_VERSION, help="версия Data Dragon")
    r.set_defaults(fn=cmd_record)
    b = sub.add_parser("base", help="паспорта → база частот")
    b.add_argument("files", nargs="+", type=Path)
    b.add_argument("--out", type=Path, default=BASE)
    b.add_argument("--patch", default=None, help="патч, если все записи с одного")
    b.set_defaults(fn=cmd_base)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
