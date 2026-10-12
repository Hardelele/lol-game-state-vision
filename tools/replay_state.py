"""Журнал Live Client API → снимки `/state` по времени журнала (#804).

Зачем. Клиенту coach нужно прогонять макро-правила по записанным играм, а
не только по живой. Инструмент проигрывает журнал `allgamedata` (формат
`lolstream.live_client.read_journal`: журналы `data/live_client/*.jsonl`,
`data/own/<сеанс>/api.jsonl`) тем же путём, что сервис без сеанса live:
`live_client.replay` (Tracker) → `state_service.api_only_provider` (свой
MatchState только из API) → `build_snapshot` → `check_snapshot`.

Часы. Снимок такой, будто его отдал живой сервис в момент строки журнала:
и провайдер, и `build_snapshot` получают `now` = `t` строки, поэтому все
`age` и `sources.*.age` — относительно этой строки, а не «сейчас». Провайдер
зовётся на каждой строке (MatchState видит каждый ответ, как в живом
сервисе), снимок выводится раз в `--step` секунд времени журнала (unix-время
опроса; в игре оно идёт вровень с временем матча) и на последней строке.

Выход — JSONL, по строке на снимок: `{"t_journal": t, "snapshot": {...}}`.
Снимок не по контракту — ошибки в stderr, код выхода 1 (выход всё равно
пишется целиком).

Ники. `events` ников не содержит, но `me.riot_id` и `champions[].riot_id` —
ники. `--no-names` (по умолчанию включён; `--names` — оставить) меняет их на
"me" (свой игрок) и "player1".."playerN" (по порядку появления в журнале), а
во всех остальных строках снимка — вхождения полного riotId и имени до «#»
(целым словом). Так файл можно отдать в другой репозиторий как фикстуру.

Пример:
    python tools/replay_state.py data/live_client/20261012-171753.jsonl \\
        --out runs/replay/20261012-171753.jsonl --step 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream.live_client import JournalRow, LiveStatus, read_journal, replay
from lolstream.state_service import api_only_provider, build_snapshot, check_snapshot
from lolvision import ddragon

STEP = 5.0


class _Replayed:
    """Подставной LiveClient для `api_only_provider`: `status()` — состояние
    API на текущей строке журнала."""

    def __init__(self) -> None:
        self.current: LiveStatus | None = None

    def status(self) -> LiveStatus:
        assert self.current is not None
        return self.current


def snapshots(rows: Iterable[JournalRow], champion_ids: Iterable[str] | None,
              step: float = STEP) -> Iterator[tuple[float, dict, LiveStatus]]:
    """Строки журнала → (t строки, снимок, LiveStatus) раз в `step` с времени
    журнала и на последней строке. `step <= 0` — снимок на каждой строке."""
    client = _Replayed()
    now = {"t": 0.0}
    provide = api_only_provider(client, clock=lambda: now["t"])
    seq, due, last = 0, None, None
    for status, _new in replay(rows, champion_ids):
        now["t"] = status.t
        client.current = status
        inp = provide()                          # MatchState видит каждую строку
        if due is None or status.t >= due:
            seq += 1
            yield status.t, build_snapshot(inp, status.t, seq), status
            due = status.t + step if due is None else due + step
            while step > 0 and due <= status.t:  # дыра в журнале: сетку не догоняем
                due += step
            last = None
        else:
            last = (status, inp)
    if last is not None:                         # последняя строка — всегда снимок
        status, inp = last
        seq += 1
        yield status.t, build_snapshot(inp, status.t, seq), status


# ---------------------------------------------------------------- ники

class Anonymizer:
    """Ники → "me" / "playerN". Таблица копится по ходу журнала (`learn`):
    свой — по activePlayer, остальные — по порядку появления."""

    def __init__(self) -> None:
        self.alias: dict[str, str] = {}
        self._n = 0
        self._re: re.Pattern | None = None
        self._map: dict[str, str] = {}

    def learn(self, status: LiveStatus) -> None:
        game = status.game
        if game is None:
            return
        me = game.my_player
        if me is not None and me.riot_id not in self.alias:
            self.alias[me.riot_id] = "me"
            self._re = None
        for p in game.players:
            if p.riot_id not in self.alias:
                self._n += 1
                self.alias[p.riot_id] = f"player{self._n}"
                self._re = None

    def _pattern(self) -> re.Pattern | None:
        if self._re is None and self.alias:
            # полный riotId и имя до «#» целым словом; длинные — раньше
            keys = dict(self.alias)
            for k, v in self.alias.items():
                keys.setdefault(k.split("#", 1)[0], v)
            self._map = keys
            alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True) if k)
            self._re = re.compile(rf"(?<!\w)(?:{alt})(?!\w)")
        return self._re

    def text(self, s: str) -> str:
        pat = self._pattern()
        return s if pat is None else pat.sub(lambda m: self._map[m.group(0)], s)

    def apply(self, x: Any) -> Any:
        """Копия снимка без ников: значения `riot_id` — по таблице, во всех
        остальных строках (причины отказов и т. п.) — замена вхождений."""
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k == "riot_id" and isinstance(v, dict) and isinstance(v.get("value"), str):
                    v = {**v, "value": self.alias.get(v["value"], self.text(v["value"]))}
                out[self.text(k)] = self.apply(v)
            return out
        if isinstance(x, list):
            return [self.apply(v) for v in x]
        if isinstance(x, str):
            return self.text(x)
        return x


# ---------------------------------------------------------------- запуск

def run(journal: Path, out: Path, step: float = STEP, names: bool = False,
        champion_ids: Iterable[str] | None = None) -> tuple[int, list[str]]:
    """Журнал → файл снимков. Возвращает (число снимков, ошибки контракта)."""
    anon = None if names else Anonymizer()
    errors: list[str] = []
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for t, snap, status in snapshots(read_journal(journal), champion_ids, step):
            errors += [f"снимок {snap['seq']} (t={t:.3f}): {e}" for e in check_snapshot(snap)]
            if anon is not None:
                anon.learn(status)
                snap = anon.apply(snap)
            f.write(json.dumps({"t_journal": t, "snapshot": snap}, ensure_ascii=False) + "\n")
            n += 1
    return n, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("journal", type=Path, help="журнал allgamedata (JSONL read_journal)")
    ap.add_argument("--out", type=Path, required=True, help="выход JSONL")
    ap.add_argument("--step", type=float, default=STEP,
                    help="шаг снимков, с времени журнала (0 — каждая строка)")
    ap.add_argument("--no-names", dest="names", action="store_false",
                    help="ники → me / playerN (по умолчанию)")
    ap.add_argument("--names", dest="names", action="store_true", help="оставить ники")
    ap.add_argument("--ddragon", type=Path, default=ddragon.CACHE,
                    help="кэш Data Dragon (Id чемпионов)")
    ap.add_argument("--version", default=ddragon.DEFAULT_VERSION, help="версия Data Dragon")
    ap.set_defaults(names=False)
    a = ap.parse_args(argv)
    try:
        ids = ddragon.champions(a.version, a.ddragon)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"нет списка чемпионов Data Dragon в {a.ddragon}: {e}; {ddragon.HINT}",
              file=sys.stderr)
        return 2
    n, errors = run(a.journal, a.out, a.step, a.names, ids)
    for e in errors:
        print(e, file=sys.stderr)
    size = a.out.stat().st_size
    print(f"снимков {n}, {size / 1e6:.2f} МБ → {a.out}"
          + (f"; ошибок контракта {len(errors)}" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
