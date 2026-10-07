"""Дублирование вывода в лог внутри папки прогона.

Вывод обучения — часть результата: по нему видно, на чём учили и как шли
фолды. Поэтому скрипт пишет его сам, рядом с чекпоинтом, а не полагается на
перенаправление в оболочке, которое кладёт лог куда придётся.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout(path: Path) -> Iterator[None]:
    """Пока открыт контекст, всё напечатанное идёт и на экран, и в `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as log:
        original = sys.stdout
        sys.stdout = _Tee(original, log)
        try:
            yield
        finally:
            sys.stdout = original
