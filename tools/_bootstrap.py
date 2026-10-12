"""Корень репозитория в sys.path для скриптов из tools/.

`python tools/<скрипт>.py` кладёт в sys.path только саму tools/, а скрипты
импортируют пакеты lolvision и lolstream из корня. Каждый скрипт первым
делом делает `import _bootstrap` — и работает без установки проекта; после
`pip install -e .` эта строка ничего не меняет.

Корень ставится сразу за tools/, а не в конец: копия пакетов в рабочем
дереве важнее установленной где-то ещё.

Вывод — в UTF-8: консоль Windows в cp1251, и первый print со «→» ронял
build_coords (UnicodeEncodeError) посреди пересборки.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)

# Непечатное в консоли — «?», а не исключение посреди работы.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
