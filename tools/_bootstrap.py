"""Корень репозитория в sys.path для скриптов из tools/.

`python tools/<скрипт>.py` кладёт в sys.path только саму tools/, а скрипты
импортируют пакеты lolvision и lolstream из корня. Каждый скрипт первым
делом делает `import _bootstrap` — и работает без установки проекта; после
`pip install -e .` эта строка ничего не меняет.

Корень ставится сразу за tools/, а не в конец: копия пакетов в рабочем
дереве важнее установленной где-то ещё.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
