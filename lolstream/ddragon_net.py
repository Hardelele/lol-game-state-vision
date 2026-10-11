"""Скачивание для lolvision.ddragon: HTTPS через прокси из окружения.

lolvision в сеть не ходит (tests/test_boundary.py), поэтому функция
скачивания живёт здесь и передаётся в `lolvision.ddragon.load(fetch=...)`.
Data Dragon из этой сети доступен только через прокси: берётся
HTTPS_PROXY, а если его нет — локальный http://127.0.0.1:3128.
"""

from __future__ import annotations

import os
import time
import urllib.request

DEFAULT_PROXY = "http://127.0.0.1:3128"


def _opener() -> urllib.request.OpenerDirector:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or DEFAULT_PROXY
    return urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy,
                                                                    "http": proxy}))


def fetch(url: str, tries: int = 3, timeout: float = 30.0) -> bytes:
    """Тело ответа; до `tries` попыток с паузой — прокси иногда рвёт соединение."""
    opener = _opener()
    for i in range(tries):
        try:
            with opener.open(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(1.0 + i)
    raise RuntimeError("недостижимо")
