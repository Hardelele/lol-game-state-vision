"""Сеанс «свой экран» одной командой, без смотрелки: окно игры, Live Client API
и сервис состояния для coach (#801, #796).

Открывает live.start_screen (по умолчанию — как кнопка «свой экран» в
смотрелке: 2 кадра/с, захват auto), печатает адрес сервиса состояния
(`state_url`) или причину, почему его нет, и раз в 10 с — строку о сеансе:
состояние сеанса, источника (кадр есть / пауза / окна нет), API, вид, сколько
кадров. Ctrl+C — сеанс останавливается, и скрипт ждёт, пока он отпустит
захват, API и порт.

Смотрелка с открытым «своим экраном» держит порт сервиса состояния
(127.0.0.1:8733): тогда второй сеанс не запускается — сказано, что делать.

Миникарта ищется по эталону Ущелья, на другой карте (ARAM) — по рамке и по
прямоугольнику, запомненному в прошлой игре на Ущелье (%LOCALAPPDATA%/
lolstream/minimap_fit.json). `--minimap-scale` — масштаб миникарты из
настроек клиента (1.0 — стандартный): тогда без поиска (#803).

Запуск:
    python tools/live_screen.py
    python tools/live_screen.py --fps 2 --backend dxcam --run patches/cnn
    python tools/live_screen.py --minimap-scale 1.2
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Callable

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream import live
from lolstream.state_service import HOST, PORT

PERIOD = 10.0          # с между строками о сеансе
URL_WAIT = 10.0        # с: сколько ждать, пока сеанс поднимет сервис состояния
STOP_WAIT = 20.0       # с: сколько ждать остановки сеанса после Ctrl+C
BUSY_HINT = ("сервис состояния уже занят — вероятно, смотрелка держит сеанс «свой экран». "
             "Остановите его в смотрелке (или закройте её) и запустите снова; "
             "либо берите состояние из смотрелки: http://{host}:{port}/state")


def port_busy(host: str = HOST, port: int = PORT) -> bool:
    """Кто-то уже слушает порт сервиса состояния (смотрелка, другой запуск)."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def line(st: dict) -> str:
    """Строка о сеансе из Session.status()."""
    src = st.get("source_state") or {}
    key = src.get("state")
    if key is None:
        source = "кадров ещё нет"
    elif key == "ok":
        source = "кадр есть"
    else:
        source = f"{key}" + (f" ({src['reason']})" if src.get("reason") else "")
    api = st.get("api")
    if api is None:
        api_s = f"нет ({st.get('api_reason') or 'не подключён'})"
    else:
        api_s = api.get("state") or "?"
        if api.get("reason"):
            api_s += f" ({api['reason']})"
    view = st.get("view") or "?"
    out = (f"сеанс: {st.get('state')} · источник: {source} · API: {api_s} · "
           f"вид: {view} · кадров: {st.get('count', 0)}")
    if not st.get("active", True) and st.get("note"):
        out += f" · {st['note']}"
    return out


def wait_url(sess, wait: float, sleep: Callable[[float], None]) -> dict:
    """Ждать, пока сеанс поднимет сервис состояния (или скажет, почему нет)."""
    end = time.monotonic() + wait
    while True:
        st = sess.status(10**9)
        # «ещё не поднят» — временная причина: ждём дальше, это не отказ.
        pending = st.get("state_reason") == live.STATE_PENDING
        if st.get("state_url") or (st.get("state_reason") and not pending) \
                or not st.get("active", True) or time.monotonic() >= end:
            return st
        sleep(0.2)


def shutdown(sess, wait: float = STOP_WAIT) -> None:
    """Остановить сеанс и дождаться конца его потока."""
    live.stop(live.SCREEN_ID)
    if sess is not None:
        sess.join(wait)


def run(fps: float, backend: str, run_id: str, period: float = PERIOD,
        sleep: Callable[[float], None] = time.sleep,
        busy: Callable[[], bool] = port_busy, minimap_scale: float | None = None) -> int:
    """Сеанс до Ctrl+C или до его конца. → код выхода."""
    have = live.get(live.SCREEN_ID)
    if (have is not None and have.status(10**9).get("active")) or busy():
        print(BUSY_HINT.format(host=HOST, port=PORT), file=sys.stderr)
        return 2
    try:
        if minimap_scale is None:
            live.start_screen(run_id, fps, backend)
        else:
            live.start_screen(run_id, fps, backend, minimap_scale=minimap_scale)
    except (ValueError, FileNotFoundError) as exc:        # fps, бэкенд, прогон, масштаб
        print(f"сеанс не открыт: {exc}", file=sys.stderr)
        return 2
    sess = live.get(live.SCREEN_ID)
    try:
        st = wait_url(sess, URL_WAIT, sleep)
        if st.get("state_url"):
            print(f"сервис состояния: {st['state_url']}/state  (события — /events)")
        else:
            why = st.get("state_reason") or "сеанс ещё не поднял его"
            print(f"сервис состояния не поднят: {why}")
            if why.startswith("не поднят:"):               # порт занят чужим процессом
                print(BUSY_HINT.format(host=HOST, port=PORT), file=sys.stderr)
                shutdown(sess)
                return 2
        print("Ctrl+C — остановить")
        while True:
            st = sess.status(10**9)
            print(line(st), flush=True)
            if not st.get("active", True):
                return 0 if st.get("state") != live.State.FAILED.label else 1
            sleep(period)
    except KeyboardInterrupt:
        print("\nостанавливаю сеанс…", flush=True)
        shutdown(sess)
        print(f"остановлено: {sess.status(10**9).get('state')}")
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=2.0, help="кадров в секунду (как в смотрелке: 2)")
    ap.add_argument("--backend", default="auto", help="захват: auto / dxcam / mss")
    ap.add_argument("--run", default=live.DEFAULT_RUN,
                    help="прогон под runs/ (модель камеры; тип — из чекпоинта)")
    ap.add_argument("--minimap-scale", type=float, default=None,
                    help="масштаб миникарты из настроек клиента (1.0 — стандартный); "
                         "без него миникарта ищется в кадре")
    args = ap.parse_args(argv)
    return run(args.fps, args.backend, args.run, minimap_scale=args.minimap_scale)


if __name__ == "__main__":
    sys.exit(main())
