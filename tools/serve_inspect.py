"""Локальный сервер смотрелки: кадры, предсказания и тепловые карты по HTTP.

Запуск сервера из lolstream/serve_inspect.py (статика — lolstream/webapp/).

Запуск:
    python tools/serve_inspect.py
    # затем открыть http://127.0.0.1:8732
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolstream import live
from lolstream.serve_inspect import Server, discover_runs, projection
from lolvision.paths import RUNS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="127.0.0.1",
                    help="адрес, который слушать; запросы принимаются только "
                         "с Host этим именем или 127.0.0.1/localhost")
    ap.add_argument("--live-run", default=live.DEFAULT_RUN,
                    help="прогон под runs/ для потока по умолчанию; тип "
                         "модели (патчевая или CoordNet) берётся из чекпоинта")
    args = ap.parse_args()

    runs = discover_runs()
    if not runs:
        raise SystemExit(f"в {RUNS} не нашлось файлов predictions_*.csv")
    print(f"прогонов найдено: {len(runs)}")
    for r in runs:
        print(f"  {r['id']}: ролики {', '.join(r['videos'])}")
    if not projection():
        print("внимание: нет dataset/layouts/projection.json — "
              "трапеция видимой области и сетка ячеек не будут показаны")
    srv = Server(args.host, args.port, args.live_run)
    print(f"\nсмотрелка на http://{args.host}:{args.port}  (Ctrl+C — остановить)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")


if __name__ == "__main__":
    main()
