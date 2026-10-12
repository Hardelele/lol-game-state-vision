"""Запись своей игры для проверочного набора 2D (#800): кадры экрана + журнал Live Client API.

Что пишется в `data/own/<YYYYMMDD-HHMMSS>/` (docs/own-recordings.md):

* `full/<i:06d>.jpg` — полный кадр окна игры (JPEG, качество 90), `i` —
  номер ответа источника; на отказах (пауза, свёрнуто, окна нет) файла нет;
* `frames.jsonl` — строка на каждый ответ источника: `i`, `t` (unix-время
  снимка), `src_t` (время по шкале источника), `status`, `reason`, `file`,
  `size` — по `t` кадр сопоставляется с журналом API;
* `api.jsonl` — журнал `allgamedata` в формате `live_client.read_journal`:
  `{"t": unix, "code": 200 | текст ошибки, "body": текст | null}`, ~1 Гц;
* `manifest.json` — сводка записи, пишется атомарно раз в ~30 с и в конце.

Источник кадров — `lolstream.frame_source.ScreenSource`, опрос API —
`lolstream.live_client.LiveClient` в своём потоке с транспортом
`HttpTransport`, обёрнутым в журнал. Конец игры решает `Tracker` (состояние
`ended`: GameEnd или пропажа API после игры); запись останавливается через
`--end-grace` секунд (120) после него, по `--max-min` минутам (40) или по
Ctrl+C — манифест дописывается в любом случае.

Ничего не удаляется и не перезаписывается: занятый каталог — новый id.

    python tools/record_own_game.py              # окно игры, 2 кадра/с
    python tools/record_own_game.py --fps 4 --max-min 25
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from PIL import Image

from lolstream.frame_source import BACKENDS, ScreenSource, Status
from lolstream.live_client import (
    HttpTransport,
    LiveClient,
    LiveStatus,
    Response,
    State,
    Transport,
    ddragon_champion_ids,
)
from lolstream.video_stream import write_atomic
from lolvision import paths

MANIFEST_SCHEMA = 1
JPEG_QUALITY = 90


def new_record_dir(root: Path, now: float) -> Path:
    """Каталог новой записи `<YYYYMMDD-HHMMSS>`; занят — `-2`, `-3`, … Не перезаписывает."""
    base = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        d = root / (base if n == 1 else f"{base}-{n}")
        try:
            d.mkdir()
        except FileExistsError:
            continue
        return d
    raise RuntimeError(f"не нашлось свободного id записи в {root} для {base}")


class JournalTransport:
    """Транспорт, который пишет каждый ответ строкой журнала и отдаёт его дальше.

    `t` — unix-время перед запросом (как у `LiveClient.poll_once`). Ошибка
    соединения — `code` текстом ошибки, так её читает `read_journal`.
    Исключение транспорта тоже попадает в журнал, затем идёт наверх
    (`LiveClient` превратит его в `error`).
    """

    def __init__(self, inner: Transport, path: Path, clock: Callable[[], float] = time.time):
        self.inner, self.clock = inner, clock
        self._f = path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self.rows = 0

    def _write(self, t: float, code, body: str | None) -> None:
        with self._lock:
            if self._f.closed:
                return
            self._f.write(json.dumps({"t": t, "code": code, "body": body},
                                     ensure_ascii=False) + "\n")
            self._f.flush()
            self.rows += 1

    def __call__(self, endpoint: str) -> Response:
        t = self.clock()
        try:
            r = self.inner(endpoint)
        except Exception as e:                                # noqa: BLE001
            self._write(t, f"исключение транспорта: {type(e).__name__}: {e}", None)
            raise
        self._write(t, r.code if r.code is not None else (r.error or "нет ответа"), r.body)
        return r

    def close(self) -> None:
        with self._lock:
            self._f.close()


class Recorder:
    """Одна запись: цикл кадров в вызывающем потоке, опрос API — в потоке `LiveClient`.

    `source` — `FrameSource` (в тестах подставной), `transport` — транспорт
    Live Client (в тестах подставной), `clock` — unix-время, `stop` — внешний
    сигнал остановки. `end_grace` — сколько секунд писать после конца игры
    по API, `max_sec` — предел длины записи.
    """

    def __init__(self, out: Path, source, transport: Transport, *,
                 api_period: float = 1.0, champion_ids=None, end_grace: float = 120.0,
                 max_sec: float = 40 * 60, manifest_every: float = 30.0,
                 print_every: float = 10.0, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = print):
        self.out, self.source = out, source
        self.end_grace, self.max_sec = end_grace, max_sec
        self.manifest_every, self.print_every = manifest_every, print_every
        self.clock, self.log = clock, log
        self.stop = threading.Event()
        (out / "full").mkdir(parents=True, exist_ok=False)
        self.journal = JournalTransport(transport, out / "api.jsonl", clock)
        self.api = LiveClient(self.journal, period=api_period, champion_ids=champion_ids,
                              clock=clock)
        self.started = clock()
        self.finished: float | None = None
        self.stop_reason: str | None = None
        self.n = 0                                 # ответов источника
        self.saved = 0                             # записанных кадров
        self.gaps: Counter[str] = Counter()        # отказы по состоянию
        self.sizes: Counter[str] = Counter()       # размеры кадров «WxH»
        self.ended_at: float | None = None         # когда API впервые сказал ended
        self.api_seen: dict = {}                   # что известно из API о своём игроке
        self.last_api: LiveStatus | None = None

    # ---------- API ----------

    def _watch_api(self, now: float) -> None:
        """Последнее состояние API: свой игрок и отсчёт после конца игры."""
        st = self.api.status()
        self.last_api = st
        game = st.game
        if game is not None:
            me = game.my_player
            self.api_seen.update(game_mode=game.game_mode, game_time=game.game_time)
            if me is not None:
                self.api_seen.update(my_team=me.team, my_champion=me.champion,
                                     my_champion_reason=me.champion_reason,
                                     my_champion_raw=me.champion_raw)
            else:
                self.api_seen.setdefault("me_reason", game.me_reason)
        if st.state is State.ENDED:
            if self.ended_at is None:
                self.ended_at = now
                self.api_seen["ended_reason"] = st.reason
                self.log(f"API: конец игры ({st.reason}); пишу ещё {self.end_grace:.0f} с")
        elif st.state in (State.IN_GAME, State.LOADING) and self.ended_at is not None:
            self.ended_at = None                   # пошла новая игра — отсчёт сначала
            self.api_seen.pop("ended_reason", None)

    # ---------- кадры ----------

    def _frame(self) -> bool:
        """Один ответ источника → строка frames.jsonl (и кадр). False — источник кончился."""
        fr = self.source.read()
        t = self.clock()
        if fr.status is Status.ENDED:
            self.stop_reason = self.stop_reason or f"источник: {fr.reason}"
            return False
        i = self.n
        self.n += 1
        row = {"i": i, "t": t, "src_t": fr.t, "status": fr.status.key, "reason": fr.reason,
               "file": None, "size": None}
        if fr.ok:
            name = f"full/{i:06d}.jpg"
            try:
                Image.fromarray(fr.image).save(self.out / name, quality=JPEG_QUALITY)
            except (OSError, ValueError) as e:
                row.update(status=Status.NO_FRAME.key, reason=f"JPEG не записан: {e}")
                self.gaps[Status.NO_FRAME.key] += 1
            else:
                w, h = fr.size
                row.update(file=name, size=[w, h])
                self.saved += 1
                self.sizes[f"{w}x{h}"] += 1
        else:
            self.gaps[fr.status.key] += 1
        self._frames.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._frames.flush()
        return True

    # ---------- манифест и печать ----------

    def manifest(self) -> dict:
        d = self.source.describe() if hasattr(self.source, "describe") else {}
        st = self.last_api
        return {
            "schema": MANIFEST_SCHEMA,
            "id": self.out.name,
            "started": self.started, "started_iso": _iso(self.started),
            "finished": self.finished,
            "finished_iso": _iso(self.finished) if self.finished else None,
            "complete": self.finished is not None,
            "stop_reason": self.stop_reason,
            "fps": getattr(self.source, "fps", None),
            "capture": {"kind": d.get("kind"), "backend": d.get("backend"),
                        "backend_note": d.get("backend_note"), "title": d.get("title"),
                        "notes": d.get("notes", [])},
            "frames": {"responses": self.n, "saved": self.saved,
                       "gaps": sum(self.gaps.values()), "gaps_by_status": dict(self.gaps),
                       "sizes": dict(self.sizes), "jpeg_quality": JPEG_QUALITY},
            "api": {"rows": self.journal.rows, "period": self.api.period,
                    "last_state": st.state.value if st else None,
                    "last_reason": st.reason if st else None,
                    "game_ended_at": self.ended_at, **self.api_seen},
            "files": {"frames": "full/", "frames_log": "frames.jsonl", "api": "api.jsonl"},
        }

    def write_manifest(self) -> None:
        text = json.dumps(self.manifest(), ensure_ascii=False, indent=2) + "\n"
        write_atomic(self.out / "manifest.json", lambda f: f.write(text))

    def _print(self, now: float) -> None:
        st = self.last_api
        api = "—" if st is None else st.state.value + (f" ({st.reason})" if st.reason else "")
        gt = self.api_seen.get("game_time")
        clock = "—" if gt is None else f"{int(gt) // 60}:{int(gt) % 60:02d}"
        self.log(f"[{now - self.started:6.0f} с] кадров {self.saved}, отказов "
                 f"{sum(self.gaps.values())} {dict(self.gaps) or ''}, API {api}, матч {clock}")

    # ---------- цикл ----------

    def run(self) -> dict:
        self._frames = (self.out / "frames.jsonl").open("a", encoding="utf-8", newline="\n")
        self.api.start()
        last_manifest = last_print = self.started
        try:
            self.source.open()
            self.write_manifest()
            while not self.stop.is_set():
                if not self._frame():
                    break
                now = self.clock()
                self._watch_api(now)
                if self.ended_at is not None and now - self.ended_at >= self.end_grace:
                    self.stop_reason = (f"конец игры по API + {self.end_grace:.0f} с "
                                        f"({self.api_seen.get('ended_reason')})")
                    break
                if now - self.started >= self.max_sec:
                    self.stop_reason = f"предел длины {self.max_sec / 60:.0f} мин"
                    break
                if now - last_print >= self.print_every:
                    self._print(now)
                    last_print = now
                if now - last_manifest >= self.manifest_every:
                    self.write_manifest()
                    last_manifest = now
            if self.stop.is_set() and self.stop_reason is None:
                self.stop_reason = "остановлено"
        except KeyboardInterrupt:
            self.stop_reason = "Ctrl+C"
        finally:
            self.api.stop()
            self.source.close()
            self.journal.close()
            self._frames.close()
            self.last_api = self.api.status()
            self.finished = self.clock()
            self.write_manifest()
        return self.manifest()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=2.0, help="кадров в секунду (2)")
    ap.add_argument("--backend", default="auto", choices=("auto", *BACKENDS),
                    help="бэкенд захвата экрана (auto: dxcam, запасной mss)")
    ap.add_argument("--max-min", type=float, default=40.0,
                    help="предел длины записи, минут (40)")
    ap.add_argument("--end-grace", type=float, default=120.0,
                    help="сколько секунд писать после конца игры по API (120)")
    ap.add_argument("--api-period", type=float, default=1.0,
                    help="период опроса Live Client API, с (1)")
    ap.add_argument("--out-root", type=Path, default=paths.OWN,
                    help=f"куда класть записи ({paths.OWN})")
    a = ap.parse_args(argv)

    out = new_record_dir(a.out_root, time.time())
    source = ScreenSource(fps=a.fps, backend=a.backend)
    rec = Recorder(out, source, HttpTransport(), api_period=a.api_period,
                   champion_ids=ddragon_champion_ids(), end_grace=a.end_grace,
                   max_sec=a.max_min * 60)
    print(f"запись в {out}; остановка — Ctrl+C, предел {a.max_min:.0f} мин, "
          f"или {a.end_grace:.0f} с после конца игры")
    m = rec.run()
    f, api = m["frames"], m["api"]
    print(f"готово: {out}\n  {m['stop_reason']}; кадров {f['saved']}, отказов {f['gaps']} "
          f"{f['gaps_by_status']}; строк API {api['rows']}; сторона {api.get('my_team')}, "
          f"чемпион {api.get('my_champion')}; захват {m['capture']['backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
