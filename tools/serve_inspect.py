"""Локальный сервер смотрелки: кадры, предсказания и тепловые карты по HTTP.

Статические страницы со встроенными картинками упирались в собственный вес:
10 МБ на ролик и только каждый двенадцатый кадр. Здесь картинки отдаются с
диска по запросу, поэтому доступны все кадры сразу, страница открывается
мгновенно, а по ролику можно листать и проигрывать его как видео.

Зависимостей сверх тех, что уже есть в проекте, не нужно: сервер на
стандартной библиотеке, интерфейс — обычный HTML с JavaScript.

Запуск:
    python tools/serve_inspect.py
    # затем открыть http://127.0.0.1:8732
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import traceback
import threading
from collections.abc import Callable
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np

import activations as act
import channels
import explain_miss
import inspect_pages
import live
import video_side
from geometry import MAP_UNITS, apply_h, load_projection
from channels import youtube_id
from checkpoints import checkpoint_kind, read_checkpoint
from paths import COORDS_DATA, PROJECTION, RUNS, TOOLS, VIDEOS, run_dir

WEB = TOOLS / "webapp"

_lock = threading.Lock()


def run_id(run: str) -> str:
    """Прогон из адреса или тела запроса → его имя под runs/, проверенное.

    Имя приводится к одному виду (`a/./b` и `a/b` — один прогон), чтобы
    кеши ниже не заводили по записи на каждое написание.
    """
    return run_dir(run).relative_to(RUNS.resolve()).as_posix()


def discover_runs() -> list[dict]:
    """Прогоны — это каталоги с файлами predictions_<ролик>.csv.

    Типы моделей берутся одним проходом live.models() на весь список, а не
    по проходу runs/ на каждый прогон.
    """
    kinds = {m["id"]: m["kind"] for m in live.models()}
    out: dict[str, dict] = {}
    for d in sorted(RUNS.rglob("predictions_*.csv")):
        run = d.parent
        rid = run.relative_to(RUNS).as_posix()
        rec = out.get(rid)
        if rec is None:
            meta = {}
            f = run / "eval_coords.json"
            if f.exists():
                meta = json.loads(f.read_text(encoding="utf-8"))
            rec = out[rid] = {"id": rid, "title": rid,
                              "trained_on": meta.get("trained_on", []),
                              "kind": kinds.get(rid), "videos": []}
        rec["videos"].append(d.name[len("predictions_"):-len(".csv")])
    for r in out.values():
        r["videos"].sort()
    return list(out.values())


@lru_cache(maxsize=32)
def frames_of(run: str, vid: str) -> list[dict]:
    path = run_dir(run) / f"predictions_{vid}.csv"
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        rows.append({
            "i": int(r["idx"]), "t": float(r["t_sec"]),
            "cx": float(r["cx_true"]), "cy": float(r["cy_true"]),
            "px": float(r["cx_pred"]), "py": float(r["cy_pred"]),
            "qx": float(r["cx_peak"]), "qy": float(r["cy_peak"]),
            "err": float(r["err"]), "spread": float(r["spread"]),
            "q": float(r["quality"]),
        })
    rows.sort(key=lambda r: r["i"])
    return rows


@lru_cache(maxsize=8)
def heatmaps_of(run: str, vid: str):
    z = np.load(run_dir(run) / f"heatmaps_{vid}.npz")
    idx = {int(v): k for k, v in enumerate(z["idx"])}
    return z["heatmaps"].astype(np.float32), idx


@lru_cache(maxsize=4096)
def heat_png(run: str, vid: str, idx: int, size: int = 320) -> bytes:
    hm, pos = heatmaps_of(run, vid)
    return inspect_pages.heat_png(hm[pos[idx]], size)


def scene_file(vid: str, idx: int) -> Path:
    return COORDS_DATA / youtube_id(vid) / "scene" / f"{int(idx):06d}.jpg"


@lru_cache(maxsize=64)
def _ckpt_kind(path: str, mtime: float) -> str | None:
    """Тип модели («patch», «coordnet») по полям чекпоинта; до смены файла."""
    return checkpoint_kind(read_checkpoint(Path(path), mmap=True))


def ckpt_of(run: str) -> str:
    """Чекпоинт для разбора по слоям — он умеет только CoordNet."""
    p = live.run_checkpoint(run)
    if not p.is_file():
        raise FileNotFoundError(f"нет чекпоинта {p}")
    if _ckpt_kind(str(p), p.stat().st_mtime) == "patch":
        raise ValueError("разбор по слоям есть только у CoordNet, "
                         "у патчевой модели его нет")
    return str(p)


@lru_cache(maxsize=256)
def net_stages(run: str, vid: str, idx: int) -> list[dict]:
    """Этапы сети с формами и сводкой по каналам для конкретного кадра."""
    ck = ckpt_of(run)
    model, _ = act.load_model(ck)
    a = act.activations(ck, str(scene_file(vid, idx)))
    out = []
    prev = None
    for st in act.stage_plan(model):
        t = a.get(st["id"])
        if t is None:
            continue
        c, h, w = (t.shape + (1, 1))[:3] if t.ndim == 3 else (t.shape[0], 1, 1)
        alive = float((t.reshape(len(t), -1) > 0).mean()) if st["id"] != "input" else 1.0
        # Раскладка контактного листа повторяет activations.grid_png, чтобы
        # клик по клетке в интерфейсе попадал в тот же канал.
        cols, cell = act.sheet_cell(int(c))
        out.append({**st, "shape": [int(c), int(h), int(w)],
                    "channels": int(c), "alive": round(alive, 3),
                    "input_of": prev,
                    "grid": {"cols": cols, "rows": int(np.ceil(c / cols)),
                             "cw": cell, "ch": max(8, round(cell * h / w)),
                             "pad": 2}})
        prev = st["id"]
    return out


@lru_cache(maxsize=1)
def video_meta() -> dict:
    """Название, роль и сторона по каждому ролику.

    Без этого в интерфейсе виден только идентификатор вида 4AIX8QtRid4, и
    невозможно понять ни что за матч, ни участвовал ли он в обучении.
    """
    out = {}
    for p in sorted(VIDEOS.glob("*.info.json")):
        vid = p.stem.replace(".info", "")
        try:
            a = video_side.analyse(json.loads(p.read_text(encoding="utf-8")))
        except Exception:                                     # noqa: BLE001
            continue
        out[vid] = {"title": a["title"], "role": a["role"],
                    "side": a["side_ru"], "side_sure": a["confident"]}
    return out


@lru_cache(maxsize=1)
def projection() -> dict:
    if not PROJECTION.exists():
        return {}
    Hm, (W, H) = load_projection()
    return {"scene_size": [W, H], "homography": Hm.tolist(),
            "outline": apply_h(Hm, np.array([[0, 0], [W, 0], [W, H], [0, H]],
                                            float)).tolist()}


def cell_quads(cols: int, rows: int) -> list[dict]:
    """Контуры ячеек сетки в смещениях от центра камеры (с перспективой)."""
    p = projection()
    if not p:
        return []
    Hm = np.array(p["homography"], float)
    W, H = p["scene_size"]
    out = []
    for r in range(rows):
        for c in range(cols):
            xs = [c * W / cols, (c + 1) * W / cols]
            ys = [r * H / rows, (r + 1) * H / rows]
            quad = np.array([[xs[0], ys[0]], [xs[1], ys[0]],
                             [xs[1], ys[1]], [xs[0], ys[1]]], float)
            out.append({"id": f"{r + 1}.{c + 1}",
                        "pts": apply_h(Hm, quad).round(5).tolist()})
    return out


# ---------- маршруты ----------
#
# Таблица (метод, шаблон адреса, обработчик). Шаблон сверяется с путём
# целиком, группы уходят обработчику позиционно; порядок строк — порядок
# проверки. Обработчик получает запрос `h` и сам отвечает через него.

ROUTES: list[tuple[str, re.Pattern, Callable]] = []


def route(method: str, pattern: str):
    def deco(fn: Callable) -> Callable:
        ROUTES.append((method, re.compile(pattern), fn))
        return fn
    return deco


@route("GET", r"/|/index\.html")
def page(h) -> None:
    h._file(WEB / "index.html", cache=0)


@route("GET", r"/static/(.*)")
def static(h, name: str) -> None:
    if "/" in name or ".." in name:
        return h.fail(403, "нельзя")
    h._file(WEB / name, cache=0)


# ---------- каналы и забор роликов ----------

@route("GET", r"/api/channels")
def channel_list(h) -> None:
    h._json({"channels": channels.channels(), "have": channels.have_ids(),
             "jobs": channels.jobs()})


@route("GET", r"/api/channel")
def channel_videos(h) -> None:
    q = h.query
    handle = (q.get("handle") or [""])[0]
    if not handle:
        return h.fail(400, "не указан канал")
    h._json(channels.videos(handle, (q.get("url") or [None])[0],
                            int((q.get("limit") or ["60"])[0]),
                            (q.get("refresh") or [""])[0] == "1"))


@route("GET", r"/api/jobs")
def job_list(h) -> None:
    h._json({"jobs": channels.jobs(), "have": channels.have_ids()})


@route("GET", r"/api/video/([\w-]+)")
def video(h, vid: str) -> None:
    h._json(channels.video_info(youtube_id(vid)))


@route("POST", r"/api/channels/add")
def channel_add(h) -> None:
    h._json(channels.add_channel(h.body.get("url", "")))


@route("POST", r"/api/channels/drop")
def channel_drop(h) -> None:
    channels.drop_channel(h.body.get("handle", ""))
    h._json({"ok": True})


@route("POST", r"/api/ingest")
def ingest(h) -> None:
    h._json(channels.ingest(h.body.get("id", ""), h.body.get("fps") or 1.0))


# ---------- поток ----------

@route("GET", r"/api/live-models")
def live_models(h) -> None:
    with _lock:
        h._json({"models": live.models(h.server.live_run),
                 "default": h.server.live_run})


@route("GET", r"/api/live/([\w-]+)")
def live_status(h, vid: str) -> None:
    ses = live.get(vid)
    if ses is None:
        return h.fail(404, "сеанс не открыт")
    h._json(ses.status(int((h.query.get("since") or ["0"])[0])))


@route("POST", r"/api/live/start")
def live_start(h) -> None:
    b = h.body
    h._json(live.start(b.get("id", ""), b.get("run") or h.server.live_run,
                       b.get("fps") or 1.0, float(b.get("start") or 0.0)))


@route("POST", r"/api/live/stop")
def live_stop(h) -> None:
    live.stop(h.body.get("id", ""))
    h._json({"ok": True})


@route("GET", r"/img/live/([\w-]+)/(scene|mini)/(\d+)\.jpg")
def live_image(h, vid: str, kind: str, idx: str) -> None:
    ses = live.get(vid)
    store = (ses.scene if kind == "scene" else ses.mini) if ses else {}
    body = store.get(int(idx))
    if body is None:
        return h.fail(404, "кадра нет в сеансе")
    h._send(body, "image/jpeg", cache=3600)


@route("GET", r"/img/live/([\w-]+)/heat/(\d+)\.png")
def live_heat(h, vid: str, idx: str) -> None:
    ses = live.get(vid)
    hm = ses.heat.get(int(idx)) if ses else None
    if hm is None:
        return h.fail(404, "кадра нет в сеансе")
    h._send(inspect_pages.heat_png(hm), "image/png", cache=3600)


# ---------- прогоны с диска ----------

@route("GET", r"/api/index")
def index(h) -> None:
    with _lock:
        h._json({"runs": discover_runs(), "projection": projection(),
                 "videos": video_meta(), "map_units": MAP_UNITS})


@route("GET", r"/api/frames/(.+)/([\w-]+)")
def frames(h, run: str, vid: str) -> None:
    run, vid = run_id(run), youtube_id(vid)
    with _lock:
        rows = frames_of(run, vid)
    h._json({"video": vid, "run": run, "frames": rows})


@route("GET", r"/api/cells/(\d+)x(\d+)")
def cells(h, cols: str, rows: str) -> None:
    h._json(cell_quads(int(cols), int(rows)))


@route("GET", r"/explain/(.+)/([\w-]+)/(\d+)\.html")
def explain(h, run: str, vid: str, idx: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        body = explain_miss.build(run, vid, idx)
    h._send(body.encode("utf-8"), "text/html; charset=utf-8")


@route("GET", r"/api/net/(.+)/([\w-]+)/(\d+)")
def net(h, run: str, vid: str, idx: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        h._json({"stages": net_stages(run, vid, idx)})


@route("GET", r"/api/chan/(.+)/([\w-]+)/(\d+)/([\w.]+)")
def chan(h, run: str, vid: str, idx: str, sid: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
    h._json(act.channel_stats(a[sid]))


@route("GET", r"/img/(scene|mini)/([\w-]+)/(\d+)\.jpg")
def image(h, kind: str, vid: str, idx: str) -> None:
    h._file(COORDS_DATA / youtube_id(vid) / kind / f"{int(idx):06d}.jpg")


@route("GET", r"/img/net/(.+)/([\w-]+)/(\d+)/([\w.]+)/grid\.png")
def net_grid(h, run: str, vid: str, idx: str, sid: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
        body = act.grid_png(a[sid])
    h._send(body, "image/png", cache=3600)


@route("GET", r"/img/net/(.+)/([\w-]+)/(\d+)/([\w.]+)/ch(\d+)\.png")
def net_channel(h, run: str, vid: str, idx: str, sid: str, ch: str) -> None:
    run, vid, idx, ch = run_id(run), youtube_id(vid), int(idx), int(ch)
    with _lock:
        a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
        body = (act.rgb_png(a[sid]) if sid == "input" and ch < 0
                else act.channel_png(a[sid], ch))
    h._send(body, "image/png", cache=3600)


@route("GET", r"/img/scenefull/(.+)/([\w-]+)/(\d+)\.png")
def scene_full(h, run: str, vid: str, idx: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
        body = act.rgb_png(a["input"], width=560)
    h._send(body, "image/png", cache=3600)


@route("GET", r"/img/kernel/(.+)/([\w.]+)/(\d+)\.png")
def kernel(h, run: str, sid: str, ch: str) -> None:
    run = run_id(run)
    with _lock:
        body = act.kernel_png(ckpt_of(run), sid, int(ch))
    h._send(body, "image/png", cache=86400)


@route("GET", r"/img/heat/(.+)/([\w-]+)/(\d+)\.png")
def heat(h, run: str, vid: str, idx: str) -> None:
    run, vid, idx = run_id(run), youtube_id(vid), int(idx)
    with _lock:
        body = heat_png(run, vid, idx)
    h._send(body, "image/png", cache=86400)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # без шума в консоли на каждый кадр
        pass

    def _send(self, body: bytes, ctype: str, cache: int = 0) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Без явного указания браузер кеширует страницу и стили на своё
        # усмотрение и после правки продолжает показывать старую версию.
        # Картинки кадров неизменны, их кешировать нужно; разметку — нет.
        self.send_header("Cache-Control",
                         f"max-age={cache}" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, code: int, why: str) -> None:
        """Причина — в теле: в строке статуса HTTP кириллица не кодируется."""
        body = json.dumps({"error": why}, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, cache: int = 86400) -> None:
        if not path.is_file():
            self.fail(404, "нет файла")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(path.read_bytes(), ctype, cache)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _refuse(self, method: str) -> tuple[int, str] | None:
        """Запрос не из нашей же страницы — отказ.

        Сервер запускает yt-dlp, ffmpeg и сеансы на видеокарте и читает
        чекпоинты, а слушает localhost, куда может постучаться любой
        открытый в браузере сайт:

        * Host — только наш: иначе сайт с DNS rebinding (evil.com →
          127.0.0.1) читает ответы как свои;
        * Origin, если он есть, — тот же адрес, что и Host;
        * POST — только application/json: такой запрос браузер с чужой
          страницы без preflight не отправит, а OPTIONS сервер не знает.
        """
        host = (self.headers.get("Host") or "").lower()
        if host not in self.server.hosts:
            return 403, "чужой Host"
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() != f"http://{host}":
            return 403, "запрос с чужой страницы"
        if method == "POST":
            ctype = (self.headers.get("Content-Type") or "").split(";")[0]
            if ctype.strip().lower() != "application/json":
                return 415, "нужен Content-Type: application/json"
        return None

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        p = unquote(url.path)
        self.query = parse_qs(url.query)
        bad = self._refuse(method)
        if bad:
            return self.fail(*bad)
        try:
            self.body = self._body() if method == "POST" else {}
            for meth, rx, fn in ROUTES:
                if meth != method:
                    continue
                m = rx.fullmatch(p)
                if m:
                    return fn(self, *m.groups())
        except FileNotFoundError:
            return self.fail(404, "нет данных")
        except ValueError as exc:
            return self.fail(400, str(exc))
        except Exception as exc:                                 # noqa: BLE001
            traceback.print_exc()
            return self.fail(500, f"{type(exc).__name__}: {exc}")
        self.fail(404, "нет такого адреса")

    def do_GET(self) -> None:                                    # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:                                   # noqa: N802
        self._dispatch("POST")


class Server(ThreadingHTTPServer):
    """HTTP-сервер смотрелки: имена, по которым к нему можно прийти, и
    прогон для потока по умолчанию."""

    def __init__(self, host: str, port: int, live_run: str = live.DEFAULT_RUN):
        super().__init__((host, port), Handler)
        self.live_run = live_run
        names = {"127.0.0.1", "localhost", "[::1]",
                 f"[{host}]" if ":" in host else host}
        self.hosts = {n.lower() for n in names} | {f"{n}:{port}".lower()
                                                   for n in names}


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
