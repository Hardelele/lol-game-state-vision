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
import io
import json
import mimetypes
import re
import traceback
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np
from PIL import Image

import activations as act
import channels
import explain_miss
import live
import video_side

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "webapp"
DATA = ROOT / "data" / "coords"
RUNS = ROOT / "runs"
PROJ = ROOT / "dataset" / "layouts" / "projection.json"
MAP_UNITS = 14800

_lock = threading.Lock()


def discover_runs() -> list[dict]:
    """Прогоны — это каталоги с файлами predictions_<ролик>.csv."""
    out = []
    for d in sorted(RUNS.rglob("predictions_*.csv")):
        run = d.parent
        rid = str(run.relative_to(RUNS)).replace("\\", "/")
        rec = next((r for r in out if r["id"] == rid), None)
        if rec is None:
            meta = {}
            f = run / "eval_coords.json"
            if f.exists():
                meta = json.loads(f.read_text(encoding="utf-8"))
            rec = {"id": rid, "title": rid, "trained_on": meta.get("trained_on", []),
                   "videos": []}
            out.append(rec)
        rec["videos"].append(d.name[len("predictions_"):-len(".csv")])
    for r in out:
        r["videos"].sort()
    return out


@lru_cache(maxsize=32)
def frames_of(run: str, vid: str) -> list[dict]:
    path = RUNS / run / f"predictions_{vid}.csv"
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
    z = np.load(RUNS / run / f"heatmaps_{vid}.npz")
    idx = {int(v): k for k, v in enumerate(z["idx"])}
    return z["heatmaps"].astype(np.float32), idx


@lru_cache(maxsize=4096)
def heat_png(run: str, vid: str, idx: int, size: int = 320) -> bytes:
    hm, pos = heatmaps_of(run, vid)
    h = hm[pos[idx]]
    h = h / max(float(h.max()), 1e-9)
    big = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 0] = np.clip(255 * big ** 0.6, 0, 255)
    rgba[..., 1] = np.clip(90 * big ** 1.6, 0, 255)
    rgba[..., 2] = np.clip(40 * big ** 2.0, 0, 255)
    rgba[..., 3] = np.clip(225 * big ** 0.8, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def scene_file(vid: str, idx: int) -> Path:
    return DATA / vid / "scene" / f"{idx:06d}.jpg"


def ckpt_of(run: str) -> str:
    p = RUNS / run / "model.pt"
    if not p.is_file():
        raise FileNotFoundError(f"нет чекпоинта {p}")
    return str(p)


@lru_cache(maxsize=256)
def net_stages(run: str, vid: str, idx: int) -> list[dict]:
    """Этапы сети с формами и сводкой по каналам для конкретного кадра."""
    model, _ = act.load_model(ckpt_of(run))
    a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
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
def hand_labels() -> dict:
    """Ручные метки top/not_top: показываются рядом с ответом модели."""
    out = {}
    for p in sorted((ROOT / "dataset" / "videos").glob("*/frames.csv")):
        vid = p.parent.name
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r["label"]:
                out[f"{vid}:{int(r['t_sec'])}"] = r["label"]
    return out


@lru_cache(maxsize=1)
def video_meta() -> dict:
    """Название, роль и сторона по каждому ролику.

    Без этого в интерфейсе виден только идентификатор вида 4AIX8QtRid4, и
    невозможно понять ни что за матч, ни участвовал ли он в обучении.
    """
    out = {}
    for p in sorted((ROOT / "data" / "videos").glob("*.info.json")):
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
    if not PROJ.exists():
        return {}
    d = json.loads(PROJ.read_text(encoding="utf-8"))
    Hm = np.array(d["homography"], float)
    W, H = d["scene_size"]

    def to_map(pts):
        v = np.c_[pts, np.ones(len(pts))] @ Hm.T
        return (v[:, :2] / v[:, 2:3]).tolist()

    return {"scene_size": [W, H], "homography": d["homography"],
            "outline": to_map(np.array([[0, 0], [W, 0], [W, H], [0, H]], float))}


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
            v = np.c_[quad, np.ones(4)] @ Hm.T
            out.append({"id": f"{r + 1}.{c + 1}",
                        "pts": (v[:, :2] / v[:, 2:3]).round(5).tolist()})
    return out


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

    def do_POST(self) -> None:                                   # noqa: N802
        p = unquote(urlparse(self.path).path)
        try:
            body = self._body()
            if p == "/api/channels/add":
                return self._json(channels.add_channel(body.get("url", "")))
            if p == "/api/channels/drop":
                channels.drop_channel(body.get("handle", ""))
                return self._json({"ok": True})
            if p == "/api/live/start":
                return self._json(live.start(
                    body.get("id", ""), body.get("run") or "coords/full14",
                    float(body.get("fps") or 1.0),
                    float(body.get("start") or 0.0)))
            if p == "/api/live/stop":
                live.stop(body.get("id", ""))
                return self._json({"ok": True})
            if p == "/api/ingest":
                return self._json(channels.ingest(
                    body.get("id", ""), float(body.get("fps") or 1.0)))
        except ValueError as exc:
            return self.fail(400, str(exc))
        except Exception as exc:                                 # noqa: BLE001
            traceback.print_exc()
            return self.fail(500, f"{type(exc).__name__}: {exc}")
        self.fail(404, "нет такого адреса")

    def do_GET(self) -> None:                                    # noqa: N802
        p = unquote(urlparse(self.path).path)
        try:
            if p in ("/", "/index.html"):
                return self._file(WEB / "index.html", cache=0)
            if p.startswith("/static/"):
                name = p[len("/static/"):]
                if "/" in name or ".." in name:
                    return self.fail(403, "нельзя")
                return self._file(WEB / name, cache=0)

            if p == "/api/channels":
                return self._json({"channels": channels.channels(),
                                   "have": channels.have_ids(),
                                   "jobs": channels.jobs()})
            if p == "/api/channel":
                q = parse_qs(urlparse(self.path).query)
                handle = (q.get("handle") or [""])[0]
                if not handle:
                    return self.fail(400, "не указан канал")
                return self._json(channels.videos(
                    handle, (q.get("url") or [None])[0],
                    int((q.get("limit") or ["60"])[0]),
                    (q.get("refresh") or [""])[0] == "1"))
            m = re.fullmatch(r"/api/live/([\w-]+)", p)
            if m:
                ses = live.get(m.group(1))
                if ses is None:
                    return self.fail(404, "сеанс не открыт")
                q = parse_qs(urlparse(self.path).query)
                return self._json(ses.status(int((q.get("since") or ["0"])[0])))
            if p == "/api/jobs":
                return self._json({"jobs": channels.jobs(),
                                   "have": channels.have_ids()})
            m = re.fullmatch(r"/api/video/([\w-]+)", p)
            if m:
                return self._json(channels.video_info(m.group(1)))

            if p == "/api/index":
                with _lock:
                    return self._json({"runs": discover_runs(),
                                       "projection": projection(),
                                       "videos": video_meta(),
                                       "map_units": MAP_UNITS})
            m = re.fullmatch(r"/api/frames/(.+)/([\w-]+)", p)
            if m:
                run, vid = m.group(1), m.group(2)
                with _lock:
                    rows = frames_of(run, vid)
                labels = hand_labels()
                for r in rows:
                    r["label"] = labels.get(f"{vid}:{int(round(r['t']))}")
                return self._json({"video": vid, "run": run, "frames": rows})
            m = re.fullmatch(r"/api/cells/(\d+)x(\d+)", p)
            if m:
                return self._json(cell_quads(int(m.group(1)), int(m.group(2))))
            m = re.fullmatch(r"/explain/(.+)/([\w-]+)/(\d+)\.html", p)
            if m:
                run, vid, idx = m.group(1), m.group(2), int(m.group(3))
                with _lock:
                    body = explain_miss.build(run, vid, idx)
                return self._send(body.encode("utf-8"),
                                  "text/html; charset=utf-8")
            m = re.fullmatch(r"/api/net/(.+)/([\w-]+)/(\d+)", p)
            if m:
                run, vid, idx = m.group(1), m.group(2), int(m.group(3))
                with _lock:
                    return self._json({"stages": net_stages(run, vid, idx)})
            m = re.fullmatch(r"/api/chan/(.+)/([\w-]+)/(\d+)/([\w.]+)", p)
            if m:
                run, vid, idx, sid = (m.group(1), m.group(2), int(m.group(3)),
                                      m.group(4))
                with _lock:
                    a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
                return self._json(act.channel_stats(a[sid]))

            m = re.fullmatch(r"/img/live/([\w-]+)/(scene|mini)/(\d+)\.jpg", p)
            if m:
                ses = live.get(m.group(1))
                store = (ses.scene if m.group(2) == "scene" else ses.mini) if ses else {}
                body = store.get(int(m.group(3)))
                if body is None:
                    return self.fail(404, "кадра нет в сеансе")
                return self._send(body, "image/jpeg", cache=3600)
            m = re.fullmatch(r"/img/live/([\w-]+)/heat/(\d+)\.png", p)
            if m:
                ses = live.get(m.group(1))
                h = ses.heat.get(int(m.group(2))) if ses else None
                if h is None:
                    return self.fail(404, "кадра нет в сеансе")
                return self._send(live.heat_png(h), "image/png", cache=3600)

            m = re.fullmatch(r"/img/(scene|mini)/([\w-]+)/(\d+)\.jpg", p)
            if m:
                kind, vid, idx = m.group(1), m.group(2), int(m.group(3))
                return self._file(DATA / vid / kind / f"{idx:06d}.jpg")
            m = re.fullmatch(r"/img/net/(.+)/([\w-]+)/(\d+)/([\w.]+)/grid\.png", p)
            if m:
                run, vid, idx, sid = (m.group(1), m.group(2), int(m.group(3)),
                                      m.group(4))
                with _lock:
                    a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
                    body = act.grid_png(a[sid])
                return self._send(body, "image/png", cache=3600)
            m = re.fullmatch(
                r"/img/net/(.+)/([\w-]+)/(\d+)/([\w.]+)/ch(\d+)\.png", p)
            if m:
                run, vid, idx, sid, ch = (m.group(1), m.group(2), int(m.group(3)),
                                          m.group(4), int(m.group(5)))
                with _lock:
                    a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
                    body = (act.rgb_png(a[sid]) if sid == "input" and ch < 0
                            else act.channel_png(a[sid], ch))
                return self._send(body, "image/png", cache=3600)
            m = re.fullmatch(r"/img/scenefull/(.+)/([\w-]+)/(\d+)\.png", p)
            if m:
                run, vid, idx = m.group(1), m.group(2), int(m.group(3))
                with _lock:
                    a = act.activations(ckpt_of(run), str(scene_file(vid, idx)))
                    body = act.rgb_png(a["input"], width=560)
                return self._send(body, "image/png", cache=3600)
            m = re.fullmatch(r"/img/kernel/(.+)/([\w.]+)/(\d+)\.png", p)
            if m:
                run, sid, ch = m.group(1), m.group(2), int(m.group(3))
                with _lock:
                    body = act.kernel_png(ckpt_of(run), sid, ch)
                return self._send(body, "image/png", cache=86400)
            m = re.fullmatch(r"/img/heat/(.+)/([\w-]+)/(\d+)\.png", p)
            if m:
                run, vid, idx = m.group(1), m.group(2), int(m.group(3))
                with _lock:
                    body = heat_png(run, vid, idx)
                return self._send(body, "image/png", cache=86400)
        except FileNotFoundError:
            return self.fail(404, "нет данных")
        except Exception as exc:                                 # noqa: BLE001
            traceback.print_exc()
            return self.fail(500, f"{type(exc).__name__}: {exc}")
        self.fail(404, "нет такого адреса")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8732)
    ap.add_argument("--host", default="127.0.0.1")
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
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nсмотрелка на http://{args.host}:{args.port}  (Ctrl+C — остановить)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")


if __name__ == "__main__":
    main()
