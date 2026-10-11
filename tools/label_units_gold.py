"""Ручная проверочная выборка юнитов: честный эталон вместо псевдо-меток.

Зачем. Детектор юнитов (#758, #773) меряется против псевдо-меток читателя
полосок, а тот сам пропускает около 30% миньонов: часть «промахов» сети —
ошибки учителя, часть «ложных» — настоящие юниты. Потолок такой проверки
ниже 100%, и по нему не понять, куда расти. Нужна выборка, где каждый
видимый юнит отмечен человеком.

Как. Кадры отложенных роликов (детектор на них не учился), по
`--per-video` с каждого, равномерно по времени. На каждом кадре заранее
стоят ответы текущей сети (порог ниже рабочего, в зоне ответов как в
live; что вне её — отмечает человек): человек правит, а не размечает с
нуля. Разметка — центр тела, класс, команда; хранится в dataset/units_gold/labels.jsonl (в репозитории:
это ручной труд), сохраняется после каждого кадра.

Пример:
    python tools/label_units_gold.py prepare            # очередь + предразметка (GPU, ~1 мин)
    python tools/label_units_gold.py serve              # http://127.0.0.1:8794
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision.paths import DATASET, ROOT

GOLD = DATASET / "units_gold"
QUEUE = GOLD / "queue.json"
LABELS = GOLD / "labels.jsonl"
HISTORY = GOLD / "labels.history.jsonl"   # журнал всех сохранений, только дописывается
UI = Path(__file__).resolve().parent / "labeler"
HOLDOUT = ("58w57eJ5Qks", "olmTXkkUv58", "5nBGNokOUDA", "4AIX8QtRid4")
# Классы эталона шире детектора: монстры и прочее (питомцы, клоны,
# ловушки) размечаются впрок, проверка детектора их пока пропускает.
KINDS = ("champion", "minion", "ward", "monster", "other")
TEAMS = ("ORDER", "CHAOS", "NEUTRAL")


# ---------- очередь ----------

def prepare(per_video: int, thr: float) -> None:
    import cv2
    import numpy as np
    from lolvision.unit_net import DEFAULT_NET, UnitNetRunner
    from lolvision.unit_targets import ignore_map, unit_keep

    units_labels = ROOT / "data" / "units" / "labels"
    frames = []
    for vid in HOLDOUT:
        rows = [json.loads(s)["frame"] for s in (units_labels / f"{vid}.jsonl").open(encoding="utf-8")]
        rows = sorted(r for r in rows if (ROOT / r).is_file())
        # Равномерно по времени, а не подряд: соседние кадры почти одинаковы.
        idx = np.linspace(0, len(rows) - 1, min(per_video, len(rows))).round().astype(int)
        frames += [rows[i] for i in sorted(set(idx))]
    keep = unit_keep(1920, 1080)
    # Зона ответов — как в live: вне её (HUD и полосы у его краёв) сеть не
    # училась и сыплет ложными; юнитов там человек отметит сам.
    runner = UnitNetRunner(DEFAULT_NET, keep=keep, valid=1 - ignore_map(keep), thr=thr)
    queue = []
    for k, f in enumerate(frames):
        rgb = cv2.imread(str(ROOT / f))[..., ::-1]
        if rgb.shape[:2] != (1080, 1920):
            rgb = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        pred = [{"kind": u.kind, "team": u.team, "x": round(u.x, 1), "y": round(u.y, 1),
                 "score": round(u.score, 3)} for u in runner.units(np.ascontiguousarray(rgb))]
        queue.append({"i": k, "frame": f, "video": Path(f).name.split("_t")[0], "pred": pred})
    GOLD.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps({"net": str(DEFAULT_NET.relative_to(ROOT)), "thr": thr,
                                 "frames": queue}, ensure_ascii=False, indent=0) + "\n",
                     encoding="utf-8")
    print(f"→ {QUEUE}: {len(queue)} кадров, предразметка {sum(len(q['pred']) for q in queue)} юнитов")


# ---------- хранилище ----------

class Store:
    """labels.jsonl: одна строка на кадр, последняя запись побеждает.
    Переписывается целиком через временный файл — файл маленький, а
    оборванная запись не должна портить уже сделанное."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if LABELS.exists():
            for s in LABELS.open(encoding="utf-8"):
                if s.strip():
                    r = json.loads(s)
                    self.rows[r["frame"]] = r

    def save(self, row: dict) -> None:
        with self.lock:
            # Сначала — в журнал: по нему восстанавливается любое состояние,
            # даже если labels.jsonl испортят или перезапишут.
            with HISTORY.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "
")
            self.rows[row["frame"]] = row
            tmp = LABELS.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in self.rows.values():
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, LABELS)


def clean(body: dict, frames: set[str]) -> dict:
    """Проверить и нормализовать присланную разметку кадра."""
    if body.get("frame") not in frames:
        raise ValueError("неизвестный кадр")
    status = body.get("status")
    if status not in ("done", "skip", "draft"):
        raise ValueError("status: done / skip / draft")
    units = []
    for u in body.get("units") or []:
        if u.get("kind") not in KINDS or u.get("team") not in TEAMS:
            raise ValueError("kind / team")
        units.append({"kind": u["kind"], "team": u["team"],
                      "x": round(min(1920.0, max(0.0, float(u["x"]))), 1),
                      "y": round(min(1080.0, max(0.0, float(u["y"]))), 1)})
    return {"frame": body["frame"], "status": status, "units": units,
            "note": str(body.get("note") or "")[:200], "sec": round(float(body.get("sec") or 0), 1),
            "saved": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------- сервер ----------

class Handler(BaseHTTPRequestHandler):
    server_version = "gold-labeler"

    def log_message(self, *a):        # тихо: клики сохраняются часто
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        srv = self.server
        if path == "/":
            return self._send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/labeler.js", "/labeler.css"):
            ctype = mimetypes.guess_type(path)[0] or "text/plain"
            return self._send(200, (UI / path[1:]).read_bytes(), f"{ctype}; charset=utf-8")
        if path == "/api/queue":
            return self._json({"kinds": KINDS, "teams": TEAMS, "net": srv.queue["net"],
                               "frames": srv.queue["frames"], "labels": srv.store.rows})
        if path.startswith("/frame/"):
            try:
                q = srv.queue["frames"][int(path[7:].split(".")[0])]
            except (ValueError, IndexError):
                return self._json({"error": "нет кадра"}, 404)
            return self._send(200, (ROOT / q["frame"]).read_bytes(), "image/jpeg")
        self._json({"error": "не найдено"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._json({"error": "не найдено"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            row = clean(json.loads(self.rfile.read(n)), self.server.frames)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        self.server.store.save(row)
        self._json({"ok": True, "saved": row["saved"]})


def serve(port: int) -> None:
    if not QUEUE.exists():
        raise SystemExit(f"нет {QUEUE}: сначала `python tools/label_units_gold.py prepare`")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    srv.frames = {q["frame"] for q in srv.queue["frames"]}
    srv.store = Store()
    done = sum(r["status"] in ("done", "skip") for r in srv.store.rows.values())
    print(f"разметка на http://127.0.0.1:{port}  ·  готово {done} из {len(srv.frames)}  (Ctrl+C — остановить)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="очередь кадров и предразметка сетью")
    p.add_argument("--per-video", type=int, default=100)
    p.add_argument("--thr", type=float, default=0.25)
    s = sub.add_parser("serve", help="страница разметки")
    s.add_argument("--port", type=int, default=8794)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.per_video, args.thr)
    else:
        serve(args.port)


if __name__ == "__main__":
    main()
