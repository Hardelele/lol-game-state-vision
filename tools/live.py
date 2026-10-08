"""Смотрелка прямо из потока: ролик не скачивается и не собирается в датасет.

Раньше, чтобы посмотреть ответ модели на новом ролике, надо было собрать по
нему датасет (минуты) и дождаться прогона. Здесь кадры идут потоком: ролик
читается через yt-dlp, распаковывается видеокартой, с миникарты снимается
истинное положение камеры, модель тут же отвечает, и кадры по одному
появляются в смотрелке.

Что где узко (замерено на этой машине):

    распаковка AV1 1080p60 на видеокарте   24x реального времени
    распаковка на процессоре               5.9x
    модель                                 5118 кадров/с

То есть всё упирается в распаковку: тридцатиминутный матч проходит целиком
примерно за две с половиной минуты, а первые кадры видны через несколько
секунд после нажатия.

Перемотки нет: YouTube душит любое обращение к медиа-адресу, кроме
последовательного чтения самим yt-dlp. Проверено — ffmpeg с `-ss` по прямому
адресу за 100 секунд не получил ни байта, `yt-dlp --download-sections` висит
так же. Поэтому «начать с 10-й минуты» означает промотать поток распаковкой,
а это примерно 25 секунд на каждые десять минут ролика.

Ничего не оседает на диске: кадры живут в памяти процесса, пока сеанс открыт.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paths import ROOT
from activations import load_model
from build_coords import probe_url, frames as frame_stream
from coord_model import expected_point, peak_point, spread
from mask_frames import mask_boxes
from minimap_camera import MM, white_mask, find_box, estimate_box
from scene_data import crop_box

LAYOUT = ROOT / "dataset" / "layouts" / "spectator-volibear-challenger.json"
RUNS = ROOT / "runs"
STORE_W = 720            # как в датасете: кадр показывается, а не хранится
MINI_STORE = 256
WARMUP = 12              # кадров на оценку размера рамки вьюпорта
MAX_FRAMES = 5400        # потолок памяти: полтора часа при 1 кадр/с
MAP_UNITS = 14800


class Session:
    """Один ролик, читаемый потоком. Живёт в памяти, пока его не закрыли."""

    def __init__(self, vid: str, run: str, fps: float = 1.0,
                 start: float = 0.0, device: str = "cuda"):
        self.vid, self.run, self.fps, self.start = vid, run, fps, start
        self.device = device if torch.cuda.is_available() else "cpu"
        self.rows: list[dict] = []
        self.scene: dict[int, bytes] = {}
        self.mini: dict[int, bytes] = {}
        self.heat: dict[int, np.ndarray] = {}
        self.state = "готовлюсь"
        self.note = ""
        self.duration = 0.0
        self.title = ""
        self.started = time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    # ---------- чтение состояния ----------

    def status(self, since: int = 0) -> dict:
        with self._lock:
            rows = self.rows[since:]
            n = len(self.rows)
        el = time.time() - self.started
        seen = (self.start + n / self.fps) if n else self.start
        return {"id": self.vid, "run": self.run, "state": self.state,
                "note": self.note, "fps": self.fps, "start": self.start,
                "title": self.title, "duration": round(self.duration),
                "count": n, "since": since, "frames": rows,
                "elapsed": round(el, 1),
                "speed": round(seen / el, 1) if el > 1 else None,
                "map_units": MAP_UNITS}

    def stop(self) -> None:
        self._stop.set()

    # ---------- работа ----------

    def _fail(self, why: str) -> None:
        self.state, self.note = "сбой", why

    def _run(self) -> None:
        try:
            self._work()
        except Exception as exc:                              # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")

    def _work(self) -> None:
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        url = f"https://www.youtube.com/watch?v={self.vid}"
        self.state = "спрашиваю YouTube"
        w, h, dur, _, info = probe_url(url)
        self.duration, self.title = dur, info.get("title") or self.vid

        cx0, cy0, cx1, cy1 = crop_box((w, h), layout, "dense")
        store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
        boxes = mask_boxes((w, h), layout)
        m0, n0, m1, n1 = layout["labelling"]["minimap"]
        mmbox = (int(m0 * w), int(n0 * h), int(m1 * w), int(n1 * h))

        ck = RUNS / self.run / "model.pt"
        if not ck.is_file():
            return self._fail(f"нет модели {ck}")
        model, size = load_model(str(ck))
        model = model.to(self.device)
        self.note = (f"{w}x{h}, {dur / 60:.0f} мин · вход модели "
                     f"{size[0]}x{size[1]} · {self.device}")
        self.state = "идёт"

        # Размер рамки вьюпорта постоянен внутри ролика, но по первому кадру
        # его не оценить: в начале заставка. Копим первые кадры, оцениваем по
        # ним и только потом отдаём — иначе первые метки уехали бы.
        warm: list[tuple[np.ndarray, Image.Image, Image.Image]] = []
        bw = bh = None

        for fr in frame_stream(url, w, h, self.fps, True,
                                skip=self.start):
            if self._stop.is_set() or len(self.rows) >= MAX_FRAMES:
                break
            img = Image.fromarray(fr)
            mini = img.crop(mmbox).resize((MM, MM), Image.LANCZOS)
            mask = white_mask(np.asarray(mini, dtype=np.float32))
            scene = img.copy()
            for bx in boxes:
                scene.paste((0, 0, 0), bx)
            scene = scene.crop((cx0, cy0, cx1, cy1)).resize(
                (STORE_W, store_h), Image.LANCZOS)

            if bw is None:
                warm.append((mask, scene, mini))
                if len(warm) < WARMUP:
                    continue
                bw, bh = estimate_box([m for m, _, _ in warm])
                self.note += f" · рамка {bw}x{bh}"
                for m, s, mi in warm:
                    self._emit(m, s, mi, bw, bh, model, size)
                warm.clear()
                continue
            self._emit(mask, scene, mini, bw, bh, model, size)

        if self.state == "идёт":
            self.state = "остановлено" if self._stop.is_set() else "ролик кончился"

    def _emit(self, mask, scene: Image.Image, mini: Image.Image,
              bw: int, bh: int, model, size) -> None:
        """Один кадр: истина с миникарты, ответ модели, картинки в память."""
        x, y, q = find_box(mask, bw, bh)
        cx, cy = (x + bw / 2) / MM, (y + bh / 2) / MM

        a = np.asarray(scene.resize(size, Image.BILINEAR),
                       dtype=np.float32) / 255.0
        t = torch.from_numpy(a.transpose(2, 0, 1) * 2 - 1)[None]
        g = model.grid
        with torch.no_grad():
            lo = model(t.to(self.device)).float()       # (1, g*g)
            px, py = expected_point(lo, g)[0].tolist()
            qx, qy = peak_point(lo, g)[0].tolist()
            sp = float(spread(lo, g)[0])
            hm = torch.softmax(lo, 1).reshape(g, g)

        i = len(self.rows)
        row = {"i": i, "t": round(self.start + i / self.fps, 2),
               "cx": round(cx, 4), "cy": round(cy, 4),
               "px": round(px, 4), "py": round(py, 4),
               "qx": round(qx, 4), "qy": round(qy, 4),
               "err": round(float(np.hypot(cx - px, cy - py)), 4),
               "spread": round(sp, 3), "q": round(q, 3), "label": None}
        with self._lock:
            self.scene[i] = _jpeg(scene, 85)
            self.mini[i] = _jpeg(mini.resize((MINI_STORE, MINI_STORE)), 82)
            self.heat[i] = hm.cpu().numpy().astype(np.float32)
            self.rows.append(row)


def _jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def heat_png(h: np.ndarray, size: int = 320) -> bytes:
    """Тепловая карта в ту же палитру, что и в обычной смотрелке."""
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


# ---------- реестр сеансов ----------

_sessions: dict[str, Session] = {}
_reg = threading.Lock()


def start(vid: str, run: str, fps: float = 1.0, start_sec: float = 0.0
          ) -> dict:
    """Открыть сеанс. Одновременно работает один: распаковка занимает карту."""
    with _reg:
        for other in list(_sessions.values()):
            if other.vid != vid:
                other.stop()
        # Кадры закрытого сеанса ещё можно листать, но держать их все вечно
        # нельзя: две тысячи кадров — это около 120 МБ. Оставляем последние
        # два ролика, остальные забываем.
        while len(_sessions) > 2:
            old_id = min(_sessions, key=lambda k: _sessions[k].started)
            if old_id == vid:
                break
            _sessions.pop(old_id).stop()
        old = _sessions.get(vid)
        if old and old.state in ("идёт", "готовлюсь", "спрашиваю YouTube"):
            return old.status()
        s = Session(vid, run, fps, start_sec)
        _sessions[vid] = s
    return s.status()


def get(vid: str) -> Session | None:
    return _sessions.get(vid)


def stop(vid: str) -> None:
    s = _sessions.get(vid)
    if s:
        s.stop()


def listing() -> list[dict]:
    return [{"id": s.vid, "state": s.state, "count": len(s.rows),
             "run": s.run, "title": s.title} for s in _sessions.values()]
