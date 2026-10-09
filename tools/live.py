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

Путь кадра в сеансе (замер на потоке 58w57eJ5Qks, 1 кадр/с, видеокарта
свободна, медиана): маска, обрезка и масштаб сцены на процессоре ~20 мс,
патчевая сеть ~4 мс, разбор и сведение голосов ~6 мс, сжатие картинок
~1.5 мс — около 30 кадров/с, наравне с CoordNet; поток при этом шёл
примерно в 13x реального времени.

Перемотки нет: YouTube душит любое обращение к медиа-адресу, кроме
последовательного чтения самим yt-dlp. Проверено — ffmpeg с `-ss` по прямому
адресу за 100 секунд не получил ни байта, `yt-dlp --download-sections` висит
так же. Поэтому «начать с 10-й минуты» означает промотать поток распаковкой,
а это примерно 25 секунд на каждые десять минут ролика.

Ничего не оседает на диске: кадры живут в памяти процесса, пока сеанс открыт.

Модель — любой прогон с model.pt под runs/, тип берётся из чекпоинта:
патчевая (tools/patch_model.py; по умолчанию — лучшая, DEFAULT_RUN) или
CoordNet (tools/coord_model.py). Новый чекпоинт патчевой модели того же
формата подхватывается без правки кода: достаточно передать его прогон.
На каждый кадр сеанс отдаёт точку камеры, разброс, тепловую карту по карте
и время по стадиям; у патчевой модели ещё долю согласных голосов и долю
патчей «сцена».
"""

from __future__ import annotations

import io
import json
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
from mask_frames import crop_box, mask_boxes
from minimap_camera import MM, BOX_W, BOX_H, white_mask, find_box, estimate_box
from patch_infer import PatchInfer, is_patch_checkpoint

LAYOUT = ROOT / "dataset" / "layouts" / "spectator-volibear-challenger.json"
RUNS = ROOT / "runs"
STORE_W = 720            # как в датасете: кадр показывается, а не хранится
MINI_STORE = 256
WARMUP = 12              # кадров с видимой рамкой на оценку её размера
WARMUP_MAX = 120         # дольше не ждём: оцениваем по тому, что есть
WARM_Q = 0.4             # «рамка видна»: качество при размере по умолчанию
BOX0 = (int(BOX_W * MM), int(BOX_H * MM))
MAX_FRAMES = 5400        # потолок памяти: полтора часа при 1 кадр/с
MAP_UNITS = 14800
DEFAULT_RUN = "patches/cnn-split16-notgame"
STAGES = ("decode", "prep", "net", "agg", "out")
TIMING_WINDOW = 200      # по скольким последним кадрам медиана времени


# ---------- модели ----------

class CoordInfer:
    """CoordNet: хранимая сцена → вход сети → ожидаемая точка тепловой карты."""

    kind = "coordnet"

    def __init__(self, path: Path, device: str):
        model, self.size = load_model(str(path))
        self.net = model.to(device)
        self.device = device
        self.grid = model.grid
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.trained_on = ck.get("trained_on", [])
        self.label = "CoordNet"

    def describe(self, scene_size) -> str:
        return f"{self.label}, вход {self.size[0]}x{self.size[1]}"

    def prepare(self, scene: Image.Image) -> torch.Tensor:
        a = np.asarray(scene.resize(self.size, Image.BILINEAR),
                       dtype=np.float32) / 255.0
        return torch.from_numpy(a.transpose(2, 0, 1) * 2 - 1)[None]

    @torch.no_grad()
    def predict(self, t: torch.Tensor):
        t0 = time.perf_counter()
        g = self.grid
        lo = self.net(t.to(self.device)).float()       # (1, g*g)
        px, py = expected_point(lo, g)[0].tolist()
        qx, qy = peak_point(lo, g)[0].tolist()
        sp = float(spread(lo, g)[0])
        hm = torch.softmax(lo, 1).reshape(g, g).cpu().numpy()
        ms = (time.perf_counter() - t0) * 1000
        # Разбора патчей и сведения голосов у CoordNet нет: всё время — сеть.
        return ({"px": px, "py": py, "qx": qx, "qy": qy, "spread": sp},
                hm, {"net": ms, "agg": 0.0})


def run_checkpoint(run: str) -> Path:
    """Прогон → его model.pt; только внутри runs/: чекпоинт — это код."""
    ck = (RUNS / run / "model.pt").resolve()
    if RUNS.resolve() not in ck.parents:
        raise ValueError(f"прогон вне runs/: {run}")
    return ck


def _kind_of(ck: dict) -> dict | None:
    if is_patch_checkpoint(ck):
        return {"kind": "patch", "label": f"патч-{ck.get('encoder', '?')}"}
    # У классификатора сцены (runs/scene) те же ключи, но есть «classes».
    if ("encoder" in ck and "head" in ck and "input_size" in ck
            and "classes" not in ck):
        return {"kind": "coordnet", "label": "CoordNet"}
    return None


def load_runner(path: Path, device: str):
    """Модель для потока по типу чекпоинта: патчевая или CoordNet."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    info = _kind_of(ck)
    if info is None:
        raise ValueError(f"неизвестный тип чекпоинта: {path.name}")
    return PatchInfer(path, device) if info["kind"] == "patch" else CoordInfer(path, device)


_kinds: dict[str, tuple[float, dict]] = {}


def models() -> list[dict]:
    """Все прогоны с model.pt, годные для потока, с типом модели.

    Тип читается из самого чекпоинта, а не из каталога (CoordNet для
    сравнения лежит и в runs/patches), и запоминается до смены файла.
    """
    out = []
    for p in sorted(RUNS.rglob("model.pt")):
        rid = p.parent.relative_to(RUNS).as_posix()
        mt = p.stat().st_mtime
        hit = _kinds.get(rid)
        if hit is None or hit[0] != mt:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            except Exception:                                 # noqa: BLE001
                continue
            info = _kind_of(ck)
            if info is None:
                continue
            info["trained_on"] = list(ck.get("trained_on", []))
            hit = _kinds[rid] = (mt, info)
        out.append({"id": rid, **hit[1], "default": rid == DEFAULT_RUN})
    return out


# ---------- сеанс ----------

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
        self.kind = ""
        self.model_label = ""
        self.trained_on: list[str] = []
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
            tail = self.rows[-TIMING_WINDOW:]
        el = time.time() - self.started
        seen = (self.start + n / self.fps) if n else self.start
        return {"id": self.vid, "run": self.run, "state": self.state,
                "kind": self.kind, "model": self.model_label,
                "trained_on": self.trained_on, "timing": timing(tail),
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

        ck = run_checkpoint(self.run)
        if not ck.is_file():
            return self._fail(f"нет модели runs/{self.run}/model.pt")
        model = load_runner(ck, self.device)
        self.kind, self.model_label = model.kind, model.label
        self.trained_on = list(model.trained_on)
        self.note = (f"{w}x{h}, {dur / 60:.0f} мин · "
                     f"{model.describe((STORE_W, store_h))} · {self.device}")
        self.state = "идёт"

        # Размер рамки вьюпорта постоянен внутри ролика, но по первому кадру
        # его не оценить: в начале заставка. Копим первые кадры, оцениваем по
        # ним и только потом отдаём — иначе первые метки уехали бы.
        warm: list[tuple] = []
        bw = bh = None
        n_good = 0

        stream = frame_stream(url, w, h, self.fps, True, skip=self.start)
        while True:
            # Время распаковки — ожидание следующего кадра из ffmpeg: сюда
            # входят и сеть до YouTube, и NVDEC, и прореживание до fps.
            t0 = time.perf_counter()
            fr = next(stream, None)
            t1 = time.perf_counter()
            if fr is None or self._stop.is_set() or len(self.rows) >= MAX_FRAMES:
                break
            img = Image.fromarray(fr)
            mini = img.crop(mmbox).resize((MM, MM), Image.LANCZOS)
            mask = white_mask(np.asarray(mini, dtype=np.float32))
            t2 = time.perf_counter()
            scene = img.copy()
            for bx in boxes:
                scene.paste((0, 0, 0), bx)
            scene = scene.crop((cx0, cy0, cx1, cy1)).resize(
                (STORE_W, store_h), Image.LANCZOS)
            ms = {"decode": (t1 - t0) * 1000,
                  "prep": (time.perf_counter() - t2) * 1000}

            if bw is None:
                warm.append((mask, scene, mini, ms))
                # Считаются только кадры, где рамка уже видна: по заставке
                # размер выходил меньше настоящего (58w57eJ5Qks: 56x32 вместо
                # 68x38), и все метки ролика смещались на ~380 ед.
                if find_box(mask, *BOX0)[2] >= WARM_Q:
                    n_good += 1
                if n_good < WARMUP and len(warm) < WARMUP_MAX:
                    continue
                good = [m[0] for m in warm if find_box(m[0], *BOX0)[2] >= WARM_Q]
                bw, bh = estimate_box(good or [m[0] for m in warm])
                self.note += f" · рамка {bw}x{bh}"
                for m, s, mi, t in warm:
                    self._emit(m, s, mi, bw, bh, model, t)
                warm.clear()
                continue
            self._emit(mask, scene, mini, bw, bh, model, ms)
        stream.close()                    # ffmpeg и yt-dlp не должны висеть
        if bw is None and warm and not self._stop.is_set():
            # Ролик кончился раньше, чем набралась оценка рамки.
            bw, bh = estimate_box([m[0] for m in warm])
            for m, s, mi, t in warm:
                self._emit(m, s, mi, bw, bh, model, t)

        if self.state == "идёт":
            self.state = "остановлено" if self._stop.is_set() else "ролик кончился"

    def _emit(self, mask, scene: Image.Image, mini: Image.Image,
              bw: int, bh: int, model, ms: dict) -> None:
        """Один кадр: истина с миникарты, ответ модели, картинки в память.

        `ms` — уже набежавшее время кадра (распаковка; маска, обрезка и
        масштаб сцены); сюда добавляются подготовка входа модели, сеть,
        сведение голосов и «ответ» — сжатие картинок и запись в сеанс.
        Метка с миникарты в путь модели не входит и не считается.
        """
        x, y, q = find_box(mask, bw, bh)
        cx, cy = (x + bw / 2) / MM, (y + bh / 2) / MM

        t0 = time.perf_counter()
        prep = model.prepare(scene)
        ms["prep"] += (time.perf_counter() - t0) * 1000
        out, hm, mt = model.predict(prep)
        ms.update(mt)

        t0 = time.perf_counter()
        i = len(self.rows)
        row = {"i": i, "t": round(self.start + i / self.fps, 2),
               "cx": round(cx, 4), "cy": round(cy, 4),
               **{k: _r(out[k], 4) for k in ("px", "py", "qx", "qy")},
               "err": _r(np.hypot(cx - out["px"], cy - out["py"]), 4),
               "spread": _r(out["spread"], 3), "q": round(q, 3)}
        for k in ("agree", "scene", "game"):
            if k in out:
                row[k] = _r(out[k], 3)
        for k in ("votes", "is_game"):
            if k in out:
                row[k] = out[k]
        sj = _jpeg(scene, 85)
        mj = _jpeg(mini.resize((MINI_STORE, MINI_STORE)), 82)
        ms["out"] = (time.perf_counter() - t0) * 1000
        row["ms"] = {k: round(ms.get(k, 0.0), 2) for k in STAGES}
        with self._lock:
            self.scene[i] = sj
            self.mini[i] = mj
            # float16: у патчевой модели сетка 64×64, и полтора часа потока
            # в float32 заняли бы почти 90 МБ одних тепловых карт.
            self.heat[i] = hm.astype(np.float16)
            self.rows.append(row)


def _r(v, n: int):
    """Округление для JSON; NaN (нет ни одного голоса) уходит как null."""
    v = float(v)
    return round(v, n) if np.isfinite(v) else None


def timing(rows: list[dict]) -> dict | None:
    """Медиана времени по стадиям за последние кадры, мс, и кадр/с пути
    модели — без распаковки: та ограничена потоком, а не моделью."""
    rs = [r["ms"] for r in rows if "ms" in r]
    if not rs:
        return None
    med = {k: round(float(np.median([r[k] for r in rs])), 2) for k in STAGES}
    work = sum(med[k] for k in STAGES if k != "decode")
    med["model_path"] = round(work, 2)
    med["fps_model"] = round(1000 / work, 1) if work > 0 else None
    med["n"] = len(rs)
    return med


def _jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def heat_png(h: np.ndarray, size: int = 320) -> bytes:
    """Тепловая карта в ту же палитру, что и в обычной смотрелке.

    Сетка любая: у CoordNet это softmax 32×32, у патчевой модели —
    гистограмма голосов 64×64; рисунок всё равно растягивается на карту.
    """
    h = h.astype(np.float32)
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


def start(vid: str, run: str = DEFAULT_RUN, fps: float = 1.0,
          start_sec: float = 0.0) -> dict:
    """Открыть сеанс. Одновременно работает один: распаковка занимает карту."""
    run = run or DEFAULT_RUN
    run_checkpoint(run)                  # путь вне runs/ отвергается сразу
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
            if old.run == run:
                return old.status()
            old.stop()                   # тот же ролик, но другой моделью
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
             "run": s.run, "kind": s.kind, "title": s.title}
            for s in _sessions.values()]
