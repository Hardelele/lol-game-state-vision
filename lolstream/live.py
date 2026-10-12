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

Путь кадра в сеансе у патчевой модели: подготовка входа (обрезка, маска,
масштаб, вид сверху) на видеокарте ~1 мс, сеть ~4 мс, разбор и сведение
голосов ~6 мс, сжатие картинок ~1.5 мс. Раньше подготовка шла на процессоре
через PIL LANCZOS и одна стоила ~20 мс.

Перемотки нет: YouTube душит любое обращение к медиа-адресу, кроме
последовательного чтения самим yt-dlp. Проверено — ffmpeg с `-ss` по прямому
адресу за 100 секунд не получил ни байта, `yt-dlp --download-sections` висит
так же. Поэтому «начать с 10-й минуты» означает промотать поток распаковкой,
а это примерно 25 секунд на каждые десять минут ролика.

Ничего не оседает на диске: кадры живут в памяти процесса, пока сеанс открыт.

Модель — любой прогон с model.pt под runs/, тип берётся из чекпоинта:
патчевая (lolvision/patch_model.py; по умолчанию — лучшая, DEFAULT_RUN) или
CoordNet (lolvision/coord_model.py). Новый чекпоинт патчевой модели того же
формата подхватывается без правки кода: достаточно передать его прогон.
На каждый кадр сеанс отдаёт точку камеры, разброс, тепловую карту по карте
и время по стадиям; у патчевой модели ещё долю согласных голосов и долю
патчей «сцена». Сверх камеры на кадр — тип сцены, юниты основного вида
(unit_net.py) и чемпионы: миникарта, жив ли, имена над чемпионами
(live_champions.py); поверх них — контекст матча (match_state.py): память о
чемпионах между кадрами, смерти и возрождения.
"""

from __future__ import annotations

import enum
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lolvision.paths import DEFAULT_LAYOUT, RUNS, run_dir
from lolvision.checkpoints import checkpoint_kind, load_checkpoint, read_checkpoint
from lolvision.dataset_io import MINI_STORE, STORE_W
from lolstream.video_stream import probe_url, frames as frame_stream
from lolstream.channels import check_fps, youtube_id
from lolvision.coord_model import expected_point, peak_point, spread
from lolvision.geometry import MAP_UNITS, Projection
from lolvision.unit_tracker import UnitTracker
from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.minimap_camera import (MM, BOX_W, BOX_H, white_mask, find_box,
                                      find_box_clipped, estimate_box)
from lolvision.patch_infer import PatchInfer
from lolvision.scene_kind import CLASSES as KIND_CLASSES, DEFAULT_HEAD as KIND_HEAD, SceneKind
from lolvision.scene_net import DEFAULT_NET as KIND_NET, SceneNetRunner
from lolvision.unit_net import DEFAULT_NET, REF_H, REF_W, UnitNetRunner
from lolvision.unit_targets import ignore_map, unit_keep
from lolvision.minimap_icons import DEFAULT_RUN as MINI_RUN
from lolvision.minimap_locate import Locator, box_px
from lolstream.live_champions import LiveChampions
from lolvision.match_state import MatchState

WARMUP = 12              # кадров с видимой рамкой на оценку её размера
WARMUP_MAX = 120         # дольше не ждём: оцениваем по тому, что есть
WARM_Q = 0.4             # «рамка видна»: качество при размере по умолчанию
# Поиск миникарты (minimap_locate.py, #785): ~40 мс на кадр «игра», только
# пока не найдена. Не нашлась за столько попыток — остаётся раскладка.
LOCATE_MAX = 300
# Типы сцены «игра под оверлеем»: миникарта видна, сцена закрыта окном.
OVERLAY_KINDS = ("shop", "tab")
BOX0 = (int(BOX_W * MM), int(BOX_H * MM))
MAX_FRAMES = 5400        # потолок памяти: полтора часа при 1 кадр/с
DEFAULT_RUN = "patches/cnn-split16-v2-notgame"
STAGES = ("decode", "prep", "net", "agg", "kind", "units", "mini", "who", "out")
# Кадр целиком для показа юнитов: ширина хранимой копии (16:9).
FULL_STORE_W = 640
# Тип сцены (игра / Tab / магазин / клиент / загрузка / не League). Своя сеть,
# дистиллированная из DINO (scene_net.py), стоит ~1.6 мс с уменьшением кадра —
# считаем на каждом кадре. Если её нет, работает сам DINO (scene_kind.py,
# ~14 мс) — тогда примерно KIND_HZ раз в секунду потока, между этим берём
# прошлый ответ.
KIND_HZ = 2.0
# Детектор юнитов без опоры на полоски ХП (#773, docs/unit-detection.md):
# с полосками миньоны не хуже прежнего, без них — 0.79 против 0.015. Его
# рабочий порог 0.4 (у прежней сети 0.3). Нет файла — прежняя сеть. Выбор —
# при каждом запуске сеанса (unit_net_choice), а не при загрузке модуля:
# сервер живёт дольше, чем файлы моделей на месте.
UNIT_NET_NOBARS = RUNS / "units" / "net-nobars2" / "model.pt"
UNIT_NETS = ((UNIT_NET_NOBARS, 0.4), (DEFAULT_NET, 0.3))
TIMING_WINDOW = 200      # по скольким последним кадрам медиана времени
STOP_WAIT = 20.0         # с: сколько ждать остановки прежнего сеанса перед новым


# ---------- модели ----------

class Runner(Protocol):
    """Что сеанс ждёт от модели: PatchInfer (patch_infer.py) или CoordInfer.

    `prepare` — хранимая сцена → вход `predict`. `frame_prep` — подготовка
    полного кадра на видеокарте: вызываемое «кадр → (вход predict, хранимая
    сцена)» или None, и тогда сцена готовится на процессоре.
    """

    kind: str
    label: str
    trained_on: list[str]

    def describe(self, scene_size) -> str: ...

    def prepare(self, scene: Image.Image): ...

    def frame_prep(self, frame_size, crop, boxes, store_size) -> Callable | None: ...

    def predict(self, prep) -> tuple[dict, np.ndarray, dict]: ...


class CoordInfer:
    """CoordNet: хранимая сцена → вход сети → ожидаемая точка тепловой карты."""

    kind = "coordnet"

    def __init__(self, net, ck: dict, device: str):
        self.net = net
        self.size = tuple(ck["input_size"])
        self.device = device
        self.grid = net.grid
        self.trained_on = ck.get("trained_on", [])
        self.label = "CoordNet"

    def describe(self, scene_size) -> str:
        return f"{self.label}, вход {self.size[0]}x{self.size[1]}"

    def prepare(self, scene: Image.Image) -> torch.Tensor:
        a = np.asarray(scene.resize(self.size, Image.BILINEAR),
                       dtype=np.float32) / 255.0
        return torch.from_numpy(a.transpose(2, 0, 1) * 2 - 1)[None]

    def frame_prep(self, frame_size, crop, boxes, store_size) -> None:
        """Подготовки кадра на видеокарте у CoordNet нет."""
        return None

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
    return run_dir(run) / "model.pt"


def load_runner(path: Path, device: str):
    """Модель для потока по типу чекпоинта: патчевая или CoordNet."""
    net, ck = load_checkpoint(path, device)
    runner = PatchInfer if checkpoint_kind(ck) == "patch" else CoordInfer
    return runner(net, ck, device)


_kinds: dict[str, tuple[float, dict]] = {}


def models(default: str = DEFAULT_RUN) -> list[dict]:
    """Все прогоны с model.pt, годные для потока, с типом модели.

    Тип читается из самого чекпоинта, а не из каталога (CoordNet для
    сравнения лежит и в runs/patches), и запоминается до смены файла.
    `default` — какой прогон пометить моделью по умолчанию.
    """
    out = []
    for p in sorted(RUNS.rglob("model.pt")):
        rid = p.parent.relative_to(RUNS).as_posix()
        mt = p.stat().st_mtime
        hit = _kinds.get(rid)
        if hit is None or hit[0] != mt:
            try:
                ck = read_checkpoint(p, mmap=True)
            except Exception:                                 # noqa: BLE001
                continue
            kind = checkpoint_kind(ck)
            if kind is None:
                continue
            info = {"kind": kind,
                    "label": (f"патч-{ck.get('encoder', '?')}" if kind == "patch"
                              else "CoordNet"),
                    "trained_on": list(ck.get("trained_on", []))}
            hit = _kinds[rid] = (mt, info)
        out.append({"id": rid, **hit[1], "default": rid == default})
    return out


# ---------- сеанс ----------

class State(enum.Enum):
    """Состояние сеанса. Логика сравнивает члены, а наружу (в JSON поля
    state) уходит подпись `label`: её показывает и с ней сравнивает
    webapp/app.js."""

    PREPARING = "preparing"
    PROBING = "probing"
    RUNNING = "running"
    STOPPED = "stopped"
    ENDED = "ended"
    FAILED = "failed"

    @property
    def label(self) -> str:
        return STATE_LABELS[self]

    @property
    def active(self) -> bool:
        """Сеанс ещё работает (или вот-вот начнёт)."""
        return self in (State.PREPARING, State.PROBING, State.RUNNING)


STATE_LABELS = {
    State.PREPARING: "готовлюсь",
    State.PROBING: "спрашиваю YouTube",
    State.RUNNING: "идёт",
    State.STOPPED: "остановлено",
    State.ENDED: "ролик кончился",
    State.FAILED: "сбой",
}


class Session:
    """Один ролик, читаемый потоком. Живёт в памяти, пока его не закрыли."""

    def __init__(self, vid: str, run: str, fps: float = 1.0,
                 start: float = 0.0, device: str = "cuda"):
        self.vid, self.run, self.fps, self.start = vid, run, fps, start
        self.device = device if torch.cuda.is_available() else "cpu"
        self.rows: list[dict] = []
        self.scene: dict[int, bytes] = {}
        self.mini: dict[int, bytes] = {}
        self.full: dict[int, bytes] = {}       # кадр целиком 640×360 — под юнитов
        self.heat: dict[int, np.ndarray] = {}
        self.roster: list[str] | None = None   # состав матча по HUD (L1…L5, R1…R5)
        self.minimap: dict | None = None       # прямоугольник миникарты ролика (#785)
        self.view = "spectator"                # раскладка: наблюдатель или игрок (#786)
        self.kind = ""
        self.model_label = ""
        self.trained_on: list[str] = []
        self.state = State.PREPARING
        self.note = ""
        self.duration = 0.0
        self.title = ""
        self.started = time.time()
        self._stop = threading.Event()
        self._champs: LiveChampions | None = None
        self._lock = threading.Lock()
        # Сжатие картинок и запись кадра в сеанс — в своём потоке: путь кадра
        # его не ждёт. Один исполнитель — кадры ложатся строго по порядку.
        self._n_emit = 0
        self.match: MatchState | None = None   # контекст матча (#780)
        self.tracker: UnitTracker | None = None  # id юнитов между кадрами (#782)
        self.proj: Projection | None = None    # кадр → карта (есть, когда есть юниты)
        self._out = ThreadPoolExecutor(1, thread_name_prefix="live-out")
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
        # Ошибка потока состава (#788): поток жив, но состав мог не читаться.
        champs_error = self._champs.error if self._champs is not None else None
        note = self.note
        # Раскладка может смениться по ходу ролика: в note — текущая, а не
        # дописанная однажды (прежде «вид игрока» оставался навсегда).
        if self.view == "player":
            note += " · вид игрока"
        if champs_error:
            ago = round(time.time() - champs_error["time"])
            note += f" · состав: ошибка {ago} с назад — {champs_error['text']}"
        return {"id": self.vid, "run": self.run, "state": self.state.label,
                "kind": self.kind, "model": self.model_label,
                "trained_on": self.trained_on, "timing": timing(tail),
                "note": note, "champs_error": champs_error,
                "fps": self.fps, "start": self.start,
                "title": self.title, "duration": round(self.duration),
                "count": n, "since": since, "frames": rows, "roster": self.roster,
                "minimap": self.minimap, "view": self.view, "elapsed": round(el, 1),
                "speed": round(seen / el, 1) if el > 1 else None,
                "map_units": MAP_UNITS}

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> None:
        """Дождаться конца работы сеанса (после stop)."""
        self._th.join(timeout)

    # ---------- работа ----------

    def _fail(self, why: str) -> None:
        self.state, self.note = State.FAILED, why

    def _run(self) -> None:
        try:
            self._work()
        except Exception as exc:                              # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            self._out.shutdown(wait=True)
            if self._champs is not None:
                self._champs.close()             # поток чтения состава

    def _work(self) -> None:
        layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
        url = f"https://www.youtube.com/watch?v={self.vid}"
        self.state = State.PROBING
        w, h, dur, _, info = probe_url(url)
        self.duration, self.title = dur, info.get("title") or self.vid

        cx0, cy0, cx1, cy1 = crop_box((w, h), layout, "dense")
        store_h = round(STORE_W * (cy1 - cy0) / (cx1 - cx0))
        boxes = mask_boxes((w, h), layout)
        # Миникарта: пока не найдена в кадре, режется по раскладке. Масштаб
        # миникарты у стримеров свой (от 1.0 до 1.5 от стандартной), и по
        # раскладке истина у них съезжает к левому верхнему углу (#785).
        mmbox = box_px(layout["labelling"]["minimap"], (w, h))
        self.minimap = {"box": list(layout["labelling"]["minimap"]), "scale": None,
                        "found": False}
        locator = Locator((w, h))
        rx0, ry0 = locator.region[:2]

        ck = run_checkpoint(self.run)
        if not ck.is_file():
            return self._fail(f"нет модели runs/{self.run}/model.pt")
        model: Runner = load_runner(ck, self.device)
        self.kind, self.model_label = model.kind, model.label
        self.trained_on = list(model.trained_on)
        self.note = (f"{w}x{h}, {dur / 60:.0f} мин · "
                     f"{model.describe((STORE_W, store_h))} · {self.device}")
        # Подготовка кадра: у патчевой модели на видеокарте целиком (~1 мс),
        # иначе на процессоре — обрезка видом, маска на малом, cv2 INTER_AREA
        # вместо PIL LANCZOS (~3 мс против ~17; ответ патчевой модели тот же
        # в пределах 0.3 ед.).
        gpu_prep = model.frame_prep((w, h), (cx0, cy0, cx1, cy1), boxes,
                                    (STORE_W, store_h))
        cboxes = [(max(x0, cx0) - cx0, max(y0, cy0) - cy0,
                   min(x1, cx1) - cx0, min(y1, cy1) - cy0)
                  for x0, y0, x1, y1 in boxes
                  if x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1]

        def cpu_scene(fr: np.ndarray) -> np.ndarray:
            a = fr[cy0:cy1, cx0:cx1].copy()
            for x0, y0, x1, y1 in cboxes:
                a[y0:y1, x0:x1] = 0
            return cv2.resize(a, (STORE_W, store_h), interpolation=cv2.INTER_AREA)

        # Юниты на основном виде (unit_net.py): чемпионы, миньоны, варды —
        # где, чьи, доля ХП. Маска HUD — та же, что при обучении; у краёв HUD
        # ответы отбрасываются (там сеть не училась).
        unit_model = None
        unit_net, unit_thr = unit_net_choice()
        if not self.device.startswith("cuda"):
            self.note += " · без юнитов (нет видеокарты)"
        elif unit_net is None:
            self.note += f" · без юнитов (нет {_rel(UNIT_NETS[0][0])})"
        else:
            keep, valid, valid_src = _unit_zone(w, h)
            unit_model = UnitNetRunner(unit_net, self.device, keep=keep, thr=unit_thr,
                                       valid=valid, size=(h, w))
            # Трекер юнитов (#782): сопоставление на карте через проекцию
            # экрана и центр камеры; «должен быть виден» — та же зона ответов.
            self.proj = Projection.load((cx0, cy0, cx1, cy1), (w, h))
            self.tracker = UnitTracker(self.proj, valid=valid_src)
            self.note += (" · юниты" if unit_net == UNIT_NETS[0][0] else
                          f" · юниты (прежняя сеть: нет {_rel(UNIT_NETS[0][0])})")
        # Зона детектора юнитов — под раскладку ролика (#786): пока раскладка
        # не решена — наблюдатель, у вида игрока своя маска от миникарты.
        zone = ("spectator", None)

        # Чемпионы (live_champions.py): состав по HUD, миникарта, жив ли,
        # имена над чемпионами детектора. Только на кадрах «игра».
        champs = None
        if (MINI_RUN / "model.pt").is_file():
            champs = self._champs = LiveChampions(self.device)
            self.note += " · чемпионы" + (f" ({'; '.join(champs.notes)})" if champs.notes else "")
        else:
            self.note += f" · без чемпионов (нет {_rel(MINI_RUN / 'model.pt')})"

        kind_model, kind_every = None, 1
        if KIND_NET.is_file():
            kind_model = SceneNetRunner(KIND_NET, self.device)
            self.note += " · тип сцены"
        elif KIND_HEAD.is_file():
            kind_model = SceneKind(KIND_HEAD, self.device)
            kind_every = max(1, round(self.fps / KIND_HZ))
            self.note += " · тип сцены (DINO)"
        else:
            self.note += f" · без типа сцены (нет {_rel(KIND_NET)})"
        kind_p = None
        n_frame = 0

        # Кадр грузится в видеопамять один раз, через закреплённый буфер, и
        # дальше его режут на месте подготовка камеры, детектор юнитов и
        # уменьшенная копия для показа. Раньше каждая модель грузила свои
        # 6 МБ сама.
        fr_pin = fr_dev = None
        if self.device.startswith("cuda"):
            fr_pin = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()
            fr_dev = torch.empty((h, w, 3), dtype=torch.uint8, device=self.device)

        self.state = State.RUNNING

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
            t2 = time.perf_counter()
            if fr_dev is not None:
                fr_pin.numpy()[:] = fr
                fr_dev.copy_(fr_pin, non_blocking=True)
            if gpu_prep is not None:
                prep, scene = gpu_prep(fr_dev)
            else:
                scene = cpu_scene(fr)
                prep = model.prepare(Image.fromarray(scene))
            ms = {"decode": (t1 - t0) * 1000,
                  "prep": (time.perf_counter() - t2) * 1000}
            if kind_model is not None and n_frame % kind_every == 0:
                t3 = time.perf_counter()
                kind_p = kind_model.probs(fr)
                ms["kind"] = (time.perf_counter() - t3) * 1000
            ms["kind_p"] = kind_p
            # Без модели типа сцены считаем, что всё — игра.
            kind = "game" if kind_p is None else KIND_CLASSES[int(np.argmax(kind_p))]
            game = kind == "game"
            # Магазин и Tab — игра под оверлеем (#786): сцену закрывает окно,
            # но миникарта видна, и чемпионы на ней читаются.
            overlay = kind in OVERLAY_KINDS
            region = fr[ry0:, rx0:]
            if locator.box is None and game and locator.tried < LOCATE_MAX:
                if locator.offer_region(region):
                    mmbox = box_px(locator.box, (w, h))
                    self.minimap = {"box": list(locator.box), "scale": locator.fit.scale,
                                    "found": True}
                    self.note += f" · миникарта ×{locator.fit.scale:g}"
                    if bw is not None:
                        # Найдена после прогрева: размер рамки оценён в чужой
                        # сетке. Вьюпорт постоянен в единицах карты, поэтому
                        # в сетке своей миникарты он меньше в масштаб раз.
                        bw = round(bw / locator.fit.scale)
                        bh = round(bh / locator.fit.scale)
            mini = _mini(region, mmbox, rx0, ry0)
            mask = white_mask(np.asarray(mini, dtype=np.float32))
            units = None
            if unit_model is not None:
                t4 = time.perf_counter()
                units = unit_model.units(fr_dev)
                ms["units_list"] = [
                    [u.kind, u.team, round(u.x / w, 4), round(u.y / h, 4), round(u.hp, 2),
                     round(u.score, 2)] for u in units]
                ms["units"] = (time.perf_counter() - t4) * 1000
            if champs is not None and (game or overlay):
                t5 = time.perf_counter()
                if game:
                    champs.offer(fr, n_frame / self.fps)
                ms["champs"] = champs.minimap(fr, np.asarray(mini))
                self.roster = champs.roster
                t6 = time.perf_counter()
                ms["mini"] = (t6 - t5) * 1000
                if units and game:
                    # Имя — седьмым полем юнита: старые поля на месте.
                    for row_u, name in zip(ms["units_list"], champs.who(fr, units)):
                        row_u.append(name)
                    ms["who"] = (time.perf_counter() - t6) * 1000
            if unit_model is not None and champs is not None:
                view = champs.view or "spectator"
                want = (view, locator.box if view == "player" else None)
                if want != zone:
                    zone = want
                    keep, valid, valid_src = _unit_zone(w, h, view, locator.box)
                    unit_model.set_zone(keep, valid)
                    self.tracker.valid = valid_src
                    self.view = view
            ms["full_img"] = _full_small(fr, fr_dev)
            n_frame += 1

            if bw is None:
                # Угол кадра — чтобы перерезать миникарту, если её найдут
                # позже в прогреве (буфер кадра переиспользуется).
                warm.append((mask, scene, prep, mini, ms, region.copy()))
                # Считаются только кадры, где рамка уже видна: по заставке
                # размер выходил меньше настоящего (58w57eJ5Qks: 56x32 вместо
                # 68x38), и все метки ролика смещались на ~380 ед.
                if find_box(mask, *BOX0)[2] >= WARM_Q:
                    n_good += 1
                # Ждём и миникарту: без неё размер рамки оценится в чужой сетке.
                if (n_good < WARMUP or locator.box is None) and len(warm) < WARMUP_MAX:
                    continue
                warm = _recut(warm, mmbox, rx0, ry0)
                good = [m[0] for m in warm if find_box(m[0], *BOX0)[2] >= WARM_Q]
                bw, bh = estimate_box(good or [m[0] for m in warm])
                self.note += f" · рамка {bw}x{bh}"
                for m, s, p, mi, t in warm:
                    self._emit(m, s, p, mi, bw, bh, model, t)
                warm.clear()
                continue
            self._emit(mask, scene, prep, mini, bw, bh, model, ms)
        stream.close()                    # ffmpeg и yt-dlp не должны висеть
        if bw is None and warm and not self._stop.is_set():
            # Ролик кончился раньше, чем набралась оценка рамки.
            warm = _recut(warm, mmbox, rx0, ry0)
            bw, bh = estimate_box([m[0] for m in warm])
            for m, s, p, mi, t in warm:
                self._emit(m, s, p, mi, bw, bh, model, t)

        self._out.shutdown(wait=True)    # все кадры записаны — тогда конец
        if self.state is State.RUNNING:
            self.state = State.STOPPED if self._stop.is_set() else State.ENDED

    def _emit(self, mask, scene: np.ndarray, prep, mini: Image.Image,
              bw: int, bh: int, model: Runner, ms: dict) -> None:
        """Один кадр: истина с миникарты, ответ модели, картинки в память.

        `ms` — уже набежавшее время кадра (распаковка; подготовка входа
        модели — маска, обрезка, масштаб, вид сверху); сюда добавляются сеть,
        сведение голосов и «ответ» — сжатие картинок и запись в сеанс.
        Метка с миникарты в путь модели не входит и не считается.

        Истины нет, если миникарта в кадре не найдена, а вид не наблюдателя:
        тогда она вырезана по раскладке наблюдателя мимо настоящей, и рамка
        по ней — мусор. В строке cx, cy, q, err = None и truth = "no_minimap";
        ответ модели остаётся.
        """
        truth = self.minimap is None or self.minimap["found"] or self.view == "spectator"
        if truth:
            x, y, q, _ = find_box_clipped(mask, bw, bh)
            cx, cy = (x + bw / 2) / MM, (y + bh / 2) / MM

        out, hm, mt = model.predict(prep)
        ms.update(mt)

        t0 = time.perf_counter()
        # Номер кадра — свой счётчик: строки ложатся в self.rows позже, из
        # потока сжатия.
        i = self._n_emit
        self._n_emit += 1
        row = {"i": i, "t": round(self.start + i / self.fps, 2),
               "cx": None, "cy": None,
               **{k: _r(out[k], 4) for k in ("px", "py", "qx", "qy")},
               "err": None, "spread": _r(out["spread"], 3), "q": None}
        if truth:
            row.update(cx=round(cx, 4), cy=round(cy, 4), q=round(q, 3),
                       err=_r(np.hypot(cx - out["px"], cy - out["py"]), 4))
        else:
            row["truth"] = "no_minimap"
        for k in ("agree", "scene", "game"):
            if k in out:
                row[k] = _r(out[k], 3)
        for k in ("votes", "is_game"):
            if k in out:
                row[k] = out[k]
        units = ms.pop("units_list", None)
        full_img = ms.pop("full_img", None)
        if units is not None:
            row["units"] = units
        mini_champs = ms.pop("champs", None)
        if mini_champs is not None:
            row["champs"] = mini_champs
        kp = ms.pop("kind_p", None)
        if kp is not None:
            # DINO узнаёт интерфейс, патчевая модель — места Ущелья: «игра»
            # без её подтверждения — не наша карта или чужая игра.
            c = int(np.argmax(kp))
            if KIND_CLASSES[c] == "game" and row.get("is_game") is False:
                c = KIND_CLASSES.index("other")
            row["kind"] = KIND_CLASSES[c]
            row["kind_p"] = round(float(kp[int(np.argmax(kp))]), 3)
        # Контекст матча: память о чемпионах между кадрами, смерти и
        # возрождения. В строку — снимок на этот кадр и новые события.
        if self.match is None:
            self.match = MatchState(view=(bw / MM, bh / MM))
        ev = self.match.update(row["t"], row)
        if self.match.order:
            row["ctx"] = self.match.frame_view()
        if ev:
            row["ev"] = ev
        # Треки юнитов: id, состояние (seen / unseen / lost), место на карте.
        # События юнитов — отдельно от событий чемпионов: смертей крипов
        # десятки в минуту.
        if self.tracker is not None and row.get("kind", "game") == "game" and "units" in row:
            cam = (row.get("px"), row.get("py"))
            uev = self.tracker.update(row["t"], None if cam[0] is None else cam, row["units"])
            row["tracks"] = self.tracker.frame_view(row["t"])
            if uev:
                row["uev"] = uev
        ms["out"] = (time.perf_counter() - t0) * 1000
        row["ms"] = {k: round(ms.get(k, 0.0), 2) for k in STAGES}
        self._out.submit(self._store, row, scene, full_img, mini, hm)

    def _store(self, row: dict, scene: np.ndarray, full_img: np.ndarray | None,
               mini: Image.Image, hm: np.ndarray) -> None:
        """Сжать картинки кадра и положить кадр в сеанс (поток self._out).
        Строка появляется вместе с картинками: смотрелка не попросит кадр,
        которого ещё нет. Время сжатия — row["ms"]["enc"], вне пути кадра."""
        t0 = time.perf_counter()
        sj = _jpeg(Image.fromarray(scene), 85)
        fj = _jpeg(Image.fromarray(full_img), 80) if full_img is not None else None
        mj = _jpeg(mini.resize((MINI_STORE, MINI_STORE)), 82)
        row["ms"]["enc"] = round((time.perf_counter() - t0) * 1000, 2)
        i = row["i"]
        with self._lock:
            self.scene[i] = sj
            self.mini[i] = mj
            if fj is not None:
                self.full[i] = fj
            # float16: у патчевой модели сетка 64×64, и полтора часа потока
            # в float32 заняли бы почти 90 МБ одних тепловых карт.
            self.heat[i] = hm.astype(np.float16)
            self.rows.append(row)


def unit_net_choice() -> tuple[Path | None, float | None]:
    """Сеть детектора юнитов и её порог — первая из UNIT_NETS, что есть на диске."""
    for path, thr in UNIT_NETS:
        if path.is_file():
            return path, thr
    return None, None


def _rel(path: Path) -> str:
    try:
        return path.relative_to(RUNS.parent).as_posix()
    except ValueError:
        return str(path)


def _unit_zone(w: int, h: int, view: str = "spectator",
               box=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Маски детектора юнитов: keep и valid в сетке 1080p (в ней сеть училась
    и в ней заданы полосы ignore_map) и valid в размере кадра — для трекера."""
    keep = unit_keep(REF_W, REF_H, view, box)
    valid = 1 - ignore_map(keep)
    if (w, h) == (REF_W, REF_H):
        return keep, valid, valid
    return keep, valid, cv2.resize(valid, (w, h), interpolation=cv2.INTER_NEAREST)


def _mini(region: np.ndarray, mmbox, rx0: int, ry0: int) -> Image.Image:
    """Миникарта 256×256 из угла кадра — те же пиксели, что из кадра целиком.

    LANCZOS оставлен как в build_coords: с INTER_AREA рамка в 15% кадров
    уезжала на пиксель (~58 ед.) от меток датасета. Обрезается до PIL, а не
    после: перевод в PIL всего кадра 1080p стоил сам по себе."""
    x0, y0, x1, y1 = mmbox
    return Image.fromarray(np.ascontiguousarray(
        region[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0])).resize((MM, MM), Image.LANCZOS)


def _recut(warm: list[tuple], mmbox, rx0: int, ry0: int) -> list[tuple]:
    """Кадры прогрева с миникартой по окончательному прямоугольнику.

    Ответы иконок чемпионов на кадрах до находки остаются по раскладке:
    это первые секунды игры, и перерезать их — значит держать кадры целиком."""
    out = []
    for _, s, p, _, t, region in warm:
        mini = _mini(region, mmbox, rx0, ry0)
        out.append((white_mask(np.asarray(mini, dtype=np.float32)), s, p, mini, t))
    return out


def _full_small(fr: np.ndarray, fr_dev: torch.Tensor | None) -> np.ndarray:
    """Кадр целиком шириной FULL_STORE_W для показа юнитов: на видеокарте из
    уже загруженного кадра (среднее по площади, как cv2.INTER_AREA), без неё —
    на процессоре."""
    h, w = fr.shape[:2]
    size = (FULL_STORE_W * h // w, FULL_STORE_W)
    if fr_dev is None:
        return cv2.resize(fr, size[::-1], interpolation=cv2.INTER_AREA)
    x = F.interpolate(fr_dev.permute(2, 0, 1)[None].float(), size=size, mode="area")
    return x[0].permute(1, 2, 0).round_().to(torch.uint8).cpu().numpy()


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


# ---------- реестр сеансов ----------

_sessions: dict[str, Session] = {}
_reg = threading.Lock()


def start(vid: str, run: str = DEFAULT_RUN, fps: float = 1.0,
          start_sec: float = 0.0) -> dict:
    """Открыть сеанс. Одновременно работает один: распаковка занимает карту."""
    vid, fps = youtube_id(vid), check_fps(fps)
    run = run or DEFAULT_RUN
    run_checkpoint(run)                  # путь вне runs/ отвергается сразу
    with _reg:
        stopped = []
        for other in list(_sessions.values()):
            if other.vid != vid and other.state.active:
                other.stop()
                stopped.append(other)
        # Кадры закрытого сеанса ещё можно листать, но держать их все вечно
        # нельзя: две тысячи кадров — это около 120 МБ. Оставляем последние
        # два ролика, остальные забываем.
        while len(_sessions) > 2:
            old_id = min(_sessions, key=lambda k: _sessions[k].started)
            if old_id == vid:
                break
            _sessions.pop(old_id).stop()
        old = _sessions.get(vid)
        if old and old.state.active:
            if old.run == run:
                return old.status()
            old.stop()                   # тот же ролик, но другой моделью
            stopped.append(old)
        # Новый сеанс захватывает CUDA graph; если старый в это время ещё
        # гоняет свои, захват падает (два сеанса подряд оба кончались
        # «сбоем»). Ждём, пока остановленные отпустят видеокарту.
        for other in stopped:
            other.join(STOP_WAIT)
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
    return [{"id": s.vid, "state": s.state.label, "count": len(s.rows),
             "run": s.run, "kind": s.kind, "title": s.title}
            for s in _sessions.values()]
