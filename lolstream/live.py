"""Смотрелка прямо из потока: ролик не скачивается и не собирается в датасет.

Раньше, чтобы посмотреть ответ модели на новом ролике, надо было собрать по
нему датасет (минуты) и дождаться прогона. Здесь кадры идут потоком: ролик
читается через yt-dlp, распаковывается видеокартой, с миникарты снимается
истинное положение камеры, модель тут же отвечает, и кадры по одному
появляются в смотрелке.

Откуда кадр, сеанс не знает (#794): он читает `FrameSource`
(frame_source.py) — ролик YouTube (`VideoSource`) или окно игры на своём
экране (`ScreenSource`). Кадра нет (пауза, окно свёрнуто, окна нет) — сеанс
не падает и строк не пишет, а показывает состояние источника и причину в
status() (`source_state`). Сборка моделей под размер кадра —
`build_pipeline`, путь кадра — `Session._step`; размер кадра сменился (окно
растянули) — путь собирается заново, и это видно в note. С сеансом экрана
работает Live Client своей игры (live_client.py, #795): его статус — в
status() (`api`), события копятся в `Session.api_events`.

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
from lolstream.channels import check_fps, youtube_id
from lolstream.frame_source import BACKENDS, FrameSource, ScreenSource, Status, VideoSource
from lolstream.live_client import EventRecord, LiveClient, LiveStatus, ddragon_champion_ids
from lolvision.coord_model import expected_point, peak_point, spread
from lolvision.geometry import MAP_UNITS, Projection
from lolvision.unit_tracker import UnitTracker
from lolvision.mask_frames import crop_box, mask_boxes
from lolvision.minimap_camera import (MM, BOX_W, BOX_H, white_mask, find_box,
                                      find_box_clipped, estimate_box)
from lolvision.patch_infer import PatchInfer
from lolvision.scene_kind import CLASSES as KIND_CLASSES, DEFAULT_HEAD as KIND_HEAD, SceneKind
from lolvision.scene_net import DEFAULT_NET as KIND_NET, SceneNetRunner
from lolvision.unit_net import LIVE_NET, REF_H, REF_W, UnitNetRunner, live_net
from lolvision.unit_targets import ignore_map, unit_keep
from lolvision.minimap_icons import DEFAULT_RUN as MINI_RUN
from lolvision.minimap_locate import Locator, box_px
from lolstream.live_champions import LiveChampions
from lolvision.match_state import MatchState
from lolvision.records import SCHEMA, UnitRec, dicts

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


# ---------- путь кадра ----------

class PipelineError(RuntimeError):
    """Путь кадра не собрать (нет модели): текст — причина для сеанса."""


class Pipeline:
    """Путь кадра одного размера: модели, геометрия раскладки под этот размер
    и состояние прогрева. Собирает `build_pipeline`, кадры через него гонит
    `Session._step`. Сменился размер кадра — собирается новый путь.

    `note` — что собрано и чего нет (без модели юнитов, без чемпионов…);
    `minimap` — прямоугольник миникарты, пока не найдена — по раскладке.
    """

    def __init__(self, size: tuple[int, int], device: str, fps: float, model: Runner):
        self.size, self.device, self.fps, self.model = size, device, fps, model
        self.note = ""
        self.crop: tuple[int, int, int, int] = (0, 0, *size)
        self.store: tuple[int, int] = (STORE_W, STORE_W)
        self.cboxes: list[tuple[int, int, int, int]] = []
        self.gpu_prep: Callable | None = None
        self.locator: Locator | None = None
        self.region0: tuple[int, int] = (0, 0)       # левый верх угла поиска миникарты
        self.mmbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.minimap: dict = {}
        self.unit_model: UnitNetRunner | None = None
        self.proj: Projection | None = None
        self.tracker: UnitTracker | None = None
        self.champs: LiveChampions | None = None
        self.kind_model = None
        self.kind_every = 1
        self.fr_pin: torch.Tensor | None = None
        self.fr_dev: torch.Tensor | None = None
        # Состояние пути кадра.
        self.kind_p = None
        self.n_frame = 0
        # Зона детектора юнитов — под раскладку (#786): пока раскладка не
        # решена — наблюдатель, у вида игрока своя маска от миникарты.
        self.zone: tuple = ("spectator", None)
        # Размер рамки вьюпорта постоянен, но по первому кадру его не
        # оценить (в начале заставка): копим кадры, оцениваем, потом отдаём.
        self.warm: list[tuple] = []
        self.bw: int | None = None
        self.bh: int | None = None
        self.n_good = 0

    def cpu_scene(self, fr: np.ndarray) -> np.ndarray:
        """Сцена на процессоре: обрезка видом, маска на малом, cv2 INTER_AREA
        вместо PIL LANCZOS (~3 мс против ~17; ответ патчевой модели тот же
        в пределах 0.3 ед.)."""
        cx0, cy0, cx1, cy1 = self.crop
        a = fr[cy0:cy1, cx0:cx1].copy()
        for x0, y0, x1, y1 in self.cboxes:
            a[y0:y1, x0:x1] = 0
        return cv2.resize(a, self.store, interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        """Погасить поток чтения состава."""
        if self.champs is not None:
            self.champs.close()


def build_pipeline(size: tuple[int, int], run: str, device: str, fps: float) -> Pipeline:
    """Собрать путь кадра под кадр `size` (W, H): модель камеры прогона `run`,
    детектор юнитов, чемпионы, тип сцены — что есть на диске и на `device`.
    Чего нет — записано в `note`, а не подменено. Нет модели камеры —
    PipelineError с причиной."""
    w, h = size
    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    ck = run_checkpoint(run)
    if not ck.is_file():
        raise PipelineError(f"нет модели runs/{run}/model.pt")
    model: Runner = load_runner(ck, device)
    p = Pipeline((w, h), device, fps, model)

    cx0, cy0, cx1, cy1 = p.crop = crop_box((w, h), layout, "dense")
    p.store = (STORE_W, round(STORE_W * (cy1 - cy0) / (cx1 - cx0)))
    boxes = mask_boxes((w, h), layout)
    # Миникарта: пока не найдена в кадре, режется по раскладке. Масштаб
    # миникарты у стримеров свой (от 1.0 до 1.5 от стандартной), и по
    # раскладке истина у них съезжает к левому верхнему углу (#785).
    p.mmbox = box_px(layout["labelling"]["minimap"], (w, h))
    p.minimap = {"box": list(layout["labelling"]["minimap"]), "scale": None, "found": False}
    p.locator = Locator((w, h))
    p.region0 = tuple(p.locator.region[:2])

    p.note = f"{model.describe(p.store)} · {device}"
    # Подготовка кадра: у патчевой модели на видеокарте целиком (~1 мс),
    # иначе на процессоре (Pipeline.cpu_scene).
    p.gpu_prep = model.frame_prep((w, h), p.crop, boxes, p.store)
    p.cboxes = [(max(x0, cx0) - cx0, max(y0, cy0) - cy0,
                 min(x1, cx1) - cx0, min(y1, cy1) - cy0)
                for x0, y0, x1, y1 in boxes
                if x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1]

    # Юниты на основном виде (unit_net.py): чемпионы, миньоны, варды —
    # где, чьи, доля ХП. Маска HUD — та же, что при обучении; у краёв HUD
    # ответы отбрасываются (там сеть не училась).
    # Боевой детектор и его порог выбираются в одном месте — unit_net.live_net.
    unit_net, unit_thr = live_net()
    if not device.startswith("cuda"):
        p.note += " · без юнитов (нет видеокарты)"
    elif unit_net is None:
        p.note += f" · без юнитов (нет {_rel(LIVE_NET)})"
    else:
        keep, valid, valid_src = _unit_zone(w, h)
        p.unit_model = UnitNetRunner(unit_net, device, keep=keep, thr=unit_thr,
                                     valid=valid, size=(h, w))
        # Трекер юнитов (#782): сопоставление на карте через проекцию
        # экрана и центр камеры; «должен быть виден» — та же зона ответов.
        p.proj = Projection.load(p.crop, (w, h))
        p.tracker = UnitTracker(p.proj, valid=valid_src)
        p.note += (" · юниты" if unit_net == LIVE_NET else
                   f" · юниты (прежняя сеть: нет {_rel(LIVE_NET)})")

    # Чемпионы (live_champions.py): состав по HUD, миникарта, жив ли,
    # имена над чемпионами детектора. Только на кадрах «игра».
    if (MINI_RUN / "model.pt").is_file():
        p.champs = LiveChampions(device)
        p.note += " · чемпионы" + (f" ({'; '.join(p.champs.notes)})" if p.champs.notes else "")
    else:
        p.note += f" · без чемпионов (нет {_rel(MINI_RUN / 'model.pt')})"

    if KIND_NET.is_file():
        p.kind_model = SceneNetRunner(KIND_NET, device)
        p.note += " · тип сцены"
    elif KIND_HEAD.is_file():
        p.kind_model = SceneKind(KIND_HEAD, device)
        p.kind_every = max(1, round(fps / KIND_HZ))
        p.note += " · тип сцены (DINO)"
    else:
        p.note += f" · без типа сцены (нет {_rel(KIND_NET)})"

    # Кадр грузится в видеопамять один раз, через закреплённый буфер, и
    # дальше его режут на месте подготовка камеры, детектор юнитов и
    # уменьшенная копия для показа.
    if device.startswith("cuda"):
        p.fr_pin = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()
        p.fr_dev = torch.empty((h, w, 3), dtype=torch.uint8, device=device)
    return p


# ---------- сеанс ----------

class State(enum.Enum):
    """Состояние сеанса. Логика сравнивает члены, а наружу (в JSON поля
    state) уходит подпись `label`, а признак «ещё работает» — поле active."""

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
    State.PROBING: "открываю источник",
    State.RUNNING: "идёт",
    State.STOPPED: "остановлено",
    State.ENDED: "кончился",
    State.FAILED: "сбой",
}

# Подписи источников для note и причины «API нет».
SOURCE_LABELS = {"youtube": "YouTube", "file": "файл", "screen": "свой экран"}


class Session:
    """Один поток кадров (ролик или свой экран). Живёт в памяти, пока его не
    закрыли. Откуда кадр, сеанс не знает: читает `FrameSource`.

    `api` — Live Client своей игры (live_client.py) или None: его заводит тот,
    кто открывает сеанс экрана; сеанс запускает его вместе с собой и
    останавливает вместе с собой. Его последний статус — `api_status()`,
    события — `api_events` (копятся за сеанс; логика матча из них — 2B).
    `build` — сборка пути кадра, как у `build_pipeline` (в тестах подставная).
    """

    def __init__(self, sid: str, run: str, source: FrameSource, device: str = "cuda",
                 api: LiveClient | None = None,
                 build: Callable[..., Pipeline] | None = None):
        self.sid, self.run, self.source = sid, run, source
        self.fps = float(source.fps)
        self.start = float(getattr(source, "start", 0.0))
        self.device = device if torch.cuda.is_available() else "cpu"
        self.api = api
        self.api_reason = (None if api is not None else
                           f"нет — источник {SOURCE_LABELS.get(source.kind, source.kind)}")
        self.api_events: list[EventRecord] = []
        self._build = build if build is not None else build_pipeline
        self.rows: list[dict] = []
        self.scene: dict[int, bytes] = {}
        self.mini: dict[int, bytes] = {}
        self.full: dict[int, bytes] = {}       # кадр целиком 640×360 — под юнитов
        self.heat: dict[int, np.ndarray] = {}
        self.roster: list[str] | None = None   # состав матча по HUD (L1…L5, R1…R5)
        self.minimap: dict | None = None       # прямоугольник миникарты (#785)
        self.view = "spectator"                # раскладка: наблюдатель или игрок (#786)
        self.kind = ""
        self.model_label = ""
        self.trained_on: list[str] = []
        self.state = State.PREPARING
        self.note = ""
        self.duration = 0.0
        self.title = ""
        self.started = time.time()
        # Что источник отдал последним: кадр или «кадра нет» с причиной, и
        # сколько раз кадра не было по каждой причине.
        self.source_state: dict = {"state": None, "reason": "источник ещё не открыт",
                                   "t": None, "since": None, "size": None}
        self.source_gaps: dict[str, int] = {}
        self._stop = threading.Event()
        self._pipe: Pipeline | None = None
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

    def api_status(self) -> LiveStatus | None:
        """Последний статус Live Client или None, если его в сеансе нет."""
        return None if self.api is None else self.api.status()

    def _drain_api(self) -> None:
        """Новые события Live Client — в `api_events` сеанса."""
        if self.api is None:
            return
        new = self.api.events()
        if new:
            with self._lock:
                self.api_events.extend(new)

    def _api_view(self) -> dict | None:
        """Статус API для status(): всё из LiveStatus.to_dict(), кроме полного
        снимка игры (десять игроков с предметами и статами — десятки КБ на
        каждый опрос смотрелки); вместо него — краткая сводка `brief`.
        Полностью — `api_full()`."""
        st = self.api_status()
        if st is None:
            return None
        d = st.to_dict()
        d.pop("game")
        g = st.game
        me = g.my_player if g is not None else None
        d["brief"] = None if g is None else {
            "game_time": g.game_time, "game_mode": g.game_mode, "players": len(g.players),
            "me": None if me is None else {"champion": me.champion, "team": me.team,
                                           "is_dead": me.is_dead},
            "me_reason": g.me_reason}
        d["events"] = len(self.api_events)
        return d

    def api_full(self) -> dict:
        """Статус API целиком (со снимком игры) и все события сеанса."""
        self._drain_api()
        st = self.api_status()
        with self._lock:
            events = [e.to_dict() for e in self.api_events]
        return {"id": self.sid, "api": None if st is None else st.to_dict(),
                "api_reason": self.api_reason, "events": events}

    def status(self, since: int = 0) -> dict:
        self._drain_api()
        with self._lock:
            rows = self.rows[since:]
            n = len(self.rows)
            tail = self.rows[-TIMING_WINDOW:]
            last_t = self.rows[-1]["t"] if self.rows else None
        el = time.time() - self.started
        # Скорость к реальному времени — у ролика (с промотанным началом: его
        # тоже распаковали); экран идёт в реальном времени — скорости нет.
        speed = None
        if self.source.kind != "screen" and el > 1:
            seen = last_t + 1 / self.fps if last_t is not None else self.start
            speed = round(seen / el, 1)
        # Ошибка потока состава (#788): поток жив, но состав мог не читаться.
        champs = self._pipe.champs if self._pipe is not None else None
        champs_error = champs.error if champs is not None else None
        note = self.note
        # Раскладка может смениться по ходу ролика: в note — текущая, а не
        # дописанная однажды (прежде «вид игрока» оставался навсегда).
        if self.view == "player":
            note += " · вид игрока"
        if champs_error:
            ago = round(time.time() - champs_error["time"])
            note += f" · состав: ошибка {ago} с назад — {champs_error['text']}"
        return {"id": self.sid, "run": self.run, "state": self.state.label,
                "active": self.state.active,
                "kind": self.kind, "model": self.model_label,
                "trained_on": self.trained_on, "timing": timing(tail),
                "note": note, "champs_error": champs_error,
                "fps": self.fps, "start": self.start,
                "title": self.title, "duration": round(self.duration),
                "source": self.source.describe(), "source_state": dict(self.source_state),
                "source_gaps": dict(self.source_gaps),
                "api": self._api_view(), "api_reason": self.api_reason,
                "count": n, "since": since, "frames": rows, "roster": self.roster,
                "minimap": self.minimap, "view": self.view, "elapsed": round(el, 1),
                "speed": speed, "map_units": MAP_UNITS}

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> None:
        """Дождаться конца работы сеанса (после stop)."""
        self._th.join(timeout)

    # ---------- работа ----------

    def _fail(self, why: str) -> None:
        self.state, self.note = State.FAILED, why

    def _run(self) -> None:
        if self.api is not None:
            self.api.start()                     # опрос API — свой поток
        try:
            self._work()
        except PipelineError as exc:
            self._fail(str(exc))
        except Exception as exc:                              # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            self._out.shutdown(wait=True)
            self.source.close()                  # ffmpeg, yt-dlp, захват экрана
            if self._pipe is not None:
                self._pipe.close()               # поток чтения состава
            if self.api is not None:
                self.api.stop()
                self._drain_api()

    def _seen(self, fr) -> None:
        """Запомнить ответ источника: состояние меняется — меняется since."""
        key = fr.status.key
        if key != self.source_state["state"]:
            self.source_state = {"state": key, "reason": fr.reason, "t": round(fr.t, 2),
                                 "since": time.time(), "size": fr.size}
        else:
            self.source_state.update(reason=fr.reason, t=round(fr.t, 2), size=fr.size)
        if not fr.ok and fr.status is not Status.ENDED:
            self.source_gaps[key] = self.source_gaps.get(key, 0) + 1

    def _new_pipe(self, size: tuple[int, int]) -> Pipeline:
        """Собрать путь кадра под размер и привязать к нему сеанс."""
        pl = self._build(size, self.run, self.device, self.fps)
        self._pipe = pl
        self.kind, self.model_label = pl.model.kind, pl.model.label
        self.trained_on = list(pl.model.trained_on)
        self.minimap = dict(pl.minimap)
        self.proj, self.tracker = pl.proj, pl.tracker
        if self.match is not None:
            # Память матча переживает смену размера, проекция — новая.
            self.match.proj = pl.proj
        return pl

    def _source_note(self, size: tuple[int, int] | None) -> str:
        src = self.source
        dims = f"{size[0]}x{size[1]}" if size else "размер кадра пока неизвестен"
        if src.kind == "screen":
            return f"свой экран {dims} · {src.backend_note or src.backend}"
        return f"{dims}, {self.duration / 60:.0f} мин"

    def _work(self) -> None:
        src = self.source
        self.state = State.PROBING
        src.open()                               # у экрана — в этом же потоке (mss)
        d = src.describe()
        self.title = d.get("title") or self.sid
        self.duration = float(d.get("duration") or 0.0)
        pl = None
        if src.size is not None:
            # Размер известен заранее (у ролика всегда): модели грузятся до
            # первого кадра, как и раньше.
            pl = self._new_pipe(src.size)
            self.note = f"{self._source_note(src.size)} · {pl.note}"
        else:
            self.note = self._source_note(None)
        self.state = State.RUNNING

        while not self._stop.is_set():
            if len(self.rows) >= MAX_FRAMES:
                self.note += f" · потолок памяти {MAX_FRAMES} кадров: дальше не читаю"
                break
            self._drain_api()
            # Время «decode» — ожидание кадра от источника: у ролика сеть до
            # YouTube, NVDEC и прореживание до fps; у экрана — такт и снимок.
            t0 = time.perf_counter()
            fr = src.read()
            t1 = time.perf_counter()
            if self._stop.is_set():
                break
            self._seen(fr)
            if fr.status is Status.ENDED:
                if fr.reason and fr.reason != "ролик кончился":
                    self.note += f" · {fr.reason}"
                break
            if not fr.ok:
                continue                         # пауза, окна нет — строк нет
            if pl is None or fr.size != pl.size:
                if pl is not None:
                    # Окно растянули: кадры прогрева — по старому размеру,
                    # дальше — новый путь под новый размер.
                    self._flush(pl)
                    old = pl.size
                    pl.close()
                    pl = self._new_pipe(fr.size)
                    self.note += (f" · размер кадра {old[0]}x{old[1]} → "
                                  f"{fr.size[0]}x{fr.size[1]}: путь кадра пересобран")
                else:
                    pl = self._new_pipe(fr.size)
                    self.note = f"{self._source_note(fr.size)} · {pl.note}"
            self._step(pl, fr.image, fr.t, (t1 - t0) * 1000)
        if pl is not None and not self._stop.is_set():
            self._flush(pl)                      # поток кончился раньше прогрева

        self._out.shutdown(wait=True)    # все кадры записаны — тогда конец
        if self.state is State.RUNNING:
            self.state = State.STOPPED if self._stop.is_set() else State.ENDED

    def _step(self, pl: Pipeline, fr: np.ndarray, t: float, decode_ms: float) -> None:
        """Путь одного кадра: подготовка, тип сцены, миникарта, юниты,
        чемпионы — и в _emit (после прогрева рамки)."""
        model = pl.model
        rx0, ry0 = pl.region0
        t2 = time.perf_counter()
        if pl.fr_dev is not None:
            pl.fr_pin.numpy()[:] = fr
            pl.fr_dev.copy_(pl.fr_pin, non_blocking=True)
        if pl.gpu_prep is not None:
            prep, scene = pl.gpu_prep(pl.fr_dev)
        else:
            scene = pl.cpu_scene(fr)
            prep = model.prepare(Image.fromarray(scene))
        ms = {"decode": decode_ms, "prep": (time.perf_counter() - t2) * 1000}
        if pl.kind_model is not None and pl.n_frame % pl.kind_every == 0:
            t3 = time.perf_counter()
            pl.kind_p = pl.kind_model.probs(fr)
            ms["kind"] = (time.perf_counter() - t3) * 1000
        ms["kind_p"] = pl.kind_p
        # Без модели типа сцены считаем, что всё — игра.
        kind = "game" if pl.kind_p is None else KIND_CLASSES[int(np.argmax(pl.kind_p))]
        game = kind == "game"
        # Магазин и Tab — игра под оверлеем (#786): сцену закрывает окно,
        # но миникарта видна, и чемпионы на ней читаются.
        overlay = kind in OVERLAY_KINDS
        region = fr[ry0:, rx0:]
        locator = pl.locator
        if locator.box is None and game and locator.tried < LOCATE_MAX:
            if locator.offer_region(region):
                pl.mmbox = box_px(locator.box, pl.size)
                pl.minimap = {"box": list(locator.box), "scale": locator.fit.scale,
                              "found": True}
                self.minimap = dict(pl.minimap)
                self.note += f" · миникарта ×{locator.fit.scale:g}"
                if pl.bw is not None:
                    # Найдена после прогрева: размер рамки оценён в чужой
                    # сетке. Вьюпорт постоянен в единицах карты, поэтому
                    # в сетке своей миникарты он меньше в масштаб раз.
                    pl.bw = round(pl.bw / locator.fit.scale)
                    pl.bh = round(pl.bh / locator.fit.scale)
        mini = _mini(region, pl.mmbox, rx0, ry0)
        mask = white_mask(np.asarray(mini, dtype=np.float32))
        units = None
        if pl.unit_model is not None:
            t4 = time.perf_counter()
            units = pl.unit_model.units(pl.fr_dev)
            ms["units"] = (time.perf_counter() - t4) * 1000
        names = None
        champs = pl.champs
        if champs is not None and (game or overlay):
            t5 = time.perf_counter()
            if game:
                champs.offer(fr, t)
            ms["champs"] = champs.minimap(fr, np.asarray(mini))
            self.roster = champs.roster
            t6 = time.perf_counter()
            ms["mini"] = (t6 - t5) * 1000
            if units and game:
                names = champs.who(fr, units)
                ms["who"] = (time.perf_counter() - t6) * 1000
        if units is not None:
            # Юниты детектора — записи records.UnitRec (доли кадра, имя).
            ms["unit_recs"] = [UnitRec.from_unit(u, pl.size, names[k] if names else None)
                               for k, u in enumerate(units)]
        if pl.unit_model is not None and champs is not None:
            view = champs.view or "spectator"
            want = (view, locator.box if view == "player" else None)
            if want != pl.zone:
                pl.zone = want
                keep, valid, valid_src = _unit_zone(*pl.size, view, locator.box)
                pl.unit_model.set_zone(keep, valid)
                pl.tracker.valid = valid_src
                self.view = view
        ms["full_img"] = _full_small(fr, pl.fr_dev)
        pl.n_frame += 1

        if pl.bw is None:
            # Угол кадра — чтобы перерезать миникарту, если её найдут
            # позже в прогреве (буфер кадра переиспользуется).
            pl.warm.append((mask, scene, prep, mini, ms, region.copy(), t))
            # Считаются только кадры, где рамка уже видна: по заставке
            # размер выходил меньше настоящего (58w57eJ5Qks: 56x32 вместо
            # 68x38), и все метки ролика смещались на ~380 ед.
            if find_box(mask, *BOX0)[2] >= WARM_Q:
                pl.n_good += 1
            # Ждём и миникарту: без неё размер рамки оценится в чужой сетке.
            if (pl.n_good < WARMUP or locator.box is None) and len(pl.warm) < WARMUP_MAX:
                return
            warm = _recut(pl.warm, pl.mmbox, rx0, ry0)
            good = [m[0] for m in warm if find_box(m[0], *BOX0)[2] >= WARM_Q]
            pl.bw, pl.bh = estimate_box(good or [m[0] for m in warm])
            self.note += f" · рамка {pl.bw}x{pl.bh}"
            for m, s, p, mi, msx, tt in warm:
                self._emit(m, s, p, mi, pl.bw, pl.bh, model, msx, tt)
            pl.warm.clear()
            return
        self._emit(mask, scene, prep, mini, pl.bw, pl.bh, model, ms, t)

    def _flush(self, pl: Pipeline) -> None:
        """Поток кончился (или сменился размер) раньше, чем набралась оценка
        рамки: оценить по тому, что есть, и отдать кадры прогрева."""
        if pl.bw is not None or not pl.warm:
            return
        warm = _recut(pl.warm, pl.mmbox, *pl.region0)
        pl.bw, pl.bh = estimate_box([m[0] for m in warm])
        for m, s, p, mi, msx, tt in warm:
            self._emit(m, s, p, mi, pl.bw, pl.bh, pl.model, msx, tt)
        pl.warm.clear()

    def _emit(self, mask, scene: np.ndarray, prep, mini: Image.Image,
              bw: int, bh: int, model: Runner, ms: dict, t: float) -> None:
        """Один кадр: истина с миникарты, ответ модели, картинки в память.

        `t` — время кадра по шкале источника (Frame.t): у ролика — позиция в
        ролике, у экрана — секунды от открытия источника.

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
        row = {"schema": SCHEMA, "i": i, "t": round(t, 2),
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
        # Записи кадра (records.py) идут в контекст и трекер как есть, в
        # строку — словарями: строка уходит в JSON.
        units = ms.pop("unit_recs", None)
        full_img = ms.pop("full_img", None)
        if units is not None:
            row["units"] = dicts(units)
        mini_champs = ms.pop("champs", None)
        if mini_champs is not None:
            row["champs"] = dicts(mini_champs)
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
            # Места чемпионов с экрана — той же проекцией, что у трекера.
            self.match = MatchState(proj=self.proj)
        ev = self.match.update(row["t"], row.get("kind", "game"), mini_champs, units,
                               (row["px"], row["py"]))
        if self.match.order:
            row["ctx"] = dicts(self.match.frame_view())
        if ev:
            row["ev"] = dicts(ev)
        # Треки юнитов: id, состояние (seen / unseen / lost), место на карте.
        # События юнитов — отдельно от событий чемпионов: смертей крипов
        # десятки в минуту.
        if self.tracker is not None and row.get("kind", "game") == "game" and units is not None:
            cam = (row["px"], row["py"])
            uev = self.tracker.update(row["t"], None if cam[0] is None else cam, units)
            row["tracks"] = dicts(self.tracker.frame_view(row["t"]))
            if uev:
                row["uev"] = dicts(uev)
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
    """Кадры прогрева с миникартой по окончательному прямоугольнику:
    (маска, сцена, вход, миникарта, время по стадиям, угол кадра, t) →
    (маска, сцена, вход, миникарта, время по стадиям, t).

    Ответы иконок чемпионов на кадрах до находки остаются по раскладке:
    это первые секунды игры, и перерезать их — значит держать кадры целиком."""
    out = []
    for _, s, p, _, ms, region, t in warm:
        mini = _mini(region, mmbox, rx0, ry0)
        out.append((white_mask(np.asarray(mini, dtype=np.float32)), s, p, mini, ms, t))
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
# Id сеанса своего экрана: экран один, и сеанс по нему один.
SCREEN_ID = "screen"


def _open(sid: str, run: str, make: Callable[[], Session]) -> dict:
    """Открыть сеанс `sid`. Одновременно работает один: распаковка и модели
    занимают видеокарту; прочие активные сеансы останавливаются."""
    with _reg:
        stopped = []
        for other in list(_sessions.values()):
            if other.sid != sid and other.state.active:
                other.stop()
                stopped.append(other)
        # Кадры закрытого сеанса ещё можно листать, но держать их все вечно
        # нельзя: две тысячи кадров — это около 120 МБ. Оставляем последние
        # два сеанса, остальные забываем.
        while len(_sessions) > 2:
            old_id = min(_sessions, key=lambda k: _sessions[k].started)
            if old_id == sid:
                break
            _sessions.pop(old_id).stop()
        old = _sessions.get(sid)
        if old and old.state.active:
            if old.run == run:
                return old.status()
            old.stop()                   # тот же источник, но другой моделью
            stopped.append(old)
        # Новый сеанс захватывает CUDA graph; если старый в это время ещё
        # гоняет свои, захват падает (два сеанса подряд оба кончались
        # «сбоем»). Ждём, пока остановленные отпустят видеокарту.
        for other in stopped:
            other.join(STOP_WAIT)
        s = make()
        _sessions[sid] = s
    return s.status()


def start(vid: str, run: str = DEFAULT_RUN, fps: float = 1.0,
          start_sec: float = 0.0) -> dict:
    """Сеанс по ролику YouTube (Live Client API у ролика нет)."""
    vid, fps = youtube_id(vid), check_fps(fps)
    run = run or DEFAULT_RUN
    run_checkpoint(run)                  # путь вне runs/ отвергается сразу
    url = f"https://www.youtube.com/watch?v={vid}"
    return _open(vid, run, lambda: Session(vid, run, VideoSource(url, fps, float(start_sec))))


def start_screen(run: str = DEFAULT_RUN, fps: float = 2.0, backend: str = "auto") -> dict:
    """Сеанс по своему экрану (окно игры) вместе с Live Client своей игры.

    Захват открывается в потоке сеанса (mss привязан к потоку), не здесь."""
    fps = check_fps(fps)
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"неизвестный бэкенд захвата {backend!r}: auto, {', '.join(BACKENDS)}")
    run = run or DEFAULT_RUN
    run_checkpoint(run)
    return _open(SCREEN_ID, run, lambda: Session(
        SCREEN_ID, run, ScreenSource(fps, backend=backend),
        api=LiveClient(champion_ids=ddragon_champion_ids())))


def get(sid: str) -> Session | None:
    return _sessions.get(sid)


def stop(sid: str) -> None:
    s = _sessions.get(sid)
    if s:
        s.stop()


def listing() -> list[dict]:
    return [{"id": s.sid, "state": s.state.label, "active": s.state.active,
             "count": len(s.rows), "run": s.run, "kind": s.kind, "title": s.title,
             "source": s.source.kind}
            for s in _sessions.values()]
