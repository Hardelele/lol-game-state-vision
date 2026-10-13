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

import copy
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
from lolstream.loadouts import LoadoutRecorder, live_recorder
from lolstream.live_client import (EventRecord, LiveClient, LiveStatus, ddragon_champion_ids,
                                   status_observation)
from lolstream.live_client import State as ApiState
from lolstream.state_service import PORT as STATE_PORT
from lolstream.state_service import (FrameStatus, StateInputs, StateServer, VisionFrame,
                                     match_view)
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
# Поиск миникарты (#785): ~40 мс на кадр «игра», только пока не найдена; не
# нашлась за LOCATE_MAX попыток — остаётся раскладка, а через LOCATE_PAUSE с
# поиск идёт новым кругом: сеанс «свой экран» переживает смену игры (ARAM →
# Ущелье), и сдавшийся навсегда поиск оставлял без миникарты всю следующую игру.
from lolvision.minimap_locate import (AGREE_PX, LOCATE_MAX, Locator, corner_px, box_px,
                                      SCALES, fit_file, load_fit, save_fit)
from lolstream.live_champions import LiveChampions
from lolvision.hud_spells import SpellHud
from lolvision.match_state import API_STALE, ApiObs, MatchState, rift
from lolvision.passport import UNKNOWN, Change, Passport
from lolvision.records import SCHEMA, RosterEvent, UnitRec, dicts
from lolvision.sides import SideContext, assign as assign_sides

LOCATE_PAUSE = 60.0      # с потока между кругами поиска миникарты после сдачи
WARMUP = 12              # кадров с видимой рамкой на оценку её размера
WARMUP_MAX = 120         # дольше не ждём: оцениваем по тому, что есть
WARM_Q = 0.4             # «рамка видна»: качество при размере по умолчанию
# Не Ущелье (#803): центр камеры — рамка на миникарте, если её качество не
# ниже этого. Кадры ARAM: рамка видна — 0.70–0.85, у края карты (обрезана) —
# 0.38, под окном выбора аугмента — 0.0.
MINIMAP_CAM_Q = 0.3
# Типы сцены «игра под оверлеем»: миникарта видна, сцена закрыта окном.
OVERLAY_KINDS = ("shop", "tab")
BOX0 = (int(BOX_W * MM), int(BOX_H * MM))
# Память кадров сеанса (#801). У ролика — потолок: дальше MAX_FRAMES кадров
# сеанс не читает (полтора часа при 1 кадр/с). Свой экран живёт через много
# игр подряд, и потолок обрывал его посреди игры (при 2 кадр/с — через
# 45 минут от запуска сервиса): у экрана — скользящее окно, последние
# SCREEN_WINDOW кадров, старые выбрасываются вместе с картинками. Кадр в окне
# ~120 КБ (сцена ~37 КБ, кадр целиком 640×360 ~43 КБ, миникарта ~18 КБ,
# тепловая карта 8 КБ, строка с юнитами и контекстом) — окно ~650 МБ, столько
# же, сколько занимал потолок.
MAX_FRAMES = 5400
SCREEN_WINDOW = 5400     # 45 минут при 2 кадр/с
# Событий Live Client в памяти сеанса — не больше (счёт всех — api_events_n).
API_EVENTS_KEEP = 2000
DEFAULT_RUN = "patches/cnn-split16-v2-notgame"
# Как найдена миникарта (minimap_locate.Fit.how) — для note и паспорта.
FIT_HOW = {"map": "эталон Ущелья", "frame": "рамка миникарты",
           "saved": "сохранённый прямоугольник, рамка на месте", "manual": "масштаб задан явно"}
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
# Паспорт трансляции (#799): сколько последних изменений показывать в note и
# сколько держать в status() (`passport_events`) и в памяти сеанса.
PASSPORT_NOTE = 3
# Готовность заклинаний D/F по HUD (#806): поиск панели умений — ~0.2 с на
# кадр 1080p, пока не найдена, не чаще раза в SPELL_LOCATE_EVERY с потока;
# после SPELL_LOCATE_SAY попыток без находки паспорт узнаёт причину.
SPELL_LOCATE_EVERY = 2.0
SPELL_LOCATE_SAY = 10
PASSPORT_STATUS = 50
PASSPORT_KEEP = 1000


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
        self.locate_done = False                     # поиск сдался — паспорту сказано
        self.locate_idle: float | None = None        # t конца круга поиска без находки
        self.region0: tuple[int, int] = (0, 0)       # левый верх угла поиска миникарты
        self.layout_mm: list[float] = []             # миникарта по раскладке, доли кадра
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
        # Заклинания D/F по панели умений (#806): своя на каждый размер кадра.
        self.spells = SpellHud()
        self.spell_try_t: float | None = None
        self.spell_said = False

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


def build_pipeline(size: tuple[int, int], run: str, device: str, fps: float,
                   champs: LiveChampions | None = None) -> Pipeline:
    """Собрать путь кадра под кадр `size` (W, H): модель камеры прогона `run`,
    детектор юнитов, чемпионы, тип сцены — что есть на диске и на `device`.
    Чего нет — записано в `note`, а не подменено. Нет модели камеры —
    PipelineError с причиной.

    `champs` — чемпионы прежнего пути при пересборке (окно растянули): они от
    размера кадра не зависят (всё читают в сетке 1080p), поэтому прочитанный
    состав и раскладка переживают пересборку, а не читаются заново."""
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
    p.layout_mm = list(layout["labelling"]["minimap"])
    p.mmbox = box_px(p.layout_mm, (w, h))
    p.minimap = {"box": list(p.layout_mm), "scale": None, "found": False}
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
    if champs is not None:
        p.champs = champs
        p.note += " · чемпионы (прежние: состав сохранён)"
    elif (MINI_RUN / "model.pt").is_file():
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
    события — `api_events`. Снимок API на каждом кадре идёт в контекст матча
    (`MatchState`, источник api) и в состав чемпионов (`LiveChampions.set_api`).
    `build` — сборка пути кадра, как у `build_pipeline` (в тестах подставная).

    Паспорт трансляции (`passport`, #799) строится после `source.open()` и
    меняется по сигналам пути кадра (размер кадра, миникарта, вид, API, тип
    сцены, состав, ресурсы); читатели, которым паспорт отказал, в строке
    кадра — null, причина — в `row["off"]`.

    `state_port` — поднять сервис состояния (state_service.StateServer) на
    этом порту loopback вместе с сеансом (0 — свободный порт); None — без него.

    Миникарта (#803): `minimap_store` — файл последнего прямоугольника,
    найденного эталоном Ущелья (minimap_locate.save_fit), по размеру кадра:
    на другой карте он берётся, если рамка миникарты на его месте. Только у
    своего экрана — масштаб миникарты это настройка своего клиента, у чужого
    ролика он свой. `minimap_scale` — масштаб задан явно: без поиска.
    """

    # Карта по Live Client API (mapNumber); None — API нет: ролик с Ущелья.
    map_number: int | None = None
    minimap_store: Path | None = None
    minimap_scale: float | None = None
    loadouts: LoadoutRecorder | None = None     # паспорт выбора (#807)
    window: int | None = None
    dropped = 0
    api_events_n = 0

    def __init__(self, sid: str, run: str, source: FrameSource, device: str = "cuda",
                 api: LiveClient | None = None,
                 build: Callable[..., Pipeline] | None = None,
                 state_port: int | None = None,
                 minimap_store: Path | None = None,
                 minimap_scale: float | None = None,
                 loadouts: LoadoutRecorder | None = None):
        self.sid, self.run, self.source = sid, run, source
        # Паспорт выбора (#807): наблюдатель API, пишет игру по её концу;
        # сеанс закрыт посреди игры — дописывает то, что есть (close).
        self.loadouts = loadouts
        self.minimap_store, self.minimap_scale = minimap_store, minimap_scale
        self.map_number = None
        self.fps = float(source.fps)
        self.start = float(getattr(source, "start", 0.0))
        self.device = device if torch.cuda.is_available() else "cpu"
        self.api = api
        self.api_reason = (None if api is not None else
                           f"нет — источник {SOURCE_LABELS.get(source.kind, source.kind)}")
        self.api_events: list[EventRecord] = []
        self.api_events_n = 0                  # всего событий за сеанс, с выброшенными
        self._build = build if build is not None else build_pipeline
        self.rows: list[dict] = []
        # Окно кадров (#801): у экрана — сколько последних держать, у ролика —
        # None (потолок MAX_FRAMES). Номер кадра row["i"] — сквозной за сеанс;
        # `dropped` — сколько первых кадров выброшено (номер self.rows[0]).
        self.window: int | None = SCREEN_WINDOW if source.kind == "screen" else None
        self.dropped = 0
        self.scene: dict[int, bytes] = {}
        self.mini: dict[int, bytes] = {}
        self.full: dict[int, bytes] = {}       # кадр целиком 640×360 — под юнитов
        self.heat: dict[int, np.ndarray] = {}
        self.roster: list[str] | None = None   # состав (L1…L5, R1…R5), принятый паспортом
        self.minimap: dict | None = None       # прямоугольник миникарты (#785)
        self.view = UNKNOWN                    # вид по паспорту: spectator / player / unknown
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
        self.last_ok: float | None = None      # unix-время последнего кадра
        self._stop = threading.Event()
        self._pipe: Pipeline | None = None
        self._lock = threading.Lock()
        # Сжатие картинок и запись кадра в сеанс — в своём потоке: путь кадра
        # его не ждёт. Один исполнитель — кадры ложатся строго по порядку.
        self._n_emit = 0
        self.match: MatchState | None = None   # контекст матча (#780)
        self._match_at: float | None = None    # unix-время момента match.t
        self._api_refused: str | None = None   # почему снимок API последнего кадра не принят
        self.tracker: UnitTracker | None = None  # id юнитов между кадрами (#782)
        self.proj: Projection | None = None    # кадр → карта (есть, когда есть юниты)
        # Паспорт трансляции и его изменения (Change.to_dict, по порядку).
        self.passport: Passport | None = None
        self.passport_events: list[dict] = []
        self._pnote: list[str] = []            # последние изменения паспорта для note
        # Live Client в пути кадра: последнее состояние, сказанное паспорту,
        # последнее принятое наблюдение и когда (время потока).
        self._api_state: ApiState | None = None
        self._api_obs: ApiObs | None = None
        self._api_obs_t: float | None = None
        # Последний кадр зрения для сервиса состояния (state_service.VisionFrame).
        self.vision: VisionFrame | None = None
        self.state_port = state_port
        self.state_server: StateServer | None = None
        self.state_url: str | None = None
        self.state_reason: str | None = ("сервис состояния у этого сеанса не поднимается"
                                         if state_port is None else STATE_PENDING)
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
                self.api_events_n += len(new)
                del self.api_events[:-API_EVENTS_KEEP]

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
        d["events"] = self.api_events_n
        return d

    def api_full(self) -> dict:
        """Статус API целиком (со снимком игры) и все события сеанса."""
        self._drain_api()
        st = self.api_status()
        with self._lock:
            events = [e.to_dict() for e in self.api_events]
        return {"id": self.sid, "api": None if st is None else st.to_dict(),
                "api_reason": self.api_reason, "events": events}

    def state_inputs(self) -> StateInputs:
        """Входы сервиса состояния (state_service.StateInputs) — провайдер
        StateServer, зовётся из его потока. Контекст матча, паспорт и кадр
        зрения снимаются под замком сеанса: их меняет поток кадра; паспорт —
        копией (сервис зовёт verdict/to_dict у себя)."""
        with self._lock:
            match = (None if self.match is None or self._match_at is None
                     else match_view(self.match, self._match_at, self._api_refused))
            passport = copy.deepcopy(self.passport)
            events = tuple(self.match.events) if self.match is not None else ()
            vision = self.vision
        ss, d = self.source_state, self.source.describe()
        size = ss.get("size") or d.get("size")
        frames = FrameStatus(kind=self.source.kind, state=ss["state"] or "no_frame",
                             reason=ss["reason"], backend=d.get("backend"),
                             backend_note=d.get("backend_note"), fps=self.fps,
                             size=None if size is None else (int(size[0]), int(size[1])),
                             last_ok=self.last_ok)
        return StateInputs(session=self.state.label, frames=frames, vision=vision, match=match,
                           api=self.api_status(), passport=passport, vision_events=events)

    @property
    def count(self) -> int:
        """Сколько кадров выдано за сеанс, включая выброшенные из окна."""
        return self.dropped + len(self.rows)

    def status(self, since: int = 0) -> dict:
        """Состояние сеанса и кадры с номером >= `since` (номера сквозные).

        `count` — номер следующего кадра (его и передавать в `since` дальше).
        У экрана кадры держатся окном (`window`): `first` — номер старейшего
        кадра в памяти; `since` старше него — отдаются все кадры окна, а
        `dropped_before` — сколько запрошенных кадров уже выброшено."""
        self._drain_api()
        since = max(0, since)
        with self._lock:
            first = self.dropped
            rows = self.rows[max(0, since - first):]
            n = first + len(self.rows)
            tail = self.rows[-TIMING_WINDOW:]
            last_t = self.rows[-1]["t"] if self.rows else None
            passport = None if self.passport is None else self.passport.to_dict()
            pev = self.passport_events[-PASSPORT_STATUS:]
            n_pev = len(self.passport_events)
            pnote = list(self._pnote[-PASSPORT_NOTE:])
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
        # Вид может смениться по ходу: в note — текущий, а не дописанный однажды.
        if self.view == "player":
            note += " · вид игрока"
        elif self.view == UNKNOWN and passport is not None and self._pipe is not None:
            note += f" · вид неизвестен ({passport['items']['view']['how']})"
        if champs_error:
            ago = round(time.time() - champs_error["time"])
            note += f" · состав: ошибка {ago} с назад — {champs_error['text']}"
        api_error = getattr(champs, "api_error", None) if champs is not None else None
        if api_error:
            note += f" · состав из API не принят: {api_error}"
        if pnote:
            note += " · паспорт: " + "; ".join(pnote)
        return {"id": self.sid, "run": self.run, "state": self.state.label,
                "active": self.state.active,
                "kind": self.kind, "model": self.model_label,
                "trained_on": self.trained_on, "timing": timing(tail),
                "note": note, "champs_error": champs_error, "api_error": api_error,
                "fps": self.fps, "start": self.start,
                "title": self.title, "duration": round(self.duration),
                "source": self.source.describe(), "source_state": dict(self.source_state),
                "source_gaps": dict(self.source_gaps),
                "api": self._api_view(), "api_reason": self.api_reason,
                "passport": passport, "passport_events": pev, "passport_events_n": n_pev,
                "state_url": self.state_url, "state_reason": self.state_reason,
                "loadouts": self._loadouts_view(),
                "count": n, "since": since, "frames": rows, "roster": self.roster,
                "window": self.window, "first": first,
                "dropped_before": max(0, first - since),
                "minimap": self.minimap, "view": self.view, "elapsed": round(el, 1),
                "speed": speed, "map_units": MAP_UNITS}

    def _loadouts_view(self) -> dict | None:
        """Паспорт выбора (#807): куда записано, что пропущено, ошибка; None — выключен."""
        rec = self.loadouts
        if rec is None:
            return None
        api_err = getattr(self.api, "observer_error", None)
        return {"out_dir": None if rec.out_dir is None else str(rec.out_dir),
                "games": len(rec.games),
                "written": [{"file": str(p), "rows": n} for p, n in rec.written],
                "skipped": list(rec.skipped), "error": rec.error or api_err}

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
        self._serve()
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
            if self.loadouts is not None:
                try:
                    self.loadouts.close()
                except Exception as exc:                      # noqa: BLE001
                    self.loadouts.error = f"{type(exc).__name__}: {exc}"
            if self.state_server is not None:
                self.state_server.close()
                self.state_reason = "сеанс закрыт"

    def _serve(self) -> None:
        """Поднять сервис состояния. Порт занят — сеанс идёт без него, причина
        в status() (`state_reason`), а не молчание."""
        if self.state_port is None:
            return
        try:
            self.state_server = StateServer(self.state_inputs, port=self.state_port).start()
        except OSError as exc:
            self.state_reason = f"не поднят: {type(exc).__name__}: {exc}"
            return
        self.state_url, self.state_reason = self.state_server.url, None

    def _seen(self, fr) -> None:
        """Запомнить ответ источника: состояние меняется — меняется since."""
        key = fr.status.key
        if key != self.source_state["state"]:
            self.source_state = {"state": key, "reason": fr.reason, "t": round(fr.t, 2),
                                 "since": time.time(), "size": fr.size}
        else:
            self.source_state.update(reason=fr.reason, t=round(fr.t, 2), size=fr.size)
        if fr.ok:
            self.last_ok = time.time()
        elif fr.status is not Status.ENDED:
            self.source_gaps[key] = self.source_gaps.get(key, 0) + 1

    # ---------- паспорт ----------

    def _sig(self, signal: Callable[[Passport], list[Change]]) -> list[Change]:
        """Сигнал паспорту под замком сеанса (status() и сервис читают его из
        своих потоков); изменения — в `passport_events` и note (смена типа
        сцены — только в события: она частая и видна в строке кадра)."""
        with self._lock:
            changes = signal(self.passport)
            for ch in changes:
                self.passport_events.append(ch.to_dict())
                if ch.key != "item:scene":
                    self._pnote.append(ch.text)
            del self.passport_events[:-PASSPORT_KEEP]
            del self._pnote[:-PASSPORT_NOTE]
        return changes

    def _open_passport(self, d: dict) -> None:
        """Паспорт после `source.open()`: источник и размер — из describe().
        У своего экрана вид — сразу игрока (это своя игра), у ролика — не
        решён, пока его не решат чтения HUD."""
        p = Passport.build(d, t=None)
        with self._lock:
            self.passport = p
        if self.api is None:
            self._sig(lambda p: p.set_api(None, False, None, reason=self.api_reason))
        if self.source.kind == "screen":
            self._sig(lambda p: p.set_view("player", None, how="свой экран: своя игра",
                                           source="source"))
            self.view = "player"

    def _frame_size(self, size: tuple[int, int], t: float) -> None:
        """Размер кадра → паспорт. У экрана кадр — клиентская область окна
        игры: прямоугольник игры известен сразу, перемерять нечего."""
        if self.passport.items["resolution"].value == tuple(size):
            return
        self._sig(lambda p: p.set_frame_size(size, t))
        if self.source.kind == "screen":
            self._sig(lambda p: p.set_game_rect((0, 0, *size), t, source="source",
                                                how="кадр — клиентская область окна игры"))

    def _assets(self, pl: Pipeline, t: float | None) -> None:
        """Что есть у пути кадра — в паспорт (читатели без ресурса отказывают)."""
        have = {"unit_net": pl.unit_model is not None,
                "names_bank": pl.champs is not None and pl.champs.names is not None,
                "minimap_model": pl.champs is not None}
        for name, ok in have.items():
            if self.passport.items["assets"].value.get(name) != ok:
                self._sig(lambda p, name=name, ok=ok: p.set_asset(name, ok, t))

    def _new_pipe(self, size: tuple[int, int], champs: LiveChampions | None = None,
                  t: float | None = None) -> Pipeline:
        """Собрать путь кадра под размер и привязать к нему сеанс. `champs` —
        чемпионы прежнего пути (пересборка при смене размера)."""
        pl = (self._build(size, self.run, self.device, self.fps, champs=champs)
              if champs is not None else self._build(size, self.run, self.device, self.fps))
        pl.locator = self._new_locator(pl.size)
        self._pipe = pl
        self.kind, self.model_label = pl.model.kind, pl.model.label
        self.trained_on = list(pl.model.trained_on)
        self.minimap = dict(pl.minimap)
        self.proj, self.tracker = pl.proj, pl.tracker
        if self.match is not None:
            # Память матча переживает смену размера, проекция — новая.
            self.match.proj = pl.proj
        self._assets(pl, t)
        return pl

    def _new_locator(self, size: tuple[int, int]) -> Locator:
        """Поиск миникарты под размер кадра: с сохранённым прямоугольником
        своего экрана и явным масштабом, если они есть."""
        saved = (load_fit(self.minimap_store, size) if self.minimap_store is not None
                 else None)
        return Locator(size, saved=saved, manual=self.minimap_scale)

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
        self._open_passport(d)
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
            if self.window is None and len(self.rows) >= MAX_FRAMES:
                self.note += f" · потолок памяти {MAX_FRAMES} кадров: дальше не читаю"
                break
            self._drain_api()
            # Время «decode» — ожидание кадра от источника: у ролика сеть до
            # YouTube, NVDEC и прореживание до fps; у экрана — такт и снимок.
            t0 = time.perf_counter()
            fr = src.read()
            t1 = time.perf_counter()
            wall = time.time()                   # когда кадр получен: к нему — снимок API
            if self._stop.is_set():
                break
            self._seen(fr)
            if fr.status is Status.ENDED:
                if fr.reason and fr.reason != "ролик кончился":
                    self.note += f" · {fr.reason}"
                break
            if not fr.ok:
                continue                         # пауза, окна нет — строк нет
            self._frame_size(fr.size, fr.t)
            if pl is None or fr.size != pl.size:
                if pl is not None:
                    # Окно растянули: кадры прогрева — по старому размеру,
                    # дальше — новый путь под новый размер. Чемпионы от
                    # размера не зависят: состав переходит в новый путь.
                    self._flush(pl)
                    old, champs = pl.size, pl.champs
                    pl.champs = None             # не гасить поток состава в close()
                    pl.close()
                    pl = self._new_pipe(fr.size, champs=champs, t=fr.t)
                    self.note += (f" · размер кадра {old[0]}x{old[1]} → "
                                  f"{fr.size[0]}x{fr.size[1]}: путь кадра пересобран")
                else:
                    pl = self._new_pipe(fr.size, t=fr.t)
                    self.note = f"{self._source_note(fr.size)} · {pl.note}"
            self._step(pl, fr.image, fr.t, (t1 - t0) * 1000, wall)
        if pl is not None and not self._stop.is_set():
            self._flush(pl)                      # поток кончился раньше прогрева

        self._out.shutdown(wait=True)    # все кадры записаны — тогда конец
        if self.state is State.RUNNING:
            self.state = State.STOPPED if self._stop.is_set() else State.ENDED

    # ---------- сигналы пути кадра ----------

    def _api_step(self, t: float, wall: float) -> tuple[ApiObs | None, str | None]:
        """Снимок Live Client на момент кадра → (наблюдение контекста матча
        или None, почему снимок игры не принят). Смена состояния API — сигнал
        паспорту."""
        if self.api is None:
            return None, None
        st = self.api.status()
        if st.state is not self._api_state:
            self._api_state = st.state
            self._sig(lambda p: p.set_api(st.state.value, st.state is ApiState.IN_GAME, t,
                                          st.reason))
        obs, why = status_observation(st, wall)
        if obs is not None:
            self._api_obs, self._api_obs_t = obs, t
            self.map_number = obs.map_number
            return obs, None
        if self._api_obs_t is not None and t - self._api_obs_t > API_STALE:
            self._api_obs = None
        # Отказ по содержимому (чемпион не опознан, не 5/5) — в строку кадра;
        # отказ по состоянию (no_game, error) виден в паспорте.
        content = st.game is not None and st.state in (ApiState.LOADING, ApiState.IN_GAME)
        return None, why if content else None

    def _spells_step(self, pl: Pipeline, fr: np.ndarray, t: float, game: bool,
                     view: str) -> list[dict] | None:
        """Готовность своих заклинаний D/F по HUD (#806, lolvision.hud_spells).

        Только на игровом кадре вида игрока. Пока панель умений не найдена —
        поиск раз в SPELL_LOCATE_EVERY с; находка (AGREE подряд в одном
        месте) — паспорту `set_spell_panel`. Читатель `hud_spells` паспорта
        разрешён — состояния слотов (`SlotRead.to_dict()`), иначе None.
        Заклинания слотов из API (`spell_variants`) сменились — эталон
        готовой иконки слота набирается заново."""
        hud = pl.spells
        if not game or view != "player":
            return None
        if hud.panel is None:
            if pl.spell_try_t is not None and t - pl.spell_try_t < SPELL_LOCATE_EVERY:
                return None
            pl.spell_try_t = t
            panel = hud.offer(fr)
            h, w = fr.shape[:2]
            if panel is not None:
                x0, y0, x1, y1 = panel.box("Q")[0], panel.box("Q")[1], *panel.box("F")[2:]
                box = (x0 / w, y0 / h, x1 / w, y1 / h)
                how = (f"найдена по кадру: масштаб {panel.scale:.2f} к своему HUD, иконка "
                       f"{panel.slot_px():.0f} px, совпадение {panel.score:.2f}")
                self._sig(lambda p: p.set_spell_panel(True, t, how, box, panel.scale,
                                                      panel.score))
            elif hud.tried >= SPELL_LOCATE_SAY and not pl.spell_said:
                pl.spell_said = True
                self._sig(lambda p: p.set_spell_panel(
                    False, t, f"не найдена за {hud.tried} попыток, поиск продолжается"))
            return None
        if "hud_spells" in self.passport.disabled():
            return None
        if self.api is not None:
            g = self.api.status().game
            me = None if g is None else g.my_player
            if me is not None and me.spells is not None:
                hud.set_spells(tuple(x.variant for x in me.spells))
        return [r.to_dict() for r in hud.read(fr, t).values()]

    def _my_team(self) -> tuple[str | None, str | None]:
        """Команда своего игрока по API для sides.assign или (None, причина)."""
        ok, why = self.passport.verdict("api_facts")
        if not ok:
            return None, why
        st = self.api.status()
        g = st.game
        if g is None:
            return None, f"нет снимка игры ({st.state.value})"
        me = g.my_player
        return (me.team, None) if me is not None else (None, g.me_reason)

    def _view_step(self, champs: LiveChampions | None, t: float) -> str:
        """Вид трансляции → паспорт (только смена). Свой экран — вид игрока,
        если API не сказал, что своего игрока нет (наблюдатель в клиенте). У
        ролика — по чтениям HUD (LiveChampions.view); не решён — unknown с
        причиной, а не наблюдатель по умолчанию."""
        if self.source.kind == "screen":
            if self._api_obs is not None and self._api_obs.me_team is None:
                view, how, src = "spectator", "Live Client API: своего игрока нет", "api"
            else:
                view, how, src = "player", "свой экран: своя игра", "source"
        elif champs is None:
            view, how, src = None, "нет модели иконок миникарты: HUD не читается", "hud"
        else:
            # Только по положительным признакам (#799): «портретов не видно» —
            # не вид игрока, а unknown или прежний вид.
            view, how, src = champs.view, champs.view_how(), "hud"
        if (view or UNKNOWN) != self.passport.items["view"].value:
            self._sig(lambda p: p.set_view(view, t, how=how, source=src))
        self.view = self.passport.items["view"].value
        return self.view

    def _roster_step(self, champs: LiveChampions | None, t: float) -> None:
        """Откуда состав → паспорт: api, hud (если паспорт разрешает состав
        по HUD: вид наблюдателя) или никак. Состав по портретам на виде игрока
        не принимается (#786)."""
        if champs is None:
            src, how = None, "нет модели иконок миникарты"
        elif champs.roster is None:
            src = None
            how = (f"состав из API не принят: {champs.api_error}" if champs.api_error
                   else "не прочитан")
        elif champs.match is not None and champs.match.api:
            src, how = "api", "Live Client API"
            if len(champs.roster) < 10:
                # Тренировка (#803): API отдаёт только своего — состав неполный.
                n = len(champs.roster)
                how = f"Live Client API, неполный состав: {n} из 10 (тренировка)"
        else:
            ok, why = self.passport.verdict("hud_roster")
            src, how = ("hud", "портреты HUD") if ok else (None, f"состав по HUD не принят: {why}")
        if src != self.passport.items["roster"].value:
            self._sig(lambda p: p.set_roster(src, t, how))
        self.roster = champs.roster if src is not None else None

    def _minimap_step(self, pl: Pipeline, region: np.ndarray, game: bool, t: float) -> None:
        """Поиск миникарты и его итог → паспорт: найдена, поиск сдался или
        перепроверка (только по сигналу паспорта и на игровом кадре)."""
        locator = pl.locator
        if locator.box is None:
            if game and locator.tried < LOCATE_MAX:
                if locator.offer_region(region):
                    self._found(pl, t)
            elif locator.tried >= LOCATE_MAX:
                if pl.locate_idle is None:           # круг только что кончился
                    pl.locate_idle = t
                    if not pl.locate_done:
                        pl.locate_done = True
                        self._sig(lambda p: p.minimap_missing(
                            t, f"не найдена за {LOCATE_MAX} попыток — прямоугольник раскладки"))
                elif t - pl.locate_idle >= LOCATE_PAUSE:
                    # Новый круг: паспорт уже знает об отказе и узнает о находке.
                    pl.locator, pl.locate_idle = self._new_locator(pl.size), None
            return
        if not (game and "minimap" in self.passport.pending):
            return
        # Перепроверка в том же месте: ответ в другом прямоугольнике — не она.
        fit = locator.locate(region)
        same = fit is not None and corner_px(fit.box, locator.box, pl.size) <= AGREE_PX
        self._sig(lambda p: p.check_minimap(fit.score if same else 0.0, t))
        if not self.passport.items["minimap"].value["found"]:
            self._lost(pl)

    def _found(self, pl: Pipeline, t: float) -> None:
        locator = pl.locator
        fit = locator.fit
        pl.mmbox = box_px(locator.box, pl.size)
        pl.minimap = {"box": list(locator.box), "scale": fit.scale, "found": True,
                      "how": fit.how}
        self.minimap = dict(pl.minimap)
        self.note += f" · миникарта ×{fit.scale:g}" + ("" if fit.how == "map" else
                                                       f" ({FIT_HOW[fit.how]})")
        if fit.how == "map" and self.minimap_store is not None:
            # Найдена эталоном — запомнить для другой карты (ARAM, #803).
            try:
                save_fit(self.minimap_store, pl.size, fit)
            except OSError as exc:
                self.note += f" · прямоугольник миникарты не сохранён: {exc}"
        if pl.bw is not None:
            # Найдена после прогрева: размер рамки оценён в чужой сетке.
            # Вьюпорт постоянен в единицах карты, поэтому в сетке своей
            # миникарты он меньше в масштаб раз.
            pl.bw = round(pl.bw / locator.fit.scale)
            pl.bh = round(pl.bh / locator.fit.scale)
        self._sig(lambda p: p.set_minimap(locator.box, fit.scale, fit.score, t,
                                          how=f"найдена в кадре: {FIT_HOW[fit.how]}"))

    def _lost(self, pl: Pipeline) -> None:
        """Паспорт снял находку миникарты: снова раскладка и новый поиск."""
        scale = pl.locator.fit.scale if pl.locator.fit is not None else 1.0
        pl.locator, pl.locate_done, pl.locate_idle = self._new_locator(pl.size), False, None
        pl.mmbox = box_px(pl.layout_mm, pl.size)
        pl.minimap = {"box": list(pl.layout_mm), "scale": None, "found": False}
        self.minimap = dict(pl.minimap)
        if pl.bw is not None:
            pl.bw, pl.bh = round(pl.bw * scale), round(pl.bh * scale)   # обратно в сетку раскладки

    def _step(self, pl: Pipeline, fr: np.ndarray, t: float, decode_ms: float,
              wall: float | None = None) -> None:
        """Путь одного кадра: подготовка, тип сцены, миникарта, API, вид,
        юниты, чемпионы — и в _emit (после прогрева рамки). `wall` —
        unix-время получения кадра: на него берётся снимок Live Client."""
        wall = time.time() if wall is None else wall
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
        self._sig(lambda p: p.set_scene(kind, t, game))
        # Магазин и Tab — игра под оверлеем (#786): сцену закрывает окно,
        # но миникарта видна, и чемпионы на ней читаются.
        overlay = kind in OVERLAY_KINDS
        region = fr[ry0:, rx0:]
        self._minimap_step(pl, region, game, t)
        locator = pl.locator                     # после шага: потерянную ищет новый
        mini = _mini(region, pl.mmbox, rx0, ry0)
        mask = white_mask(np.asarray(mini, dtype=np.float32))

        # Live Client: состав из API — до миникарты (иконки назначаются
        # только среди этих десяти, в тренировке — среди тех, кого отдал API,
        # #803). API пропал дольше API_STALE — состав
        # снова по HUD; одиночный сбой опроса состав не сбрасывает.
        obs, refused = self._api_step(t, wall)
        champs = pl.champs
        if champs is not None:
            if obs is not None:
                champs.set_api(obs)
            elif self._api_obs is None:
                champs.set_api(None)
            if game:
                champs.offer(fr, t)
        view = self._view_step(champs, t)
        self._roster_step(champs, t)
        off = self.passport.disabled()           # кому паспорт отказал на этом кадре

        # Детектор юнитов — только на известном виде (маска своя у каждого
        # вида, #786); вид неизвестен — не запускается: null с причиной.
        units = None
        if pl.unit_model is not None and "units" not in off:
            want = (view, locator.box if view == "player" else None)
            if want != pl.zone:
                pl.zone = want
                keep, valid, valid_src = _unit_zone(*pl.size, view, locator.box)
                pl.unit_model.set_zone(keep, valid)
                pl.tracker.valid = valid_src
            t4 = time.perf_counter()
            units = pl.unit_model.units(pl.fr_dev)
            ms["units"] = (time.perf_counter() - t4) * 1000
        names = None
        if champs is not None and (game or overlay) and "minimap_icons" not in off:
            t5 = time.perf_counter()
            ms["champs"] = champs.minimap(fr, np.asarray(mini))
            t6 = time.perf_counter()
            ms["mini"] = (t6 - t5) * 1000
            if units and game and "names" not in off:
                names = champs.who(fr, units)
                ms["who"] = (time.perf_counter() - t6) * 1000
        if units is not None:
            # Юниты детектора — записи records.UnitRec (доли кадра, имя), со
            # своими / врагами / собой и стороной (sides.py, #790).
            recs = [UnitRec.from_unit(u, pl.size, names[k] if names else None)
                    for k, u in enumerate(units)]
            my_team, my_why = self._my_team()
            ctx = SideContext(view=None if view == UNKNOWN else view, my_team=my_team,
                              my_team_reason=my_why)
            ms["unit_recs"] = assign_sides(recs, ctx, fr)
        ms["spells"] = self._spells_step(pl, fr, t, game, view)
        ms.update(full_img=_full_small(fr, pl.fr_dev), off=off, view=view, api_obs=obs,
                  api_refused=refused, wall=wall)
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
            if (pl.n_good < WARMUP or pl.locator.box is None) and len(pl.warm) < WARMUP_MAX:
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

    def _clock_reason(self, refused: str | None) -> str:
        """Почему у кадра нет времени матча по API."""
        if self.api is None:
            return f"Live Client API: {self.api_reason}"
        ok, why = self.passport.verdict("api_facts")
        if not ok:
            return why
        if refused:
            return f"снимок API не принят: {refused}"
        return f"нет свежего снимка API (дольше {API_STALE:g} с)"

    def _emit(self, mask, scene: np.ndarray, prep, mini: Image.Image,
              bw: int, bh: int, model: Runner, ms: dict, t: float) -> None:
        """Один кадр: истина с миникарты, ответ модели, картинки в память.

        `t` — время кадра по шкале источника (Frame.t): у ролика — позиция в
        ролике, у экрана — секунды от открытия источника.

        `ms` — уже набежавшее время кадра (распаковка; подготовка входа
        модели — маска, обрезка, масштаб, вид сверху) и то, что путь кадра
        передал строке (записи, отказы паспорта, снимок API); сюда
        добавляются сеть, сведение голосов и «ответ» — сжатие картинок и
        запись в сеанс. Метка с миникарты в путь модели не входит.

        Истина камеры — только если паспорт её разрешает (`camera_truth`:
        миникарта найдена в кадре или вид наблюдателя). Иначе миникарта
        вырезана по раскладке мимо настоящей, и рамка по ней — мусор: в строке
        cx, cy, q, err = None, truth = "no_minimap" и причина паспорта;
        ответ модели остаётся. Решение берётся на момент выдачи строки: кадры
        прогрева перерезаны по найденной миникарте.

        Читатели, которым паспорт отказал на кадре, — `row["off"]`
        {читатель: причина}; их выход в строке — null.
        """
        off = dict(ms.pop("off", None) or {})
        truth, why = self.passport.verdict("camera_truth")
        if truth:
            off.pop("camera_truth", None)
            x, y, q, _ = find_box_clipped(mask, bw, bh)
            cx, cy = (x + bw / 2) / MM, (y + bh / 2) / MM
        else:
            off["camera_truth"] = why

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
               "err": None, "spread": _r(out["spread"], 3), "q": None,
               "view": ms.pop("view", self.view)}
        if truth:
            row.update(cx=round(cx, 4), cy=round(cy, 4), q=round(q, 3),
                       err=_r(np.hypot(cx - out["px"], cy - out["py"]), 4))
        else:
            row["truth"], row["truth_reason"] = "no_minimap", why
        for k in ("agree", "scene", "game"):
            if k in out:
                row[k] = _r(out[k], 3)
        for k in ("votes", "is_game"):
            if k in out:
                row[k] = out[k]
        # Записи кадра (records.py) идут в контекст и трекер как есть, в
        # строку — словарями: строка уходит в JSON. Отказ паспорта — null.
        units = ms.pop("unit_recs", None)
        full_img = ms.pop("full_img", None)
        spells = ms.pop("spells", None)
        if spells is not None:
            row["spells"] = spells
        elif "hud_spells" in off:
            row["spells"] = None
        if units is not None:
            row["units"] = dicts(units)
        elif "units" in off:
            row["units"] = None
        ran = "champs" in ms
        mini_champs = ms.pop("champs", None)
        if mini_champs is not None:
            row["champs"] = dicts(mini_champs)
        elif ran or "minimap_icons" in off:
            row["champs"] = None
            if "minimap_icons" not in off:
                off["minimap_icons"] = "состав не решён"
        kp = ms.pop("kind_p", None)
        # Карта по API (#803): не Ущелье — модель камеры и проекция экрана
        # (единицы Ущелья) не годятся; камера — по рамке на миникарте.
        on_rift = rift(self.map_number)
        if kp is not None:
            # DINO узнаёт интерфейс, патчевая модель — места Ущелья: «игра»
            # без её подтверждения — не наша карта или чужая игра. Если API
            # сказал, что карта другая (ARAM), её «не Ущелье» — не довод.
            c = int(np.argmax(kp))
            if KIND_CLASSES[c] == "game" and row.get("is_game") is False and on_rift:
                c = KIND_CLASSES.index("other")
            row["kind"] = KIND_CLASSES[c]
            row["kind_p"] = round(float(kp[int(np.argmax(kp))]), 3)
        obs, refused = ms.pop("api_obs", None), ms.pop("api_refused", None)
        wall = ms.pop("wall", None)
        # Контекст матча: память о чемпионах между кадрами, смерти и
        # возрождения, снимок API (источник api). Под замком: сервис
        # состояния читает контекст из своего потока.
        with self._lock:
            if self.match is None:
                # Места чемпионов с экрана — той же проекцией, что у трекера.
                self.match = MatchState(proj=self.proj)
            ev = self.match.update(row["t"], row.get("kind", "game"), mini_champs, units,
                                   (row["px"], row["py"]) if on_rift else None, api=obs)
            self._match_at = wall if wall is not None else time.time()
            ctx = self.match.frame_view() if self.match.order else None
            clock = self.match.game_clock()
            conflicts = dict(self.match.conflicts)
            refused = refused or self.match.api_refused
            self._api_refused = refused          # сервису состояния: причина отказа API
        if ctx is not None:
            row["ctx"] = dicts(ctx)
        if ev:
            row["ev"] = dicts(ev)
        row["game_time"] = ({"value": round(clock.value, 1), "src": clock.src, "reason": None}
                            if clock is not None else
                            {"value": None, "src": None, "reason": self._clock_reason(refused)})
        row["conflicts"] = conflicts
        row["api_refused"] = refused
        row["off"] = off
        # Треки юнитов: id, состояние (seen / unseen / lost), место на карте.
        # События юнитов — отдельно от событий чемпионов: смертей крипов
        # десятки в минуту.
        tracks = None
        if self.tracker is not None and any(isinstance(e, RosterEvent) for e in ev):
            # Новая игра (другой состав): треки прошлой — не этой (#801). У
            # чемпионов срока нет, и без сброса они висели бы «unseen» вечно.
            self.tracker.reset()
        if (self.tracker is not None and row.get("kind", "game") == "game"
                and units is not None and on_rift):
            cam = (row["px"], row["py"])
            uev = self.tracker.update(row["t"], None if cam[0] is None else cam, units)
            tracks = self.tracker.frame_view(row["t"])
            row["tracks"] = dicts(tracks)
            if uev:
                row["uev"] = dicts(uev)
        if on_rift:
            cam = None if row["px"] is None or row["py"] is None else (row["px"], row["py"])
            vision = VisionFrame(i, row["t"], time.time(), row.get("kind"), cam, row["spread"],
                                 units, tracks, spells=spells,
                                 spells_why=off.get("hud_spells") or _spells_why(row))
        else:
            # Не Ущелье: центр камеры — рамка вьюпорта на миникарте (доли
            # миникарты годятся для любой карты), если миникарта найдена и
            # рамка на ней видна (под окном аугментов ARAM её нет: q = 0).
            off_map = f"карта {self.map_number} — не Ущелье"
            cam, cam_why = None, None
            if row["cx"] is None:
                cam_why = f"{off_map}: модель камеры знает только Ущелье, а миникарты нет ({why})"
            elif row["q"] < MINIMAP_CAM_Q:
                cam_why = (f"{off_map}: модель камеры знает только Ущелье, а рамка на миникарте "
                           f"не видна (q {row['q']:g} < {MINIMAP_CAM_Q:g})")
            else:
                cam = (row["cx"], row["cy"])
            vision = VisionFrame(i, row["t"], time.time(), row.get("kind"), cam, None,
                                 units, None, camera_src="minimap", camera_why=cam_why,
                                 tracks_why=f"{off_map}: треки ведутся в единицах Ущелья",
                                 spells=spells,
                                 spells_why=off.get("hud_spells") or _spells_why(row))
        with self._lock:
            self.vision = vision
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
            if self.window is not None and len(self.rows) > self.window:
                self._drop(len(self.rows) - self.window)

    def _drop(self, k: int) -> None:
        """Выбросить `k` старейших кадров окна со всеми картинками (под замком)."""
        for r in self.rows[:k]:
            j = r["i"]
            for store in (self.scene, self.mini, self.full, self.heat):
                store.pop(j, None)
        del self.rows[:k]
        self.dropped += k


def _spells_why(row: dict) -> str | None:
    """Почему у кадра нет чтения заклинаний, если паспорт его не запрещал."""
    if row.get("spells") is not None:
        return None
    if row.get("kind", "game") != "game":
        return f"кадр не игровой ({row.get('kind')})"
    return "панель умений ещё ищется"


# Сервис состояния сеанса ещё поднимается — временная причина, а не отказ.
STATE_PENDING = "ещё не поднят"


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


def start_screen(run: str = DEFAULT_RUN, fps: float = 2.0, backend: str = "auto",
                 minimap_scale: float | None = None, loadouts: bool = True) -> dict:
    """Сеанс по своему экрану (окно игры) вместе с Live Client своей игры и
    сервисом состояния для coach на 127.0.0.1:STATE_PORT (state_service.py;
    адрес — `state_url` в status()).

    Миникарта: найденный эталоном прямоугольник запоминается вне репозитория
    (minimap_locate.fit_file) и выручает на других картах; `minimap_scale` —
    масштаб миникарты из настроек клиента, тогда без поиска (#803).

    `loadouts` — писать паспорт выбора игры (роль, заклинания, руны, предметы
    всех десяти, без ников) в data/loadouts/ по концу игры (#807).

    Захват открывается в потоке сеанса (mss привязан к потоку), не здесь."""
    fps = check_fps(fps)
    if minimap_scale is not None and not SCALES[0] <= minimap_scale <= SCALES[1]:
        raise ValueError(f"масштаб миникарты {minimap_scale:g} вне {SCALES[0]:g}–{SCALES[1]:g}")
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"неизвестный бэкенд захвата {backend!r}: auto, {', '.join(BACKENDS)}")
    run = run or DEFAULT_RUN
    run_checkpoint(run)

    def make() -> Session:
        rec = live_recorder() if loadouts else None
        api = LiveClient(champion_ids=ddragon_champion_ids())
        if rec is not None:
            api.on_status = rec.observe          # каждый опрос API — в паспорт выбора
        return Session(SCREEN_ID, run, ScreenSource(fps, backend=backend), api=api,
                       state_port=STATE_PORT, minimap_store=fit_file(),
                       minimap_scale=minimap_scale, loadouts=rec)
    return _open(SCREEN_ID, run, make)


def get(sid: str) -> Session | None:
    return _sessions.get(sid)


def stop(sid: str) -> None:
    s = _sessions.get(sid)
    if s:
        s.stop()


def listing() -> list[dict]:
    return [{"id": s.sid, "state": s.state.label, "active": s.state.active,
             "count": s.count, "run": s.run, "kind": s.kind, "title": s.title,
             "source": s.source.kind}
            for s in _sessions.values()]
