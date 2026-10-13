"""Готовность своих заклинаний призывателя (D/F) по HUD вида игрока (#806).

Зачем. Live Client API говорит, какие заклинания взяты (`me.spells`), но не
говорит, готовы ли они. На HUD вида игрока это видно: у иконки слота D/F
на перезарядке поверх — тёмно-синяя заливка и белые цифры секунд; готовая
иконка — яркая картинка заклинания. Ещё два вида «не нажать»: запрет
(иконка темнеет, поверх красный крестик — контроль или своё умение) и
смерть (панель серая; заливка перезарядки при этом остаётся синей).

Как (замеры и проверка — docs/hud-spells.md).

1. Панель умений ищется по кадру (`locate_panel`): эталон — вырез панели
   своего HUD (dataset/hud_spells/panel_template.png, кадр своей игры
   1920×1080), сравнение NCC в градациях серого по рамкам и фону панели
   (иконки, цифры, полосы ХП/маны и точки уровней закрыты маской), по
   сетке масштабов SCALES — масштаб HUD у стримеров свой (0.96–1.28 от
   своего на роликах). Слоты — постоянные прямоугольники в координатах
   эталона (`SLOTS`), умноженные на масштаб.
2. Есть ли панель на этом кадре — то же совпадение на найденном месте
   (`panel_score`): ниже PRESENT_MIN — панель закрыта или кадр не игровой,
   ответ null.
3. Слот → вырез 24×24 без рамки → признаки (`slot_features`): доля ровной
   «синей заливки перезарядки» (оттенок 195–235°, насыщенность > 0.45,
   яркость 40–150, без рисунка — синие пиксели иконок Призрака и синей
   Кары с рисунком), белые цифры с тёмной обводкой в середине, красный
   крестик на диагоналях, средняя яркость и насыщенность.
4. Эталон готовой иконки — свой у каждого слота, набирается в сеансе
   (`SlotRef`): медиана последних вырезов, которые не похожи на перезарядку,
   запрет и смерть и не темнее эталона. Готовность — относительная: вырез
   похож на эталон (NCC, яркость, насыщенность), без синей заливки и цифр
   сверх эталона. Абсолютный порог «белые цифры» врёт на готовом Призраке
   (#804), а относительное сравнение — нет: белая фигура есть и в эталоне.
5. Решение по слоту (`decide`, `SlotRead.state`): `cooldown` → False
   (ровная синяя заливка; или тёмная иконка с цифрами — конец перезарядки;
   заливка видна и у мёртвого); `ready` → True; `blocked` (красный
   крестик), `dead` (полоса ХП пуста), `hidden` (панели нет), `unsure`
   (нет эталона, вырез не похож ни на что) → None с причиной. Ошибка
   «готово, когда не готово» хуже обратной: при сомнении — None.

Пример:
    hud = SpellHud()
    hud.offer(frame)                  # пока панель не найдена — ищет её
    reads = hud.read(frame, t)        # {"D": SlotRead, "F": SlotRead}
    reads["D"].value                  # True / False / None
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lolvision.paths import DATASET

TEMPLATE = DATASET / "hud_spells" / "panel_template.png"
# Эталон снят с кадра своей игры: начало выреза в кадре 1920×1080.
TEMPLATE_ORIGIN = (735, 975)
# Слоты в координатах эталона (x0, y0, x1, y1): внутренность иконки без рамки.
SLOTS = {"Q": (69, 16, 107, 53), "W": (114, 16, 152, 53), "E": (159, 16, 198, 53),
         "R": (205, 16, 243, 53), "D": (257, 16, 284, 43), "F": (291, 16, 319, 43)}
SPELL_SLOTS = ("D", "F")
# Полоса ХП в координатах эталона: у мёртвого пустая.
HP_BAR = (36, 72, 316, 82)
# Что маска эталона закрывает: внутренность слотов, точки уровней, полосы,
# подписи клавиш, портрет — меняется от кадра к кадру и от чемпиона.
_CUTS = [(x0 + 2, 15, x1 - 2, 52) for x0, x1 in
         ((33, 62), (68, 108), (112, 153), (158, 199), (204, 244), (256, 285), (290, 320))]
_CUTS += [(30, 55, 330, 64), (33, 69, 318, 97), (250, 40, 320, 52), (0, 0, 30, 105)]
# Масштаб HUD относительно эталона. Свой HUD — 1.0; на роликах 0.96–1.14.
SCALES = tuple(round(s, 2) for s in np.arange(0.70, 1.62, 0.02))
# Где искать панель: низ кадра по центру (доли кадра x0, y0, x1, y1).
SEARCH = (0.20, 0.74, 0.80, 1.0)
# Совпадение с эталоном: находка панели и «панель на месте» на кадре.
FOUND_MIN = 0.52
PRESENT_MIN = 0.35
AGREE = 3            # столько находок подряд в одном месте — панель найдена
AGREE_PX = 4.0

CROP = 24            # сторона выреза слота для признаков
NAVY_FLAT = 160.0    # пиксель заливки: перепад яркости (Собель, |gx|+|gy|) ниже
INSET = 0.08         # отступ от рамки слота, доля стороны


@dataclass(frozen=True)
class Panel:
    """Панель умений в кадре: начало эталона (x, y, пиксели кадра), масштаб
    относительно эталона и совпадение."""
    x: float
    y: float
    scale: float
    score: float

    def box(self, key: str) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = SLOTS[key]
        s = self.scale
        return (self.x + x0 * s, self.y + y0 * s, self.x + x1 * s, self.y + y1 * s)

    def slot_px(self) -> float:
        """Сторона иконки заклинания в пикселях кадра."""
        x0, _, x1, _ = SLOTS["D"]
        return (x1 - x0) * self.scale

    def to_dict(self) -> dict:
        return {"x": round(self.x, 1), "y": round(self.y, 1), "scale": round(self.scale, 3),
                "score": round(self.score, 3), "slot_px": round(self.slot_px(), 1)}


_tpl_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def template(path: Path = TEMPLATE) -> tuple[np.ndarray, np.ndarray]:
    """(серый эталон float32, маска uint8)."""
    key = str(path)
    if key not in _tpl_cache:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = np.ones(g.shape, np.uint8)
        for x0, y0, x1, y1 in _CUTS:
            m[y0:y1, x0:x1] = 0
        _tpl_cache[key] = (g, m)
    return _tpl_cache[key]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _scaled(s: float) -> tuple[np.ndarray, np.ndarray]:
    g, m = template()
    h, w = g.shape
    size = (max(8, round(w * s)), max(8, round(h * s)))
    t = cv2.resize(g, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mk = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return t, mk


def search_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return (round(SEARCH[0] * w), round(SEARCH[1] * h), round(SEARCH[2] * w), round(SEARCH[3] * h))


def _match(g: np.ndarray, s: float, half: bool = False) -> tuple[float, tuple[int, int]]:
    t, mk = _scaled(s)
    if half:
        t = cv2.resize(t, (t.shape[1] // 2, t.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(mk, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
    if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1] or not mk.any():
        return -1.0, (0, 0)
    r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED, mask=mk)
    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return float(mx), loc


def locate_panel(frame: np.ndarray, scales=SCALES, whole: bool = False) -> Panel | None:
    """Лучшее место панели в низу кадра (RGB, H×W×3) по всем масштабам.
    `whole` — искать по всей картинке (она уже вырез низа кадра).

    Грубо — на картинке вдвое меньше по каждому второму масштабу, точно — в
    окне ±12 px вокруг лучшего по пяти соседним масштабам."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if whole else search_box((w, h))
    g = _gray(frame[y0:y1, x0:x1])
    g2 = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    coarse = [(_match(g2, s, half=True), s) for s in scales[::2]]
    (sc, loc), s0 = max(coarse, key=lambda c: c[0][0])
    if sc <= -1.0:
        return None
    best: Panel | None = None
    i0 = list(scales).index(s0)
    for s in scales[max(0, i0 - 2):i0 + 3]:
        t, _ = _scaled(s)
        pad = 12
        xa, ya = max(0, 2 * loc[0] - pad), max(0, 2 * loc[1] - pad)
        xb = min(g.shape[1], 2 * loc[0] + t.shape[1] + pad)
        yb = min(g.shape[0], 2 * loc[1] + t.shape[0] + pad)
        mx, (lx, ly) = _match(g[ya:yb, xa:xb], s)
        if best is None or mx > best.score:
            best = Panel(float(x0 + xa + lx), float(y0 + ya + ly), float(s), mx)
    return best


def panel_score(frame: np.ndarray, panel: Panel, pad: int = 2) -> float:
    """Совпадение с эталоном на найденном месте (±pad px): есть ли панель."""
    t, mk = _scaled(panel.scale)
    th, tw = t.shape
    h, w = frame.shape[:2]
    x, y = round(panel.x), round(panel.y)
    xa, ya = max(0, x - pad), max(0, y - pad)
    xb, yb = min(w, x + tw + pad), min(h, y + th + pad)
    if xb - xa < tw or yb - ya < th:
        return -1.0
    r = cv2.matchTemplate(_gray(frame[ya:yb, xa:xb]), t, cv2.TM_CCOEFF_NORMED, mask=mk)
    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(r.max())


def slot_crop(frame: np.ndarray, panel: Panel, key: str, size: int = CROP) -> np.ndarray:
    """Вырез иконки слота size×size RGB без рамки."""
    x0, y0, x1, y1 = panel.box(key)
    dx, dy = (x1 - x0) * INSET, (y1 - y0) * INSET
    xa, ya, xb, yb = (round(x0 + dx), round(y0 + dy), round(x1 - dx), round(y1 - dy))
    c = frame[max(0, ya):yb, max(0, xa):xb]
    if c.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(np.ascontiguousarray(c), (size, size), interpolation=cv2.INTER_AREA)


def hp_fill(frame: np.ndarray, panel: Panel) -> float:
    """Доля зелёных пикселей полосы ХП (0 — пустая: мёртв или панель закрыта)."""
    x0, y0, x1, y1 = HP_BAR
    s = panel.scale
    bar = frame[round(panel.y + y0 * s):round(panel.y + y1 * s),
                round(panel.x + x0 * s):round(panel.x + x1 * s)].astype(np.int16)
    if bar.size == 0:
        return 0.0
    r, g, b = bar[..., 0], bar[..., 1], bar[..., 2]
    return float(((g > r + 40) & (g > b + 30) & (g > 90)).mean())


@dataclass(frozen=True)
class SlotFeat:
    v: float        # средняя яркость (0..255)
    s: float        # средняя насыщенность (0..1)
    navy: float     # доля синей заливки перезарядки
    white: float    # доля белых пикселей в середине (цифры)
    cross: float    # красный крестик запрета: доля красного на диагоналях минус вне их
    digits: float   # белое с тёмной обводкой в середине (цифры секунд), доля середины


def slot_features(crop: np.ndarray) -> SlotFeat:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hue, sat, val = hsv[..., 0] * (360 / 256), hsv[..., 1] / 255, hsv[..., 2]
    # Заливка перезарядки — ровная (перепад яркости у соседей 80–150), синие
    # пиксели самих иконок (Призрак, синяя Кара) — с рисунком (190–320).
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
            + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)))
    navy = ((hue > 195) & (hue < 235) & (sat > 0.45) & (val > 40) & (val < 150)
            & (grad < NAVY_FLAT))
    n = crop.shape[0]
    a, b = n // 4, n - n // 4
    white = (val > 200) & (sat < 0.25)
    # Крестик — чисто красные пиксели на обеих диагоналях в середине; у
    # Воспламенения красное разбросано по всей иконке, а не на диагоналях.
    red = ((hue < 12) | (hue > 348)) & (sat > 0.6) & (val > 110)
    yy, xx = _grid(n)
    diag = (np.abs(xx - yy) <= 2) | (np.abs(xx + yy - (n - 1)) <= 2)
    mid = (np.abs(xx - n / 2) < n * 0.35) & (np.abs(yy - n / 2) < n * 0.35)
    cross = float(red[diag & mid].mean() - red[~diag].mean())
    # Цифры: белые пиксели, рядом с которыми (±2 px) есть тёмные — обводка.
    bright = ((val > 190) & (sat < 0.3)).astype(np.uint8)
    dark = cv2.dilate((val < 70).astype(np.uint8), np.ones((5, 5), np.uint8))
    dig = (bright & dark)[n // 5:n - n // 5, n // 8:n - n // 8]
    return SlotFeat(float(val.mean()), float(sat.mean()), float(navy.mean()),
                    float(white[a:b, n // 8:n - n // 8].mean()), cross, float(dig.mean()))


def _grid(n: int):
    return np.mgrid[:n, :n]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная корреляция двух вырезов в сером."""
    x = _gray(a).ravel()
    y = _gray(b).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-6 else 0.0


# ---------------------------------------------------------------- решение

# Перезарядка, вид 1: синяя заливка сверх эталона (пока эталона нет — сверх
# нуля). Заливка — сектор «осталось», к концу перезарядки она сходит на нет.
# Ровных синих у перезарядки 0.34–0.78, у готовых синих иконок (голубой
# Призрак, синяя Кара) 0.10–0.23.
CD_NAVY = 0.30
# ...и иконка в целом тёмная: у готового Призрака средняя яркость 179, у
# синей Кары 135–168; у иконки с заливкой перезарядки 96–125.
CD_NAVY_V = 140.0
# Перезарядка, вид 2: иконка тёмная и на ней цифры с обводкой. Так выглядит
# конец длинной перезарядки (синего уже мало) и короткие (запрет повтора Кары).
CD_DARK = 0.70       # яркость ниже этой доли эталона
CD_DARK_ABS = 90.0   # или ниже этой яркости, пока эталона нет (готовые 125–155)
CD_DIGITS = 0.03     # у цифр 0.045–0.11; у готового Призрака белая фигура 0.18–0.21
CD_NAVY_DIG = 0.15   # заливки поменьше, но с цифрами и тёмная (v < CD_DIG_V) — тоже перезарядка
CD_DIG_V = 120.0     # у начала перезарядки Скачка 96–100, у готовой синей Кары 135–168
# Кандидат в эталон: не синий, без крестика, не мёртв, без тёмных цифр.
REF_NAVY = 0.25
REF_BRIGHT = 150.0   # или яркая иконка (синий Призрак): тоже кандидат
REF_DARK = 115.0     # тёмная иконка с цифрами — не кандидат (конец перезарядки 85–110)
# Кандидат не темнее эталона больше чем на 15% (без эталона — не темнее
# REF_MIN_V): иначе долгая перезарядка (Воспламенение, 3 готовых выреза на 2
# минуты) за несколько добавлений утягивала медиану к тёмной иконке.
REF_KEEP_V = 0.85
REF_MIN_V = 110.0    # готовые иконки D/F на роликах и своём HUD — 125–180
REF_CROSS = 0.10
REF_MIN = 8          # вырезов в эталоне, чтобы отвечать «готово»
REF_KEEP = 40        # сколько последних кандидатов держать
REF_EVERY = 2.0      # с: не чаще одного кандидата за столько секунд потока
# Готово: похоже на эталон.
READY_NCC = 0.80
READY_V = 0.80       # яркость не ниже доли яркости эталона
READY_S = 0.70       # насыщенность не ниже доли эталона (серая иконка — не готова)
READY_DIGITS = 0.02  # цифр не больше, чем у эталона, плюс столько
# Крестик запрета: у крестика 0.29–0.49 (кадры своей игры, ролики), у
# Воспламенения и прочих иконок ≤ 0.01.
BLOCK_CROSS = 0.20
# Мёртв: полоса ХП пуста (у живого с 3% ХП зелёного ещё ≈ 0.01).
DEAD_HP = 0.005

REASONS = {
    "hidden": "панели умений на кадре нет (закрыта или кадр не игровой)",
    "dead": "свой мёртв: полоса ХП пуста, иконка серая",
    "blocked": "заклинание под запретом: красный крестик (контроль или своё умение)",
    "no_ref": "эталон готовой иконки слота ещё не набран",
    "unsure": "иконка не похожа ни на готовую, ни на перезарядку",
}


@dataclass(frozen=True)
class SlotRead:
    slot: str
    state: str                  # ready / cooldown / blocked / dead / hidden / unsure
    value: bool | None
    reason: str | None
    feat: SlotFeat | None = None
    sim: float | None = None    # NCC с эталоном
    vrel: float | None = None   # яркость относительно эталона

    def to_dict(self) -> dict:
        return {"slot": self.slot, "state": self.state, "value": self.value,
                "reason": self.reason,
                "sim": None if self.sim is None else round(self.sim, 3),
                "vrel": None if self.vrel is None else round(self.vrel, 3)}


@dataclass
class SlotRef:
    """Эталон готовой иконки слота: медиана последних кандидатов."""
    crops: deque = field(default_factory=lambda: deque(maxlen=REF_KEEP))
    last_t: float | None = None
    _ref: np.ndarray | None = None
    _feat: SlotFeat | None = None

    def add(self, crop: np.ndarray, t: float) -> None:
        if self.last_t is not None and 0 <= t - self.last_t < REF_EVERY:
            return
        self.last_t = t
        self.crops.append(crop)
        self._ref = None

    @property
    def ready(self) -> bool:
        return len(self.crops) >= REF_MIN

    def ref(self) -> tuple[np.ndarray, SlotFeat]:
        if self._ref is None:
            self._ref = np.median(np.stack(self.crops), axis=0).astype(np.uint8)
            self._feat = slot_features(self._ref)
        return self._ref, self._feat

    def reset(self) -> None:
        self.crops.clear()
        self.last_t = None
        self._ref = None


def decide(slot: str, crop: np.ndarray, ref: SlotRef, present: bool, dead: bool,
           t: float, learn: bool = True) -> SlotRead:
    """Состояние слота по вырезу, эталону и признакам панели.

    Порядок: панели нет → hidden; крестик → blocked; перезарядка (синяя
    заливка или тёмная иконка с цифрами — видна и у мёртвого) → cooldown;
    мёртв → dead; похоже на эталон → ready; иначе unsure. «Готово» — только
    при эталоне и по всем признакам сразу."""
    if not present:
        return SlotRead(slot, "hidden", None, REASONS["hidden"])
    f = slot_features(crop)
    rf = ref.ref()[1] if ref.ready else None
    if f.cross >= BLOCK_CROSS:
        return SlotRead(slot, "blocked", None, REASONS["blocked"], f)
    if f.navy - (rf.navy if rf is not None else 0.0) >= CD_NAVY and f.v < CD_NAVY_V:
        return SlotRead(slot, "cooldown", False, None, f)
    dark = (f.v < CD_DARK * rf.v) if rf is not None else (f.v < CD_DARK_ABS)
    # Начало длинной перезарядки: трёхзначные цифры закрывают часть заливки.
    dark = dark or (f.navy - (rf.navy if rf is not None else 0.0) >= CD_NAVY_DIG
                    and f.v < CD_DIG_V)
    if dark and f.digits >= CD_DIGITS and not dead:
        return SlotRead(slot, "cooldown", False, None, f)
    if dead:
        return SlotRead(slot, "dead", None, REASONS["dead"], f)
    if (learn and (f.navy < REF_NAVY or f.v >= REF_BRIGHT) and f.cross < REF_CROSS
            and not (f.digits >= CD_DIGITS and f.v < REF_DARK)
            and f.v >= (REF_KEEP_V * rf.v if rf is not None else REF_MIN_V)):
        ref.add(crop, t)
    if not ref.ready:
        return SlotRead(slot, "unsure", None, REASONS["no_ref"], f)
    r, rf = ref.ref()
    sim = ncc(crop, r)
    vrel = f.v / max(rf.v, 1.0)
    srel = f.s / max(rf.s, 1e-3)
    if (sim >= READY_NCC and vrel >= READY_V and srel >= READY_S
            and f.navy - rf.navy < CD_NAVY / 2 and f.digits <= rf.digits + READY_DIGITS
            and rf.v >= REF_MIN_V):
        return SlotRead(slot, "ready", True, None, f, sim, vrel)
    return SlotRead(slot, "unsure", None,
                    f"{REASONS['unsure']} (сходство с эталоном {sim:.2f}, яркость {vrel:.2f})",
                    f, sim, vrel)


class SpellHud:
    """Чтение слотов D/F в сеансе: поиск панели, эталоны готовых иконок.

    `offer(frame)` — пока панель не найдена (или потеряна), ищет её: AGREE
    находок подряд в одном месте с совпадением ≥ FOUND_MIN. `read(frame, t)`
    — состояния слотов. `spells` — заклинания по слотам из API: сменились
    (улучшение Телепорта, Кары) — эталон слота сбрасывается."""

    def __init__(self):
        self.panel: Panel | None = None
        self.refs = {k: SlotRef() for k in SPELL_SLOTS}
        self._cand: list[Panel] = []
        self._miss = 0
        self.spells: tuple | None = None
        self.tried = 0

    def offer(self, frame: np.ndarray) -> Panel | None:
        """Искать панель, если её нет. Возвращает панель, когда найдена."""
        if self.panel is not None:
            return self.panel
        self.tried += 1
        p = locate_panel(frame)
        if p is None or p.score < FOUND_MIN:
            self._cand.clear()
            return None
        if self._cand and (abs(self._cand[-1].x - p.x) > AGREE_PX
                           or abs(self._cand[-1].y - p.y) > AGREE_PX
                           or abs(self._cand[-1].scale - p.scale) > 0.03):
            self._cand.clear()
        self._cand.append(p)
        if len(self._cand) >= AGREE:
            self.set_panel(self._cand[-1])
        return self.panel

    def set_panel(self, panel: Panel | None) -> None:
        self.panel = panel
        self._cand.clear()
        self._miss = 0
        for r in self.refs.values():
            r.reset()

    def set_spells(self, spells) -> None:
        """Заклинания по слотам (API) — сменилось — эталон слота заново."""
        spells = None if spells is None else tuple(spells)
        if self.spells is not None and spells is not None:
            for k, a, b in zip(SPELL_SLOTS, self.spells, spells):
                if a != b:
                    self.refs[k].reset()
        if spells is not None:
            self.spells = spells

    def read(self, frame: np.ndarray, t: float) -> dict[str, SlotRead]:
        if self.panel is None:
            return {k: SlotRead(k, "hidden", None, "панель умений не найдена") for k in SPELL_SLOTS}
        score = panel_score(frame, self.panel)
        present = score >= PRESENT_MIN
        dead = present and hp_fill(frame, self.panel) < DEAD_HP
        return {k: decide(k, slot_crop(frame, self.panel, k), self.refs[k], present, dead, t)
                for k in SPELL_SLOTS}
