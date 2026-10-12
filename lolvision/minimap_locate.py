"""Где в кадре миникарта: прямоугольник обрезки под масштаб миникарты ролика.

Зачем. Истина камеры снимается с рамки на миникарте (minimap_camera.py), а
миникарта вырезается прямоугольником раскладки (`labelling.minimap`),
замеренным на ролике со стандартным масштабом. Масштаб миникарты игрок
меняет в настройках, и у такого стримера рамка пересчитывается в чужую
сетку: истина съезжает к левому верхнему углу. DQpQYfrrbHk (RamyAura,
миникарта крупнее на 5%): ответ модели ≈ 0.95·истина + 0.053, медиана
ошибки 0.057; с истиной по найденному прямоугольнику — 0.003 (#785).

Как ищется. Миникарта привязана к правому нижнему углу и меняет только
масштаб. Эталон — миникарта 256×256 в обрезке раскладки, медиана чистых
фонов всех обучающих роликов (dataset/layouts/minimap_reference.png,
tools/minimap_reference.py). Сравниваются не яркости, а модуль градиента
после размытия: у игрока туман войны и светлые базы, у наблюдателя карта
открыта, и по яркости половина роликов совпадала с эталоном на 0.45, а по
градиентам — все на 0.80–0.85 (замер по фонам 16 обучающих роликов: у
всех масштаб 1.00 и сдвиг 0). Перебор масштаба сначала грубо на
половинном разрешении, потом точно в окрестности.

Ответ — прямоугольник в долях кадра, который для этого ролика значит то
же, что `labelling.minimap` для стандартного: та же точка карты ложится в
ту же клетку сетки 256. Поэтому всё дальше (рамка, DRAW_X0/Y0, иконки
чемпионов) работает без правок.

Другие карты (#803). Эталон — содержимое Ущелья, и на Воющей бездне (ARAM,
map 12) поиск по нему не находит ничего: на 15 кадрах ARAM лучшее
совпадение 0.15–0.16 при пороге 0.40. А рамка интерфейса миникарты (золотая
линия, тёмная окантовка, уголок слева сверху) одна на все карты, и масштаб
миникарты — настройка клиента, одна на все карты. Поэтому, когда эталон не
нашёл, миникарта ищется по рамке (`locate_frame`):

* сравнивается полоса рамки — верхние и левые FRAME_BAND клеток сетки 256
  эталона, а не градиенты, а сами яркости (после лёгкого размытия): у
  градиентов любой прямой край давал 0.6–0.7, у яркостей — тонкая светлая
  линия на тёмном, как у рамки;
* миникарта привязана к правому нижнему углу, и её левый верхний угол при
  масштабе s лежит на прямой (ANCHOR, `scale_box`): по трём масштабам
  0.88 / 1.00 / 1.205 ошибка прямой ≤ 0.7 px кадра 1080p. Перебирается
  только масштаб, место — следствие (±1 px), поэтому полоса без правого и
  нижнего края всё равно задаёт масштаб.

Замер (полоса 10 клеток, совпадение у найденного масштаба): кадры с
нынешней рамкой — свой экран на Ущелье (8 кадров, ×1.205) и на ARAM
(15 кадров), наблюдатель j8_-kWWh5qM (14, ×1.00) — 0.89–0.99, медиана 0.97,
прямоугольник в пределах 1.4 px от найденного эталоном; кадр ARAM, где
поверх игры терминал, — 0.52; ролики со старой рамкой без золотой линии
(mXIVjtxn5p0, свой 720p 2025 г.) — не выше 0.62. Порог FRAME_MIN = 0.80.
На 1280×720, 1600×900 и 2560×1440 (тот же кадр ARAM) — тот же масштаб.

Последний найденный эталоном прямоугольник для размера экрана хранится вне
репозитория (`save_fit` / `load_fit`, live для своего экрана): на другой
карте он проверяется по рамке первым и, если рамка на месте, берётся как
есть — точнее прямой. Масштаб можно задать и явно (`manual`): прямоугольник
по прямой, без поиска.

Пример:
    loc = Locator((w, h))
    for fr in frames:
        if loc.offer(fr):            # True, когда прямоугольник найден
            break
    box = loc.box or layout["labelling"]["minimap"]
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lolvision.paths import LAYOUTS

REFERENCE = LAYOUTS / "minimap_reference.png"
# Обрезка раскладки, в которой снят эталон (spectator-volibear-challenger).
STD_BOX = (0.857, 0.74, 0.993, 0.984)
# Где искать: правый нижний угол кадра. Вмещает миникарту до SCALES[-1].
SEARCH = (0.70, 0.50)
# Масштаб миникарты относительно стандартной. У стримов набора not_game
# встречается от 1.00 до 1.50 (две трети — не 1.00), 1.50 — у двух каналов.
SCALES = (0.70, 1.80)
# Сравнивается внутренняя часть эталона: края обрезки — тёмный интерфейс,
# он у разных масштабов ведёт себя не как карта.
INNER = (24, 232)
SIGMA = 1.5             # размытие перед градиентом, в клетках сетки 256
COARSE_STEP, FINE_STEP = 0.02, 0.005
COARSE_DOWN = 4         # грубый проход — на четверти разрешения
# Найденное в пределах SNAP_PX от прямоугольника раскладки — это раскладка:
# поиск на стандартной миникарте отвечает на пиксель мимо (масштаб 0.995),
# а пиксель обрезки — это ~58 ед. истины. Стандартные ролики не меняются.
SNAP_PX = 2.5
# Совпадение по градиентам (540p, кадры набора not_game, 20 стримов):
# игра — медиана 0.53, не игра (клиент, загрузка, другие игры, вебкамера
# во весь кадр) — не выше 0.22. Миникарта под вебкамерой — тоже ниже порога.
MIN_SCORE = 0.40
AGREE = 3               # столько уверенных кадров подряд с одним ответом
AGREE_PX = 2.0          # «один ответ»: углы прямоугольника ближе, px кадра
# Сколько попыток (кадров «игра») поиск миникарты делает, прежде чем сдаться
# и остаться на прямоугольнике раскладки: ~40 мс каждая, только до находки.
# Один на live (lolstream/live.py) и сборку датасета (tools/build_coords.py).
LOCATE_MAX = 300

# Поиск по рамке (см. докстринг модуля). Полоса рамки — верхние и левые
# клетки сетки 256 эталона: золотая линия и тёмная окантовка с уголком.
FRAME_BAND = 10
FRAME_BLUR = 0.7        # размытие перед сравнением, в клетках сетки 256
# Совпадение полосы рамки: нынешняя рамка 0.89–0.99, старая рамка и не
# игра — не выше 0.62 (замер в докстринге модуля).
FRAME_MIN = 0.80
FRAME_COARSE, FRAME_FINE = 0.01, 0.0025
FRAME_SLACK = (2, 1)    # px вокруг места по прямой: грубый и точный проход
FRAME_SUB = 4           # грубый проход — по каждой 4-й точке полосы
# Левый верхний угол миникарты при масштабе s: STD_BOX[:2] − ANCHOR·(s − 1),
# доли кадра. МНК по трём замерам эталоном: ×0.88 (mXIVjtxn5p0), ×1.00
# (раскладка), ×1.205 (свой экран); ошибка ≤ 0.7 px на 1080p.
ANCHOR = (0.1455, 0.268)
# Где лежит последний найденный эталоном прямоугольник своего экрана: вне
# репозитория, по одной записи на размер кадра.
FIT_FILE = "minimap_fit.json"


@dataclass
class Fit:
    box: tuple[float, float, float, float]   # доли кадра, как labelling.minimap
    scale: float                             # 1.0 — стандартная миникарта
    score: float                             # совпадение: эталона (map) или рамки
    # Как найдена: map — по эталону Ущелья, frame — по рамке, saved — сохранённый
    # прямоугольник, подтверждённый рамкой, manual — масштаб задан явно.
    how: str = "map"


def scale_box(scale: float) -> tuple[float, float, float, float]:
    """Прямоугольник миникарты масштаба `scale` у правого нижнего угла, доли
    кадра (прямая ANCHOR; 1.0 — ровно STD_BOX)."""
    if abs(scale - 1.0) < 1e-9:
        return STD_BOX
    d = scale - 1.0
    x0, y0 = STD_BOX[0] - ANCHOR[0] * d, STD_BOX[1] - ANCHOR[1] * d
    return (round(x0, 5), round(y0, 5),
            round(x0 + (STD_BOX[2] - STD_BOX[0]) * scale, 5),
            round(y0 + (STD_BOX[3] - STD_BOX[1]) * scale, 5))


def fit_file() -> Path:
    """Файл сохранённых прямоугольников: %LOCALAPPDATA%/lolstream (или
    ~/.cache/lolstream) — не data/, runs/, dataset/."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "lolstream" / FIT_FILE


def _size_key(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def load_fit(path: Path, size: tuple[int, int]) -> Fit | None:
    """Сохранённый прямоугольник для кадра `size` или None (нет, файл битый)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))[_size_key(size)]
        box = tuple(float(v) for v in d["box"])
        if len(box) != 4:
            return None
        return Fit(box, float(d["scale"]), float(d["score"]), "saved")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_fit(path: Path, size: tuple[int, int], fit: Fit) -> None:
    """Запомнить прямоугольник для кадра `size` (остальные размеры — как были)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[_size_key(size)] = {"box": list(fit.box), "scale": fit.scale, "score": fit.score,
                             "how": fit.how, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _grad(gray: np.ndarray, sigma: float) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def search_region(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Угол кадра, в котором ищется миникарта, в пикселях (x0, y0, x1, y1)."""
    w, h = size
    return int(SEARCH[0] * w), int(SEARCH[1] * h), w, h


def box_px(box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Доли кадра → пиксели, округление как у мест, где режется миникарта."""
    w, h = size
    return int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)


class Locator:
    """Ищет миникарту по кадрам ролика одного размера, пока не найдёт.

    Один кадр может обмануть (миникарту закрыл оверлей, на экране меню),
    поэтому прямоугольник принимается, когда AGREE уверенных кадров подряд
    дали один и тот же ответ.

    Сначала эталон Ущелья; не нашёл (совпадение ниже MIN_SCORE) — рамка
    (`by_frame`): сохранённый прямоугольник `saved`, если рамка на его
    месте, иначе перебор масштаба по рамке. `manual` — масштаб задан явно:
    прямоугольник по прямой ANCHOR, принимается с первого кадра, без поиска.
    """

    def __init__(self, size: tuple[int, int], reference: np.ndarray | None = None, *,
                 saved: Fit | None = None, manual: float | None = None,
                 by_frame: bool = True):
        self.size = size
        w, h = size
        ref = reference if reference is not None else cv2.imread(str(REFERENCE))[:, :, ::-1]
        self._ref = _gray(ref)
        self.region = search_region(size)
        # Пиксели кадра на клетку сетки 256 у стандартной миникарты.
        self._cell = ((STD_BOX[2] - STD_BOX[0]) * w / 256, (STD_BOX[3] - STD_BOX[1]) * h / 256)
        self._templates: dict[tuple[float, int], np.ndarray] = {}
        self._bands: dict[tuple[float, int], tuple] = {}
        self._ref_soft = cv2.GaussianBlur(self._ref, (0, 0), FRAME_BLUR)
        self.saved = saved
        self.manual = manual
        self.by_frame = by_frame
        self.box: tuple[float, float, float, float] | None = None
        self.fit: Fit | None = None
        self.tried = 0
        self._run: list[Fit] = []

    def _template(self, scale: float, down: int) -> np.ndarray:
        key = (round(scale, 4), down)
        t = self._templates.get(key)
        if t is None:
            a, b = INNER
            sx, sy = (scale * c / down for c in self._cell)
            g = _grad(self._ref, SIGMA)[a:b, a:b]
            t = cv2.resize(g, (max(1, round((b - a) * sx)), max(1, round((b - a) * sy))),
                           interpolation=cv2.INTER_AREA)
            self._templates[key] = t
        return t

    def _best(self, feat: np.ndarray, scales, down: int,
              window: tuple[int, int, int, int] | None = None):
        """Лучший масштаб и положение левого верхнего угла внутреннего окна."""
        best = (-2.0, 0.0, 0, 0)
        ox = oy = 0
        if window is not None:
            ox, oy = window[0], window[1]
            feat = feat[window[1]:window[3], window[0]:window[2]]
        for s in scales:
            t = self._template(s, down)
            if t.shape[0] >= feat.shape[0] or t.shape[1] >= feat.shape[1]:
                continue
            r = cv2.matchTemplate(feat, t, cv2.TM_CCOEFF_NORMED)
            _, v, _, (x, y) = cv2.minMaxLoc(r)
            if v > best[0]:
                best = (float(v), float(s), x + ox, y + oy)
        return best

    def locate(self, region: np.ndarray) -> Fit | None:
        """Одна попытка по углу кадра `region` (RGB, вырезан search_region):
        эталон Ущелья, а если он не нашёл — рамка (см. класс)."""
        if self.manual is not None:
            return self._manual()
        fit = self.locate_map(region)
        if (fit is not None and fit.score >= MIN_SCORE) or not self.by_frame:
            return fit
        alt = self.locate_frame(region)
        return alt if alt is not None else fit

    def _manual(self) -> Fit:
        return Fit(scale_box(self.manual), float(self.manual), 1.0, "manual")

    def locate_map(self, region: np.ndarray) -> Fit | None:
        """По эталону Ущелья (градиенты содержимого карты)."""
        cw, ch = self._cell
        feat = _grad(_gray(region), SIGMA * cw)
        d = COARSE_DOWN
        small = cv2.resize(feat, (feat.shape[1] // d, feat.shape[0] // d),
                           interpolation=cv2.INTER_AREA)
        coarse = np.arange(SCALES[0], SCALES[1] + 1e-9, COARSE_STEP)
        v, s, x, y = self._best(small, coarse, d)
        if v < -1:
            return None
        fine = np.arange(s - COARSE_STEP, s + COARSE_STEP + 1e-9, FINE_STEP)
        pad = 2 * d
        big = self._template(fine[-1], 1).shape
        win = (max(0, d * x - pad), max(0, d * y - pad),
               min(feat.shape[1], d * x + pad + big[1]),
               min(feat.shape[0], d * y + pad + big[0]))
        v, s, x, y = self._best(feat, fine, 1, win)
        if v < -1:
            return None
        a = INNER[0]
        w, h = self.size
        x0 = self.region[0] + x - a * s * cw
        y0 = self.region[1] + y - a * s * ch
        box = (x0 / w, y0 / h, (x0 + 256 * s * cw) / w, (y0 + 256 * s * ch) / h)
        # Миникарта целиком в углу поиска и в кадре: иначе это не она, а
        # совпадение на краю, и обрезка по нему вышла бы за кадр.
        if x0 < self.region[0] or y0 < self.region[1] or box[2] > 1.0 or box[3] > 1.0:
            return None
        if corner_px(box, STD_BOX, self.size) <= SNAP_PX:
            return Fit(STD_BOX, 1.0, round(v, 3))
        return Fit(tuple(round(b, 5) for b in box), round(s, 4), round(v, 3))

    # ---------- по рамке ----------

    def _band(self, scale: float, sub: int = 1):
        """Полоса рамки при масштабе `scale`: строки, столбцы (px от левого
        верха миникарты), яркости эталона (среднее 0, норма 1) и наибольшие
        строка и столбец. `sub` — каждая sub-я точка полосы (грубый проход)."""
        key = (round(scale, 4), sub)
        b = self._bands.get(key)
        if b is None:
            cw, ch = self._cell
            size = (max(1, round(256 * scale * cw)), max(1, round(256 * scale * ch)))
            t = cv2.resize(self._ref_soft, size, interpolation=cv2.INTER_AREA)
            bx, by = round(FRAME_BAND * scale * cw), round(FRAME_BAND * scale * ch)
            m = np.zeros(t.shape, bool)
            m[:by, :] = True
            m[:, :bx] = True
            ys, xs = np.nonzero(m)
            ys, xs = ys[::sub], xs[::sub]
            v = t[ys, xs].astype(np.float64)
            v -= v.mean()
            v /= max(np.linalg.norm(v), 1e-9)
            b = self._bands[key] = (ys, xs, v.astype(np.float32), int(ys.max()), int(xs.max()))
        return b

    def _frame_score(self, feat: np.ndarray, scale: float, x0: int, y0: int,
                     slack: int, sub: int = 1) -> tuple[float, int, int]:
        """Лучшее совпадение полосы рамки масштаба `scale` с левым верхом в
        (x0, y0) ± slack px (координаты угла поиска) → (совпадение, dx, dy)."""
        ys, xs, v, ymax, xmax = self._band(scale, sub)
        d = np.arange(-slack, slack + 1)
        dy, dx = (a.ravel() for a in np.meshgrid(d, d, indexing="ij"))
        ok = ((x0 + dx >= 0) & (y0 + dy >= 0) & (x0 + dx + xmax < feat.shape[1])
              & (y0 + dy + ymax < feat.shape[0]))
        if not ok.any():
            return -2.0, 0, 0
        dx, dy = dx[ok], dy[ok]
        f = feat[(y0 + dy)[:, None] + ys, (x0 + dx)[:, None] + xs]       # (смещения, n)
        n = len(v)
        den = np.sqrt(np.maximum((f * f).sum(1) - f.sum(1) ** 2 / n, 1e-9))
        r = (f @ v) / den
        k = int(np.argmax(r))
        return float(r[k]), int(dx[k]), int(dy[k])

    def _corner(self, box) -> tuple[int, int]:
        """Левый верх прямоугольника `box` (доли кадра) в px угла поиска."""
        w, h = self.size
        return round(box[0] * w) - self.region[0], round(box[1] * h) - self.region[1]

    def locate_frame(self, region: np.ndarray) -> Fit | None:
        """По рамке миникарты, не по содержимому карты: годится для любой
        карты. None — рамки нет (совпадение ниже FRAME_MIN)."""
        feat = cv2.GaussianBlur(_gray(region), (0, 0), FRAME_BLUR * self._cell[0])
        w, h = self.size
        if self.saved is not None:
            x0, y0 = self._corner(self.saved.box)
            v, _, _ = self._frame_score(feat, self.saved.scale, x0, y0, FRAME_SLACK[1])
            if v >= FRAME_MIN:
                return Fit(self.saved.box, self.saved.scale, round(v, 3), "saved")

        def scan(scales, slack, sub):
            best = (-2.0, 1.0, 0, 0)
            for s in scales:
                x0, y0 = self._corner(scale_box(s))
                v, dx, dy = self._frame_score(feat, float(s), x0, y0, slack, sub)
                if v > best[0]:
                    best = (v, float(s), dx, dy)
            return best

        v, s, _, _ = scan(np.arange(SCALES[0], SCALES[1] + 1e-9, FRAME_COARSE),
                       FRAME_SLACK[0], FRAME_SUB)
        if v < FRAME_MIN - 0.1:
            return None
        v, s, dx, dy = scan(np.arange(s - FRAME_COARSE, s + FRAME_COARSE + 1e-9, FRAME_FINE),
                            FRAME_SLACK[1], 1)
        if v < FRAME_MIN:
            return None
        s = round(s, 4)
        b = scale_box(s)
        box = (b[0] + dx / w, b[1] + dy / h, b[2] + dx / w, b[3] + dy / h)
        if box[2] > 1.0 or box[3] > 1.0:
            return None
        if corner_px(box, STD_BOX, self.size) <= SNAP_PX:
            return Fit(STD_BOX, 1.0, round(v, 3), "frame")
        return Fit(tuple(round(c, 5) for c in box), s, round(v, 3), "frame")

    def offer(self, frame: np.ndarray) -> bool:
        """Кадр целиком (RGB). True — прямоугольник найден, он в self.box."""
        if self.box is not None:
            return True
        x0, y0, x1, y1 = self.region
        return self.offer_region(frame[y0:y1, x0:x1])

    def offer_region(self, region: np.ndarray) -> bool:
        if self.box is not None:
            return True
        self.tried += 1
        if self.manual is not None:
            # Масштаб задан явно: искать нечего, согласия кадров не нужно.
            self.fit = self._manual()
            self.box = self.fit.box
            return True
        fit = self.locate(region)
        if fit is None or fit.score < MIN_SCORE:
            self._run.clear()
            return False
        w, h = self.size
        if self._run and corner_px(self._run[-1].box, fit.box, (w, h)) > AGREE_PX:
            self._run.clear()
        self._run.append(fit)
        if len(self._run) < AGREE:
            return False
        self.fit = max(self._run, key=lambda f: f.score)
        self.box = self.fit.box
        return True


def corner_px(a, b, size) -> float:
    w, h = size
    return max(abs(a[0] - b[0]) * w, abs(a[1] - b[1]) * h,
               abs(a[2] - b[2]) * w, abs(a[3] - b[3]) * h)

