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

Пример:
    loc = Locator((w, h))
    for fr in frames:
        if loc.offer(fr):            # True, когда прямоугольник найден
            break
    box = loc.box or layout["labelling"]["minimap"]
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Fit:
    box: tuple[float, float, float, float]   # доли кадра, как labelling.minimap
    scale: float                             # 1.0 — стандартная миникарта
    score: float                             # корреляция градиентов


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
    """

    def __init__(self, size: tuple[int, int], reference: np.ndarray | None = None):
        self.size = size
        w, h = size
        ref = reference if reference is not None else cv2.imread(str(REFERENCE))[:, :, ::-1]
        self._ref = _gray(ref)
        self.region = search_region(size)
        # Пиксели кадра на клетку сетки 256 у стандартной миникарты.
        self._cell = ((STD_BOX[2] - STD_BOX[0]) * w / 256, (STD_BOX[3] - STD_BOX[1]) * h / 256)
        self._templates: dict[tuple[float, int], np.ndarray] = {}
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
        """Одна попытка по углу кадра `region` (RGB, вырезан search_region)."""
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
        if _corner_px(box, STD_BOX, self.size) <= SNAP_PX:
            return Fit(STD_BOX, 1.0, round(v, 3))
        return Fit(tuple(round(b, 5) for b in box), round(s, 4), round(v, 3))

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
        fit = self.locate(region)
        if fit is None or fit.score < MIN_SCORE:
            self._run.clear()
            return False
        w, h = self.size
        if self._run and _corner_px(self._run[-1].box, fit.box, (w, h)) > AGREE_PX:
            self._run.clear()
        self._run.append(fit)
        if len(self._run) < AGREE:
            return False
        self.fit = max(self._run, key=lambda f: f.score)
        self.box = self.fit.box
        return True


def _corner_px(a, b, size) -> float:
    w, h = size
    return max(abs(a[0] - b[0]) * w, abs(a[1] - b[1]) * h,
               abs(a[2] - b[2]) * w, abs(a[3] - b[3]) * h)

