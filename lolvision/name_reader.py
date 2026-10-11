"""Имя над плашкой → чемпион: закрытый словарь, ближайший образец.

Зачем. Над плашкой канал пишет английское имя чемпиона («Volibear»,
«Irelia») или вместо него состояние («CHANNELING», «STUNNED»). Словарь
закрыт — 173 имени Data Dragon и десяток состояний, — поэтому OCR общего
вида не нужен: как с уровнем в health_bars.py, вырез сравнивается с
образцами целиком.

Признак — «белизна»: минимум по каналам RGB, растянутый в 0..1 (текст
белый, обводка тёмная, фон сцены цветной). Сравнение — нормированная
корреляция со сдвигом по x ±4 px и по y ±2 px.

Образцы двух сортов:
* отрисованные шрифтом Windows (Constantia 12 px — из семи шрифтов лучше
  всех совпал с кадрами по разметке составом) — есть для каждого имени словаря, нужны для разметки
  сбора образцов и для чемпионов, которых в данных не было;
* настоящие вырезы — медиана автоматически размеченных вырезов чемпиона
  (разметка: состав матча по HUD + команда плашки, см. tools/train_names.py).
  Настоящий образец точнее отрисованного, и при наличии он главный.

Ограничение кандидатами: если состав известен, для плашки команды T
выбираем из пяти чемпионов T и состояний — ошибки «похожее имя чужого
чемпиона» исчезают. Известное ограничение: Neeko и другие копирующие облик
пишут над плашкой чужое имя (docs/champion-identity.md).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lolvision.identity import NAME_BOX

# Состояния над плашкой (английский клиент). Видели CHANNELING и STUNNED;
# остальные — из перечня эффектов контроля игры, могут не встречаться.
STATUSES = ("CHANNELING", "STUNNED", "ROOTED", "SILENCED", "TAUNTED", "FEARED", "CHARMED",
            "AIRBORNE", "ASLEEP", "SUPPRESSED", "POLYMORPHED", "BLINDED", "GROUNDED",
            "DISARMED", "NEARSIGHTED", "SLOWED", "KNOCKED UP", "DROWSY", "PULLED",
            "STASIS", "RECALLING")
STATUS = "status:"                  # префикс метки состояния
NONE = "none"                       # пусто или закрыто
CROP_H = NAME_BOX[3] - NAME_BOX[1]  # 19
CROP_W = NAME_BOX[2] - NAME_BOX[0]  # 147
DX, DY = 4, 2
WIN_H, WIN_W = CROP_H - 2 * DY, CROP_W - 2 * DX
FONT = Path("C:/Windows/Fonts/constan.ttf")
FONT_PX = 12                        # кегль, при котором ширина имён совпала с кадром
TRACK = 0.0                         # разрядка, px кадра
TEXT_CX = CROP_W / 2                # имя центрировано над плашкой
TEXT_Y = 4                          # верх заглавных в вырезе
# Состояние — заглавными, с разрядкой, ниже и правее имени (слева значок
# эффекта); подобрано по вырезам «STUNNED» (сходство 0.85).
STATUS_PX = 12
STATUS_DX = 10
STATUS_Y = 6
STATUS_TRACK = 0.5


def whiteness(crop: np.ndarray) -> np.ndarray:
    """Белизна выреза (H, W, 3) uint8 → (H, W) float32 в 0..1."""
    m = crop.min(2).astype(np.float32)
    return np.clip((m - 110.0) / 100.0, 0.0, 1.0)


def _norm(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return a / (np.sqrt((a * a).sum(axis=(-2, -1), keepdims=True)) + 1e-6)


def render(text: str, font: Path = FONT, px: int = FONT_PX,
           track: float = TRACK, dx: float = 0.0, top: float = TEXT_Y) -> np.ndarray:
    """Отрисованный образец: белизна (CROP_H, CROP_W), текст там, где он в кадре.
    track — добавка к промежутку между буквами в пикселях кадра (у канала
    разрядка заметная)."""
    big = 4                                           # рисуем крупно и сжимаем — мягкий край
    fb = ImageFont.truetype(str(font), px * big)
    img = Image.new("L", (CROP_W * big, CROP_H * big), 0)
    d = ImageDraw.Draw(img)
    adv = [d.textlength(ch, font=fb) + track * big for ch in text]
    width = sum(adv) - track * big
    _, y0, _, _ = d.textbbox((0, 0), "H", font=fb)
    x = (TEXT_CX + dx) * big - width / 2
    for ch, a in zip(text, adv):
        d.text((x, top * big - y0), ch, fill=255, font=fb)
        x += a
    return np.asarray(img.resize((CROP_W, CROP_H), Image.BILINEAR), np.float32) / 255.0


def windows(w: np.ndarray) -> np.ndarray:
    """Все сдвиги окна (WIN_H, WIN_W) по белизне выреза → (сдвиги, WIN_H·WIN_W), нормированы."""
    v = np.lib.stride_tricks.sliding_window_view(w, (WIN_H, WIN_W))
    v = v.reshape(-1, WIN_H, WIN_W)
    return _norm(v).reshape(len(v), -1)


def center(w: np.ndarray) -> np.ndarray:
    """Окно без сдвига — так хранятся образцы."""
    return w[DY:DY + WIN_H, DX:DX + WIN_W]


class NameReader:
    """Банк образцов: labels[i] — Id чемпиона, «status:…» или «none»."""

    def __init__(self, labels: list[str], templates: np.ndarray, kinds: list[str] | None = None):
        self.labels = list(labels)
        self.kinds = list(kinds or ["real"] * len(labels))   # real / rendered
        t = np.asarray(templates, np.float32).reshape(len(labels), WIN_H, WIN_W)
        self.bank = _norm(t).reshape(len(labels), -1).T          # (признаки, образцы)
        self.uniq = list(dict.fromkeys(self.labels))              # метки без повторов
        u = {lab: k for k, lab in enumerate(self.uniq)}
        self.group = np.array([u[lab] for lab in self.labels])    # образец → метка

    @classmethod
    def rendered(cls, names: dict[str, str], font: Path = FONT, px: int = FONT_PX,
                 track: float = TRACK) -> NameReader:
        """Только отрисованные образцы: names — Id → отображаемое имя."""
        labels = list(names) + [STATUS + s for s in STATUSES]
        temps = [center(render(t, font, px, track)) for t in names.values()]
        temps += [center(render(t, font, STATUS_PX, STATUS_TRACK, STATUS_DX, STATUS_Y))
                  for t in STATUSES]
        return cls(labels, np.stack(temps), ["rendered"] * len(labels))

    @classmethod
    def load(cls, path: Path) -> NameReader:
        z = np.load(path)
        return cls([str(s) for s in z["labels"]], z["templates"], [str(s) for s in z["kinds"]])

    def save(self, path: Path) -> None:
        t = (self.bank.T.reshape(-1, WIN_H, WIN_W)).astype(np.float16)
        np.savez(path, labels=np.array(self.labels), templates=t, kinds=np.array(self.kinds))

    def label_sims(self, crops: np.ndarray, batch: int = 64) -> np.ndarray:
        """Сходство вырезов (N, H, W, 3) с каждой меткой: (N, len(uniq)), где
        uniq = self.uniq — метки без повторов; по образцам метки и сдвигам — максимум."""
        out = np.empty((len(crops), len(self.uniq)), np.float32)
        for i in range(0, len(crops), batch):
            part = crops[i:i + batch]
            w = np.stack([windows(whiteness(c)) for c in part])          # (n, сдвиги, D)
            n, k, d = w.shape
            sims = (w.reshape(n * k, d) @ self.bank).reshape(n, k, -1).max(1)   # (n, образцы)
            lab = np.full((len(part), len(self.uniq)), -2.0, np.float32)
            np.maximum.at(lab.T, self.group, sims.T)
            out[i:i + len(part)] = lab
        return out

    def scores(self, crop: np.ndarray) -> dict[str, float]:
        """Лучшее сходство с каждой меткой (по всем её образцам и сдвигам)."""
        return dict(zip(self.uniq, self.label_sims(crop[None])[0].tolist()))

    def pick(self, sims: np.ndarray, candidates: list[str] | None = None,
             statuses: bool = True) -> tuple[str, float, float]:
        """Решение по строке label_sims: (метка, сходство, отрыв от второй).
        candidates — Id допустимых чемпионов (например, пять из команды
        плашки); состояния добавляются, если statuses."""
        uniq = self.uniq
        if candidates is None:
            idx = np.arange(len(uniq))
        else:
            keep = set(candidates)
            idx = np.array([k for k, lab in enumerate(uniq)
                            if lab in keep or (statuses and lab.startswith(STATUS)) or lab == NONE],
                           int)
        if len(idx) == 0:
            return NONE, float("nan"), 0.0
        sub = sims[idx]
        order = np.argsort(-sub)
        s1 = float(sub[order[0]])
        s2 = float(sub[order[1]]) if len(order) > 1 else -1.0
        return uniq[idx[order[0]]], s1, s1 - s2

    def read(self, crop: np.ndarray, candidates: list[str] | None = None,
             statuses: bool = True) -> tuple[str, float, float]:
        """Один вырез: (метка, сходство, отрыв от второй)."""
        return self.pick(self.label_sims(crop[None])[0], candidates, statuses)


# --- настоящие образцы из вырезов -------------------------------------

ADAPT_SIM = 0.5         # вырез идёт в медиану: сходство с образцом не ниже
ADAPT_MARGIN = 0.03     # и отрыв от второго кандидата
ADAPT_COUNT = 5         # медиана не меньше чем из пяти вырезов
DUP_SIM = 0.85          # две медианы одной группы совпали со сдвигом — одно слово


def as_crop(win: np.ndarray) -> np.ndarray:
    """Окно белизны (WIN_H, WIN_W) → вырез RGB полного размера с той же белизной."""
    img = np.zeros((CROP_H, CROP_W), np.float32)
    w = (win - win.min()) / (np.ptp(win) + 1e-6)
    img[DY:DY + WIN_H, DX:DX + WIN_W] = w
    return np.repeat((110 + 100 * img).clip(0, 255).astype(np.uint8)[..., None], 3, 2)


def build_medians(groups: dict[tuple[str, str], list[np.ndarray]],
                  min_count: int = ADAPT_COUNT) -> tuple[list[tuple[str, str]], list]:
    """Медианы окон белизны по ключам (группа, метка).

    Чистка: две медианы одной группы (ролика), совпавшие со сдвигом выше
    DUP_SIM, — одно и то же слово; остаётся метка, у которой вырезов больше.
    Так бывает, когда младшая метка «ворует» вырезы старшей: Camille
    (ведомая каналом, сотни плашек) и Ezreal в одной команде, отрисованный
    образец путает эти имена, и медиана «Ezreal» — это слово Camille.
    Отрисованные образцы здесь не судьи: на чистой медиане они дают
    0.5–0.6 и путают те же пары. Группа «*» (состояния) не чистится."""
    keys, temps, counts = [], [], []
    for key, ws in sorted(groups.items()):
        if len(ws) >= min_count:
            keys.append(key)
            temps.append(np.median(np.stack(ws), 0))
            counts.append(len(ws))
    drop = set()
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j or keys[i][0] != keys[j][0] or keys[i][0] == "*" or counts[j] <= counts[i]:
                continue
            if NameReader([keys[j][1]], temps[j][None]).read(as_crop(temps[i]))[1] > DUP_SIM:
                drop.add(i)
    keep = [i for i in range(len(keys)) if i not in drop]
    return [keys[i] for i in keep], [temps[i] for i in keep]


def adapted(reader: NameReader, crops: np.ndarray, cands: list[list[str]],
            iters: int = 2) -> NameReader:
    """Банк, подстроенный под один матч: медианы уверенно прочитанных вырезов
    этого матча (среди пяти кандидатов команды) добавляются как настоящие
    образцы. Нужен для чемпионов, у которых в банке только отрисованный
    образец: на нём сходство 0.5–0.6, и имя часто не проходит порог, хотя
    видно. Верные метки не нужны — только состав и команда плашки."""
    base = reader
    for _ in range(iters):
        sims = reader.label_sims(crops)
        groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for c, s, cand in zip(crops, sims, cands):
            lab, sim, m = reader.pick(s, cand)
            if sim >= ADAPT_SIM and m >= ADAPT_MARGIN and not lab.startswith(STATUS):
                groups[("match", lab)].append(center(whiteness(c)))
        keys, temps = build_medians(groups)
        if not keys:
            return base
        old = base.bank.T.reshape(-1, WIN_H, WIN_W)
        reader = NameReader(base.labels + [k[1] for k in keys],
                            np.concatenate([old, np.stack(temps)]),
                            base.kinds + ["match"] * len(keys))
    return reader
