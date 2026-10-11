"""Чемпионы на миникарте по кадру: кто где, команда, видим ли.

Поверх детектора и классификатора иконок (lolvision/minimap_icons.py):

1. Состав матча — 10 чемпионов, по 5 на команду. Классификатор знает всех
   чемпионов Data Dragon; состав набирается голосованием по уверенным
   детекциям за ролик (или задаётся снаружи — модуль состава, заголовок).
2. Кто есть кто в кадре — назначение детекций чемпионам состава
   (венгерский алгоритм): каждый чемпион в кадре не больше одного раза,
   детекция может остаться «лишней» (пинг, призрак, ложная находка).
3. Положение — из сетки миникарты 256 в единицы карты.

Миникарта говорит только «видим / не видим»: исчезновение — это и туман,
и невидимость (Твич, Ивлин, Акали), и Юми на союзнике, и проглатывание
Таам Кенчем, и царство Мордекайзера, и стазис. Жив/мёртв — только по HUD
(lolvision/hud_panels.py).

Перевод в единицы карты (замер Replay API, docs/replay-api.md): поле
0…14 800 по x — экранные x 1653.96…1908.42 при 1920×1080; по y верх
миникарты — 812.06, низ — 1065.97 (рамка камеры по y сдвинута на 3.5 px
относительно точки земли, край карты — по границе отрисовки
minimap_camera.DRAW_Y0). Ось y карты смотрит вверх (синяя база внизу
слева), как у позиций в игре. Что центр иконки совпадает с позицией
чемпиона — допущение, эталона позиций чемпионов нет.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from lolvision.minimap_icons import MM, TEAMS

MAP_UNITS = 14800.0
# Миникарта на экране 1920×1080 (пиксели), поле 0…MAP_UNITS.
MAP_X0, MAP_X1 = 1653.96, 1908.42
MAP_Y_TOP, MAP_Y_BOTTOM = 812.06, 1065.97
REF_W, REF_H = 1920, 1080
MIN_SCORE_VOTE = 0.6      # детекции для голосования за состав
NEW_COST = 4.0            # цена «детекция не из состава», в единицах −log p


def crop_box(layout: dict, w: int = REF_W, h: int = REF_H) -> tuple[int, int, int, int]:
    """Прямоугольник миникарты в кадре так же, как его режет minimap_camera.minimap."""
    x0, y0, x1, y1 = layout["labelling"]["minimap"]
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def grid_to_map(u: float, v: float, layout: dict, w: int = REF_W,
                h: int = REF_H) -> tuple[float, float]:
    """Точка сетки 256 (пиксель i занимает [i, i+1)) → единицы карты (x, y)."""
    bx0, by0, bx1, by1 = crop_box(layout, w, h)
    fx = (bx0 + u * (bx1 - bx0) / MM) * REF_W / w     # в пикселях кадра 1920×1080
    fy = (by0 + v * (by1 - by0) / MM) * REF_H / h
    x = (fx - MAP_X0) / (MAP_X1 - MAP_X0) * MAP_UNITS
    y = (MAP_Y_BOTTOM - fy) / (MAP_Y_BOTTOM - MAP_Y_TOP) * MAP_UNITS
    return x, y


def grid_px_to_units(layout: dict, w: int = REF_W, h: int = REF_H) -> float:
    """Единиц карты на пиксель сетки 256 (среднее по осям)."""
    bx0, by0, bx1, by1 = crop_box(layout, w, h)
    sx = (bx1 - bx0) / MM * REF_W / w * MAP_UNITS / (MAP_X1 - MAP_X0)
    sy = (by1 - by0) / MM * REF_H / h * MAP_UNITS / (MAP_Y_BOTTOM - MAP_Y_TOP)
    return (sx + sy) / 2


class Composition:
    """Голосование за состав по детекциям ролика: копит вероятности."""

    def __init__(self, n_champ: int):
        self.votes = np.zeros((2, n_champ))
        self.n = 0

    def add(self, dets: list[dict]) -> None:
        for d in dets:
            if d["score"] >= MIN_SCORE_VOTE:
                self.votes[d["team"]] += np.exp(d["logp"])
                self.n += 1

    def teams(self, k: int = 5) -> list[list[int]]:
        """По k чемпионов на команду; чемпион — только в одной команде
        (берётся там, где за него больше голосов)."""
        v = self.votes.copy()
        out: list[list[int]] = [[], []]
        while min(len(out[0]), len(out[1])) < k and v.max() > 0:
            t, c = np.unravel_index(int(v.argmax()), v.shape)
            if len(out[t]) < k:
                out[t].append(int(c))
                v[:, c] = 0
            else:
                v[t] = 0
        return out


def assign(dets: list[dict], teams: list[list[int]], new_cost: float = NEW_COST) -> list[int]:
    """Детекции кадра → индекс чемпиона из состава или −1.

    Цена пары — −log p(чемпион) плюс −log p(команда детектора совпадает с
    командой чемпиона). Детекция, которой ни один чемпион не подходит
    дешевле NEW_COST, остаётся без чемпиона.
    """
    if not dets:
        return []
    cand = [(c, t) for t in (0, 1) for c in teams[t]]
    n, m = len(dets), len(cand)
    cost = np.full((n, m + n), 1e6)
    for i, d in enumerate(dets):
        tp = np.asarray(d.get("team_p", [1.0, 1.0]), dtype=float)
        for j, (c, t) in enumerate(cand):
            team_cost = -np.log(max(tp[t], 1e-4)) if d["team"] != t else 0.0
            cost[i, j] = -float(d["logp"][c]) + team_cost
        cost[i, m + i] = new_cost
    rows, cols = linear_sum_assignment(cost)
    out = [-1] * n
    for i, j in zip(rows, cols):
        if j < m:
            out[i] = cand[j][0]
    return out


def frame_records(dets: list[dict], teams: list[list[int]], names: list[str],
                  layout: dict, w: int = REF_W, h: int = REF_H) -> list[dict]:
    """10 записей кадра: чемпион, команда, видим, x, y (ед. карты), сетка 256.
    Жив/мёртв и таймер добавляет HUD (hud_panels)."""
    who = assign(dets, teams)
    seen = defaultdict(lambda: None)
    for d, c in zip(dets, who):
        if c >= 0:
            seen[c] = d
    rec = []
    for t in (0, 1):
        for c in teams[t]:
            d = seen[c]
            r = {"champion": names[c], "team": TEAMS[t], "visible": d is not None,
                 "x": None, "y": None}
            if d is not None:
                x, y = grid_to_map(d["x"], d["y"], layout, w, h)
                r.update(x=round(x), y=round(y), mm=[round(d["x"], 2), round(d["y"], 2)],
                         score=round(d["score"], 3), p=round(float(np.exp(d["logp"][c])), 3))
            rec.append(r)
    return rec


# --- Состав по портретам HUD и чтение кадра целиком -------------------------

PORTRAIT_MIN = 0.45       # ниже — портрет слота не виден (панель скрыта, заставка)


def roster_from_hud(frames_bgr: list[np.ndarray], icons: dict[str, np.ndarray],
                    names: list[str]) -> tuple[list[str], np.ndarray]:
    """Чемпионы слотов L1…L5, R1…R5 по портретам нескольких кадров.

    Сходство усредняется по кадрам (серый портрет мёртвого и перекрытия
    размываются), затем назначение «слот → чемпион» без повторов
    (венгерский алгоритм по всем чемпионам Data Dragon). Возвращает имена
    и отрыв лучшего чемпиона слота от второго — мера уверенности.
    """
    from lolvision import hud_panels

    tm = hud_panels.portrait_templates({n: icons[n] for n in names})
    S = np.mean([hud_panels.portrait_scores(f, tm, names) for f in frames_bgr], 0)
    rows, cols = linear_sum_assignment(-S)
    picked = [names[c] for c in cols[np.argsort(rows)]]
    srt = np.sort(S, 1)
    return picked, srt[:, -1] - srt[:, -2]


class FrameReader:
    """Кадр канала → 10 записей {чемпион, команда, слот, видим, x, y, жив, таймер}.

    roster — 10 имён Data Dragon в порядке слотов HUD (L1…L5 синие, R1…R5
    красные). Видимость и положение — с миникарты, жив/мёртв и таймер —
    только с HUD: исчезновение с миникарты смертью не считается.
    """

    def __init__(self, model, roster: list[str], layout: dict, icons: dict[str, np.ndarray]):
        from lolvision import hud_panels

        self.hp = hud_panels
        self.model = model
        self.layout = layout
        self.roster = roster
        idx = {n: i for i, n in enumerate(model.champions)}
        self.ids = [idx[n] for n in roster]
        self.teams = [self.ids[:5], self.ids[5:]]
        self.digits = hud_panels.DigitReader()
        self.tm = hud_panels.portrait_templates({n: icons[n] for n in roster})
        self.scales = None
        self.new_cost = NEW_COST

    @staticmethod
    def minimap_of(frame_rgb: np.ndarray, layout: dict) -> np.ndarray:
        """Миникарта 256×256 так же, как minimap_camera.minimap (LANCZOS)."""
        from PIL import Image

        h, w = frame_rgb.shape[:2]
        x0, y0, x1, y1 = crop_box(layout, w, h)
        im = Image.fromarray(frame_rgb[y0:y1, x0:x1]).resize((MM, MM), Image.LANCZOS)
        return np.asarray(im)

    def minimap(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.minimap_of(frame_rgb, self.layout)

    def __call__(self, frames_rgb: list[np.ndarray]) -> list[list[dict]]:
        minis = np.stack([self.minimap(f) for f in frames_rgb])
        dets = self.model(minis)
        return [self.records(f, d) for f, d in zip(frames_rgb, dets)]

    def records(self, frame_rgb: np.ndarray, dets: list[dict]) -> list[dict]:
        h, w = frame_rgb.shape[:2]
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        slots = self.hp.read_panels(bgr, self.digits)
        # Масштаб портрета запоминается по первому уверенному кадру.
        own, best = self.hp.portrait_own(bgr, self.tm, self.roster, self.scales)
        if self.scales is None and (own >= PORTRAIT_MIN).all():
            self.scales = best
        who = assign(dets, self.teams, self.new_cost)
        seen = {c: d for d, c in zip(dets, who) if c >= 0}
        out = []
        for s, (name, cid) in enumerate(zip(self.roster, self.ids)):
            d = seen.get(cid)
            sl = slots[s]
            hud_ok = sl.dead or own[s] >= PORTRAIT_MIN
            r = {"champion": name, "team": TEAMS[s // 5], "slot": f"{'LR'[s // 5]}{s % 5 + 1}",
                 "visible": d is not None, "x": None, "y": None,
                 "alive": (not sl.dead) if hud_ok else None,
                 "respawn": sl.timer if sl.dead else None}
            if d is not None:
                x, y = grid_to_map(d["x"], d["y"], self.layout, w, h)
                r.update(x=round(x), y=round(y), mm=[round(d["x"], 2), round(d["y"], 2)],
                         score=round(d["score"], 3), p=round(float(np.exp(d["logp"][cid])), 3))
            out.append(r)
        return out
