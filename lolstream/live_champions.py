"""Чемпионы в живом потоке: состав матча, миникарта, жив ли и кто есть кто
на основном виде.

Зачем. Модели миникарты (#759, lolvision/minimap_champions.py) и «кто это»
(#760, lolvision/roster.py, lolvision/name_reader.py) работали только по
кадрам на диске. Здесь они собраны под поток: один объект на сеанс, кадр
1080p RGB на входе, короткие записи для строки кадра на выходе.

Что и как часто (канал наблюдателя, раскладка spectator-volibear-challenger;
замер на этой машине):

* состав — портреты HUD (roster.py, ~250 мс на кадр на процессоре). Идёт
  в своём потоке раз в ROSTER_EVERY секунд потока на игровых кадрах, путь
  кадра его не ждёт; решение — по последним ROSTER_FRAMES кадрам с HUD.
  Там же раз в ROSTER_EVERY проверяется, видны ли портреты своих чемпионов
  (hud_panels.portrait_own, ~28 мс): по ним «жив / мёртв» отличается от
  «панели не видно». Новый состав (новый матч) сбрасывает старые кадры;
* миникарта и боковые панели — на каждом игровом кадре: детектор и
  классификатор иконок на видеокарте (~8 мс), назначение в состав, серый
  портрет и таймер возрождения (hud_panels.read_panels, ~2 мс на полосах
  панелей);
* имя над чемпионом на основном виде — по юнитам детектора (unit_net.py),
  а не классическим поиском плашек (330–450 мс на кадр). От центра тела
  до плашки — постоянное смещение разметки детектора (unit_bars.BODY), имя
  ищется в окне ±6×±8 px вокруг ожидаемого места и только среди пяти
  чемпионов команды и надписей-состояний (name_reader). Где имя не
  прочиталось, берётся имя ближайшего чемпиона той же команды с прошлого
  кадра; внутри команды все разные (венгерский алгоритм).

Вид игрока (#786). Портретов наблюдателя у него нет, а над плашкой ник, а
не имя чемпиона. Раскладка решается по последним чтениям портретов
(`view`): у наблюдателя таблица совпадает с иконками на 0.97, у игрока —
на 0.48–0.49. У игрока состав не решается и имена не выдаются: прежде
заставка стримера случайно прошла проверку HUD, состав решился по ней, и
Brand был «Ivern» в 100% кадров (ANXVB571YkE).

Своя игра (#798, 2B). Если есть Live Client API, состав берётся из него
(`set_api`): портреты HUD и боковые панели не читаются, иконки миникарты
назначаются только среди десяти из API, у вида игрока синие иконки —
своя команда, раскладка — «player» (свой игрок есть) или «spectator».

Облик (appearance.py, DINOv2) сюда не взят: признак стоит ~10 мс на вырез
и вторую модель в видеопамяти, а без имени по соседним кадрам он даёт
лишь +2–5 п. п. (docs/champion-identity.md).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from lolvision import ddragon, hud_panels
from lolvision.health_bars import W as PLATE_W
from lolvision.identity import NAME_BOX, crop_pad
from lolvision.match_state import ApiObs
from lolvision.minimap_champions import PORTRAIT_MIN, assign
from lolvision.minimap_icons import (CROP, DEFAULT_RUN as MINI_RUN, MM, TEAMS, IconModel,
                                     crops, decode)
from lolvision.name_reader import NONE, STATUS, WIN_H, WIN_W, NameReader, whiteness
from lolvision.paths import RUNS
from lolvision.records import MinimapChamp
from lolvision.roster import IconBank, decide, read_frame
from lolvision.unit_bars import BODY

NAMES = RUNS / "identity" / "names.npz"
ROSTER_EVERY = 5.0        # с потока, как часто читать портреты HUD
ROSTER_FRAMES = 8         # по скольким последним кадрам с HUD решать
# Первый состав — не меньше чем по стольким кадрам с HUD: один случайный
# кадр иначе закрепляет состав до конца ролика (#786). Больше двух — дорого:
# чтение портретов идёт 2–3 с (замер 2026-10-12, а не ~250 мс, как выше), и
# при потоке ×5 три чтения — это ~50 с ролика без имён.
ROSTER_MIN = 2
# Раскладка ролика (#786) — по последним чтениям портретов на игровых кадрах:
# у наблюдателя таблица портретов есть почти всегда, у вида игрока её нет.
# Решение — когда чтений не меньше VIEW_MIN, по большинству из VIEW_READS;
# принятое решение меняется, только если против него не меньше VIEW_FLIP
# чтений: кадры со скрытым HUD (затемнение, заставка, клавиша H) у
# наблюдателя иначе дёргают раскладку туда-обратно (j8_-kWWh5qM: 15 из 54
# кадров без таблицы).
VIEW_READS, VIEW_MIN, VIEW_FLIP = 8, 4, 6
# Полосы боковых панелей в кадре 1080p — всё, что читает hud_panels.
PANEL_COLS = ((0, 80), (1840, 1920))
PANEL_ROWS = (140, 680)
# Центр тела чемпиона у детектора — верх плашки + dy + h/2 (unit_bars.BODY).
_, BODY_DY, _, BODY_H = BODY["champion"]
PLATE_DY = BODY_DY + BODY_H / 2
# Запас окна поиска имени сверх сдвигов name_reader (±4 по x, ±2 по y), px
# 1080p: имя от центра детектора — в 99% в пределах ±4 по x и ±7 по y
# (435 уверенных прочтений, 58w57eJ5Qks, olmTXkkUv58, 4AIX8QtRid4).
SLACK_X, SLACK_Y = 2, 6
NAME_MIN = 0.5            # сходство имени не ниже (name_reader.ADAPT_SIM)
NAME_MARGIN = 0.03        # и отрыв от второго кандидата
TRACK_PX = 260            # px 1080p: имя переносится с прошлого кадра не дальше
TRACK_SCORE = 0.35        # «вес» перенесённого имени против прочитанного
MAX_ICONS = 20            # детекций миникарты на кадр не больше (minimap_icons.decode)


def _few(names: list[str], n: int = 5) -> str:
    more = f" и ещё {len(names) - n}" if len(names) > n else ""
    return ", ".join(names[:n]) + more


def as_1080(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == (1080, 1920):
        return frame
    return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)


def _to_input(u8: torch.Tensor) -> torch.Tensor:
    """uint8 N×H×W×3 на устройстве → вход сетей миникарты (как to_tensor)."""
    return u8.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


class IconGraph:
    """Детектор и классификатор иконок миникарты на одну миникарту за раз,
    каждый — CUDA graph. Сети мелкие, и без графа время уходит на запуск
    ~60 слоёв из Python: ~26 мс на кадр против ~8 мс. Ответ — как у
    IconModel на пачке из одной миникарты."""

    def __init__(self, model: IconModel):
        self.model, dev = model, model.device
        self.x = torch.zeros(1, MM, MM, 3, dtype=torch.uint8, device=dev)
        self.c = torch.zeros(MAX_ICONS, CROP, CROP, 3, dtype=torch.uint8, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._det()
                self._cls()
        torch.cuda.current_stream().wait_stream(s)
        self.g_det, self.g_cls = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_det), torch.no_grad():
            self.heat, self.off = self._det()
        with torch.cuda.graph(self.g_cls), torch.no_grad():
            self.logp, self.pt = self._cls()

    def _det(self):
        return self.model.det(_to_input(self.x))

    def _cls(self):
        lc, lt = self.model.cls(_to_input(self.c))
        return F.log_softmax(lc, 1), F.softmax(lt, 1)

    @torch.no_grad()
    def __call__(self, mini: np.ndarray) -> list[dict]:
        self.x.copy_(torch.from_numpy(np.ascontiguousarray(mini))[None])
        self.g_det.replay()
        ds = decode(self.heat, self.off, self.model.thr, MAX_ICONS)[0]
        if not ds:
            return []
        n = len(ds)
        self.c[:n].copy_(torch.from_numpy(crops(mini, [(d.x, d.y) for d in ds])))
        self.g_cls.replay()
        logp, pt = self.logp[:n].cpu().numpy(), self.pt[:n].cpu().numpy()
        return [{"x": d.x, "y": d.y, "team": d.team, "score": d.score,
                 "team_p": pt[i].tolist(), "logp": logp[i]} for i, d in enumerate(ds)]


class Match:
    """Состав и всё, что от него зависит; заменяется целиком, не правится.

    `roster` — по пять чемпионов на цвет иконок миникарты: сначала синие,
    затем красные. `sides` — какая команда у синих и у красных: у
    наблюдателя (и состава по портретам HUD) — ORDER и CHAOS, у вида игрока
    цвета относительны — синие свои (`api=True`, состав из Live Client API).
    У состава из API нет порядка слотов HUD, поэтому панели и портреты не
    читаются: жив / мёртв даёт сам API."""

    def __init__(self, roster: list[str], model: IconModel, icons: dict,
                 sides: tuple[str, str] = TEAMS, api: bool = False):
        self.roster = roster
        self.sides, self.api = sides, api
        idx = {n: i for i, n in enumerate(model.champions)}
        self.ids = [idx[n] for n in roster]
        self.teams = [self.ids[:5], self.ids[5:]]
        self.tm = None if api else hud_panels.portrait_templates({n: icons[n] for n in roster})
        self.own = np.zeros(10, np.float32)       # видны ли портреты (portrait_own)
        self.scales: list[int] | None = None
        self.me_team: str | None = None           # своя команда по API

    def slot(self, s: int) -> tuple[str, str]:
        """Команда и слот (L1…L5 — ORDER, R1…R5 — CHAOS) записи `s`."""
        team = self.sides[s // 5]
        return team, f"{'LR'[TEAMS.index(team)]}{s % 5 + 1}"


class LiveChampions:
    """Чемпионы одного сеанса. `minimap` и `who` зовутся из рабочего потока
    сеанса, состав считается в своём потоке (`offer`)."""

    def __init__(self, device: str = "cuda", run: Path = MINI_RUN, names: Path = NAMES):
        self.model = IconModel(run, device)
        self.icon_net = (IconGraph(self.model) if self.model.device.startswith("cuda")
                         else lambda mini: self.model(mini[None])[0])
        self.icons = ddragon.load_icons()
        self.bank = IconBank(self.icons)
        self.names = NameReader.load(names) if names.is_file() else None
        # Чего сеанс не умеет и почему — в note сеанса (#788): без банка имён
        # who() молчит, и об этом надо сказать, а не писать «чемпионы».
        self.notes: list[str] = []
        if self.names is None:
            self.notes.append("без имён: нет банка имён")
        # Списки чемпионов модели миникарты и Data Dragon должны совпадать:
        # чемпион состава, которого нет у модели, роняет Match (KeyError).
        self._model_ids = set(self.model.champions)
        no_dd = sorted(set(self.model.champions) - set(self.icons))
        no_model = sorted(set(self.icons) - set(self.model.champions))
        if no_dd:
            self.notes.append(f"нет в Data Dragon: {_few(no_dd)}")
        if no_model:
            self.notes.append(f"модель миникарты не знает: {_few(no_model)}")
        # Последняя ошибка потока состава: {"text", "time"} или None.
        self.error: dict | None = None
        self.api_error: str | None = None     # почему состав из API не принят
        # Образцы имён окнами (WIN_H, WIN_W) для cv2.matchTemplate.
        self.name_tm = (None if self.names is None else
                        np.ascontiguousarray(self.names.bank.T.reshape(-1, WIN_H, WIN_W)))
        self.digits = hud_panels.DigitReader()
        self.match: Match | None = None
        self.hud: deque = deque(maxlen=ROSTER_FRAMES)
        self.reads: deque = deque(maxlen=VIEW_READS)        # был ли HUD наблюдателя
        self._view: str | None = None                       # принятая раскладка
        self.next_roster = 0.0
        self.prev: list[tuple[str, str, float, float]] = []   # имя, команда, x, y (px)
        self.bgr = np.zeros((1080, 1920, 3), np.uint8)       # только полосы панелей
        self._job: np.ndarray | None = None
        self._wake = threading.Event()
        self._busy = False
        self._closed = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    @property
    def roster(self) -> list[str] | None:
        m = self.match
        return m.roster if m else None

    @property
    def view(self) -> str | None:
        """«spectator», «player» или None — чтений пока мало. При составе из
        Live Client API — по нему: есть свой игрок — вид игрока, нет —
        наблюдатель в клиенте."""
        m = self.match
        if m is not None and m.api:
            return "player" if m.me_team is not None else "spectator"
        n, spec = len(self.reads), sum(self.reads)
        if n < VIEW_MIN:
            return None
        prev = self._view
        if prev == "spectator" and n - spec < VIEW_FLIP:
            return prev
        if prev == "player" and spec < VIEW_FLIP:
            return prev
        self._view = "spectator" if spec * 2 >= n else "player"
        return self._view

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    # ---------- состав из Live Client API ----------

    def set_api(self, obs: ApiObs | None) -> str | None:
        """Состав из Live Client API (match_state.ApiObs) вместо портретов HUD;
        None — API пропал, состав снова по портретам. Возвращает причину
        отказа или None. Иконки миникарты назначаются только среди этих
        десяти; у вида игрока синие — своя команда."""
        if obs is None:
            if self.match is not None and self.match.api:
                self.match = None
                self.hud.clear()
            return None
        why = obs.problem()
        if why is None:
            unknown = [p.name for p in obs.players if p.name not in self._model_ids]
            if unknown:
                why = f"модель миникарты не знает: {_few(unknown)}"
        if why is not None:
            self.api_error = why
            return why
        self.api_error = None
        first = obs.me_team or TEAMS[0]           # синие на миникарте
        sides = (first, TEAMS[1 - TEAMS.index(first)])
        roster = [p.name for side in sides for p in obs.players if p.team == side]
        m = self.match
        if m is None or not m.api or m.roster != roster or m.sides != sides:
            m = Match(roster, self.model, self.icons, sides, api=True)
        m.me_team = obs.me_team
        self.match = m
        return None

    # ---------- состав (свой поток) ----------

    def offer(self, frame: np.ndarray, t: float) -> None:
        """Игровой кадр потока: раз в ROSTER_EVERY с уходит на чтение портретов.
        Пока прошлый кадр не прочитан, новые не копятся. При составе из API
        портреты не читаются."""
        m = self.match
        if (m is not None and m.api) or t < self.next_roster or self._busy:
            return
        self.next_roster = t + ROSTER_EVERY
        self._busy = True
        self._job = as_1080(frame).copy()
        self._wake.set()

    def _loop(self) -> None:
        while not self._closed:
            self._wake.wait()
            self._wake.clear()
            if self._closed:
                return
            frame, self._job = self._job, None
            try:
                if frame is not None:
                    self.read_roster(frame)
            except Exception as exc:                          # noqa: BLE001
                # Поток не умирает: состав неизвестен до следующего окна,
                # ошибка видна в статусе сеанса (champs_error).
                self.match = None
                self.hud.clear()
                self.error = {"text": f"{type(exc).__name__}: {exc}",
                              "time": round(time.time(), 1)}
            finally:
                self._busy = False

    def read_roster(self, frame: np.ndarray) -> bool:
        """Портреты HUD кадра 1080p → состав. True — состав сменился."""
        if self.match is not None and self.match.api:
            return False                    # состав дал API: портреты не нужны
        fs = read_frame(frame, self.bank)
        self.reads.append(fs.hud)
        if not fs.hud:
            return False
        self.hud.append(fs)
        m = self.match
        if m is None and len(self.hud) < ROSTER_MIN:
            return False
        r = decide(list(self.hud), self.bank.ids)
        if m is not None and len(set(r.champions) & set(m.roster)) < 5:
            # Другой матч: старые кадры HUD к нему не относятся.
            self.hud.clear()
            self.hud.append(fs)
            r = decide(list(self.hud), self.bank.ids)
        changed = m is None or r.champions != m.roster
        if changed:
            m = Match(r.champions, self.model, self.icons)
        bgr = np.ascontiguousarray(frame[..., ::-1])
        own, best = hud_panels.portrait_own(bgr, m.tm, m.roster, m.scales)
        if m.scales is None and (own >= PORTRAIT_MIN).all():
            m.scales = best
        m.own = own
        if changed and not (self.match is not None and self.match.api):
            self.match = m                  # пока читали, состав мог прийти из API
        return changed

    # ---------- миникарта и панели ----------

    def minimap(self, frame: np.ndarray, mini: np.ndarray) -> list[MinimapChamp] | None:
        """10 записей по составу (records.MinimapChamp): u, v — доли миникарты
        (0..1, как у рамки камеры), None — не видим; жив — True / False /
        None (панель не видна). None вместо списка — состав не решён.
        `mini` — миникарта 256×256 RGB, нарезанная так же, как в
        minimap_camera (LANCZOS)."""
        m = self.match
        if m is None:
            return None
        dets = self.icon_net(mini)
        slots = None
        if not m.api:
            # Панели HUD — по слотам портретов; у состава из API слотов нет,
            # а жив / мёртв даёт сам API.
            f = as_1080(frame)
            y0, y1 = PANEL_ROWS
            for x0, x1 in PANEL_COLS:
                self.bgr[y0:y1, x0:x1] = f[y0:y1, x0:x1, ::-1]
            slots = hud_panels.read_panels(self.bgr, self.digits)
        # Только среди десяти состава: детекция без пары — «лишняя».
        who = assign(dets, m.teams)
        seen = {c: d for d, c in zip(dets, who) if c >= 0}
        out = []
        for s, (name, cid) in enumerate(zip(m.roster, m.ids)):
            d = seen.get(cid)
            alive = timer = None
            if slots is not None:
                sl = slots[s]
                if sl.dead or m.own[s] >= PORTRAIT_MIN:
                    alive = not sl.dead
                timer = sl.timer if sl.dead else None
            team, slot = m.slot(s)
            out.append(MinimapChamp(name, team, slot, d is not None,
                                    d["x"] / MM if d else None, d["y"] / MM if d else None,
                                    alive, timer))
        return out

    # ---------- кто есть кто на основном виде ----------

    def who(self, frame: np.ndarray, units: list) -> list[str | None]:
        """Имена для юнитов детектора (Unit из unit_net, пиксели кадра):
        по списку юнитов — Id Data Dragon или None (не чемпион, нет
        состава, не прочиталось и не перенеслось)."""
        out: list[str | None] = [None] * len(units)
        roster = self.roster
        # У вида игрока над плашкой ник, а не имя чемпиона: читать нечего.
        if roster is None or self.names is None or self.view == "player":
            return out
        s = 1920 / frame.shape[1]                       # всё в пикселях 1080p
        f = as_1080(frame)
        cur = []
        for team_i, team in enumerate(TEAMS):
            cands = roster[5 * team_i:5 * team_i + 5]
            idx = [k for k, u in enumerate(units) if u.kind == "champion" and u.team == team]
            if not idx:
                continue
            score = np.zeros((len(idx), 5), np.float32)
            for a, k in enumerate(idx):
                x, y = units[k].x * s, units[k].y * s
                sims = self.name_scores(f, x, y, cands)
                if sims is not None:
                    order = np.argsort(-sims)
                    best, gap = sims[order[0]], sims[order[0]] - sims[order[1]]
                    if best >= NAME_MIN and gap >= NAME_MARGIN:
                        score[a, order[0]] += best
                for name, pt, px, py in self.prev:
                    if pt == team and name in cands and np.hypot(px - x, py - y) <= TRACK_PX:
                        score[a, cands.index(name)] += TRACK_SCORE
            rows, cols = linear_sum_assignment(-score)
            for a, c in zip(rows, cols):
                if score[a, c] > 0:
                    k = idx[a]
                    out[k] = cands[c]
                    cur.append((cands[c], team, units[k].x * s, units[k].y * s))
        self.prev = cur
        return out

    def name_scores(self, f: np.ndarray, x: float, y: float,
                    cands: list[str]) -> np.ndarray | None:
        """Сходство надписи над чемпионом (центр тела x, y в кадре 1080p) с
        именами `cands`: нормированная корреляция по сдвигам в окне
        (SLACK_X, SLACK_Y) — та же, что у NameReader, только через
        cv2.matchTemplate, — и по образцам — максимум. Если надпись-состояние
        или «пусто» ближе любого имени — None: это не голос."""
        px, py = round(x - PLATE_W / 2), round(y - PLATE_DY)
        ex, ey = SLACK_X, SLACK_Y
        x0, y0, x1, y1 = NAME_BOX
        wn = whiteness(crop_pad(f, px, py, (x0 - ex, y0 - ey, x1 + ex, y1 + ey)))
        if wn.max() <= 0:
            return None
        keep = set(cands)
        best: dict[str, float] = {}
        for k, lab in enumerate(self.names.labels):
            if lab in keep or lab.startswith(STATUS) or lab == NONE:
                sim = float(cv2.matchTemplate(wn, self.name_tm[k], cv2.TM_CCOEFF_NORMED).max())
                best[lab] = max(best.get(lab, -2.0), sim)
        other = max([v for lab, v in best.items() if lab not in keep] or [-2.0])
        own = np.array([best.get(c, -2.0) for c in cands], np.float32)
        if other >= own.max():
            return None
        return own
