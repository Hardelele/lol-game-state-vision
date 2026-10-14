"""Офлайн-прогон ролика через путь кадра live, кадры пачками на видеокарте (#843).

Зачем. Набору данных для модели решений нужен /state по каждому кадру
ролика — тот же снимок (state_service.build_snapshot), что видит coach в
игре, но по сотням роликов. Live проходит кадр за кадром: сеть на одном
кадре почти не считает, время уходит на запуск ядер. Здесь сети считают
сразу пачку кадров, а всё, что помнит прошлое (паспорт, поиск миникарты и
панели умений, вид, состав, MatchState, часы HUD, трекер, добивания,
окно размена), идёт по готовым выходам строго по порядку кадров — тем же
кодом `live.Session._step` / `_emit`, без копии логики.

Как устроено.

* `OfflineSession` — сеанс live без своего потока: кадры подаёт
  `process`. Сжатие картинок для смотрелки не делается, снимок строится
  сразу после выдачи кадра; часы модуля live подменены временем кадра
  (`_Clock`), поэтому давности в снимке считаются по ролику, а не по
  тому, как быстро шла видеокарта. Чтение состава по портретам — в том же
  потоке, а не в своём (в live оно в фоне, и его итог приходил бы в
  случайный кадр).
* `BatchPre` — чистый инференс пачки: тип сцены, модель камеры (подготовка
  кадра, сеть и сведение голосов), иконки миникарты, детектор юнитов.
  Выходы подставляются в путь кадра заместителями моделей. Вход иконок и
  юнитов зависит от состояния (прямоугольник миникарты, маска вида):
  пачка считается по состоянию на её начало, а если к кадру оно сменилось
  (нашли миникарту, решился вид) — заместитель считает этот кадр обычной
  моделью. Поэтому пакетный путь отличается от последовательного только
  числами пакетного счёта сетей (docs/extract-states.md, «Эквивалентность»).

Стадии (`STAGES`): scene — тип сцены; camera — модель камеры; minimap —
иконки миникарты, состав и вид по HUD (без неё вида нет, и HUD своего
игрока не читается); hud — свой HUD: Q/W/E/R и D/F, CS и часы; units —
юниты на экране (и трекер, добивания, окно размена); names — имена над
чемпионами. По умолчанию — `DEFAULT_STAGES`.
"""

from __future__ import annotations

import contextlib
import gzip
import os
import json
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from lolstream import live
from lolstream import hud_pool
from lolstream import live_champions as lc
from lolstream.frame_source import Frame, Status, VideoSource
from lolstream.state_service import build_snapshot
from lolvision.match_state import ApiObs, ApiPlayer, MatchState
from lolvision.patch_infer import FramePrep, PatchInfer, vote_stats
from lolvision.patch_model import canvases_post
from lolvision.scene_net import SceneNetRunner, small as scene_small
from lolvision.unit_net import decode as unit_decode, prepare as unit_prepare, to_units

STAGES = ("scene", "camera", "minimap", "hud", "units", "names")
DEFAULT_STAGES = ("scene", "camera", "minimap", "hud")
TEAMS = ("ORDER", "CHAOS")


def parse_stages(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """«scene,camera» / «default,+units» / «all,-names» → набор стадий по порядку STAGES."""
    if spec is None:
        return DEFAULT_STAGES
    parts = [p.strip() for p in (spec.split(",") if isinstance(spec, str) else spec) if p.strip()]
    have: set[str] = set()
    for p in parts:
        op, name = (p[0], p[1:]) if p[0] in "+-" else ("=", p)
        names = (DEFAULT_STAGES if name == "default" else STAGES if name == "all" else (name,))
        bad = [n for n in names if n not in STAGES]
        if bad:
            raise ValueError(f"неизвестная стадия {bad[0]!r}: бывают {', '.join(STAGES)}")
        if op == "-":
            have -= set(names)
        else:
            have |= set(names)
    if "names" in have and "units" not in have:
        raise ValueError("имена (names) читаются только над юнитами: нужна и стадия units")
    return tuple(s for s in STAGES if s in have)


# ---------------------------------------------------------------- состав


@dataclass(frozen=True)
class Roster:
    """Состав, заданный снаружи (`--roster`): по пять чемпионов (Id Data
    Dragon) на команду и свой чемпион. Нужен ролику вида игрока без Live
    Client API: состав по портретам там не принимается (паспорт, #786), и без
    состава иконки миникарты не назначаются. Свой чемпион обязателен: на
    миникарте вида игрока синие — своя команда.

    Факты API (уровень, золото, предметы, время матча) отсюда не берутся:
    состав идёт только в назначение иконок (`LiveChampions.set_api`), в
    MatchState его нет."""
    order: tuple[str, ...]
    chaos: tuple[str, ...]
    me: str
    me_team: str

    @classmethod
    def from_dict(cls, d: dict) -> "Roster":
        order, chaos = tuple(d.get("ORDER") or ()), tuple(d.get("CHAOS") or ())
        if len(order) != 5 or len(chaos) != 5:
            raise ValueError(f"состав: нужно 5 + 5, а дано {len(order)} + {len(chaos)}")
        me = d.get("me")
        teams = [t for t, side in zip(TEAMS, (order, chaos)) if me in side]
        if not teams:
            raise ValueError(f"состав: свой чемпион {me!r} не в составе")
        team = d.get("me_team") or teams[0]
        if team not in teams:
            raise ValueError(f"состав: {me!r} не в команде {team}")
        return cls(order, chaos, me, team)

    @classmethod
    def load(cls, spec: str) -> "Roster":
        """Файл JSON или сама строка JSON."""
        p = Path(spec)
        text = p.read_text(encoding="utf-8") if p.is_file() else spec
        return cls.from_dict(json.loads(text))

    def obs(self) -> ApiObs:
        players = tuple(ApiPlayer(n, t, False, 0.0)
                        for t, side in zip(TEAMS, (self.order, self.chaos)) for n in side)
        return ApiObs(0.0, players, me_team=self.me_team, me=self.me, map_number=11,
                      game_mode="CLASSIC")

    def to_dict(self) -> dict:
        return {"ORDER": list(self.order), "CHAOS": list(self.chaos), "me": self.me,
                "me_team": self.me_team}


# ---------------------------------------------------------------- подмены


class _Clock:
    """Часы модуля live на время прогона: `time()` — время текущего кадра
    ролика, остальное — настоящий модуль time."""

    def __init__(self):
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _SyncPool:
    """Исполнитель «сразу здесь»: `_store` сеанса — в потоке кадра, по порядку."""

    def submit(self, fn, *a, **kw):
        fn(*a, **kw)

    def shutdown(self, wait: bool = True) -> None:
        pass


@contextlib.contextmanager
def patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


class _Token:
    """Вход модели камеры в пакетном пути: номер кадра, ответ уже посчитан."""
    __slots__ = ("i",)

    def __init__(self, i: int):
        self.i = i


def _no_camera(prep):
    """Модель камеры выключена (`-camera`): ответа нет, как у кадра без голосов."""
    nan = float("nan")
    return ({"px": nan, "py": nan, "qx": nan, "qy": nan, "spread": nan},
            np.zeros((1, 1), np.float32), {"net": 0.0, "agg": 0.0})


def sync_offer(ch: lc.LiveChampions) -> Callable:
    """`LiveChampions.offer` без своего потока: чтение портретов — сразу, в
    потоке кадра. Окно ROSTER_EVERY и отказ при составе из API — те же."""
    ch.close()

    def offer(frame: np.ndarray, t: float) -> None:
        m = ch.match
        if (m is not None and m.api) or t < ch.next_roster:
            return
        ch.next_roster = t + lc.ROSTER_EVERY
        try:
            ch.read_roster(lc.as_1080(frame))
        except Exception as exc:                              # noqa: BLE001
            ch.match = None
            ch.hud.clear()
            ch.error = {"text": f"{type(exc).__name__}: {exc}", "time": round(time.time(), 1)}
    return offer


# ---------------------------------------------------------------- пачка


def patch_rows(model: PatchInfer, out: np.ndarray) -> list[tuple[dict, np.ndarray]]:
    """Плоские ответы модели камеры (B, 9 + grid²) → строки и тепловые карты,
    как у `PatchInfer.predict`."""
    rows = []
    for o in out:
        cx, cy, agree, qx, qy, sp, scene_share, n_vote, p_game = o[:9].tolist()
        heat = o[9:].reshape(model.grid, model.grid)
        is_game = not model.has_game or p_game >= model.net.frame_threshold
        ok = bool(np.isfinite([cx, cy]).all()) and is_game
        px, py = (cx, cy) if ok else (float("nan"),) * 2
        row = {"px": px, "py": py, "qx": qx, "qy": qy, "spread": sp, "agree": agree,
               "scene": scene_share, "votes": int(n_vote)}
        if model.has_game:
            row["game"] = p_game
            row["is_game"] = bool(is_game)
        rows.append((row, heat))
    return rows


@torch.no_grad()
def patch_batch(model: PatchInfer, fp: FramePrep, frames: torch.Tensor) -> np.ndarray:
    """Пачка кадров (B, H, W, 3) uint8 в видеопамяти → ответы модели камеры
    (B, 9 + grid²): подготовка — как FramePrep.__call__, сеть и сведение
    голосов — как PatchInfer._post, только для всей пачки разом."""
    b = frames.shape[0]
    cx0, cy0, cx1, cy1 = fp.crop
    a = frames[:, cy0:cy1, cx0:cx1].permute(0, 3, 1, 2).float() * fp.keep
    s = F.interpolate(a, size=(fp.store[1], fp.store[0]), mode="bicubic",
                      antialias=True, align_corners=False).clamp_(0, 255).round_()
    del a
    cv = F.grid_sample(s, fp.grid.expand(b, -1, -1, -1), mode="bilinear",
                       padding_mode="zeros", align_corners=True).div_(255)
    del s
    v = fp.val.expand(b, -1, -1)
    q = torch.tensor(fp.q0, device=cv.device, dtype=torch.float32)[None].expand(b, -1)
    net = model.net
    emb, lo, sc = net(cv)
    cams, agree, info = canvases_post(net, emb, lo, sc, v, q, model.units)
    out = []
    for i in range(b):
        cam = cams[i].float()
        votes = (info["pts"][i] - info["off"][i]).reshape(-1, 2).float()
        w = info["w"][i].reshape(-1).float()
        live_p = (info["valid"][i].reshape(-1) >= 0.5).float()
        heat, small = vote_stats(votes, w, cam, model.grid)
        sc_ = (info["scene"][i].reshape(-1) > 0.5).float()
        game = (info["game"][i].reshape(1).float() if "game" in info
                else torch.ones(1, device=w.device))
        out.append(torch.cat([cam, agree[i:i + 1].float(), small,
                              ((sc_ * live_p).sum() / live_p.sum().clamp(min=1))[None],
                              (w > 0).sum()[None].float(), game, heat.reshape(-1)]))
    return torch.stack(out).cpu().numpy()


@torch.no_grad()
def units_batch(runner, frames: torch.Tensor) -> list:
    """Детектор юнитов на пачке: та же подготовка, сеть в fp16 и decode, что у
    UnitNetRunner._run, только без CUDA graph на один кадр."""
    x = torch.cat([unit_prepare(frames[k], runner.keep) for k in range(frames.shape[0])])
    with torch.autocast("cuda", dtype=runner.dtype, enabled=runner.dtype == torch.float16):
        h, t, o, hp = runner.net(x)
    rows = unit_decode(h, t, o, hp, runner.thr, valid=runner.valid).cpu().numpy()
    return [to_units(r, runner.scale) for r in rows]


def _mini_key(region: np.ndarray, mmbox) -> tuple:
    return (region.__array_interface__["data"][0], region.shape, tuple(mmbox))


class BatchPre:
    """Чистый счёт пачки кадров по состоянию пути кадра на её начало.

    Сети — пачкой на видеокарте: тип сцены `kind[k]`, камера `cam[i]`
    (по сквозному номеру кадра), иконки `dets` (по содержимому миникарты),
    юниты `units[k]` (при маске вида `zone`). Чистые функции кадра на
    процессоре — в пуле потоков сеанса: миникарта `minis` (по месту кадра в
    памяти, как live._mini), её белая маска `masks` и рамка `boxes` (по
    содержимому), полоса статистики HUD `stats[k]`. Заместители в
    `OfflineSession` берут отсюда, а при сменившемся состоянии или другом
    входе считают кадр обычным кодом."""

    def __init__(self, s: "OfflineSession", pl: live.Pipeline, imgs: list[np.ndarray],
                 first: int, times: dict | None = None):
        self.imgs, self.first, self.pl = imgs, first, pl
        self.times = times if times is not None else {}
        dev = pl.device
        b = len(imgs)
        pool = s._pool
        rx0, ry0 = pl.region0
        mmbox, bw, bh = tuple(pl.mmbox), pl.bw, pl.bh
        champs = pl.champs is not None
        hud = "hud" in s.stages

        # Процессор: всё, что от состояния не зависит или зависит от
        # прямоугольника миникарты и рамки на начало пачки.
        def cpu(k):
            im = imgs[k]
            out = {}
            if s._kind_orig is not None:
                out["small"] = scene_small(im)
            if champs:
                region = im[ry0:, rx0:]
                mini = s._mini_orig(region, mmbox, rx0, ry0)
                arr = np.asarray(mini)
                mf = np.asarray(mini, dtype=np.float32)
                mask = s._white_orig(mf)
                out.update(key=_mini_key(region, mmbox), mini=mini, arr=arr,
                           mkey=mf.tobytes(), mask=mask)
                if bw is not None:
                    out["box"] = (mask.tobytes(), bw, bh, s._box_orig(mask, bw, bh))
            return out
        t0 = time.perf_counter()
        futs = [pool.submit(cpu, k) for k in range(b)]
        # Полоса HUD — в процессах (lolstream.hud_pool): в потоках её держит GIL.
        strips = ([np.ascontiguousarray(live.hud_stats.strip(im)[..., ::-1]) for im in imgs]
                  if hud else None)
        sfuts = None
        if hud:
            sfuts = ([s._procs.submit(hud_pool.read, st) for st in strips] if s._procs is not None
                     else [pool.submit(s._stats_orig, pl.stats, st) for st in strips])

        pin = s._pinned(b, imgs[0].shape)
        for k, im in enumerate(imgs):
            pin[k].numpy()[:] = im
        self.frames = pin[:b].to(dev, non_blocking=True)
        self._tick("upload", t0)

        self.cam: dict[int, tuple] = {}
        if isinstance(s._prep_orig, FramePrep) and isinstance(pl.model, PatchInfer):
            t1 = time.perf_counter()
            out = patch_batch(pl.model, s._prep_orig, self.frames)
            ms = (time.perf_counter() - t1) * 1000 / b
            for k, (row, heat) in enumerate(patch_rows(pl.model, out)):
                self.cam[first + k] = (row, heat, {"net": ms, "agg": 0.0})
            self._tick("camera", t1)

        self.units: list | None = None
        self.zone = pl.zone
        if pl.unit_model is not None:
            t1 = time.perf_counter()
            self.units = units_batch(pl.unit_model, self.frames)
            self._tick("units", t1)

        t1 = time.perf_counter()
        res = [f.result() for f in futs]
        self._tick("cpu_wait", t1)

        self.kind: list | None = None
        km = s._kind_orig
        if km is not None and isinstance(km, SceneNetRunner):
            t1 = time.perf_counter()
            x = torch.from_numpy(np.stack([r["small"] for r in res])).to(dev)
            with torch.no_grad():
                self.kind = list(km.net(x.permute(0, 3, 1, 2).float() / 255).cpu().numpy())
            self._tick("scene", t1)

        self.minis: dict[tuple, object] = {}
        self.masks: dict[bytes, np.ndarray] = {}
        self.boxes: dict[tuple, tuple] = {}
        self.dets: dict[bytes, list] = {}
        if champs:
            for r in res:
                self.minis[r["key"]] = r["mini"]
                self.masks[r["mkey"]] = r["mask"]
                if "box" in r:
                    mk, w_, h_, box = r["box"]
                    self.boxes[(mk, w_, h_)] = box
            t1 = time.perf_counter()
            with torch.no_grad():
                dets = pl.champs.model(np.stack([r["arr"] for r in res]))
            for r, d in zip(res, dets):
                self.dets[r["arr"].tobytes()] = d
            self._tick("icons", t1)

        self.stats: list | None = None
        if hud:
            t1 = time.perf_counter()
            self.stats = [(st, f.result()) for st, f in zip(strips, sfuts)]
            self._tick("hud_wait", t1)

    def _tick(self, key: str, t0: float) -> None:
        if self.frames.is_cuda:
            torch.cuda.synchronize()
        self.times[key] = self.times.get(key, 0.0) + (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- сеанс


class OfflineSession(live.Session):
    """Сеанс live, которому кадры подаёт `process` (без своего потока).

    `stages` — что считать (STAGES); `batch` — по сколько кадров считать
    сети (1 — обычный последовательный путь live, без заместителей);
    `roster` — состав для вида игрока без API (Roster); `on_record(rec)` —
    каждая готовая запись: время ролика, время матча, снимок /state и строка
    кадра `row` (как у смотрелки, без картинок)."""

    def __init__(self, source: VideoSource, *, run: str = live.DEFAULT_RUN,
                 device: str = "cuda", stages: Iterable[str] = DEFAULT_STAGES,
                 batch: int = 1, roster: Roster | None = None,
                 on_record: Callable[[dict], None] | None = None, keep_row: bool = False,
                 live_extras: bool = False, cpu_workers: int = 1, hud_procs: int = 2):
        self.stages = frozenset(stages)
        self.batch = max(1, int(batch))
        self.roster_in = roster
        self._roster_obs = roster.obs() if roster is not None else None
        self.on_record = on_record
        self.keep_row = keep_row
        # Как в live: уменьшенный кадр целиком и сжатие картинок для смотрелки
        # (нужно только замеру пути live, набору — нет).
        self.live_extras = live_extras
        self.on_pipe: Callable[[live.Pipeline], None] | None = None
        self.clock = _Clock()
        self.seq = 0
        self.times: dict[str, float] = {}       # мс по стадиям пакетного счёта, сумма
        self.misses: dict[str, int] = {}        # кадров, посчитанных мимо пачки
        self._pre: BatchPre | None = None
        self._cur = 0                           # номер кадра в пачке
        self._gidx = 0                          # сквозной номер кадра (для камеры)
        self._pin: torch.Tensor | None = None
        self._kind_orig = self._prep_orig = None
        self._mini_orig = live._mini
        self._white_orig = live.white_mask
        self._box_orig = live.find_box_clipped
        self._stats_orig = live.hud_stats.StatsReader.read_strip
        self._pool = ThreadPoolExecutor(max(1, cpu_workers), thread_name_prefix="extract-cpu")
        # Процессы чтения полосы HUD — только пакетному пути и только со стадией hud.
        self._procs = (ProcessPoolExecutor(hud_procs, initializer=hud_pool.init)
                       if hud_procs > 0 and int(batch) > 1 and "hud" in self.stages else None)
        self._pre_cam_store: dict[int, tuple] = {}     # ответы камеры пачки до выдачи кадра
        super().__init__("extract", run, source, device=device, build=self._build)
        self._out.shutdown(wait=True)
        self._out = _SyncPool()

    # ---------- сборка пути кадра ----------

    def _run(self) -> None:
        """Своего потока нет: кадры подаёт process()."""

    def _build(self, size, run, device, fps, champs=None) -> live.Pipeline:
        no_units = "units" not in self.stages
        with contextlib.ExitStack() as st:
            if no_units:
                st.enter_context(patched(live, "live_net", lambda: (None, None)))
            pl = (live.build_pipeline(size, run, device, fps, champs=champs) if champs is not None
                  else live.build_pipeline(size, run, device, fps))
        if no_units:
            pl.note += " · юниты выключены (стадия units)"
        if "scene" not in self.stages:
            pl.kind_model = None
            pl.note += " · тип сцены выключен: всё — игра"
        if "camera" not in self.stages:
            pl.gpu_prep = lambda fr: (None, None)
            pl.model.predict = _no_camera
            pl.note += " · модель камеры выключена"
        if pl.champs is not None and "minimap" not in self.stages:
            pl.champs.close()
            pl.champs = None
            pl.note += " · миникарта выключена: вида и HUD нет"
        if pl.champs is not None and champs is None:
            if "names" not in self.stages:
                pl.champs.names = None
            pl.champs.offer = sync_offer(pl.champs)
        if self.batch > 1:
            self._install(pl)
        if self.on_pipe is not None:
            self.on_pipe(pl)
        return pl

    def _install(self, pl: live.Pipeline) -> None:
        """Заместители моделей: ответ из пачки, мимо неё — обычная модель."""
        self._kind_orig, self._prep_orig = pl.kind_model, pl.gpu_prep
        if pl.kind_model is not None:
            orig = pl.kind_model

            class Kind:
                def probs(_, rgb):
                    pre = self._pre
                    if pre is not None and pre.kind is not None and rgb is pre.imgs[self._cur]:
                        return pre.kind[self._cur]
                    self._miss("scene")
                    return orig.probs(rgb)
            pl.kind_model = Kind()
        if isinstance(pl.gpu_prep, FramePrep):
            prep = pl.gpu_prep
            model = pl.model
            orig_predict = type(model).predict

            def gpu_prep(fr_dev):
                if self._pre is not None and self._gidx in self._pre.cam:
                    return _Token(self._gidx), None
                self._miss("camera")
                return prep(self._pre.frames[self._cur] if self._pre is not None else fr_dev)

            def predict(p):
                if isinstance(p, _Token):
                    return self._pre_cam_store.pop(p.i)
                return orig_predict(model, p)
            pl.gpu_prep = gpu_prep
            model.predict = predict
        if pl.champs is not None:
            ch = pl.champs
            orig_icons = ch.icon_net

            def icon_net(mini):
                pre = self._pre
                if pre is not None:
                    d = pre.dets.get(np.ascontiguousarray(mini).tobytes())
                    if d is not None:
                        return d
                self._miss("icons")
                return orig_icons(mini)
            ch.icon_net = icon_net
        if pl.unit_model is not None:
            um = pl.unit_model
            orig_units = type(um).units

            def units(rgb):
                pre = self._pre
                if pre is not None and pre.units is not None and pl.zone == pre.zone:
                    return pre.units[self._cur]
                self._miss("units")
                return orig_units(um, pre.frames[self._cur] if pre is not None else rgb)
            um.units = units
        if "hud" in self.stages:
            st_orig = pl.stats.read_strip

            def read_strip(st):
                pre = self._pre
                if pre is not None and pre.stats is not None:
                    st0, res = pre.stats[self._cur]
                    if st0.shape == st.shape and np.array_equal(st0, st):
                        return res
                self._miss("hud_stats")
                return st_orig(st)
            pl.stats.read_strip = read_strip
        # Кадр целиком в видеопамять по одному больше не грузится: пачка уже там.
        pl.fr_pin = pl.fr_dev = None

    def _miss(self, key: str) -> None:
        self.misses[key] = self.misses.get(key, 0) + 1

    def _pinned(self, b: int, shape) -> torch.Tensor:
        if self._pin is None or self._pin.shape[0] < b or self._pin.shape[1:] != shape:
            pin = torch.empty((b, *shape), dtype=torch.uint8)
            self._pin = pin.pin_memory() if torch.cuda.is_available() else pin
        return self._pin

    def _white(self, mf: np.ndarray) -> np.ndarray:
        """live.white_mask с памятью пачки (по содержимому миникарты)."""
        pre = self._pre
        if pre is not None and pre.masks:
            m = pre.masks.get(mf.tobytes())
            if m is not None:
                return m
        return self._white_orig(mf)

    def _box(self, mask: np.ndarray, bw: int, bh: int):
        """live.find_box_clipped с памятью пачки (по маске и размеру рамки)."""
        pre = self._pre
        if pre is not None and pre.boxes:
            r = pre.boxes.get((mask.tobytes(), bw, bh))
            if r is not None:
                return r
        return self._box_orig(mask, bw, bh)

    def _mini(self, region, mmbox, rx0, ry0):
        """live._mini с памятью пачки: та же миникарта не режется дважды."""
        pre = self._pre
        if pre is not None:
            m = pre.minis.get(_mini_key(region, mmbox))
            if m is not None:
                return m
        return self._mini_orig(region, mmbox, rx0, ry0)

    # ---------- подмены шагов сеанса ----------

    def _api_step(self, t: float, wall: float):
        """API у ролика нет. Состав снаружи (`roster`) — только иконкам
        миникарты и только на виде игрока: у наблюдателя состав читается по
        портретам, и он точнее (там же жив / мёртв по панелям)."""
        ch = self._pipe.champs if self._pipe is not None else None
        r = self._roster_obs
        if r is not None and ch is not None and self.view == "player" \
                and not (ch.match is not None and ch.match.api):
            ch.set_api(r)
            self._api_obs = r            # чтобы _step не сбросил состав (API «пропал»)
        return None, None

    def _roster_step(self, champs, t: float) -> None:
        if champs is not None and champs.match is not None and champs.match.api \
                and self.roster_in is not None:
            if self.passport.items["roster"].value != "manual":
                self._sig(lambda p: p.set_roster("manual", t, "состав задан снаружи (--roster)"))
            self.roster = champs.roster
            return
        super()._roster_step(champs, t)

    def _spells_step(self, pl, fr, t, game, view):
        return super()._spells_step(pl, fr, t, game, view) if "hud" in self.stages else None

    def _stats_step(self, pl, fr, game, view, off):
        return super()._stats_step(pl, fr, game, view, off) if "hud" in self.stages else None

    def _store(self, row, scene, full_img, mini, hm) -> None:
        """Кадр выдан: снимок /state на его момент — в запись; картинки не сжимаются."""
        if self.live_extras:
            live._jpeg(live.Image.fromarray(scene), 85)
            if full_img is not None:
                live._jpeg(live.Image.fromarray(full_img), 80)
            live._jpeg(mini.resize((live.MINI_STORE, live.MINI_STORE)), 82)
        self.dropped += len(self.rows)
        self.rows = [row]
        self.seq += 1
        snap = build_snapshot(self.state_inputs(), self.clock.now, self.seq)
        gt = (snap.get("match") or {}).get("time") or {}
        rec = {"video_t": row["t"], "game_t": gt.get("value"), "frame": row["i"],
               "state": snap}
        if self.keep_row:
            rec["row"] = {k: v for k, v in row.items() if k != "ms"}
        if self.on_record is not None:
            self.on_record(rec)

    # ---------- работа ----------

    def process(self, frames: Iterable[Frame] | None = None) -> None:
        """Все кадры источника (или `frames`) через путь кадра; записи — в
        on_record. `frames` по умолчанию — чтение источника в своём потоке
        с запасом на две пачки (распаковка идёт, пока считают сети)."""
        src = self.source
        small = live._full_small if self.live_extras else (lambda fr, dev: None)
        with contextlib.ExitStack() as stack:
            for name, fn in (("time", self.clock), ("_mini", self._mini), ("_full_small", small)):
                stack.enter_context(patched(live, name, fn))
            if self.batch > 1:
                stack.enter_context(patched(live, "white_mask", self._white))
                stack.enter_context(patched(live, "find_box_clipped", self._box))
            self.state = live.State.PROBING
            src.open()
            d = src.describe()
            self.title = d.get("title") or self.sid
            self.duration = float(d.get("duration") or 0.0)
            self._open_passport(d)
            pl = self._new_pipe(src.size) if src.size is not None else None
            if pl is not None:
                self.note = f"{self._source_note(src.size)} · {pl.note}"
            self.state = live.State.RUNNING
            it = frames if frames is not None else prefetch(src, 3 * self.batch + 2)
            # Пачка n+1 считается в своём потоке, пока по порядку идёт логика
            # пачки n: по состоянию на момент её запуска (заместители сверяют).
            ahead = (ThreadPoolExecutor(1, thread_name_prefix="extract-pre")
                     if self.batch > 1 else None)
            try:
                cs = chunks(it, self.batch)
                cur = next(cs, None)
                fut = self._ahead(ahead, pl, cur, self._gidx)
                while cur is not None:
                    nxt = next(cs, None)
                    pre = fut.result() if fut is not None else None
                    fut = self._ahead(ahead, pl, nxt,
                                      self._gidx + sum(fr.ok for fr, _ in cur))
                    pl = self._chunk(pl, cur, pre)
                    cur = nxt
                if pl is not None:
                    self._flush(pl)
            finally:
                if ahead is not None:
                    ahead.shutdown(wait=True)
                close = getattr(it, "close", None)
                if close is not None:
                    close()
                src.close()
                if self._pipe is not None:
                    self._pipe.close()
                self._pool.shutdown(wait=True)
                if self._procs is not None:
                    self._procs.shutdown(wait=True)
            self.state = live.State.ENDED

    @staticmethod
    def _lead(pl: live.Pipeline | None, chunk) -> list[Frame]:
        """Кадры пачки до первого кадра другого размера (их и считают сети)."""
        out = []
        for fr, _ in chunk or ():
            if not fr.ok:
                continue
            if pl is None or fr.size != pl.size:
                break
            out.append(fr)
        return out

    def _ahead(self, ex_, pl, chunk, first: int):
        """Запустить счёт пачки в фоне; None — нечего или некому считать."""
        if ex_ is None or pl is None:
            return None
        lead = self._lead(pl, chunk)
        if not lead:
            return None
        return ex_.submit(BatchPre, self, pl, [fr.image for fr in lead], first, self.times)

    def _chunk(self, pl: live.Pipeline | None, chunk: list[tuple[Frame, float]],
               pre: "BatchPre | None" = None):
        ok = [(fr, ms) for fr, ms in chunk if fr.ok]
        for fr, _ in chunk:
            if not fr.ok:
                self._seen(fr)
        if not ok:
            return pl
        # Размер кадра сменился посреди пачки — как в live: путь пересобирается
        # (у ролика так не бывает, но пачка не должна соврать).
        if pl is None or pl.size != ok[0][0].size:
            pl = self._resize(pl, ok[0][0])
        lead = self._lead(pl, chunk)
        if pre is not None and not (pre.pl is pl and pre.first == self._gidx
                                    and len(pre.imgs) == len(lead)):
            pre = None                           # путь пересобран — пачка не та
        if pre is None and self.batch > 1:
            pre = BatchPre(self, pl, [fr.image for fr in lead], self._gidx, self.times)
        self._pre = pre
        if pre is not None:
            self._pre_cam_store.update(pre.cam)
        for k, (fr, ms) in enumerate(ok[:len(lead)]):
            self._cur = k
            self._one(pl, fr, ms)
        self._pre = None
        for fr, ms in ok[len(lead):]:            # кадры нового размера — по одному
            pl = self._resize(pl, fr) if fr.size != pl.size else pl
            self._one(pl, fr, ms)
        return pl

    def _one(self, pl: live.Pipeline, fr: Frame, decode_ms: float) -> None:
        self.clock.now = fr.t
        self._seen(fr)
        self._frame_size(fr.size, fr.t)
        self._step(pl, fr.image, fr.t, decode_ms, fr.t)
        self._gidx += 1

    def _resize(self, pl: live.Pipeline | None, fr: Frame) -> live.Pipeline:
        self._frame_size(fr.size, fr.t)
        if pl is None:
            pl = self._new_pipe(fr.size, t=fr.t)
            self.note = f"{self._source_note(fr.size)} · {pl.note}"
            return pl
        self._flush(pl)
        old, champs = pl.size, pl.champs
        pl.champs = None
        pl.close()
        pl = self._new_pipe(fr.size, champs=champs, t=fr.t)
        self.note += f" · размер кадра {old[0]}x{old[1]} → {fr.size[0]}x{fr.size[1]}"
        return pl


# ---------------------------------------------------------------- замер


class Profiler:
    """Время пути кадра по стадиям (`--profile`): обёртки с синхронизацией
    видеокарты до и после, так что время сети приходится на свою стадию, а не
    на первую выгрузку после неё. Обёртки вложены (step содержит почти всё),
    поэтому в отчёте — время каждой обёртки целиком и число вызовов."""

    SESSION = ("_step", "_emit", "_minimap_step", "_api_step", "_view_step", "_roster_step",
               "_spells_step", "_stats_step", "_self_place_step", "_last_hits_step",
               "_lane_step", "_store", "state_inputs", "_mini_orig")
    MODULE = ((live, "white_mask"), (live, "find_box"), (live, "find_box_clipped"),
              (live, "assign_sides"), (live, "_full_small"), (live, "_jpeg"))

    def __init__(self, s: OfflineSession):
        self.s = s
        self.ms: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self._stack = contextlib.ExitStack()
        for name in self.SESSION:
            setattr(s, name, self.wrap(getattr(s, name), name.lstrip("_")))
        for mod, name in self.MODULE:
            self._stack.enter_context(patched(mod, name, self.wrap(getattr(mod, name), name)))
        self._stack.enter_context(patched(MatchState, "update", self.wrap_method(
            MatchState.update, "match_state")))
        import lolstream.extract as me
        self._stack.enter_context(patched(me, "build_snapshot", self.wrap(build_snapshot,
                                                                          "build_snapshot")))
        s.on_pipe = self.pipe

    def wrap(self, fn, key: str, cuda: bool = False):
        def timed(*a, **kw):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                if cuda:
                    torch.cuda.synchronize()
                self.ms[key] = self.ms.get(key, 0.0) + (time.perf_counter() - t0) * 1000
                self.n[key] = self.n.get(key, 0) + 1
        return timed

    def wrap_method(self, fn, key: str):
        w = self.wrap(fn, key)

        def method(obj, *a, **kw):
            return w(obj, *a, **kw)
        return method

    def pipe(self, pl: live.Pipeline) -> None:
        cuda = pl.device.startswith("cuda")
        if pl.gpu_prep is not None:
            pl.gpu_prep = self.wrap(pl.gpu_prep, "camera_prep", cuda)
        if pl.kind_model is not None:
            pl.kind_model.probs = self.wrap(pl.kind_model.probs, "scene_net", cuda)
        pl.model.predict = self.wrap(pl.model.predict, "camera_net", cuda)
        if pl.unit_model is not None:
            pl.unit_model.units = self.wrap(pl.unit_model.units, "units_net", cuda)
        if pl.tracker is not None:
            pl.tracker.update = self.wrap(pl.tracker.update, "tracker")
        ch = pl.champs
        if ch is not None:
            ch.offer = self.wrap(ch.offer, "roster_read")
            ch.icon_net = self.wrap(ch.icon_net, "icons_net", cuda)
            ch.minimap = self.wrap(ch.minimap, "champs_minimap")
            ch.who = self.wrap(ch.who, "names")

    def close(self) -> None:
        self._stack.close()

    def report(self, frames: int) -> dict:
        return {k: {"ms_per_frame": round(v / max(1, frames), 2), "calls": self.n[k]}
                for k, v in sorted(self.ms.items(), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------- кадры


def prefetch(src, depth: int) -> Iterator[tuple[Frame, float]]:
    """Кадры источника из своего потока: (кадр, мс ожидания кадра в потоке
    чтения). Кончается на ENDED (его не отдаёт)."""
    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    stop = threading.Event()

    def work():
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                fr = src.read()
                q.put((fr, (time.perf_counter() - t0) * 1000))
                if fr.status is Status.ENDED:
                    return
        except BaseException as exc:                          # noqa: BLE001
            q.put(exc)

    th = threading.Thread(target=work, daemon=True, name="extract-read")
    th.start()

    def gen():
        try:
            while True:
                item = q.get()
                if isinstance(item, BaseException):
                    raise item
                fr, ms = item
                if fr.status is Status.ENDED:
                    return
                yield fr, ms
        finally:
            stop.set()
            while th.is_alive():                 # освободить поток чтения
                try:
                    q.get_nowait()
                except queue.Empty:
                    th.join(0.05)
    return gen()


def chunks(it: Iterable, n: int) -> Iterator[list]:
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def open_source(src: str, fps: float, start: float = 0.0, decode: str = "auto") -> VideoSource:
    """Файл или ссылка (http/https — поток YouTube через yt-dlp, как frame_source)."""
    url = src.startswith(("http://", "https://"))
    return VideoSource(src, fps, start, from_url=url, decode=decode)


class Writer:
    """Записи в .jsonl.gz (или .jsonl) по одной на строку."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.f = (gzip.open(self.path, "wt", encoding="utf-8", compresslevel=5)
                  if self.path.suffix == ".gz" else self.path.open("w", encoding="utf-8"))

    def __call__(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=_json) + "\n")
        self.n += 1

    def close(self) -> None:
        self.f.close()


def _json(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, frozenset, tuple)):
        return list(v)
    return str(v)
