"""Ответ патчевой модели по одному кадру — для live и смотрелки.

Зачем отдельный модуль. В patch_model.py есть всё нужное для пачки картинок
(infer_images, predict_canvases), но live нужно другое: один кадр за раз,
разбивка времени по стадиям и то, что смотрелка умеет показать, — точка
камеры, разброс голосов, их тепловая карта, доля патчей «сцена». Сама
модель, геометрия и сведение голосов здесь не повторяются: выходы сети
идут в canvases_post — ту же функцию, что внутри predict_canvases у
probe_hud, поэтому ответ live на тех же кадрах совпадает с проверками.
На видеокарте сеть и сведение голосов записываются в CUDA graphs.

Вход — либо хранимая сцена (после маски и плотной обрезки, 720 px по
ширине; аффинное A «пиксель картинки → пиксель хранимой сцены» для неё
единичное), либо полный кадр через FramePrep, целиком на видеокарте.

Пример (проверка на сохранённых сценах датасета):
    python tools/patch_infer.py runs/patches/cnn-split16/model.pt 58w57eJ5Qks
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import torch.nn.functional as F

from patch_model import (UNITS_PER_PX, load_patchnet, load_projection, rectify,
                         canvases_post, to_world)


def is_patch_checkpoint(ck: dict) -> bool:
    return ck.get("kind") == "patch"


class PatchInfer:
    """Патчевая модель для потока: кадр → точка камеры и сведения о голосах.

    `amp` — autocast на видеокарте. При batch 1 он не ускоряет (время уходит
    на запуск ядер, см. docs/patch-embeddings.md), поэтому по умолчанию
    выключен; включённый даёт тот же ответ в пределах долей единицы.

    `graphs` — CUDA graphs. При batch 1 сеть почти не считает: время уходит
    на запуск сотен мелких ядер по одному. Граф записывает цепочку один раз
    и запускает её одной командой; холст в live одного размера на весь
    сеанс, поэтому запись делается на первом кадре. Замер на холсте 432×272:
    сеть 3.6 → 0.9 мс, сеть + разбор и сведение голосов 8.0 → 1.2 мс, ответ
    тот же побитово. Если запись не удалась, работает обычный режим.
    """

    kind = "patch"

    def __init__(self, path: Path, device: str, amp: bool = False,
                 graphs: bool = True):
        self.device = device
        self.net = load_patchnet(Path(path), device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.units = float(ck.get("units_per_px", UNITS_PER_PX))
        self.trained_on = ck.get("trained_on", [])
        self.label = f"патч-{self.net.kind}"
        self.h0, self.store = load_projection()
        self.amp = amp and device == "cuda"
        self.grid = self.net.grid
        self.has_game = self.net.frame is not None
        # Графы — по форме холста; autocast в них не записываем.
        self.graphs = graphs and device == "cuda" and not self.amp
        self._graphs: dict[tuple, dict | None] = {}

    def describe(self, scene_size) -> str:
        cv, _, _ = rectify(np.zeros((scene_size[1], scene_size[0], 3), np.uint8),
                           np.eye(3), self.h0, self.units, self.net.stride)
        return f"{self.label}, вид сверху {cv.shape[1]}x{cv.shape[0]}"

    def prepare(self, scene):
        """Хранимая сцена (PIL или (H, W, 3) uint8) → холст вида сверху,
        маска следа, центр камеры на холсте."""
        a = np.asarray(scene.convert("RGB")) if hasattr(scene, "convert") else scene
        return rectify(a, np.eye(3), self.h0, self.units, self.net.stride)

    def frame_prep(self, frame_size, crop, boxes, store_size) -> "FramePrep | None":
        """Подготовка полного кадра на видеокарте; без неё — None."""
        if self.device != "cuda":
            return None
        return FramePrep(self, frame_size, crop, boxes, store_size)

    def _post(self, emb, lo, sc, v, q) -> torch.Tensor:
        """Выходы сети → один плоский тензор ответа, целиком на устройстве:
        [камера x, y, согласие, пик x, y, разброс, доля «сцены», число
        голосов, P(игра), тепловая карта grid²]. Один тензор — одна выгрузка."""
        cams, agree, info = canvases_post(self.net, emb, lo, sc, v, q, self.units)
        cam = cams[0].float()
        votes = (info["pts"][0] - info["off"][0]).reshape(-1, 2).float()
        w = info["w"][0].reshape(-1).float()
        live_p = (info["valid"][0].reshape(-1) >= 0.5).float()
        heat, small = vote_stats(votes, w, cam, self.grid)
        sc_ = (info["scene"][0].reshape(-1) > 0.5).float()
        # Голова кадра «игра / не игра» есть не в каждом чекпойнте; без неё
        # кадр считается игрой, как и раньше.
        game = (info["game"][0].reshape(1).float() if "game" in info
                else torch.ones(1, device=w.device))
        return torch.cat([cam, agree[:1].float(), small,
                          ((sc_ * live_p).sum() / live_p.sum().clamp(min=1))[None],
                          (w > 0).sum()[None].float(), game, heat.reshape(-1)])

    def _graph(self, x, v, q) -> dict | None:
        """Записанные графы под эту форму холста или None (обычный режим)."""
        key = (tuple(x.shape), tuple(v.shape))
        if key in self._graphs:
            return self._graphs[key]
        g = None
        try:
            sx, sv, sq = x.clone(), v.clone(), q.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):          # прогрев вне графа обязателен
                for _ in range(3):
                    self._post(*self.net(sx), sv, sq)
            torch.cuda.current_stream().wait_stream(side)
            g_net, g_post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(g_net, pool=pool):
                outs = self.net(sx)
            with torch.cuda.graph(g_post, pool=pool):
                res = self._post(*outs, sv, sq)
            g = {"x": sx, "v": sv, "q": sq, "net": g_net, "post": g_post, "res": res}
        except Exception:                                     # noqa: BLE001
            torch.cuda.synchronize()
        self._graphs[key] = g
        return g

    @torch.no_grad()
    def predict(self, prep) -> tuple[dict, np.ndarray, dict]:
        """Ответ по подготовленному кадру: поля строки, тепловая карта, мс.

        `prep` — от `prepare` (numpy) или от `FramePrep` (холст и маска уже
        на видеокарте, (1, 3, H, W) и (1, H, W) в долях единицы).
        """
        cv, val, q0 = prep
        cuda = self.device == "cuda"
        t0 = time.perf_counter()
        if isinstance(cv, torch.Tensor):
            x, v = cv, val
        else:
            x = torch.from_numpy(cv).to(self.device).permute(2, 0, 1)[None].float().div_(255)
            v = torch.from_numpy(val).to(self.device)[None].float().div_(255)
        q = torch.tensor(q0[None], device=self.device, dtype=torch.float32)
        g = self._graph(x, v, q) if self.graphs else None
        if cuda:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            ev[0].record()
        if g is not None:
            g["x"].copy_(x)
            g["v"].copy_(v)
            g["q"].copy_(q)
            g["net"].replay()
            ev[1].record()
            g["post"].replay()
            res = g["res"]
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                outs = self.net(x)
                if cuda:
                    ev[1].record()
                res = self._post(*outs, v, q)
        if cuda:
            ev[2].record()
        # Одна выгрузка на кадр: каждая лишняя синхронизация с занятой
        # видеокартой стоит миллисекунды.
        out = res.cpu().numpy()
        if cuda:
            net_ms, agg_ms = ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2])
        else:
            net_ms, agg_ms = (time.perf_counter() - t0) * 1000, 0.0
        cx, cy, agree, qx, qy, sp, scene_share, n_vote, p_game = out[:9].tolist()
        heat = out[9:].reshape(self.grid, self.grid)
        # На кадре «не игра» точка камеры бессмысленна: ответа нет, как и у
        # кадра без единого голоса.
        is_game = not self.has_game or p_game >= self.net.frame_threshold
        ok = bool(np.isfinite([cx, cy]).all()) and is_game
        px, py = (cx, cy) if ok else (float("nan"),) * 2
        row = {"px": px, "py": py, "qx": qx, "qy": qy,
               "spread": sp, "agree": agree,
               "scene": scene_share, "votes": int(n_vote)}
        if self.has_game:
            row["game"] = p_game
            row["is_game"] = bool(is_game)
        return row, heat, {"net": net_ms, "agg": agg_ms}


class FramePrep:
    """Полный кадр → вход патчевой модели, целиком на видеокарте.

    Зачем. На процессоре подготовка стоила больше самой модели: копия кадра
    1080p, маска, LANCZOS до хранимой сцены и вид сверху — около 19 мс на
    кадр против ~4 мс сети. Здесь то же самое делается за ~1 мс: обрезка
    по плотной области (видом numpy, без копии всего кадра), маска, сглаженное
    уменьшение до хранимой сцены (bicubic с antialias вместо LANCZOS) и вид
    сверху через grid_sample. Геометрия от ролика к ролику постоянна, поэтому
    сетка выборки, маска следа и центр камеры считаются один раз.

    Совпадение с прежним путём замерено на 172 кадрах 1080p отложенных
    роликов: ответ модели отличается на 0.2 ед. по медиане (p95 0.6) при
    карте в 14 800 ед.; ошибка к метке та же.
    """

    def __init__(self, infer: "PatchInfer", frame_size, crop, boxes, store_size):
        dev = infer.device
        self.dev = dev
        self.crop = tuple(int(v) for v in crop)
        cx0, cy0, cx1, cy1 = self.crop
        self.store = tuple(int(v) for v in store_size)
        sw, sh = self.store
        # Маска в координатах обрезки: прямоугольники HUD, попавшие в неё.
        keep = torch.ones(1, 1, cy1 - cy0, cx1 - cx0, device=dev)
        for x0, y0, x1, y1 in boxes:
            if x1 > cx0 and x0 < cx1 and y1 > cy0 and y0 < cy1:
                keep[..., max(y0, cy0) - cy0:min(y1, cy1) - cy0,
                     max(x0, cx0) - cx0:min(x1, cx1) - cx0] = 0
        self.keep = keep
        # Холст, маска следа и центр камеры — ровно как у rectify на
        # хранимой сцене этого размера; отсюда же обратное отображение
        # «пиксель холста → пиксель сцены» для grid_sample.
        cv0, val0, q0 = infer.prepare(np.zeros((sh, sw, 3), np.uint8))
        hc, wc = cv0.shape[:2]
        t = (np.array([[1, 0, q0[0]], [0, 1, q0[1]], [0, 0, 1]], dtype=np.float64)
             @ to_world(infer.h0, np.eye(3), infer.units))
        ys, xs = np.mgrid[0:hc, 0:wc].astype(np.float64)
        p = np.linalg.inv(t) @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
        gx = p[0] / p[2] / (sw - 1) * 2 - 1
        gy = p[1] / p[2] / (sh - 1) * 2 - 1
        self.grid = torch.tensor(np.stack([gx, gy], -1).reshape(1, hc, wc, 2),
                                 dtype=torch.float32, device=dev)
        self.val = torch.from_numpy(val0).to(dev)[None].float().div_(255)
        self.q0 = q0

    @torch.no_grad()
    def __call__(self, frame: np.ndarray):
        """(H, W, 3) uint8 → (prep для predict, хранимая сцена uint8 для показа)."""
        cx0, cy0, cx1, cy1 = self.crop
        a = torch.from_numpy(np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])).to(self.dev)
        a = a.permute(2, 0, 1)[None].float() * self.keep
        s = F.interpolate(a, size=(self.store[1], self.store[0]), mode="bicubic",
                          antialias=True, align_corners=False).clamp_(0, 255).round_()
        cv = F.grid_sample(s, self.grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True).div_(255)
        scene = s[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        return (cv, self.val, self.q0), scene


def vote_stats(votes: torch.Tensor, w: torch.Tensor, cam: torch.Tensor,
               grid: int):
    """Голоса патчей → тепловая карта (grid, grid) и (пик x, пик y, разброс).

    Тепловая карта — сумма весов голосов по клеткам сетки карты: то же,
    что смотрелка рисует для CoordNet (распределение по карте), только это
    не softmax одной головы, а гистограмма голосов. Разброс — взвешенное
    СКО голосов от ответа в долях карты, как spread у CoordNet: малый —
    патчи согласны, большой — голоса расходятся по нескольким местам. Всё
    остаётся на устройстве голосов; без голосов пик и разброс — NaN.
    """
    ij = (votes.clamp(0, 1 - 1e-6) * grid).long()
    k = ij[:, 1] * grid + ij[:, 0]
    ww = w * ((votes >= 0) & (votes < 1)).all(1)
    heat = torch.zeros(grid * grid, device=w.device).index_add_(0, k, ww)
    tot = ww.sum()
    top = heat.argmax()
    peak = torch.stack([(top % grid).float() + 0.5, (top // grid).float() + 0.5]) / grid
    sp = torch.sqrt((((votes - cam[None]) ** 2).sum(1) * ww).sum() / tot)
    small = torch.cat([peak, sp[None]])
    small = torch.where((tot > 0) & torch.isfinite(cam).all(), small,
                        torch.full_like(small, float("nan")))
    return heat.reshape(grid, grid), small


def _main() -> None:
    import argparse
    import csv
    from PIL import Image
    from paths import COORDS_DATA

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PatchInfer(args.model, dev)
    rows = [r for r in csv.DictReader((COORDS_DATA / f"{args.video}.csv").open(encoding="utf-8"))
            if float(r["quality"]) >= 0.4][:args.n]
    err = []
    for r in rows:
        with Image.open(COORDS_DATA / r["scene"]) as s:
            a = np.asarray(s.convert("RGB"))
        out, _, ms = m.predict(m.prepare(a))
        err.append(np.hypot(out["px"] - float(r["cx"]), out["py"] - float(r["cy"])) * 14800)
    print(f"{m.describe((720, 372))}: {len(err)} кадров, медиана {np.median(err):.0f} ед.,"
          f" последний кадр {ms}")


if __name__ == "__main__":
    _main()
