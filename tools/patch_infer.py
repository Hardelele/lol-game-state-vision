"""Ответ патчевой модели по одному кадру — для live и смотрелки.

Зачем отдельный модуль. В patch_model.py есть всё нужное для пачки картинок
(infer_images, predict_canvases), но live нужно другое: один кадр за раз,
разбивка времени по стадиям и то, что смотрелка умеет показать, — точка
камеры, разброс голосов, их тепловая карта, доля патчей «сцена». Сама
модель, геометрия и сведение голосов здесь не повторяются: кадр уходит в
predict_canvases ровно так же, как в probe_hud, поэтому ответ live на тех
же кадрах совпадает с проверками.

Вход — хранимая сцена (после маски и плотной обрезки, 720 px по ширине),
та же, на которой модель училась и которую live и так держит для показа;
аффинное A «пиксель картинки → пиксель хранимой сцены» для неё единичное.

Пример (проверка на сохранённых сценах датасета):
    python tools/patch_infer.py runs/patches/cnn-split16/model.pt 58w57eJ5Qks
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from patch_model import (UNITS_PER_PX, load_patchnet, load_projection, rectify,
                         predict_canvases)


def is_patch_checkpoint(ck: dict) -> bool:
    return ck.get("kind") == "patch"


class PatchInfer:
    """Патчевая модель для потока: кадр → точка камеры и сведения о голосах.

    `amp` — autocast на видеокарте. При batch 1 он не ускоряет (время уходит
    на запуск ядер, см. docs/patch-embeddings.md), поэтому по умолчанию
    выключен; включённый даёт тот же ответ в пределах долей единицы.
    """

    kind = "patch"

    def __init__(self, path: Path, device: str, amp: bool = False):
        self.device = device
        self.net = load_patchnet(Path(path), device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.units = float(ck.get("units_per_px", UNITS_PER_PX))
        self.trained_on = ck.get("trained_on", [])
        self.label = f"патч-{self.net.kind}"
        self.h0, self.store = load_projection()
        self.amp = amp and device == "cuda"
        self.grid = self.net.grid
        # Конец прямого прохода сети отмечается событием видеокарты прямо из
        # хука: так время делится на «сеть» и «разбор + сведение голосов»
        # без второй копии predict_canvases.
        self._ev = None
        self.net.register_forward_hook(self._mark)

    def _mark(self, *_):
        if self._ev is not None:
            self._ev[1].record()

    def describe(self, scene_size) -> str:
        cv, _, _ = rectify(np.zeros((scene_size[1], scene_size[0], 3), np.uint8),
                           np.eye(3), self.h0, self.units, self.net.stride)
        return f"{self.label}, вид сверху {cv.shape[1]}x{cv.shape[0]}"

    def prepare(self, scene):
        """Хранимая сцена (PIL или (H, W, 3) uint8) → холст вида сверху,
        маска следа, центр камеры на холсте."""
        a = np.asarray(scene.convert("RGB")) if hasattr(scene, "convert") else scene
        return rectify(a, np.eye(3), self.h0, self.units, self.net.stride)

    @torch.no_grad()
    def predict(self, prep) -> tuple[dict, np.ndarray, dict]:
        """Ответ по подготовленному кадру: поля строки, тепловая карта, мс."""
        cv, val, q0 = prep
        cuda = self.device == "cuda"
        t0 = time.perf_counter()
        x = torch.from_numpy(cv).to(self.device).permute(2, 0, 1)[None].float().div_(255)
        v = torch.from_numpy(val).to(self.device)[None].float().div_(255)
        q = torch.tensor(q0[None], device=self.device, dtype=torch.float32)
        if cuda:
            self._ev = (torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))
            self._ev[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            cams, agree, info = predict_canvases(self.net, x, v, q, self.units)
        cam = cams[0]
        votes = (info["pts"][0] - info["off"][0]).reshape(-1, 2).float()
        w = info["w"][0].reshape(-1).float()
        live_p = (info["valid"][0].reshape(-1) >= 0.5).float()
        heat, small = vote_stats(votes, w, torch.from_numpy(cam).to(votes), self.grid)
        sc = (info["scene"][0].reshape(-1) > 0.5).float()
        # Голова кадра «игра / не игра» есть не в каждом чекпойнте; без неё
        # кадр считается игрой, как и раньше.
        game = info["game"][0].reshape(1).float() if "game" in info else torch.ones(1, device=w.device)
        small = torch.cat([small, ((sc * live_p).sum() / live_p.sum().clamp(min=1))[None],
                           (w > 0).sum()[None].float(), game])
        # Одна выгрузка на кадр: каждая лишняя синхронизация с занятой
        # видеокартой стоит миллисекунды.
        heat, (qx, qy, sp, scene_share, n_vote, p_game) = heat.cpu().numpy(), small.cpu().tolist()
        if cuda:
            self._ev[2].record()
            torch.cuda.synchronize()
            net_ms = self._ev[0].elapsed_time(self._ev[1])
            agg_ms = self._ev[1].elapsed_time(self._ev[2])
            self._ev = None
        else:
            net_ms = (time.perf_counter() - t0) * 1000
            agg_ms = 0.0
        # На кадре «не игра» точка камеры бессмысленна: ответа нет, как и у
        # кадра без единого голоса.
        is_game = "game" not in info or p_game >= self.net.frame_threshold
        ok = bool(np.isfinite(cam).all()) and is_game
        px, py = (float(cam[0]), float(cam[1])) if ok else (float("nan"),) * 2
        row = {"px": px, "py": py, "qx": qx, "qy": qy,
               "spread": sp, "agree": float(agree[0]),
               "scene": scene_share, "votes": int(n_vote)}
        if "game" in info:
            row["game"] = p_game
            row["is_game"] = bool(is_game)
        return row, heat, {"net": net_ms, "agg": agg_ms}


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
