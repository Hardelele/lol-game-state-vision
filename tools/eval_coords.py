"""Проверка модели координат по роликам, с контрольными опытами.

Кроме обычных метрик делает два контроля, без которых хорошему результату
верить нельзя:

* чёрный вход — модель обязана скатиться к средней точке обучения. Если
  она и на пустом кадре «угадывает» разные места, сигнал приходит не из
  картинки, а откуда-то ещё;
* перемешанные пары кадр/координата — ошибка обязана подняться до
  базового уровня. Если нет, метрика считается неправильно.

Попутно сохраняет предсказания по кадрам и тепловые карты — для смотрелки.

Пример:
    python tools/eval_coords.py runs/coords/heatmap/model.pt olmTXkkUv58 zJvTSjEnKNE
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paths import COORDS_DATA
from coord_model import CoordNet, GRID, expected_point, peak_point, spread
from train_coords import Scenes, load_rows, hand_labels, in_top, MAP_UNITS


def load_model(path: Path, dev: str):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = CoordNet(grid=ck.get("grid", GRID)).to(dev)
    m.encoder.load_state_dict(ck["encoder"])
    m.head.load_state_dict(ck["head"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, rows, root: Path, size, dev: str, workers: int):
    dl = DataLoader(Scenes(rows, root, size), batch_size=64, shuffle=False,
                    num_workers=workers, pin_memory=True)
    pk, ex, sp, hm = [], [], [], []
    for x, _, _ in dl:
        lo = model(x.to(dev, non_blocking=True)).float()
        pk.append(peak_point(lo, GRID).cpu())
        ex.append(expected_point(lo, GRID).cpu())
        sp.append(spread(lo, GRID).cpu())
        hm.append(F.softmax(lo, 1).cpu().reshape(-1, GRID, GRID).half())
    return (torch.cat(pk).numpy(), torch.cat(ex).numpy(),
            torch.cat(sp).numpy(), torch.cat(hm).numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--root", type=Path, default=COORDS_DATA)
    ap.add_argument("--min-quality", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.checkpoint, dev)
    size = tuple(ck["input_size"])
    out_dir = args.out or args.checkpoint.parent
    trained = set(ck.get("trained_on", []))
    print(f"чекпоинт обучен на: {', '.join(sorted(trained))}; вход {size}")

    hand = hand_labels(args.videos)
    summary = {}
    for vid in args.videos:
        tag = " (ОБУЧАЮЩИЙ — не отложенный)" if vid in trained else ""
        rows = load_rows([vid], args.root, args.min_quality)
        pk, ex, sp, hm = predict(model, rows, args.root, size, dev, args.workers)
        true = np.array([[float(r["cx"]), float(r["cy"])] for r in rows])
        d = np.linalg.norm(ex - true, axis=1)

        res = {"n": len(rows), "median": float(np.median(d)),
               "mean": float(d.mean()), "p90": float(np.percentile(d, 90)),
               "median_units": float(np.median(d) * MAP_UNITS),
               "within_viewport": float((d < 0.133).mean())}
        idx = [i for i, r in enumerate(rows)
               if (vid, int(round(float(r["t_sec"])))) in hand]
        if idx:
            y = np.array([hand[(vid, int(round(float(rows[i]["t_sec"]))))] == "top"
                          for i in idx])
            p = in_top(pk[idx, 0], pk[idx, 1])
            rec = [float(((p == y) & (y == c)).sum() / max((y == c).sum(), 1))
                   for c in (False, True)]
            res["vs_hand"] = {"n": len(idx), "top": int(y.sum()),
                              "accuracy": float((p == y).mean()),
                              "balanced_accuracy": float(np.mean(rec))}
        summary[vid] = res

        print(f"\n{vid}{tag}: кадров {res['n']}")
        print(f"  ошибка: медиана {res['median']:.3f} (~{res['median_units']:.0f} ед.), "
              f"среднее {res['mean']:.3f}, 90-й перцентиль {res['p90']:.3f}")
        print(f"  в пределах половины вьюпорта: {res['within_viewport'] * 100:.1f}% кадров")
        if "vs_hand" in res:
            v = res["vs_hand"]
            print(f"  против ручных меток ({v['n']} шт, из них top {v['top']}): "
                  f"acc {v['accuracy']:.3f}, bal {v['balanced_accuracy']:.3f}")

        with (out_dir / f"predictions_{vid}.csv").open("w", encoding="utf-8",
                                                       newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "video_id", "idx", "t_sec", "cx_true", "cy_true", "cx_pred",
                "cy_pred", "cx_peak", "cy_peak", "err", "spread", "quality",
                "scene", "mini"])
            w.writeheader()
            for i, r in enumerate(rows):
                w.writerow({
                    "video_id": vid, "idx": r["idx"], "t_sec": r["t_sec"],
                    "cx_true": r["cx"], "cy_true": r["cy"],
                    "cx_pred": round(float(ex[i, 0]), 4),
                    "cy_pred": round(float(ex[i, 1]), 4),
                    "cx_peak": round(float(pk[i, 0]), 4),
                    "cy_peak": round(float(pk[i, 1]), 4),
                    "err": round(float(d[i]), 4),
                    "spread": round(float(sp[i]), 4),
                    "quality": r["quality"], "scene": r["scene"], "mini": r["mini"]})
        np.savez_compressed(out_dir / f"heatmaps_{vid}.npz", heatmaps=hm,
                            idx=np.array([int(r["idx"]) for r in rows]))

    # Контроль 1: пустой вход.
    with torch.no_grad():
        lo = model(torch.zeros(8, 3, size[1], size[0], device=dev)).float()
        e = expected_point(lo, GRID).cpu().numpy()
    tr_rows = load_rows(sorted(trained), args.root, args.min_quality) if trained else []
    mp = (np.array([np.mean([float(r["cx"]) for r in tr_rows]),
                    np.mean([float(r["cy"]) for r in tr_rows])])
          if tr_rows else np.array([np.nan, np.nan]))
    print(f"\nконтроль 1, чёрный вход: ответ ({e[0, 0]:.3f}, {e[0, 1]:.3f}), "
          f"разброс по батчу {float(e.std(0).max()):.4f}")
    print(f"   средняя точка обучения ({mp[0]:.3f}, {mp[1]:.3f}); "
          f"расхождение {float(np.linalg.norm(e[0] - mp)):.3f} — "
          "на пустом кадре модель и должна отвечать примерно ею")

    # Контроль 2: перемешанные пары кадр/координата.
    last_rows, last_ex = rows, ex
    true = np.array([[float(r["cx"]), float(r["cy"])] for r in last_rows])
    perm = np.random.default_rng(0).permutation(len(true))
    d_sh = np.linalg.norm(last_ex - true[perm], axis=1)
    print(f"контроль 2, перемешанные пары на {last_rows[0]['video_id']}: "
          f"медиана {np.median(d_sh):.3f} против {np.median(np.linalg.norm(last_ex - true, axis=1)):.3f} "
          "на правильных — должна быть заметно хуже")

    (out_dir / "eval_coords.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "trained_on": sorted(trained),
                    "videos": summary,
                    "control_black_input": [float(e[0, 0]), float(e[0, 1])],
                    "train_mean_point": [float(mp[0]), float(mp[1])]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nпредсказания, тепловые карты и сводка → {out_dir}")


if __name__ == "__main__":
    main()
