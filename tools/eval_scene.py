"""Проверка обученной модели на отложенных роликах.

Ролики для проверки должны быть из других матчей, чем обучающие: валидация
внутри матча меряет устойчивость к сдвигу по ходу игры, но не перенос на
другой матч, канал и чемпиона. Скрипт отказывается считать ролик отложенным,
если он указан в `trained_on` чекпоинта.

Пример:
    python tools/eval_scene.py runs/scene/cnn-dense/model.pt \
        dataset/videos/olmTXkkUv58 dataset/videos/zJvTSjEnKNE dataset/videos/ibUVbSX7ARU \
        --layout dataset/layouts/spectator-volibear-challenger.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scene_data import CLASSES, load_video
from scene_model import SceneClassifier
from train_scene import abstention_table, metrics


def load_model(path: Path) -> tuple[SceneClassifier, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if list(ckpt["classes"]) != list(CLASSES):
        raise ValueError(f"классы чекпоинта {ckpt['classes']} не совпадают с {CLASSES}")
    model = SceneClassifier(num_classes=len(CLASSES), embed_dim=ckpt["embed_dim"])
    model.encoder.load_state_dict(ckpt["encoder"])
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    return model, ckpt


def evaluate(model: SceneClassifier, data, train_share_top: float | None) -> dict:
    x = torch.from_numpy(data.x)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).numpy()
    lab = data.labelled()
    pred = probs.argmax(axis=1)
    out = metrics(data.y[lab], pred[lab])
    out["video"] = data.video_id
    out["unknown"] = int((data.y < 0).sum())
    out["top_share"] = float((data.y[lab] == CLASSES.index("top")).mean())
    # Базовый уровень «всегда самый частый класс обучающего матча».
    if train_share_top is not None:
        const = CLASSES.index("top") if train_share_top >= 0.5 else CLASSES.index("not_top")
        out["majority"] = metrics(data.y[lab], np.full(len(lab), const))
    out["abstention"] = abstention_table(probs, data.y)
    return out, probs, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("video_dirs", type=Path, nargs="+")
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="по умолчанию папка прогона, где лежит чекпоинт")
    ap.add_argument("--train-video", type=Path, default=None,
                    help="обучающий ролик: нужен только чтобы взять долю top для базового уровня")
    args = ap.parse_args()

    args.out = args.out or args.checkpoint.parent
    model, ckpt = load_model(args.checkpoint)
    trained_on = set(ckpt.get("trained_on", []))
    # Предобработку берём из чекпоинта, а не из умолчаний: модель, обученная
    # на другом размере входа или другой обрезке, на чужой предобработке
    # выдаст правдоподобные, но неверные числа.
    size = tuple(ckpt["input_size"])
    # Чекпоинты до появления поля crop обучались на обрезке bbox.
    crop = ckpt.get("crop", "bbox")
    if "crop" not in ckpt:
        print("в чекпоинте нет поля crop (старый формат) — считаю обрезку bbox")
    print(f"чекпоинт обучен на: {', '.join(sorted(trained_on)) or '—'}; "
          f"вход {size}, обрезка {crop}, раскладка {ckpt.get('layout')}")

    share_top = None
    if args.train_video:
        tr = load_video(args.train_video, args.layout, size, crop)
        lab = tr.labelled()
        share_top = float((tr.y[lab] == CLASSES.index("top")).mean())
        print(f"доля top в обучающем матче: {share_top:.3f}")

    rows, all_true, all_pred = [], [], []
    preds_csv = []
    for vdir in args.video_dirs:
        data = load_video(vdir, args.layout, size, crop)
        if data.video_id in trained_on:
            raise SystemExit(
                f"{data.video_id} есть в trained_on — это не отложенный ролик, "
                "оценка по нему бессмысленна")
        res, probs, pred = evaluate(model, data, share_top)
        rows.append(res)
        lab = data.labelled()
        all_true.append(data.y[lab])
        all_pred.append(pred[lab])
        for i in range(len(data.names)):
            preds_csv.append({
                "video_id": data.video_id, "frame": data.names[i],
                "t_sec": int(data.t_sec[i]),
                "label": CLASSES[data.y[i]] if data.y[i] >= 0 else "unknown",
                "pred": CLASSES[pred[i]],
                "p_top": round(float(probs[i, CLASSES.index("top")]), 4),
                "correct": bool(data.y[i] >= 0 and pred[i] == data.y[i]),
            })

        base = f", базовый {res['majority']['accuracy']:.3f}" if "majority" in res else ""
        print(f"\n{data.video_id}: кадров с меткой {res['n']} "
              f"(top {res['top_share']:.3f}), unknown {res['unknown']}")
        print(f"  acc {res['accuracy']:.3f}{base}, bal_acc {res['balanced_accuracy']:.3f}, "
              f"top P/R/F1 {res['top_precision']:.3f}/{res['top_recall']:.3f}/"
              f"{res['top_f1']:.3f}")
        print(f"  матрица [истина][предсказание]: {res['confusion']}")

    pooled = metrics(np.concatenate(all_true), np.concatenate(all_pred))
    print(f"\nвсе отложенные ролики вместе: acc {pooled['accuracy']:.3f}, "
          f"bal_acc {pooled['balanced_accuracy']:.3f}, "
          f"top P/R/F1 {pooled['top_precision']:.3f}/{pooled['top_recall']:.3f}/"
          f"{pooled['top_f1']:.3f}")
    print(f"матрица [истина][предсказание]: {pooled['confusion']}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "trained_on": sorted(trained_on),
                    "videos": rows, "pooled": pooled}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (args.out / "holdout_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds_csv[0]))
        w.writeheader()
        w.writerows(preds_csv)
    print(f"\nотчёт и предсказания по кадрам → {args.out}")


if __name__ == "__main__":
    main()
