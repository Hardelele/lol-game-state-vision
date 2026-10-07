"""Обучение базовой CNN «камера показывает топ / не топ» на кадрах одного матча.

Это проверка сквозного тракта (кадры → маска → тензор → CNN → метрики),
а не измерение переноса на другие матчи: все кадры из одного ролика, одна
раскладка, один чемпион. Честные цифры появятся, когда в датасете будет
несколько матчей и сплит пойдёт по роликам.

Пример:
    python tools/train_scene.py dataset/videos/58w57eJ5Qks \
        --layout dataset/layouts/spectator-volibear-challenger.json --out runs/scene/cnn-dense
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scene_data import CLASSES, INPUT_SIZE, FrameSet, load_video, time_block_folds
from paths import SCENE_RUNS, TRAIN_LOG
from runlog import tee_stdout
from scene_model import SceneClassifier, count_params


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Сдвиг, масштаб и яркость/контраст. Без отражений.

    Горизонтальное отражение здесь запрещено: карта League of Legends не
    симметрична относительно вертикали, отражённый топ выглядит как бот.
    Такая аугментация ломала бы метку.
    """
    n = x.shape[0]

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, generator=gen) * (hi - lo) + lo

    # Сдвиг до ±4% кадра и масштаб ±5% — одной афинной сеткой.
    scale = u(0.95, 1.05)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = u(-0.04, 0.04)
    theta[:, 1, 2] = u(-0.04, 0.04)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # padding_mode="zeros": вылезшие края становятся чёрными, как закрытые маской.
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    bright = u(-0.15, 0.15).view(n, 1, 1, 1)
    contrast = u(0.85, 1.15).view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * contrast + mean + bright).clamp(-1.0, 1.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Точность, сбалансированная точность и precision/recall/F1 для класса top."""
    top = CLASSES.index("top")
    cm = np.zeros((2, 2), dtype=int)  # [истина, предсказание]
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = int(cm[top, top])
    fp = int(cm[1 - top, top])
    fn = int(cm[top, 1 - top])
    recalls = [cm[c, c] / cm[c].sum() for c in range(2) if cm[c].sum()]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "top_precision": float(prec),
        "top_recall": float(rec),
        "top_f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "confusion": cm.tolist(),
    }


def train_once(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    seed: int,
    x_va: torch.Tensor | None = None,
    y_va: torch.Tensor | None = None,
    log: bool = False,
    use_augment: bool = True,
    dropout: float = 0.3,
) -> tuple[SceneClassifier, list[dict]]:
    """Обучить классификатор; вернуть модель и историю по эпохам."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = SceneClassifier(num_classes=len(CLASSES), dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Веса классов: кадров top больше, чем not_top; без весов модель смещается к top.
    counts = torch.bincount(y_tr, minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts.clamp(min=1))

    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        total = 0.0
        for i in range(0, len(perm), batch):
            sel = perm[i : i + batch]
            xb = augment(x_tr[sel], gen) if use_augment else x_tr[sel]
            loss = F.cross_entropy(model(xb), y_tr[sel], weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()

        row = {"epoch": epoch + 1, "train_loss": total / len(x_tr)}
        with torch.no_grad():
            model.eval()
            row["train_acc"] = float((model(x_tr).argmax(1) == y_tr).float().mean())
            if x_va is not None and len(x_va):
                row["val_acc"] = float((model(x_va).argmax(1) == y_va).float().mean())
        history.append(row)
        if log and (epoch + 1) % 10 == 0:
            extra = f" val_acc {row['val_acc']:.3f}" if "val_acc" in row else ""
            print(f"    эпоха {epoch + 1:3d}  loss {row['train_loss']:.4f}"
                  f"  train_acc {row['train_acc']:.3f}{extra}")
    return model, history


def majority_baseline(y_tr: np.ndarray, y_va: np.ndarray) -> dict:
    """Тривиальный ориентир: всегда предсказывать самый частый класс обучения.

    Без него accuracy не читается: доля top внутри матча сильно плывёт по
    времени, и на отдельном блоке «всегда top» может дать за 0.9.
    """
    const = int(np.bincount(y_tr, minlength=len(CLASSES)).argmax())
    return metrics(y_va, np.full(len(y_va), const))


def cross_validate(data: FrameSet, args) -> dict:
    """Кросс-валидация по подряд идущим блокам времени внутри матча."""
    lab = data.labelled()
    x_all = torch.from_numpy(data.x)
    y_all = torch.from_numpy(data.y)
    folds = time_block_folds(data.t_sec, args.folds)

    per_fold, pooled_true, pooled_pred, base_pred = [], [], [], []
    # Предсказание для каждого кадра моделью, которая его не видела.
    # Только по ним можно разбирать ошибки: финальная модель эти кадры учила.
    oof = {}
    for k, fold in enumerate(folds):
        va = np.array([i for i in fold if data.y[i] >= 0])
        tr = np.array([i for i in lab if i not in set(fold.tolist())])
        if len(va) == 0 or len(tr) == 0:
            print(f"  фолд {k}: пропущен (нет размеченных кадров)")
            continue
        print(f"  фолд {k}: обучение {len(tr)}, валидация {len(va)} "
              f"({data.t_sec[va[0]]}–{data.t_sec[va[-1]]} с)")
        model, hist = train_once(
            x_all[tr], y_all[tr], args.epochs, args.batch, args.lr,
            args.weight_decay, args.seed + k, x_all[va], y_all[va], log=args.verbose,
            use_augment=not args.no_augment, dropout=args.dropout)
        with torch.no_grad():
            model.eval()
            logits = model(x_all[va])
            probs_va = F.softmax(logits, dim=1).numpy()
            pred = probs_va.argmax(axis=1)
        for j, idx in enumerate(va):
            oof[int(idx)] = {"fold": k, "pred": int(pred[j]),
                             "p_top": float(probs_va[j, CLASSES.index("top")])}
        m = metrics(data.y[va], pred)
        m["fold"] = k
        m["window_sec"] = [int(data.t_sec[va[0]]), int(data.t_sec[va[-1]])]
        m["final_train_acc"] = hist[-1]["train_acc"]
        base = majority_baseline(data.y[tr], data.y[va])
        m["majority_accuracy"] = base["accuracy"]
        m["top_share"] = float((data.y[va] == CLASSES.index("top")).mean())
        per_fold.append(m)
        pooled_true.append(data.y[va])
        pooled_pred.append(pred)
        const = int(np.bincount(data.y[tr], minlength=len(CLASSES)).argmax())
        base_pred.append(np.full(len(va), const))
        print(f"    итог: acc {m['accuracy']:.3f} (базовый уровень "
              f"{m['majority_accuracy']:.3f}, доля top {m['top_share']:.3f}), "
              f"bal_acc {m['balanced_accuracy']:.3f}, train_acc {m['final_train_acc']:.3f}")

    accs = [m["accuracy"] for m in per_fold]
    y_pooled = np.concatenate(pooled_true)
    pooled = metrics(y_pooled, np.concatenate(pooled_pred))
    return {
        "folds": per_fold,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "majority_accuracy_mean": float(np.mean([m["majority_accuracy"] for m in per_fold])),
        "pooled": pooled,
        "pooled_majority": metrics(y_pooled, np.concatenate(base_pred)),
        "oof": oof,
    }


def abstention_table(
    probs: np.ndarray, y: np.ndarray, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)
) -> list[dict]:
    """Доля ответов и точность на них при пороге уверенности.

    Считается по всем кадрам, включая unknown: от честной модели ждём, что
    неопределимые кадры она отбрасывает, а не угадывает.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    out = []
    for t in thresholds:
        keep = conf >= t
        kept_lab = keep & (y >= 0)
        unknown = y < 0
        out.append({
            "threshold": t,
            "coverage": float(keep.mean()),
            "accuracy_on_covered": (
                float((pred[kept_lab] == y[kept_lab]).mean()) if kept_lab.any() else None
            ),
            "unknown_answered": int((keep & unknown).sum()),
            "unknown_total": int(unknown.sum()),
        })
    return out


def benchmark(model: SceneClassifier, x: torch.Tensor) -> dict:
    """Время инференса на кадр и пропускная способность."""
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x[:1])
        t0 = time.perf_counter()
        for i in range(20):
            model(x[i % len(x) : i % len(x) + 1])
        single = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(5):
            model(x[:16])
        batched = (time.perf_counter() - t0) / (5 * 16)
    return {
        "ms_per_frame_batch1": round(single * 1000, 2),
        "ms_per_frame_batch16": round(batched * 1000, 2),
        "fps_batch16": round(1 / batched, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=SCENE_RUNS / "cnn-dense")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-augment", action="store_true",
                    help="без аугментации: проверка, что модель вообще способна "
                         "выучить обучающую выборку наизусть")
    ap.add_argument("--crop", choices=("bbox", "dense", "none"), default="dense",
                    help="как обрезать кадр после маски перед масштабом")
    ap.add_argument("--size", default="x".join(map(str, INPUT_SIZE)),
                    help="вход модели, ШИРИНАxВЫСОТА")
    ap.add_argument("--threads", type=int, default=0, help="0 — оставить как есть")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    with tee_stdout(args.out / TRAIN_LOG):
        run(args)


def run(args: argparse.Namespace) -> None:

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(False)

    size = tuple(int(v) for v in args.size.lower().split("x"))
    data = load_video(args.video_dir, args.layout, size, args.crop)
    lab = data.labelled()
    counts = {c: int((data.y == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"ролик {data.video_id}, раскладка {data.layout}")
    print(f"вход {tuple(data.x.shape[1:])}, обрезка {args.crop}, кадров {len(data.y)}: "
          f"{counts}, unknown {int((data.y < 0).sum())}")

    probe = SceneClassifier(num_classes=len(CLASSES))
    print(f"параметров: энкодер {count_params(probe.encoder):,}, "
          f"голова {count_params(probe.head):,}, всего {count_params(probe):,}")

    print(f"\nкросс-валидация по {args.folds} блокам времени:")
    cv = cross_validate(data, args)
    print(f"среднее по фолдам: acc {cv['accuracy_mean']:.3f} "
          f"± {cv['accuracy_std']:.3f} против базового уровня "
          f"{cv['majority_accuracy_mean']:.3f}")
    print(f"объединённо: acc {cv['pooled']['accuracy']:.3f} "
          f"(базовый {cv['pooled_majority']['accuracy']:.3f}), bal_acc "
          f"{cv['pooled']['balanced_accuracy']:.3f} "
          f"(базовый {cv['pooled_majority']['balanced_accuracy']:.3f}), "
          f"top P/R {cv['pooled']['top_precision']:.3f}/"
          f"{cv['pooled']['top_recall']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    oof = cv.pop("oof")
    with (args.out / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frame", "t_sec", "label",
                                          "pred", "p_top", "correct", "fold"])
        w.writeheader()
        for idx in sorted(oof):
            rec = oof[idx]
            w.writerow({
                "video_id": data.video_id, "frame": data.names[idx],
                "t_sec": int(data.t_sec[idx]),
                "label": CLASSES[data.y[idx]],
                "pred": CLASSES[rec["pred"]], "p_top": round(rec["p_top"], 4),
                "correct": bool(rec["pred"] == data.y[idx]), "fold": rec["fold"],
            })
    print(f"предсказания вне своего фолда → {args.out / 'oof_predictions.csv'}")

    print("\nфинальная модель на всех размеченных кадрах:")
    x_all = torch.from_numpy(data.x)
    y_all = torch.from_numpy(data.y)
    model, history = train_once(
        x_all[lab], y_all[lab], args.epochs, args.batch, args.lr,
        args.weight_decay, args.seed, log=True,
        use_augment=not args.no_augment, dropout=args.dropout)

    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(x_all), dim=1).numpy()
        emb = model.embed(x_all).numpy()
    fit = metrics(data.y[lab], probs[lab].argmax(axis=1))
    print(f"  посадка на обучающие кадры: acc {fit['accuracy']:.3f} "
          f"(проверка, что модель вообще способна выучить задачу)")

    abst = abstention_table(probs, data.y)
    print("\nотказ от ответа по порогу уверенности (все 94 кадра):")
    for row in abst:
        acc = "—" if row["accuracy_on_covered"] is None else f"{row['accuracy_on_covered']:.3f}"
        print(f"  порог {row['threshold']:.1f}: покрытие {row['coverage']:.3f}, "
              f"acc {acc}, отвечено unknown "
              f"{row['unknown_answered']}/{row['unknown_total']}")

    bench = benchmark(model, x_all)
    print(f"\nинференс: {bench['ms_per_frame_batch1']} мс/кадр (по одному), "
          f"{bench['ms_per_frame_batch16']} мс/кадр батчем 16 "
          f"({bench['fps_batch16']} кадр/с)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": model.encoder.state_dict(),
        "head": model.head.state_dict(),
        "classes": list(CLASSES),
        "input_size": list(size),
        "crop": args.crop,
        "layout": data.layout,
        "trained_on": [data.video_id],
        "embed_dim": model.encoder.embed_dim,
    }, args.out / "model.pt")
    np.savez(args.out / "embeddings.npz", embeddings=emb, t_sec=data.t_sec,
             label=data.y, names=np.array(data.names), probs=probs)
    report = {
        "video": data.video_id,
        "layout": data.layout,
        "input_size": list(size),
        "crop": args.crop,
        "label_counts": {**counts, "unknown": int((data.y < 0).sum())},
        "params": {"encoder": count_params(model.encoder),
                   "head": count_params(model.head),
                   "total": count_params(model)},
        "hyperparams": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                        "weight_decay": args.weight_decay, "folds": args.folds,
                        "seed": args.seed, "dropout": args.dropout,
                        "augment": not args.no_augment},
        "cross_validation": cv,
        "train_fit": fit,
        "abstention": abst,
        "inference": bench,
        "history_final_model": history,
        "caveat": "Все кадры из одного матча: это проверка тракта, а не оценка "
                  "переноса. Для честных цифр нужен сплит по роликам.",
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nчекпоинт, эмбеддинги и отчёт → {args.out}")


if __name__ == "__main__":
    main()
