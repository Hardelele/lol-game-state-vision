"""Загрузка размеченных кадров в тензоры: маска HUD, обрезка, масштаб.

Вход модели готовится здесь и только здесь, чтобы обучение и инференс
использовали одинаковую предобработку (требование оценки из README).

Пример:
    python tools/scene_data.py dataset/videos/58w57eJ5Qks \
        --layout dataset/layouts/spectator-volibear-challenger.json --preview runs/checks/model-input
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mask_frames import apply_mask, mask_boxes

# Порядок классов фиксирован: индекс в этом списке — это метка модели.
CLASSES = ("not_top", "top")
ABSTAIN = "unknown"
# Ширина и высота входа модели. Плотная обрезка + этот размер дают ~12
# исходных пикселей на пиксель входа; на прежних (256, 112) с обрезкой bbox
# было 53, и рельеф в таком масштабе не читался — см. docs в README.
INPUT_SIZE = (288, 256)


@dataclass
class FrameSet:
    """Кадры одного ролика, уже приведённые к входу модели."""

    video_id: str
    x: np.ndarray  # (N, 3, H, W) float32 в диапазоне [-1, 1]
    y: np.ndarray  # (N,) int64; -1 для unknown
    t_sec: np.ndarray  # (N,) int64, время кадра в ролике
    names: list[str]
    layout: str

    def labelled(self) -> np.ndarray:
        """Индексы кадров с меткой top/not_top (unknown исключён)."""
        return np.flatnonzero(self.y >= 0)


def keep_grid(size: tuple[int, int], layout: dict) -> np.ndarray:
    """Булева карта кадра: True там, где пиксель не закрыт маской."""
    w, h = size
    keep = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in mask_boxes(size, layout):
        keep[y0:y1, x0:x1] = False
    return keep


def unmasked_bbox(size: tuple[int, int], layout: dict) -> tuple[int, int, int, int]:
    """Границы области, где хоть один пиксель не закрыт маской.

    Обрезка одинакова для всех кадров ролика, поэтому не может служить
    подсказкой о фазе матча. Внутри области остаются закрытые прямоугольники
    (чат, объявления, лента убийств) — они постоянны и информации не несут.
    """
    keep = keep_grid(size, layout)
    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Маска закрывает кадр целиком — нечего подавать модели")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def dense_bbox(size: tuple[int, int], layout: dict, rel: float = 0.6
               ) -> tuple[int, int, int, int]:
    """Границы плотной середины кадра: строки и столбцы, открытые не хуже,
    чем `rel` от самой открытой строки/столбца этой раскладки.

    Обрезка по `unmasked_bbox` сохраняет всю ширину кадра: достаточно пары
    открытых уголков над боковыми панелями, чтобы строка попала в границы.
    В итоге большую часть входа занимают постоянные чёрные поля, а рельеф
    сжимается в несколько раз сильнее нужного. Порог берётся относительным,
    потому что абсолютная доля открытых пикселей зависит от раскладки:
    при плотном HUD открытых строк выше 60% может не быть вообще.
    """
    keep = keep_grid(size, layout)
    rows_p, cols_p = keep.mean(axis=1), keep.mean(axis=0)
    rows = np.flatnonzero(rows_p >= rel * rows_p.max())
    cols = np.flatnonzero(cols_p >= rel * cols_p.max())
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Маска не оставляет плотной области")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_box(size: tuple[int, int], layout: dict, mode: str = "dense"
             ) -> tuple[int, int, int, int]:
    if mode == "bbox":
        return unmasked_bbox(size, layout)
    if mode == "dense":
        return dense_bbox(size, layout)
    if mode == "none":
        return (0, 0, size[0], size[1])
    raise ValueError(f"неизвестный режим обрезки {mode!r}")


def prepare_frame(
    img: Image.Image, layout: dict, size: tuple[int, int] = INPUT_SIZE,
    crop: str = "dense"
) -> Image.Image:
    """Маска → обрезка по незакрытой области → масштаб ко входу модели."""
    masked = apply_mask(img, layout)
    return masked.crop(crop_box(masked.size, layout, crop)).resize(size, Image.BILINEAR)


def to_tensor(img: Image.Image) -> np.ndarray:
    """(H, W, 3) uint8 → (3, H, W) float32 в [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)) * 2.0 - 1.0


def load_video(
    video_dir: Path, layout_path: Path | None = None, size: tuple[int, int] = INPUT_SIZE,
    crop: str = "dense"
) -> FrameSet:
    """Собрать кадры ролика по frames.csv. Раскладка берётся из layout или source.json."""
    video_dir = Path(video_dir)
    rows = list(csv.DictReader((video_dir / "frames.csv").open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{video_dir}/frames.csv пуст")

    if layout_path is None:
        source = json.loads((video_dir / "source.json").read_text(encoding="utf-8"))
        name = source.get("layout")
        if not name:
            raise ValueError(
                f"{video_dir}/source.json не содержит поле layout — укажите --layout"
            )
        layout_path = video_dir.parents[1] / "layouts" / f"{name}.json"
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))

    xs, ys, ts, names = [], [], [], []
    for row in rows:
        label = row["label"].strip()
        if label == ABSTAIN:
            y = -1
        elif label in CLASSES:
            y = CLASSES.index(label)
        else:
            raise ValueError(f"Неизвестная метка {label!r} в {row['frame']}")
        with Image.open(video_dir / row["frame"]) as img:
            xs.append(to_tensor(prepare_frame(img, layout, size, crop)))
        ys.append(y)
        ts.append(int(row["t_sec"]))
        names.append(Path(row["frame"]).stem)

    order = np.argsort(np.array(ts))  # кадры по времени: нужно для блочных фолдов
    return FrameSet(
        video_id=rows[0]["video_id"],
        x=np.stack(xs)[order],
        y=np.array(ys, dtype=np.int64)[order],
        t_sec=np.array(ts, dtype=np.int64)[order],
        names=[names[i] for i in order],
        layout=layout["name"],
    )


def time_block_folds(t_sec: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    """Разбить кадры на n_folds подряд идущих отрезков времени.

    Блоки, а не случайное разбиение — не из-за похожести соседних кадров:
    при шаге 15 с камера успевает уйти, и корреляция соседних кадров всего
    0.059 против -0.010 у случайных пар. Дело в другом: доля `top` сильно
    меняется по ходу матча (0.88 в начале, 0.06 в конце). Случайный сплит
    дал бы в обучении и валидации одинаковую долю классов и скрыл бы
    неустойчивость модели к этому сдвигу, которая на практике её и ломает.
    Это всё равно не измерение переноса на другие матчи.
    """
    if n_folds < 2:
        raise ValueError("нужно минимум 2 фолда")
    idx = np.arange(len(t_sec))
    return [np.sort(part) for part in np.array_split(idx, n_folds)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--preview", type=Path, default=None,
                    help="сохранить готовый вход модели как PNG для визуальной проверки")
    args = ap.parse_args()

    data = load_video(args.video_dir, args.layout)
    counts = {c: int((data.y == i).sum()) for i, c in enumerate(CLASSES)}
    counts[ABSTAIN] = int((data.y < 0).sum())
    print(f"ролик {data.video_id}, раскладка {data.layout}")
    print(f"вход модели {data.x.shape}, {data.x.dtype}, "
          f"диапазон [{data.x.min():.2f}, {data.x.max():.2f}]")
    print("метки:", counts)
    for i, fold in enumerate(time_block_folds(data.t_sec)):
        lab = fold[data.y[fold] >= 0]
        print(f"  фолд {i}: {len(fold)} кадров, {data.t_sec[fold[0]]}–"
              f"{data.t_sec[fold[-1]]} с, с метками {len(lab)}")

    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
        for i in range(len(data.names)):
            arr = ((data.x[i].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255)
            Image.fromarray(arr.astype(np.uint8)).save(
                args.preview / f"{data.names[i]}.png")
        print(f"{len(data.names)} превью → {args.preview}")


if __name__ == "__main__":
    main()
