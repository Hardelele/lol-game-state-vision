"""Страница для глазной проверки: опознаётся ли кусочек кадра по размеру.

Вопрос, на который отвечает страница: с какого поля зрения по куску сцены
вообще можно понять, где на карте он снят. Это решает, на каком масштабе
имеет смысл учить модель узнавать рельеф.

Два раздела:

1. «Одно место, разное поле зрения» — один и тот же центр, вырезанный
   окнами 10, 20, 40, 80, 160, 320 px. Видно, на каком размере пятно травы
   превращается в узнаваемое место.
2. «Двойники» — кусочек и самый похожий на него кусочек из другого кадра,
   рядом расстояние между этими кадрами по карте. Если двойник найден на
   другом конце карты, кусочек этого размера места не определяет.

Похожесть считается так же, как в замере: оттенки серого, окно сжато до
10x10 и нормировано по яркости и контрасту, поэтому поле зрения — это
единственное, чем окна отличаются. Показываются кусочки в цвете: человеку
так судить легче, и это честнее по отношению к проверяемой идее.

Координаты камеры берутся из runs/minimap/camera_coords.csv (tools/minimap_camera.py).

Пример:
    python tools/inspect_patches.py --layout dataset/layouts/spectator-volibear-challenger.json \
        --coords runs/minimap/camera_coords.csv --out runs/inspect/patches.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mask_frames import apply_mask
from paths import DATASET_VIDEOS, INSPECT, MINIMAP
from inspect_scene import NAV_OPEN, NAV_CLOSE, b64, sync_nav, update_manifest, write_index

SCALES = (10, 20, 40, 80, 160, 320)
CELL = 150          # сторона показа кусочка на странице
MAP_UNITS = 14800   # примерный размер карты League of Legends в игровых единицах
VIEW_FRAC = 0.266   # ширина вьюпорта камеры в долях карты


def units(scale: int) -> int:
    """Во что превращается окно в экранных пикселях в игровых единицах."""
    return round(scale * VIEW_FRAC * MAP_UNITS / 1920)


def descriptor(patch: np.ndarray) -> np.ndarray:
    """То же описание, что в замере: 10x10 серого, без яркости и контраста."""
    d = np.asarray(Image.fromarray(patch).resize((10, 10), Image.BILINEAR),
                   dtype=np.float32)
    return ((d - d.mean()) / (d.std() + 1e-6)).ravel()


def load_frames(layout: dict, coords: dict, per_video: int) -> list[dict]:
    out = []
    for vid in sorted({v for v, _ in coords}):
        vd = DATASET_VIDEOS / vid
        rows = [r for r in csv.DictReader((vd / "frames.csv").open(encoding="utf-8"))
                if coords.get((vid, Path(r["frame"]).stem), (0, 0, 0))[2] >= 0.6]
        for r in rows[::max(1, len(rows) // per_video)][:per_video]:
            with Image.open(vd / r["frame"]) as im:
                rgb = apply_mask(im, layout)
            cx, cy, _ = coords[(vid, Path(r["frame"]).stem)]
            out.append({
                "video": vid, "t_sec": int(r["t_sec"]), "label": r["label"],
                "rgb": rgb, "gray": np.asarray(rgb.convert("L"), dtype=np.float32),
                "c": np.array([cx, cy]),
            })
    return out


def usable(gray: np.ndarray, x: int, y: int, s: int) -> bool:
    """Окно целиком в сцене: не задевает маску и не пустая заливка."""
    p = gray[y:y + s, x:x + s]
    return p.shape == (s, s) and (p < 8).mean() <= 0.05 and p.std() >= 6


def pick_centers(frames: list[dict], n: int, rng) -> list[tuple[int, int, int]]:
    """Точки, пригодные на всех масштабах сразу — иначе ряды несопоставимы."""
    big = max(SCALES)
    picks = []
    while len(picks) < n:
        fi = int(rng.integers(0, len(frames)))
        g = frames[fi]["gray"]
        h, w = g.shape
        y = int(rng.integers(big // 2, h - big // 2))
        x = int(rng.integers(big // 2, w - big // 2))
        if all(usable(g, x - s // 2, y - s // 2, s) for s in SCALES):
            picks.append((fi, x, y))
    return picks


def show(img: Image.Image) -> str:
    """Увеличить до единого размера без сглаживания: видно реальную детализацию."""
    return b64(img.resize((CELL, CELL), Image.NEAREST), "PNG")


def section_scales(frames: list[dict], picks: list, layout: dict) -> str:
    mx0, my0, mx1, my1 = layout["labelling"]["minimap"]
    rows = []
    for fi, x, y in picks:
        f = frames[fi]
        cells = []
        for s in SCALES:
            patch = f["rgb"].crop((x - s // 2, y - s // 2, x + s // 2, y + s // 2))
            cells.append(f'<figure><img src="{show(patch)}" alt="{s}px">'
                         f'<figcaption>{s} px<br><span>≈{units(s)} ед.</span></figcaption></figure>')
        full = f["rgb"].copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(full)
        d.rectangle([x - 160, y - 160, x + 160, y + 160], outline=(255, 90, 70), width=6)
        d.rectangle([x - 5, y - 5, x + 5, y + 5], outline=(120, 255, 160), width=4)
        rows.append(
            f'<div class="prow"><div class="meta">{html.escape(f["video"])} · '
            f't={f["t_sec"]}с · метка {html.escape(f["label"] or "—")}<br>'
            f'камера на карте {f["c"][0]:.2f}, {f["c"][1]:.2f}</div>'
            f'<figure class="ctx"><img src="{b64(full.resize((360, 203)))}" alt="кадр">'
            f'<figcaption>кадр после маски; красным — окно 320 px</figcaption></figure>'
            f'<div class="strip">{"".join(cells)}</div></div>')
    return "".join(rows)


def section_twins(frames: list[dict], rng, per_scale: int, samples: int) -> str:
    blocks = []
    for s in SCALES:
        P, meta = [], []
        for fi, f in enumerate(frames):
            g = f["gray"]
            h, w = g.shape
            got, tries = 0, 0
            while got < samples and tries < samples * 40:
                tries += 1
                y = int(rng.integers(0, h - s))
                x = int(rng.integers(0, w - s))
                if not usable(g, x, y, s):
                    continue
                P.append(descriptor(g[y:y + s, x:x + s]))
                meta.append((fi, x, y))
                got += 1
        P = np.array(P)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        S = P @ P.T
        fidx = np.array([m[0] for m in meta])
        S[fidx[:, None] == fidx[None, :]] = -9
        nn = S.argmax(1)
        dist = np.array([np.linalg.norm(frames[meta[i][0]]["c"] - frames[meta[nn[i]][0]]["c"])
                         for i in range(len(meta))])
        rnd = float(np.mean([np.linalg.norm(frames[a]["c"] - frames[b]["c"])
                             for a, b in zip(rng.integers(0, len(frames), 3000),
                                             rng.integers(0, len(frames), 3000))]))
        order = rng.permutation(len(meta))[:per_scale]

        pairs = []
        for i in order:
            j = nn[i]
            imgs = []
            for k in (i, j):
                fi, x, y = meta[k]
                imgs.append((frames[fi], frames[fi]["rgb"].crop((x, y, x + s, y + s))))
            far = dist[i] > 0.25
            pairs.append(
                f'<div class="pair {"far" if far else "near"}">'
                + "".join(
                    f'<figure><img src="{show(im)}" alt="кусочек">'
                    f'<figcaption>{html.escape(fr["video"][:7])} t={fr["t_sec"]}с<br>'
                    f'<span>карта {fr["c"][0]:.2f}, {fr["c"][1]:.2f}</span></figcaption></figure>'
                    for fr, im in imgs)
                + f'<div class="dist">по карте<br><b>{dist[i]:.2f}</b></div></div>')

        gain = (1 - dist.mean() / rnd) * 100 if rnd else 0
        blocks.append(
            f'<h3>{s} px <small>≈{units(s)} игровых единиц · '
            f'среднее расстояние до двойника {dist.mean():.3f} '
            f'против {rnd:.3f} у случайной пары · польза {gain:.1f}%</small></h3>'
            f'<div class="pairs">{"".join(pairs)}</div>')
    return "".join(blocks)


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Кусочки кадра: с какого размера видно, где это</title>
<style>
 :root {{ color-scheme: dark; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:0 16px 48px; background:#14161a; color:#e8e8ea;
        font:14px/1.5 system-ui,Segoe UI,sans-serif; }}
 .top {{ position:sticky; top:0; z-index:9; background:#14161a;
         border-bottom:1px solid #2a2e36; padding:12px 0 10px; margin-bottom:16px; }}
 h1 {{ font-size:18px; margin:0 0 4px; }}
 .sum {{ color:#9aa0aa; max-width:900px; }}
 .tabs {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }}
 .tab {{ display:block; padding:5px 10px; border:1px solid #39404c; border-radius:6px;
         color:#cfd4dc; text-decoration:none; background:#1b1f26; font-size:13px; }}
 .tab em {{ display:block; font-style:normal; color:#878d98; font-size:11px; }}
 .tab.on {{ background:#3a5bd9; border-color:#3a5bd9; color:#fff; }}
 .tab.home {{ background:#262b34; }}
 h2 {{ font-size:16px; margin:26px 0 6px; }}
 h3 {{ font-size:14px; margin:20px 0 8px; color:#cfd4dc; }}
 h3 small {{ color:#878d98; font-weight:400; }}
 .lead {{ color:#9aa0aa; max-width:900px; margin-bottom:10px; }}
 .prow {{ display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap;
          border:1px solid #2a2e36; border-radius:10px; background:#1a1d23;
          padding:12px; margin-bottom:12px; }}
 .meta {{ min-width:190px; color:#9aa0aa; font-size:12px; }}
 .strip {{ display:flex; gap:8px; flex-wrap:wrap; }}
 figure {{ margin:0; }}
 figure img {{ display:block; border-radius:5px; image-rendering:pixelated; }}
 .ctx img {{ image-rendering:auto; }}
 figcaption {{ color:#878d98; font-size:11px; margin-top:3px; text-align:center; }}
 figcaption span {{ color:#6a7080; }}
 .pairs {{ display:flex; gap:10px; flex-wrap:wrap; }}
 .pair {{ display:flex; gap:8px; align-items:center; padding:8px; border-radius:9px;
          border:1px solid #2f353f; background:#1a1d23; }}
 .pair.far {{ border-color:#b4473f; }}
 .pair.near {{ border-color:#2f6d43; }}
 .dist {{ font-size:11px; color:#9aa0aa; text-align:center; min-width:62px; }}
 .dist b {{ font-size:16px; color:#e8e8ea; display:block; }}
 .pair.far .dist b {{ color:#ff7b6e; }}
 .pair.near .dist b {{ color:#6cd08a; }}
 .note {{ color:#767d89; font-size:12px; max-width:900px; margin-top:10px;
          border-top:1px solid #2a2e36; padding-top:10px; }}
</style></head><body>
<div class="top">
  <h1>Кусочки кадра: с какого размера видно, где это</h1>
  <div class="sum">Вопрос простой: можно ли по куску сцены понять, в каком месте карты он снят.
  От ответа зависит, на каком масштабе учить модель узнавать рельеф.</div>
  {nav_open}{nav}{nav_close}
</div>

<h2>1. Одно место, разное поле зрения</h2>
<div class="lead">Один и тот же центр, вырезанный окнами разного размера и растянутый
до одного размера на экране. Смотрите, с какого окна пятно превращается в узнаваемое место.</div>
{scales}

<h2>2. Двойники</h2>
<div class="lead">Слева кусочек, справа самый похожий на него кусочек <b>из другого кадра</b>,
между ними — расстояние по карте (1.0 — это диагональ всей карты). Зелёная рамка:
двойник рядом, значит вид о месте что-то говорит. Красная: двойник нашёлся далеко,
значит по такому кусочку место не определяется.</div>
{twins}

<div class="note">Похожесть считается так же, как в замере: оттенки серого, окно сжато до 10×10
и нормировано по яркости и контрасту, поэтому поле зрения — единственное, чем окна отличаются.
Показаны кусочки в цвете: человеку так судить легче. Это поиск по сырым пикселям, а не обученное
представление — сеть из того же окна выжмет больше, поэтому «польза» в заголовках разделов
является нижней оценкой, а не потолком. Сравнивать стоит не абсолютные числа, а порядок между
размерами. Окна берутся только там, где они целиком попадают в сцену и не задевают маску.</div>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--coords", type=Path, default=MINIMAP / "camera_coords.csv")
    ap.add_argument("--out", type=Path, default=INSPECT / "patches.html")
    ap.add_argument("--frames", type=int, default=10, help="кадров на ролик")
    ap.add_argument("--rows", type=int, default=8, help="строк в первом разделе")
    ap.add_argument("--pairs", type=int, default=8, help="пар на масштаб во втором")
    ap.add_argument("--samples", type=int, default=60, help="окон на кадр для поиска двойников")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    coords = {(r["video_id"], r["frame"]): (float(r["cx"]), float(r["cy"]),
                                            float(r["quality"]))
              for r in csv.DictReader(args.coords.open(encoding="utf-8"))}
    rng = np.random.default_rng(args.seed)

    frames = load_frames(layout, coords, args.frames)
    print(f"кадров в разборе: {len(frames)}")
    picks = pick_centers(frames, args.rows, rng)
    scales_html = section_scales(frames, picks, layout)
    twins_html = section_twins(frames, rng, args.pairs, args.samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.format(
        nav_open=NAV_OPEN, nav="", nav_close=NAV_CLOSE,
        scales=scales_html, twins=twins_html), encoding="utf-8")

    pages = update_manifest(args.out.parent, {
        "file": args.out.name, "title": "Кусочки кадра: какой размер опознаётся",
        "video": "разбор", "accuracy": 0.0, "scored": 0,
        "total": len(frames), "correct": 0, "errors": 0, "strip": "",
        "kind": "note",
    })
    sync_nav(args.out.parent, pages)
    write_index(args.out.parent, pages)
    print(f"страница ({args.out.stat().st_size / 1e6:.1f} МБ) → {args.out}")


if __name__ == "__main__":
    main()
