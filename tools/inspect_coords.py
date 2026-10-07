"""Смотрелка предсказаний координат: сцена, карта, ответ модели.

Для каждого кадра показывает:

1. вход модели — обрезанную сцену без HUD и без миникарты, разбитую на
   ячейки сеткой;
2. карту: фон — миникарта этого же кадра, поверх неё красным распределение
   вероятности, белым прямоугольником область, которую камера показывает на
   самом деле. Внутри прямоугольника — та же сетка, ячейка в ячейку.

Главное, что из этого видно: ячейка 2.3 на картинке — это вот этот кусок
карты. Наведение на ячейку подсвечивает её сразу на обоих изображениях и
показывает её координаты на карте. Так понятно, какую область считает
каждый квадратик и почему по одному квадратику место не определяется.

Отметки — элементы SVG, а не запечённые пиксели, поэтому наведение на строку
с числами подсвечивает нужную отметку. Таймкод — ссылка на этот момент
в ролике на YouTube.

Пример:
    python tools/inspect_coords.py zJvTSjEnKNE --run runs/coords --cells 8x5
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inspect_scene import NAV_OPEN, NAV_CLOSE, b64, sync_nav, update_manifest, write_index

MAP_UNITS = 14800
# Видимая область берётся из калиброванной проекции (tools/calibrate_projection.py):
# камера наклонена, поэтому на карте она покрывает трапецию, а не прямоугольник,
# и масштаб меняется примерно на 30% от верха кадра к низу. Прямоугольник,
# который рисует сама игра на миникарте, этой трапеции соответствует лишь грубо.
PROJECTION = Path("dataset/layouts/projection.json")
VIEW_W, VIEW_H = 0.25, 0.166   # запасной вариант, если проекции ещё нет
SCENE_W = 460
MAP_W = 320
GOOD, OKAY = 0.05, VIEW_W / 2


def load_projection():
    """Гомография «пиксель хранимой сцены → смещение от центра камеры»."""
    if not PROJECTION.exists():
        return None
    d = json.loads(PROJECTION.read_text(encoding="utf-8"))
    return np.array(d["homography"], float), tuple(d["scene_size"])


def scene_to_map(Hm, pts: np.ndarray) -> np.ndarray:
    v = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return v[:, :2] / v[:, 2:3]


def cell_polys(Hm, size, cols: int, rows: int):
    """Контуры ячеек сетки, перенесённые на карту с учётом перспективы."""
    W, H = size
    out = []
    for r in range(rows):
        for c in range(cols):
            xs = [c * W / cols, (c + 1) * W / cols]
            ys = [r * H / rows, (r + 1) * H / rows]
            quad = np.array([[xs[0], ys[0]], [xs[1], ys[0]],
                             [xs[1], ys[1]], [xs[0], ys[1]]], float)
            out.append((f"{r + 1}.{c + 1}", scene_to_map(Hm, quad)))
    return out


def heat_png(heat: np.ndarray) -> str:
    """Распределение ответа отдельным полупрозрачным слоем поверх карты."""
    h = heat / max(heat.max(), 1e-9)
    big = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(
        (MAP_W, MAP_W), Image.BILINEAR), dtype=np.float32) / 255.0
    rgba = np.zeros((MAP_W, MAP_W, 4), np.uint8)
    rgba[..., 0] = np.clip(255 * big ** 0.6, 0, 255)
    rgba[..., 1] = np.clip(90 * big ** 1.6, 0, 255)
    rgba[..., 2] = np.clip(40 * big ** 2.0, 0, 255)
    rgba[..., 3] = np.clip(225 * big ** 0.8, 0, 255)
    return b64(Image.fromarray(rgba, "RGBA"), "PNG")


def cell_rects(x0, y0, w, h, cols, rows, cls, with_label=False, label_px=9):
    """Сетка прозрачных прямоугольников с общими номерами ячеек.

    Номера совпадают на сцене и на карте — по ним и связываются подсветки.
    """
    out = []
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            cid = f"{r + 1}.{c + 1}"
            x, y = x0 + c * cw, y0 + r * ch
            out.append(f'<rect class="{cls}" data-cell="{cid}" x="{x:.1f}" '
                       f'y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}"/>')
            if with_label:
                out.append(f'<text class="clab" data-cell="{cid}" '
                           f'x="{x + 3:.1f}" y="{y + label_px + 1:.1f}" '
                           f'font-size="{label_px}">{cid}</text>')
    return "".join(out)


def scene_svg(img_b64: str, w: int, h: int, cols: int, rows: int) -> str:
    return (f'<svg class="scene" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<image href="{img_b64}" x="0" y="0" width="{w}" height="{h}"/>'
            f'<g class="cells">{cell_rects(0, 0, w, h, cols, rows, "cell", True, 10)}'
            f'</g></svg>')


def map_svg(mini, heat, t, p, cols, rows, grid, proj) -> str:
    """Карта: фон, вероятность, трапеция видимой области и сетка ячеек.

    Область рисуется трапецией, а не прямоугольником: камера наклонена, и
    верх кадра покрывает примерно на 40% больше ширины карты, чем низ.
    """
    W = MAP_W
    cell = W / grid
    hgrid = "".join(
        f'<line x1="{i * cell:.1f}" y1="0" x2="{i * cell:.1f}" y2="{W}"/>'
        f'<line x1="0" y1="{i * cell:.1f}" x2="{W}" y2="{i * cell:.1f}"/>'
        for i in range(1, grid))

    Hm, size = proj
    out = scene_to_map(Hm, np.array([[0, 0], [size[0], 0],
                                     [size[0], size[1]], [0, size[1]]], float))

    def poly(pts, origin):
        return " ".join(f"{(origin[0] + q[0]) * W:.1f},{(origin[1] + q[1]) * W:.1f}"
                        for q in pts)

    cells = "".join(
        f'<polygon class="cell" data-cell="{cid}" points="{poly(q, t)}"/>'
        for cid, q in cell_polys(Hm, size, cols, rows))
    tx, ty = t[0] * W, t[1] * W
    px, py = p[0] * W, p[1] * W
    return (
        f'<svg class="map" viewBox="0 0 {W} {W}" width="{W}" height="{W}">'
        f'<image href="{mini}" x="0" y="0" width="{W}" height="{W}"/>'
        f'<image class="heat" href="{heat}" x="0" y="0" width="{W}" height="{W}"/>'
        f'<g class="hgrid">{hgrid}</g>'
        f'<g class="m-true"><polygon points="{poly(out, t)}"/>'
        f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="5"/></g>'
        f'<g class="cells">{cells}</g>'
        f'<g class="m-pred"><polygon points="{poly(out, p)}"/>'
        f'<line x1="{px - 8:.1f}" y1="{py:.1f}" x2="{px + 8:.1f}" y2="{py:.1f}"/>'
        f'<line x1="{px:.1f}" y1="{py - 8:.1f}" x2="{px:.1f}" y2="{py + 8:.1f}"/>'
        f'</g>'
        f'<line class="link" x1="{tx:.1f}" y1="{ty:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>'
        f'</svg>')


def build(vid, run: Path, root: Path, every, worst, grid, cols, rows, proj):
    src = list(csv.DictReader((run / f"predictions_{vid}.csv").open(encoding="utf-8")))
    z = np.load(run / f"heatmaps_{vid}.npz")
    heat = z["heatmaps"].astype(np.float32)
    err = np.array([float(r["err"]) for r in src])

    pick = set(range(0, len(src), every))
    if worst:
        pick |= set(np.argsort(-err)[:worst].tolist())
    pick = sorted(pick)

    cards, dots = [], []
    for i, r in enumerate(src):
        e = err[i]
        state = "ok" if e < GOOD else ("mid" if e < OKAY else "bad")
        sec = int(float(r["t_sec"]))
        tc = f"{sec // 60}:{sec % 60:02d}"
        dots.append((state, f"#{r['idx']} {tc} · промах {e:.3f}"))
        if i not in pick:
            continue
        with Image.open(root / r["scene"]) as im:
            sh = round(SCENE_W * im.height / im.width)
            scene = b64(im.convert("RGB").resize((SCENE_W, sh), Image.BILINEAR))
        with Image.open(root / r["mini"]) as mi:
            mini = b64(mi.convert("RGB").resize((MAP_W, MAP_W), Image.LANCZOS))
        t = (float(r["cx_true"]), float(r["cy_true"]))
        p = (float(r["cx_pred"]), float(r["cy_pred"]))
        url = f"https://www.youtube.com/watch?v={vid}&t={sec}s"

        cards.append(
            f'<article class="card" id="f{i}" data-state="{state}" '
            f'data-err="{e:.4f}" data-spread="{r["spread"]}" '
            f'data-cx="{t[0]:.4f}" data-cy="{t[1]:.4f}">'
            f'<header><a class="tc" href="{url}" target="_blank" rel="noopener">'
            f'&#9654; {tc}</a>'
            f'<span class="hd">кадр {r["idx"]} · промах <b>{e:.3f}</b> '
            f'&asymp; {e * MAP_UNITS:.0f} ед. · разброс ответа '
            f'{float(r["spread"]):.3f}</span>'
            f'<span class="cellinfo"></span></header>'
            f'<div class="grid-row">'
            f'<figure>{scene_svg(scene, SCENE_W, sh, cols, rows)}'
            f'<figcaption>вход модели: это и есть содержимое белого '
            f'прямоугольника на карте</figcaption></figure>'
            f'<figure>{map_svg(mini, heat_png(heat[i]), t, p, cols, rows, grid, proj)}'
            f'<figcaption>карта: красное — распределение ответа</figcaption></figure>'
            f'<div class="num">'
            f'<div class="row hl-true"><span>истина</span>'
            f'<b>{t[0]:.3f}, {t[1]:.3f}</b></div>'
            f'<div class="row hl-pred"><span>модель</span>'
            f'<b>{p[0]:.3f}, {p[1]:.3f}</b></div>'
            f'<div class="row hl-link"><span>промах</span><b>{e:.3f}</b></div>'
            f'<div class="row"><span>в единицах</span><b>{e * MAP_UNITS:.0f}</b></div>'
            f'<div class="row"><span>доля вьюпорта</span>'
            f'<b>{e / VIEW_W * 100:.0f}%</b></div>'
            f'<div class="tip">наведите на строку — подсветится отметка на карте;'
            f' наведите на ячейку — подсветится её место на карте</div>'
            f'</div></div></article>')

    dots_html = "".join(
        f'<a class="dot {s}" href="#f{i}" title="{html.escape(tip)}"></a>'
        for i, (s, tip) in enumerate(dots))
    stats = {"total": len(src), "shown": len(pick),
             "median": float(np.median(err)), "mean": float(err.mean()),
             "good": int((err < GOOD).sum()), "bad": int((err >= OKAY).sum()),
             "strip": "".join({"ok": "o", "bad": "x"}.get(s, ".") for s, _ in dots)}
    return "\n".join(cards), dots_html, stats


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Координаты камеры — {video}</title>
<style>
 :root {{ color-scheme: dark; --true:#ffffff; --pred:#5aff8c; --cell:#ffd166; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:0 16px 48px; background:#14161a; color:#e8e8ea;
        font:14px/1.5 system-ui,Segoe UI,sans-serif; }}
 .top {{ position:sticky; top:0; z-index:9; background:#14161a;
         border-bottom:1px solid #2a2e36; padding:12px 0 10px; margin-bottom:16px; }}
 h1 {{ font-size:17px; margin:0 0 3px; }}
 .sum {{ color:#9aa0aa; margin-bottom:8px; }}
 .sum b.ok {{ color:#6cd08a; }} .sum b.bad {{ color:#ff7b6e; }}
 .legend {{ display:flex; gap:18px; flex-wrap:wrap; color:#9aa0aa; font-size:12px;
            background:#1a1d23; border:1px solid #2a2e36; border-radius:8px;
            padding:8px 12px; margin-bottom:10px; }}
 .legend i {{ display:inline-block; width:22px; height:12px; vertical-align:-2px;
              margin-right:6px; border-radius:2px; }}
 .lg-true {{ border:2px solid var(--true); }}
 .lg-pred {{ border:2px dashed var(--pred); }}
 .lg-heat {{ background:linear-gradient(90deg,#3a1a10,#ff5a28); }}
 .lg-cell {{ border:2px solid var(--cell); }}
 .tabs {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
 .tab {{ display:block; padding:5px 10px; border:1px solid #39404c; border-radius:6px;
         color:#cfd4dc; text-decoration:none; background:#1b1f26; font-size:13px; }}
 .tab em {{ display:block; font-style:normal; color:#878d98; font-size:11px; }}
 .tab.on {{ background:#3a5bd9; border-color:#3a5bd9; color:#fff; }}
 .tab.home {{ background:#262b34; }}
 .strip {{ display:flex; flex-wrap:wrap; gap:2px; margin-bottom:10px; }}
 .dot {{ width:7px; height:13px; border-radius:2px; display:block; background:#4a515d; }}
 .dot.ok {{ background:#3fa863; }} .dot.mid {{ background:#c9922f; }}
 .dot.bad {{ background:#d6483c; }}
 .dot:hover {{ outline:2px solid #e8e8ea; }}
 .dot.dim {{ opacity:.22; }}
 .controls {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
 button {{ background:#222730; color:#e8e8ea; border:1px solid #39404c;
           border-radius:6px; padding:6px 12px; cursor:pointer; font:inherit; }}
 button.on {{ background:#3a5bd9; border-color:#3a5bd9; }}
 .hint {{ color:#767d89; font-size:12px; }}
 .card {{ border:1px solid #2a2e36; border-radius:10px; margin-bottom:14px;
          background:#1a1d23; overflow:hidden; scroll-margin-top:220px; }}
 .card[data-state=bad] {{ border-color:#b4473f; }}
 .card[data-state=ok] {{ border-color:#2f6d43; }}
 header {{ padding:8px 12px; background:#20242b; border-bottom:1px solid #2a2e36;
           display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }}
 .tc {{ color:#8ab4ff; text-decoration:none; font-weight:600;
        border:1px solid #33425e; border-radius:5px; padding:1px 9px; }}
 .tc:hover {{ background:#2a3a57; color:#fff; }}
 .hd {{ color:#cfd4dc; }}
 .cellinfo {{ color:var(--cell); font-variant-numeric:tabular-nums; }}
 .grid-row {{ display:flex; gap:14px; padding:12px; flex-wrap:wrap;
              align-items:flex-start; }}
 figure {{ margin:0; }}
 figcaption {{ color:#878d98; font-size:11px; margin-top:4px; max-width:320px; }}
 svg {{ display:block; border-radius:6px; }}
 svg.scene {{ outline:2px solid transparent; }}
 .card.show-true svg.scene {{ outline-color:var(--true); }}
 .cells rect, .cells polygon {{ fill:transparent; stroke:#fff; stroke-width:.5;
                opacity:0; cursor:crosshair; }}
 .cells text {{ fill:#fff; opacity:0; pointer-events:none;
                paint-order:stroke; stroke:#000; stroke-width:2.5px; }}
 body.cells-on .cells rect, body.cells-on .cells polygon {{ opacity:.35; }}
 body.cells-on svg.scene .cells text {{ opacity:.8; }}
 .cells rect.lit, .cells polygon.lit {{ opacity:1 !important; stroke:var(--cell);
                    stroke-width:2.5; fill:rgba(255,209,102,.22); }}
 .hgrid line {{ stroke:#fff; stroke-width:.5; opacity:0; }}
 body.grid-on .hgrid line {{ opacity:.2; }}
 .m-true polygon {{ fill:none; stroke:var(--true); stroke-width:1.5; opacity:.85; }}
 .m-true circle {{ fill:none; stroke:var(--true); stroke-width:3; }}
 .m-pred polygon {{ fill:none; stroke:var(--pred); stroke-width:1; opacity:.45;
                 stroke-dasharray:4 3; }}
 .m-pred line {{ stroke:var(--pred); stroke-width:3; }}
 .link {{ stroke:var(--cell); stroke-width:2; opacity:0; }}
 .card.show-true .m-true polygon {{ stroke-width:3.5; opacity:1; }}
 .card.show-true .m-true circle {{ stroke-width:4; }}
 .card.show-pred .m-pred polygon {{ opacity:1; stroke-width:2.5; }}
 .card.show-pred .m-pred line {{ stroke-width:5; }}
 .card.show-link .link {{ opacity:1; }}
 .card.hide-heat .heat {{ opacity:0; }}
 .num {{ min-width:220px; padding:10px 12px; border-radius:8px; background:#20242b;
         border:1px solid #2f353f; }}
 .row {{ display:flex; justify-content:space-between; gap:14px; padding:3px 6px;
         border-radius:5px; }}
 .row span {{ color:#9aa0aa; }}
 .row.hl-true:hover {{ background:#39404c; }}
 .row.hl-pred:hover {{ background:#1d3b28; }}
 .row.hl-link:hover {{ background:#3b3522; }}
 .tip {{ color:#767d89; font-size:11px; margin-top:8px; }}
 .hide {{ display:none; }}
 .note {{ color:#767d89; font-size:12px; margin-top:12px; max-width:920px; }}
</style></head><body>
<div class="top">
  <h1>Координаты камеры · {video}{tag}</h1>
  <div class="sum">{summary}</div>
  <div class="legend">
    <span><i class="lg-true"></i>белый прямоугольник — область карты, которую камера
      показывает на самом деле; её содержимое и есть картинка слева</span>
    <span><i class="lg-pred"></i>зелёный — та же область по мнению модели</span>
    <span><i class="lg-heat"></i>красное — распределение вероятности,
      ячейка {grid}&times;{grid} &asymp; {cell_units:.0f} игровых единиц</span>
    <span><i class="lg-cell"></i>сетка {cols}&times;{rows}: наведите на ячейку
      картинки — подсветится её место на карте</span>
  </div>
  {nav_open}{nav}{nav_close}
  <div class="strip">{dots}</div>
  <div class="controls">
    <button class="on" data-f="all">все показанные</button>
    <button data-f="bad">крупные промахи</button>
    <button data-f="ok">точные</button>
    <button data-f="unsure">модель сомневается</button>
    <button id="cells" class="on">сетка {cols}&times;{rows}</button>
    <button id="grid">сетка вероятности</button>
    <button id="heat">скрыть распределение</button>
    <span class="hint" id="count"></span>
  </div>
</div>
{cards}
<div class="note">Полоса сверху покрывает все {total} кадров ролика; подробно показано
{shown} (каждый {every}-й плюс самые крупные промахи). Таймкод — ссылка на этот момент
в ролике. «Промах» — расстояние между центрами белого и зелёного прямоугольников.
Сейчас модель выдаёт <b>одну</b> точку на весь кадр; сетка {cols}&times;{rows} показана,
чтобы было видно, какой кусок карты приходится на каждую ячейку картинки — это цель
для поячеечной модели, а не её текущий ответ. Перевод ячейки в координаты карты идёт через
калиброванную гомографию, поэтому ячейки на карте — четырёхугольники разного размера:
ближние к верху кадра шире. Проекция подобрана на двух роликах и проверена на двух
других (остаток 29 px при шаге метки с миникарты около 16 px).</div>
<script>
const cards=[...document.querySelectorAll('.card')];
const dots=[...document.querySelectorAll('.dot')];
const btns=[...document.querySelectorAll('[data-f]')];
const shown=cards.map(c=>+c.id.slice(1));
const VIEW_W={view_w}, VIEW_H={view_h}, COLS={cols}, ROWS={rows};

function show(f){{
  let n=0;
  cards.forEach(c=>{{
    const st=c.dataset.state, sp=parseFloat(c.dataset.spread);
    const ok = f==='all' || (f==='bad'&&st==='bad') || (f==='ok'&&st==='ok')
            || (f==='unsure'&&sp>0.15);
    c.classList.toggle('hide',!ok); if(ok)n++;
  }});
  dots.forEach((d,i)=>d.classList.toggle('dim', !shown.includes(i)));
  document.getElementById('count').textContent='показано '+n+' из '+cards.length;
}}
dots.forEach((d,i)=>d.addEventListener('click',e=>{{
  const t=document.getElementById('f'+i);
  if(t){{e.preventDefault(); t.classList.remove('hide');
        t.scrollIntoView({{behavior:'smooth',block:'start'}});}}
}}));
btns.forEach(b=>b.onclick=()=>{{
  btns.forEach(x=>x.classList.toggle('on',x===b)); show(b.dataset.f);
}});

for(const pair of [['hl-true','show-true'],['hl-pred','show-pred'],['hl-link','show-link']]){{
  document.querySelectorAll('.'+pair[0]).forEach(r=>{{
    const card=r.closest('.card');
    r.addEventListener('mouseenter',()=>card.classList.add(pair[1]));
    r.addEventListener('mouseleave',()=>card.classList.remove(pair[1]));
  }});
}}

// Ячейка картинки и ячейка карты связаны номером: подсвечиваем обе разом
// и показываем, в какие координаты карты попадает центр этой ячейки.
cards.forEach(card=>{{
  const info=card.querySelector('.cellinfo');
  const cx=parseFloat(card.dataset.cx), cy=parseFloat(card.dataset.cy);
  card.querySelectorAll('.cells polygon').forEach(rc=>{{
    rc.addEventListener('mouseenter',()=>{{
      const id=rc.dataset.cell;
      card.querySelectorAll('[data-cell="'+id+'"]')
          .forEach(x=>x.classList.add('lit'));
      const p=id.split('.'), r=+p[0], c=+p[1];
      const mx=cx-VIEW_W/2+VIEW_W*(c-0.5)/COLS;
      const my=cy-VIEW_H/2+VIEW_H*(r-0.5)/ROWS;
      info.textContent='ячейка '+id+' → карта '+mx.toFixed(3)+', '+my.toFixed(3);
    }});
    rc.addEventListener('mouseleave',()=>{{
      card.querySelectorAll('.lit').forEach(x=>x.classList.remove('lit'));
      info.textContent='';
    }});
  }});
}});

document.getElementById('cells').onclick=e=>{{
  document.body.classList.toggle('cells-on');
  e.target.classList.toggle('on',document.body.classList.contains('cells-on'));
}};
document.getElementById('grid').onclick=e=>{{
  document.body.classList.toggle('grid-on');
  e.target.classList.toggle('on',document.body.classList.contains('grid-on'));
}};
document.getElementById('heat').onclick=e=>{{
  const on=!cards[0].classList.contains('hide-heat');
  cards.forEach(c=>c.classList.toggle('hide-heat',on));
  e.target.classList.toggle('on',on);
  e.target.textContent=on?'показать распределение':'скрыть распределение';
}};
document.body.classList.add('cells-on');
show('all');
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--run", type=Path, default=Path("runs/coords"))
    ap.add_argument("--root", type=Path, default=Path("data/coords"))
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--cells", default="8x5", help="сетка на картинке, СТОЛБЦЫxСТРОКИ")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.cells.lower().split("x"))
    ck = json.loads((args.run / "eval_coords.json").read_text(encoding="utf-8"))
    grid = int(torch.load(args.run / "model.pt", map_location="cpu",
                          weights_only=False).get("grid", 32))
    trained = args.video in ck.get("trained_on", [])
    out = args.out or Path("runs/inspect") / f"coords-{args.video}.html"
    proj = load_projection()
    if proj is None:
        raise SystemExit("нет dataset/layouts/projection.json — "
                         "сначала tools/calibrate_projection.py")
    cards, dots, st = build(args.video, args.run, args.root, args.every,
                            args.worst, grid, cols, rows, proj)

    summary = (f'кадров {st["total"]} · медианный промах <b>{st["median"]:.3f}</b> '
               f'(&asymp;{st["median"] * MAP_UNITS:.0f} игровых единиц, '
               f'{st["median"] / VIEW_W * 100:.0f}% ширины вьюпорта) · '
               f'точных <b class="ok">{st["good"]}</b> · '
               f'крупных промахов <b class="bad">{st["bad"]}</b>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(
        video=html.escape(args.video),
        tag=" — ОБУЧАЮЩИЙ, не отложенный" if trained else " — отложенный",
        summary=summary, dots=dots, cards=cards, total=st["total"],
        shown=st["shown"], every=args.every, grid=grid, cols=cols, rows=rows,
        cell_units=MAP_UNITS / grid, view_w=VIEW_W, view_h=VIEW_H,
        nav_open=NAV_OPEN, nav="", nav_close=NAV_CLOSE), encoding="utf-8")

    pages = update_manifest(out.parent, {
        "file": out.name,
        "title": f"Координаты камеры{' (обучающий)' if trained else ''}",
        "video": args.video, "accuracy": 1 - st["median"], "scored": st["total"],
        "total": st["total"], "correct": st["good"], "errors": st["bad"],
        "strip": st["strip"][:400]})
    sync_nav(out.parent, pages)
    write_index(out.parent, pages)
    print(f'{args.video}: медиана {st["median"]:.3f}, показано {st["shown"]} '
          f'из {st["total"]}')
    print(f"страница ({out.stat().st_size / 1e6:.1f} МБ) → {out}")


if __name__ == "__main__":
    main()
