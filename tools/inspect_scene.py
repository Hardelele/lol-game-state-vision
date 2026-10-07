"""Смотрелка кадров: что на кадре, какая метка, что ответила модель.

Собирает самодостаточную HTML-страницу (картинки внутри файла, ничего
подгружать не нужно — можно открыть двойным кликом или переслать).

Для каждого кадра показывает четыре вещи:

1. кадр целиком — что происходило в игре;
2. миникарту крупно — по рамке камеры на ней ставилась метка, то есть это
   основание разметки, которое можно перепроверить;
3. вход модели — ровно те пиксели после маски, обрезки и масштаба, которые
   видит сеть. Разрыв между 1 и 3 обычно и объясняет странные ответы;
4. метку, ответ модели и уверенность.

Сверху — полоса точек, по одной на кадр в порядке времени: зелёная верно,
красная ошибка, серая без метки. Полоса сразу показывает, чего больше и где
ошибки скапливаются, и по ней же можно перейти к нужному кадру.

Страницы в одной папке знают друг о друге: каждая новая перестраивает
переключатель во всех остальных, поэтому прогоны можно сравнивать.

Предсказания берутся из CSV (`oof_predictions.csv` от train_scene.py или
`holdout_predictions.csv` от eval_scene.py). Для обучающего ролика нужен
именно файл «вне своего фолда»: финальная модель эти кадры учила, и её
ответы по ним ничего не говорят о качестве.

Пример:
    python tools/inspect_scene.py dataset/videos/58w57eJ5Qks \
        --layout dataset/layouts/spectator-volibear-challenger.json \
        --predictions runs/scene/cnn-dense/oof_predictions.csv \
        --checkpoint runs/scene/cnn-dense/model.pt --title "dense 288x256"
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from scene_data import CLASSES, INPUT_SIZE, prepare_frame
from label_sheets import is_dark, minimap_box
from paths import INSPECT

FRAME_W = 520   # ширина превью кадра на странице
MINI_W = 240    # ширина миникарты
INPUT_SCALE = 2  # во сколько раз увеличить вход модели, чтобы его было видно
NAV_OPEN, NAV_CLOSE = "<!--NAV-->", "<!--/NAV-->"
MANIFEST = "pages.json"
INDEX = "index.html"
# Компактная запись результата по кадрам для главной: o верно, x ошибка, точка — без метки.
DOT_CHAR = {"ok": "o", "bad": "x", "skip": ".", "none": "."}


def b64(img: Image.Image, fmt: str = "JPEG", quality: int = 82) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, fmt, quality=quality)
    else:
        img.save(buf, fmt)
    return (f"data:image/{fmt.lower()};base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def read_predictions(path: Path, video_id: str) -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r.get("video_id") and r["video_id"] != video_id:
            continue
        rows[r["frame"]] = r
    return rows


def build_cards(video_dir: Path, layout: dict, preds: dict, limit: int | None,
                size: tuple[int, int], crop: str) -> tuple[list[str], str, dict]:
    rows = list(csv.DictReader((video_dir / "frames.csv").open(encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    mx0, my0, mx1, my1 = minimap_box(layout)

    cards, dots = [], []
    stats = {"total": 0, "scored": 0, "correct": 0, "errors": 0, "no_pred": 0}

    for i, row in enumerate(rows):
        stem = Path(row["frame"]).stem
        with Image.open(video_dir / row["frame"]) as img:
            w, h = img.size
            frame_img = img.resize((FRAME_W, int(FRAME_W * h / w)), Image.BILINEAR)
            mini = img.crop((int(mx0 * w), int(my0 * h), int(mx1 * w), int(my1 * h))
                            ).resize((MINI_W, MINI_W), Image.LANCZOS)
            model_in = prepare_frame(img, layout, size, crop)
            dark = is_dark(img)
        shown = model_in.resize(
            (size[0] * INPUT_SCALE, size[1] * INPUT_SCALE), Image.NEAREST)

        p = preds.get(stem)
        label = row["label"].strip() or "—"
        stats["total"] += 1
        tip = f"#{i + 1} t={row['t_sec']}с · метка {label}"

        if p:
            scored = label in CLASSES
            correct = p["correct"].lower() == "true"
            p_top = float(p["p_top"])
            conf = max(p_top, 1 - p_top)
            if scored:
                stats["scored"] += 1
                stats["correct" if correct else "errors"] += 1
                state = "ok" if correct else "bad"
                verdict = "верно" if correct else "ОШИБКА"
            else:
                state = "skip"
                verdict = "без метки — в счёт не идёт"
            tip += f" · модель {p['pred']} · p(top)={p_top:.2f}"
            pred_html = (
                f'<div class="pred {state}">'
                f'<div class="row"><span>метка</span><b>{html.escape(label)}</b></div>'
                f'<div class="row"><span>модель</span><b>{html.escape(p["pred"])}</b></div>'
                f'<div class="row"><span>p(top)</span><b>{p_top:.3f}</b></div>'
                f'<div class="bar"><i style="width:{p_top * 100:.1f}%"></i></div>'
                f'<div class="row"><span>уверенность</span><b>{conf:.3f}</b></div>'
                f'<div class="verdict">{verdict}</div>'
                + (f'<div class="row"><span>фолд</span><b>{html.escape(p["fold"])}</b></div>'
                   if p.get("fold") else "")
                + "</div>")
        else:
            stats["no_pred"] += 1
            state, conf, p_top = "none", 0.0, -1.0
            pred_html = (f'<div class="pred none">'
                         f'<div class="row"><span>метка</span>'
                         f'<b>{html.escape(label)}</b></div>'
                         f'<div class="verdict">предсказания нет</div></div>')

        dots.append({"state": state, "tip": tip})
        note = html.escape(row.get("note", ""))
        cards.append(
            f'<article class="card" id="f{i + 1}" data-state="{state}" '
            f'data-label="{html.escape(label)}" data-conf="{conf:.4f}">'
            f'<header><b>#{i + 1}</b> t={row["t_sec"]}с · {html.escape(row["timecode"])}'
            + (' · <span class="dark">ЧЁРНЫЙ КАДР</span>' if dark else "")
            + '</header><div class="grid">'
            f'<figure><img src="{b64(frame_img)}" alt="кадр">'
            f'<figcaption>кадр целиком</figcaption></figure>'
            f'<figure><img src="{b64(mini)}" alt="миникарта">'
            f'<figcaption>миникарта — основание метки</figcaption></figure>'
            f'<figure><img class="pix" src="{b64(shown, "PNG")}" alt="вход модели">'
            f'<figcaption>вход модели, {size[0]}×{size[1]}, обрезка {crop}</figcaption>'
            f'</figure>{pred_html}</div>'
            + (f'<footer>{note}</footer>' if note else "")
            + "</article>")

    dots_html = "".join(
        f'<a class="dot {d["state"]}" href="#f{i + 1}" title="{html.escape(d["tip"])}"></a>'
        for i, d in enumerate(dots))
    stats["strip"] = "".join(DOT_CHAR[d["state"]] for d in dots)
    return cards, dots_html, stats


def nav_html(pages: list[dict], current: str) -> str:
    """Переключатель между страницами плюс ссылка на главную."""
    items = [f'<a class="tab home" href="{INDEX}">все прогоны<em>главная</em></a>']
    for p in pages:
        cls = "tab on" if p["file"] == current else "tab"
        if p.get("kind") == "note":
            sub = "разбор"
        else:
            acc = f'{p["accuracy"]:.3f}' if p["scored"] else "—"
            sub = f'{html.escape(p["video"])} · точность {acc}'
        items.append(
            f'<a class="{cls}" href="{html.escape(p["file"])}">'
            f'{html.escape(p["title"])}<em>{sub}</em></a>')
    return '<div class="tabs">' + "".join(items) + "</div>"


def sync_nav(out_dir: Path, pages: list[dict]) -> None:
    """Перестроить переключатель во всех страницах папки.

    Собранная раньше страница не знает о появившихся позже, поэтому блок
    навигации в ней заменяется целиком при каждой новой сборке.
    """
    pattern = re.compile(re.escape(NAV_OPEN) + ".*?" + re.escape(NAV_CLOSE), re.S)
    for p in pages:
        f = out_dir / p["file"]
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        block = NAV_OPEN + nav_html(pages, p["file"]) + NAV_CLOSE
        new = pattern.sub(lambda _: block, text, count=1)
        if new != text:
            f.write_text(new, encoding="utf-8")


INDEX_TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Прогоны — распознавание сцены</title>
<style>
 :root {{ color-scheme: dark; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:24px 16px 48px; background:#14161a; color:#e8e8ea;
        font:14px/1.5 system-ui,Segoe UI,sans-serif; }}
 .wrap {{ max-width:1100px; margin:0 auto; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 .lead {{ color:#9aa0aa; margin-bottom:22px; }}
 h2 {{ font-size:15px; margin:26px 0 10px; color:#cfd4dc;
       border-bottom:1px solid #2a2e36; padding-bottom:6px; }}
 h2 small {{ color:#878d98; font-weight:400; }}
 a.run {{ display:block; text-decoration:none; color:inherit; background:#1a1d23;
          border:1px solid #2a2e36; border-radius:10px; padding:12px 14px;
          margin-bottom:10px; }}
 a.run:hover {{ border-color:#3a5bd9; background:#1d2129; }}
 .head {{ display:flex; justify-content:space-between; gap:16px;
          align-items:baseline; flex-wrap:wrap; }}
 .name {{ font-weight:600; font-size:15px; }}
 .acc {{ font-variant-numeric:tabular-nums; font-size:15px; }}
 .meta {{ color:#9aa0aa; font-size:12px; margin-top:2px; }}
 .meta b.ok {{ color:#6cd08a; }} .meta b.bad {{ color:#ff7b6e; }}
 .mini {{ display:flex; gap:2px; flex-wrap:wrap; margin-top:9px; }}
 .mini i {{ width:7px; height:14px; border-radius:2px; background:#4a515d; }}
 .mini i.o {{ background:#3fa863; }}
 .mini i.x {{ background:#d6483c; }}
 .empty {{ color:#878d98; }}
 footer {{ margin-top:32px; color:#767d89; font-size:12px;
           border-top:1px solid #2a2e36; padding-top:12px; }}
 code {{ background:#20242b; padding:1px 5px; border-radius:4px; }}
</style></head><body><div class="wrap">
<h1>Прогоны: распознавание сцены</h1>
<div class="lead">{lead}</div>
{groups}
<footer>
Собирается автоматически при каждом запуске <code>tools/inspect_scene.py</code>.
Полоска под каждым прогоном — кадры по порядку времени: зелёный верно, красный
ошибка, серый без метки. Прогоны на отложенных роликах важнее прогонов на
обучающем: на обучающем модель мерит сама себя.
</footer>
</div></body></html>
"""


def write_index(out_dir: Path, pages: list[dict]) -> Path:
    """Главная со списком всех прогонов, сгруппированных по ролику."""
    by_video: dict[str, list[dict]] = {}
    notes = []
    for p in pages:
        # Разборы и заметки не меряются точностью — им отдельный раздел наверху.
        (notes if p.get("kind") == "note" else
         by_video.setdefault(p["video"], [])).append(p)

    groups = []
    if notes:
        rows = [f'<a class="run" href="{html.escape(p["file"])}">'
                f'<div class="head"><span class="name">{html.escape(p["title"])}</span></div>'
                + (f'<div class="meta">собран {html.escape(p["built"])}</div>'
                   if p.get("built") else "") + "</a>"
                for p in sorted(notes, key=lambda p: p["title"])]
        groups.append("<h2>Разборы <small>— без метрик</small></h2>" + "".join(rows))

    for video in sorted(by_video):
        runs = sorted(by_video[video], key=lambda p: p["title"])
        rows = []
        for p in runs:
            acc = f'{p["accuracy"]:.3f}' if p.get("scored") else "—"
            strip = p.get("strip", "")
            mini = "".join(f'<i class="{c if c != "." else ""}"></i>' for c in strip)
            rows.append(
                f'<a class="run" href="{html.escape(p["file"])}">'
                f'<div class="head"><span class="name">{html.escape(p["title"])}</span>'
                f'<span class="acc">{acc}</span></div>'
                f'<div class="meta">кадров {p.get("total", "?")} · '
                f'верно <b class="ok">{p.get("correct", "?")}</b> · '
                f'ошибок <b class="bad">{p.get("errors", "?")}</b>'
                + (f' · собран {html.escape(p["built"])}' if p.get("built") else "")
                + f'</div><div class="mini">{mini}</div></a>')
        groups.append(f'<h2>{html.escape(video)} <small>'
                      f'— прогонов {len(runs)}</small></h2>' + "".join(rows))

    lead = (f"страниц {len(pages)}, роликов {len(by_video)}"
            if pages else "пока ничего не собрано")
    path = out_dir / INDEX
    path.write_text(
        INDEX_TEMPLATE.format(lead=html.escape(lead),
                              groups="".join(groups) or
                              '<div class="empty">Пусто.</div>'),
        encoding="utf-8")
    return path


def update_manifest(out_dir: Path, entry: dict) -> list[dict]:
    """Список страниц папки: обновить свою запись, забыть удалённые файлы."""
    path = out_dir / MANIFEST
    pages = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    pages = [p for p in pages
             if p["file"] != entry["file"] and (out_dir / p["file"]).exists()]
    pages.append(entry)
    pages.sort(key=lambda p: (p["video"], p["title"]))
    path.write_text(json.dumps(pages, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return pages


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — {video}</title>
<style>
 :root {{ color-scheme: dark; }}
 * {{ box-sizing: border-box; }}
 body {{ margin:0; padding:0 16px 40px; background:#14161a; color:#e8e8ea;
        font:14px/1.45 system-ui,Segoe UI,sans-serif; }}
 .top {{ position:sticky; top:0; z-index:9; background:#14161a;
         border-bottom:1px solid #2a2e36; padding:12px 0 10px; margin-bottom:16px; }}
 h1 {{ font-size:17px; margin:0 0 3px; }}
 .sum {{ color:#9aa0aa; margin-bottom:10px; }}
 .sum b.ok {{ color:#6cd08a; }} .sum b.bad {{ color:#ff7b6e; }}
 .tabs {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
 .tab {{ display:block; padding:5px 10px; border:1px solid #39404c; border-radius:6px;
         color:#cfd4dc; text-decoration:none; background:#1b1f26; font-size:13px; }}
 .tab em {{ display:block; font-style:normal; color:#878d98; font-size:11px; }}
 .tab.on {{ background:#3a5bd9; border-color:#3a5bd9; color:#fff; }}
 .tab.on em {{ color:#d6ddf7; }}
 .tab.home {{ background:#262b34; }}
 .strip {{ display:flex; flex-wrap:wrap; gap:3px; margin-bottom:10px; }}
 .dot {{ width:12px; height:12px; border-radius:3px; display:block;
         background:#4a515d; transition:transform .08s; }}
 .dot.ok {{ background:#3fa863; }}
 .dot.bad {{ background:#d6483c; }}
 .dot:hover {{ transform:scale(1.5); }}
 .dot.dim {{ opacity:.2; }}
 .dot.here {{ outline:2px solid #e8e8ea; outline-offset:1px; }}
 .controls {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
 button {{ background:#222730; color:#e8e8ea; border:1px solid #39404c;
           border-radius:6px; padding:6px 12px; cursor:pointer; font:inherit; }}
 button.on {{ background:#3a5bd9; border-color:#3a5bd9; }}
 .hint {{ color:#767d89; font-size:12px; }}
 .card {{ border:1px solid #2a2e36; border-radius:10px; margin-bottom:14px;
          background:#1a1d23; overflow:hidden; scroll-margin-top:170px; }}
 .card[data-state=bad] {{ border-color:#b4473f; }}
 .card[data-state=ok] {{ border-color:#2f6d43; }}
 .card.flash {{ box-shadow:0 0 0 2px #e8e8ea; }}
 header {{ padding:8px 12px; background:#20242b; border-bottom:1px solid #2a2e36; }}
 .dark {{ color:#ffb347; }}
 .grid {{ display:flex; gap:14px; padding:12px; flex-wrap:wrap; align-items:flex-start; }}
 figure {{ margin:0; }}
 figure img {{ display:block; border-radius:6px; max-width:100%; }}
 img.pix {{ image-rendering:pixelated; }}
 figcaption {{ color:#878d98; font-size:12px; margin-top:4px; }}
 .pred {{ min-width:220px; padding:10px 12px; border-radius:8px; background:#20242b;
          border:1px solid #2f353f; }}
 .pred.bad {{ background:#2e1d1c; border-color:#b4473f; }}
 .pred.ok {{ background:#17251b; border-color:#2f6d43; }}
 .row {{ display:flex; justify-content:space-between; gap:16px; padding:2px 0; }}
 .row span {{ color:#9aa0aa; }}
 .bar {{ height:7px; background:#2f353f; border-radius:4px; overflow:hidden; margin:6px 0; }}
 .bar i {{ display:block; height:100%; background:#3a5bd9; }}
 .verdict {{ margin-top:6px; font-weight:600; }}
 .bad .verdict {{ color:#ff7b6e; }}
 .ok .verdict {{ color:#6cd08a; }}
 footer {{ padding:8px 12px; border-top:1px solid #2a2e36; color:#9aa0aa; }}
 .hide {{ display:none; }}
</style></head><body>
<div class="top">
  <h1>{title} · {video}</h1>
  <div class="sum">{summary}</div>
  {nav_open}{nav}{nav_close}
  <div class="strip" id="strip">{dots}</div>
  <div class="controls">
    <button class="on" data-f="all">все</button>
    <button data-f="bad">только ошибки</button>
    <button data-f="ok">только верные</button>
    <button data-f="top">метка top</button>
    <button data-f="not_top">метка not_top</button>
    <button data-f="unknown">метка unknown</button>
    <button data-f="lowconf">уверенность &lt; 0.7</button>
    <span class="hint" id="count"></span>
    <span class="hint">← → — по показанным кадрам</span>
  </div>
</div>
{cards}
<script>
const cards=[...document.querySelectorAll('.card')];
const dots=[...document.querySelectorAll('.dot')];
const btns=[...document.querySelectorAll('[data-f]')];
let visible=[],cur=-1;

const match=(f,st,lab,cf)=> f==='all' || (f==='bad'&&st==='bad') || (f==='ok'&&st==='ok')
  || (f==='lowconf'&&cf>0&&cf<0.7) || (f===lab);

function show(f){{
  visible=[];
  cards.forEach((c,i)=>{{
    const ok=match(f,c.dataset.state,c.dataset.label,parseFloat(c.dataset.conf));
    c.classList.toggle('hide',!ok);
    dots[i].classList.toggle('dim',!ok);
    if(ok)visible.push(i);
  }});
  cur=-1;
  document.getElementById('count').textContent=
    `показано ${{visible.length}} из ${{cards.length}}`;
}}

function goto(i){{
  cur=visible.indexOf(i);
  const c=cards[i];
  c.scrollIntoView({{behavior:'smooth',block:'start'}});
  dots.forEach(d=>d.classList.remove('here'));
  dots[i].classList.add('here');
  c.classList.add('flash');
  setTimeout(()=>c.classList.remove('flash'),900);
}}

dots.forEach((d,i)=>d.addEventListener('click',e=>{{e.preventDefault();goto(i);}}));
btns.forEach(b=>b.onclick=()=>{{
  btns.forEach(x=>x.classList.toggle('on',x===b));
  show(b.dataset.f);
}});
addEventListener('keydown',e=>{{
  if(!visible.length)return;
  if(e.key==='ArrowRight'){{cur=Math.min(cur+1,visible.length-1);goto(visible[Math.max(cur,0)]);}}
  if(e.key==='ArrowLeft'){{cur=Math.max(cur-1,0);goto(visible[cur]);}}
}});
show('all');
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="oof_predictions.csv (обучающий ролик) или holdout_predictions.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="файл страницы; по умолчанию runs/inspect/<ролик>-<прогон>.html")
    ap.add_argument("--title", default=None,
                    help="подпись в переключателе; по умолчанию имя папки прогона")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N кадров")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="взять размер входа и режим обрезки из чекпоинта модели")
    ap.add_argument("--crop", choices=("bbox", "dense", "none"), default="dense")
    ap.add_argument("--size", default="x".join(map(str, INPUT_SIZE)))
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    run = args.title or args.predictions.parent.name
    out = args.out or (INSPECT
                       / f"{args.video_dir.name}-{args.predictions.parent.name}.html")

    size = tuple(int(v) for v in args.size.lower().split("x"))
    crop = args.crop
    if args.checkpoint:
        import torch
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # Чекпоинты до появления поля crop обучались на обрезке bbox. Подставлять
        # сюда текущее умолчание нельзя: страница показала бы не тот вход,
        # который видела сеть, и выводы об ошибках были бы неверны.
        size, crop = tuple(ck["input_size"]), ck.get("crop", "bbox")
        if "crop" not in ck:
            print("в чекпоинте нет поля crop (старый формат) — считаю обрезку bbox")
        print(f"предобработка из чекпоинта: вход {size}, обрезка {crop}")

    preds = read_predictions(args.predictions, args.video_dir.name)
    if not preds:
        raise SystemExit(f"в {args.predictions} нет строк для ролика {args.video_dir.name}")

    cards, dots, stats = build_cards(args.video_dir, layout, preds, args.limit, size, crop)
    acc = stats["correct"] / stats["scored"] if stats["scored"] else 0.0
    summary = (f'кадров {stats["total"]} · верно <b class="ok">{stats["correct"]}</b> · '
               f'ошибок <b class="bad">{stats["errors"]}</b> · точность {acc:.3f}'
               + (f' · без метки {stats["total"] - stats["scored"]}'
                  if stats["total"] > stats["scored"] else ""))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(
        title=html.escape(run), video=html.escape(args.video_dir.name),
        summary=summary, dots=dots, cards="\n".join(cards),
        nav_open=NAV_OPEN, nav="", nav_close=NAV_CLOSE), encoding="utf-8")

    pages = update_manifest(out.parent, {
        "file": out.name, "title": run, "video": args.video_dir.name,
        "accuracy": acc, "scored": stats["scored"], "total": stats["total"],
        "correct": stats["correct"], "errors": stats["errors"],
        "strip": stats["strip"],
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    sync_nav(out.parent, pages)
    index = write_index(out.parent, pages)

    print(f'{stats["total"]} кадров: верно {stats["correct"]}, ошибок {stats["errors"]}, '
          f"точность {acc:.3f}")
    print(f"страница ({out.stat().st_size / 1e6:.1f} МБ) → {out}")
    print(f"главная ({len(pages)} прогонов) → {index}")


if __name__ == "__main__":
    main()
