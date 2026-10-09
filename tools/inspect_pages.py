"""Общая обвязка статических HTML-страниц смотрелки (`runs/inspect/`).

Страницы в одной папке знают друг о друге: каждая новая сборка обновляет
список `pages.json`, перестраивает переключатель во всех страницах и главную
`index.html`. Используется `inspect_coords.py` и `inspect_patches.py`.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

from PIL import Image

NAV_OPEN, NAV_CLOSE = "<!--NAV-->", "<!--/NAV-->"
MANIFEST = "pages.json"
INDEX = "index.html"


def b64(img: Image.Image, fmt: str = "JPEG", quality: int = 82) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, fmt, quality=quality)
    else:
        img.save(buf, fmt)
    return (f"data:image/{fmt.lower()};base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


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
<title>Страницы смотрелки</title>
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
<h1>Страницы смотрелки</h1>
<div class="lead">{lead}</div>
{groups}
<footer>
Собирается автоматически при каждом запуске <code>tools/inspect_coords.py</code>
или <code>tools/inspect_patches.py</code>. Полоска под каждым прогоном — кадры
по порядку времени: зелёный — малая ошибка, красный — крупный промах.
Прогоны на отложенных роликах важнее прогонов на обучающем: на обучающем
модель мерит сама себя.
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
