"""Ключевые кадры спора «метка ↔ модель ↔ траектория» — страница для глаз.

Зачем. Цифры tools/label_noise.py говорят, сколько в метке шума, но
грубую ошибку (рамка схвачена не там, рамка обрезана краем карты, кадр не
игра) надёжнее всего подтверждает человек. Страница показывает
стратифицированную выборку кадров и собирает вердикты.

Выборка (seed фиксирован, состав пишется в keyframes.csv):
* `model_vs_label` — самые большие расхождения модели с меткой на
  отложенных роликах;
* `label_vs_traj` — самые большие остатки метки относительно траектории по
  сцене (label_noise.py analyze), отложенные и обучающие ролики;
* `random` — равномерная случайная выборка кадров отложенных роликов с
  качеством рамки ≥ 0.4. Без неё долю грубых ошибок не оценить без смещения:
  первые две страты нарочно собраны из худших кадров;
* `clip_vs_label` — случайные кадры отложенных роликов, где рамка, найденная
  с учётом обрезки краем миникарты (label_noise.find_box_clipped), расходится
  с меткой больше чем на 200 ед.: кто прав в этой группе.

Страница слепая: гипотез на ней не видно. Человек ставит рамку размера
ролика туда, где на увеличенной миникарте белая рамка игры (рамка может
выходить за край), а вердикт считается сам по расстоянию до метки и
модели: совпала — ближе 1.5 пикселя миникарты 256. Так глаз не путается в
пяти квадратах и не подстраивается под подсказку. Положение и вердикт
хранятся в localStorage и выгружаются в CSV в формате
dataset/label_noise/verdicts.csv (см. dataset/README.md). Вердикт агента
(dataset/label_noise/verdicts_agent.csv) показывается только после ответа.

Пример:
    python tools/label_noise_page.py
    # → runs/label_noise/keyframes.html (открыть в браузере файлом)
    python tools/label_noise_page.py --sheets   # ещё и JPG-листы для просмотра
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401
from label_noise import HELD_OUT, OUT, read_csv
from lolvision.geometry import MAP_UNITS, apply_h, load_projection
from lolvision.mask_frames import crop_box
from lolvision.minimap_camera import MM
from lolvision.patch_model import scene_from_box
from lolvision.paths import COORDS_DATA, DATASET

PX = MAP_UNITS / MM
STRATA = ("model_vs_label", "label_vs_traj", "clip_vs_label", "random")
AGENT = DATASET / "label_noise" / "verdicts_agent.csv"
COLORS = {"label": "#ffd84a", "model": "#4ad3ff", "traj": "#ff5ad1", "sub": "#7dff7a",
          "clip": "#ff9b3d"}
NAMES = {"label": "метка", "model": "модель", "traj": "траектория", "sub": "субпиксель",
         "clip": "рамка с обрезкой"}


def f(v: str) -> float:
    return float(v) if v not in ("", None) else float("nan")


def load_frames() -> dict[str, list[dict]]:
    out = {}
    for p in sorted((OUT / "frames").glob("*.csv")):
        out[p.stem] = read_csv(p)
    return out


def select(frames: dict, n_model: int, n_traj: int, n_rand: int, n_clip: int,
           seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    held = [r for v in HELD_OUT if v in frames for r in frames[v] if f(r["quality"]) >= 0.4]
    picks: dict[tuple, dict] = {}

    def add(r, stratum, score):
        k = (r["video_id"], int(r["idx"]))
        if k not in picks:
            picks[k] = {"row": r, "stratum": stratum, "score": round(float(score), 1)}

    em = [(np.hypot(f(r["mx"]) - f(r["lx"]), f(r["my"]) - f(r["ly"])), r) for r in held]
    em = sorted([x for x in em if np.isfinite(x[0])], key=lambda x: -x[0])
    for s, r in em[:n_model]:
        add(r, "model_vs_label", s)
    allr = [r for v in frames.values() for r in v]
    er = [(np.hypot(f(r["lx"]) - f(r["tx"]), f(r["ly"]) - f(r["ty"])), r) for r in allr
          if f(r["quality"]) >= 0.4]
    er = sorted([x for x in er if np.isfinite(x[0])], key=lambda x: -x[0])
    held_er = [x for x in er if x[1]["video_id"] in HELD_OUT]
    train_er = [x for x in er if x[1]["video_id"] not in HELD_OUT]
    k_held = (2 * n_traj) // 3
    for s, r in held_er[:k_held] + train_er[:n_traj - k_held]:
        add(r, "label_vs_traj", s)
    # Рамка с учётом обрезки против метки: случайные кадры из тех, где они
    # расходятся больше чем на 200 ед. (почти все — у края карты). Случайные,
    # а не худшие: по ним оценивается, кто прав в этой группе вообще.
    clip = [r for r in held
            if np.hypot(f(r["kx"]) - f(r["lx"]), f(r["ky"]) - f(r["ly"])) > 200]
    for i in rng.choice(len(clip), size=min(n_clip, len(clip)), replace=False):
        r = clip[int(i)]
        add(r, "clip_vs_label", np.hypot(f(r["kx"]) - f(r["lx"]), f(r["ky"]) - f(r["ly"])))
    idx = rng.choice(len(held), size=min(n_rand, len(held)), replace=False)
    for i in idx:
        r = held[int(i)]
        k = (r["video_id"], int(r["idx"]))
        # Случайная страта — сама по себе: если кадр уже попал в другую, он
        # всё равно числится и здесь (иначе выборка перестаёт быть случайной).
        if k in picks:
            picks[k]["also_random"] = True
        else:
            add(r, "random", 0)
    return list(picks.values())


def footprint_offsets(A: np.ndarray) -> np.ndarray:
    """След полного кадра 1920×1080 вокруг точки камеры, ед. меток (4, 2)."""
    h0, store = load_projection()
    layout = json.loads((DATASET / "layouts" / "spectator-volibear-challenger.json")
                        .read_text(encoding="utf-8"))
    crop = crop_box((1920, 1080), layout, "dense")
    a = scene_from_box((0, 0, 1920, 1080), (1920, 1080), crop, store)
    fp = apply_h(h0 @ a, np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], float))
    return (fp * MAP_UNITS) @ A.T


def b64(img: Image.Image, q: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def hyps(r: dict) -> dict:
    out = {}
    for k, (a, b) in {"label": ("lx", "ly"), "model": ("mx", "my"),
                      "traj": ("tx", "ty"), "sub": ("sx", "sy")}.items():
        x, y = f(r[a]), f(r[b])
        if np.isfinite(x) and np.isfinite(y):
            out[k] = [round(x, 1), round(y, 1)]
    kx, ky = f(r.get("kx", "")), f(r.get("ky", ""))
    if "label" in out and np.isfinite(kx) and np.hypot(kx - out["label"][0], ky - out["label"][1]) > 1:
        out["clip"] = [round(kx, 1), round(ky, 1)]
    return out


def _font(size: int = 14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(scene: Image.Image, mini: Image.Image, h: dict, box: tuple, fp: np.ndarray,
          title: str) -> Image.Image:
    """Лист для просмотра глазами: сцена, миникарта целиком, увеличенный
    участок миникарты без пометок (чтобы видеть саму белую рамку игры) и тот
    же участок с рамками гипотез."""
    Z, half = 4, 50
    pts = [np.array(v) / PX + 0.5 for v in h.values()]
    c = np.mean(pts, 0) if pts else np.array([MM / 2, MM / 2])
    x0, y0 = int(c[0] - half), int(c[1] - half)
    big = mini.resize((MM * Z, MM * Z), Image.BICUBIC)
    raw = big.crop((x0 * Z, y0 * Z, (x0 + 2 * half) * Z, (y0 + 2 * half) * Z))
    over = raw.copy()
    d = ImageDraw.Draw(over)
    bw, bh = box
    for k in ("traj", "clip", "model", "sub", "label"):
        if k not in h:
            continue
        p = np.array(h[k]) / PX + 0.5
        col = COLORS[k]
        X, Y = (p[0] - x0) * Z, (p[1] - y0) * Z
        d.rectangle([X - bw / 2 * Z, Y - bh / 2 * Z, X + bw / 2 * Z, Y + bh / 2 * Z],
                    outline=col, width=1)
        d.ellipse([X - 4, Y - 4, X + 4, Y + 4], outline=col, width=2)
    S = 2 * half * Z
    W = scene.width + 2 * S + 8
    out = Image.new("RGB", (W, max(scene.height + 256, S) + 48), (20, 22, 26))
    out.paste(scene, (0, 26))
    out.paste(mini, (0, 26 + scene.height))
    out.paste(raw, (scene.width + 4, 26))
    out.paste(over, (scene.width + 8 + S, 26))
    g = ImageDraw.Draw(out)
    font = _font(14)
    g.text((6, 4), title, fill=(230, 230, 230), font=font)
    x = scene.width + 8 + S
    for k in ("label", "model", "traj", "sub", "clip"):
        if k in h:
            g.text((x, S + 30), NAMES[k], fill=COLORS[k], font=font)
            x += 9 * len(NAMES[k]) + 14
    g.text((scene.width + 4, S + 30), "без пометок", fill=(200, 200, 200), font=font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-model", type=int, default=30)
    ap.add_argument("--n-traj", type=int, default=30)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-clip", type=int, default=20)
    ap.add_argument("--seed", type=int, default=753)
    ap.add_argument("--out", type=Path, default=OUT / "keyframes.html")
    ap.add_argument("--sheets", action="store_true",
                    help="ещё и JPG-листы в runs/label_noise/sheets/")
    args = ap.parse_args()

    frames = load_frames()
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    A = np.array(summ["A"])
    fp = footprint_offsets(A)
    boxes = {}
    for v in frames:
        m = OUT / "mini" / f"{v}.csv"
        if m.exists():
            r0 = read_csv(m)[0]
            boxes[v] = (int(r0["bw"]), int(r0["bh"]))
    agent = {}
    if AGENT.exists():
        for r in read_csv(AGENT):
            agent[(r["video_id"], int(r["idx"]))] = r
    picks = select(frames, args.n_model, args.n_traj, args.n_random, args.n_clip, args.seed)
    picks.sort(key=lambda p: (STRATA.index(p["stratum"]),
                              -p["score"]))

    items = []
    sheets = OUT / "sheets"
    if args.sheets:
        sheets.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for n, p in enumerate(picks):
        r = p["row"]
        v, i = r["video_id"], int(r["idx"])
        scene = Image.open(COORDS_DATA / v / "scene" / f"{i:06d}.jpg").convert("RGB")
        mini = Image.open(COORDS_DATA / v / "mini" / f"{i:06d}.jpg").convert("RGB")
        h = hyps(r)
        box = boxes.get(v, (68, 38))
        a = agent.get((v, i))
        items.append({
            "key": f"{v}:{i}", "video_id": v, "idx": i, "t_sec": r["t_sec"],
            "stratum": p["stratum"], "also_random": bool(p.get("also_random")),
            "score": p["score"], "quality": f(r["quality"]), "game": r.get("game", ""),
            "hyp": h, "box": box,
            "scene": b64(scene, 80), "mini": b64(mini, 90),
            "agent": ({"verdict": a["verdict"], "note": a.get("note", "")} if a else None),
        })
        key_rows.append({"video_id": v, "idx": i, "t_sec": r["t_sec"], "stratum": p["stratum"],
                         "also_random": int(bool(p.get("also_random"))), "score": p["score"]})
        if args.sheets:
            title = (f"#{n:03d} {v} idx {i} t={r['t_sec']}s  {p['stratum']} "
                     f"score {p['score']}  q={f(r['quality']):.2f}")
            sheet(scene, mini, h, box, fp, title).save(sheets / f"{n:03d}_{v}_{i:06d}.jpg",
                                                      quality=88)
    with (OUT / "keyframes.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)

    page = PAGE.replace("__DATA__", json.dumps({
        "items": items, "px": PX, "mm": MM}, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"кадров {len(items)} (страты: "
          + ", ".join(f"{s} {sum(1 for x in items if x['stratum'] == s)}"
                      for s in STRATA)
          + f") → {args.out}")


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Шум меток: рамка вслепую</title>
<style>
 :root { color-scheme: dark; --bg:#14161a; --panel:#1b1e24; --line:#2a2e36; --text:#e8e8ea;
         --dim:#9aa0aa; --mine:#00ffa6; }
 * { box-sizing:border-box; }
 body { margin:0; padding:16px; background:var(--bg); color:var(--text);
        font:14px/1.45 system-ui,Segoe UI,sans-serif; }
 .wrap { max-width:1280px; margin:0 auto; }
 header { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
 h1 { font-size:18px; margin:0; }
 .dim { color:var(--dim); }
 select, button, input { background:#20242b; color:var(--text); border:1px solid #3a3f49;
        border-radius:6px; padding:5px 10px; font:inherit; }
 button { cursor:pointer; }
 button.primary { background:#1f6f54; border-color:#2a9a74; }
 .main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:16px; }
 @media (max-width:900px) { .main { grid-template-columns:1fr; } }
 #view { width:min(100%, calc(100vh - 130px)); aspect-ratio:1; background:#0c0d10;
         border-radius:8px; display:block;
         cursor:crosshair; touch-action:none; }
 aside { display:flex; flex-direction:column; gap:12px; }
 .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
 .card img, .card canvas { width:100%; display:block; border-radius:4px; }
 kbd { background:#2a2e36; border-radius:4px; padding:0 5px; font-size:12px; }
 .keys { font-size:13px; color:var(--dim); line-height:1.8; }
 .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
 #result { min-height:3em; }
 .ok { color:#7dff7a; } .bad { color:#ff7a7a; }
 progress { width:160px; }
</style></head><body><div class="wrap">
<header>
 <h1>Где белая рамка камеры?</h1>
 <span class="dim" id="count"></span>
 <progress id="prog" max="1" value="0"></progress>
 <select id="group">
  <option value="must">минимум: случайные + край</option>
  <option value="all">все кадры</option>
  <option value="random">только случайные</option>
  <option value="clip_vs_label">только край</option>
  <option value="model_vs_label">модель ≠ метка</option>
  <option value="label_vs_traj">метка ≠ траектория</option>
 </select>
 <label class="dim"><input type="checkbox" id="todo" checked> только без ответа</label>
 <span style="flex:1"></span>
 <button id="csv">CSV</button>
 <button id="clear">Стереть</button>
</header>
<div class="main">
 <div>
  <canvas id="view" width="720" height="720"></canvas>
  <div class="row" style="margin-top:8px">
   <button id="prev">← назад</button>
   <button class="primary" id="ok">Готово <kbd>Enter</kbd></button>
   <button id="skip">Не разобрать <kbd>N</kbd></button>
   <input id="note" placeholder="заметка (необязательно)" style="flex:1;min-width:160px">
  </div>
 </div>
 <aside>
  <div class="card"><div class="dim" id="meta"></div></div>
  <div class="card"><div class="dim" style="margin-bottom:4px">Миникарта целиком</div>
   <canvas id="whole" width="256" height="256"></canvas></div>
  <div class="card"><div class="dim" style="margin-bottom:4px">Сцена (что видит модель)</div>
   <img id="scene" alt=""></div>
  <div class="card" id="result"></div>
  <div class="card keys">
   Наведите зелёную рамку на белую рамку игры и кликните; её можно таскать.
   Рамка может выходить за край миникарты.<br>
   <kbd>←↑→↓</kbd> сдвиг ¼ px, с <kbd>Shift</kbd> — 1 px<br>
   <kbd>Z</kbd> удерживать — спрятать свою рамку<br>
   <kbd>колесо</kbd> масштаб · <kbd>S</kbd> сглаживание<br>
   <kbd>Enter</kbd> готово · <kbd>N</kbd> не разобрать · <kbd>Backspace</kbd> назад
  </div>
 </aside>
</div>
</div>
<script>
const D = __DATA__;
const KEY = "label_noise_place_v1";
// Порог «гипотеза совпала с рамкой человека»: 1.5 пикселя миникарты 256.
const TOL = 1.5 * D.px;
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

const view = document.getElementById("view"), ctx = view.getContext("2d");
const whole = document.getElementById("whole"), wctx = whole.getContext("2d");
const imgs = {};
function img(src) {
  if (!imgs[src]) { imgs[src] = new Image(); imgs[src].src = src; }
  return imgs[src];
}

let list = [], pos = 0, it = null;
let win = null;          // [x0, y0, size] окна в пикселях миникарты
let mine = null;         // центр своей рамки, px миникарты (null — не поставлена)
let hover = null, dragging = false, hide = false, smooth = false;

// Ед. меток ↔ координаты миникарты: так же, как рамки рисовала прежняя
// страница (пиксель k занимает [k, k+1], центр рамки — x + bw/2 + 0.5).
const toPx = v => [v[0] / D.px + 0.5, v[1] / D.px + 0.5];
const toUnits = p => [(p[0] - 0.5) * D.px, (p[1] - 0.5) * D.px];

function inGroup(x, g) {
  if (g === "all") return true;
  if (g === "must") return x.stratum === "random" || x.also_random
                            || x.stratum === "clip_vs_label";
  if (g === "random") return x.stratum === "random" || x.also_random;
  return x.stratum === g;
}
function rebuild(keepKey) {
  const g = document.getElementById("group").value;
  const todo = document.getElementById("todo").checked;
  // Случайные первыми: по ним и считается доля ошибок.
  const order = x => (x.stratum === "random" || x.also_random) ? 0
                     : x.stratum === "clip_vs_label" ? 1 : 2;
  list = D.items.filter(x => inGroup(x, g) && (!todo || !store[x.key] || x.key === keepKey))
                .sort((a, b) => order(a) - order(b));
  pos = Math.max(0, list.findIndex(x => x.key === keepKey));
  show();
}

function show() {
  const all = D.items.filter(x => inGroup(x, document.getElementById("group").value));
  const done = all.filter(x => store[x.key]).length;
  document.getElementById("count").textContent = `готово ${done} из ${all.length}`;
  const pr = document.getElementById("prog"); pr.max = all.length || 1; pr.value = done;
  it = list[pos] || null;
  if (!it) {
    ctx.clearRect(0, 0, view.width, view.height);
    ctx.fillStyle = "#9aa0aa"; ctx.font = "20px system-ui";
    ctx.fillText("В этой группе всё отмечено. Выгрузите CSV.", 40, 60);
    return;
  }
  const pts = ["label", "model", "clip"].filter(k => it.hyp[k]).map(k => toPx(it.hyp[k]));
  const c = pts.length ? [pts.reduce((s, p) => s + p[0], 0) / pts.length,
                          pts.reduce((s, p) => s + p[1], 0) / pts.length] : [128, 128];
  // Окно с запасом: охватывает все гипотезы, но не подсказывает, где какая.
  let span = 0;
  for (const p of pts) span = Math.max(span, Math.abs(p[0] - c[0]), Math.abs(p[1] - c[1]));
  const s = Math.max(90, 2 * span + it.box[0] + 24);
  win = [c[0] - s / 2, c[1] - s / 2, s];
  const st = store[it.key];
  mine = st && st.x !== "" && st.x != null ? toPx([st.x, st.y]) : null;
  document.getElementById("note").value = st ? st.note || "" : "";
  document.getElementById("scene").src = it.scene;
  document.getElementById("meta").innerHTML =
    `${esc(it.video_id)} · ${esc(it.t_sec)} с · кадр ${pos + 1} из ${list.length}<br>` +
    `<a style="color:#8fb0ff" target="_blank" href="https://www.youtube.com/watch?v=` +
    `${esc(it.video_id)}&t=${Math.floor(+it.t_sec)}s">открыть ролик</a>`;
  document.getElementById("result").innerHTML = st ? verdictHtml(it, st) : "";
  draw();
}

function draw() {
  if (!it) return;
  const im = img(it.mini);
  if (!im.complete) { im.onload = draw; return; }
  const [x0, y0, s] = win, Z = view.width / s;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#2b2f38"; ctx.fillRect(0, 0, view.width, view.height);
  ctx.imageSmoothingEnabled = smooth;
  ctx.setTransform(Z, 0, 0, Z, -x0 * Z, -y0 * Z);
  ctx.drawImage(im, 0, 0, D.mm, D.mm);
  ctx.strokeStyle = "#555"; ctx.lineWidth = 1 / Z; ctx.strokeRect(0, 0, D.mm, D.mm);
  const p = dragging || !mine ? (hover || mine) : mine;
  if (p && !hide) {
    const [bw, bh] = it.box;
    ctx.strokeStyle = mine ? "#00ffa6" : "rgba(0,255,166,.55)";
    ctx.lineWidth = 1.5 / Z;
    ctx.strokeRect(p[0] - bw / 2, p[1] - bh / 2, bw, bh);
    ctx.beginPath(); ctx.arc(p[0], p[1], 3 / Z, 0, 7); ctx.stroke();
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  // Вся миникарта с окном.
  wctx.imageSmoothingEnabled = true;
  wctx.drawImage(im, 0, 0, 256, 256);
  wctx.strokeStyle = "#00ffa6"; wctx.lineWidth = 1.5;
  const k = 256 / D.mm; wctx.strokeRect(x0 * k, y0 * k, s * k, s * k);
}

function evPx(e) {
  const r = view.getBoundingClientRect(), [x0, y0, s] = win;
  return [x0 + (e.clientX - r.left) / r.width * s, y0 + (e.clientY - r.top) / r.height * s];
}
view.addEventListener("pointerdown", e => {
  if (!it) return;
  view.setPointerCapture(e.pointerId); dragging = true; mine = evPx(e); hover = mine; draw();
});
view.addEventListener("pointermove", e => {
  if (!it) return;
  hover = evPx(e);
  if (dragging) mine = hover;
  draw();
});
view.addEventListener("pointerup", () => { dragging = false; draw(); });
view.addEventListener("pointerleave", () => { hover = null; draw(); });
view.addEventListener("wheel", e => {
  if (!it) return;
  e.preventDefault();
  const p = evPx(e), f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const s = Math.min(256, Math.max(24, win[2] * f));
  win = [p[0] - (p[0] - win[0]) * s / win[2], p[1] - (p[1] - win[1]) * s / win[2], s];
  draw();
}, {passive: false});

function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function judge(it, x, y) {
  const me = [x, y], d = {};
  for (const k of ["label", "model", "clip"]) if (it.hyp[k]) d[k] = dist(it.hyp[k], me);
  const okL = d.label <= TOL, okM = d.model <= TOL;
  const verdict = okL && okM ? "both_ok" : okL ? "label" : okM ? "model" : "both_wrong";
  return {d, verdict};
}
const NAMES = {label: "метка", model: "модель", clip: "рамка с учётом края"};
const VNAME = {both_ok: "обе верны", label: "права метка", model: "права модель",
               both_wrong: "обе мимо", unclear: "не разобрать"};
function verdictHtml(it, st) {
  if (st.verdict === "unclear") return `<b>не разобрать</b>`;
  const {d} = judge(it, st.x, st.y);
  let h = `<b>${VNAME[st.verdict]}</b><br>`;
  for (const k in d) {
    const ok = d[k] <= TOL;
    h += `<span class="${ok ? "ok" : "bad"}">${NAMES[k]}: ${Math.round(d[k])} ед.</span><br>`;
  }
  if (it.agent) h += `<span class="dim">агент: ${esc(it.agent.verdict)}</span>`;
  return h;
}

function commit(verdict) {
  if (!it) return;
  const note = document.getElementById("note").value;
  if (verdict === "unclear") {
    store[it.key] = {x: "", y: "", verdict, note};
  } else {
    if (!mine) { flash("Сначала поставьте рамку кликом"); return; }
    const [x, y] = toUnits(mine);
    store[it.key] = {x: +x.toFixed(1), y: +y.toFixed(1), verdict: judge(it, x, y).verdict, note};
  }
  save();
  const prev = it;
  const todo = document.getElementById("todo").checked;
  if (todo) { list.splice(pos, 1); } else { pos++; }
  if (pos >= list.length) pos = Math.max(0, list.length - 1);
  if (todo && !list.length) pos = 0;
  show();
  // Итог предыдущего кадра — только после ответа, чтобы не подсказывать.
  document.getElementById("result").innerHTML =
    `<div class="dim">предыдущий кадр:</div>` + verdictHtml(prev, store[prev.key]);
}
function flash(t) {
  const r = document.getElementById("result");
  r.innerHTML = `<span class="bad">${esc(t)}</span>`;
}
function back() {
  const todo = document.getElementById("todo").checked;
  if (todo) { document.getElementById("todo").checked = false; }
  const key = it ? it.key : null;
  rebuild(key);
  if (pos > 0) { pos--; show(); }
}

document.getElementById("ok").onclick = () => commit();
document.getElementById("skip").onclick = () => commit("unclear");
document.getElementById("prev").onclick = back;
document.getElementById("group").onchange = () => rebuild();
document.getElementById("todo").onchange = () => rebuild(it && it.key);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") {
    if (e.key === "Enter") { commit(); e.preventDefault(); }
    return;
  }
  const step = e.shiftKey ? 1 : 0.25;
  const mv = {ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step],
              ArrowDown: [0, step]}[e.key];
  if (mv && mine) { mine = [mine[0] + mv[0], mine[1] + mv[1]]; draw(); e.preventDefault(); }
  else if (e.key === "Enter") { commit(); e.preventDefault(); }
  else if (e.key === "n" || e.key === "N" || e.key === "т" || e.key === "Т") commit("unclear");
  else if (e.key === "Backspace") { back(); e.preventDefault(); }
  else if ((e.key === "z" || e.key === "Z" || e.key === "я" || e.key === "Я") && !hide) {
    hide = true; draw();
  }
  else if (e.key === "s" || e.key === "S" || e.key === "ы" || e.key === "Ы") {
    smooth = !smooth; draw();
  }
});
document.addEventListener("keyup", e => {
  if (["z", "Z", "я", "Я"].includes(e.key)) { hide = false; draw(); }
});

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name; a.click();
}
document.getElementById("csv").onclick = () => {
  const cols = ["video_id", "idx", "t_sec", "stratum", "also_random", "x", "y",
                "verdict", "note"];
  const q = v => /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"'
                                         : String(v);
  const rows = D.items.filter(x => store[x.key]).map(x => ({
    video_id: x.video_id, idx: x.idx, t_sec: x.t_sec, stratum: x.stratum,
    also_random: x.also_random ? 1 : 0, ...store[x.key]}));
  const text = [cols.join(",")].concat(rows.map(r => cols.map(c => q(r[c] ?? "")).join(",")));
  download("verdicts.csv", text.join("\n") + "\n", "text/csv");
};
document.getElementById("clear").onclick = () => {
  // Без confirm(): двойной клик в течение 2 с.
  const b = document.getElementById("clear");
  if (b.dataset.armed) { store = {}; save(); delete b.dataset.armed; b.textContent = "Стереть";
                         rebuild(); return; }
  b.dataset.armed = 1; b.textContent = "Точно? ещё раз";
  setTimeout(() => { delete b.dataset.armed; b.textContent = "Стереть"; }, 2000);
};
window.addEventListener("resize", draw);
rebuild();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
