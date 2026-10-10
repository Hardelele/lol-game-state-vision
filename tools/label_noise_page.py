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

Для каждого кадра: хранимая сцена (то, что видит модель), миникарта целиком
и увеличенный участок с рамками гипотез (метка, модель, траектория,
субпиксельная рамка) и следом кадра по гомографии для каждой гипотезы,
плюс ход x/y за ±10 с. Вердикт: «права метка», «права модель», «обе
верны», «обе мимо», «не разобрать» — хранится в localStorage, выгружается в
CSV/JSON в формате dataset/label_noise/verdicts.csv (см. dataset/README.md).
Вердикты агента (dataset/label_noise/verdicts_agent.csv), если есть,
показаны отдельной пометкой и в выгрузку человека не попадают.

Пример:
    python tools/label_noise_page.py
    # → runs/label_noise/keyframes.html (открыть в браузере файлом)
    python tools/label_noise_page.py --sheets   # ещё и JPG-листы для просмотра
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paths import COORDS_DATA, DATASET
from label_noise import HELD_OUT, OUT, read_csv
from mask_frames import crop_box
from minimap_camera import MM
from patch_model import MAP_UNITS, apply_h, load_projection, scene_from_box

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


def series(rows: list[dict], i: int, k: int = 10) -> dict:
    lo, hi = max(0, i - k), min(len(rows), i + k + 1)
    s = {"t0": lo, "i": i}
    for name, (a, b) in {"label": ("lx", "ly"), "model": ("mx", "my"),
                         "traj": ("tx", "ty")}.items():
        s[name] = [[round(f(rows[j][a]), 1) if np.isfinite(f(rows[j][a])) else None,
                    round(f(rows[j][b]), 1) if np.isfinite(f(rows[j][b])) else None]
                   for j in range(lo, hi)]
    return s


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
        man = frames[v]
        scene = Image.open(COORDS_DATA / v / "scene" / f"{i:06d}.jpg").convert("RGB")
        mini = Image.open(COORDS_DATA / v / "mini" / f"{i:06d}.jpg").convert("RGB")
        h = hyps(r)
        box = boxes.get(v, (68, 38))
        a = agent.get((v, i))
        items.append({
            "key": f"{v}:{i}", "video_id": v, "idx": i, "t_sec": r["t_sec"],
            "stratum": p["stratum"], "also_random": bool(p.get("also_random")),
            "score": p["score"], "quality": f(r["quality"]), "game": r.get("game", ""),
            "hyp": h, "box": box, "series": series(man, i),
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
        "items": items, "fp": fp.round(1).tolist(), "px": PX, "mm": MM,
        "colors": COLORS, "names": NAMES,
        "summary": {"A": summ["A"]}}, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"кадров {len(items)} (страты: "
          + ", ".join(f"{s} {sum(1 for x in items if x['stratum'] == s)}"
                      for s in STRATA)
          + f") → {args.out}")


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Шум меток: ключевые кадры</title>
<style>
 :root { color-scheme: dark; }
 * { box-sizing:border-box; }
 body { margin:0; padding:20px 16px 60px; background:#14161a; color:#e8e8ea;
        font:14px/1.45 system-ui,Segoe UI,sans-serif; }
 .wrap { max-width:1500px; margin:0 auto; }
 h1 { font-size:20px; margin:0 0 4px; }
 .lead { color:#9aa0aa; margin-bottom:14px; max-width:1000px; }
 .bar { position:sticky; top:0; z-index:5; background:#14161a; padding:8px 0;
        border-bottom:1px solid #2a2e36; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
 button, select { background:#20242b; color:#e8e8ea; border:1px solid #3a3f49;
        border-radius:6px; padding:5px 10px; font:inherit; cursor:pointer; }
 button:hover { border-color:#5a7bf0; }
 button.on { background:#2d4fd0; border-color:#5a7bf0; }
 .card { background:#1a1d23; border:1px solid #2a2e36; border-radius:10px;
         padding:12px; margin:14px 0; }
 .card.done { border-color:#2f6b45; }
 .head { display:flex; gap:12px; flex-wrap:wrap; align-items:baseline; margin-bottom:8px; }
 .head b { font-size:15px; }
 .tag { font-size:12px; padding:1px 7px; border-radius:9px; background:#2a2e36; color:#cfd4dc; }
 .tag.model_vs_label { background:#3b2a52; } .tag.label_vs_traj { background:#523a2a; }
 .tag.random { background:#2a4a3a; } .tag.clip_vs_label { background:#4a4a2a; }
 .row { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; }
 .scene { max-width:100%; width:720px; border-radius:6px; }
 svg { background:#0c0d10; border-radius:6px; }
 .legend span { margin-right:12px; font-size:12px; }
 .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
 .nums { font:12px/1.5 ui-monospace,Consolas,monospace; color:#b9bec8; white-space:pre; }
 .verd { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; align-items:center; }
 .agent { font-size:12px; color:#e0b46a; }
 textarea { background:#101216; color:#e8e8ea; border:1px solid #3a3f49; border-radius:6px;
            width:min(600px,100%); height:34px; font:inherit; }
 .hide { display:none; }
 code { background:#20242b; padding:1px 5px; border-radius:4px; }
</style></head><body><div class="wrap">
<h1>Шум меток с миникарты: ключевые кадры</h1>
<div class="lead">Метка — центр белой рамки на миникарте (целый пиксель сетки 256, шаг ≈58 ед.).
Модель — патчевая модель по сцене без миникарты. Траектория — положение камеры по сдвигам самой
сцены между соседними кадрами, привязанное ко всем меткам цепочки сразу (своя метка кадра в
неё почти не входит). Прямоугольники — рамка размера ролика в каждой гипотезе, тонкая трапеция —
след полного кадра по гомографии (для метки и модели). Сравнивайте белую рамку игры с гипотезами
и картинку сцены с местностью миникарты. Расхождения меньше пикселя миникарты (≈58 ед.) глазом
не различить — для них честный вердикт «обе верны» или «не разобрать».
Вердикты сохраняются в этом браузере; выгрузка — кнопками справа.</div>
<div class="bar">
 <span>Страта:</span>
 <button data-f="all" class="on">все</button>
 <button data-f="model_vs_label">модель ↔ метка</button>
 <button data-f="label_vs_traj">метка ↔ траектория</button>
 <button data-f="clip_vs_label">рамка с обрезкой ↔ метка</button>
 <button data-f="random">случайные</button>
 <label><input type="checkbox" id="todo"> только без вердикта</label>
 <span id="count" style="color:#9aa0aa"></span>
 <span style="flex:1"></span>
 <button id="csv">CSV</button><button id="json">JSON</button>
 <button id="clear" title="стереть вердикты этого браузера">сброс</button>
</div>
<div id="list"></div>
<div class="lead" style="margin-top:30px">Выгрузку CSV положить в
<code>dataset/label_noise/verdicts.csv</code> (формат описан в <code>dataset/README.md</code>) и
закоммитить. Вердикты агента лежат отдельно в <code>verdicts_agent.csv</code> и в выгрузку не входят.</div>
</div>
<script>
const D = __DATA__;
const KEY = "label_noise_verdicts_v1";
const VERD = [["label","права метка"],["model","права модель"],["both_ok","обе верны"],
              ["both_wrong","обе мимо"],["unclear","не разобрать"]];
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function mapSvg(it, win) {
  // win: [x0, y0, size] в пикселях миникарты 256; точки: ед. / px + 0.5
  const [x0, y0, s] = win, px = D.px;
  const W = 420;
  let g = `<svg width="${W}" height="${W}" viewBox="${x0} ${y0} ${s} ${s}">`;
  g += `<image href="${it.mini}" x="0" y="0" width="${D.mm}" height="${D.mm}" style="image-rendering:${s <= 100 ? "pixelated" : "auto"}"/>`;
  const sw = s / W;
  for (const k of ["traj","clip","model","sub","label"]) {
    const p = it.hyp[k]; if (!p) continue;
    const cx = p[0] / px + 0.5, cy = p[1] / px + 0.5, c = D.colors[k];
    const [bw, bh] = it.box;
    g += `<rect x="${cx-bw/2}" y="${cy-bh/2}" width="${bw}" height="${bh}" fill="none" stroke="${c}" stroke-width="${(k=="label"?2.2:1.5)*sw}" stroke-dasharray="${k=="traj"?4*sw:0}"/>`;
    g += `<circle cx="${cx}" cy="${cy}" r="${4*sw}" fill="none" stroke="${c}" stroke-width="${2*sw}"/>`;
    if (k == "label" || k == "model") {
      const pts = D.fp.map(q => `${cx + q[0]/px},${cy + q[1]/px}`).join(" ");
      g += `<polygon points="${pts}" fill="none" stroke="${c}" stroke-opacity="0.7" stroke-width="${1*sw}"/>`;
    }
  }
  return g + "</svg>";
}

function chart(it) {
  const s = it.series, W = 300, H = 90;
  let out = "";
  for (const ax of [0, 1]) {
    const vals = [];
    for (const k of ["label","model","traj"]) for (const v of s[k]) if (v[ax] != null) vals.push(v[ax]);
    if (!vals.length) continue;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 200) { const m = (hi + lo) / 2; lo = m - 100; hi = m + 100; }
    const n = s.label.length, X = j => 6 + j * (W - 12) / Math.max(n - 1, 1),
          Y = v => H - 6 - (v - lo) / (hi - lo) * (H - 12);
    let g = `<svg width="${W}" height="${H}"><text x="4" y="12" fill="#9aa0aa" font-size="11">${ax ? "y" : "x"}, диапазон ${Math.round(hi-lo)} ед.</text>`;
    const ci = s.i - s.t0;
    g += `<line x1="${X(ci)}" x2="${X(ci)}" y1="0" y2="${H}" stroke="#555"/>`;
    const tr = s.traj.map((v, j) => v[ax] == null ? null : `${X(j)},${Y(v[ax])}`);
    let seg = [];
    for (const p of tr.concat([null])) { if (p) seg.push(p); else { if (seg.length > 1) g += `<polyline points="${seg.join(" ")}" fill="none" stroke="${D.colors.traj}" stroke-width="1.5"/>`; seg = []; } }
    s.label.forEach((v, j) => { if (v[ax] != null) g += `<circle cx="${X(j)}" cy="${Y(v[ax])}" r="2.5" fill="${D.colors.label}"/>`; });
    s.model.forEach((v, j) => { if (v[ax] != null) g += `<path d="M${X(j)-3},${Y(v[ax])-3}l6,6m0,-6l-6,6" stroke="${D.colors.model}"/>`; });
    out += g + "</svg>";
  }
  return out;
}

function dist(a, b) { return (a && b) ? Math.hypot(a[0]-b[0], a[1]-b[1]).toFixed(0) : "—"; }

function card(it) {
  const h = it.hyp, hs = Object.values(h);
  // окно увеличения — вокруг середины всех гипотез, чтобы влезли все рамки
  const c = [hs.reduce((a, p) => a + p[0], 0) / hs.length / D.px + 0.5,
             hs.reduce((a, p) => a + p[1], 0) / hs.length / D.px + 0.5];
  const v = store[it.key] || {};
  const nums = `метка–модель ${dist(h.label, h.model)} ед.   метка–траектория ${dist(h.label, h.traj)}   модель–траектория ${dist(h.model, h.traj)}
метка–субпиксель ${dist(h.label, h.sub)}   качество рамки ${it.quality.toFixed(2)}   P(игра) ${it.game === "" ? "—" : (+it.game).toFixed(2)}`;
  const legend = ["label","model","traj","sub","clip"].filter(k => h[k]).map(k => `<span><i style="background:${D.colors[k]}"></i>${D.names[k]}</span>`).join("");
  return `<div class="card ${v.verdict ? "done" : ""}" data-key="${it.key}" data-s="${it.stratum}">
  <div class="head"><b>${esc(it.video_id)} · кадр ${it.idx} · ${it.t_sec} с</b>
   <span class="tag ${it.stratum}">${it.stratum}${it.also_random ? " + random" : ""}</span>
   <span class="tag">оценка ${it.score}</span>
   <a href="https://www.youtube.com/watch?v=${esc(it.video_id)}&t=${Math.floor(+it.t_sec)}s" target="_blank" style="color:#8fb0ff">ролик</a>
   ${it.agent ? `<span class="agent">агент: ${esc(it.agent.verdict)}${it.agent.note ? " — " + esc(it.agent.note) : ""}</span>` : ""}</div>
  <div class="row"><img class="scene" src="${it.scene}" alt="">
   <div>${mapSvg(it, [c[0]-50, c[1]-50, 100])}</div>
   <div>${mapSvg(it, [0, 0, D.mm])}</div></div>
  <div class="row" style="margin-top:6px"><div class="legend">${legend}</div></div>
  <div class="row" style="margin-top:6px"><div class="nums">${nums}</div>${chart(it)}</div>
  <div class="verd">${VERD.map(([k, t]) => `<button data-v="${k}" class="${v.verdict == k ? "on" : ""}">${t}</button>`).join("")}
   <textarea placeholder="заметка">${esc(v.note || "")}</textarea></div></div>`;
}

let filt = "all";
function render() {
  const todo = document.getElementById("todo").checked;
  const list = D.items.filter(it => (filt == "all" || it.stratum == filt || (filt == "random" && it.also_random))
                                    && (!todo || !(store[it.key] || {}).verdict));
  document.getElementById("list").innerHTML = list.map(card).join("");
  const done = D.items.filter(it => (store[it.key] || {}).verdict).length;
  document.getElementById("count").textContent = `показано ${list.length} · с вердиктом ${done} из ${D.items.length}`;
}
document.querySelectorAll(".bar [data-f]").forEach(b => b.onclick = () => {
  filt = b.dataset.f; document.querySelectorAll(".bar [data-f]").forEach(x => x.classList.toggle("on", x == b)); render(); });
document.getElementById("todo").onchange = render;
document.getElementById("list").addEventListener("click", e => {
  const b = e.target.closest("button[data-v]"); if (!b) return;
  const key = b.closest(".card").dataset.key;
  store[key] = Object.assign(store[key] || {}, {verdict: b.dataset.v}); save();
  const cardEl = b.closest(".card"); cardEl.classList.add("done");
  cardEl.querySelectorAll("button[data-v]").forEach(x => x.classList.toggle("on", x == b));
  const done = D.items.filter(it => (store[it.key] || {}).verdict).length;
  document.getElementById("count").textContent = document.getElementById("count").textContent.replace(/с вердиктом \d+/, "с вердиктом " + done);
});
document.getElementById("list").addEventListener("input", e => {
  if (e.target.tagName != "TEXTAREA") return;
  const key = e.target.closest(".card").dataset.key;
  store[key] = Object.assign(store[key] || {}, {note: e.target.value}); save();
});
function rows() {
  return D.items.filter(it => (store[it.key] || {}).verdict).map(it => ({
    video_id: it.video_id, idx: it.idx, t_sec: it.t_sec, stratum: it.stratum,
    also_random: it.also_random ? 1 : 0, verdict: store[it.key].verdict, note: store[it.key].note || ""}));
}
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; a.click();
}
document.getElementById("csv").onclick = () => {
  const r = rows(), cols = ["video_id","idx","t_sec","stratum","also_random","verdict","note"];
  const q = v => /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v);
  download("verdicts.csv", [cols.join(",")].concat(r.map(x => cols.map(c => q(x[c])).join(","))).join("\n") + "\n", "text/csv");
};
document.getElementById("json").onclick = () => download("verdicts.json", JSON.stringify(rows(), null, 1), "application/json");
document.getElementById("clear").onclick = () => { if (confirm("Стереть вердикты этого браузера?")) { store = {}; save(); render(); } };
render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
