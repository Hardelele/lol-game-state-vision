"""Разбор одного промаха: где именно ответ разошёлся с истиной.

У сети прямого распространения нет «правильного» промежуточного состояния,
с которым можно было бы сравнить её собственное, поэтому вопрос «на каком
слое ошибка» сам по себе некорректен. Здесь он ставится так, чтобы на него
можно было ответить измерением:

1. Что на самом деле находится там, куда показала модель. Если сцена в том
   месте карты похожа на разбираемый кадр, ответ объясняется сходством
   местности, а не поломкой.
2. На каком слое кадр перестал быть похож на «кадры из истинного места» и
   стал похож на «кадры из предсказанного». Для каждого этапа берутся
   признаки, усреднённые по кадру, и считается, к какому из двух облаков
   кадр ближе. Слой, после которого перевес переходит к неверному облаку, и
   есть точка расхождения.
3. Какие части кадра тянут ответ в сторону. Каждая ячейка сетки по очереди
   закрывается, и смотрится, как меняется вероятность истинного места. Если
   от закрытия ячейки становится лучше — эта ячейка и сбивала.
4. Короткий вывод из этих чисел.

Пример:
    python tools/explain_miss.py coords/heatmap olmTXkkUv58 4
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coord_model import CoordNet, GRID, grid_centers
from paths import DATASET, ROOT

DATA = ROOT / "data" / "coords"
RUNS = ROOT / "runs"
MAP_UNITS = 14800
NEAR = 0.035          # «рядом» по карте, в долях карты
POOL = 48             # сколько кадров берём в каждое облако


def b64(img: Image.Image, fmt="JPEG", q=84) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=q) if fmt == "JPEG" else img.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load_model(run: str, dev: str):
    ck = torch.load(RUNS / run / "model.pt", map_location=dev, weights_only=False)
    m = CoordNet(grid=ck.get("grid", GRID)).to(dev)
    m.encoder.load_state_dict(ck["encoder"])
    m.head.load_state_dict(ck["head"])
    m.eval()
    return m, tuple(ck["input_size"]), ck.get("trained_on", [])


def coords_rows(vid: str) -> list[dict]:
    return list(csv.DictReader((DATA / f"{vid}.csv").open(encoding="utf-8")))


def load_batch(paths: list[Path], size, dev) -> torch.Tensor:
    out = []
    for p in paths:
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize(size, Image.BILINEAR),
                           dtype=np.float32) / 255.0
        out.append(a.transpose(2, 0, 1) * 2 - 1)
    return torch.from_numpy(np.stack(out)).to(dev)


@torch.no_grad()
def stage_means(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Признаки каждого этапа, усреднённые по кадру: (B, C) на этап."""
    caught = {}

    def hook(name):
        def fn(_m, _i, out):
            caught[name] = out.detach().mean(dim=(2, 3))
        return fn

    hs = []
    for name, mod in model.encoder.stages.named_modules():
        if isinstance(mod, torch.nn.ReLU) and name:
            hs.append(mod.register_forward_hook(hook("stages." + name)))
    logits = model(x)
    for h in hs:
        h.remove()
    caught["вход"] = x.mean(dim=(2, 3))
    return caught, logits


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1 - F.cosine_similarity(a.flatten()[None], b.flatten()[None]))


def pick_pool(rows: list[dict], point, n: int, exclude=None) -> list[dict]:
    """Кадры, чья истинная точка ближе всего к заданной."""
    cand = []
    for r in rows:
        if float(r["quality"]) < 0.5:
            continue
        if exclude and (r["video_id"], int(r["idx"])) == exclude:
            continue
        d = np.hypot(float(r["cx"]) - point[0], float(r["cy"]) - point[1])
        cand.append((d, r))
    cand.sort(key=lambda t: t[0])
    return [r for d, r in cand[:n]]


def stage_titles(model) -> list[str]:
    names = ["вход"]
    for bi, block in enumerate(model.encoder.stages):
        k = 0
        for i, m in enumerate(block):
            if isinstance(m, torch.nn.ReLU):
                k += 1
                names.append(f"блок {bi + 1} · после ReLU {k}")
    return names


def stage_keys(model) -> list[str]:
    keys = ["вход"]
    for name, mod in model.encoder.stages.named_modules():
        if isinstance(mod, torch.nn.ReLU) and name:
            keys.append("stages." + name)
    return keys


def build(run: str, vid: str, idx: int, cells=(8, 5)) -> str:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size, trained = load_model(run, dev)
    grid = model.grid

    rows = {r["idx"]: r for r in coords_rows(vid)}
    me = rows[str(idx)] if str(idx) in rows else rows[f"{idx}"]
    true = (float(me["cx"]), float(me["cy"]))
    scene_path = DATA / me["scene"]

    x = load_batch([scene_path], size, dev)
    with torch.no_grad():
        logits = model(x).float()
        prob = F.softmax(logits, 1)[0]
    centers = grid_centers(grid, logits.device)
    pred = tuple(float(v) for v in (prob @ centers))
    err = float(np.hypot(pred[0] - true[0], pred[1] - true[1]))

    # Облака кадров: из обучающих роликов, потому что именно их сеть видела.
    ref_rows = []
    for v in (trained or [vid]):
        f = DATA / f"{v}.csv"
        if f.exists():
            ref_rows += coords_rows(v)
    pool_t = pick_pool(ref_rows, true, POOL, exclude=(vid, idx))
    pool_p = pick_pool(ref_rows, pred, POOL, exclude=(vid, idx))

    xt = load_batch([DATA / r["scene"] for r in pool_t], size, dev)
    xp = load_batch([DATA / r["scene"] for r in pool_p], size, dev)
    mq, _ = stage_means(model, x)
    mt, _ = stage_means(model, xt)
    mp, _ = stage_means(model, xp)

    keys, titles = stage_keys(model), stage_titles(model)
    steps = []
    flip = None
    for k, t in zip(keys, titles):
        q, ct, cp = mq[k][0], mt[k].mean(0), mp[k].mean(0)
        dt, dp = cos_dist(q, ct), cos_dist(q, cp)
        sep = cos_dist(ct, cp)
        # Сырой запас по этапам несравним: облака расходятся вглубь сети, и
        # вместе с ними растёт любой отрыв. Сравнивать нужно долю от
        # разделимости — насколько кадр смещён к истине в масштабе самих облаков.
        rel = (dp - dt) / sep if sep > 1e-6 else 0.0
        steps.append({"name": t, "d_true": dt, "d_pred": dp,
                      "margin": dp - dt, "sep": sep, "rel": rel})
        if flip is None and dp < dt:
            flip = t

    # Закрываем по одной ячейке и смотрим, как меняется вероятность истины.
    cols, rws = cells
    ti = int(np.argmin(np.linalg.norm(
        centers.cpu().numpy() - np.array(true), axis=1)))
    base_p = float(prob[ti])
    H, W = size[1], size[0]
    occ = []
    batch = []
    for r in range(rws):
        for c in range(cols):
            z = x.clone()
            z[:, :, r * H // rws:(r + 1) * H // rws,
              c * W // cols:(c + 1) * W // cols] = 0
            batch.append(z)
    with torch.no_grad():
        pr = F.softmax(model(torch.cat(batch)).float(), 1)[:, ti]
    for i, v in enumerate(pr.tolist()):
        occ.append({"id": f"{i // cols + 1}.{i % cols + 1}",
                    "p": v, "delta": v - base_p})
    worst = sorted(occ, key=lambda o: -o["delta"])[:5]

    return render(run, vid, idx, me, true, pred, err, float(prob.max()),
                  scene_path, pool_t, pool_p, steps, flip, base_p, occ, worst,
                  cells, grid, prob.reshape(grid, grid).cpu().numpy())


def render(run, vid, idx, me, true, pred, err, pmax, scene_path, pool_t, pool_p,
           steps, flip, base_p, occ, worst, cells, grid, heat) -> str:
    with Image.open(scene_path) as im:
        scene = b64(im.convert("RGB").resize((520, round(520 * im.height / im.width))))
    with Image.open(DATA / me["mini"]) as im:
        mini = b64(im.convert("RGB").resize((300, 300)))

    def strip(pool, n=6):
        out = []
        for r in pool[:n]:
            with Image.open(DATA / r["scene"]) as im:
                out.append(f'<figure><img src="{b64(im.convert("RGB").resize((190, 98)))}">'
                           f'<figcaption>{html.escape(r["video_id"][:7])} '
                           f'{float(r["t_sec"]):.0f}с<br>{float(r["cx"]):.3f}, '
                           f'{float(r["cy"]):.3f}</figcaption></figure>')
        return "".join(out)

    rowsh = "".join(
        f'<tr class="{"flip" if s["name"] == flip else ""}">'
        f'<td>{html.escape(s["name"])}</td>'
        f'<td>{s["d_true"]:.4f}</td><td>{s["d_pred"]:.4f}</td>'
        f'<td class="{"neg" if s["margin"] < 0 else "pos"}">{s["margin"]:+.4f}</td>'
        f'<td>{s["sep"]:.4f}</td>'
        f'<td class="{"neg" if s["rel"] < 0.5 else "pos"}">{s["rel"]:.2f}</td>'
        f'</tr>' for s in steps)

    cols, rws = cells
    occmax = max(abs(o["delta"]) for o in occ) or 1e-9
    cellsh = ""
    for i, o in enumerate(occ):
        v = o["delta"] / occmax
        col = (f"rgba(90,255,140,{min(abs(v), 1) * .75:.2f})" if v > 0
               else f"rgba(214,72,60,{min(abs(v), 1) * .75:.2f})")
        cellsh += (f'<div class="oc" style="background:{col}" '
                   f'title="закрыть {o["id"]} → p(истина) {o["p"]:.4f}">'
                   f'{o["id"]}<small>{o["delta"]:+.3f}</small></div>')

    verdict = []
    if pmax < 0.15:
        verdict.append(f"Модель не была уверена: на свою же лучшую ячейку она "
                       f"даёт всего {pmax:.3f}. По порогу отказа такой кадр "
                       f"отбрасывается, то есть это не уверенная ошибка.")
    if flip:
        verdict.append(f"Перевес переходит к неверному месту на этапе "
                       f"<b>{html.escape(flip)}</b>.")
    else:
        first, last = steps[0]["rel"], steps[-1]["rel"]
        if last < first - 0.1:
            verdict.append(
                f"Ни на одном этапе кадр не оказался ближе к неверному месту, но "
                f"в масштабе облаков перевес тает: {first:.2f} на входе против "
                f"{last:.2f} на последнем блоке. То есть признаки постепенно "
                f"теряют то, чем истинное место отличается от предсказанного.")
        else:
            verdict.append(
                "Ни на одном этапе кадр не стал ближе к неверному месту, и "
                "относительный перевес не падает: по усреднённым признакам "
                "сеть на стороне истины.")
    if worst and worst[0]["delta"] > 0.01:
        ids = ", ".join(w["id"] for w in worst if w["delta"] > 0.01)
        verdict.append(f"Закрытие ячеек <b>{ids}</b> повышает вероятность "
                       f"истинного места — именно они сбивают ответ.")
    else:
        verdict.append("Нет отдельной ячейки, закрытие которой заметно "
                       "исправляет ответ: сбивает кадр целиком, а не его часть.")

    return TEMPLATE.format(
        run=html.escape(run), vid=html.escape(vid), idx=idx,
        t=float(me["t_sec"]), scene=scene, mini=mini,
        tx=true[0], ty=true[1], px=pred[0], py=pred[1],
        err=err, units=err * MAP_UNITS, pmax=pmax, base_p=base_p,
        pool_t=strip(pool_t), pool_p=strip(pool_p), rows=rowsh,
        cols=cols, cells=cellsh, verdict=" ".join(verdict),
        yt=f"https://www.youtube.com/watch?v={vid}&t={int(float(me['t_sec']))}s",
        npool=len(pool_t))


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Разбор промаха — {vid} кадр {idx}</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; padding:20px 24px 48px; background:#14161a; color:#e8e8ea;
   font:14px/1.5 system-ui,Segoe UI,sans-serif; max-width:1180px; }}
 h1 {{ font-size:19px; margin:0 0 4px; }}
 h2 {{ font-size:15px; margin:26px 0 8px; border-bottom:1px solid #2a2e36;
   padding-bottom:6px; }}
 .lead {{ color:#9aa0aa; margin-bottom:14px; }}
 .row {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
 figure {{ margin:0; }} figure img {{ display:block; border-radius:6px; }}
 figcaption {{ color:#878d98; font-size:11px; margin-top:3px; }}
 table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
 th,td {{ padding:4px 8px; text-align:right; border-bottom:1px solid #22262e; }}
 th:first-child, td:first-child {{ text-align:left; }}
 th {{ color:#9aa0aa; font-weight:500; }}
 tr.flip td {{ background:#3b2420; }}
 .pos {{ color:#6cd08a; }} .neg {{ color:#ff7b6e; }}
 .facts td:first-child {{ color:#9aa0aa; }}
 .grid {{ display:grid; grid-template-columns:repeat({cols},1fr); gap:3px;
   max-width:560px; }}
 .oc {{ border-radius:4px; padding:6px 4px; text-align:center; font-size:11px;
   border:1px solid #2a2e36; }}
 .oc small {{ display:block; color:#cfd4dc; }}
 .verdict {{ background:#1a1d23; border:1px solid #2a2e36; border-radius:10px;
   padding:12px 14px; margin-top:12px; }}
 a {{ color:#8ab4ff; }}
 .note {{ color:#767d89; font-size:12px; margin-top:8px; }}
</style></head><body>

<h1>Разбор промаха · {vid}, кадр {idx}</h1>
<div class="lead">прогон {run} · <a href="{yt}" target="_blank">момент {t:.0f} с
на YouTube</a></div>

<h2>Шаг 0. Что произошло</h2>
<div class="row">
  <figure><img src="{scene}"><figcaption>вход модели</figcaption></figure>
  <figure><img src="{mini}"><figcaption>миникарта этого кадра</figcaption></figure>
  <table class="facts">
    <tr><td>истина</td><td>{tx:.3f}, {ty:.3f}</td></tr>
    <tr><td>модель</td><td>{px:.3f}, {py:.3f}</td></tr>
    <tr><td>промах</td><td>{err:.3f}</td></tr>
    <tr><td>в игровых единицах</td><td>{units:.0f}</td></tr>
    <tr><td>уверенность в своей ячейке</td><td>{pmax:.3f}</td></tr>
    <tr><td>вероятность истинной ячейки</td><td>{base_p:.4f}</td></tr>
  </table>
</div>

<h2>Шаг 1. Что находится там, куда показала модель</h2>
<div class="lead">Сверху — кадры из обучающих роликов, снятые рядом с истинной
точкой. Снизу — рядом с предсказанной. Если нижние похожи на разбираемый кадр,
ответ объясняется сходством местности.</div>
<div class="row">{pool_t}</div>
<div class="row" style="margin-top:10px">{pool_p}</div>

<h2>Шаг 2. На каком этапе перевес ушёл к неверному месту</h2>
<div class="lead">Для каждого этапа признаки усредняются по кадру и считается
косинусное расстояние до среднего по {npool} кадрам из истинного места и до
среднего по кадрам из предсказанного. Запас — насколько кадр ближе к истине;
отрицательный означает, что он уже похож на чужое место. Разделимость
показывает, насколько сами облака различаются на этом этапе. Сравнивать по
этапам нужно последний столбец: сырой запас растёт просто потому, что вглубь
сети расходятся сами облака.</div>
<div class="lead"><b>Чего эта таблица не говорит.</b> Признаки усредняются по
кадру, то есть расположение найденного внутри кадра отбрасывается. А голова
смотрит именно на расположение: перед ней стоит пулинг до сетки 5×9, а не до
одного числа на канал. Поэтому «перевес на стороне истины» означает только то,
что набор откликов похож на истинное место, и не доказывает, что правильно
разложенные по кадру признаки тоже верны.</div>
<table>
  <tr><th>этап</th><th>до истины</th><th>до предсказания</th><th>запас</th>
    <th>разделимость</th><th>запас / разделимость</th></tr>
  {rows}
</table>

<h2>Шаг 3. Какие части кадра сбивают</h2>
<div class="lead">Каждая ячейка по очереди закрывается чёрным, и заново
считается вероятность истинного места. Зелёная ячейка — без неё ответ лучше,
то есть она мешала. Красная — без неё хуже, она помогала.</div>
<div class="grid">{cells}</div>

<div class="verdict">{verdict}</div>
<div class="note">Все числа получены на этом кадре и на кадрах обучающих
роликов. Облака собираются по истинным координатам с миникарты, предсказание
модели в их отборе не участвует.</div>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("video")
    ap.add_argument("idx", type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    html_doc = build(args.run, args.video, args.idx)
    out = args.out or (ROOT / "runs" / "inspect" /
                       f"miss-{args.video}-{args.idx}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"отчёт ({out.stat().st_size / 1e6:.1f} МБ) → {out}")


if __name__ == "__main__":
    main()
