"""Готовность заклинаний призывателя (D/F) с HUD вида игрока: данные и проверка (#806).

Подкоманды (подробно — docs/hud-spells.md):

* `extract <id>` — вырезы низа кадра (где панель умений при любом масштабе
  HUD) из скачанного ролика `data/videos/<id>.mp4`, 1 кадр/с, в
  `data/hud_spells/<id>/crops/t<сек>.jpg`; источник ролика — в
  `dataset/hud_spells/<id>/source.json` (как у dataset/not_game);
* `locate <id>…` — панель умений по медиане вырезов (lolvision.hud_spells.
  locate_panel): масштаб HUD и место слотов → `data/hud_spells/<id>/panel.json`;
  и сколько одиночных кадров находят то же место;
* `draft <id>…` — черновик разметки грубым правилом (эталон по окну ±150 с,
  читателю недоступен) → `dataset/hud_spells/<id>/draft.csv`;
* `strips <id>…` — ленты слотов Q/D/F по минутам с цветом ответа читателя
  или разметки (`--labels`) — для проверки глазами;
* `questions <id>…` — лист интервалов разметки, которые не «готово» и не
  «перезарядка» (запрет, смерть, «?»): панель целиком, решить глазами;
* `score` — читатель по роликам подряд против разметки (`labels.csv`):
  матрица «класс × ответ», по заклинаниям и роликам, лист-контакт ошибок и
  выборка ответов для проверки глазами → `runs/hud_spells/score/`.

Видео качается так же, как остальные ролики (dataset/README.md):
    python -m yt_dlp -f "bv*[height<=1080][fps<=60]" --write-info-json \
        -o "data/videos/%(id)s.%(ext)s" <url>

Пример:
    python tools/hud_spells.py extract JKZgBDX1WfY
    python tools/hud_spells.py locate JKZgBDX1WfY
    python tools/hud_spells.py score --split test
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision import hud_spells as hs
from lolvision.paths import DATA, DATASET, RUNS, VIDEOS
from lolstream.video_stream import frames, probe

SPELLS_DATA = DATA / "hud_spells"
SPELLS_LABELS = DATASET / "hud_spells"
# Вырез низа кадра: панель умений внизу по центру при любом масштабе HUD
# (доли кадра x0, y0, x1, y1).
CROP = (0.20, 0.74, 0.80, 1.0)


def game_running() -> bool:
    """Идёт ли сейчас игра пользователя (Live Client отвечает временем матча):
    тогда видеокарту не трогаем — распаковка на процессоре."""
    try:
        out = subprocess.run(["curl", "-sk", "-m2",
                              "https://127.0.0.1:2999/liveclientdata/gamestats"],
                             capture_output=True, text=True, timeout=5).stdout
        return "gameTime" in out
    except (OSError, subprocess.SubprocessError):
        return False


def source_json(vid: str) -> dict:
    """Источник ролика из .info.json yt-dlp — те же поля, что у dataset/not_game."""
    info = json.loads((VIDEOS / f"{vid}.info.json").read_text(encoding="utf-8"))
    return {"video_id": vid, "url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
            "title": info.get("title"), "channel": info.get("channel"),
            "channel_url": info.get("channel_url"), "upload_date": info.get("upload_date"),
            "duration_sec": info.get("duration"), "width": info.get("width"),
            "height": info.get("height"), "fps": info.get("fps"),
            "format_id": info.get("format_id"), "license": info.get("license"),
            "sampling": {"fps": 1.0, "crop": list(CROP),
                         "tool": "tools/hud_spells.py extract (вырезы низа кадра, "
                                 "видео в data/videos, в Git не идёт)"}}


def cmd_extract(args) -> None:
    vid = args.video
    video = VIDEOS / f"{vid}.mp4"
    out = SPELLS_DATA / vid / "crops"
    out.mkdir(parents=True, exist_ok=True)
    src = SPELLS_LABELS / vid / "source.json"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(json.dumps(source_json(vid), ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    w, h, dur = probe(video)
    x0, y0, x1, y1 = (round(CROP[0] * w), round(CROP[1] * h), round(CROP[2] * w),
                      round(CROP[3] * h))
    decode = "cpu" if game_running() else args.decode
    n = 0
    for i, fr in enumerate(frames(video, w, h, 1.0, decode=decode)):
        p = out / f"t{i:05d}.jpg"
        if p.exists():
            continue
        Image.fromarray(fr[y0:y1, x0:x1]).save(p, quality=93)
        n += 1
    print(f"{vid}: {w}x{h}, {dur:.0f} с, новых вырезов {n} → {out}")


def crops(vid: str) -> list[Path]:
    return sorted((SPELLS_DATA / vid / "crops").glob("t*.jpg"))


def load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def video_panel(vid: str, every: int = 10) -> hs.Panel:
    """Панель ролика по медиане вырезов (панель неподвижна, остальное
    меняется); кеш — data/hud_spells/<id>/panel.json."""
    path = SPELLS_DATA / vid / "panel.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return hs.Panel(d["x"], d["y"], d["scale"], d["score"])
    fs = crops(vid)[::every]
    med = np.median(np.stack([load(p) for p in fs]), axis=0).astype(np.uint8)
    p = hs.locate_panel(med, whole=True)
    path.write_text(json.dumps({"x": p.x, "y": p.y, "scale": p.scale, "score": p.score,
                                "median_of": len(fs)}) + "\n", encoding="utf-8")
    return p


def cmd_locate(args) -> None:
    for vid in args.videos:
        p = video_panel(vid)
        fs = crops(vid)
        # Ищется ли та же панель по одиночным кадрам — как в живом сеансе.
        single = [hs.locate_panel(load(f), whole=True) for f in fs[30::max(1, len(fs) // 12)]]
        same = sum(abs(q.x - p.x) <= hs.AGREE_PX and abs(q.y - p.y) <= hs.AGREE_PX
                   and q.score >= hs.FOUND_MIN for q in single)
        print(f"{vid}: масштаб {p.scale:.2f}, иконка {p.slot_px():.0f} px, совпадение "
              f"медианы {p.score:.2f}; одиночных кадров на том же месте {same} из {len(single)}")


def run_video(vid: str, learn: bool = True) -> list[dict]:
    """Читатель по всем вырезам ролика подряд, как в сеансе (эталоны копятся)."""
    p = video_panel(vid)
    hud = hs.SpellHud()
    hud.set_panel(p)
    rows = []
    for f in crops(vid):
        t = float(f.stem[1:])
        fr = load(f)
        score = hs.panel_score(fr, p)
        present = score >= hs.PRESENT_MIN
        dead = present and hs.hp_fill(fr, p) < hs.DEAD_HP
        for k in hs.SPELL_SLOTS:
            c = hs.slot_crop(fr, p, k)
            r = hs.decide(k, c, hud.refs[k], present, dead, t, learn=learn)
            ft = r.feat
            rows.append({"video": vid, "t": t, "slot": k, "state": r.state,
                         "sim": r.sim, "vrel": r.vrel, "score": score,
                         "navy": None if ft is None else ft.navy,
                         "white": None if ft is None else ft.white,
                         "cross": None if ft is None else ft.cross,
                         "v": None if ft is None else ft.v})
    return rows


LABELS = ("ready", "cooldown", "blocked", "dead", "hidden", "?")


def frame_table(vid: str) -> dict:
    """Признаки всех вырезов ролика: присутствие панели, ХП, вырезы слотов."""
    p = video_panel(vid)
    out = {"t": [], "score": [], "hp": [], "crop": {k: [] for k in hs.SPELL_SLOTS}}
    for f in crops(vid):
        fr = load(f)
        out["t"].append(float(f.stem[1:]))
        out["score"].append(hs.panel_score(fr, p))
        out["hp"].append(hs.hp_fill(fr, p))
        for k in hs.SPELL_SLOTS:
            out["crop"][k].append(hs.slot_crop(fr, p, k))
    return out


def draft_states(tab: dict, k: str, win: int = 150) -> list[str]:
    """Черновик разметки слота — грубое правило для проверки глазами, не
    читатель: эталон — медиана «не синих» вырезов в окне ±win с (прошлое и
    будущее, читателю недоступно). По кадру: крестик → запрет; синяя
    заливка ≥ 0.30 и иконка тёмная → перезарядка; почти совпадает с эталоном
    (NCC ≥ 0.9, яркость ≥ 0.9) → готово; тёмная (< 0.7 эталона) или с
    заливкой ≥ 0.15, с цифрами, не мёртв → перезарядка; полоса ХП пуста →
    мёртв; иначе «?» — решать глазами."""
    n = len(tab["t"])
    cs = tab["crop"][k]
    fs = [hs.slot_features(c) for c in cs]
    present = [sc >= hs.PRESENT_MIN for sc in tab["score"]]
    dead = [present[i] and tab["hp"][i] < 0.005 for i in range(n)]
    cand = [present[i] and not dead[i] and (fs[i].navy < 0.25 or fs[i].v >= 150)
            and fs[i].cross < 0.1
            and not (fs[i].digits >= 0.03 and fs[i].v < 115) for i in range(n)]
    out = []
    for i in range(n):
        f = fs[i]
        if not present[i]:
            out.append("hidden")
            continue
        idx = [j for j in range(max(0, i - win), min(n, i + win)) if cand[j]]
        sim = vrel = None
        if len(idx) >= 5:
            ref = np.median(np.stack([cs[j] for j in idx[::max(1, len(idx) // 40)]]), 0)
            ref = ref.astype(np.uint8)
            sim = hs.ncc(cs[i], ref)
            vrel = f.v / max(hs.slot_features(ref).v, 1)
        if f.cross >= 0.2:
            out.append("blocked")
        elif f.navy >= 0.30 and f.v < 140:
            out.append("cooldown")
        elif sim is not None and sim >= 0.9 and vrel >= 0.9:
            out.append("ready")
        elif (((vrel is not None and vrel < 0.7) or (f.navy >= 0.15 and f.v < 120))
              and f.digits >= 0.03 and not dead[i]):
            out.append("cooldown")
        elif dead[i]:
            out.append("dead")
        else:
            out.append("?")
    return out


def intervals(ts: list[float], states: list[str]) -> list[tuple[float, float, str]]:
    out = []
    for t, s in zip(ts, states):
        if out and out[-1][2] == s and t == out[-1][1]:
            out[-1] = (out[-1][0], t + 1, s)
        else:
            out.append((t, t + 1, s))
    return out


def cmd_draft(args) -> None:
    import csv
    for vid in args.videos:
        path = SPELLS_LABELS / vid / "draft.csv"
        tab = frame_table(vid)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["slot", "start_sec", "end_sec", "state", "note"])
            for k in hs.SPELL_SLOTS:
                for a, b, st in intervals(tab["t"], draft_states(tab, k)):
                    w.writerow([k, int(a), int(b), st, ""])
        print(path)


def read_labels(vid: str, name: str = "labels.csv") -> dict[tuple[float, str], str]:
    """Разметка интервалами → {(t, слот): класс}; поздние строки сильнее."""
    import csv
    path = SPELLS_LABELS / vid / name
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        for t in range(int(r["start_sec"]), int(r["end_sec"])):
            out[(float(t), r["slot"])] = r["state"]
    return out


COLORS = {"?": (255, 0, 255), "ready": (40, 200, 40), "cooldown": (40, 90, 230),
          "blocked": (230, 30, 30),
          "dead": (150, 150, 150), "hidden": (0, 0, 0), "unsure": (240, 200, 0)}


def cmd_strips(args) -> None:
    """Ленты Q / D / F по 60 с в строке, 10 минут на лист; под вырезом D/F —
    полоска цвета ответа читателя (зелёный готово, синий перезарядка,
    красный запрет, серый мёртв, жёлтый не уверен, чёрный панели нет)."""
    from PIL import ImageDraw
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for vid in args.videos:
        p = video_panel(vid)
        if args.labels:
            rows = read_labels(vid, args.labels)
        else:
            rows = {(r["t"], r["slot"]): r["state"] for r in run_video(vid)}
        fs = crops(vid)
        z, per, block = args.size, 60, 600
        for b0 in range(0, len(fs), block):
            part = fs[b0:b0 + block]
            nrow = (len(part) + per - 1) // per
            rh = 12 + 3 * z + 2 * 4
            sheet = Image.new("RGB", (per * z, nrow * (rh + 4)), (60, 0, 60))
            d = ImageDraw.Draw(sheet)
            for i, f in enumerate(part):
                t = float(f.stem[1:])
                r, c = divmod(i, per)
                x, y = c * z, r * (rh + 4)
                fr = load(f)
                if int(t) % 10 == 0:
                    d.text((x + 1, y), str(int(t)), fill=(255, 255, 0))
                yy = y + 12
                for k in ("Q", "D", "F"):
                    sheet.paste(Image.fromarray(hs.slot_crop(fr, p, k, z - 2)), (x, yy))
                    yy += z
                    if k != "Q":
                        d.rectangle([x, yy - 1, x + z - 2, yy + 2],
                                    fill=COLORS[rows.get((t, k), "?")])
                        yy += 4
            name = out / f"{vid}_{b0 // block}.png"
            sheet.save(name)
            print(name)


SPLIT = {"tune": ["JKZgBDX1WfY", "InEwM0UBAbc", "8lkxSQHq81k", "1Z9DJyLsXHo", "Pq2V04mhRTE"],
         "test": ["tEMffCUg-Js", "htjWVBQd0gw", "D37cbVNs0EU", "LIZxk-kM-6o", "ANXVB571YkE"]}
CLASSES = ("ready", "cooldown", "blocked", "dead", "hidden")


def read_spells(vid: str) -> dict[tuple[float, str], str]:
    """Какое заклинание в слоте (dataset/hud_spells/<id>/spells.csv, глазами)."""
    import csv
    path = SPELLS_LABELS / vid / "spells.csv"
    out = {}
    if path.exists():
        for r in csv.DictReader(path.open(encoding="utf-8")):
            for t in range(int(r["start_sec"]), int(r["end_sec"])):
                out[(float(t), r["slot"])] = r["spell"]
    return out


def summary(cn) -> str:
    """Строка метрик по счётчику (класс разметки, ответ)."""
    notready = sum(cn[(c, p)] for c in CLASSES if c != "ready" for p in ("true", "false", "null"))
    fr = sum(cn[(c, "true")] for c in CLASSES if c != "ready")
    rd = sum(cn[("ready", p)] for p in ("true", "false", "null"))
    cd = sum(cn[("cooldown", p)] for p in ("true", "false", "null"))
    tr = sum(cn[(c, "true")] for c in CLASSES)
    fa = sum(cn[(c, "false")] for c in CLASSES)
    return (f"ложное «готово» {fr} из {notready} не готовых ({100 * fr / max(notready, 1):.2f}%); "
            f"точность true {100 * cn[('ready', 'true')] / max(tr, 1):.2f}%; "
            f"точность false {100 * cn[('cooldown', 'false')] / max(fa, 1):.2f}%; "
            f"готово найдено {cn[('ready', 'true')]}/{rd} "
            f"({100 * cn[('ready', 'true')] / max(rd, 1):.1f}%); "
            f"перезарядка найдена {cn[('cooldown', 'false')]}/{cd} "
            f"({100 * cn[('cooldown', 'false')] / max(cd, 1):.1f}%)")


def cmd_score(args) -> None:
    """Читатель по роликам подряд (как в сеансе) против разметки глазами.
    Кадры в пределах ±margin с от смены класса в разметке и класс «?» не
    считаются. Выход — матрица «класс разметки × ответ», по заклинаниям и
    роликам, лист-контакт ошибок «готово ↔ не готово»."""
    from collections import Counter
    vids = args.videos or SPLIT[args.split]
    name = args.split if not args.videos else "_".join(v[:4] for v in vids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conf: Counter = Counter()
    by_spell: dict[str, Counter] = {}
    by_video: dict[str, Counter] = {}
    bad, answered = [], []
    for vid in vids:
        lab = read_labels(vid)
        spells = read_spells(vid)
        for r in run_video(vid):
            key = (r["t"], r["slot"])
            truth = lab.get(key, "?")
            if truth == "?":
                continue
            near = [lab.get((r["t"] + d, r["slot"])) for d in range(-args.margin, args.margin + 1)]
            if any(x is not None and x != truth for x in near):
                continue
            pred = {"ready": "true", "cooldown": "false"}.get(r["state"], "null")
            conf[(truth, pred)] += 1
            by_spell.setdefault(spells.get(key, "?"), Counter())[(truth, pred)] += 1
            by_video.setdefault(vid, Counter())[(truth, pred)] += 1
            answered.append((vid, r["t"], r["slot"], truth, pred))
            if (pred == "true" and truth != "ready") or (pred == "false" and truth != "cooldown"):
                bad.append((vid, r["t"], r["slot"], truth, pred))
    lines = [f"# Готовность D/F по HUD: {name} ({', '.join(vids)}), допуск ±{args.margin} с", "",
             "| разметка \\ ответ | true | false | null | всего |",
             "| --- | --- | --- | --- | --- |"]
    for c in CLASSES:
        n = sum(conf[(c, p)] for p in ("true", "false", "null"))
        lines.append(f"| {c} | {conf[(c, 'true')]} | {conf[(c, 'false')]} | "
                     f"{conf[(c, 'null')]} | {n} |")
    lines += ["", "Итого: " + summary(conf), "", "## По заклинаниям", ""]
    lines += [f"* {sp}: " + summary(cn) for sp, cn in sorted(by_spell.items())]
    lines += ["", "## По роликам", ""]
    lines += [f"* {v}: " + summary(cn) for v, cn in by_video.items()]
    lines += ["", f"Ответов true/false не по разметке: {len(bad)}"]
    text = "\n".join(lines) + "\n"
    (out / f"score_{name}.md").write_text(text, encoding="utf-8")
    print(text)
    if bad:
        contact(bad[:400], out / f"errors_{name}.png")
    # Выборка ответов для проверки глазами: 120 «готово», 60 «перезарядка»,
    # 40 «нельзя» (seed 0) — подпись: время, класс разметки > ответ.
    rng = np.random.default_rng(0)
    pick = []
    for want, n in (("true", 120), ("false", 60), ("null", 40)):
        pool = [x for x in answered if x[4] == want]
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False) if pool else []
        pick += [pool[i] for i in sorted(idx)]
    contact(pick, out / f"sample_{name}.png")


def contact(items, path: Path, z: int = 44, per: int = 20) -> None:
    """Лист-контакт вырезов слотов: (ролик, t, слот, разметка, ответ)."""
    from PIL import ImageDraw
    sheet = Image.new("RGB", (per * z, ((len(items) + per - 1) // per) * (z + 24)))
    d = ImageDraw.Draw(sheet)
    panels = {}
    for i, (vid, t, k, truth, pred) in enumerate(items):
        panels.setdefault(vid, video_panel(vid))
        fr = load(SPELLS_DATA / vid / "crops" / f"t{int(t):05d}.jpg")
        x, y = (i % per) * z, (i // per) * (z + 24)
        sheet.paste(Image.fromarray(hs.slot_crop(fr, panels[vid], k, z - 2)), (x, y + 24))
        d.text((x + 1, y), f"{vid[:4]}{k}", fill=(255, 255, 0))
        d.text((x + 1, y + 11), f"{int(t)} {truth[:2]}>{pred[0]}", fill=(255, 128, 0))
    sheet.save(path)


def cmd_questions(args) -> None:
    """Лист «спорных» интервалов разметки (не готово и не перезарядка):
    панель целиком в начале, середине и конце интервала — решить глазами."""
    from PIL import ImageDraw
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for vid in args.videos:
        p = video_panel(vid)
        fs = {float(f.stem[1:]): f for f in crops(vid)}
        lab = read_labels(vid, args.labels)          # с правками: поздние строки сильнее
        rows = []
        for k in hs.SPELL_SLOTS:
            ts = sorted(t for (t, s) in lab if s == k)
            for a, b, st in intervals(ts, [lab[(t, k)] for t in ts]):
                if st not in ("ready", "cooldown", "hidden"):
                    rows.append({"slot": k, "start_sec": a, "end_sec": b, "state": st})
        s = p.scale
        box = (round(p.x - 5 * s), round(p.y), round(p.x + 330 * s), round(p.y + 100 * s))
        w, h = 330, 100
        tiles = []
        for r in rows:
            a, b = int(r["start_sec"]), int(r["end_sec"]) - 1
            for t in sorted({a, (a + b) // 2, b}):
                if float(t) in fs:
                    im = Image.open(fs[float(t)]).convert("RGB").crop(box).resize((w, h))
                    d = ImageDraw.Draw(im)
                    d.rectangle([0, 0, w, 11], fill=(0, 0, 0))
                    d.text((2, 0), f"{r['slot']} {a}-{b + 1} {r['state']} t={t}",
                           fill=(255, 255, 0))
                    tiles.append(im)
        per = 4
        for k in range(0, len(tiles), per * 8):
            part = tiles[k:k + per * 8]
            sheet = Image.new("RGB", (per * w, ((len(part) + per - 1) // per) * h))
            for i, im in enumerate(part):
                sheet.paste(im, ((i % per) * w, (i // per) * h))
            name = out / f"{vid}_q{k // (per * 8)}.png"
            sheet.save(name)
            print(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract", help="вырезы низа кадра 1 кадр/с")
    a.add_argument("video")
    a.add_argument("--decode", choices=("auto", "gpu", "cpu"), default="cpu")
    a.set_defaults(fn=cmd_extract)
    a = sub.add_parser("locate", help="панель умений по медиане вырезов")
    a.add_argument("videos", nargs="+")
    a.set_defaults(fn=cmd_locate)
    a = sub.add_parser("strips", help="ленты слотов для разметки глазами")
    a.add_argument("videos", nargs="+")
    a.add_argument("--out", default=str(RUNS / "hud_spells" / "strips"))
    a.add_argument("--size", type=int, default=28)
    a.add_argument("--labels", default="", help="цвет по разметке (draft.csv / labels.csv)")
    a.set_defaults(fn=cmd_strips)
    a = sub.add_parser("score", help="читатель против разметки")
    a.add_argument("videos", nargs="*")
    a.add_argument("--split", default="tune", choices=tuple(SPLIT))
    a.add_argument("--margin", type=int, default=1)
    a.add_argument("--out", default=str(RUNS / "hud_spells" / "score"))
    a.set_defaults(fn=cmd_score)
    a = sub.add_parser("questions", help="лист спорных интервалов разметки")
    a.add_argument("videos", nargs="+")
    a.add_argument("--labels", default="draft.csv")
    a.add_argument("--out", default=str(RUNS / "hud_spells" / "questions"))
    a.set_defaults(fn=cmd_questions)
    a = sub.add_parser("draft", help="черновик разметки грубым правилом")
    a.add_argument("videos", nargs="+")
    a.set_defaults(fn=cmd_draft)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
