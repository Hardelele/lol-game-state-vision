"""Готовность умений Q/W/E/R по панели своего HUD: прогон и проверка глазами.

Тот же читатель, что у заклинаний D/F (lolvision.hud_spells, #806): панель
та же, перезарядка у умений выглядит так же — ровная тёмно-синяя заливка и
белые цифры секунд. Свои у умений «не изучено» (серая тусклая иконка) и
смена иконки (усиление, повторное нажатие) — читатель отвечает «не похоже
ни на что» (unsure), а не «готово».

Подкоманды (вырезы и панели роликов — от tools/hud_spells.py extract / locate):

* `run [id…]` — читатель по всем вырезам ролика подряд, как в сеансе, →
  `runs/hud_abilities/reads.csv` и счёт состояний по слотам;
* `sheets [--per N]` — листы для проверки глазами: случайные N ответов
  «готово» и N «перезарядка» на ролик → `runs/hud_abilities/<ответ>_<id>.png`.

Пример:
    python tools/hud_abilities.py run
    python tools/hud_abilities.py sheets --per 40
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision import hud_spells as hs
from lolvision.paths import RUNS

import hud_spells as tool

OUT = RUNS / "hud_abilities"
VIDEOS = ("JKZgBDX1WfY", "InEwM0UBAbc", "8lkxSQHq81k", "1Z9DJyLsXHo", "Pq2V04mhRTE",
          "tEMffCUg-Js", "htjWVBQd0gw", "D37cbVNs0EU", "LIZxk-kM-6o", "ANXVB571YkE")


def run_video(vid: str) -> list[dict]:
    p = tool.video_panel(vid)
    refs = {k: hs.SlotRef() for k in hs.ABILITY_SLOTS}
    rows = []
    for f in tool.crops(vid):
        t = float(f.stem[1:])
        fr = tool.load(f)
        present = hs.panel_score(fr, p) >= hs.PRESENT_MIN
        dead = present and hs.hp_fill(fr, p) < hs.DEAD_HP
        for k in hs.ABILITY_SLOTS:
            r = hs.decide(k, hs.slot_crop(fr, p, k), refs[k], present, dead, t)
            rows.append({"video": vid, "t": t, "slot": k, "state": r.state,
                         "sim": "" if r.sim is None else round(r.sim, 3)})
    return rows


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in args.videos or VIDEOS:
        rs = run_video(vid)
        rows += rs
        c = Counter((r["slot"], r["state"]) for r in rs)
        line = " ".join(f"{k}: " + "/".join(str(c[(k, s)]) for s in ("ready", "cooldown", "unsure"))
                        for k in hs.ABILITY_SLOTS)
        print(f"{vid}: готово/перезарядка/не знаю — {line}")
    with open(OUT / "reads.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_sheets(args) -> None:
    rows = list(csv.DictReader(open(OUT / "reads.csv", encoding="utf-8")))
    rnd = random.Random(816)
    S = 48
    for state in ("ready", "cooldown"):
        for vid in VIDEOS:
            pick = [r for r in rows if r["video"] == vid and r["state"] == state]
            pick = sorted(rnd.sample(pick, min(args.per, len(pick))),
                          key=lambda r: (float(r["t"]), r["slot"]))
            if not pick:
                continue
            p = tool.video_panel(vid)
            cols = 10
            img = Image.new("RGB", (cols * (S + 6), ((len(pick) - 1) // cols + 1) * (S + 16)))
            d = ImageDraw.Draw(img)
            files = {float(f.stem[1:]): f for f in tool.crops(vid)}
            for i, r in enumerate(pick):
                fr = tool.load(files[float(r["t"])])
                c = hs.slot_crop(fr, p, r["slot"], size=S)
                x, y = (i % cols) * (S + 6), (i // cols) * (S + 16)
                img.paste(Image.fromarray(c), (x, y))
                d.text((x, y + S + 1), f"{r['slot']}{int(float(r['t']))}", fill=(255, 255, 0))
            img.save(OUT / f"{state}_{vid}.png")
    print("листы:", OUT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="читатель по роликам → reads.csv")
    a.add_argument("videos", nargs="*")
    a.set_defaults(func=cmd_run)
    a = sub.add_parser("sheets", help="листы «готово» / «перезарядка» для глаз")
    a.add_argument("--per", type=int, default=40)
    a.set_defaults(func=cmd_sheets)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
