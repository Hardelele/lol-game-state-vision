"""Проверить состав матча по портретам HUD на собранных роликах.

Зачем. Задача #760, п. 1. Состав по ролику — решение по всем кадрам с HUD
(lolvision/roster.py). Здесь меряем:
* сходится ли он с названием ролика («Ekko Top vs Zac» — два чемпиона и
  их роли: ведомый в синей или красной команде, соперник — в другой, на
  той же роли);
* как часто состав по одному кадру совпадает с составом по ролику и
  через сколько секунд от начала решение по накопленным кадрам перестаёт
  меняться (состав «на кадр» в live);
* лист для проверки глазами: портреты HUD с середины ролика и иконки
  Data Dragon, выбранные для них.

Пример:
    python tools/eval_roster.py            # все data/identity/harvest/*.npz
    # → runs/identity/roster_report.json, roster_sheet.png
"""

from __future__ import annotations

import argparse
import json
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from identity_harvest import HARVEST, find_video
from lolstream.video_stream import frames as video_frames
from lolvision import ddragon, name_reader as nr, roster
from lolvision.paths import RUNS
from train_names import rendered_reader

OUT = RUNS / "identity"
ROLE_WORDS = {"top": 0, "jungle": 1, "mid": 2, "adc": 3, "support": 4}


def title_pair(title: str, dd) -> tuple[str, str, int] | None:
    """«Ekko Top vs Zac - …» → (Ekko, Zac, роль 0); «(Comp) Thresh Support vs Janna»."""
    m = re.search(r"([A-Za-z'. ]+?)\s+(Top|Jungle|Mid|ADC|Support)\s+vs\s+([A-Za-z'. ]+?)"
                  r"(?:\s*\(|\s+-)", title)
    if not m:
        return None
    a, role, b = m.group(1).split(")")[-1].strip(), m.group(2).lower(), m.group(3).strip()
    ca, cb = dd.by_name(a), dd.by_name(b)
    if not ca or not cb:
        return None
    return ca, cb, ROLE_WORDS[role]


def mid_frame(vid: str, t: float) -> np.ndarray:
    for rgb in video_frames(find_video(vid), 1920, 1080, 1.0, skip=t):
        return rgb.copy()
    raise RuntimeError(f"{vid}: кадр {t} с не прочитан")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="16.19.1")
    ap.add_argument("--no-sheet", action="store_true")
    args = ap.parse_args()
    dd = ddragon.DDragon(args.version)
    report, rows = {}, []
    for p in sorted(HARVEST.glob("*.npz")):
        vid = p.stem
        z = np.load(p)
        ids = [str(s) for s in z["ids"]]
        fr = [roster.FrameSlots(s.astype(np.float32)) for s in z["slots"]]
        r = roster.decide(fr, ids)
        # Слоты с иконкой не из Data Dragon (Kayn в форме) — по именам над плашками.
        r = nr.refine_roster(r, z["names"], z["bars"]["team"], rendered_reader(args.version))
        hud = np.array([f.hud for f in fr])
        # Состав по одному кадру против состава по ролику.
        single = [roster.decide([f], ids) for f, h in zip(fr, hud) if h]
        slot_agree = np.mean([np.mean([a == b for a, b in zip(s.champions, r.champions)])
                              for s in single])
        frame_agree = np.mean([s.champions == r.champions for s in single])
        # Через сколько секунд накопленное решение стало окончательным.
        t = z["t"]
        stable_at = None
        acc = []
        for i, f in enumerate(fr):
            if f.hud:
                acc.append(f)
            if acc and (i % 5 == 0 or i == len(fr) - 1):
                d = roster.decide(acc, ids)
                if d.champions == r.champions:
                    stable_at = stable_at if stable_at is not None else float(t[i])
                else:
                    stable_at = None
        pair = title_pair(str(z["title"]), dd)
        title_ok = None
        if pair:
            a, b, role = pair
            title_ok = ((r.champions[role] == a and r.champions[5 + role] == b)
                        or (r.champions[role] == b and r.champions[5 + role] == a))
        report[vid] = {"title": str(z["title"]), "roster": r.to_json(),
                       "title_pair": pair, "title_ok": title_ok,
                       "frames": len(fr), "hud_frames": int(hud.sum()),
                       "single_frame_slot_agree": round(float(slot_agree), 4),
                       "single_frame_all10_agree": round(float(frame_agree), 4),
                       "stable_from_sec": stable_at,
                       "min_margin": round(min(r.margin), 3)}
        print(vid, title_ok, r.champions, f"слот/кадр {slot_agree:.3f}", flush=True)
        if not args.no_sheet:
            try:                 # ролик могли удалить после сбора — тогда без листа
                rows.append((vid, mid_frame(vid, float(t[len(t) // 2])), r))
            except FileNotFoundError:
                print(f"{vid}: ролика нет, в лист не попадает", flush=True)
    ok = [v["title_ok"] for v in report.values() if v["title_ok"] is not None]
    summary = {"videos": len(report), "title_checked": len(ok), "title_ok": int(sum(ok)),
               "single_frame_slot_agree_mean": round(float(np.mean(
                   [v["single_frame_slot_agree"] for v in report.values()])), 4),
               "single_frame_all10_agree_mean": round(float(np.mean(
                   [v["single_frame_all10_agree"] for v in report.values()])), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roster_report.json").write_text(
        json.dumps({"summary": summary, "videos": report}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if rows:
        cell = 40
        sheet = Image.new("RGB", (cell * 20 + 110, (cell + 4) * len(rows)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for k, (vid, rgb, r) in enumerate(rows):
            y = k * (cell + 4)
            d.text((2, y + 12), vid, fill=(255, 255, 0))
            for slot, vs in enumerate(roster.views()):
                v = vs[1]                                   # портрет боковой панели
                crop = rgb[v.y:v.y + v.size, v.x:v.x + v.size]
                icon = cv2.cvtColor(cv2.imread(str(dd.icon(r.champions[slot]))),
                                    cv2.COLOR_BGR2RGB)
                x = 110 + slot * 2 * cell
                sheet.paste(Image.fromarray(crop).resize((cell - 2, cell - 2)), (x, y))
                sheet.paste(Image.fromarray(icon).resize((cell - 2, cell - 2)), (x + cell, y))
        sheet.save(OUT / "roster_sheet.png")


if __name__ == "__main__":
    main()
