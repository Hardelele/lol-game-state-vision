"""Собрать банк образцов имён над плашкой и проверить чтение имён.

Зачем. Задача #760, п. 2: имя над плашкой → чемпион без OCR общего вида.
Образцы размечаются автоматически: плашка команды T несёт имя одного из
пяти чемпионов T по составу из HUD (lolvision/roster.py). Начальная
разметка — отрисованными шрифтом образцами среди этих пяти; дальше
медиана вырезов каждого чемпиона становится настоящим образцом, и
разметка повторяется уже по ним (медиана шумной группы — правильное имя,
если правильных в группе большинство). Итог — банк: настоящие образцы для
чемпионов обучающих роликов, отрисованные — для остальных, состояния.

Проверка на отложенных роликах: (1) чтение по всему словарю против чтения
среди пяти кандидатов команды — насколько имя само по себе определяет
чемпиона; (2) лист вырезов с прочитанным для проверки глазами.

Пример:
    python tools/train_names.py --heldout 58w57eJ5Qks olmTXkkUv58 5nBGNokOUDA 4AIX8QtRid4
    # → runs/identity/names.npz, names_report.json, names_medians.png, names_check_*.png
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from lolvision import ddragon, name_reader as nr, roster
from lolvision.paths import DATA, RUNS

HARVEST = DATA / "identity" / "harvest"
OUT = RUNS / "identity"
MIN_MARGIN = nr.ADAPT_MARGIN   # разметка в образец: отрыв лучшего кандидата
MIN_SIM = nr.ADAPT_SIM         # и сходство не ниже
MIN_COUNT = nr.ADAPT_COUNT     # образец из медианы не меньше чем пяти вырезов


_RENDERED: dict[str, nr.NameReader] = {}


def rendered_reader(version: str = "16.19.1") -> nr.NameReader:
    if version not in _RENDERED:
        _RENDERED[version] = nr.NameReader.rendered(ddragon.DDragon(version).names())
    return _RENDERED[version]


def load_video(vid: str, harvest: Path = HARVEST, refine: bool = True) -> dict:
    """Сбор ролика и состав по HUD; слоты с чужой иконкой — поправка по именам."""
    z = np.load(harvest / f"{vid}.npz")
    ids = [str(s) for s in z["ids"]]
    slots = z["slots"].astype(np.float32)
    r = roster.decide([roster.FrameSlots(s) for s in slots], ids)
    if refine:
        r = nr.refine_roster(r, z["names"], z["bars"]["team"], rendered_reader())
    return {"vid": vid, "roster": r, "bars": z["bars"], "names": z["names"],
            "title": str(z["title"])}


def candidates(v: dict, team: int) -> list[str]:
    return v["roster"].team(roster.TEAMS[team]) if team >= 0 else v["roster"].champions


def label_all(reader: nr.NameReader, videos: list[dict]) -> dict[str, list[tuple]]:
    """Чтение всех вырезов среди пяти кандидатов команды: (метка, сходство, отрыв)."""
    out = {}
    for v in videos:
        sims = reader.label_sims(v["names"])
        out[v["vid"]] = [reader.pick(s, candidates(v, int(b["team"])))
                         for s, b in zip(sims, v["bars"])]
    return out


def medians(videos: list[dict], labels: dict[str, list[tuple]]) -> tuple[list[str], list]:
    """Образцы: медиана белизны по (ролик, чемпион) и по каждому состоянию
    (чистка совпавших медиан — в nr.build_medians)."""
    groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for v in videos:
        for c, (lab, s, m) in zip(v["names"], labels[v["vid"]]):
            if s >= MIN_SIM and m >= MIN_MARGIN:
                key = ("*", lab) if lab.startswith(nr.STATUS) else (v["vid"], lab)
                groups[key].append(nr.center(nr.whiteness(c)))
    keys, temps = nr.build_medians(groups, MIN_COUNT)
    return [k[1] for k in keys], temps


def build(videos: list[dict], dd, font: Path, px: int, track: float, iters: int) -> nr.NameReader:
    rendered = nr.NameReader.rendered(dd.names(), font, px, track)
    reader = rendered
    for it in range(iters):
        labels = label_all(reader, videos)
        labs, temps = medians(videos, labels)
        have = set(labs)
        # Отрисованные — для чемпионов и состояний без настоящего образца.
        rl = [lab for lab in rendered.labels if lab not in have]
        rt = [rendered.bank[:, rendered.labels.index(lab)].reshape(nr.WIN_H, nr.WIN_W)
              for lab in rl]
        reader = nr.NameReader(labs + rl, np.stack(temps + rt),
                               ["real"] * len(labs) + ["rendered"] * len(rl))
        print(f"проход {it + 1}: настоящих образцов {len(labs)} "
              f"({len({x for x in labs if not x.startswith(nr.STATUS)})} чемпионов)", flush=True)
    return reader


def sheet(path: Path, tiles: list[tuple[np.ndarray, str]], cols: int = 3, k: int = 3) -> None:
    w, h = nr.CROP_W * k, nr.CROP_H * k + 14
    img = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), (30, 30, 30))
    d = ImageDraw.Draw(img)
    for i, (crop, text) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * h
        img.paste(Image.fromarray(crop).resize((nr.CROP_W * k, nr.CROP_H * k), Image.NEAREST),
                  (x, y))
        d.text((x + 4, y + nr.CROP_H * k), text, fill=(255, 255, 0))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--font", default=str(nr.FONT))
    ap.add_argument("--px", type=int, default=nr.FONT_PX)
    ap.add_argument("--track", type=float, default=nr.TRACK)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--check", type=int, default=90, help="вырезов на лист проверки")
    ap.add_argument("--version", default="16.19.1")
    ap.add_argument("--harvest", type=Path, default=HARVEST, help="папка сборов")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dd = ddragon.DDragon(args.version)
    vids = sorted(p.stem for p in args.harvest.glob("*.npz") if not p.stem.endswith(".tmp"))
    train = [load_video(v, args.harvest) for v in vids if v not in args.heldout]
    held = [load_video(v, args.harvest) for v in vids if v in args.heldout]
    print(f"обучение: {len(train)} роликов, отложено: {len(held)}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    reader = build(train, dd, Path(args.font), args.px, args.track, args.iters)
    reader.save(out / "names.npz")

    # Лист медиан: глазами видно, что образец каждого чемпиона — его имя.
    real = [(lab, i) for i, (lab, k) in enumerate(zip(reader.labels, reader.kinds)) if k == "real"]
    tiles = []
    for lab, i in real:
        t = reader.bank[:, i].reshape(nr.WIN_H, nr.WIN_W)
        t = (t - t.min()) / (np.ptp(t) + 1e-6)
        img = np.zeros((nr.CROP_H, nr.CROP_W), np.float32)
        img[nr.DY:nr.DY + nr.WIN_H, nr.DX:nr.DX + nr.WIN_W] = t
        tiles.append((np.repeat((img * 255).astype(np.uint8)[..., None], 3, 2), lab))
    sheet(out / "names_medians.png", tiles, cols=4, k=2)

    report = {"train": [v["vid"] for v in train], "heldout": {},
              "bank": {"real": len(real), "total": len(reader.labels)}}
    rng = np.random.default_rng(0)
    for v in held:
        t0 = time.perf_counter()
        sims = reader.label_sims(v["names"])
        cons = [reader.pick(s, candidates(v, int(b["team"]))) for s, b in zip(sims, v["bars"])]
        dt = (time.perf_counter() - t0) / max(len(cons), 1)
        free = [reader.pick(s) for s in sims]
        champs = set(v["roster"].champions)
        n = len(cons)
        sure = [i for i, (lab, s, m) in enumerate(cons) if s >= MIN_SIM and lab in champs]
        agree = sum(free[i][0] == cons[i][0] for i in sure)
        report["heldout"][v["vid"]] = {
            "title": v["title"], "roster": v["roster"].to_json(), "bars": n,
            "labels": Counter(lab for lab, s, m in cons if s >= MIN_SIM).most_common(),
            "below_min_sim": sum(s < MIN_SIM for _, s, _ in cons),
            "champion_reads": len(sure),
            "free_equals_constrained": round(agree / max(len(sure), 1), 4),
            "ms_per_crop": round(dt * 1000, 2)}
        pick = rng.permutation(n)[:args.check]
        sheet(out / f"names_check_{v['vid']}.png",
              [(v["names"][i], f"{i} {cons[i][0]} {cons[i][1]:.2f} | free {free[i][0]}")
               for i in pick])
        np.save(out / f"names_check_{v['vid']}.npy", pick)
    (out / "names_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
