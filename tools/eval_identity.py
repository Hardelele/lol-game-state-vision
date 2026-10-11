"""Итог «кто это» для найденных плашек: имя, облик, соседние кадры, состав.

Зачем. Задача #760, п. 4. Сводит три источника на отложенных роликах:
* имя над плашкой (банк образцов, tools/train_names.py) среди пяти
  чемпионов команды — если прочитано уверенно, это ответ;
* облик (признаки DINOv2 в своей проекции из train_appearance.py,
  ближайший прототип того же матча — вырезы с прочитанным именем);
* голос соседних кадров «с той же плашкой» (lolvision/identity.neighbours)
  и запрет двух одинаковых чемпионов в одной команде на кадре.

Как меряем. Верной метки для плашки без имени нет, поэтому проверка —
на плашках, где имя прочитано: имя этой плашки прячем (её собственное имя
не идёт ни в ответ, ни в прототипы ±30 с), отвечаем обликом и соседями и
сравниваем с прочитанным именем. Два режима прототипов: offline — весь
ролик, кроме ±30 с вокруг кадра; online — только прошлое (как в live).
Отдельно — сколько плашек без имени и что о них сказано.

Пример:
    python tools/eval_identity.py --heldout 58w57eJ5Qks olmTXkkUv58 5nBGNokOUDA 4AIX8QtRid4
    # → runs/identity/identity_report.json, identity_check_<id>.png
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401
from lolvision import appearance as ap_, identity, name_reader as nr
from lolvision.paths import RUNS
from lolvision.scene_kind import load_dino
from train_appearance import GAP_SEC, body_features, name_labels
from train_names import HARVEST, candidates, load_video

OUT = RUNS / "identity"
NB_WEIGHT = 0.5         # вес голоса соседнего кадра относительно своего облика


def app_scores(f: np.ndarray, t: np.ndarray, labs: list[str], sure: np.ndarray,
               i: int, cands: list[str], past: bool) -> np.ndarray:
    """Сходство с ближайшим прототипом каждого кандидата; нет прототипа — 0."""
    far = np.abs(t - t[i]) >= GAP_SEC
    if past:
        far &= t < t[i]
    m = sure & far
    out = np.zeros(len(cands), np.float32)
    if m.any():
        sims = f[m] @ f[i]
        lab = np.array(labs)[m]
        for j, c in enumerate(cands):
            s = sims[lab == c]
            if len(s):
                out[j] = s.max()
    return out


def to_probs(s: np.ndarray, temp: float = 0.03) -> np.ndarray:
    """Сходства → вероятности по кандидатам (нулевой кандидат — без прототипа)."""
    z = np.where(s > 0, s / temp, -1e9)
    if not np.isfinite(z).any() or (s <= 0).all():
        return np.full(len(s), 1.0 / len(s), np.float32)
    z = np.exp(z - z.max())
    return (z / z.sum()).astype(np.float32)


def name_probs(nsim: np.ndarray, temp: float = 0.03) -> np.ndarray:
    """Сходства с именами кандидатов (неуверенное чтение) → вероятности."""
    z = np.exp((nsim - nsim.max(1, keepdims=True)) / temp)
    return z / z.sum(1, keepdims=True)


def evaluate(v: dict, f: np.ndarray, labs: list[str], sure: np.ndarray, past: bool,
             nsim: np.ndarray) -> dict:
    bars = v["bars"]
    t = v["t"][bars["frame"]]
    team = bars["team"].astype(int)
    n = len(bars)
    cands = [candidates(v, int(team[i])) for i in range(n)]
    # Своё: имя (если прочитано) или облик; для проверки — облик всегда.
    app = np.stack([to_probs(app_scores(f, t, labs, sure, i, cands[i], past))
                    for i in range(n)])
    named = np.zeros_like(app)
    for i in np.flatnonzero(sure):
        named[i, cands[i].index(labs[i])] = 1.0
    # Без уверенного имени: облик × неуверенное имя (часть букв видна).
    weak = app * name_probs(nsim)
    weak /= weak.sum(1, keepdims=True) + 1e-9
    own = np.where(sure[:, None], named, weak)
    nb = identity.neighbours(bars["frame"], bars["x"], bars["y"], team, bars["level"])

    def decide(i: int, hide: bool) -> tuple[np.ndarray, np.ndarray]:
        mine = app[i] if hide else own[i]
        vote = mine.copy()
        for j in nb[i]:
            # кандидаты соседа те же (та же команда)
            vote += NB_WEIGHT * own[j]
        return mine, vote

    res = {"app": [], "app_nb": [], "app_nb_distinct": []}
    per_champ: dict[str, list[bool]] = {}
    by_frame: dict[int, list[int]] = {}
    for i in range(n):
        by_frame.setdefault(int(bars["frame"][i]), []).append(i)
    for i in np.flatnonzero(sure):
        mine, vote = decide(i, hide=True)
        truth = labs[i]
        res["app"].append(cands[i][int(mine.argmax())] == truth)
        res["app_nb"].append(cands[i][int(vote.argmax())] == truth)
        # Запрет повторов в команде на кадре: остальные плашки кадра — как есть.
        same = [j for j in by_frame[int(bars["frame"][i])] if team[j] == team[i]]
        votes = np.stack([decide(j, hide=(j == i))[1] for j in same])
        pick = identity.assign_distinct(votes)
        k = same.index(i)
        ok = cands[i][int(pick[k])] == truth
        res["app_nb_distinct"].append(ok)
        per_champ.setdefault(truth, []).append(ok)
    # Плашки без имени: итоговый ответ системы (имя соседей + облик).
    unnamed = Counter()
    for i in np.flatnonzero(~sure):
        _, vote = decide(i, hide=False)
        unnamed[cands[i][int(vote.argmax())]] += 1
    m = int(sure.sum())
    out = {"bars": n, "named": m, "unnamed": n - m,
           "with_neighbour": round(float(np.mean([len(x) > 0 for x in nb])), 4),
           **{k: round(float(np.mean(r)), 4) if r else None for k, r in res.items()},
           "balanced_app_nb_distinct": round(float(np.mean([np.mean(r) for r in
                                                            per_champ.values()])), 4),
           "per_champion": {c: [int(np.sum(r)), len(r)] for c, r in sorted(per_champ.items())},
           "unnamed_answers": unnamed.most_common()}
    return out


def check_sheet(path: Path, v: dict, labs: list[str], sure: np.ndarray, answers: list[str],
                idx: np.ndarray, bodies: np.ndarray) -> None:
    """Лист для проверки глазами: вырез имени, облик и ответ."""
    cw, ch = 160, 190
    img = Image.new("RGB", (cw * 6, ch * ((len(idx) + 5) // 6)), (25, 25, 25))
    d = ImageDraw.Draw(img)
    for k, i in enumerate(idx):
        x, y = (k % 6) * cw, (k // 6) * ch
        img.paste(Image.fromarray(v["names"][i]), (x + 6, y))
        img.paste(Image.fromarray(bodies[i]), (x + 16, y + 22))
        d.text((x + 4, y + 154), f"{i} {answers[i]}", fill=(255, 255, 0))
        d.text((x + 4, y + 168), f"name: {labs[i] or '-'}", fill=(180, 180, 180))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--names", type=Path, default=OUT / "names.npz")
    ap.add_argument("--harvest", type=Path, default=HARVEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", type=int, default=48, help="плашек без имени на лист проверки")
    ap.add_argument("--proj", type=Path, default=OUT / "appearance_proj.npy",
                    help="проекция облика из train_appearance.py (нет файла — сырые признаки)")
    args = ap.parse_args()

    reader = nr.NameReader.load(args.names)
    proj = ap_.Projection(np.load(args.proj)) if args.proj.exists() else None
    vit = load_dino()
    report: dict = {"heldout": {}, "projection": proj is not None}
    rng = np.random.default_rng(0)
    for vid in args.heldout:
        v = load_video(vid, args.harvest)
        z = np.load(args.harvest / f"{vid}.npz")
        v["t"] = z["t"]
        f = body_features(vid, z["bodies"], vit)
        if proj is not None:
            f = proj(f)
        labs, sure, nsim = name_labels(v, reader, soft=True)
        r = {"title": v["title"], "roster": v["roster"].to_json()}
        for mode, past in (("offline", False), ("online", True)):
            r[mode] = evaluate(v, f, labs, sure, past, nsim)
        report["heldout"][vid] = r
        print(vid, json.dumps({m: {k: r[m][k] for k in ("named", "unnamed", "app", "app_nb",
                                                        "app_nb_distinct")}
                               for m in ("offline", "online")}), flush=True)
        # Лист плашек без имени с ответом системы (offline).
        bars = v["bars"]
        t = v["t"][bars["frame"]]
        team = bars["team"].astype(int)
        cands = [candidates(v, int(team[i])) for i in range(len(bars))]
        answers = []
        nb = identity.neighbours(bars["frame"], bars["x"], bars["y"], team, bars["level"])
        own = []
        for i in range(len(bars)):
            if sure[i]:
                p = np.zeros(5, np.float32)
                p[cands[i].index(labs[i])] = 1
            else:
                p = to_probs(app_scores(f, t, labs, sure, i, cands[i], False))
                p = p * name_probs(nsim[i:i + 1])[0]
                p /= p.sum() + 1e-9
            own.append(p)
        for i in range(len(bars)):
            vote = own[i] + NB_WEIGHT * sum((own[j] for j in nb[i]), np.zeros(5, np.float32))
            answers.append(cands[i][int(vote.argmax())])
        pool = np.flatnonzero(~sure)
        idx = rng.permutation(pool)[:args.check]
        check_sheet(args.out / f"identity_check_{vid}.png", v, labs, sure, answers, idx,
                    z["bodies"])
        np.save(args.out / f"identity_check_{vid}.npy", idx)

    # Скорость частей на кадр (CPU — имя; GPU — облик пачкой из 4 вырезов).
    names = np.load(args.harvest / f"{args.heldout[0]}.npz")["names"][:256]
    t0 = time.perf_counter()
    reader.label_sims(names)
    report["ms_name_per_bar_cpu"] = round((time.perf_counter() - t0) / len(names) * 1000, 2)
    (args.out / "identity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = {}
    for mode in ("offline", "online"):
        for k in ("app", "app_nb", "app_nb_distinct"):
            num = sum(r[mode][k] * r[mode]["named"] for r in report["heldout"].values())
            den = sum(r[mode]["named"] for r in report["heldout"].values())
            tot[f"{mode}_{k}"] = round(num / max(den, 1), 4)
    report["total"] = tot
    (args.out / "identity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
