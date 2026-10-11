"""Облик под плашкой → чемпион: признаки DINOv2, общая голова, прототипы матча.

Зачем. Задача #760, п. 3. Метки — имя над плашкой, прочитанное банком
образцов (tools/train_names.py) среди пяти чемпионов команды по составу
из HUD («HUD как учитель»). Проверка — на отложенных роликах против тех же
меток; кадры, где имя не читается, отдельно: для них верной метки нет,
в отчёт идёт число и распределение ответов.

Три способа (все — выбор из пяти чемпионов команды плашки):
* head — линейная голова на признаках DINOv2, обучена на обучающих роликах;
* proto — прототипы того же ролика: вырезы с прочитанным именем, взятые
  не ближе 30 с к проверяемому кадру (чтобы не сравнивать почти одинаковые
  кадры); вариант proto_past — только из прошлого (как в live);
* mix — прототипы, если у кандидата они есть, иначе голова.

Пример:
    python tools/train_appearance.py --heldout 58w57eJ5Qks olmTXkkUv58 5nBGNokOUDA 4AIX8QtRid4
    # → runs/identity/appearance_head.npz, appearance_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
from lolvision import appearance as ap_, name_reader as nr
from lolvision.paths import DATA, RUNS
from lolvision.scene_kind import load_dino
from train_names import HARVEST, candidates, load_video

FEATS = DATA / "identity" / "feats"
OUT = RUNS / "identity"
NAME_SIM = 0.6          # имя считается прочитанным: сходство с образцом
NAME_MARGIN = 0.05      # и отрыв от второго кандидата
GAP_SEC = 30.0          # прототипы не ближе этого к проверяемому кадру


def body_features(vid: str, bodies: np.ndarray, vit) -> np.ndarray:
    """Признаки вырезов облика, с кэшем на диске (data/identity/feats/<id>.npy)."""
    FEATS.mkdir(parents=True, exist_ok=True)
    p = FEATS / f"{vid}.npy"
    if p.exists():
        f = np.load(p)
        if len(f) == len(bodies):
            return f.astype(np.float32)
    f = ap_.features(vit, bodies)
    np.save(p, f.astype(np.float16))
    return f


def name_labels(v: dict, reader: nr.NameReader) -> tuple[list[str], np.ndarray]:
    """Метка по имени для каждой плашки: Id чемпиона, если имя прочитано
    уверенно, иначе ""; и маска уверенных."""
    sims = reader.label_sims(v["names"])
    champs = set(v["roster"].champions)
    labs = []
    for s, b in zip(sims, v["bars"]):
        lab, sim, m = reader.pick(s, candidates(v, int(b["team"])))
        labs.append(lab if lab in champs and sim >= NAME_SIM and m >= NAME_MARGIN else "")
    labs_arr = np.array(labs)
    return labs, labs_arr != ""


def train_head(x: np.ndarray, y: list[str], epochs: int = 300, wd: float = 1e-3) -> ap_.LinearHead:
    ids = sorted(set(y))
    yi = torch.tensor([ids.index(c) for c in y], device="cuda")
    xt = torch.tensor(x, device="cuda")
    lin = torch.nn.Linear(x.shape[1], len(ids)).cuda()
    opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=wd)
    # Классы сильно неравны (ведомый чемпион канала на половине вырезов) —
    # веса обратно пропорциональны частоте.
    cnt = torch.bincount(yi, minlength=len(ids)).float()
    w = (cnt.sum() / (cnt * len(ids))).clamp(max=20)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(lin(xt * 10), yi, weight=w)
        loss.backward()
        opt.step()
    return ap_.LinearHead(ids, lin.weight.detach().cpu().numpy() * 10,
                          lin.bias.detach().cpu().numpy())


def proto_scores(f: np.ndarray, t: np.ndarray, labs: list[str], sure: np.ndarray,
                 i: int, cands: list[str], past: bool, k: int = 5) -> np.ndarray:
    """Сходство выреза i с прототипами кандидатов из того же ролика."""
    far = np.abs(t - t[i]) >= GAP_SEC
    if past:
        far &= t < t[i]
    m = sure & far
    out = np.full(len(cands), -1.0, np.float32)
    if not m.any():
        return out
    sims = f[m] @ f[i]
    lab = np.array(labs)[m]
    for j, c in enumerate(cands):
        s = sims[lab == c]
        if len(s):
            out[j] = np.sort(s)[::-1][:k].mean()
    return out


def evaluate(v: dict, f: np.ndarray, labs: list[str], sure: np.ndarray,
             head: ap_.LinearHead) -> dict:
    t = v["t"][v["bars"]["frame"]]
    res = {m: [] for m in ("head", "proto", "proto_past", "mix")}
    known = 0
    for i in np.flatnonzero(sure):
        cands = candidates(v, int(v["bars"]["team"][i]))
        truth = labs[i]
        known += truth in head.ids
        hp = head.pick(f[i:i + 1], cands)[1][0]
        ps = proto_scores(f, t, labs, sure, i, cands, past=False)
        pp = proto_scores(f, t, labs, sure, i, cands, past=True)
        res["head"].append(cands[int(hp.argmax())] == truth)
        res["proto"].append(ps.max() > -1 and cands[int(ps.argmax())] == truth)
        res["proto_past"].append(pp.max() > -1 and cands[int(pp.argmax())] == truth)
        # mix: прототипы там, где есть, остальным кандидатам — вероятность головы
        mix = np.where(ps > -1, ps, -1.0 + hp)
        res["mix"].append(cands[int(mix.argmax())] == truth)
    n = int(sure.sum())
    return {"labelled": n, "unlabelled": int((~sure).sum()),
            "truth_known_to_head": round(known / max(n, 1), 4),
            **{m: round(float(np.mean(r)), 4) if r else None for m, r in res.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--names", type=Path, default=OUT / "names.npz")
    ap.add_argument("--harvest", type=Path, default=HARVEST)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    reader = nr.NameReader.load(args.names)
    vit = load_dino()
    vids = sorted(p.stem for p in args.harvest.glob("*.npz"))
    data = {}
    for vid in vids:
        v = load_video(vid, args.harvest)
        z = np.load(args.harvest / f"{vid}.npz")
        v["t"] = z["t"]
        t0 = time.perf_counter()
        f = body_features(vid, z["bodies"], vit)
        labs, sure = name_labels(v, reader)
        data[vid] = (v, f, labs, sure)
        print(f"{vid}: плашек {len(labs)}, с именем {int(sure.sum())}, "
              f"{time.perf_counter() - t0:.0f} с", flush=True)

    train = [vid for vid in vids if vid not in args.heldout]
    x = np.concatenate([data[v][1][data[v][3]] for v in train])
    y = [lab for v in train for lab, s in zip(data[v][2], data[v][3]) if s]
    head = train_head(x, y)
    args.out.mkdir(parents=True, exist_ok=True)
    head.save(args.out / "appearance_head.npz")

    # Скорость признаков: пачка из 4 вырезов (типичное число плашек на кадре).
    bodies = np.load(args.harvest / f"{vids[0]}.npz")["bodies"][:4]
    for _ in range(3):
        ap_.features(vit, bodies)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        ap_.features(vit, bodies)
    torch.cuda.synchronize()
    ms4 = (time.perf_counter() - t0) / 20 * 1000

    report = {"train": train, "train_crops": len(y), "head_classes": len(head.ids),
              "ms_dino_4_crops": round(ms4, 2), "heldout": {}}
    for vid in args.heldout:
        if vid not in data:
            continue
        v, f, labs, sure = data[vid]
        r = evaluate(v, f, labs, sure, head)
        r["roster"] = v["roster"].to_json()
        r["roster_unseen_by_head"] = [c for c in v["roster"].champions if c not in head.ids]
        report["heldout"][vid] = r
        print(vid, json.dumps(r, ensure_ascii=False), flush=True)
    tot = {}
    for m in ("head", "proto", "proto_past", "mix"):
        num = sum(r[m] * r["labelled"] for r in report["heldout"].values() if r[m] is not None)
        den = sum(r["labelled"] for r in report["heldout"].values() if r[m] is not None)
        tot[m] = round(num / max(den, 1), 4)
    report["heldout_total"] = tot
    (args.out / "appearance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
