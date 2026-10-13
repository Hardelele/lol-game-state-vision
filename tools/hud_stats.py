"""Полоса статистики HUD вида игрока — CS и время матча: данные и проверка (#816).

Подкоманды (подробно — docs/last-hits.md):

* `strips <id>…` — вырез полосы (y 0…36, x 1500…1920 кадра 1920×1080) из
  скачанного ролика `data/videos/<id>.mp4` подряд, `--fps` кадров/с, в
  `data/hud_stats/<id>.<fps>fps.raw` (BGR, 36×420×3 на кадр);
* `cluster` — цифры роликов подбора (каждый 3-й кадр 1 кадр/с), k-средних
  на 40 групп → `data/hud_stats/clusters.npz` и лист групп
  `runs/hud_stats/clusters.png`: подписать глазами;
* `samples` — подписи групп из `dataset/hud_stats/cluster_labels.txt`
  (по строке на группу: цифра, `:` или `x` — мусор) → образцы
  `dataset/hud_stats_digits.npz` и значок миньона `dataset/hud_stats_icon.png`;
* `score` — читатель по роликам: время матча против хода ролика (на
  ролике без склеек «время − секунда ролика» постоянно), CS — убывания и
  скачки; выборка прочтений CS для проверки глазами → `runs/hud_stats/score/`.

Ролики — те же 10 роликов вида игрока, что у #806 (docs/hud-spells.md), и то
же деление на подбор и проверку.

Пример:
    python tools/hud_stats.py strips --split tune --split test
    python tools/hud_stats.py cluster
    python tools/hud_stats.py samples
    python tools/hud_stats.py score --split test
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision import hud_stats as hs
from lolvision.paths import DATA, DATASET, RUNS, VIDEOS

STATS_DATA = DATA / "hud_stats"
LABELS = DATASET / "hud_stats" / "cluster_labels.txt"
OUT = RUNS / "hud_stats"
SPLIT = {"tune": ["JKZgBDX1WfY", "InEwM0UBAbc", "8lkxSQHq81k", "1Z9DJyLsXHo", "Pq2V04mhRTE"],
         "test": ["tEMffCUg-Js", "htjWVBQd0gw", "D37cbVNs0EU", "LIZxk-kM-6o", "ANXVB571YkE"]}
SH = (36, 420, 3)
K = 40


def strip_path(vid: str, fps: float) -> Path:
    return STATS_DATA / f"{vid}.{fps:g}fps.raw"


def load(vid: str, fps: float = 1.0) -> np.ndarray:
    p = strip_path(vid, fps)
    if not p.exists():
        raise SystemExit(f"нет {p}: сначала `strips {vid} --fps {fps:g}`")
    return np.fromfile(p, np.uint8).reshape(-1, *SH)


def videos(args) -> list[str]:
    out = list(args.videos or [])
    for s in args.split or []:
        out += SPLIT[s]
    return out


def cmd_strips(args) -> None:
    STATS_DATA.mkdir(parents=True, exist_ok=True)
    for vid in videos(args):
        src = VIDEOS / f"{vid}.mp4"
        # Полоса берётся из кадра 1920×1080: другой размер сначала приводится.
        vf = f"fps={args.fps:g},scale=1920:1080,crop=420:36:1500:0"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", vf,
                        "-f", "rawvideo", "-pix_fmt", "bgr24", str(strip_path(vid, args.fps))],
                       check=True)
        print(f"{vid}: {len(load(vid, args.fps))} кадров")


def _tiles(imgs: list[np.ndarray], n: int) -> np.ndarray:
    t = [cv2.copyMakeBorder((g * 255).astype(np.uint8), 1, 1, 1, 1,
                            cv2.BORDER_CONSTANT, value=80) for g in imgs[:n]]
    while len(t) < n:
        t.append(np.zeros((hs.GLYPH[1] + 2, hs.GLYPH[0] + 2), np.uint8))
    return cv2.resize(np.hstack(t), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def cmd_cluster(args) -> None:
    gl = []
    for vid in SPLIT["tune"]:
        for st in load(vid)[::3]:
            f = hs.fields(st)
            gl += [g.img for g in hs.glyphs(f["time"]) + hs.glyphs(f["cs"])]
    g = np.stack(gl).astype(np.float32)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, lab, cen = cv2.kmeans(g.reshape(len(g), -1), K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    np.savez_compressed(STATS_DATA / "clusters.npz", glyphs=g, labels=lab, centers=cen)
    rng = np.random.default_rng(0)
    rows = []
    for c in range(K):
        idx = np.nonzero(lab == c)[0]
        pick = rng.choice(idx, min(16, len(idx)), replace=False)
        row = cv2.cvtColor(_tiles([g[i] for i in pick], 16), cv2.COLOR_GRAY2BGR)
        head = np.zeros((row.shape[0], 110, 3), np.uint8)
        cv2.putText(head, f"{c}:{len(idx)}", (2, 35), 0, 0.7, (0, 255, 255), 2)
        rows.append(np.hstack([head, row]))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "clusters.png"), np.vstack(rows))
    print(f"{len(g)} цифр, {K} групп → {OUT / 'clusters.png'}")


def cmd_samples(args) -> None:
    z = np.load(STATS_DATA / "clusters.npz")
    g, lab, cen = z["glyphs"], z["labels"], z["centers"]
    names = [ln.split("#")[0].strip() for ln in LABELS.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n]
    if len(names) != K:
        raise SystemExit(f"{LABELS}: {len(names)} подписей, групп {K}")
    rng = np.random.default_rng(0)
    xs, ys = [], []
    x = g.reshape(len(g), -1)
    for c, name in enumerate(names):
        idx = np.nonzero(lab == c)[0]
        # Дальние от центра 10% группы — то, что k-средних прибило к чужим.
        d = ((x[idx] - cen[c]) ** 2).mean(1)
        keep = idx if name == "x" else idx[d <= np.percentile(d, 90)]
        keep = rng.choice(keep, min(80, len(keep)), replace=False)
        xs.append(g[keep])
        ys += [name] * len(keep)
    np.savez_compressed(hs.DIGITS, glyphs=np.concatenate(xs), labels=np.array(ys))
    icons = [np.median(np.stack([hs.fields(st)["icon"] for st in load(v)[::7]]), 0)
             for v in SPLIT["tune"]]
    icon = cv2.cvtColor(np.median(np.stack(icons), 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(hs.ICON), icon)
    u, n = np.unique(ys, return_counts=True)
    print(f"{hs.DIGITS.name}: " + ", ".join(f"{a} {b}" for a, b in zip(u, n)))


def check(stats: list[hs.Stats]) -> dict:
    """Самопроверка без разметки. Время: «время − номер кадра» в окне ±15
    кадров (1 кадр/с) — медиана; расходится больше чем на 1 с — ошибка
    (склейка ролика тоже даёт расхождение, поэтому лист для глаз). CS:
    убывание к прошлому прочтению не дальше 3 кадров — ошибка или новая
    игра; рост больше 8 за кадр — скачок."""
    n = len(stats)
    t = np.array([np.nan if s.time is None else s.time for s in stats], float)
    off = t - np.arange(n)
    odd = [i for i in range(n) if not np.isnan(off[i])
           and abs(off[i] - np.nanmedian(off[max(0, i - 15):i + 16])) > 1]
    dec, jump, prev = [], [], None
    for i, s in enumerate(stats):
        if s.cs is None:
            continue
        if prev is not None and i - prev[0] <= 3:
            if s.cs < prev[1]:
                dec.append(i)
            elif s.cs - prev[1] > 8:
                jump.append(i)
        prev = (i, s.cs)
    present = [s.present for s in stats]
    return {"frames": n, "present": sum(present),
            "time": int(np.sum(~np.isnan(t))), "time_odd": odd,
            "cs": sum(s.cs is not None for s in stats), "cs_dec": dec, "cs_jump": jump}


def cmd_score(args) -> None:
    rd = hs.StatsReader()
    out = OUT / "score"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    lines = ["| ролик | кадров | полоса | время прочтено | время расходится | CS прочтён | CS убывает | CS +9 и больше |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    tot = np.zeros(7, int)
    sample, odd_tiles = [], []
    for vid in videos(args) or SPLIT["test"]:
        a = load(vid)
        st = [rd.read_strip(x) for x in a]
        c = check(st)
        row = [c["frames"], c["present"], c["time"], len(c["time_odd"]), c["cs"],
               len(c["cs_dec"]), len(c["cs_jump"])]
        tot += row
        lines.append(f"| {vid} | " + " | ".join(map(str, row)) + " |")
        read = [i for i, s in enumerate(st) if s.cs is not None]
        for i in rng.choice(read, min(args.sample, len(read)), replace=False):
            sample.append((vid, int(i), a[i], f"{st[i].cs} {_mmss(st[i].time)}"))
        for i in c["time_odd"] + c["cs_dec"] + c["cs_jump"]:
            odd_tiles.append((vid, i, a[i], f"{st[i].cs} {_mmss(st[i].time)}"))
    lines.append("| всего | " + " | ".join(map(str, tot)) + " |")
    md = out / f"score_{'_'.join(args.split or ['videos'])}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for name, tiles in (("sample", sample), ("odd", odd_tiles)):
        if tiles:
            cv2.imwrite(str(out / f"{name}_{'_'.join(args.split or ['videos'])}.png"), _sheet(tiles))
    print(f"→ {out}")


def _mmss(t: int | None) -> str:
    return "—" if t is None else f"{t // 60:02d}:{t % 60:02d}"


def _sheet(tiles: list[tuple]) -> np.ndarray:
    """Лист: вырез CS и времени ×3, справа ответ читателя."""
    rows = []
    for vid, i, st, txt in tiles:
        part = cv2.resize(st[:, 255:], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        lab = np.zeros((part.shape[0], 260, 3), np.uint8)
        cv2.putText(lab, txt, (4, 45), 0, 1.0, (0, 255, 255), 2)
        cv2.putText(lab, f"{vid} {i}", (4, 95), 0, 0.5, (200, 200, 200), 1)
        rows.append(np.hstack([part, lab]))
    cols = [np.vstack(rows[k:k + 30]) for k in range(0, len(rows), 30)]
    hmax = max(c.shape[0] for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 8, cv2.BORDER_CONSTANT) for c in cols]
    return np.hstack(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("strips", "score"):
        a = sub.add_parser(name)
        a.add_argument("videos", nargs="*")
        a.add_argument("--split", action="append", choices=tuple(SPLIT))
        if name == "strips":
            a.add_argument("--fps", type=float, default=1.0)
        else:
            a.add_argument("--sample", type=int, default=12,
                           help="прочтений CS на ролик в лист для глаз")
    sub.add_parser("cluster")
    sub.add_parser("samples")
    args = ap.parse_args()
    {"strips": cmd_strips, "cluster": cmd_cluster, "samples": cmd_samples,
     "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
