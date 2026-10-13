"""Добивания по роликам вида игрока: прогон сеанса и сопоставление (#816).

Подкоманды (подробно — docs/last-hits.md):

* `session <id> --start S --dur D --fps F` — сеанс live (lolstream.live) по
  скачанному ролику `data/videos/<id>.mp4`: строки кадров (юниты, треки,
  смерти трекера) → `data/last_hits/<id>.jsonl`. Вид игрока сеанс решает
  по паспорту за ~1 мин потока — начинать стоит на минуту раньше нужного;
* `match <id> --fps F` — прочтения CS из полосы HUD (`tools/hud_stats.py
  strips <id> --fps F`) и строки сеанса → события lolvision.last_hits:
  счётчики, `runs/last_hits/<id>.json` и листы `runs/last_hits/<id>_<тип>_NN.png` —
  по каждой ближней смерти три выреза вокруг крипа (до, в момент, после) и
  CS этих кадров, для проверки глазами.

Пример:
    python tools/hud_stats.py strips D37cbVNs0EU --fps 4
    python tools/last_hits.py session D37cbVNs0EU --start 30 --dur 600 --fps 4
    python tools/last_hits.py match D37cbVNs0EU --fps 4
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision import hud_stats as hs
from lolvision import last_hits as lh
from lolvision.paths import DATA, RUNS, VIDEOS
from lolstream.video_stream import frames, probe

import hud_stats as tool_stats

LH_DATA = DATA / "last_hits"
OUT = RUNS / "last_hits"
ROW_KEYS = ("i", "t", "kind", "px", "py", "units", "tracks", "uev", "game_time", "hud", "last_hits", "view")


def cmd_session(args) -> None:
    from lolstream import live
    from lolstream.frame_source import VideoSource

    LH_DATA.mkdir(parents=True, exist_ok=True)
    src = VideoSource(str(VIDEOS / f"{args.id}.mp4"), args.fps, args.start, from_url=False)
    s = live.Session(args.id, live.DEFAULT_RUN, src)
    n, t0, end = 0, time.time(), args.start + args.dur
    with open(LH_DATA / f"{args.id}.jsonl", "w", encoding="utf-8") as f:
        while True:
            st = s.status(n)
            for r in st["frames"]:
                f.write(json.dumps({k: r.get(k) for k in ROW_KEYS}, ensure_ascii=False) + "\n")
            n = st["count"]
            last = st["frames"][-1]["t"] if st["frames"] else None
            if not st["active"] or (last is not None and last >= end):
                break
            time.sleep(0.5)
    s.stop()
    s.join(10)
    print(f"{args.id}: {n} кадров за {time.time() - t0:.0f} с, вид {st['view']}")


def cmd_match(args) -> None:
    rows = [json.loads(ln) for ln in open(LH_DATA / f"{args.id}.jsonl", encoding="utf-8")]
    strips = tool_stats.load(args.id, args.fps)
    rd = hs.StatsReader()
    m = lh.LastHits()
    me_place = lh.SelfPlace()
    events, seen_xy, cs_at = [], {}, {}
    n_me = n_game = 0
    for r in rows:
        k = round(r["t"] * args.fps)
        cs = rd.read_strip(strips[k]).cs if k < len(strips) else None
        cs_at[r["t"]] = cs
        tracks = r.get("tracks")
        if tracks is None:              # не игра или вид не решён: трекер не шёл
            continue
        n_game += 1
        me = me_place.update(r["t"], r.get("units"), tracks)
        n_me += me is not None
        deaths, born = lh.frame_inputs(r.get("uev"), tracks)
        for e in m.update(r["t"], cs, deaths, me, born):
            events.append(e)
        for tr in tracks:
            if tr.get("x") is not None:
                seen_xy[tr["id"]] = (r["t"], tr["x"], tr["y"])
    events += m.flush()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"id": args.id, "fps": args.fps, "frames": len(rows), "tracked": n_game,
           "me_found": n_me, "counts": m.counts, "cs_last": m.cs.cs,
           "events": [dict(e.to_dict(), xy=seen_xy.get(e.id)) for e in events]}
    (OUT / f"{args.id}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"{args.id}: кадров {len(rows)}, с треками {n_game}, свой найден {n_me}; "
          f"CS в конце {m.cs.cs}; {m.counts}")
    near = [e for e in events if e.type != "cs_unseen" and seen_xy.get(e.id)
            and (args.only is None or e.type == args.only)]
    if near:
        sheet(args.id, args.fps, near, seen_xy, cs_at, args.only or "all")


def sheet(vid: str, fps: float, evs: list, seen_xy: dict, cs_at: dict, tag: str) -> None:
    """Лист: по событию три выреза 320×200 вокруг последнего места крипа —
    за 0,5 с до смерти, в момент, через 0,5 с; подпись — тип и CS кадров."""
    want: dict[int, list] = {}
    for j, e in enumerate(evs):
        for d, dt in enumerate((-0.5, 0.0, 0.5)):
            want.setdefault(round((seen_xy[e.id][0] + dt) * fps), []).append((j, d))
    w, h, _ = probe(VIDEOS / f"{vid}.mp4")
    cells: dict[tuple[int, int], np.ndarray] = {}
    last = max(want)
    for k, fr in enumerate(frames(VIDEOS / f"{vid}.mp4", w, h, fps)):
        for j, d in want.get(k, ()):
            _, x, y = seen_xy[evs[j].id]
            cx, cy = int(x * w), int(y * h)
            x0, y0 = max(0, cx - 160), max(0, cy - 130)
            crop = np.ascontiguousarray(fr[y0:y0 + 200, x0:x0 + 320][..., ::-1])
            crop = cv2.copyMakeBorder(crop, 0, 200 - crop.shape[0], 0, 320 - crop.shape[1],
                                      cv2.BORDER_CONSTANT)
            cv2.circle(crop, (cx - x0, cy - y0), 14, (0, 255, 255), 1)
            cs = cs_at.get(round(k / fps, 2))
            cv2.putText(crop, f"CS {cs}", (4, 192), 0, 0.5, (0, 255, 255), 1)
            cells[(j, d)] = crop
        if k >= last:
            break
    rows = []
    for j, e in enumerate(evs):
        lab = np.zeros((200, 200, 3), np.uint8)
        col = (0, 255, 0) if e.type == "last_hit" else (0, 0, 255)
        cv2.putText(lab, e.type.replace("_last_hit", ""), (4, 40), 0, 0.7, col, 2)
        cv2.putText(lab, f"t {e.t:.2f}", (4, 80), 0, 0.6, (220, 220, 220), 1)
        far = "self ?" if e.dist is None else f"{e.dist:.0f} ed"
        cv2.putText(lab, far, (4, 115), 0, 0.6, (220, 220, 220), 1)
        cv2.putText(lab, f"#{j}", (4, 150), 0, 0.6, (220, 220, 220), 1)
        blank = np.zeros((200, 320, 3), np.uint8)
        rows.append(np.hstack([lab] + [cells.get((j, d), blank) for d in range(3)]))
    for p in range(0, len(rows), 12):
        cv2.imwrite(str(OUT / f"{vid}_{tag}_{p // 12:02d}.png"), np.vstack(rows[p:p + 12]))
    print(f"листы: {OUT}/{vid}_{tag}_NN.png ({len(rows)} событий)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("session")
    a.add_argument("id")
    a.add_argument("--start", type=float, default=0.0)
    a.add_argument("--dur", type=float, default=600.0)
    a.add_argument("--fps", type=float, default=4.0)
    a = sub.add_parser("match")
    a.add_argument("id")
    a.add_argument("--fps", type=float, default=4.0)
    a.add_argument("--only", choices=("last_hit", "missed_last_hit"),
                   help="в лист — только события этого типа")
    args = ap.parse_args()
    {"session": cmd_session, "match": cmd_match}[args.cmd](args)


if __name__ == "__main__":
    main()
