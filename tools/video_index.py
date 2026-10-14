"""Индекс роликов YouTube для набора: годится ли ролик, не скачивая его (#843).

Подкоманды (подробно — docs/video-index.md):

* `channel <url> [--tab videos|streams] [--since patch|season|26.18|2026-09-01]`
  — пройти канал: плоский список, отбор по длине и дате, предпросмотр
  каждого нового ролика, строка в индексе;
* `search "<query>" [--n 30] [--since ...]` — то же по поиску YouTube;
* `video <id> [<id> ...] [--force]` — разобрать названные ролики;
* `show [--champion Volibear] [--role top] [--view player] [--usable]
  [--channel ...] [--json]` — выборка из индекса;
* `assets` — докачать иконки умений и чемпионов Data Dragon (один раз).

Предпросмотр ролика (lolstream/youtube_preview.py): сведения `yt-dlp -J`,
раскадровка → лента типа сцены → отрезки игры → 5–10 точечных кадров 1080p
из HLS-потока → модели (тип сцены, раскладка HUD, часы матча, миникарта и
камера, свой чемпион по иконкам умений) → итог lolvision/video_verdict.py.

Индекс — SQLite `D:\\bc-data\\index\\videos.sqlite` (одна строка на ролик);
кадры предпросмотра и лист для проверки глазами — `D:\\bc-data\\previews\\<id>\\`.
Повторный проход берёт только ролики, которых в индексе ещё нет (`--force` —
все). Модели — из runs/ основного каталога (`--runs`), в git их нет.

Пример:
    python tools/video_index.py assets
    python tools/video_index.py channel https://www.youtube.com/@ProTopArchive --since season
    python tools/video_index.py search "Volibear top challenger" --n 40
    python tools/video_index.py show --champion Volibear --role top --usable
"""

# ruff: noqa: E402 — потоки BLAS задаются до импорта numpy
from __future__ import annotations

import os

# Скромно по ЦП (машина — рабочая, на ней же игра, coach и сервис): потоки
# BLAS/OpenMP — до импорта numpy, приоритет процесса — ниже обычного (его
# наследуют yt-dlp и ffmpeg). Замер 2026-10-15: два прохода по 2–4 ролика без
# этого заняли ~98% ЦП.
CPU_THREADS = os.environ.get("VIDEO_INDEX_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, CPU_THREADS)

import argparse
import datetime as dt
import json
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401 — корень репозитория в sys.path
from lolvision import hud_champion, hud_spells, hud_stats, minimap_locate, patch_calendar
from lolvision import rift, roster, video_meta, video_verdict as vv
from lolvision.ddragon import DDragon
from lolvision.minimap_camera import BOX_H, BOX_W, MM, find_box_clipped, white_mask
from lolvision.paths import RUNS
from lolstream import youtube_preview as yp
from lolstream.live_champions import player_hud, spectator_hud

INDEX = Path(os.environ.get("BC_INDEX", r"D:\bc-data\index"))
PREVIEWS = Path(os.environ.get("BC_PREVIEWS", r"D:\bc-data\previews"))
DDRAGON = INDEX / "ddragon"
DD_VERSION = "16.20.1"          # патч 26.20 (lolvision/patch_calendar.py)
# Модели не в git: рабочая копия берёт их из основного каталога.
MAIN_RUNS = Path(r"C:\develop\games\lol\lol-game-state-vision\runs")
SCENE_NET = Path("scene_kind") / "net" / "model.pt"
SCHEMA = 1                      # версия разбора: строки старее — повод разобрать заново
MIN_LIST_S = 8 * 60             # короче по списку — не полная игра, кадры не тянем
APPROX_SLACK_D = 10             # грубая дата списка: запас на округление «N weeks ago»
CAMERA_Q = 0.35                 # качество рамки камеры не ниже — камере верим
BOX0 = (int(BOX_W * MM), int(BOX_H * MM))
BOT_PAUSE_S = 300.0             # пауза после «Sign in to confirm you're not a bot»

COLUMNS = {
    "id": "TEXT PRIMARY KEY", "channel": "TEXT", "channel_id": "TEXT", "sources": "TEXT",
    "title": "TEXT", "upload_date": "TEXT", "approx_date": "TEXT", "duration": "REAL",
    "patch": "TEXT", "patch_source": "TEXT", "status": "TEXT",
    "view": "TEXT", "view_share": "REAL", "champion": "TEXT", "champion_source": "TEXT",
    "role": "TEXT", "role_source": "TEXT", "rank_claim": "TEXT", "champion_claim": "TEXT",
    "role_claim": "TEXT", "full_game": "INTEGER", "cut_s": "REAL", "cuts": "INTEGER",
    "usable": "INTEGER", "confidence": "REAL",
    "reasons": "TEXT", "notes": "TEXT", "segments": "TEXT", "games": "TEXT",
    "minimap_share": "REAL", "minimap_scale": "REAL", "hud_share": "REAL",
    "frames": "TEXT", "timing": "TEXT", "traffic_bytes": "INTEGER", "preview_s": "REAL",
    "analyzed_at": "TEXT", "schema": "INTEGER", "error": "TEXT", "eye": "TEXT",
}


def be_nice() -> str:
    """Приоритет процесса «ниже обычного» (Windows) или nice 10; потоки
    OpenCV и torch — CPU_THREADS. Возвращает, что сделано."""
    n = int(CPU_THREADS)
    cv2.setNumThreads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL
        return f"приоритет BelowNormal {'задан' if ok else 'НЕ задан'}, потоков {n}"
    os.nice(10)
    return f"nice 10, потоков {n}"


# ---------------------------------------------------------------- индекс


class Index:
    """SQLite: одна строка на ролик. Пишет один поток за раз (lock)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        cols = ", ".join(f"{k} {v}" for k, v in COLUMNS.items())
        self.db.execute(f"CREATE TABLE IF NOT EXISTS videos ({cols})")
        have = {r[1] for r in self.db.execute("PRAGMA table_info(videos)")}
        for k, v in COLUMNS.items():
            if k not in have:
                self.db.execute(f"ALTER TABLE videos ADD COLUMN {k} {v}")
        self.db.commit()

    def get(self, vid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
        return dict(r) if r else None

    def put(self, row: dict) -> None:
        old = self.get(row["id"]) or {}
        src = set(filter(None, (old.get("sources") or "").split("|")))
        src |= set(filter(None, (row.get("sources") or "").split("|")))
        row = {**{k: old.get(k) for k in COLUMNS}, **row, "sources": "|".join(sorted(src))}
        for k, v in list(row.items()):
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v, ensure_ascii=False)
        keys = [k for k in COLUMNS if k in row]
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO videos ({', '.join(keys)}) VALUES "
                            f"({', '.join('?' for _ in keys)})", [row[k] for k in keys])
            self.db.commit()

    def add_source(self, vid: str, source: str) -> None:
        old = self.get(vid)
        if old is None:
            return
        src = set(filter(None, (old.get("sources") or "").split("|"))) | {source}
        with self.lock:
            self.db.execute("UPDATE videos SET sources = ? WHERE id = ?",
                            ("|".join(sorted(src)), vid))
            self.db.commit()

    def rows(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM videos")]


# ---------------------------------------------------------------- модели


def runs_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return RUNS if (RUNS / SCENE_NET).exists() else MAIN_RUNS


class Readers:
    """Модели и банки одного прохода. Вызовы моделей — под замком: сеть
    типа сцены — CUDA graph на батч 1, он не для нескольких потоков."""

    def __init__(self, runs: Path, device: str = "cuda"):
        from lolvision import scene_net
        self.lock = threading.Lock()
        self.scene = scene_net.SceneNetRunner(runs / SCENE_NET, device)
        self.classes = list(self.scene.classes)
        self.stats = hud_stats.StatsReader()
        dd = DDragon(DD_VERSION, root=DDRAGON)
        self.dd = dd
        self.champs = dd.champions()
        self.names = video_meta.ChampionNames({c: v["name"] for c, v in self.champs.items()})
        self.icons = roster.IconBank.from_ddragon(dd)
        self.abilities = hud_champion.AbilityBank.from_ddragon(dd)
        self._locators: dict[tuple[int, int], minimap_locate.Locator] = {}

    def support_like(self, cid: str | None) -> bool | None:
        if cid not in self.champs:
            return None
        tags = self.champs[cid].get("tags") or []
        return "Support" in tags and "Marksman" not in tags

    def scene_of(self, rgb: np.ndarray) -> tuple[str, float]:
        with self.lock:
            p = self.scene.probs(rgb)
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])

    def locator(self, size: tuple[int, int]) -> minimap_locate.Locator:
        with self.lock:
            if size not in self._locators:
                self._locators[size] = minimap_locate.Locator(size)
            return self._locators[size]


def _minimap_found(fit) -> bool:
    if fit is None:
        return False
    if fit.how == "frame":
        return fit.score >= minimap_locate.FRAME_MIN
    return fit.how in ("saved", "manual") or fit.score >= minimap_locate.MIN_SCORE


def analyze_frame(rd: Readers, rgb: np.ndarray, t: float, seg: int) -> tuple[vv.FrameFacts,
                                                                              np.ndarray | None]:
    """Факты одного кадра и оценки чемпионов по иконкам умений (или None)."""
    f = vv.FrameFacts(t=t, segment=seg)
    f.scene, f.scene_p = rd.scene_of(rgb)
    if f.scene not in vv.IN_GAME:
        return f, None
    h, w = rgb.shape[:2]
    img = rgb if (w, h) == (1920, 1080) else cv2.resize(rgb, (1920, 1080),
                                                        interpolation=cv2.INTER_AREA)
    # Раскладка: портреты наблюдателя сильнее полосы здоровья игрока.
    panel = hud_spells.locate_panel(img)
    f.panel_score = None if panel is None else panel.score
    st = rd.stats.read(img[..., ::-1])
    f.clock = st.time
    if player_hud(img) or (panel is not None and panel.score >= hud_spells.FOUND_MIN) \
            or (st.present and st.time is not None):
        # Положительный признак игрока; портреты наблюдателя проверяются
        # всё равно — у наблюдателя полосы игрока нет, но мало ли оверлей.
        fs = roster.read_frame(img, rd.icons)
        f.view_read = "spectator" if spectator_hud(fs) else "player"
    else:
        fs = roster.read_frame(img, rd.icons)
        f.view_read = "spectator" if spectator_hud(fs) else None
    # Миникарта и камера.
    loc = rd.locator((1920, 1080))
    x0, y0, x1, y1 = loc.region
    fit = loc.locate(img[y0:y1, x0:x1])
    f.minimap_found = _minimap_found(fit)
    f.minimap_score = None if fit is None else fit.score
    f.minimap_scale = None if fit is None else fit.scale
    box = fit.box if f.minimap_found else (minimap_locate.STD_BOX
                                           if f.view_read == "spectator" else None)
    if box is not None:
        bx0, by0, bx1, by1 = minimap_locate.box_px(box, (1920, 1080))
        mini = Image.fromarray(np.ascontiguousarray(img[by0:by1, bx0:bx1])).resize(
            (MM, MM), Image.LANCZOS)
        mask = white_mask(np.asarray(mini, dtype=np.float32))
        x, y, q, _ = find_box_clipped(mask, *BOX0)
        if q >= CAMERA_Q:
            u, v = (x + BOX0[0] / 2) / MM, (y + BOX0[1] / 2) / MM
            f.camera = (u, v)
            f.zone = rift.zone(*rift.crop_to_world(u, v))
    scores = None
    if f.view_read == "player" and panel is not None and panel.score >= hud_spells.PRESENT_MIN:
        scores = rd.abilities.scores(img, panel)
        top = np.argsort(-scores)[:5]
        f.champion_top = [(rd.abilities.ids[i], float(scores[i])) for i in top]
    return f, scores


# ---------------------------------------------------------------- ролик


def _rle(tile_cls: list[tuple[float, str]]) -> list:
    """Лента раскадровки сжато: [[t начала, класс, число кадров], ...]."""
    out: list = []
    for t, c in tile_cls:
        if out and out[-1][1] == c:
            out[-1][2] += 1
        else:
            out.append([round(t, 1), c, 1])
    return out


def _jpeg(path: Path, rgb: np.ndarray, width: int = 1280) -> None:
    h, w = rgb.shape[:2]
    img = cv2.resize(rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def contact_sheet(path: Path, shots: list, facts: list[vv.FrameFacts], title: str,
                  verdict: dict) -> None:
    """Лист для проверки глазами: кадры 480×270 с подписями и итог."""
    tw, th, cols = 480, 270, 4
    cells = [(s, f) for s, f in zip(shots, facts) if s.image is not None]
    rows = max(1, (len(cells) + cols - 1) // cols)
    sheet = np.full((rows * th + 60, cols * tw, 3), 24, np.uint8)
    for k, (s, f) in enumerate(cells):
        r, c = divmod(k, cols)
        im = cv2.resize(s.image, (tw, th), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        lab = (f"{s.t:.0f}s {f.scene} {f.view_read or '-'} clk={f.clock} "
               f"mm={'Y' if f.minimap_found else 'n'} {f.zone or ''}")
        lab2 = " ".join(f"{c_}:{v:.2f}" for c_, v in f.champion_top[:2])
        for i, text in enumerate((lab, lab2)):
            cv2.putText(im, text, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(im, text, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
        sheet[60 + r * th:60 + (r + 1) * th, c * tw:(c + 1) * tw] = im
    head = (f"{'USABLE' if verdict['usable'] else 'no'} view={verdict['view']} "
            f"champ={verdict['champion']}({verdict['champion_source']}) role={verdict['role']} "
            f"full={verdict['full_game']} conf={verdict['confidence']:.2f}")
    t_ascii = title.encode("ascii", "replace").decode()[:110]
    cv2.putText(sheet, t_ascii, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(sheet, head, (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def preview(rd: Readers, vid: str, since: dt.date | None, source: str,
            max_frames: int = vv.MAX_FRAMES, list_entry: dict | None = None) -> dict:
    """Предпросмотр и итог по одному ролику → строка индекса."""
    t0 = time.time()
    timing: dict[str, float] = {}
    traffic = yp.Traffic()
    info = yp.video_info(vid)
    timing["info"] = info.seconds
    upload = patch_calendar.parse_date(info.upload_date)
    patch, psrc = patch_calendar.patch_of(info.title, info.description, upload)
    claims = video_meta.claims(info.title, info.description, rd.names)
    row = {"id": info.id, "channel": info.channel, "channel_id": info.channel_id,
           "sources": source, "title": info.title, "upload_date": info.upload_date,
           "approx_date": (list_entry or {}).get("approx_date"), "duration": info.duration,
           "patch": patch, "patch_source": psrc, "rank_claim": claims["rank"],
           "champion_claim": claims["champion"], "role_claim": claims["role"],
           "analyzed_at": dt.datetime.now().isoformat(timespec="seconds"), "schema": SCHEMA}
    if since is not None and upload is not None and upload < since:
        return {**row, "status": "old", "usable": 0, "reasons": [f"опубликован {upload} < {since}"],
                "preview_s": round(time.time() - t0, 2), "traffic_bytes": traffic.bytes}
    if info.live_status in ("is_live", "is_upcoming"):
        return {**row, "status": "live", "usable": 0, "reasons": ["трансляция идёт или будет"],
                "preview_s": round(time.time() - t0, 2)}
    # Раскадровка → лента.
    t1 = time.time()
    tiles = yp.storyboard(info, traffic)
    timing["storyboard"] = time.time() - t1
    t1 = time.time()
    tile_cls = [(tl.t, rd.scene_of(tl.image)[0]) for tl in tiles]
    timing["storyboard_scene"] = time.time() - t1
    segs = vv.timeline(tile_cls, info.duration)
    share = vv.game_share(tile_cls)
    picks = vv.pick_times(segs, max_frames)
    if not tiles:
        # Без раскадровки — равномерно по ролику, как будто он весь игра.
        segs = [vv.Segment(0.0, info.duration, 0, 1.0)]
        picks = vv.pick_times(segs, max_frames)
        share = 1.0
    # Точечные кадры.
    t1 = time.time()
    shots = []
    if picks and info.hls is not None:
        hls = yp.HlsStream(info, traffic)
        shots = hls.shots([t for _, t in picks])
    timing["shots"] = time.time() - t1
    bad = [s for s in shots if s.image is None]
    if shots and len(bad) > len(shots) // 2:
        # Сеть, а не ролик: строка — ошибка, следующий проход разберёт заново.
        raise RuntimeError(f"кадры не получены: {len(bad)} из {len(shots)} ({bad[0].error})")
    t1 = time.time()
    facts, per_frame = [], []
    for (seg, _), s in zip(picks, shots):
        if s.image is None:
            facts.append(vv.FrameFacts(t=s.t, segment=seg, error=s.error))
            continue
        f, sc = analyze_frame(rd, s.image, s.t, seg)
        facts.append(f)
        if sc is not None:
            per_frame.append(sc)
    timing["models"] = time.time() - t1
    champ = hud_champion.decide(rd.abilities.ids, per_frame).to_dict() if per_frame else None
    cid = (champ or {}).get("champion") or claims["champion"]
    ver = vv.verdict(segs, facts, claims, champ, share, rd.support_like(cid)).to_dict()
    if info.hls is None:
        ver["reasons"].append("нет HLS-потока: точечных кадров нет")
    # Предпросмотр на диск.
    d = PREVIEWS / info.id
    d.mkdir(parents=True, exist_ok=True)
    for s in shots:
        if s.image is not None:
            _jpeg(d / f"t{int(s.t):05d}.jpg", s.image)
    if tiles:
        strip = np.concatenate([tl.image for tl in tiles[::max(1, len(tiles) // 24)][:24]], 1)
        _jpeg(d / "storyboard.jpg", strip, width=min(3840, strip.shape[1]))
    contact_sheet(d / "sheet.jpg", shots, facts, info.title, ver)
    shot_s = [s.seconds for s in shots]
    timing = {k: round(v, 2) for k, v in timing.items()}
    timing["per_shot_median"] = round(float(np.median(shot_s)), 2) if shot_s else None
    timing["shots_n"] = len(shots)
    timing["shot_errors"] = sum(s.image is None for s in shots)
    timing["tiles"] = len(tiles)
    out = {**row, "status": "done", "view": ver["view"], "view_share": ver["view_share"],
           "champion": ver["champion"], "champion_source": ver["champion_source"],
           "role": ver["role"], "role_source": ver["role_source"],
           "full_game": int(ver["full_game"]), "usable": int(ver["usable"]),
           "cut_s": ver["cut_s"], "cuts": ver["cuts"],
           "confidence": ver["confidence"], "reasons": ver["reasons"], "notes": ver["notes"],
           "segments": [s.to_dict() for s in segs], "games": ver["games"],
           "minimap_share": ver["minimap_share"], "minimap_scale": ver["minimap_scale"],
           "hud_share": ver["hud_share"],
           "frames": {"facts": [f.to_dict() for f in facts], "champion": champ,
                      "tiles_game_share": round(share, 4), "tiles": _rle(tile_cls),
                      "description": info.description[:2000]},
           "timing": timing, "traffic_bytes": traffic.bytes,
           "preview_s": round(time.time() - t0, 2), "error": None}
    (d / "facts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- проход


def _keep(e: dict, since: dt.date | None) -> str | None:
    """Почему запись списка не берётся (None — брать)."""
    if (e.get("url") or "").find("/shorts/") >= 0:
        return "шорт"
    dur = e.get("duration")
    if dur is not None and dur < MIN_LIST_S:
        return f"короче {MIN_LIST_S // 60} мин ({dur:.0f} с)"
    if e.get("live_status") in ("is_live", "is_upcoming"):
        return "трансляция идёт или будет"
    ad = patch_calendar.parse_date(e.get("approx_date"))
    if since is not None and ad is not None and ad < since - dt.timedelta(days=APPROX_SLACK_D):
        return f"старый (≈{ad})"
    return None


def run_listing(rd: Readers, idx: Index, listing: dict, source: str, since: dt.date | None,
                force: bool, workers: int, max_videos: int | None, max_frames: int) -> dict:
    entries = listing["entries"]
    todo, skipped, known, old_run = [], {}, 0, 0
    for e in entries:
        why = _keep(e, since)
        if why is not None:
            skipped[why.split(" (")[0]] = skipped.get(why.split(" (")[0], 0) + 1
            if why.startswith("старый"):
                old_run += 1
            continue
        old = idx.get(e["id"])
        if old is not None and not force and (old.get("schema") or 0) >= SCHEMA:
            idx.add_source(e["id"], source)
            known += 1
            continue
        todo.append(e)
    if max_videos:
        todo = todo[:max_videos]
    print(f"[{source}] в списке {len(entries)}, к разбору {len(todo)}, уже в индексе {known}, "
          f"пропущено {skipped}", flush=True)
    t0 = time.time()
    done = usable = errors = 0

    def one(e: dict) -> dict:
        try:
            return preview(rd, e["id"], since, source, max_frames, e)
        except yp.ExpiredError:
            return preview(rd, e["id"], since, source, max_frames, e)
        except yp.BotCheckError:
            # YouTube просит войти: пауза на весь поток и одна попытка ещё.
            print(f"  бот-проверка на {e['id']}: пауза {BOT_PAUSE_S:.0f} с", flush=True)
            time.sleep(BOT_PAUSE_S)
            return preview(rd, e["id"], since, source, max_frames, e)

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, e): e for e in todo}
        for fu in as_completed(futs):
            e = futs[fu]
            try:
                row = fu.result()
            except Exception as exc:                          # noqa: BLE001
                errors += 1
                row = {"id": e["id"], "title": e.get("title"), "sources": source,
                       "duration": e.get("duration"), "approx_date": e.get("approx_date"),
                       "status": "error", "usable": 0, "schema": 0,
                       "error": f"{type(exc).__name__}: {exc}"[:500],
                       "analyzed_at": dt.datetime.now().isoformat(timespec="seconds")}
                traceback.print_exc()
            idx.put(row)
            done += 1
            usable += int(row.get("usable") or 0)
            print(f"  {done}/{len(todo)} {row['id']} {row.get('status')} "
                  f"usable={row.get('usable')} view={row.get('view')} "
                  f"champ={row.get('champion')} role={row.get('role')} "
                  f"{row.get('preview_s')}s {((row.get('traffic_bytes') or 0) / 1e6):.1f}MB "
                  f"| {(row.get('title') or '')[:60]}", flush=True)
    dt_s = time.time() - t0
    rate = done / (dt_s / 60) if dt_s > 0 else 0.0
    print(f"[{source}] разобрано {done} за {dt_s:.0f} с ({rate:.1f} роликов/мин), "
          f"годных {usable}, ошибок {errors}", flush=True)
    return {"done": done, "usable": usable, "errors": errors, "seconds": dt_s}


def channel_url(url: str, tab: str) -> str:
    url = url.rstrip("/")
    if url.startswith("@"):
        url = f"https://www.youtube.com/{url}"
    for t in ("/videos", "/streams", "/shorts", "/featured"):
        if url.endswith(t):
            url = url[: -len(t)]
    return f"{url}/{tab}"


# ---------------------------------------------------------------- show


def reverdict(idx: Index) -> None:
    """Итог заново по сохранённым отрезкам и фактам кадров — без сети и
    моделей: после правки правил lolvision/video_verdict.py."""
    dd = DDragon(DD_VERSION, root=DDRAGON)
    champs = dd.champions()
    names = video_meta.ChampionNames({c: v["name"] for c, v in champs.items()})

    def support_like(cid):
        tags = (champs.get(cid) or {}).get("tags")
        return None if tags is None else ("Support" in tags and "Marksman" not in tags)

    n = changed = 0
    for r in idx.rows():
        if r.get("status") != "done" or not r.get("frames"):
            continue
        fr = json.loads(r["frames"])
        segs = [vv.Segment(x["start"], x["end"], x["tiles"], x["game_share"])
                for x in json.loads(r.get("segments") or "[]")]
        facts = [vv.FrameFacts.from_dict(f) for f in fr["facts"]]
        claims = video_meta.claims(r.get("title") or "", fr.get("description") or "", names)
        if r.get("champion_claim"):
            claims["champion"] = r["champion_claim"]
        claims["role"] = claims["role"] or r.get("role_claim")
        champ = fr.get("champion")
        tiles_share = fr.get("tiles_game_share")
        if tiles_share is None:
            tiles_share = (sum(s.tiles * s.game_share for s in segs)
                           / max(1, sum(s.tiles for s in segs))) if segs else 0.0
        cid = (champ or {}).get("champion") or claims["champion"]
        v = vv.verdict(segs, facts, claims, champ, tiles_share, support_like(cid)).to_dict()
        new = {"id": r["id"], "view": v["view"], "view_share": v["view_share"],
               "champion": v["champion"], "champion_source": v["champion_source"],
               "role": v["role"], "role_source": v["role_source"],
               "full_game": int(v["full_game"]), "usable": int(v["usable"]),
               "cut_s": v["cut_s"], "cuts": v["cuts"],
               "confidence": v["confidence"], "reasons": v["reasons"], "notes": v["notes"],
               "games": v["games"], "minimap_share": v["minimap_share"],
               "minimap_scale": v["minimap_scale"], "hud_share": v["hud_share"]}
        n += 1
        changed += int(bool(r.get("usable")) != bool(new["usable"])
                       or r.get("champion") != new["champion"] or r.get("role") != new["role"])
        idx.put(new)
    print(f"итог пересчитан у {n} роликов, изменился у {changed}")


def show(idx: Index, a) -> None:
    rows = [r for r in idx.rows() if r.get("status") == "done"]
    if a.champion:
        key = video_meta.ChampionNames({a.champion: a.champion})
        rows = [r for r in rows if (r.get("champion") or "").lower() == a.champion.lower()
                or (a.claims and key.find(r.get("title") or ""))]
    if a.role:
        rows = [r for r in rows if r.get("role") == a.role]
    if a.view:
        rows = [r for r in rows if r.get("view") == a.view]
    if a.usable:
        rows = [r for r in rows if r.get("usable")]
    if a.channel:
        rows = [r for r in rows if a.channel.lower() in (r.get("channel") or "").lower()]
    if a.patch:
        rows = [r for r in rows if r.get("patch") == patch_calendar.norm(a.patch)]
    if a.max_cut is not None:
        rows = [r for r in rows if r.get("cut_s") is not None and r["cut_s"] <= a.max_cut]
    if a.minimap:
        rows = [r for r in rows if (r.get("minimap_share") or 0) >= vv.MINIMAP_SHARE]
    rows.sort(key=lambda r: (r.get("upload_date") or "", r["id"]), reverse=True)
    if a.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return
    for r in rows:
        g = json.loads(r.get("games") or "[]")
        start = f"матч с {g[0]['start']:.0f}с" if g else ""
        print(f"{r['id']} {r.get('upload_date')} p{r.get('patch')} {r.get('view'):9} "
              f"{(r.get('champion') or '-'):12} {(r.get('role') or '-'):7} "
              f"{'Y' if r.get('usable') else 'n'} {r.get('confidence') or 0:.2f} "
              f"mm={r.get('minimap_share') if r.get('minimap_share') is not None else '-'} "
              f"{(r.get('duration') or 0) / 60:.0f}мин {start} "
              f"cut={'-' if r.get('cut_s') is None else int(r['cut_s'])} | {r.get('channel')} | "
              f"{(r.get('title') or '')[:70]}")
    print(f"всего {len(rows)}")


def fetch_assets() -> None:
    from lolstream.ddragon_net import fetch
    dd = DDragon(DD_VERSION, root=DDRAGON, fetch=fetch)
    ch = dd.champions()

    def one(cid: str) -> None:
        dd.icon(cid)
        for s in dd.champion_full(cid)["spells"]:
            dd.spell_icon(s["image"]["full"])

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, sorted(ch)))
    print(f"Data Dragon {DD_VERSION}: {len(ch)} чемпионов, иконки умений → {DDRAGON}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", default=str(INDEX / "videos.sqlite"))
    ap.add_argument("--runs", default=None, help="каталог runs/ с моделями")
    ap.add_argument("--device", default="cuda")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--since", default="season",
                       help="patch | season | 26.18 | 2026-09-01 | all")
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--max", type=int, default=None, help="не больше стольких роликов")
        p.add_argument("--frames", type=int, default=vv.MAX_FRAMES)
        p.add_argument("--force", action="store_true")

    p = sub.add_parser("channel")
    p.add_argument("url")
    p.add_argument("--tab", default="videos", choices=("videos", "streams"))
    p.add_argument("--limit", type=int, default=200, help="длина плоского списка")
    common(p)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--n", type=int, default=30)
    common(p)
    p = sub.add_parser("video")
    p.add_argument("ids", nargs="+")
    common(p)
    p = sub.add_parser("show")
    p.add_argument("--champion")
    p.add_argument("--claims", action="store_true", help="и по названию, если HUD не решил")
    p.add_argument("--role", choices=video_meta.ROLES)
    p.add_argument("--view", choices=("player", "spectator", "unknown"))
    p.add_argument("--usable", action="store_true")
    p.add_argument("--minimap", action="store_true", help="только с видимой миникартой")
    p.add_argument("--max-cut", type=float, default=None,
                   help="вырезано не больше стольких секунд матча (по часам HUD)")
    p.add_argument("--channel")
    p.add_argument("--patch")
    p.add_argument("--json", action="store_true")
    sub.add_parser("assets")
    sub.add_parser("reverdict")
    a = ap.parse_args(argv)

    if a.cmd == "assets":
        fetch_assets()
        return 0
    idx = Index(Path(a.index))
    if a.cmd == "show":
        show(idx, a)
        return 0
    if a.cmd == "reverdict":
        reverdict(idx)
        return 0
    since = None if a.since == "all" else patch_calendar.since(a.since)
    print(be_nice(), flush=True)
    rd = Readers(runs_dir(a.runs), a.device)
    if a.cmd == "channel":
        url = channel_url(a.url, a.tab)
        listing = yp.list_playlist(url, a.limit)
        run_listing(rd, idx, listing, f"channel:{url}", since, a.force, a.workers, a.max,
                    a.frames)
    elif a.cmd == "search":
        listing = yp.search(a.query, a.n)
        run_listing(rd, idx, listing, f"search:{a.query}", since, a.force, a.workers, a.max,
                    a.frames)
    elif a.cmd == "video":
        listing = {"entries": [{"id": v} for v in a.ids]}
        run_listing(rd, idx, listing, "video", since, a.force, a.workers, a.max, a.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
