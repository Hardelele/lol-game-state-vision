"""Каталог каналов, список роликов канала и забор ролика в датасет.

Роликов в сети десятки тысяч, а в датасете их пока восемнадцать. Выбирать
следующий вслепую по идентификатору неудобно: нужно видеть канал, название
матча, роль, патч — и сразу понимать, брали мы этот ролик или нет.

Ключевые каналы уже есть: `dataset/sources/champion-replay-channels.json`
хранит по каналу на каждого из 173 чемпионов, а
`dataset/sources/top-player-channels.json` — каналы топ-игроков и стримеров
с записями от первого лица (другой HUD). Свои добавляются поверх и
лежат отдельно, в `data/channels/custom.json`, чтобы собранный каталог
оставался неизменным.

Список роликов берётся плоским запросом (`--flat-playlist`): он не ходит на
страницу каждого ролика и потому быстр — десяток роликов за 1.7 с. Ответ
кешируется на диск: без кеша каждое открытие канала — это сетевой запрос.

Сам ролик не скачивается нигде: в датасет он попадает потоком через
build_coords.py --url, а смотреть его можно плеером YouTube.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from paths import ROOT

CATALOGS = (ROOT / "dataset" / "sources" / "champion-replay-channels.json",
            ROOT / "dataset" / "sources" / "top-player-channels.json")
STORE = ROOT / "data" / "channels"
CACHE = STORE / "cache"
CUSTOM = STORE / "custom.json"
COORDS = ROOT / "data" / "coords"
VIDEOS = ROOT / "data" / "videos"
LAYOUT = ROOT / "dataset" / "layouts" / "spectator-volibear-challenger.json"
INGEST_LOGS = ROOT / "runs" / "ingest"

FRESH = 6 * 3600          # сколько секунд считаем кеш списка годным
ROLES = ("Top", "Jungle", "Mid", "ADC", "Bot", "Support")
YT = ["python", "-m", "yt_dlp"]


def _run(args: list[str], timeout: float = 180) -> str:
    out = subprocess.run(YT + args, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
    if out.returncode != 0 and not out.stdout.strip():
        tail = (out.stderr or "").strip().splitlines()[-1:] or ["неизвестно"]
        raise RuntimeError(tail[0][:300])
    return out.stdout


# ---------- каталог ----------

def _custom() -> list[dict]:
    if not CUSTOM.exists():
        return []
    return json.loads(CUSTOM.read_text(encoding="utf-8"))


def _save_custom(rows: list[dict]) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    CUSTOM.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")


def have_ids() -> dict[str, dict]:
    """Какие ролики уже у нас: кадры с координатами и сведения о ролике."""
    out: dict[str, dict] = {}
    for p in sorted(COORDS.glob("*.csv")):
        if p.stem == "roles":
            continue
        n = sum(1 for _ in p.open(encoding="utf-8")) - 1
        out[p.stem] = {"frames": max(0, n)}
    for p in sorted(VIDEOS.glob("*.info.json")):
        out.setdefault(p.stem.replace(".info", ""), {})["info"] = True
    return out


def _owner(vid: str) -> tuple[str | None, str | None]:
    """Канал, с которого взят ролик: по сохранённому описанию."""
    p = VIDEOS / f"{vid}.info.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return None, None
    return d.get("uploader_id"), d.get("channel_id")


def channels() -> list[dict]:
    """Каталог: собранные каналы плюс добавленные руками.

    Канал, с которого уже брались ролики, помечается — именно с таких и
    хочется продолжать, а искать их в списке из 173 штук утомительно.
    """
    cat = [c for p in CATALOGS if p.exists()
           for c in json.loads(p.read_text(encoding="utf-8"))["channels"]]
    mine: dict[str, int] = {}
    for vid in have_ids():
        handle, cid = _owner(vid)
        for key in (handle, cid):
            if key:
                mine[key] = mine.get(key, 0) + 1

    rows = []
    for c in cat:
        rows.append({"handle": c["handle"], "url": c["url"],
                     "name": (c.get("champion") or c.get("player")
                              or c.get("channel_title")),
                     "title": c.get("channel_title"), "id": c.get("channel_id"),
                     "videos": c.get("videos"), "subs": c.get("subscribers"),
                     "custom": False})
    for c in _custom():
        if any(r["handle"].lower() == c["handle"].lower() for r in rows):
            continue
        rows.append({**c, "custom": True})
    for r in rows:
        r["mine"] = mine.get(r["handle"], 0) or mine.get(r["id"] or "", 0)
        r["cached"] = _cache_file(r["handle"]).exists()
    rows.sort(key=lambda r: (-r["mine"], not r["custom"], (r["name"] or "").lower()))
    return rows


def add_channel(url: str) -> dict:
    """Добавить канал по ссылке или @имени; название спрашиваем у YouTube."""
    url = url.strip()
    if not url:
        raise ValueError("пустая ссылка")
    if "://" not in url:
        url = f"https://www.youtube.com/{url.lstrip('/') if url.startswith('@') else '@' + url}"
    base = url.split("/videos")[0].rstrip("/")
    d = json.loads(_run(["--flat-playlist", "--playlist-end", "1",
                         "--dump-single-json", base + "/videos"], timeout=90))
    handle = d.get("uploader_id") or d.get("channel") or base.rsplit("/", 1)[-1]
    rec = {"handle": handle, "url": base,
           "name": d.get("channel") or d.get("title") or handle,
           "title": d.get("channel") or d.get("title"),
           "id": d.get("channel_id"),
           "videos": (f"{d['playlist_count']} videos"
                      if d.get("playlist_count") else None),
           "subs": (f"{d['channel_follower_count']} subscribers"
                    if d.get("channel_follower_count") else None)}
    rows = [r for r in _custom() if r["handle"].lower() != handle.lower()]
    rows.append(rec)
    _save_custom(rows)
    return rec


def drop_channel(handle: str) -> None:
    """Удаляются только добавленные руками: каталог правится в репозитории."""
    rows = [r for r in _custom() if r["handle"].lower() != handle.lower()]
    _save_custom(rows)


# ---------- ролики канала ----------

def _cache_file(handle: str) -> Path:
    safe = re.sub(r"[^\w@.-]", "_", handle)
    return CACHE / f"{safe}.json"


def parse_title(title: str) -> dict:
    """Роль, соперник, патч и регион — из названия ролика.

    Канал пишет название по шаблону «Ahri Mid vs Cassiopeia - KR Grandmaster
    Patch 26.19», так что фильтры получаются без единого запроса в сеть.
    Стороны в названии нет — она только в описании, см. video_side.py.
    """
    t = title or ""
    role = next((r for r in ROLES if re.search(rf"\b{r}\b", t, re.I)), None)
    vs = re.search(r"\bvs\.?\s+([A-Z][\w'. ]*?)(?:\s*\(|\s+-|$)", t)
    patch = re.search(r"Patch\s+([\d.]+)", t, re.I)
    region = re.search(r"\b(KR|EUW|EUNE|NA|BR|JP|OCE|TR|LAN|LAS|RU|VN|PH|SG)\b", t)
    rank = re.search(r"\b(Challenger|Grandmaster|Master|Diamond)\b", t, re.I)
    return {"role": role, "vs": (vs.group(1).strip() if vs else None),
            "patch": patch.group(1) if patch else None,
            "region": region.group(1) if region else None,
            "rank": rank.group(1).title() if rank else None}


def _with_mine(handle: str, items: list[dict]) -> list[dict]:
    """Ролики этого канала, уже взятые в датасет, — наверх списка.

    Иначе их не найти: в датасете лежат матчи полугодовой давности, а канал
    отдаёт последние. Человек открывает канал и не понимает, брали отсюда
    что-то или нет.
    """
    seen = {v["id"] for v in items}
    head = []
    for vid in have_ids():
        if vid in seen:
            continue
        p = VIDEOS / f"{vid}.info.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        if d.get("uploader_id") != handle and d.get("channel_id") != handle:
            continue
        head.append({"id": vid, "title": d.get("title") or vid,
                     "duration": d.get("duration"), "views": None,
                     "thumb": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
                     **parse_title(d.get("title") or "")})
    return head + items


def videos(handle: str, url: str | None = None, limit: int = 60,
           refresh: bool = False) -> dict:
    """Список роликов канала с диска, при надобности — из сети."""
    f = _cache_file(handle)
    cached = {}
    if f.exists():
        try:
            cached = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            cached = {}
    fresh = (time.time() - cached.get("at", 0)) < FRESH
    enough = len(cached.get("items", [])) >= limit
    if cached and fresh and enough and not refresh:
        return {**cached, "items": _with_mine(handle, cached["items"][:limit]),
                "source": "кеш"}

    base = (url or f"https://www.youtube.com/{handle}").split("/videos")[0]
    txt = _run(["--flat-playlist", "--dump-json", "--playlist-end", str(limit),
                base.rstrip("/") + "/videos"])
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not d.get("id"):
            continue
        thumbs = d.get("thumbnails") or []
        items.append({"id": d["id"], "title": d.get("title") or d["id"],
                      "duration": d.get("duration"),
                      "views": d.get("view_count"),
                      "thumb": (thumbs[-1].get("url") if thumbs else None),
                      **parse_title(d.get("title") or "")})
    out = {"handle": handle, "at": time.time(), "items": _with_mine(handle, items)}
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {**out, "source": "сеть"}


# ---------- забор ролика в датасет ----------

_jobs: dict[str, dict] = {}
_jlock = threading.Lock()


def jobs() -> list[dict]:
    with _jlock:
        out = []
        for j in _jobs.values():
            p = j["proc"]
            code = p.poll()
            out.append({"id": j["id"], "state": ("идёт" if code is None else
                                                 "готово" if code == 0 else "сбой"),
                        "code": code, "started": j["started"],
                        "log": _tail(j["logfile"]), "fps": j["fps"]})
        return sorted(out, key=lambda r: -r["started"])


def _tail(p: Path, lines: int = 6) -> str:
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-lines:])
    except OSError:
        return ""


def ingest(vid: str, fps: float = 1.0) -> dict:
    """Запустить сборку кадров с координатами прямо из YouTube, без скачивания.

    Процесс отдельный, а не поток: распаковка видео занимает минуты, и
    держать её в сервере смотрелки — значит заморозить интерфейс.
    """
    if not re.fullmatch(r"[\w-]{6,20}", vid):
        raise ValueError("непохоже на идентификатор ролика")
    with _jlock:
        old = _jobs.get(vid)
        if old and old["proc"].poll() is None:
            return {"id": vid, "state": "уже идёт"}
    INGEST_LOGS.mkdir(parents=True, exist_ok=True)
    logfile = INGEST_LOGS / f"{vid}.log"
    fh = logfile.open("w", encoding="utf-8")
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT / "tools")}
    proc = subprocess.Popen(
        ["python", "-u", str(ROOT / "tools" / "build_coords.py"),
         "--url", vid, "--layout", str(LAYOUT), "--fps", str(fps)],
        cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
        env={**os.environ, **env})
    with _jlock:
        _jobs[vid] = {"id": vid, "proc": proc, "logfile": logfile,
                      "started": time.time(), "fps": fps}
    return {"id": vid, "state": "идёт"}


# ---------- сведения о конкретном ролике ----------

INFO = STORE / "info"


def video_info(vid: str) -> dict:
    """Роль и сторона записанного игрока по описанию ролика.

    Сторона есть только в описании, и достаётся она сетевым запросом, поэтому
    ответ кешируется. Ролики, уже взятые в датасет, читаются с диска.
    """
    import video_side

    local = VIDEOS / f"{vid}.info.json"
    cached = INFO / f"{vid}.json"
    if local.exists():
        raw = json.loads(local.read_text(encoding="utf-8"))
    elif cached.exists():
        raw = json.loads(cached.read_text(encoding="utf-8"))
    else:
        raw = video_side.fetch_info(vid)
        INFO.mkdir(parents=True, exist_ok=True)
        keep = ("id", "title", "description", "duration", "channel",
                "channel_id", "uploader_id", "upload_date", "width", "height")
        cached.write_text(json.dumps({k: raw.get(k) for k in keep},
                                     ensure_ascii=False), encoding="utf-8")
    a = video_side.analyse(raw)
    a["size"] = (f"{raw.get('width')}x{raw.get('height')}"
                 if raw.get("width") else None)
    a["channel"] = raw.get("channel")
    a["handle"] = raw.get("uploader_id")
    a["upload_date"] = raw.get("upload_date")
    return a
