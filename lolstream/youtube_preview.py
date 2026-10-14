"""Предпросмотр ролика YouTube без скачивания: сведения, раскадровка, точечные кадры (#843).

Зачем. Индекс роликов (tools/video_index.py) решает «годится ли ролик для
набора» по 5–10 кадрам, а не по всему видео (1080p60 — ~700 МБ на 25 мин).

Три уровня, от дешёвого к дорогому:

1. **Список канала / поиска** — `yt-dlp --flat-playlist -J` (~2 с на 30
   роликов): id, название, длина. Даты публикации в плоском списке нет;
   `youtubetab:approximate_date` даёт грубую («2 weeks ago» → одна метка на
   неделю) — её хватает, чтобы отсечь старое. Точная дата — из шага 2.
2. **Сведения о ролике** — `yt-dlp -J --skip-download` (~2.5 с): точная
   дата, описание, форматы. Из них берутся раскадровка и HLS-поток.
3. **Раскадровка** (`sb0`: листы 3×3 кадров 320×180, кадр раз в ~10 с у
   ролика 25 мин; у длинных реже) — ~1 МБ на ролик, сразу, без потока.
   Годится для грубой ленты «игра / не игра» (вход модели типа сцены —
   256×144, меньше кадра раскадровки). HUD на ней не прочитать.
4. **Точечные кадры полного качества** — из HLS-потока (`m3u8_native`,
   1080p avc): плейлист — список сегментов по ~6 с; сегмент начинается с
   ключевого кадра, и для одного кадра хватает первых ~0.5 МБ сегмента:
   читаем начало ответа, закрываем соединение, ffmpeg распаковывает первый
   кадр из трубы. Замер (6k5-vQ_GQ_M, 1080p60): ~0.7 с и 0.5 МБ на кадр
   против 3 с и 2.3–3.2 МБ целого сегмента; 8 потоков параллельно.

Почему не прямой адрес `videoplayback` с `ffmpeg -ss`: YouTube его душит —
ffmpeg за 90 с не получил ни кадра (замер 2026-10-15; то же записано в
lolstream/video_stream.py). Range в адресах сегментов HLS сервер не
соблюдает (отвечает 200 и всем телом), поэтому чтение обрывается самим
клиентом.

Срок жизни. Адреса потока и плейлиста подписаны и живут ~6 ч (параметр
`expire` в адресе). `HlsStream` помнит срок и перед кадром проверяет
его; истёкший — ошибка `ExpiredError`, вызывающий заново берёт сведения (шаг 2).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

YTDLP = [sys.executable, "-m", "yt_dlp"]
# Сколько байт начала сегмента читать за раз, пока ffmpeg не распакует кадр
# без ошибок (ключевой кадр 1080p — 0.2–0.45 МБ).
HEAD_BYTES = 450_000
HEAD_STEP = 300_000
HEAD_MAX = 4_000_000
TIMEOUT = 30.0
EXPIRE_SLACK = 300.0       # с: адрес, которому жить меньше, считается истёкшим
WORKERS = 4                # параллельных загрузок кадров одного ролика


class ExpiredError(RuntimeError):
    """Адрес потока истёк — взять сведения о ролике заново."""


# Запросы к YouTube через yt-dlp — по одному и не чаще MIN_GAP: при 4
# параллельных `-J` на втором десятке роликов YouTube ответил «Sign in to
# confirm you're not a bot» (замер 2026-10-15, 3 из 24). Временные сбои
# (бот-проверка, обрыв прокси, сброс соединения) повторяются с паузой RETRY_S.
MIN_GAP = 2.0
RETRIES = 3
RETRY_S = (20.0, 60.0, 180.0)
TRANSIENT = ("not a bot", "Unable to connect to proxy", "10054", "10061", "timed out",
             "HTTP Error 429", "HTTP Error 5", "Connection reset", "Read timed out")
_gate = threading.Lock()
_last = [0.0]


class BotCheckError(RuntimeError):
    """YouTube просит войти («not a bot») и после повторов."""


def _run(args: list[str], timeout: float = 120.0) -> str:
    err = ""
    for i in range(RETRIES + 1):
        with _gate:
            wait = _last[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            p = subprocess.run(YTDLP + args, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            err = f"timed out ({timeout:.0f} с)"
        else:
            if p.returncode == 0:
                return p.stdout
            err = " | ".join(p.stderr.strip().splitlines()[-2:])[:400]
        if i < RETRIES and any(k in err for k in TRANSIENT):
            time.sleep(RETRY_S[min(i, len(RETRY_S) - 1)])
            continue
        break
    cls = BotCheckError if "not a bot" in err else RuntimeError
    raise cls(f"yt-dlp {' '.join(args[-1:])}: {err}")


# ---------------------------------------------------------------- списки


def _entry(e: dict) -> dict:
    ts = e.get("timestamp")
    return {"id": e.get("id"), "title": e.get("title") or "", "duration": e.get("duration"),
            "approx_date": (time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else None),
            "live_status": e.get("live_status"), "url": e.get("url")}


def list_playlist(url: str, limit: int | None = None) -> dict:
    """Плоский список канала (`.../videos`, `.../streams`) или плейлиста:
    {"channel", "channel_id", "entries": [...]}. Шорты (`/shorts/`) и
    записи без длины отбрасывает вызывающий."""
    args = ["--flat-playlist", "-J", "--extractor-args", "youtubetab:approximate_date"]
    if limit:
        args += ["--playlist-end", str(limit)]
    d = json.loads(_run(args + [url], timeout=300))
    return {"channel": d.get("channel") or d.get("uploader") or d.get("title"),
            "channel_id": d.get("channel_id"), "url": url,
            "entries": [_entry(e) for e in d.get("entries") or [] if e.get("id")]}


def search(query: str, n: int = 30) -> dict:
    """Поиск YouTube: первые `n` роликов (`ytsearchN:`)."""
    d = json.loads(_run(["--flat-playlist", "-J", f"ytsearch{n}:{query}"], timeout=300))
    out = []
    for e in d.get("entries") or []:
        if e.get("id"):
            r = _entry(e)
            r["channel"] = e.get("channel") or e.get("uploader")
            r["channel_id"] = e.get("channel_id")
            out.append(r)
    return {"channel": None, "channel_id": None, "url": f"ytsearch:{query}", "entries": out}


# ---------------------------------------------------------------- сведения


def _expire(url: str) -> float | None:
    m = re.search(r"[/?&]expire[/=](\d+)", url)
    return float(m.group(1)) if m else None


def pick_hls(formats: list[dict], max_h: int = 1080) -> dict | None:
    """HLS-поток только видео: самый высокий не выше `max_h`, avc лучше
    (распаковка дешевле и ffmpeg читает его из трубы без сюрпризов)."""
    cand = [f for f in formats if f.get("protocol") == "m3u8_native"
            and (f.get("vcodec") or "none") != "none" and (f.get("height") or 0) <= max_h]
    if not cand:
        return None
    return max(cand, key=lambda f: (f.get("height") or 0,
                                    str(f.get("vcodec", "")).startswith("avc"),
                                    f.get("fps") or 0))


def pick_storyboard(formats: list[dict]) -> dict | None:
    """Раскладка с самыми крупными кадрами (sb0 — 320×180)."""
    sb = [f for f in formats if f.get("protocol") == "mhtml" and f.get("fragments")]
    return max(sb, key=lambda f: f.get("width") or 0) if sb else None


@dataclass
class VideoInfo:
    id: str
    title: str
    channel: str | None
    channel_id: str | None
    upload_date: str | None
    duration: float
    description: str
    live_status: str | None
    width: int | None
    height: int | None
    hls: dict | None
    storyboard: dict | None
    fetched: float
    seconds: float              # сколько стоил запрос сведений

    def to_meta(self) -> dict:
        return {"id": self.id, "title": self.title, "channel": self.channel,
                "channel_id": self.channel_id, "upload_date": self.upload_date,
                "duration": self.duration, "live_status": self.live_status,
                "width": self.width, "height": self.height,
                "hls": None if self.hls is None else
                {k: self.hls.get(k) for k in ("format_id", "height", "fps", "vcodec")},
                "storyboard": None if self.storyboard is None else
                {k: self.storyboard.get(k) for k in ("format_id", "width", "height", "rows",
                                                      "columns", "fps")}}


def video_info(vid: str) -> VideoInfo:
    """Сведения о ролике без скачивания (`yt-dlp -J --skip-download`)."""
    t0 = time.time()
    url = vid if vid.startswith("http") else f"https://www.youtube.com/watch?v={vid}"
    d = json.loads(_run(["-J", "--skip-download", "--no-warnings", url], timeout=120))
    fmts = d.get("formats") or []
    hls = pick_hls(fmts)
    return VideoInfo(id=d["id"], title=d.get("title") or "", channel=d.get("channel"),
                     channel_id=d.get("channel_id"), upload_date=d.get("upload_date"),
                     duration=float(d.get("duration") or 0.0),
                     description=d.get("description") or "", live_status=d.get("live_status"),
                     width=d.get("width"), height=d.get("height"), hls=hls,
                     storyboard=pick_storyboard(fmts), fetched=time.time(),
                     seconds=time.time() - t0)


# ---------------------------------------------------------------- сеть


@dataclass
class Traffic:
    """Счётчик байт и запросов — на ролик."""
    bytes: int = 0
    requests: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, n: int) -> None:
        with self.lock:
            self.bytes += n
            self.requests += 1


def _get(url: str, traffic: Traffic, limit: int | None = None, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                data = r.read() if limit is None else r.read(limit)
            traffic.add(len(data))
            return data
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))
    raise RuntimeError("недостижимо")


# ---------------------------------------------------------------- раскадровка


@dataclass(frozen=True)
class Tile:
    t: float                    # секунда ролика, к которой относится кадр раскадровки
    image: np.ndarray           # RGB (h, w, 3)


def storyboard(info: VideoInfo, traffic: Traffic, workers: int = WORKERS) -> list[Tile]:
    """Все кадры раскадровки по порядку. Кадр k листа — момент
    (начало листа + k / fps); пустые клетки последнего листа отбрасываются."""
    sb = info.storyboard
    if sb is None:
        return []
    rows, cols = int(sb["rows"]), int(sb["columns"])
    h = int(sb["height"])
    step = 1.0 / float(sb["fps"]) if sb.get("fps") else None
    frags = sb["fragments"]
    with ThreadPoolExecutor(workers) as ex:
        sheets = list(ex.map(lambda f: _get(f["url"], traffic), frags))
    out, t0 = [], 0.0
    for frag, data in zip(frags, sheets):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        per = rows * cols
        dt = step or float(frag.get("duration") or 0.0) / per
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ch, cw = img.shape[0] // rows, img.shape[1] // cols
            for k in range(per):
                t = t0 + k * dt
                if t >= info.duration:
                    break
                r, c = divmod(k, cols)
                tile = img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                if tile.shape[0] < h // 2 or float(tile.std()) < 1.0:
                    continue                    # пустая клетка в конце листа
                out.append(Tile(round(t, 2), np.ascontiguousarray(tile)))
        t0 += float(frag.get("duration") or per * (step or 0.0))
    return out


# ---------------------------------------------------------------- HLS


@dataclass(frozen=True)
class Segment:
    t: float
    dur: float
    url: str


def parse_playlist(text: str) -> list[Segment]:
    """Медиаплейлист HLS → сегменты с началом по сумме длительностей."""
    out, t, lines = [], 0.0, text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#EXTINF:"):
            d = float(line[8:].split(",")[0])
            nxt = next((x for x in lines[i + 1:] if x and not x.startswith("#")), None)
            if nxt:
                out.append(Segment(t, d, nxt.strip()))
            t += d
    return out


def decode_first(data: bytes) -> tuple[np.ndarray | None, bool]:
    """Первый кадр из начала сегмента MPEG-TS через ffmpeg: (RGB, чисто ли).
    Нечисто — ffmpeg ругался на обрыв, кадр мог выйти с серыми полосами."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", "pipe:0", "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "bmp", "-"],
                       input=data, capture_output=True, timeout=60)
    if not p.stdout:
        return None, False
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, False
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), b"error" not in p.stderr.lower()


@dataclass(frozen=True)
class Shot:
    want: float                 # просили эту секунду ролика
    t: float                    # получили кадр этой секунды (начало сегмента)
    image: np.ndarray | None
    bytes: int
    seconds: float
    error: str | None = None


class HlsStream:
    """Точечные кадры полного качества из HLS-потока ролика."""

    def __init__(self, info: VideoInfo, traffic: Traffic):
        if info.hls is None:
            raise RuntimeError("у ролика нет HLS-потока (только DASH или трансляция)")
        self.info, self.traffic = info, traffic
        self.url = info.hls["url"]
        self.expire = _expire(self.url)
        self._segs: list[Segment] | None = None

    def _check(self) -> None:
        if self.expire is not None and time.time() > self.expire - EXPIRE_SLACK:
            raise ExpiredError(f"адрес потока {self.info.id} истёк")

    def segments(self) -> list[Segment]:
        if self._segs is None:
            self._check()
            self._segs = parse_playlist(_get(self.url, self.traffic).decode("utf-8", "replace"))
        return self._segs

    def shot(self, t: float, tries: int = 3) -> Shot:
        """Кадр не позже `t`: первый (ключевой) кадр сегмента, в который попала
        секунда `t`. Ошибка сети или распаковки — Shot без картинки с причиной
        (после `tries` попыток)."""
        for i in range(tries):
            s = self._shot(t)
            if s.image is not None or i == tries - 1:
                return s
            time.sleep(2.0 * (i + 1))
        return s

    def _shot(self, t: float) -> Shot:
        t0 = time.time()
        try:
            segs = self.segments()
            if not segs:
                return Shot(t, t, None, 0, 0.0, "пустой плейлист")
            k = max(0, max((i for i, s in enumerate(segs) if s.t <= t), default=0))
            seg = segs[k]
            self._check()
            got, n, img = b"", HEAD_BYTES, None
            with urllib.request.urlopen(seg.url, timeout=TIMEOUT) as r:
                while True:
                    part = r.read(n - len(got))
                    got += part
                    img, clean = decode_first(got)
                    if (img is not None and clean) or not part or len(got) >= HEAD_MAX:
                        break
                    n = len(got) + HEAD_STEP
            self.traffic.add(len(got))
            err = None if img is not None else "кадр не распакован"
            return Shot(t, round(seg.t, 3), img, len(got), time.time() - t0, err)
        except ExpiredError:
            raise
        except Exception as exc:                              # noqa: BLE001
            return Shot(t, t, None, 0, time.time() - t0, f"{type(exc).__name__}: {exc}")

    def shots(self, times: list[float], workers: int = WORKERS) -> list[Shot]:
        self.segments()
        with ThreadPoolExecutor(min(workers, max(1, len(times)))) as ex:
            return list(ex.map(self.shot, times))
