"""Ролик как поток кадров: сведения о ролике и последовательная распаковка.

Общее для сборщиков датасета (build_coords.py, build_not_game.py) и
live-режима (live.py): локальный файл или ссылка на YouTube превращается в
поток кадров (H, W, 3) uint8 через ffmpeg, на диск видео не пишется.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np



def probe_url(url: str) -> tuple[int, int, float, str, dict]:
    """Размеры и длительность ролика без его скачивания."""
    out = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]", "--skip-download",
         "--dump-json", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"не удалось получить сведения о ролике: {url}")
    d = json.loads(out.stdout)
    return int(d["width"]), int(d["height"]), float(d["duration"]), d["id"], d


def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return int(s["width"]), int(s["height"]), float(d["format"]["duration"])


@lru_cache(maxsize=1)
def have_nvdec() -> bool:
    """Умеет ли ffmpeg распаковывать видео на видеокарте."""
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-hwaccels"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in out.split()


def frames(video, w: int, h: int, fps: float, from_url: bool = False,
           decode: str = "auto", skip: float = 0.0):
    """Последовательный поток кадров через ffmpeg; память не растёт.

    Из YouTube читаем через трубу от yt-dlp, а не отдаём ffmpeg прямой
    медиа-адрес: YouTube душит прямое обращение к нему, и ffmpeg на таком
    адресе просто висит (замерено: 7 минут без единого байта, при том что
    yt-dlp по той же ссылке идёт на 18 МиБ/с — он снимает ограничение,
    решая проверочный скрипт). На диск при этом не пишется ничего.

    Распаковка — узкое место всей сборки, а не модель и не сеть. Замер на
    пяти минутах матча (AV1 1080p60, одинаковый выход 1920×1080): на
    половине ядер процессора 53 с, на видеокарте 12.5 с. Поэтому по
    умолчанию берём NVDEC, а на машине без него откатываемся на процессор.
    """
    # Без ограничения ffmpeg забирает все ядра на распаковку 1080p60,
    # и машина перестаёт отзываться.
    threads = max(2, (os.cpu_count() or 8) // 2)
    gpu = decode == "gpu" or (decode == "auto" and have_nvdec())
    puller = None
    if from_url:
        puller = subprocess.Popen(
            [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]", "-o", "-",
             "--quiet", "--no-warnings", str(video)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 22)
        src, stdin = "pipe:0", puller.stdout
    else:
        src, stdin = str(video), None
    # Кадры остаются в видеопамяти до прореживания, вниз едут только
    # отобранные: выгружать 60 кадров в секунду, чтобы взять один, — это
    # и есть та разница в четыре раза.
    head = (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if gpu
            else ["-threads", str(threads)])
    vf = f"fps={fps},hwdownload,format=nv12" if gpu else f"fps={fps}"
    # Перемотка только «пропуском»: по трубе назад не прыгнуть, а прямой
    # медиа-адрес YouTube душит (проверено: ffmpeg с -ss по нему за 100 с не
    # получил ни байта). Пропуск стоит распаковки, то есть около 25 секунд
    # на каждые десять минут ролика.
    seek = ["-ss", f"{skip:.3f}"] if skip > 0 else []
    cmd = ["ffmpeg", "-v", "error", *head,
           "-i", src, *seek, "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=stdin, bufsize=n * 2)
    if puller is not None:
        puller.stdout.close()          # трубой дальше владеет ffmpeg
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        # Сначала гасим ffmpeg, потом закрываем трубу: в обратном порядке он
        # успевает написать в закрытый канал и завалить вывод руганью про
        # broken pipe, хотя всё в порядке — просто читатель ушёл.
        if p.poll() is None:
            p.terminate()
        p.stdout.close()
        p.wait()
        if puller is not None:
            puller.terminate()
            puller.wait()
