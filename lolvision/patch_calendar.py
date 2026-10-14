"""Календарь патчей League of Legends 2026: патч ↔ дата выхода (#843).

Зачем. Индекс роликов (tools/video_index.py) отбирает ролики «текущего
патча» и «патчей этого сезона» по дате публикации, а патч в названии есть
не у всех. Даты — официальное расписание Riot (Pacific Time, патч выходит
в среду, если не сказано иное):
https://support.riotgames.com/league-of-legends/gameplay/patch-schedule-league-of-legends
(снято 2026-10-15). Riot предупреждает, что даты могут сдвинуться; сдвиг
здесь надо внести руками.

Нумерация. С 2025 года патч называется по году: 26.01 — первый патч 2026,
26.20 — двадцатый. Клиент и Data Dragon продолжают старую нумерацию:
версия 16.20.x — это патч 26.20 (на 10 меньше по первому числу;
lolvision.ddragon.patch_to_version). Сезон 2026 — патчи 26.01…26.24.

Патч по дате публикации ролика — только верхняя граница: игру могли
сыграть раньше и выложить позже (перезаливы стримов отстают на дни). Поэтому
у ответа есть источник: «title» (патч назван в названии) или «upload»
(по дате публикации).
"""

from __future__ import annotations

import datetime as dt
import re

SOURCE = ("https://support.riotgames.com/league-of-legends/gameplay/"
          "patch-schedule-league-of-legends")
SEASON = 2026
# Патч → дата выхода (PT), официальное расписание 2026 (см. докстринг).
SCHEDULE: dict[str, dt.date] = {
    "26.01": dt.date(2026, 1, 8), "26.02": dt.date(2026, 1, 22),
    "26.03": dt.date(2026, 2, 4), "26.04": dt.date(2026, 2, 19),
    "26.05": dt.date(2026, 3, 4), "26.06": dt.date(2026, 3, 18),
    "26.07": dt.date(2026, 4, 1), "26.08": dt.date(2026, 4, 15),
    "26.09": dt.date(2026, 4, 29), "26.10": dt.date(2026, 5, 13),
    "26.11": dt.date(2026, 5, 28), "26.12": dt.date(2026, 6, 10),
    "26.13": dt.date(2026, 6, 24), "26.14": dt.date(2026, 7, 15),
    "26.15": dt.date(2026, 7, 29), "26.16": dt.date(2026, 8, 12),
    "26.17": dt.date(2026, 8, 26), "26.18": dt.date(2026, 9, 10),
    "26.19": dt.date(2026, 9, 23), "26.20": dt.date(2026, 10, 7),
    "26.21": dt.date(2026, 10, 21), "26.22": dt.date(2026, 11, 4),
    "26.23": dt.date(2026, 11, 18), "26.24": dt.date(2026, 12, 9),
}
_ORDER = sorted(SCHEDULE, key=SCHEDULE.get)
# «Patch 26.19», «26.19», «v26.19», «패치 26.19», «16.19» (клиент) — не дата и не счёт.
_PATCH_RE = re.compile(r"(?<![\d.:/])(?:patch|패치|版本|v)?\s*(1[56]|2[56])\.(\d{1,2})(?![\d.:/])",
                       re.IGNORECASE)


def norm(patch: str) -> str | None:
    """«26.1» / «16.1» / «26.01» → «26.01»; не патч — None."""
    m = re.fullmatch(r"\s*(\d{2})\.(\d{1,2})(?:\.\d+)?\s*", patch)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2))
    if major < 20:                       # версия клиента 16.x → патч 26.x
        major += 10
    return f"{major}.{minor:02d}"


def release(patch: str) -> dt.date | None:
    """Дата выхода патча (PT) или None, если его нет в календаре."""
    p = norm(patch)
    return SCHEDULE.get(p) if p else None


def at(day: dt.date) -> str | None:
    """Патч, действовавший в этот день (последний вышедший не позже), или
    None — день раньше 26.01 (прошлый сезон, календаря 2025 здесь нет)."""
    cur = None
    for p in _ORDER:
        if SCHEDULE[p] <= day:
            cur = p
        else:
            break
    return cur


def current(today: dt.date | None = None) -> str | None:
    return at(today or dt.date.today())


def season_start(year: int = SEASON) -> dt.date:
    return SCHEDULE[f"{year % 100}.01"]


def from_text(text: str) -> str | None:
    """Патч, названный в тексте (название или описание ролика), если он есть в
    календаре: «Patch 26.19» → «26.19». Счёт «10.5» или время «12:30» — не патч:
    первое число — 15/16 (клиент) или 25/26 (год)."""
    for m in _PATCH_RE.finditer(text or ""):
        p = norm(f"{m.group(1)}.{m.group(2)}")
        if p in SCHEDULE:
            return p
    return None


def parse_date(s: str | None) -> dt.date | None:
    """«20260928» (yt-dlp upload_date) или «2026-09-28» → дата."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def since(spec: str, today: dt.date | None = None) -> dt.date:
    """Нижняя граница отбора: «patch» — начало текущего патча, «season» —
    начало сезона, «26.18» — начало этого патча, или дата «2026-09-01»."""
    today = today or dt.date.today()
    if spec == "patch":
        p = current(today)
        return SCHEDULE[p] if p else season_start()
    if spec == "season":
        return season_start(today.year if f"{today.year % 100}.01" in SCHEDULE else SEASON)
    r = release(spec)
    if r is not None:
        return r
    d = parse_date(spec)
    if d is None:
        raise ValueError(f"не патч и не дата: {spec!r} (patch / season / 26.18 / 2026-09-01)")
    return d


def patch_of(title: str, description: str, upload: dt.date | None) -> tuple[str | None, str]:
    """Патч ролика и откуда он: из названия, из описания, по дате публикации
    (верхняя граница) или неизвестен."""
    p = from_text(title)
    if p:
        return p, "title"
    p = from_text(description)
    if p:
        return p, "description"
    if upload is not None:
        p = at(upload)
        if p:
            return p, "upload"
    return None, "unknown"
