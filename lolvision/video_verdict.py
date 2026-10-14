"""Годится ли ролик для набора: разбор раскадровки и точечных кадров (#843).

Только данные и правила, без сети и моделей: сеть — lolstream/youtube_preview.py,
модели зовёт tools/video_index.py и кладёт сюда их ответы (`FrameFacts`).
Поэтому всё здесь проверяется тестами на сохранённых ответах.

Шаги:

1. **Лента по раскадровке** (`timeline`): тип сцены каждого кадра
   раскадровки (~10 с) → отрезки игры. Игра — классы game / tab / shop
   (Tab и магазин — игра под окном). Отрезок рвётся на загрузке, клиенте или
   не-LoL дольше BREAK_S; короткие провалы (смерть с затемнением, сцена-
   заставка стримера) внутри отрезка терпятся.
2. **Моменты для кадров** (`pick_times`): по 1–10 кадров на отрезок не
   короче MIN_SEGMENT_S, сначала ранние (роль видна в 2–8 мин), потом по
   всей игре.
3. **Кадры** (`FrameFacts`) — тип сцены, чтение раскладки HUD
   (spectator / player / ничего — lolstream.live_champions.read_kind),
   часы матча с полосы HUD вида игрока (hud_stats), миникарта (найдена ли,
   масштаб, совпадение), камера на миникарте → зона карты (rift.zone).
4. **Итог** (`verdict`): вид, свой чемпион (по иконкам умений на виде
   игрока — lolvision/hud_champion.py; у наблюдателя — только заявка),
   роль, полнота игры, видимость миникарты и HUD, годность и причины,
   уверенность.

Уверенность — не вероятность, а грубая шкала 0…1 из согласия признаков
(доля кадров за вид, отрыв чемпиона, согласие с заявкой); проверка глазами —
docs/video-index.md.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from lolvision import hud_champion as hc

IN_GAME = ("game", "tab", "shop")
BREAK_S = 45.0              # не-игра дольше — отрезок игры кончился
MIN_SEGMENT_S = 240.0       # короче — не игра, а заставка/клип: кадры не берутся
FULL_GAME_S = 14 * 60.0     # полная игра не короче (сдача с 15-й минуты, минус шаг)
START_SLACK_S = 90.0        # начало матча по часам не позже начала отрезка + это
EARLY = (100.0, 190.0, 300.0, 430.0)           # с от начала отрезка: роль
LATE_SHARE = (0.35, 0.55, 0.75, 0.92)          # доли отрезка: остальное
MAX_FRAMES = 10
CLOCK_TOL_S = 8.0           # расхождение часов и ролика меньше — не вырез (кадр HLS — начало
                            # сегмента)
HEAVY_CUT = 0.4             # вырезано больше этой доли матча между кадрами — нарезка
ROLE_WINDOW = (80.0, 600.0)  # с матча: где камера показывает линию
ROLE_SHARE = 0.6
MINIMAP_SHARE = 0.5         # миникарта найдена хотя бы на такой доле игровых кадров


# ---------------------------------------------------------------- лента


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    tiles: int
    game_share: float        # доля игровых кадров раскадровки внутри

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 1), "end": round(self.end, 1), "tiles": self.tiles,
                "game_share": round(self.game_share, 3)}


def timeline(tiles: Sequence[tuple[float, str]], duration: float,
             break_s: float = BREAK_S) -> list[Segment]:
    """(t, класс сцены) кадров раскадровки по порядку → отрезки игры.
    Конец отрезка — момент последнего игрового кадра плюс шаг раскадровки
    (не дальше длины ролика)."""
    if not tiles:
        return []
    step = (tiles[-1][0] - tiles[0][0]) / max(1, len(tiles) - 1) if len(tiles) > 1 else 10.0
    out: list[Segment] = []
    start = last = None
    n = g = gap = 0                 # gap — не-игра после последнего игрового кадра
    for t, cls in tiles:
        if cls in IN_GAME:
            if start is not None and t - last > break_s + step:
                out.append(Segment(start, min(duration, last + step), n, g / max(1, n)))
                start = None
            if start is None:
                start, n, g = t, 0, 0
            n, g, gap = n + gap + 1, g + 1, 0
            last = t
        elif start is not None:
            if t - last > break_s:
                out.append(Segment(start, min(duration, last + step), n, g / max(1, n)))
                start, gap = None, 0
            else:
                gap += 1
    if start is not None:
        out.append(Segment(start, min(duration, last + step), n, g / max(1, n)))
    return out


def game_share(tiles: Sequence[tuple[float, str]]) -> float:
    return sum(c in IN_GAME for _, c in tiles) / len(tiles) if tiles else 0.0


def pick_times(segments: Sequence[Segment], max_frames: int = MAX_FRAMES,
               min_len: float = MIN_SEGMENT_S) -> list[tuple[int, float]]:
    """(номер отрезка, секунда ролика) для точечных кадров. Ранние моменты
    (EARLY) — у каждого длинного отрезка, остальное — доли LATE_SHARE, пока
    не наберётся `max_frames`; длинные отрезки первыми."""
    segs = [(i, s) for i, s in enumerate(segments) if s.length >= min_len]
    if not segs:
        return []
    segs.sort(key=lambda x: -x[1].length)
    plan: list[list[float]] = []
    for _, s in segs:
        ts = [s.start + e for e in EARLY if s.start + e < s.end - 20]
        ts += [s.start + f * s.length for f in LATE_SHARE]
        plan.append(ts)
    out: list[tuple[int, float]] = []
    k = 0
    while len(out) < max_frames and any(k < len(p) for p in plan):
        for (i, _), p in zip(segs, plan):
            if k < len(p) and len(out) < max_frames:
                out.append((i, round(p[k], 1)))
        k += 1
    return sorted(out, key=lambda x: x[1])


# ---------------------------------------------------------------- кадры


@dataclass
class FrameFacts:
    """Что прочитали модели на одном точечном кадре."""
    t: float                         # секунда ролика (начало сегмента HLS)
    segment: int
    scene: str | None = None
    scene_p: float | None = None
    view_read: str | None = None     # spectator / player / None (признаков нет)
    panel_score: float | None = None  # совпадение панели умений своего HUD
    clock: int | None = None         # секунда матча по полосе HUD (вид игрока)
    minimap_found: bool = False
    minimap_score: float | None = None
    minimap_scale: float | None = None
    camera: tuple[float, float] | None = None   # центр рамки камеры, доли рамки миникарты
    zone: str | None = None
    champion_top: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("scene_p", "panel_score", "minimap_score", "minimap_scale"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        if d["camera"] is not None:
            d["camera"] = [round(float(c), 4) for c in d["camera"]]
        d["champion_top"] = [[c, round(float(s), 3)] for c, s in self.champion_top]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FrameFacts:
        d = dict(d)
        if d.get("camera") is not None:
            d["camera"] = tuple(d["camera"])
        d["champion_top"] = [tuple(x) for x in d.get("champion_top") or []]
        return cls(**d)


# ---------------------------------------------------------------- итог


def decide_view(frames: Sequence[FrameFacts]) -> tuple[str, float, dict]:
    """Вид по голосам кадров: только положительные чтения голосуют.
    Ответ: (вид, доля голосов за него среди голосовавших, счёт)."""
    votes = Counter(f.view_read for f in frames if f.view_read)
    total = sum(votes.values())
    if total == 0:
        return "unknown", 0.0, dict(votes)
    view, n = votes.most_common(1)[0]
    share = n / total
    if n < 2 or share < 0.7:
        return "unknown", share, dict(votes)
    return view, share, dict(votes)


def clock_games(frames: Sequence[FrameFacts]) -> list[dict]:
    """Прочтения часов → игры: смещение «секунда ролика − секунда матча»
    одно у всех кадров одной игры без склеек. Группы по CLOCK_TOL_S."""
    pts = sorted((f.t - f.clock, f.t, f.clock) for f in frames if f.clock is not None)
    groups: list[list[tuple[float, float, int]]] = []
    for p in pts:
        if groups and abs(p[0] - groups[-1][-1][0]) <= CLOCK_TOL_S:
            groups[-1].append(p)
        else:
            groups.append([p])
    out = []
    for g in groups:
        offs = sorted(x[0] for x in g)
        out.append({"start": round(offs[len(offs) // 2], 1), "reads": len(g),
                    "first_clock": min(x[2] for x in g), "last_clock": max(x[2] for x in g)})
    return out


def coverage(frames: Sequence[FrameFacts], tol: float = CLOCK_TOL_S) -> dict | None:
    """Часы кадров одной игры по порядку → начало матча в ролике, сколько
    секунд матча вырезано (`cut_s`) и в скольких местах. Прочтение, у
    которого матч отстал от ролика больше чем на `tol` (часы «назад»), —
    ошибка чтения или пауза: отбрасывается (`dropped`). Меньше двух
    прочтений — None."""
    reads = sorted((f.t, f.clock) for f in frames if f.clock is not None)
    if not reads:
        return None
    # Самая длинная согласная цепочка: между соседними прочтениями матч
    # прошёл не меньше, чем ролик (минус tol). Чужое прочтение — ошибка
    # чтения часов или пауза игры — в цепочку не входит (`dropped`); при
    # равной длине — цепочка с меньшим вырезом.
    n = len(reads)
    best = [(1, 0.0, -1)] * n                # (длина, вырез, предыдущий)
    for j in range(n):
        tj, cj = reads[j]
        for i in range(j):
            ti, ci = reads[i]
            if cj - ci >= (tj - ti) - tol:
                ln, cut_i, _ = best[i]
                cand = (ln + 1, cut_i + max(0.0, (cj - ci) - (tj - ti)), i)
                if (cand[0], -cand[1]) > (best[j][0], -best[j][1]):
                    best[j] = cand
    end = max(range(n), key=lambda k: (best[k][0], -best[k][1]))
    kept, k = [], end
    while k >= 0:
        kept.append(reads[k])
        k = best[k][2]
    kept.reverse()
    dropped = n - len(kept)
    cuts = cut = 0.0
    for (pt, pc), (t, c) in zip(kept, kept[1:]):
        gap = (c - pc) - (t - pt)
        if gap > tol:
            cuts += 1
            cut += gap
    return {"start": round(kept[0][0] - kept[0][1], 1), "first_t": kept[0][0],
            "first_clock": kept[0][1], "last_t": kept[-1][0], "last_clock": kept[-1][1],
            "span": kept[-1][1] - kept[0][1], "cut_s": round(cut, 1), "cuts": int(cuts),
            "reads": len(kept), "dropped": dropped}


GAME_RESET_S = 60.0         # часы назад больше чем на столько — следующая игра


def split_games(frames: Sequence[FrameFacts], reset: float = GAME_RESET_S
                ) -> list[list[FrameFacts]]:
    """Кадры одного отрезка → игры по сбросу часов. Кадр без часов идёт в
    игру предыдущего кадра с часами (до первого — в первую). Одиночное
    прочтение «назад» (ошибка чтения), за которым часы снова прежние, игру
    не начинает."""
    fs = sorted(frames, key=lambda f: f.t)
    clocked = [f for f in fs if f.clock is not None]
    cut_at: list[float] = []
    for i in range(1, len(clocked)):
        prev, cur = clocked[i - 1], clocked[i]
        if cur.clock < prev.clock - reset:
            nxt = clocked[i + 1] if i + 1 < len(clocked) else None
            # Следующее прочтение согласно с новым (матч прошёл не меньше
            # ролика) и не согласно с прежним — новая игра; согласно с
            # прежним — это прочтение было ошибкой.
            with_prev = nxt is not None and \
                nxt.clock - prev.clock >= (nxt.t - prev.t) - CLOCK_TOL_S
            if not with_prev:
                cut_at.append(cur.t)
    out: list[list[FrameFacts]] = [[]]
    for f in fs:
        if cut_at and f.t >= cut_at[0]:
            cut_at.pop(0)
            if out[-1]:
                out.append([])
        out[-1].append(f)
    return [p for p in out if p]


def champion_from_tops(frames: Sequence[FrameFacts], min_score: float = hc.MIN_SCORE,
                       margin: float = hc.MARGIN) -> dict:
    """Свой чемпион по лучшим оценкам кадров (`champion_top`): у чемпиона,
    которого среди лучших кадра нет, оценка — худшая из лучших этого кадра."""
    tops = [dict(f.champion_top) for f in frames if f.champion_top]
    if not tops:
        return {"champion": None, "score": None, "margin": None}
    names = sorted({c for t in tops for c in t})
    m = {c: sum(t.get(c, min(t.values())) for t in tops) / len(tops) for c in names}
    order = sorted(m, key=m.get, reverse=True)
    a = order[0]
    gap = m[a] - (m[order[1]] if len(order) > 1 else 0.0)
    ok = m[a] >= min_score and gap >= margin
    return {"champion": a if ok else None, "score": round(m[a], 3), "margin": round(gap, 3)}


def decide_role(frames: Sequence[FrameFacts], match_start: float | None,
                seg_start: float) -> tuple[str | None, str, dict]:
    """Роль по зоне камеры в начале игры: топ / мид / бот — линия, лес и река —
    лесник. Время матча — по часам, а без них — от начала отрезка (у
    наблюдателя часов нет). Бот не делится на стрелка и поддержку."""
    base = match_start if match_start is not None else seg_start
    zones = Counter(f.zone for f in frames
                    if f.zone and ROLE_WINDOW[0] <= f.t - base <= ROLE_WINDOW[1])
    lanes = {k: zones.get(k, 0) for k in ("top", "mid", "bot")}
    wild = zones.get("jungle", 0) + zones.get("river", 0)
    total = sum(lanes.values()) + wild
    if total < 2:
        return None, f"камера в начале игры прочитана на {total} кадрах", dict(zones)
    best = max(lanes, key=lanes.get)
    if wild / total >= ROLE_SHARE:
        return "jungle", f"камера в лесу/реке {wild} из {total}", dict(zones)
    if lanes[best] / total >= ROLE_SHARE:
        return best, f"камера на линии {best} {lanes[best]} из {total}", dict(zones)
    return None, f"зоны камеры спорные: {dict(zones)}", dict(zones)


@dataclass
class Verdict:
    usable: bool
    reasons: list[str]
    confidence: float
    view: str
    view_share: float
    champion: str | None
    champion_source: str
    role: str | None
    role_source: str
    full_game: bool
    cut_s: float | None          # секунд матча нет в ролике (по часам), None — часов нет
    cuts: int | None
    games: list[dict]
    minimap_share: float | None
    minimap_scale: float | None
    hud_share: float | None
    notes: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("confidence", "view_share", "minimap_share", "minimap_scale", "hud_share"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 3)
        return d


def verdict(segments: Sequence[Segment], frames: Sequence[FrameFacts], claims: dict,
            champ: dict | None, tiles_game_share: float,
            support_like: bool | None = None) -> Verdict:
    """Итог по ролику. `claims` — lolvision.video_meta.claims; `champ` —
    hud_champion.Decision.to_dict() по кадрам вида игрока (или None);
    `support_like` — у заявленного/найденного чемпиона тег Support без
    Marksman (Data Dragon): им делится бот на стрелка и поддержку."""
    reasons: list[str] = []
    notes: list[str] = []
    game_frames = [f for f in frames if f.scene in IN_GAME]
    long_segs = [s for s in segments if s.length >= MIN_SEGMENT_S]

    view, view_share, votes = decide_view(game_frames)
    if votes:
        notes.append(f"вид: голоса {votes}")

    # Игры главного отрезка. Архивы клеят 2–3 игры подряд без загрузки между
    # ними — раскадровка видит один отрезок, а часы HUD сбрасываются
    # (BA7OqRG-QVE: 1546 с на 1497 с ролика, 756 с на 2346 с). Каждая игра:
    # видна от начала (матч начался не раньше чем за START_SLACK_S до начала
    # её участка ролика) и до конца (длина матча не меньше FULL_GAME_S).
    # Вырезы внутри (смерти, ожидание — так режут архивы стримов) игру не
    # портят: это `cut_s`; сильно порезанная (> HEAVY_CUT) — нарезка.
    games_clock = clock_games(game_frames)
    main = max(long_segs, key=lambda s: s.length) if long_segs else None
    main_i = segments.index(main) if main is not None else None
    match_start = None
    full = False
    cov = None
    game_rows: list[dict] = []
    main_frames = [f for f in game_frames if f.segment == main_i]
    parts = split_games(main_frames) if main is not None else []
    covs = [coverage(p) for p in parts]
    for k, (part, c) in enumerate(zip(parts, covs)):
        if c is None:
            continue
        lower = main.start if k == 0 else max(f.t for f in parts[k - 1])
        nxt = next((x for x in covs[k + 1:] if x is not None), None)
        end = main.end if nxt is None else nxt["start"]
        length = c["last_clock"] + (end - c["last_t"])
        heavy = c["span"] > 0 and c["cut_s"] / c["span"] > HEAVY_CUT
        why = []
        if c["start"] < lower - START_SLACK_S:
            why.append(f"начало игры вырезано: игра видна с "
                       f"{(lower - c['start']) / 60:.1f} мин матча")
        if length < FULL_GAME_S:
            why.append(f"игра короче {FULL_GAME_S / 60:.0f} мин ({length / 60:.1f} мин)")
        if heavy:
            why.append(f"нарезка: вырезано {c['cut_s']:.0f} с из {c['span']:.0f} с "
                       "матча между кадрами")
        cd = champion_from_tops(part)
        game_rows.append({"start": c["start"], "end": round(end, 1),
                          "length": round(length, 1), "first_clock": c["first_clock"],
                          "last_clock": c["last_clock"], "reads": c["reads"],
                          "dropped": c["dropped"], "cut_s": c["cut_s"], "cuts": c["cuts"],
                          "full": not why, "why": why, "champion": cd["champion"],
                          "champion_score": cd["score"], "champion_margin": cd["margin"],
                          "frames": len(part), "_cov": c, "_part": part})
    if game_rows:
        g = max(game_rows, key=lambda x: (x["full"], x["frames"], x["length"]))
        cov, full, match_start = g["_cov"], g["full"], g["start"]
        reasons.extend(g["why"])
        notes.append(f"матч начался на {match_start:.0f} с ролика "
                     f"(игра видна с {main.start:.0f} с); вырезано {cov['cut_s']:.0f} с "
                     f"матча в {cov['cuts']} местах")
        if cov["dropped"]:
            notes.append(f"прочтений часов не к месту: {cov['dropped']} (ошибка чтения или пауза)")
        if len(game_rows) > 1:
            notes.append(f"в ролике {len(game_rows)} игр подряд (по сбросу часов); "
                         f"полных {sum(x['full'] for x in game_rows)}; разобрана "
                         f"{game_rows.index(g) + 1}-я")
        role_frames = g["_part"]
        if len(game_rows) > 1 or champ is None:
            champ = {"champion": g["champion"], "margin": g["champion_margin"],
                     "score": g["champion_score"],
                     "reason": None if g["champion"] else "по кадрам этой игры не решён"}
    elif main is not None:
        role_frames = main_frames
        full = main.length >= FULL_GAME_S
        notes.append("часы HUD не прочитаны — полнота только по раскадровке")
        if not full:
            reasons.append(f"самый длинный отрезок игры {main.length / 60:.1f} мин "
                           f"< {FULL_GAME_S / 60:.0f}")
    else:
        role_frames = []
    # «Varus · Renekton · Jayce» в названии ролика из трёх игр — по порядку.
    named = claims.get("champions") or []
    if len(game_rows) > 1 and len(named) == len(game_rows):
        for x, c in zip(game_rows, named):
            x["champion_claim"] = c
    games = [{k: v for k, v in x.items() if not k.startswith("_")} for x in game_rows]
    games = games or games_clock
    if tiles_game_share < 0.05 or main is None:
        reasons.append(f"игры почти нет: {tiles_game_share:.0%} кадров раскадровки"
                       if tiles_game_share < 0.05 else
                       f"нет отрезка игры длиннее {MIN_SEGMENT_S / 60:.0f} мин")
    n_full = sum(1 for s in long_segs if s.length >= FULL_GAME_S)
    if n_full > 1:
        notes.append(f"в ролике {n_full} отрезков игры ≥ {FULL_GAME_S / 60:.0f} мин "
                     "(несколько игр); разобран самый длинный")

    # Свой чемпион.
    claim = claims.get("champion")
    champion, csrc = None, "unknown"
    hud_c = (champ or {}).get("champion")
    if view == "player" and hud_c:
        champion, csrc = hud_c, "hud"
        if claim and claim != hud_c:
            notes.append(f"HUD: {hud_c}, заявка: {claim}")
            csrc = "hud≠claim"
    elif claim:
        champion, csrc = claim, "claim"
        if view == "player" and champ and champ.get("reason"):
            notes.append(f"чемпион по HUD не решён: {champ['reason']}")
    if champion is None:
        reasons.append("свой чемпион неизвестен")

    # Роль.
    role, rnote, _ = decide_role(role_frames, match_start, main.start if main else 0.0)
    rsrc = "camera" if role else "unknown"
    rclaim = claims.get("role")
    if role is None and rclaim:
        role, rsrc = rclaim, "claim"
    elif role == "bot":
        if rclaim in ("bot", "support"):
            role, rsrc = rclaim, "camera+claim"
        elif support_like is not None:
            role = "support" if support_like else "bot"
            rsrc = "camera+tags"
    elif role and rclaim and rclaim != role and not (role == "bot" and rclaim == "support"):
        notes.append(f"роль по камере {role}, заявка {rclaim}")
        rsrc = "camera≠claim"
    notes.append(rnote)

    # Миникарта и HUD.
    mm_share = (sum(f.minimap_found for f in game_frames) / len(game_frames)
                if game_frames else None)
    scales = sorted(f.minimap_scale for f in game_frames
                    if f.minimap_found and f.minimap_scale)
    mm_scale = scales[len(scales) // 2] if scales else None
    if mm_share is not None and mm_share < MINIMAP_SHARE:
        notes.append(f"миникарта найдена на {mm_share:.0%} игровых кадров: "
                     "закрыта вебкой/оверлеем или не на месте")
    hud_share = None
    if view == "player" and game_frames:
        hud_share = sum((f.panel_score or 0) >= 0.35 for f in game_frames) / len(game_frames)
        if hud_share < 0.5:
            notes.append(f"панель умений видна на {hud_share:.0%} игровых кадров")

    if view == "unknown":
        reasons.append("вид (игрока / наблюдателя) не решён")
    if claims.get("not_full"):
        reasons.append("название: нарезка / гайд / шорт")
    if not full and main is not None and not any(r.startswith(("нарезка", "начало", "игра короче",
                                                               "самый длинный")) for r in reasons):
        reasons.append("полная игра не подтверждена")

    usable = full and view in ("player", "spectator") and champion is not None \
        and not claims.get("not_full")

    # Уверенность: согласие признаков.
    parts = [view_share if view != "unknown" else 0.0]
    if csrc == "hud":
        parts.append(min(1.0, 0.5 + 5 * float((champ or {}).get("margin") or 0)))
    elif csrc == "claim":
        parts.append(0.6)
    elif csrc == "hud≠claim":
        parts.append(0.4)
    else:
        parts.append(0.0)
    if full and cov is not None:
        parts.append(1.0 - min(0.5, cov["cut_s"] / max(1.0, cov["span"])))
    else:
        parts.append(0.7 if full else 0.3)
    conf = float(sum(parts) / len(parts))
    return Verdict(usable=usable, reasons=reasons, confidence=conf, view=view,
                   view_share=view_share, champion=champion, champion_source=csrc, role=role,
                   role_source=rsrc, full_game=full,
                   cut_s=None if cov is None else cov["cut_s"],
                   cuts=None if cov is None else cov["cuts"], games=games, minimap_share=mm_share,
                   minimap_scale=mm_scale, hud_share=hud_share, notes=notes)
