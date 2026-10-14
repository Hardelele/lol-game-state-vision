"""Ущелье призывателей: место на карте и его смысл (#823).

Зачем. Сервис отдавал место чемпиона в долях вырезанной рамки миникарты
(вместе с тёмной окантовкой слева и сверху), а coach считал те же числа
долями поля карты `x/14820`. Между ними сдвиг до 0.05 карты (~750 ед.) —
«их верхний лес» вместо топа, «под их башней» в 1200 ед. от башни (разбор
игры на Экко 2026-10-14, coach docs/reviews/2026-10-14-ekko). Теперь
окантовка учитывается здесь, в одном месте: доли рамки → мировые
координаты игры → доли поля и смысл места (линия / лес / река / база,
половина, ближайшая башня, яма). Coach геометрии не держит.

Системы координат:

* **доли рамки** `(u, v)` — как у иконок миникарты, рамки камеры, MatchState
  и `position` снимка: доли прямоугольника, которым вырезается миникарта
  (minimap_locate.STD_BOX и найденный под масштаб), вместе с окантовкой;
* **мировые** `(x, y)` — единицы игры, ось y вверх, синяя база (ORDER)
  внизу слева: поле 0…MAP_UNITS по замеру Replay API (docs/replay-api.md);
* **доли поля** `(u, v)` — `x / MAP_UNITS`, `1 − y / MAP_UNITS`: u вправо,
  v вниз, как на миникарте, но без окантовки.

Перевод «доли рамки ↔ мировые» — замер Replay API на миникарте 1920×1080
стандартного размера (поле 0…14 800 по x — экранные x 1653.96…1908.42; по y
верх поля 812.06, низ 1065.97 — по точке земли, не по рамке камеры) и
прямоугольник вырезки STD_BOX в пикселях так, как его режет box_px. Locator
находит под масштаб миникарты прямоугольник, в котором та же точка карты
ложится в ту же клетку сетки 256, поэтому формула от масштаба не зависит:

    u_рамки = 0.0343 + 0.9749 · u_поля,   v_рамки = 0.0497 + 0.9654 · v_поля

Проверка (разбор Экко, ×1.205): 22 башни на кадре t=30 ложатся на иконки
башен — среднее (−0.004, −0.004), медиана |d| 0.011 рамки; на t=865 —
медиана 0.014.

Смысл места — грубое деление карты, а не чтение названий: базы — углы за
башнями ингибиторов, линии — полосы у краёв карты с загибом углов, мид — полоса
вдоль диагонали между базами, река — полоса вдоль другой диагонали, остальное
— лес. CHAOS-сторона — отражение ORDER-стороны через центр симметрии
(CENTER, по 11 парам башен). Модуль без torch: его берут и лёгкие скрипты.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from lolvision.geometry import MAP_UNITS

TEAMS = ("ORDER", "CHAOS")
# Поле карты на экране 1920×1080 (пиксели), замер Replay API (docs/replay-api.md).
MAP_X0, MAP_X1 = 1653.96, 1908.42
MAP_Y_TOP, MAP_Y_BOTTOM = 812.06, 1065.97
REF_W, REF_H = 1920, 1080
# Прямоугольник вырезки стандартной миникарты (minimap_locate.STD_BOX) в
# пикселях 1920×1080, округление как у box_px: доли рамки отсчитываются от него.
STD_BOX = (0.857, 0.74, 0.993, 0.984)
CROP_PX = tuple(int(f * s) for f, s in zip(STD_BOX, (REF_W, REF_H, REF_W, REF_H)))

# Центр симметрии карты: середина пар башен ORDER ↔ CHAOS (одиннадцать пар,
# разброс ±40 ед.). Через него CHAOS-сторона — отражение ORDER-стороны.
CENTER = (7405.0, 7450.0)

# Фонтаны (место возрождения), мировые. CHAOS — наблюдение: своя иконка Экко
# после шести возрождений (разбор 2026-10-14, 12 кадров, t = 225–1331) стоит в
# (0.9775, 0.0903) рамки, разброс ±0.001. Прежняя константа (0.965, 0.035)
# рамки расходилась с ним на 0.055 по v. ORDER — отражение через CENTER; на
# кадрах своей игры за ORDER не проверен.
_CHAOS_SEEN = (0.9775, 0.0903)

# Башни: (команда, линия, номер P из имени API) → мировые (x, y). Имена API
# Turret_T<Order|Chaos>_L<0 бот|1 мид|2 топ>_P<3 внешняя|2 внутренняя|1 у
# ингибитора|4, 5 нексуса> (журналы своей игры 2026-10-12). Башни нексуса в
# имени — L1, но это не мид: линия здесь — `nexus`. Координаты — общепринятые
# мировые; на кадре t=30 игры на Экко все 22 ложатся на иконки башен.
TURRETS = {
    ("ORDER", "top", 3): (981, 10441), ("ORDER", "top", 2): (1512, 6699),
    ("ORDER", "top", 1): (1169, 4287),
    ("ORDER", "mid", 3): (5846, 6396), ("ORDER", "mid", 2): (5048, 4812),
    ("ORDER", "mid", 1): (3651, 3696),
    ("ORDER", "nexus", 4): (1748, 2270), ("ORDER", "nexus", 5): (2177, 1807),
    ("ORDER", "bot", 3): (10504, 1029), ("ORDER", "bot", 2): (6919, 1483),
    ("ORDER", "bot", 1): (4281, 1253),
    ("CHAOS", "top", 3): (4318, 13875), ("CHAOS", "top", 2): (7943, 13411),
    ("CHAOS", "top", 1): (10481, 13650),
    ("CHAOS", "mid", 3): (8955, 8510), ("CHAOS", "mid", 2): (9767, 10113),
    ("CHAOS", "mid", 1): (11134, 11207),
    ("CHAOS", "nexus", 4): (12611, 13084), ("CHAOS", "nexus", 5): (13052, 12612),
    ("CHAOS", "bot", 3): (13866, 4505), ("CHAOS", "bot", 2): (13327, 8226),
    ("CHAOS", "bot", 1): (13624, 10572),
}
# Ямы эпических монстров (центр ямы, общепринятые мировые): Барон / Герольд /
# личинки — одна яма сверху, дракон — снизу.
PITS = {"dragon": (9866, 4414), "baron": (5007, 10471)}

# Пороги, мировые единицы.
# Башня бьёт на 750 ед. от края; плюс радиусы башни и чемпиона — ~900 от центра.
AT_TURRET = 900.0
AT_PIT = 900.0             # «у дракона» / «у Барона»
# Базы — угол за башнями ингибиторов: x, y < BASE_EDGE и x + y < BASE_SUM
# (башни ингибиторов уже линия, сами ингибиторы — база).
BASE_EDGE, BASE_SUM = 4200.0, 7000.0
# Топ — полоса у левого края (x < LANE_EDGE) и у верхнего (y > MAP − …), угол
# загибается внутрь: y − x > LANE_CORNER. Бот — отражение топа.
LANE_EDGE, LANE_TOP, LANE_CORNER = 2300.0, 12500.0, 8700.0
MID_HALF = 1250.0          # |x − y − (CX − CY)| — полоса мида вдоль диагонали баз
RIVER_HALF = 1000.0        # |x + y − (CX + CY)| — полоса реки
DEEP = 5200.0              # так далеко от реки на их половине — глубоко на линии
# Ошибка места по источнику (ед., ~p90): иконка миникарты — 150
# (docs/minimap-champions.md), экран — камера и проекция ~250 (replay-api.md,
# entity-model-plan.md), выведено (фонтан, память) — 250.
ERR = {"minimap": 150.0, "screen": 250.0, "inferred": 250.0}
ERR_DEFAULT = 250.0

_NEW_NAME = re.compile(r"_T(Order|Chaos)_L([012])_P([1-5])", re.IGNORECASE)
_OLD_NAME = re.compile(r"_T([12])_([LCR])_0?([1-5])_", re.IGNORECASE)
_LANE_NEW = {"0": "bot", "1": "mid", "2": "top"}
_LANE_OLD = {"L": "top", "C": "mid", "R": "bot"}
_OLD_MID = {5: 3, 4: 2, 3: 1, 2: 5, 1: 4}


# ---------------------------------------------------------------- координаты

def crop_to_world(u: float, v: float) -> tuple[float, float]:
    """Доли рамки миникарты (с окантовкой) → мировые (x, y)."""
    x0, y0, x1, y1 = CROP_PX
    fx, fy = x0 + u * (x1 - x0), y0 + v * (y1 - y0)
    return ((fx - MAP_X0) / (MAP_X1 - MAP_X0) * MAP_UNITS,
            (MAP_Y_BOTTOM - fy) / (MAP_Y_BOTTOM - MAP_Y_TOP) * MAP_UNITS)


def world_to_crop(x: float, y: float) -> tuple[float, float]:
    """Мировые (x, y) → доли рамки миникарты (с окантовкой)."""
    x0, y0, x1, y1 = CROP_PX
    fx = MAP_X0 + x / MAP_UNITS * (MAP_X1 - MAP_X0)
    fy = MAP_Y_BOTTOM - y / MAP_UNITS * (MAP_Y_BOTTOM - MAP_Y_TOP)
    return (fx - x0) / (x1 - x0), (fy - y0) / (y1 - y0)


def world_to_field(x: float, y: float) -> tuple[float, float]:
    """Мировые → доли поля карты (u вправо, v вниз, без окантовки)."""
    return x / MAP_UNITS, 1.0 - y / MAP_UNITS


def mirror(x: float, y: float) -> tuple[float, float]:
    """Отражение через центр симметрии карты: ORDER-сторона ↔ CHAOS-сторона."""
    return 2 * CENTER[0] - x, 2 * CENTER[1] - y


_cx, _cy = crop_to_world(*_CHAOS_SEEN)
FOUNTAIN_WORLD = {"CHAOS": (round(_cx), round(_cy)),
                  "ORDER": tuple(round(c) for c in mirror(_cx, _cy))}
# Фонтаны в долях рамки — для MatchState (места возрождения `inferred`).
FOUNTAIN_CROP = {t: tuple(round(c, 4) for c in world_to_crop(*FOUNTAIN_WORLD[t])) for t in TEAMS}


# ---------------------------------------------------------------- башни

def turret_key(name: str) -> tuple[str, str, int] | None:
    """Имя башни API (Turret_TChaos_L1_P4_…; старое Turret_T2_C_05_A: на миде
    C_05 — внешняя, C_01 / C_02 — нексуса, на L / R 03 … 01 — как P) →
    (команда, линия, номер P); башни нексуса — линия `nexus`. Не башня — None."""
    if not isinstance(name, str):
        return None
    m = _NEW_NAME.search(name)
    if m:
        team, lane, pos = m.group(1).upper(), _LANE_NEW[m.group(2)], int(m.group(3))
    else:
        m = _OLD_NAME.search(name)
        if not m:
            return None
        team, lane, pos = ("ORDER" if m.group(1) == "1" else "CHAOS"), \
            _LANE_OLD[m.group(2).upper()], int(m.group(3))
        if lane == "mid":           # C_05 внешняя … C_03 у ингибитора, C_01 / C_02 нексуса
            pos = _OLD_MID[pos]
    return team, ("nexus" if pos in (4, 5) else lane), pos


def fallen_turrets(names: Iterable[str]) -> frozenset[tuple[str, str, int]]:
    """Ключи башен по именам из событий TurretKilled."""
    return frozenset(k for k in map(turret_key, names) if k is not None)


# ---------------------------------------------------------------- смысл места

def _top_lane(x: float, y: float) -> bool:
    return x < LANE_EDGE or y > LANE_TOP or y - x > LANE_CORNER


def _base(x: float, y: float) -> bool:
    return x < BASE_EDGE and y < BASE_EDGE and x + y < BASE_SUM


def zone(x: float, y: float) -> str:
    """base / top / bot / mid / river / jungle (мировые координаты)."""
    mx, my = mirror(x, y)
    if _base(x, y) or _base(mx, my):
        return "base"
    if _top_lane(x, y):
        return "top"
    if _top_lane(mx, my):
        return "bot"
    if abs(x - y - (CENTER[0] - CENTER[1])) < MID_HALF:
        return "mid"
    if abs(x + y - (CENTER[0] + CENTER[1])) < RIVER_HALF:
        return "river"
    return "jungle"


def side(x: float, y: float) -> str:
    """Сторона карты от оси мида: top (над диагональю баз), mid (полоса мида), bot."""
    s = x - y - (CENTER[0] - CENTER[1])
    return "mid" if abs(s) < MID_HALF else ("top" if s < 0 else "bot")


def team_half(x: float, y: float) -> str:
    """Чья половина по реке: ORDER (низ-лево) / CHAOS."""
    return "ORDER" if x + y < CENTER[0] + CENTER[1] else "CHAOS"


def _river_off(x: float, y: float) -> float:
    return abs(x + y - (CENTER[0] + CENTER[1]))


def _relation(team: str, own: str | None) -> str | None:
    return None if own not in TEAMS else ("ally" if team == own else "enemy")


def turrets_by_distance(x: float, y: float, own_team: str | None = None,
                        fallen: Iterable[tuple] | None = None) -> list[dict]:
    """Башни, не павшие по данным сервиса, от ближней к дальней. `fallen` None —
    павшие неизвестны (нет API): тогда все башни, `standing` = None."""
    known = fallen is not None
    gone = frozenset(fallen or ())
    out = []
    for key, (tx, ty) in TURRETS.items():
        if key in gone:
            continue
        d = math.hypot(x - tx, y - ty)
        out.append({"team": key[0], "lane": key[1], "tier": key[2],
                    "relation": _relation(key[0], own_team), "dist": round(d),
                    "near": d <= AT_TURRET, "standing": True if known else None})
    out.sort(key=lambda t: t["dist"])
    return out


def describe(x: float, y: float, own_team: str | None = None,
             fallen: Iterable[tuple] | None = None, err: float = ERR_DEFAULT,
             with_turrets: bool = False) -> dict:
    """Смысл точки (мировые): доли поля, зона, сторона, половина, база,
    глубина, ближайшая живая башня, ямы, ошибка и уверенность.

    `own_team` — своя команда (ORDER / CHAOS) или None: тогда относительные
    поля (`half`, `deep`, `relation` башни) — None. `err` — ошибка места (ед.):
    `sure` — зона, половина и «у башни» те же во всём круге этого радиуса.
    `with_turrets` — ещё и список всех живых башен с расстояниями."""
    fu, fv = world_to_field(x, y)
    z, th = zone(x, y), team_half(x, y)
    in_river = _river_off(x, y) < RIVER_HALF and z != "base"
    if own_team in TEAMS:
        half = "river" if in_river else ("ally" if th == own_team else "enemy")
        deep = half == "enemy" and (z in ("jungle", "base") or _river_off(x, y) >= DEEP)
    else:
        half = deep = None
    base = th if z == "base" else None
    towers = turrets_by_distance(x, y, own_team, fallen)
    near = towers[0] if towers else None
    pits = {k: round(math.hypot(x - px, y - py)) for k, (px, py) in PITS.items()}
    pit = min(pits, key=pits.get)

    tx, ty = (None, None) if near is None else TURRETS[(near["team"], near["lane"],
                                                         near["tier"])]

    def key(px: float, py: float) -> tuple:
        h = "river" if _river_off(px, py) < RIVER_HALF else team_half(px, py)
        t = None if tx is None else math.hypot(px - tx, py - ty) <= AT_TURRET
        return zone(px, py), h, t

    here = key(x, y)
    sure = all(key(x + err * math.cos(a), y + err * math.sin(a)) == here
               for a in (k * math.pi / 4 for k in range(8)))
    out = {"x": round(x), "y": round(y), "u": round(fu, 4), "v": round(fv, 4),
           "zone": z, "side": side(x, y), "team_half": th, "half": half, "base": base,
           "deep": deep, "turret": near, "pit": pit if pits[pit] <= AT_PIT else None,
           "pits": pits, "err": round(err), "sure": sure}
    if with_turrets:
        out["turrets"] = towers
    return out


def describe_crop(u: float, v: float, src: str | None, own_team: str | None = None,
                  fallen: Iterable[tuple] | None = None, with_turrets: bool = False) -> dict:
    """То же для места в долях рамки миникарты (как в MatchState); ошибка — по
    источнику места (ERR)."""
    x, y = crop_to_world(u, v)
    return describe(x, y, own_team, fallen, ERR.get(src or "", ERR_DEFAULT), with_turrets)
