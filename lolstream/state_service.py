"""Сервис состояния своей игры для coach (#796, задача 1C): снимок и поток событий.

Что это. coach спрашивает языковую модель по кадру экрана — это секунды,
деньги и выдумки. Здесь — проверенные факты о собственной игре пользователя,
несколько раз в секунду, на loopback:

* `GET /state` — снимок состояния (JSON, `schema: 1`);
* `GET /events` — поток событий Server-Sent Events (почему не WebSocket —
  docs/state-api.md, «Поток событий»);
* `GET /health` — жив ли сервис и когда собирал снимок последний раз.

Контракт полей — docs/state-api.md; машинная схема — `snapshot_schema()`,
проверка снимка без зависимостей — `check_snapshot()`.

Факт. Каждое значение выхода — объект `{"value", "source", "age", "reason"}`:
либо значение с источником (`api` / `minimap` / `screen` / `hud` /
`inferred`) и давностью в секундах, либо `value = null` с причиной.
Молчаливых подстановок нет: нет входа — null и почему.

Приоритет источников: Live Client API > HUD > зрение. Где API и зрение
говорят разное про одно и то же (команда, жив ли, таймер, состав), в выход
идёт API, а расхождение — в `conflicts` (сигнал #781), без тихого
перетирания.

Устройство. `build_snapshot` — чистая функция: входы (`StateInputs`) →
словарь снимка; ни сети, ни потоков. `StateHub` раз в `period` секунд зовёт
провайдера (callable → `StateInputs`), собирает снимок и рассылает новые
события подписчикам. `StateServer` — HTTP поверх хаба
(`ThreadingHTTPServer`, только 127.0.0.1). Сервис не импортирует live.py:
провайдера из сеанса live подключает интегратор.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lolstream.live_client import NEW_GAME_BACKSTEP, GameSnapshot, LiveStatus, PlayerRecord
from lolvision.records import ChampCtx, Event, TrackRec, UnitRec

SCHEMA = 1
HOST = "127.0.0.1"
PORT = 8733                    # смотрелка — 8732
SOURCES = ("api", "minimap", "screen", "hud", "inferred")
RELATIONS = ("self", "ally", "enemy")
TEAMS = ("ORDER", "CHAOS")
OTHER = {"ORDER": "CHAOS", "CHAOS": "ORDER"}
# Цвет плашки на виде игрока — отношение к игроку (unit_bars.TEAM_BY_COLOR["player"]).
RELATION_BY_COLOR = {"SELF": "self", "ALLY": "ally", "ENEMY": "enemy"}
# Расхождение таймеров возрождения API и HUD, которое считается конфликтом, с.
RESPAWN_TOLERANCE = 3.0

# Причины отказов, общие для многих полей.
WHY_NO_API = "Live Client API не подключён к сервису"
WHY_NO_VISION = "нет обработанного кадра: сеанс зрения не подключён или ещё не дал кадр"
WHY_NO_PASSPORT = "паспорт трансляции ещё не построен (2C)"
WHY_NO_RELATION = "свои/враги по экрану ещё не определяются (2A, #790)"
WHY_PLAYER_COLORS = ("на виде игрока цвет плашки относителен: ORDER/CHAOS по цвету "
                     "недостоверны (2A, #790)")


# ---------------------------------------------------------------- входы

@dataclass(frozen=True)
class FrameStatus:
    """Источник кадров (lolstream.frame_source) на момент снимка.

    `state` — `Status.key` последнего ответа источника (ok / paused /
    minimized / no_window / no_frame / ended); `reason` — почему кадра нет;
    `last_ok` — unix-время последнего снятого кадра или None."""
    kind: str | None             # screen / youtube / file
    state: str
    reason: str | None = None
    backend: str | None = None
    backend_note: str | None = None
    fps: float | None = None
    size: tuple[int, int] | None = None
    last_ok: float | None = None


@dataclass(frozen=True)
class VisionFrame:
    """Итог зрения по последнему обработанному кадру.

    `at` — unix-время, когда кадр обработан (от него считается давность
    фактов зрения); `t` — время кадра по шкале потока. `view` — раскладка:
    spectator / player / None (неизвестна). `camera` — центр камеры (доли
    карты) или None — модель камеры не ответила. Списки None — этот читатель
    не работал (нет модели, кадр не «игра»), пустой список — работал и ничего
    не нашёл."""
    i: int
    t: float
    at: float
    view: str | None
    scene: str | None = None
    camera: tuple[float, float] | None = None
    spread: float | None = None
    units: Sequence[UnitRec] | None = None
    tracks: Sequence[TrackRec] | None = None
    ctx: Sequence[ChampCtx] | None = None


@dataclass(frozen=True)
class StateInputs:
    """Всё, из чего собирается снимок. None — входа нет (не подключён).

    `session` — состояние сеанса live (подпись, например «идёт») или None;
    `vision_events` — события контекста матча (records.Event), можно
    накопительно: хаб сам отсеивает уже разосланные; `error` — провайдер
    упал, все отказы получают эту причину."""
    session: str | None = None
    frames: FrameStatus | None = None
    vision: VisionFrame | None = None
    api: LiveStatus | None = None
    passport: Mapping[str, Any] | None = None
    vision_events: Sequence[Event] = field(default_factory=tuple)
    error: str | None = None


# ---------------------------------------------------------------- факты

def _age(a: float | None) -> float | None:
    return None if a is None else round(max(0.0, float(a)), 2)


def fact(value: Any, source: str, age: float | None = None) -> dict:
    """Значение с источником и давностью (с)."""
    if value is None:
        raise ValueError("fact без значения — это refuse с причиной")
    if source not in SOURCES:
        raise ValueError(f"неизвестный источник {source!r}")
    return {"value": value, "source": source, "age": _age(age), "reason": None}


def refuse(reason: str) -> dict:
    """Значения нет; почему — `reason`."""
    if not reason:
        raise ValueError("отказ без причины")
    return {"value": None, "source": None, "age": None, "reason": reason}


def maybe(value: Any, source: str, age: float | None, reason: str) -> dict:
    return refuse(reason) if value is None else fact(value, source, age)


def _uv(u: float, v: float) -> dict:
    return {"u": round(float(u), 4), "v": round(float(v), 4)}


# ---------------------------------------------------------------- сборка

@dataclass
class _Api:
    """Разобранный вход API для сборки: снимок игры или почему его нет."""
    game: GameSnapshot | None
    age: float | None            # с от получения снимка игры
    why: str
    me: PlayerRecord | None = None
    me_why: str = ""
    side: str | None = None      # своя команда ORDER / CHAOS


def _api_view(api: LiveStatus | None, now: float, miss: Callable[[str], str]) -> _Api:
    if api is None:
        return _Api(None, None, miss(WHY_NO_API), me_why=miss(WHY_NO_API))
    if api.game is None:
        why = f"Live Client: {api.state.value}" + (f" ({api.reason})" if api.reason else "")
        return _Api(None, None, why, me_why=why)
    age = None if api.game_t is None else now - api.game_t
    out = _Api(api.game, age, "")
    me = api.game.my_player
    if me is None:
        out.me_why = f"свой игрок не найден: {api.game.me_reason}"
    else:
        out.me, out.side = me, me.team
    return out


def _player_key(p: PlayerRecord, i: int) -> str:
    return p.champion if p.champion is not None else f"#{i}"


def _relation(i: int, p: PlayerRecord, a: _Api) -> dict:
    if a.side is None:
        return refuse(a.me_why or a.why)
    if a.game is not None and a.game.me == i:
        return fact("self", "api", a.age)
    return fact("ally" if p.team == a.side else "enemy", "api", a.age)


def _unit_side(team: str, view: str | None, side: str | None, side_why: str,
               lag: float | None) -> tuple[dict, dict]:
    """Цвет плашки → (команда, отношение к игроку) как факты или отказы."""
    if team in RELATION_BY_COLOR:                       # вид игрока: цвет — отношение
        rel = RELATION_BY_COLOR[team]
        if side is None:
            t = refuse(f"цвет даёт отношение, а своя сторона неизвестна: {side_why}")
        else:
            t = fact(side if rel in ("self", "ally") else OTHER[side], "inferred", lag)
        return t, fact(rel, "screen", lag)
    if team not in TEAMS:
        why = f"цвет плашки вне ожидаемого: {team!r}"
        return refuse(why), refuse(why)
    if view == "spectator":
        t = fact(team, "screen", lag)
        if side is None:
            return t, refuse(f"своя сторона неизвестна: {side_why}")
        return t, fact("ally" if team == side else "enemy", "inferred", lag)
    if view == "player":
        return refuse(WHY_PLAYER_COLORS), refuse(WHY_NO_RELATION)
    why = "вид (наблюдатель / игрок) неизвестен: цвет плашки не переводится в сторону"
    return refuse(why), refuse(why)


def _screen_names(vis: VisionFrame | None) -> dict[str, UnitRec]:
    if vis is None or vis.units is None:
        return {}
    return {u.name: u for u in vis.units if u.kind == "champion" and u.name}


def _on_screen(name: str | None, vis: VisionFrame | None, lag: float | None,
               named: dict[str, UnitRec], why_vis: str) -> tuple[dict, dict]:
    """(виден ли на экране, ХП с плашки) — только по чемпиону с прочитанным именем."""
    if vis is None:
        return refuse(why_vis), refuse(why_vis)
    if vis.units is None:
        why = "юниты на экране не ищутся (нет детектора или кадр не «игра»)"
        return refuse(why), refuse(why)
    if name is not None and name in named:
        return fact(True, "screen", lag), fact(round(named[name].hp, 2), "screen", lag)
    if vis.view == "player":
        why = "на виде игрока имена над плашками не читаются"
    else:
        why = "на экране не опознан: имени над плашкой нет в кадре или оно не прочитано"
    return refuse(why), refuse(why)


def _champion_entry(name: str | None, p: PlayerRecord | None, i: int | None, a: _Api,
                    c: ChampCtx | None, vis: VisionFrame | None, lag: float | None,
                    named: dict[str, UnitRec], why_vis: str, conflicts: list[dict]) -> dict:
    """Один из десяти: API (если есть) поверх зрения, расхождения — в conflicts."""
    why_ctx = why_vis if vis is None else (
        "контекст матча не ведёт этого чемпиона" if vis.ctx is not None
        else "контекст матча пуст: состава по HUD ещё нет")
    vage = None if lag is None or c is None or c.age is None else lag + c.age

    # кто и чей
    if p is not None:
        champion = maybe(p.champion, "api", a.age, p.champion_reason or "чемпион не опознан")
        team = fact(p.team, "api", a.age)
        relation = _relation(i, p, a)
        riot_id = fact(p.riot_id, "api", a.age)
        level = fact(p.level, "api", a.age)
        items = fact([it.item_id for it in p.items], "api", a.age)
        scores = fact(p.scores.to_dict(), "api", a.age)
    else:
        champion = fact(name, "hud", lag)
        team = fact(c.team, "hud", lag)
        relation = refuse(a.me_why or a.why)
        riot_id = level = items = scores = refuse(a.why)

    # жив ли и таймер: API > HUD; зрение — только где API нет
    if p is not None:
        alive = fact(not p.is_dead, "api", a.age)
        respawn = (fact(p.respawn_timer, "api", a.age) if p.is_dead
                   else refuse("жив (api)"))
    elif c is None:
        alive = respawn = refuse(why_ctx)
    elif c.state == "dead":
        alive = fact(False, "hud", lag)
        respawn = maybe(c.respawn_in, "hud", lag, "мёртв, таймер возрождения не прочитан")
    elif c.state == "visible":
        alive = fact(True, c.src or "inferred", lag)
        respawn = refuse("жив (виден)")
    else:
        alive = respawn = refuse(f"зрение не знает: чемпион {c.state}")

    if p is not None and c is not None:
        _conflicts_of(conflicts, p, a, c, lag)

    if c is None:
        position = map_state = refuse(why_ctx)
    else:
        position = (refuse("место не наблюдалось") if c.u is None or c.v is None
                    else fact(_uv(c.u, c.v), c.src or "inferred", vage))
        map_state = fact(c.state, "inferred", lag)
    on_screen, hp = _on_screen(name, vis, lag, named, why_vis)
    return {"champion": champion, "riot_id": riot_id, "team": team, "relation": relation,
            "level": level, "alive": alive, "respawn_in": respawn, "position": position,
            "map_state": map_state, "on_screen": on_screen, "hp": hp, "items": items,
            "scores": scores}


def _conflicts_of(conflicts: list[dict], p: PlayerRecord, a: _Api, c: ChampCtx,
           lag: float | None) -> None:
    """Расхождения API и зрения по одному чемпиону. В выход идёт API."""
    who = p.champion
    if c.team != p.team:
        conflicts.append({"field": "team", "champion": who, "used": "api",
                          "values": {"api": p.team, "hud": c.team}})
    vis_alive = {"dead": False, "visible": True}.get(c.state)
    if vis_alive is not None and vis_alive == p.is_dead:
        conflicts.append({"field": "alive", "champion": who, "used": "api",
                          "values": {"api": not p.is_dead,
                                     ("hud" if c.state == "dead" else c.src or "inferred"):
                                     vis_alive}})
    if p.is_dead and c.state == "dead" and c.respawn_in is not None and a.age is not None \
            and lag is not None:
        api_now = p.respawn_timer - a.age
        hud_now = c.respawn_in - lag
        if abs(api_now - hud_now) > RESPAWN_TOLERANCE:
            conflicts.append({"field": "respawn_in", "champion": who, "used": "api",
                              "values": {"api": round(api_now, 1), "hud": round(hud_now, 1)}})


def _sources(inp: StateInputs, now: float, miss: Callable[[str], str]) -> dict:
    f = inp.frames
    frames = ({"state": "absent", "reason": miss("источник кадров не подключён"),
               "kind": None, "backend": None, "backend_note": None, "fps": None,
               "size": None, "last_frame_age": None} if f is None else
              {"state": f.state, "reason": f.reason, "kind": f.kind, "backend": f.backend,
               "backend_note": f.backend_note, "fps": f.fps,
               "size": None if f.size is None else list(f.size),
               "last_frame_age": None if f.last_ok is None else _age(now - f.last_ok)})
    if f is not None and f.state != "ok" and not f.reason:
        frames["reason"] = f"кадра нет: {f.state}"
    v = inp.vision
    vision = ({"state": "absent", "reason": miss(WHY_NO_VISION), "frame": None,
               "stream_t": None, "age": None, "view": None, "session": inp.session}
              if v is None else
              {"state": "ok", "reason": None, "frame": v.i, "stream_t": v.t,
               "age": _age(now - v.at), "view": v.view, "session": inp.session})
    a = inp.api
    api = ({"state": "absent", "reason": miss(WHY_NO_API), "age": None, "game_age": None}
           if a is None else
           {"state": a.state.value, "reason": a.reason, "age": _age(now - a.t),
            "game_age": None if a.game_t is None else _age(now - a.game_t)})
    return {"frames": frames, "vision": vision, "api": api}


def build_snapshot(inp: StateInputs, now: float, seq: int = 0) -> dict:
    """Входы → снимок состояния по контракту docs/state-api.md (schema 1).

    Чистая функция: `now` — unix-время снимка, от него считаются все
    давности; `seq` — номер снимка (ставит хаб)."""
    def miss(why: str) -> str:
        return why if inp.error is None else f"сбой провайдера состояния: {inp.error}"

    a = _api_view(inp.api, now, miss)
    vis = inp.vision
    why_vis = miss(WHY_NO_VISION)
    lag = None if vis is None else max(0.0, now - vis.at)
    named = _screen_names(vis)
    conflicts: list[dict] = []

    # --- матч и свой игрок (только API) ---
    g = a.game
    if g is None:
        match = {"time": refuse(a.why), "mode": refuse(a.why), "map": refuse(a.why),
                 "side": refuse(a.why), "state": refuse(a.why) if inp.api is None
                 else fact(inp.api.state.value, "api", now - inp.api.t)}
    else:
        match = {"time": fact(g.game_time, "api", a.age), "mode": fact(g.game_mode, "api", a.age),
                 "map": maybe(g.map_number, "api", a.age, "API не дал mapNumber"),
                 "side": maybe(a.side, "api", a.age, a.me_why or "свой игрок не найден"),
                 "state": fact(inp.api.state.value, "api", now - inp.api.t)}
    me_fields = ("riot_id", "champion", "team", "level", "gold", "alive", "respawn_in", "stats")
    if a.me is None:
        me = {k: refuse(a.me_why or a.why) for k in me_fields}
    else:
        p, act = a.me, g.active
        me = {"riot_id": fact(p.riot_id, "api", a.age),
              "champion": maybe(p.champion, "api", a.age, p.champion_reason or "не опознан"),
              "team": fact(p.team, "api", a.age), "level": fact(p.level, "api", a.age),
              "gold": (fact(act.current_gold, "api", a.age) if act is not None
                       else refuse("нет activePlayer")),
              "alive": fact(not p.is_dead, "api", a.age),
              "respawn_in": (fact(p.respawn_timer, "api", a.age) if p.is_dead
                             else refuse("жив (api)")),
              "stats": (fact(dict(act.stats), "api", a.age) if act is not None
                        else refuse("нет activePlayer"))}

    # --- десять чемпионов: состав API, иначе контекст зрения ---
    ctx = {c.name: c for c in (vis.ctx or ())} if vis is not None else {}
    if g is not None and g.players:
        entries = []
        keys = set()
        for i, p in enumerate(g.players):
            key = _player_key(p, i)
            keys.add(key)
            entries.append(_champion_entry(p.champion, p, i, a, ctx.get(key), vis, lag, named,
                                           why_vis, conflicts))
        extra = sorted(set(ctx) - keys)
        if extra:
            conflicts.append({"field": "roster", "champion": None, "used": "api",
                              "values": {"api": sorted(k for k in keys if not k.startswith("#")),
                                         "hud": sorted(ctx)}})
        champions = fact(entries, "api", a.age)
    elif ctx:
        entries = [_champion_entry(c.name, None, None, a, c, vis, lag, named, why_vis, conflicts)
                   for c in vis.ctx]
        champions = fact(entries, "hud", lag)
    else:
        why = a.why if vis is not None and vis.ctx is not None else \
            f"{a.why}; {why_vis if vis is None else 'контекст матча не ведётся'}"
        champions = refuse(f"состава нет: {why}")

    # --- враги с давностью места ---
    if champions["value"] is None:
        enemies = refuse(champions["reason"])
    elif a.side is None:
        enemies = refuse(f"своя сторона неизвестна: {a.me_why or a.why}")
    else:
        enemies = fact([{k: e[k] for k in ("champion", "position", "map_state", "alive",
                                            "respawn_in", "on_screen")}
                        for e in champions["value"] if e["relation"]["value"] == "enemy"],
                       "api", a.age)

    # --- камера, сцена, юниты, треки ---
    if vis is None:
        camera = {"center": refuse(why_vis), "spread": refuse(why_vis)}
        scene = refuse(why_vis)
        units = tracks = refuse(why_vis)
    else:
        cam = vis.camera
        ok = cam is not None and cam[0] is not None and cam[1] is not None
        camera = {"center": fact(_uv(*cam), "screen", lag) if ok
                  else refuse("модель камеры не ответила"),
                  "spread": maybe(None if vis.spread is None else round(vis.spread, 4),
                                  "screen", lag, "разброс камеры не считается")}
        scene = maybe(vis.scene, "screen", lag, "тип сцены не считается (нет модели)")
        side_why = a.me_why or a.why
        if vis.units is None:
            units = refuse("юниты на экране не ищутся (нет детектора или кадр не «игра»)")
        else:
            out = []
            for u in vis.units:
                team, rel = _unit_side(u.team, vis.view, a.side, side_why, lag)
                out.append({"kind": u.kind, "x": round(u.x, 4), "y": round(u.y, 4),
                            "hp": round(u.hp, 2), "score": round(u.score, 2),
                            "name": maybe(u.name, "screen", lag,
                                          "не чемпион или имя не прочитано"),
                            "team": team, "relation": rel})
            units = fact(out, "screen", lag)
        if vis.tracks is None:
            tracks = refuse("треки юнитов не ведутся")
        else:
            out = []
            for tr in vis.tracks:
                team, rel = _unit_side(tr.team, vis.view, a.side, side_why, lag + tr.age)
                d = tr.to_dict()
                d.update(team=team, relation=rel)
                out.append(d)
            tracks = fact(out, "screen", lag)

    passport = (refuse(miss(WHY_NO_PASSPORT)) if inp.passport is None
                else fact(dict(inp.passport), "inferred"))
    return {"schema": SCHEMA, "seq": seq, "t": round(now, 3), "error": inp.error,
            "sources": _sources(inp, now, miss), "passport": passport, "match": match,
            "me": me, "champions": champions, "enemies": enemies, "camera": camera,
            "scene": scene, "units": units, "tracks": tracks,
            "conflicts": {"count": len(conflicts), "items": conflicts}}


# ---------------------------------------------------------------- схема

def _fact_of(value: dict | None = None) -> dict:
    s: dict = {"$ref": "#/$defs/fact"}
    if value is not None:
        s["properties"] = {"value": {"anyOf": [value, {"type": "null"}]}}
    return s


def _obj(props: dict, extra: bool = False) -> dict:
    return {"type": "object", "required": sorted(props), "properties": props,
            "additionalProperties": extra}


def snapshot_schema() -> dict:
    """JSON Schema (draft 2020-12) снимка. Проверка без зависимостей — `check_snapshot`."""
    num, s, b, i = {"type": "number"}, {"type": "string"}, {"type": "boolean"}, \
        {"type": "integer"}
    nnum, ns, ni = {"type": ["number", "null"]}, {"type": ["string", "null"]}, \
        {"type": ["integer", "null"]}
    uv = _obj({"u": num, "v": num})
    team = {"enum": list(TEAMS)}
    rel = {"enum": list(RELATIONS)}
    champ = _obj({"champion": _fact_of(s), "riot_id": _fact_of(s), "team": _fact_of(team),
                  "relation": _fact_of(rel), "level": _fact_of(i), "alive": _fact_of(b),
                  "respawn_in": _fact_of(num), "position": _fact_of(uv),
                  "map_state": _fact_of({"enum": ["visible", "remembered", "dead", "unknown"]}),
                  "on_screen": _fact_of(b), "hp": _fact_of(num),
                  "items": _fact_of({"type": "array", "items": i}),
                  "scores": _fact_of({"type": "object"})})
    enemy = _obj({k: champ["properties"][k] for k in ("champion", "position", "map_state",
                                                      "alive", "respawn_in", "on_screen")})
    unit = _obj({"kind": s, "x": num, "y": num, "hp": num, "score": num,
                 "name": _fact_of(s), "team": _fact_of(team), "relation": _fact_of(rel)})
    track = _obj({"id": s, "kind": s, "state": {"enum": ["seen", "unseen", "lost"]},
                  "u": num, "v": num, "x": nnum, "y": nnum, "hp": num, "age": num,
                  "team": _fact_of(team), "relation": _fact_of(rel)})
    conflict = _obj({"field": {"enum": ["team", "alive", "respawn_in", "roster"]},
                     "champion": ns, "used": {"enum": list(SOURCES)},
                     "values": {"type": "object"}})
    frames = _obj({"state": {"enum": ["absent", "ok", "paused", "minimized", "no_window",
                                      "no_frame", "ended"]},
                   "reason": ns, "kind": ns, "backend": ns, "backend_note": ns, "fps": nnum,
                   "size": {"anyOf": [{"type": "array", "items": i, "minItems": 2,
                                       "maxItems": 2}, {"type": "null"}]},
                   "last_frame_age": nnum})
    vision = _obj({"state": {"enum": ["absent", "ok"]}, "reason": ns, "frame": ni,
                   "stream_t": nnum, "age": nnum,
                   "view": {"enum": ["spectator", "player", None]}, "session": ns})
    api = _obj({"state": {"enum": ["absent", "no_game", "loading", "in_game", "ended",
                                   "error"]},
                "reason": ns, "age": nnum, "game_age": nnum})
    fact_def = {
        "type": "object", "required": ["value", "source", "age", "reason"],
        "properties": {"value": {}, "source": {"enum": [*SOURCES, None]},
                       "age": nnum, "reason": ns},
        "additionalProperties": False,
        # value есть ⇔ есть источник и нет причины; value null ⇔ есть причина
        "oneOf": [
            {"properties": {"value": {"not": {"type": "null"}}, "source": s,
                            "reason": {"type": "null"}}},
            {"properties": {"value": {"type": "null"}, "source": {"type": "null"},
                            "age": {"type": "null"}, "reason": {"type": "string",
                                                                "minLength": 1}}},
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "lol-game-state-vision/state-api/1",
        "$defs": {"fact": fact_def},
        **_obj({
            "schema": {"const": SCHEMA}, "seq": i, "t": num, "error": ns,
            "sources": _obj({"frames": frames, "vision": vision, "api": api}),
            "passport": _fact_of({"type": "object"}),
            "match": _obj({"time": _fact_of(num), "mode": _fact_of(s), "map": _fact_of(i),
                           "side": _fact_of(team),
                           "state": _fact_of({"enum": ["no_game", "loading", "in_game",
                                                       "ended", "error"]})}),
            "me": _obj({"riot_id": _fact_of(s), "champion": _fact_of(s), "team": _fact_of(team),
                        "level": _fact_of(i), "gold": _fact_of(num), "alive": _fact_of(b),
                        "respawn_in": _fact_of(num), "stats": _fact_of({"type": "object"})}),
            "champions": _fact_of({"type": "array", "items": champ, "maxItems": 10}),
            "enemies": _fact_of({"type": "array", "items": enemy, "maxItems": 5}),
            "camera": _obj({"center": _fact_of(uv), "spread": _fact_of(num)}),
            "scene": _fact_of(s),
            "units": _fact_of({"type": "array", "items": unit}),
            "tracks": _fact_of({"type": "array", "items": track}),
            "conflicts": _obj({"count": i, "items": {"type": "array", "items": conflict}}),
        }),
    }


_TYPES: dict[str, Callable[[Any], bool]] = {
    "null": lambda x: x is None,
    "boolean": lambda x: isinstance(x, bool),
    "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "string": lambda x: isinstance(x, str),
    "array": lambda x: isinstance(x, list),
    "object": lambda x: isinstance(x, dict),
}


def _check(x: Any, s: dict, root: dict, path: str, out: list[str]) -> None:
    """Подмножество JSON Schema, которым написана `snapshot_schema`."""
    if "$ref" in s:
        _check(x, root["$defs"][s["$ref"].rsplit("/", 1)[1]], root, path, out)
    if "type" in s:
        kinds = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_TYPES[k](x) for k in kinds):
            out.append(f"{path}: {type(x).__name__}, ожидался {s['type']}")
            return
    if "const" in s and x != s["const"]:
        out.append(f"{path}: {x!r} ≠ {s['const']!r}")
    if "enum" in s and x not in s["enum"]:
        out.append(f"{path}: {x!r} не из {s['enum']}")
    if "minLength" in s and isinstance(x, str) and len(x) < s["minLength"]:
        out.append(f"{path}: пустая строка")
    if "not" in s and not _errors(x, s["not"], root, path):
        out.append(f"{path}: не должно подходить под {s['not']}")
    if "anyOf" in s and all(_errors(x, sub, root, path) for sub in s["anyOf"]):
        out.append(f"{path}: не подходит ни под один вариант anyOf")
    if "oneOf" in s:
        n = sum(not _errors(x, sub, root, path) for sub in s["oneOf"])
        if n != 1:
            out.append(f"{path}: подходит под {n} вариантов oneOf из {len(s['oneOf'])}")
    if isinstance(x, dict):
        props = s.get("properties", {})
        for k in s.get("required", ()):
            if k not in x:
                out.append(f"{path}: нет поля {k}")
        for k, v in x.items():
            if k in props:
                _check(v, props[k], root, f"{path}.{k}", out)
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: лишнее поле {k}")
    if isinstance(x, list):
        if "minItems" in s and len(x) < s["minItems"]:
            out.append(f"{path}: меньше {s['minItems']} элементов")
        if "maxItems" in s and len(x) > s["maxItems"]:
            out.append(f"{path}: больше {s['maxItems']} элементов")
        if "items" in s:
            for n, v in enumerate(x):
                _check(v, s["items"], root, f"{path}[{n}]", out)


def _errors(x: Any, s: dict, root: dict, path: str) -> list[str]:
    out: list[str] = []
    _check(x, s, root, path, out)
    return out


def check_snapshot(snap: Any) -> list[str]:
    """Ошибки снимка против `snapshot_schema()`; пустой список — снимок по контракту.
    Стандартная библиотека: jsonschema сервису не нужен."""
    root = snapshot_schema()
    return _errors(json.loads(json.dumps(snap)), root, root, "$")


# ---------------------------------------------------------------- хаб

Provider = Callable[[], StateInputs]


def _event_key(e: Event) -> str:
    return json.dumps(e.to_dict(), sort_keys=True, ensure_ascii=False)


class _Sub:
    """Подписчик потока: своя очередь; переполнилась — подписчик отключается
    (клиент переподключится и получит свежий снимок в hello)."""

    def __init__(self, size: int):
        self.q: queue.Queue = queue.Queue(size)
        self.dropped = False


class StateHub:
    """Опрашивает провайдера раз в `period` с: снимок + новые события.

    События потока (`event` SSE): `state` — снимок, не чаще раза в
    `state_every` с; `game` — новое событие Live Client (по EventID, новая
    игра — счёт заново); `vision` — новое событие контекста матча; `source`
    — сменилось состояние источника кадров, зрения или API. Провайдер упал —
    снимок из одних отказов с причиной, хаб живёт дальше."""

    def __init__(self, provider: Provider, period: float = 0.25, state_every: float = 1.0,
                 clock: Callable[[], float] = time.time, queue_size: int = 256):
        self.provider, self.period, self.state_every = provider, period, state_every
        self.clock, self.queue_size = clock, queue_size
        self._lock = threading.Lock()
        self._subs: list[_Sub] = []
        self._snap: dict | None = None
        self._seq = 0
        self._eid = 0
        self._last_state: float | None = None
        self._last_tick: float | None = None
        self._api_seen: set[int] = set()
        self._api_time: float | None = None
        self._vision_seen: set[str] = set()
        self._src: tuple | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------- такт ----------

    def tick(self) -> dict:
        """Один опрос провайдера: снимок и рассылка. Возвращает снимок."""
        now = self.clock()
        try:
            inp = self.provider()
        except Exception as e:  # noqa: BLE001 — сбой провайдера становится отказом
            inp = StateInputs(error=f"{type(e).__name__}: {e}")
        with self._lock:
            self._seq += 1
            snap = build_snapshot(inp, now, self._seq)
            self._snap, self._last_tick = snap, now
            out = self._new_events(inp, snap)
            if self._last_state is None or now - self._last_state >= self.state_every:
                self._last_state = now
                out.append(("state", snap))
            for kind, data in out:
                self._publish(kind, data, now)
        return snap

    def _new_events(self, inp: StateInputs, snap: dict) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        src = tuple(snap["sources"][k]["state"] for k in ("frames", "vision", "api"))
        if src != self._src:
            self._src = src
            out.append(("source", snap["sources"]))
        game = inp.api.game if inp.api is not None else None
        if game is not None:
            if self._api_time is not None and game.game_time < self._api_time - NEW_GAME_BACKSTEP:
                self._api_seen.clear()           # новая игра: EventID с нуля
            self._api_time = game.game_time
            for e in game.events:
                if e.event_id not in self._api_seen:
                    self._api_seen.add(e.event_id)
                    out.append(("game", e.to_dict()))
        for e in inp.vision_events:
            k = _event_key(e)
            if k not in self._vision_seen:
                self._vision_seen.add(k)
                out.append(("vision", {**e.to_dict(), "source": "inferred"}))
        return out

    def _publish(self, kind: str, data: dict, now: float) -> None:
        self._eid += 1
        msg = (self._eid, kind, {"schema": SCHEMA, "t": round(now, 3), "data": data})
        for sub in list(self._subs):
            try:
                sub.q.put_nowait(msg)
            except queue.Full:
                sub.dropped = True
                self._subs.remove(sub)

    # ---------- чтение ----------

    def snapshot(self) -> dict:
        """Последний снимок; до первого такта — собрать сейчас."""
        with self._lock:
            snap = self._snap
        return snap if snap is not None else self.tick()

    def subscribe(self) -> tuple[_Sub, dict]:
        """Новый подписчик и снимок для приветствия (hello)."""
        snap = self.snapshot()
        sub = _Sub(self.queue_size)
        with self._lock:
            self._subs.append(sub)
        return sub, snap

    def unsubscribe(self, sub: _Sub) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def health(self) -> dict:
        now = self.clock()
        with self._lock:
            snap, last, n = self._snap, self._last_tick, len(self._subs)
        age = None if last is None else round(now - last, 3)
        running = self._thread is not None and self._thread.is_alive()
        error = None if snap is None else snap["error"]
        fresh = age is not None and age <= max(5 * self.period, 2.0)
        return {"ok": bool(running and fresh and error is None), "schema": SCHEMA,
                "running": running, "seq": None if snap is None else snap["seq"],
                "age": age, "error": error, "subscribers": n}

    # ---------- поток ----------

    def _run(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            self.tick()
            self._stop.wait(max(0.0, self.period - (time.monotonic() - start)))

    def start(self) -> StateHub:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="state-hub", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        with self._lock:
            subs, self._subs = self._subs, []
        for sub in subs:                         # разбудить потоки /events: конец
            try:
                sub.q.put_nowait(None)
            except queue.Full:
                sub.dropped = True


# ---------------------------------------------------------------- HTTP

def sse(eid: int, kind: str, data: dict) -> bytes:
    """Одно сообщение Server-Sent Events: id, event, data одной строкой JSON."""
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {eid}\nevent: {kind}\ndata: {body}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    server_version = "lol-state/1"
    hub: StateHub                   # ставит StateServer
    keepalive: float

    def log_message(self, fmt: str, *args) -> None:   # тихо: запросы раз в доли секунды
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — имя задаёт http.server
        path = self.path.split("?", 1)[0]
        if path == "/state":
            self._json(200, self.hub.snapshot())
        elif path == "/health":
            h = self.hub.health()
            self._json(200 if h["ok"] else 503, h)
        elif path == "/events":
            self._events()
        else:
            self._json(404, {"error": f"нет такого пути: {path}",
                             "paths": ["/state", "/events", "/health"]})

    def _events(self) -> None:
        sub, snap = self.hub.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.write(sse(0, "hello", {"schema": SCHEMA, "t": snap["t"], "data": snap}))
            self.wfile.flush()
            while True:
                try:
                    msg = sub.q.get(timeout=self.keepalive)
                except queue.Empty:
                    if sub.dropped:
                        return
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if msg is None:                 # сервис останавливается
                    return
                self.wfile.write(sse(*msg))
                self.wfile.flush()
        except OSError:                          # клиент ушёл (BrokenPipe, Reset, Aborted)
            return
        finally:
            self.hub.unsubscribe(sub)


class StateServer:
    """HTTP сервиса состояния на loopback поверх `StateHub`.

    `port=0` — свободный порт (тесты); адрес — `url`. Только loopback:
    сервис отдаёт состояние своей игры, наружу ему незачем."""

    def __init__(self, provider: Provider, host: str = HOST, port: int = PORT,
                 period: float = 0.25, state_every: float = 1.0, keepalive: float = 10.0,
                 clock: Callable[[], float] = time.time):
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError(f"сервис состояния только на loopback, а не {host}")
        self.hub = StateHub(provider, period, state_every, clock)
        handler = type("StateHandler", (_Handler,), {"hub": self.hub, "keepalive": keepalive})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> StateServer:
        self.hub.start()
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="state-http",
                                        daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.hub.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(2.0)


# ---------------------------------------------------------------- без сеанса live

def api_only_provider(client) -> Provider:
    """Провайдер только из Live Client (`live_client.LiveClient`): зрения нет,
    его поля — отказы с причиной. Сеанс live подключает свой провайдер."""
    return lambda: StateInputs(api=client.status())


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m lolstream.state_service` — сервис только на Live Client API."""
    from lolstream.live_client import LiveClient, ddragon_champion_ids

    ap = argparse.ArgumentParser(description="Сервис состояния своей игры (Live Client API)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--period", type=float, default=0.25, help="такт снимка, с")
    a = ap.parse_args(argv)
    client = LiveClient(champion_ids=ddragon_champion_ids()).start()
    srv = StateServer(api_only_provider(client), port=a.port, period=a.period).start()
    print(f"состояние: {srv.url}/state  поток: {srv.url}/events  здоровье: {srv.url}/health",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        client.stop()


if __name__ == "__main__":
    main()
