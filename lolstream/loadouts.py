"""Паспорт выбора игры (#807): роль, заклинания, руны, предметы всех десяти — для статистики.

Зачем. Нужна статистика «кто что берёт» и детектор нетипичного выбора
(АДК с Телепортом, лесник без Кары, странная ключевая руна): по каждой игре —
строка на игрока, без ников, в `data/loadouts/<дата>.jsonl` (вне Git).
Нетипичное помечает `lolstream.surprises` (правила + база частот).

Откуда. Только Live Client API (источник `api`): `live_client.LiveStatus`
по порядку опросов — из журнала (`tools/loadouts.py`) или в живом сеансе
(`LoadoutRecorder` как наблюдатель `LiveClient.on_status`, live.start_screen).
`GameLog` копит игру по снимкам: состав, роль, заклинания (и когда
улучшились), руны, инвентарь по времени матча; `records()` — строки.

Что API не даёт и как это видно в записи:

* id матча — нет: `game_id` — хэш режима, карты, состава (команда +
  rawChampionName) и минуты начала (`game_id_source: "hash"`); повторный
  прогон того же журнала или живой сеанс той же игры строку не дублирует
  (`write_records`: тот же состав и начало ближе `DUP_START` с);
* патч — нет (`patch: null` с причиной); `ddragon` — версия item.json, по
  которой предметы отнесены к законченным, а не версия игры;
* время покупки — время матча снимка, где предмет впервые виден (опрос
  1–2 Гц: точность ~1 с); проданный и купленный заново — две покупки;
* итог — `GameEnd.Result` своего игрока (Win / Lose); врагу — обратный;
  без своего игрока (наблюдатель) или без GameEnd — null с причиной.

Предметы: `start` — инвентарь на последнем игровом снимке до
`START_WINDOW` с (стартовая покупка); `completed` — законченные по item.json
Data Dragon (`ItemCatalog.kind`: без улучшений дальше и дороже 2000 —
`legendary`, ботинки второго уровня и выше — `boots`) по порядку появления;
`purchases` — все появления предметов; `final` — инвентарь в конце.

Тренировка (PRACTICETOOL) по умолчанию не пишется: это не выбор для
статистики (`skip_modes`).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from lolstream.live_client import (ROLES, GameSnapshot, JournalRow, LiveStatus, PlayerRecord,
                                   State, replay)
from lolvision.match_state import NEW_GAME_BACKSTEP, TEAMS

SCHEMA = 1
BASE_SCHEMA = 1
START_WINDOW = 90.0         # с матча: до этого — стартовая покупка
DUP_START = 180.0           # с: та же игра, если состав тот же и начало ближе
SKIP_MODES = ("PRACTICETOOL",)
LEGENDARY_GOLD = 2000
WHY_NO_PATCH = "Live Client API не отдаёт версию игры"


# ---------------------------------------------------------------- предметы

@dataclass(frozen=True)
class ItemCatalog:
    """item.json Data Dragon одной версии: какие предметы законченные."""
    version: str
    data: Mapping[str, Mapping]

    def kind(self, item_id: int) -> str | None:
        """`legendary` / `boots` или None (компонент, расходник, тринкет, нет в каталоге)."""
        x = self.data.get(str(item_id))
        if x is None:
            return None
        tags = x.get("tags") or []
        if x.get("consumed") or "Trinket" in tags or "Consumable" in tags:
            return None
        if "Boots" in tags:
            return "boots" if (x.get("depth") or 1) >= 2 else None
        gold = (x.get("gold") or {}).get("total") or 0
        return "legendary" if not x.get("into") and gold >= LEGENDARY_GOLD else None

    def name(self, item_id: int) -> str | None:
        x = self.data.get(str(item_id))
        return None if x is None else x.get("name")

    @classmethod
    def load(cls, version: str | None = None,
             root: Path | None = None) -> tuple[ItemCatalog | None, str | None]:
        """Из кэша Data Dragon → (каталог, None) или (None, причина). В сеть не ходит."""
        from lolvision import ddragon
        v = version or ddragon.DEFAULT_VERSION
        try:
            return cls(v, ddragon.DDragon(v, root or ddragon.CACHE).items()), None
        except (FileNotFoundError, KeyError, ValueError) as e:
            return None, f"нет item.json Data Dragon {v}: {e}"


# ---------------------------------------------------------------- одна игра

def _inventory(p: PlayerRecord) -> Counter[int]:
    inv: Counter[int] = Counter()
    for it in p.items:
        inv[it.item_id] += max(1, it.count)
    return inv


def _key(p: PlayerRecord) -> tuple[str, str, str]:
    # Ник — только ключ в памяти, наружу не идёт.
    return (p.riot_id, p.team, p.champion_raw)


@dataclass
class _Player:
    first: PlayerRecord
    last: PlayerRecord
    inv: Counter[int] = field(default_factory=Counter)
    purchases: list[tuple[float, int]] = field(default_factory=list)
    start: list[int] | None = None
    upgrades: list[tuple[float, int, str]] = field(default_factory=list)


class GameLog:
    """Одна игра по снимкам API. `add` — каждый снимок по порядку."""

    def __init__(self) -> None:
        self.players: dict[tuple[str, str, str], _Player] = {}
        self.mode: str | None = None
        self.map_number: int | None = None
        self.first_time: float | None = None    # время матча первого игрового снимка
        self.last_time = 0.0
        self.start_unix: float | None = None
        self.playing = False                    # был снимок с событиями (игра шла)
        self.me: tuple[str, str, str] | None = None
        self.me_reason: str | None = "снимков не было"
        self.end: GameSnapshot | None = None
        self.done = False

    def add(self, snap: GameSnapshot, t_unix: float | None) -> None:
        live = bool(snap.events)
        self.mode, self.map_number = snap.game_mode, snap.map_number
        self.last_time = max(self.last_time, snap.game_time)
        if live:
            if not self.playing and t_unix is not None:
                self.start_unix = t_unix - snap.game_time
            if self.first_time is None:
                self.first_time = snap.game_time
            self.playing = True
        elif self.start_unix is None and t_unix is not None:
            self.start_unix = t_unix - snap.game_time
        me = snap.my_player
        if me is not None:
            self.me, self.me_reason = _key(me), None
        elif self.me is None:
            self.me_reason = snap.me_reason or "своего игрока нет"
        for p in snap.players:
            k = _key(p)
            st = self.players.get(k)
            if st is None:
                st = self.players[k] = _Player(p, p)
            self._spells(st, p, snap.game_time)
            st.last = p
            inv = _inventory(p)
            for item, n in (inv - st.inv).items():
                st.purchases += [(snap.game_time, item)] * n
            st.inv = inv
            if live and snap.game_time <= START_WINDOW:
                st.start = sorted(inv.elements())
        if snap.ended is not None:
            self.end = snap

    @staticmethod
    def _spells(st: _Player, p: PlayerRecord, t: float) -> None:
        old = st.last.spells
        if old is None or p.spells is None:
            return
        for a, b in zip(old, p.spells):
            if a.variant != b.variant:
                st.upgrades.append((t, b.slot, b.variant))

    # --- строки ---

    def game_key(self) -> str:
        comp = sorted(f"{st.first.team}:{st.first.champion_raw}" for st in self.players.values())
        text = f"{self.mode}|{self.map_number}|" + ",".join(comp)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def game_id(self) -> str:
        minute = "?" if self.start_unix is None else str(int(self.start_unix // 60))
        return hashlib.sha1(f"{self.game_key()}|{minute}".encode()).hexdigest()[:16]

    def result(self, team: str) -> tuple[str | None, str | None]:
        if self.end is None:
            return None, "GameEnd не пришёл (запись оборвалась или API пропал)"
        if self.me is None:
            return None, f"итог только у своего игрока: {self.me_reason}"
        res = self.end.ended.data.get("Result") if self.end.ended else None
        if res not in ("Win", "Lose"):
            return None, f"GameEnd.Result незнакомый: {res!r}"
        if team == self.me[1]:
            return res, None
        return ("Lose" if res == "Win" else "Win"), None

    def records(self, catalog: ItemCatalog | None, catalog_reason: str | None = None,
                origin: str = "live") -> list[dict]:
        """Строки по игроку (без ников; `surprises` добавляет surprises.annotate)."""
        start = None if self.start_unix is None else datetime.fromtimestamp(self.start_unix)
        head = {"schema": SCHEMA, "game_id": self.game_id(), "game_id_source": "hash",
                "game_key": self.game_key(),
                "game_start": None if start is None else start.isoformat(timespec="seconds"),
                "game_start_unix": None if self.start_unix is None else round(self.start_unix),
                "mode": self.mode, "map": self.map_number,
                "patch": None, "patch_reason": WHY_NO_PATCH,
                "ddragon": None if catalog is None else catalog.version,
                "duration": round(self.last_time, 1), "complete": self.end is not None,
                "origin": origin, "source": "api"}
        out = []
        for k, st in self.players.items():
            p = st.last
            if self.me is None:
                rel, rel_why = None, f"своего игрока нет: {self.me_reason}"
            else:
                rel = "self" if k == self.me else "ally" if p.team == self.me[1] else "enemy"
                rel_why = None
            res, res_why = self.result(p.team)
            first = st.first
            spells = None if first.spells is None else {
                "families": [x.family for x in first.spells],
                "variants": [x.variant for x in first.spells],
                "final": [x.variant for x in p.spells] if p.spells is not None else None,
                "upgrades": [{"t": round(t, 1), "slot": s, "variant": v}
                             for t, s, v in st.upgrades]}
            out.append({**head,
                        "team": p.team, "relation": rel, "relation_reason": rel_why,
                        "champion": p.champion, "champion_reason": p.champion_reason,
                        "is_bot": p.is_bot,
                        "role": p.role, "role_reason": p.role_reason,
                        "spells": spells, "spells_reason": first.spells_reason,
                        "runes": None if first.runes is None else first.runes.to_dict(),
                        "runes_reason": first.runes_reason,
                        "items": self._items(st, catalog, catalog_reason),
                        "result": res, "result_reason": res_why})
        order = {r: i for i, r in enumerate(ROLES)}
        out.sort(key=lambda r: (TEAMS.index(r["team"]), order.get(r["role"], len(ROLES))))
        return out

    def _items(self, st: _Player, catalog: ItemCatalog | None,
               catalog_reason: str | None) -> dict:
        if st.start is not None:
            start, start_why = st.start, None
        elif self.first_time is not None and self.first_time > START_WINDOW:
            start, start_why = None, f"игра подхвачена с {self.first_time:.0f} с матча"
        else:
            start, start_why = None, "игровых снимков в начале матча нет"
        if catalog is None:
            completed, why = None, catalog_reason or "нет каталога предметов"
        else:
            completed, why = [], None
            for t, item in st.purchases:
                kind = catalog.kind(item)
                if kind is not None:
                    completed.append({"t": round(t, 1), "item": item,
                                      "name": catalog.name(item), "kind": kind})
        return {"start": start, "start_reason": start_why,
                "completed": completed, "completed_reason": why,
                "purchases": [{"t": round(t, 1), "item": i} for t, i in st.purchases],
                "final": sorted(st.inv.elements())}


# ---------------------------------------------------------------- запись

def _same_game(a: Mapping, b: Mapping) -> bool:
    if a.get("game_key") != b.get("game_key"):
        return False
    sa, sb = a.get("game_start_unix"), b.get("game_start_unix")
    return sa is None or sb is None or abs(sa - sb) <= DUP_START


def read_records(paths: Iterable[Path]) -> list[dict]:
    """Строки паспортов из JSONL; битые строки пропускаются (файл мог дописываться)."""
    out = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        for line in text.splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def write_records(records: Sequence[dict], out_dir: Path) -> tuple[Path | None, int]:
    """Дописать строки игры в `<out_dir>/<дата начала>.jsonl`; та же игра уже
    есть — не пишет. → (файл, сколько строк записано)."""
    if not records:
        return None, 0
    head = records[0]
    day = (head.get("game_start") or datetime.now().isoformat())[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day}.jsonl"
    if any(_same_game(head, r) for r in read_records([path])):
        return path, 0
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, len(records)


Annotate = Callable[[list[dict]], None]


class LoadoutRecorder:
    """Наблюдатель `LiveStatus`: копит игру, по концу (`ended`) пишет строки.

    `out_dir` None — не пишет, только копит готовые игры в `games` (журналы,
    тесты). `annotate` — пометить сюрпризы (surprises.annotate) перед записью.
    `close()` — сеанс закрыт посреди игры: пишет то, что есть (`complete:
    false`). Вызывается из потока опроса API; ошибки внутри не роняют опрос
    (их ловит LiveClient), последняя — в `error`."""

    def __init__(self, out_dir: Path | None, catalog: ItemCatalog | None = None,
                 catalog_reason: str | None = None, annotate: Annotate | None = None,
                 origin: str = "live", skip_modes: Sequence[str] = SKIP_MODES,
                 log: Callable[[str], None] | None = None):
        self.out_dir, self.catalog, self.catalog_reason = out_dir, catalog, catalog_reason
        self.annotate, self.origin, self.skip_modes = annotate, origin, tuple(skip_modes)
        self.log_fn = log
        self.games: list[list[dict]] = []
        self.written: list[tuple[Path, int]] = []
        self.skipped: list[str] = []
        self.error: str | None = None
        self._game: GameLog | None = None
        self._lock = threading.Lock()

    def observe(self, status: LiveStatus) -> list[dict] | None:
        with self._lock:
            done = None
            g = status.game
            if g is not None and g.players:
                cur = self._game
                if cur is not None and g.game_time < cur.last_time - NEW_GAME_BACKSTEP:
                    if not cur.done:
                        done = self._finish(cur)
                    cur = self._game = None
                if cur is None:
                    cur = self._game = GameLog()
                if not cur.done:
                    cur.add(g, status.game_t)
            cur = self._game
            if status.state is State.ENDED and cur is not None and not cur.done:
                done = self._finish(cur)
            return done

    def close(self) -> list[dict] | None:
        with self._lock:
            cur = self._game
            if cur is None or cur.done:
                return None
            return self._finish(cur)

    def _finish(self, game: GameLog) -> list[dict] | None:
        game.done = True
        if not game.playing:
            return None
        if game.mode in self.skip_modes:
            self.skipped.append(f"режим {game.mode} не пишется")
            return None
        recs = game.records(self.catalog, self.catalog_reason, self.origin)
        if self.annotate is not None:
            self.annotate(recs)
        self.games.append(recs)
        if self.out_dir is not None:
            path, n = write_records(recs, self.out_dir)
            if path is not None:
                self.written.append((path, n))
                if self.log_fn is not None:
                    self.log_fn(f"паспорт выбора: {n} строк → {path}" if n
                                else f"паспорт выбора: игра уже записана в {path}")
        return recs


def records_from_journal(rows: Iterable[JournalRow], champion_ids: Iterable[str] | None,
                         catalog: ItemCatalog | None = None, catalog_reason: str | None = None,
                         annotate: Annotate | None = None, origin: str = "journal",
                         skip_modes: Sequence[str] = SKIP_MODES) -> LoadoutRecorder:
    """Журнал allgamedata → рекордер с готовыми играми в `games` (без записи)."""
    rec = LoadoutRecorder(None, catalog, catalog_reason, annotate, origin, skip_modes)
    for status, _new in replay(rows, champion_ids):
        rec.observe(status)
    rec.close()
    return rec


# ---------------------------------------------------------------- база «типичного»

@dataclass
class Base:
    """Частоты выбора: по чемпиону — роли; по (чемпион, роль) — семьи
    заклинаний (доля игр с семьёй), ключевые руны, первый законченный
    легендарный предмет. Формат файла — `to_dict` (docs/loadouts.md)."""
    source: str                                 # own / riot-match-v5 / …
    source_detail: str = ""
    built: str = ""
    patch: str | None = None
    note: str | None = None
    champions: dict[str, dict] = field(default_factory=dict)
    entries: dict[tuple[str, str], dict] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return sum(c["games"] for c in self.champions.values())

    def champion(self, champ: str) -> dict | None:
        return self.champions.get(champ)

    def entry(self, champ: str, role: str) -> dict | None:
        return self.entries.get((champ, role))

    def to_dict(self) -> dict:
        return {"schema": BASE_SCHEMA, "kind": "loadout_base", "source": self.source,
                "source_detail": self.source_detail, "built": self.built, "patch": self.patch,
                "note": self.note, "games": self.games,
                "champions": {c: {"games": v["games"], "roles": dict(v["roles"])}
                              for c, v in sorted(self.champions.items())},
                "entries": [{"champion": c, "role": r, "games": v["games"],
                             "spells": dict(v["spells"]), "keystones": dict(v["keystones"]),
                             "first_item": {str(k): n for k, n in v["first_item"].items()}}
                            for (c, r), v in sorted(self.entries.items())]}

    @classmethod
    def from_dict(cls, d: Any) -> Base:
        if not isinstance(d, Mapping) or d.get("kind") != "loadout_base":
            raise ValueError("не база паспортов выбора (kind != loadout_base)")
        if d.get("schema") != BASE_SCHEMA:
            raise ValueError(f"версия базы {d.get('schema')!r}, ожидалась {BASE_SCHEMA}")
        b = cls(str(d.get("source") or "?"), d.get("source_detail") or "", d.get("built") or "",
                d.get("patch"), d.get("note"))
        for c, v in (d.get("champions") or {}).items():
            b.champions[c] = {"games": int(v["games"]), "roles": Counter(v.get("roles") or {})}
        for e in d.get("entries") or []:
            b.entries[(e["champion"], e["role"])] = {
                "games": int(e["games"]), "spells": Counter(e.get("spells") or {}),
                "keystones": Counter(e.get("keystones") or {}),
                "first_item": Counter({str(k): n for k, n in (e.get("first_item") or {}).items()})}
        return b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple[Base | None, str | None]:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))), None
        except FileNotFoundError:
            return None, f"базы нет: {path}"
        except (ValueError, KeyError, TypeError) as e:
            return None, f"база не читается ({path}): {e}"


def first_legendary(rec: Mapping) -> int | None:
    done = (rec.get("items") or {}).get("completed") or []
    return next((c["item"] for c in done if c.get("kind") == "legendary"), None)


SMALL_SAMPLE = 100


def build_base(records: Iterable[Mapping], source: str = "own", source_detail: str = "",
               patch: str | None = None, now: datetime | None = None) -> Base:
    """Строки паспортов → база. Берутся игроки с чемпионом и ролью (режимы с
    ролями), каждая игра — один раз (game_id, команда, чемпион)."""
    b = Base(source, source_detail, (now or datetime.now()).isoformat(timespec="seconds"), patch)
    seen = set()
    for r in records:
        champ, role = r.get("champion"), r.get("role")
        if not champ or not role:
            continue
        key = (r.get("game_id"), r.get("team"), champ)
        if key in seen:
            continue
        seen.add(key)
        c = b.champions.setdefault(champ, {"games": 0, "roles": Counter()})
        c["games"] += 1
        c["roles"][role] += 1
        e = b.entries.setdefault((champ, role), {"games": 0, "spells": Counter(),
                                                 "keystones": Counter(),
                                                 "first_item": Counter()})
        e["games"] += 1
        sp = r.get("spells") or {}
        for fam in set(sp.get("families") or []):
            e["spells"][fam] += 1
        ks = ((r.get("runes") or {}).get("keystone") or {}).get("name")
        if ks:
            e["keystones"][ks] += 1
        fi = first_legendary(r)
        if fi is not None:
            e["first_item"][str(fi)] += 1
    if b.games < SMALL_SAMPLE:
        b.note = (f"малая выборка: {b.games} игроко-игр (< {SMALL_SAMPLE}); доли не "
                  "статистика, детектор сравнивает только при min_games")
    return b


# ---------------------------------------------------------------- живой сеанс

def live_recorder(out_dir: Path | None = None, base_path: Path | None = None,
                  log: Callable[[str], None] | None = print) -> LoadoutRecorder:
    """Рекордер для live.start_screen: каталог предметов из кэша Data Dragon,
    правила по умолчанию, база `<out_dir>/base.json`, если есть."""
    from lolstream import surprises
    from lolvision import paths
    out = out_dir or paths.LOADOUTS
    catalog, why = ItemCatalog.load()
    rules = surprises.load_rules()
    base, base_why = Base.load(base_path or out / "base.json")
    return LoadoutRecorder(out, catalog, why,
                           annotate=lambda rs: surprises.annotate(rs, rules, base, base_why),
                           origin="live", log=log)
