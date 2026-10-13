"""Паспорт трансляции: чего ждать от кадров этого сеанса и кому можно отвечать (#781, #799).

Зачем. Читатели (камера по миникарте, детектор юнитов, имена над плашками,
состав по HUD, иконки миникарты, свои/враги, факты Live Client API) писаны
под определённые условия: вид наблюдателя, найденную миникарту, банк имён,
живой API. Вне своих условий они не молчат, а выдают правдоподобный мусор:
на виде игрока состав решился по заставке, и Brand был «Ivern» в 100% кадров
(docs/pitfalls.md, 0.3, #786); по прямоугольнику раскладки истина камеры
у стримера с крупной миникартой съезжала к углу (#785). Паспорт собирает
условия сеанса в одном месте, каждый читатель объявляет, что ему нужно, и
ответ «можно ли отвечать» — да или нет с причиной. Не выполнено — выход
читателя `null` + причина, а не запасной путь.

Пункты паспорта (`Item`: значение, источник, как получено, когда):

* `source` — откуда кадры: screen / youtube / file (`FrameSource.describe`);
* `resolution` — (W, H) кадра;
* `game_rect` — прямоугольник игры в кадре, пиксели (x0, y0, x1, y1);
* `minimap` — найдена ли миникарта, прямоугольник (доли кадра), масштаб и
  совпадение с эталоном (minimap_locate.Locator);
* `view` — spectator / player / **unknown**: «неизвестен» — честное
  значение, пока раскладка не решена или не распознана. По HUD вид решается
  только по положительным признакам (lolstream/live_champions.py,
  `read_kind`): «портретов наблюдателя не видно» — не вид игрока, а
  unknown (или прежний вид, если он уже решён);
* `api` — состояние Live Client API и доступны ли его факты;
* `scene` — тип сцены последнего кадра (сигнал, не условие читателей);
* `roster` — откуда известен состав: api / hud или неизвестен;
* `assets` — какие модели и банки есть у сеанса (unit_net, names_bank,
  minimap_model);
* `spell_panel` — найдена ли панель умений своего HUD (вид игрока), её
  масштаб и совпадение с эталоном (lolvision/hud_spells.py, #806).

Пересчёт — по сигналу, а не на каждом кадре. Сеанс сообщает факт
(`set_*`, `check_minimap`), паспорт отвечает списком изменений (`Change`:
пункты и читатели, которые включились или отключились) — их сеанс кладёт в
поток событий и в note. Пункты, которые надо перемерить, лежат в `pending`:
сменился размер кадра — миникарту и прямоугольник игры искать заново; сцена
вернулась в игру после перерыва — проверить, на месте ли миникарта.

Паспорт — только данные: без сети, захвата и моделей; всё мерит сеанс.

Пример:
    p = Passport.build(source.describe(), t=0.0,
                       assets={"unit_net": True, "names_bank": False})
    p.set_view("player", t, how="HUD: портреты наблюдателя в 0, панель игрока в 5 из 8 чтений")
    for ch in p.set_api("in_game", available=True, t=t):
        note(ch.text)
    ok, why = p.verdict("names")         # (False, "нужен вид наблюдателя, …")
    row["names"] = p.gate("names", names)  # {"value": None, "reason": …}
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from lolvision.minimap_locate import AGREE, MIN_SCORE

VIEWS = ("spectator", "player", "unknown")
UNKNOWN = "unknown"
# Модели и банки сеанса, о которых спрашивают читатели. Чего нет в паспорте —
# того «нет сведений», и читатель отказывает, а не надеется.
ASSETS = {
    "unit_net": "детектор юнитов",
    "names_bank": "банк имён над плашками",
    "minimap_model": "модель иконок миникарты",
    "hud_stats": "образцы цифр полосы статистики HUD",
}
ROSTER_SOURCES = ("api", "hud")
# Сколько проверок подряд с совпадением ниже MIN_SCORE снимают находку
# миникарты: одна плохая проверка — пинг, оверлей или затемнение, не смена
# раскладки. Столько же уверенных кадров подряд Locator требует для находки.
LOST_AFTER = AGREE


@dataclass(frozen=True)
class Item:
    """Пункт паспорта: значение, откуда оно (`source`), как получено, когда.

    `source`: source — описание источника кадров; locator — поиск
    миникарты по кадру; layout — раскладка по умолчанию; hud — чтения HUD;
    api — Live Client API; scene — модель типа сцены; session — сеанс
    объявил сам; assumed — допущение (сказано в `how`); none — нет сведений.
    `t` — секунды потока, когда пункт получен (None — до первого кадра).
    """
    value: Any
    source: str
    how: str | None = None
    t: float | None = None

    def to_dict(self) -> dict:
        return {"value": _plain(self.value), "source": self.source, "how": self.how,
                "t": None if self.t is None else round(float(self.t), 2)}


@dataclass(frozen=True)
class Change:
    """Что изменилось после сигнала: пункт (`item:<имя>`) или читатель
    (`reader:<имя>`). У читателя old/new — "on" или причина отказа."""
    key: str
    old: Any
    new: Any
    why: str
    t: float | None

    @property
    def text(self) -> str:
        """Строка для note сеанса."""
        return f"{self.key}: {_short(self.old)} → {_short(self.new)} ({self.why})"

    def to_dict(self) -> dict:
        return {"key": self.key, "old": _plain(self.old), "new": _plain(self.new),
                "why": self.why, "t": None if self.t is None else round(float(self.t), 2)}


@dataclass(frozen=True)
class Verdict:
    """Можно ли читателю отвечать: ok, или нет и почему."""
    ok: bool
    reason: str | None = None

    def __iter__(self):
        return iter((self.ok, self.reason))


# ---------------------------------------------------------------- требования

Need = Callable[["Passport"], "str | None"]   # None — выполнено, иначе причина


def need_frame(p: Passport) -> str | None:
    return None if p.items["resolution"].value is not None else "размер кадра неизвестен"


def need_view(*views: str) -> Need:
    """Вид — один из `views`. Неизвестный вид не подходит никому."""
    def check(p: Passport) -> str | None:
        it = p.items["view"]
        if it.value in views:
            return None
        want = " или ".join({"spectator": "наблюдателя", "player": "игрока"}[v] for v in views)
        if it.value == UNKNOWN:
            return f"нужен вид {want}, а вид неизвестен ({it.how or 'не решён'})"
        have = {"spectator": "наблюдателя", "player": "игрока"}[it.value]
        return f"нужен вид {want}, а это вид {have}"
    return check


def need_minimap_or_spectator(p: Passport) -> str | None:
    """Миникарта найдена в кадре — или вид наблюдателя: тогда прямоугольник
    раскладки — та же миникарта (у обучающих роликов масштаб 1.00)."""
    mm = p.items["minimap"]
    if mm.value["found"] or p.items["view"].value == "spectator":
        return None
    view = p.items["view"].value
    who = "вид неизвестен" if view == UNKNOWN else "вид игрока"
    return (f"миникарта не найдена ({mm.how or 'не искали'}), а {who}: "
            "по прямоугольнику раскладки она вырезана мимо настоящей")


def need_asset(name: str) -> Need:
    def check(p: Passport) -> str | None:
        have = p.items["assets"].value
        if name not in have:
            return f"нет сведений, есть ли {ASSETS[name]}"
        return None if have[name] else f"нет: {ASSETS[name]}"
    return check


def need_spell_panel(p: Passport) -> str | None:
    """Панель умений своего HUD найдена в кадре (lolvision.hud_spells)."""
    it = p.items["spell_panel"]
    if it.value["found"]:
        return None
    return f"панель умений не найдена ({it.how or 'не искали'})"


def need_roster(p: Passport) -> str | None:
    it = p.items["roster"]
    return None if it.value is not None else f"состав неизвестен ({it.how or 'не прочитан'})"


def need_api(p: Passport) -> str | None:
    it = p.items["api"]
    if it.value["available"]:
        return None
    state = it.value["state"] or "не подключён"
    return f"Live Client API: {state}" + (f" — {it.how}" if it.how else "")


@dataclass(frozen=True)
class Reader:
    """Читатель и его требования к паспорту — по порядку, первый невыполненный
    и есть причина отказа."""
    name: str
    title: str
    needs: tuple[Need, ...]


READERS: dict[str, Reader] = {r.name: r for r in (
    # Истина камеры по рамке на миникарте (live._emit, «truth»).
    Reader("camera_truth", "камера по миникарте",
           (need_frame, need_minimap_or_spectator)),
    # Маска детектора юнитов своя у каждого вида (#786): по чужой маске он
    # видит 32% кадра у игрока или HUD игрока как сцену.
    Reader("units", "детектор юнитов",
           (need_frame, need_asset("unit_net"), need_view("spectator", "player"))),
    # Над плашкой у наблюдателя Volibear — имя чемпиона, у игрока — ник.
    Reader("names", "имена над плашками",
           (need_view("spectator"), need_asset("names_bank"), need_roster)),
    # Портреты HUD наблюдателя; на виде игрока состав — из API (2B).
    Reader("hud_roster", "состав по HUD",
           (need_frame, need_view("spectator"))),
    # Иконки назначаются только среди известных десяти (2B).
    Reader("minimap_icons", "иконки миникарты",
           (need_frame, need_asset("minimap_model"), need_minimap_or_spectator, need_roster)),
    # Свои / враги / сам (#790): цвета относительны только у вида игрока,
    # своя сторона — из API.
    Reader("sides", "свои / враги",
           (need_view("player"), need_api)),
    Reader("api_facts", "факты Live Client API", (need_api,)),
    # Готовность своих заклинаний D/F по панели умений (#806): панель есть
    # только у вида игрока, слоты — по найденной панели и её масштабу.
    Reader("hud_spells", "готовность заклинаний по HUD",
           (need_frame, need_view("player"), need_spell_panel)),
    # CS и время матча по полосе статистики вида игрока (#816): полоса стоит
    # на одном месте при любом масштабе HUD, искать её не нужно.
    Reader("hud_stats", "CS и время по полосе HUD",
           (need_frame, need_view("player"), need_asset("hud_stats"))),
    # Добивания (#816): рост CS по полосе против смертей миньонов трекера.
    Reader("last_hits", "добивания по CS и смертям крипов",
           (need_frame, need_view("player"), need_asset("hud_stats"), need_asset("unit_net"))),
)}


# ---------------------------------------------------------------- паспорт


class Passport:
    """Ожидания сеанса. Строится в начале (`build`), меняется по сигналам."""

    def __init__(self, items: dict[str, Item], pending: set[str] | None = None):
        self.items = items
        self.pending: set[str] = set(pending or ())
        self._low = 0           # проверок миникарты подряд с совпадением ниже порога
        self._game: bool | None = None   # последний тип сцены — игра?
        self._buf: list[Change] = []

    @classmethod
    def build(cls, source: Mapping, t: float | None = None,
              game_rect: tuple[int, int, int, int] | None = None,
              assets: Mapping[str, bool] | None = None) -> Passport:
        """Паспорт по `FrameSource.describe()` (kind, size) в начале сеанса.

        `game_rect` — прямоугольник игры в кадре, если сеанс его знает. Нет —
        кадр целиком: у экрана кадр и есть клиентская область окна игры, у
        ролика это допущение (раскладку стрима с рамкой и вебкамерой никто не
        разбирал) — так и сказано в `how`.
        """
        kind = source.get("kind")
        size = source.get("size")
        size = None if size is None else (int(size[0]), int(size[1]))
        items = {
            "source": Item(kind, "source", _describe_how(source), t),
            "resolution": Item(size, "source", None if size else "источник ещё не знает", t),
            "game_rect": _rect_item(kind, size, game_rect, t),
            "minimap": Item(_mm(False), "none", "ещё не искали", t),
            "view": Item(UNKNOWN, "none", "раскладка не решена", t),
            "api": Item({"state": None, "available": False}, "none", "клиент API не подключён", t),
            "scene": Item(None, "none", "кадров ещё не было", t),
            "roster": Item(None, "none", "не прочитан", t),
            "assets": Item({k: bool(v) for k, v in (assets or {}).items()}, "session", None, t),
            "spell_panel": Item(_panel(False), "none", "ещё не искали", t),
        }
        for k in (assets or {}):
            if k not in ASSETS:
                raise ValueError(f"неизвестный ресурс сеанса: {k}")
        return cls(items, {"minimap"})

    # ---------- читатели ----------

    def verdict(self, reader: str) -> Verdict:
        """Можно ли читателю `reader` отвечать по этому паспорту."""
        for need in READERS[reader].needs:
            why = need(self)
            if why is not None:
                return Verdict(False, why)
        return Verdict(True)

    def gate(self, reader: str, value: Any) -> dict:
        """Выход читателя: значение, если ему можно отвечать, иначе null + причина."""
        v = self.verdict(reader)
        return {"value": value, "reason": None} if v.ok else {"value": None, "reason": v.reason}

    def disabled(self) -> dict[str, str]:
        """Отключённые читатели и причины."""
        out = {}
        for name in READERS:
            v = self.verdict(name)
            if not v.ok:
                out[name] = v.reason
        return out

    # ---------- сигналы ----------

    def set_frame_size(self, size: tuple[int, int], t: float | None) -> list[Change]:
        """Размер кадра (у экрана меняется вместе с окном игры). Сменился —
        находка миникарты и прямоугольник игры были в чужой сетке: снимаются
        и уходят в `pending`."""
        size = (int(size[0]), int(size[1]))
        old = self.items["resolution"].value
        if size == old:
            return []
        changes = self._watch()
        self._set("resolution", Item(size, "source", None, t), changes, "размер кадра")
        if old is None and self.items["game_rect"].value is None:
            kind = self.items["source"].value
            self._set("game_rect", _rect_item(kind, size, None, t), changes, "размер кадра")
        elif old is not None:
            why = f"размер кадра сменился: {old[0]}x{old[1]} → {size[0]}x{size[1]}"
            kind = self.items["source"].value
            self._set("game_rect", _rect_item(kind, size, None, t), changes, why)
            self._set("minimap", Item(_mm(False), "none", why + ", искать заново", t),
                      changes, why)
            self._set("spell_panel", Item(_panel(False), "none", why + ", искать заново", t),
                      changes, why)
            self._low = 0
            self.pending |= {"minimap", "game_rect"}
        return self._done(changes, t)

    def set_game_rect(self, rect: tuple[int, int, int, int], t: float | None,
                      how: str, source: str = "session") -> list[Change]:
        changes = self._watch()
        self._set("game_rect", Item(tuple(int(v) for v in rect), source, how, t), changes,
                  how)
        self.pending.discard("game_rect")
        return self._done(changes, t)

    def set_minimap(self, box: tuple[float, float, float, float], scale: float,
                    score: float, t: float | None,
                    how: str = "найдена в кадре (minimap_locate)") -> list[Change]:
        """Миникарта найдена (Locator принял прямоугольник)."""
        changes = self._watch()
        val = _mm(True, box, scale, score)
        self._set("minimap", Item(val, "locator", how, t), changes, how)
        self._low = 0
        self.pending.discard("minimap")
        return self._done(changes, t)

    def check_minimap(self, score: float, t: float | None) -> list[Change]:
        """Перепроверка найденной миникарты: совпадение с эталоном в том же
        прямоугольнике. Звать на игровом кадре — на загрузке или в клиенте
        совпадение низкое у любой раскладки. LOST_AFTER проверок подряд ниже
        MIN_SCORE — миникарта потеряна (раскладка сменилась, закрыло
        оверлеем); одна низкая — оставляет находку и просит ещё проверку."""
        changes = self._watch()
        mm = self.items["minimap"]
        if not mm.value["found"]:
            return []
        score = round(float(score), 3)
        if score >= MIN_SCORE:
            self._low = 0
            self.pending.discard("minimap")
            # Совпадение — значение пункта, но не событие: меняется каждую проверку.
            self.items["minimap"] = Item({**mm.value, "score": score}, mm.source, mm.how, t)
            return []
        self._low += 1
        if self._low < LOST_AFTER:
            self.pending.add("minimap")
            return []
        why = f"совпадение упало до {score:g} (< {MIN_SCORE:g}) {self._low} проверок подряд"
        self._set("minimap", Item(_mm(False), "locator", why + ", искать заново", t),
                  changes, why)
        self._low = 0
        self.pending.add("minimap")
        return self._done(changes, t)

    def minimap_missing(self, t: float | None, how: str) -> list[Change]:
        """Поиск миникарты сдался (например, LOCATE_MAX кадров без находки):
        остаётся прямоугольник раскладки, читатели вне вида наблюдателя
        отказывают."""
        changes = self._watch()
        self._set("minimap", Item(_mm(False), "layout", how, t), changes, how)
        self.pending.discard("minimap")
        return self._done(changes, t)

    def set_spell_panel(self, found: bool, t: float | None, how: str,
                        box: tuple[float, float, float, float] | None = None,
                        scale: float | None = None, score: float | None = None) -> list[Change]:
        """Панель умений своего HUD: найдена (место — доли кадра, масштаб к
        эталону, совпадение) или потеряна / не найдена (`how` — почему)."""
        changes = self._watch()
        val = _panel(found, box, scale, score)
        self._set("spell_panel", Item(val, "hud" if found else "none", how, t), changes, how)
        return self._done(changes, t)

    def set_view(self, view: str | None, t: float | None, how: str,
                 source: str = "hud") -> list[Change]:
        """Вид трансляции. None — не решён: в паспорте `unknown`."""
        view = UNKNOWN if view is None else view
        if view not in VIEWS:
            raise ValueError(f"вид {view!r}, а бывает {VIEWS}")
        changes = self._watch()
        src = "none" if view == UNKNOWN and source == "hud" else source
        self._set("view", Item(view, src, how, t), changes, how)
        return self._done(changes, t)

    def set_api(self, state: str | None, available: bool, t: float | None,
                reason: str | None = None) -> list[Change]:
        """Состояние Live Client API (`LiveStatus.state.value`) и доступны ли
        его факты — решает сеанс (обычно state in_game). None — клиента нет."""
        changes = self._watch()
        val = {"state": state, "available": bool(available)}
        how = reason or (None if state is not None else "клиент API не подключён")
        self._set("api", Item(val, "api" if state is not None else "none", how, t), changes,
                  _api_why(self.items["api"].value, val, reason))
        return self._done(changes, t)

    def set_scene(self, kind: str | None, t: float | None, game: bool) -> list[Change]:
        """Тип сцены кадра (`game` — это игровой кадр). Событие — только смена
        типа. Возврат в игру после не-игры (загрузка, клиент, заставка) —
        раскладка могла смениться: найденную миникарту проверить заново."""
        old = self.items["scene"]
        if old.value == kind:
            return []
        changes = self._watch()
        back = game and self._game is False
        self._game = bool(game)
        self._set("scene", Item(kind, "scene", "игра" if game else "не игра", t), changes,
                  "тип сцены сменился")
        if back and self.items["minimap"].value["found"]:
            self.pending.add("minimap")
        return self._done(changes, t)

    def set_roster(self, source: str | None, t: float | None, how: str | None = None
                   ) -> list[Change]:
        """Откуда известен состав: api / hud, None — неизвестен."""
        if source is not None and source not in ROSTER_SOURCES:
            raise ValueError(f"источник состава {source!r}, а бывает {ROSTER_SOURCES}")
        changes = self._watch()
        self._set("roster", Item(source, source or "none", how, t), changes,
                  how or ("состав прочитан" if source else "состав потерян"))
        return self._done(changes, t)

    def set_asset(self, name: str, ok: bool, t: float | None) -> list[Change]:
        if name not in ASSETS:
            raise ValueError(f"неизвестный ресурс сеанса: {name}")
        changes = self._watch()
        have = {**self.items["assets"].value, name: bool(ok)}
        self._set("assets", Item(have, "session", None, t), changes,
                  f"{ASSETS[name]}: {'есть' if ok else 'нет'}")
        return self._done(changes, t)

    # ---------- выход ----------

    def to_dict(self) -> dict:
        """Поле `passport` снимка состояния."""
        dis = self.disabled()
        return {"items": {k: it.to_dict() for k, it in self.items.items()},
                "pending": sorted(self.pending),
                "disabled": [{"reader": k, "title": READERS[k].title, "reason": r}
                             for k, r in dis.items()],
                "enabled": [k for k in READERS if k not in dis]}

    # ---------- внутреннее ----------

    def _watch(self) -> dict[str, str]:
        """Снимок решений читателей до сигнала: "on" или причина."""
        return {k: (v.reason if not v.ok else "on")
                for k, v in ((k, self.verdict(k)) for k in READERS)}

    def _set(self, key: str, item: Item, before: dict, why: str) -> None:
        """Записать пункт; смена значения — в изменения сигнала."""
        old = self.items[key]
        self.items[key] = item
        if old.value != item.value:
            self._buf.append(Change(f"item:{key}", old.value, item.value, why, item.t))

    def _done(self, before: dict, t: float | None) -> list[Change]:
        """Изменения пунктов этого сигнала и читатели, решение которых сменилось."""
        out, self._buf = self._buf, []
        after = self._watch()
        # Событие читателя — только включение или отключение: смена причины
        # отказа (другие слова того же «нет») в поток не идёт, она в to_dict.
        for k, old in before.items():
            new = after[k]
            if (old == "on") != (new == "on"):
                why = "требования выполнены" if new == "on" else new
                out.append(Change(f"reader:{k}", old, new, why, t))
        return out


# ---------------------------------------------------------------- помощники


def _mm(found: bool, box=None, scale: float | None = None,
        score: float | None = None) -> dict:
    return {"found": found,
            "box": None if box is None else tuple(round(float(b), 5) for b in box),
            "scale": None if scale is None else round(float(scale), 4),
            "score": None if score is None else round(float(score), 3)}


def _panel(found: bool, box=None, scale: float | None = None,
           score: float | None = None) -> dict:
    return _mm(found, box, scale, score)


def _rect_item(kind: str | None, size: tuple[int, int] | None,
               rect: tuple[int, int, int, int] | None, t: float | None) -> Item:
    if rect is not None:
        return Item(tuple(int(v) for v in rect), "session", "задан сеансом", t)
    if size is None:
        return Item(None, "none", "размер кадра неизвестен", t)
    full = (0, 0, size[0], size[1])
    if kind == "screen":
        return Item(full, "source", "кадр — клиентская область окна игры", t)
    return Item(full, "assumed", "кадр целиком: раскладка стрима не разобрана", t)


def _describe_how(source: Mapping) -> str | None:
    parts = [str(source[k]) for k in ("backend", "title") if source.get(k)]
    if source.get("backend_note"):
        parts.append(str(source["backend_note"]))
    return " · ".join(parts) or None


def _api_why(old: dict, new: dict, reason: str | None) -> str:
    if old["available"] != new["available"]:
        base = "API появился" if new["available"] else "API пропал"
    else:
        base = "состояние API сменилось"
    return base + (f": {reason}" if reason else "")


def _plain(v: Any) -> Any:
    """Значение для JSON: кортежи → списки, словари — поэлементно."""
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _short(v: Any) -> str:
    if v is None:
        return "нет"
    if isinstance(v, dict) and "found" in v:
        return f"×{v['scale']:g}" if v["found"] else "не найдена"
    if isinstance(v, dict) and "available" in v:
        return v["state"] or "не подключён"
    if isinstance(v, tuple):
        return "x".join(str(x) for x in v)
    return str(v)
