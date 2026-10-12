# Сервис состояния для coach: контракт выхода (#796, задача 1C)

Код — `lolstream/state_service.py`, тесты — `tests/test_state_service.py`,
пример клиента — `tools/state_client_example.py`. ТЗ —
[mvp-state-service.md](mvp-state-service.md); входы — Live Client API
([live-client.md](live-client.md)), источник кадров
([screen-source.md](screen-source.md)), контекст матча
([match-context.md](match-context.md)), паспорт трансляции
([passport.md](passport.md)), свои / враги ([sides.md](sides.md)).

Сервис отдаёт проверенные факты о **собственной игре пользователя** на
loopback, несколько раз в секунду:

| Путь | Что | Ответ |
| --- | --- | --- |
| `GET /state` | снимок состояния | `application/json`, схема ниже |
| `GET /events` | поток событий | `text/event-stream` (SSE) |
| `GET /health` | жив ли сервис | 200 — снимки идут без сбоя провайдера, 503 — нет; тело `{"ok", "schema", "running", "seq", "age", "error", "subscribers"}` |

Адрес по умолчанию `http://127.0.0.1:8733` (смотрелка — 8732). Сервер
слушает только loopback; адрес не loopback — отказ при запуске.

**Сервис ничего не сводит сам — он переводит в контракт.** У каждого
сведения один хозяин:

| Что | Хозяин | Как попадает в снимок |
| --- | --- | --- |
| состав, жив / мёртв, таймер возрождения, место, ХП, время матча, расхождения источников | контекст матча `lolvision.match_state.MatchState` (приоритет API > HUD > зрение, счётчик `conflicts`) | снимок `MatchView` (`match_view(ms, at)`) |
| ник, уровень, предметы, счёт, золото и статы своего игрока, режим, карта, своя команда | Live Client API (`LiveStatus`) — без конкуренции со зрением | прямо |
| можно ли читателю зрения отвечать (имена над плашками, детектор юнитов) | паспорт трансляции `lolvision.passport.Passport` | `Passport.verdict(reader)` |
| свои / враги / сам и сторона у юнитов и треков | `lolvision.sides` (поля `relation`, `side` записей `UnitRec` / `TrackRec`) | поля записи как есть |
| камера, тип сцены, юниты, треки | кадр сеанса live | `VisionFrame` |

## Версия схемы

Каждый снимок и каждое сообщение потока несут `"schema": 1`. Смена имени,
типа или смысла поля — новая версия; новые поля в объектах сервиса — тоже
(схема закрытая: `additionalProperties: false`). Исключение — записи
юнитов и треков и пункты паспорта: это `to_dict()` записей lolvision как
есть, их поля версионируются там (`records.SCHEMA`, docs/passport.md), а
схема сервиса проверяет только поля, на которые он опирается. Клиент,
увидев незнакомую версию, должен отказаться читать, а не угадывать.

Машинная схема — `state_service.snapshot_schema()` (JSON Schema 2020-12);
проверка без зависимостей — `state_service.check_snapshot(snapshot)` →
список ошибок (пусто — снимок по контракту).

## Факт: значение или отказ с причиной

Каждое значение снимка — **факт**:

```json
{"value": 337.3, "source": "api", "age": 0.5, "reason": null}
{"value": null,  "source": null,  "age": null, "reason": "Live Client: no_game (нет соединения: …)"}
```

| Поле | Тип | Смысл |
| --- | --- | --- |
| `value` | любой / `null` | значение; `null` — значения нет |
| `source` | `api` / `minimap` / `screen` / `hud` / `inferred` / `null` | откуда значение; `null` только у отказа |
| `age` | число, с / `null` | давность наблюдения относительно `t` снимка; `null` у отказа или когда давность не определена |
| `reason` | строка / `null` | почему значения нет; у значения — `null` |

Инвариант (проверяется схемой): `value ≠ null` ⇔ `source ≠ null` и
`reason = null`; `value = null` ⇔ непустой `reason`. **Молчаливых подстановок
нет**: нет входа — `null` и почему.

| `source` | Что значит |
| --- | --- |
| `api` | Live Client Data API своей игры (`127.0.0.1:2999`) |
| `minimap` | иконка на миникарте (зрение) |
| `screen` | основной вид кадра: детектор юнитов, имя над плашкой, цвет полоски, модель камеры, тип сцены |
| `hud` | панели HUD (состав, жив/мёртв, таймер) |
| `inferred` | выведено: память и правила контекста матча, паспорт, сторона по правилу sides.py |

**Давность.** Факты контекста матча и кадра: `age` = задержка (сколько
прошло от момента кадра до снимка, `sources.match.age` / `sources.vision.age`)
плюс возраст наблюдения в контексте. Факты API: от получения снимка API.
Значения даны **как их отдал хозяин**: время матча и таймер возрождения
контекст матча уже приводит к моменту кадра (`MatchState.game_clock()`,
`ChampCtx.respawn_in`); сервис их не двигает. Момент наблюдения в
unix-времени — `t − age`.

Единицы: время, давность, таймеры — секунды; `u, v` — доли миникарты (0..1,
`u` вправо, `v` вниз, как у рамки камеры); `x, y` — доли кадра; ХП — доля
0..1; команды — `ORDER` (синие, низ-лево) / `CHAOS`.

## Снимок

Верхний уровень (все поля есть всегда):

| Поле | Тип | Смысл |
| --- | --- | --- |
| `schema` | 1 | версия |
| `seq` | целое | номер снимка, растёт на 1 с каждым тактом сервиса |
| `t` | число | unix-время сборки снимка; от него все `age` |
| `error` | строка / `null` | провайдер состояния упал: тогда все отказы несут эту причину |
| `sources` | объект | состояние входов (ниже) |
| `passport` | факт | паспорт трансляции |
| `match` | объект фактов | матч |
| `me` | объект фактов | свой игрок |
| `champions` | факт: массив | десять чемпионов |
| `enemies` | факт: массив | враги с местом и давностью |
| `camera` | объект фактов | камера |
| `scene` | факт | тип сцены |
| `units` | факт: массив | юниты на экране, последний кадр |
| `tracks` | факт: массив | треки юнитов (постоянные id) |
| `conflicts` | факт | расхождения источников за матч (MatchState) |

### `sources` — входы

Не факты, а состояния входов; `reason` непуст, когда вход не в порядке.

| Поле | Тип | Смысл |
| --- | --- | --- |
| `frames.state` | `absent` / `ok` / `paused` / `minimized` / `no_window` / `no_frame` / `ended` | последний ответ источника кадров (`frame_source.Status`); `absent` — не подключён |
| `frames.reason` | строка / `null` | почему кадра нет |
| `frames.kind` | `screen` / `youtube` / `file` / `null` | какой источник |
| `frames.backend`, `frames.backend_note` | строка / `null` | чем снимается кадр (dxcam / mss) и почему так |
| `frames.fps` | число / `null` | запрошенные кадры в секунду |
| `frames.size` | `[W, H]` / `null` | размер последнего кадра |
| `frames.last_frame_age` | с / `null` | давность последнего снятого кадра |
| `vision.state` | `absent` / `ok` | есть ли обработанный кадр |
| `vision.frame`, `vision.stream_t` | целое, с / `null` | номер кадра сеанса и его время по шкале потока |
| `vision.age` | с / `null` | задержка кадра; входит в `age` фактов кадра |
| `vision.session` | строка / `null` | состояние сеанса live («идёт», «сбой»…) |
| `match.state` | `absent` / `ok` | подключён ли контекст матча |
| `match.stream_t`, `match.age` | с / `null` | момент контекста (время потока) и задержка до снимка |
| `match.roster` | `api` / `hud` / `null` | откуда состав в контексте |
| `match.api_live` | bool / `null` | `MatchState.api_live()`: свежее ли наблюдение API (иначе жив/мёртв снова по HUD) |
| `match.api_refused` | строка / `null` | `MatchState.api_refused`: почему последний снимок API не принят (не 10, не 5/5, повторы) |
| `api.state` | `absent` / `no_game` / `loading` / `in_game` / `ended` / `error` | состояние Live Client (`live_client.State`); `absent` — не подключён |
| `api.reason` | строка / `null` | причина состояния |
| `api.age`, `api.game_age` | с / `null` | давность последнего опроса и последнего снимка игры |

### `passport` — паспорт трансляции

Факт: `value` — `Passport.to_dict()` как есть, `source = "inferred"`; паспорт
не подключён — отказ «паспорт трансляции не подключён к сервису».

| Поле `value` | Смысл |
| --- | --- |
| `items` | пункты паспорта: `source`, `resolution`, `game_rect`, `minimap`, `view` (`spectator` / `player` / `unknown`), `api`, `scene`, `roster`, `assets`; каждый — `{"value", "source", "how", "t"}` (источник пункта: source / locator / layout / hud / api / scene / session / assumed / none; `t` — секунды потока) |
| `pending` | пункты, которые надо перемерить (`minimap`, `game_rect`) |
| `disabled` | отключённые читатели: `{"reader", "title", "reason"}` |
| `enabled` | включённые читатели: `camera_truth`, `units`, `names`, `hud_roster`, `minimap_icons`, `sides`, `api_facts` |

Где выход сервиса зависит от читателя, решает `Passport.verdict`:
`names` — `champions[].on_screen` и `units[].name`; `units` — `units` и
`tracks`. Отказ читателя → `null` с его причиной. **Без паспорта** условия
читателей не проверены — эти поля отказ («паспорт трансляции не подключён
к сервису: условия читателя «…» не проверены»). Камера и тип сцены от
паспорта не зависят.

### `match` — матч

| Поле | Тип `value` | Источник | Смысл, отказ |
| --- | --- | --- | --- |
| `time` | число, с | api | время матча на момент кадра — `MatchState.game_clock()`; API нет, пропал (`api_live` = false) или снимок не принят — отказ с причиной |
| `mode` | строка | api | режим (`CLASSIC`, `ARAM`…) |
| `map` | целое | api | номер карты (11 — Ущелье) |
| `side` | `ORDER` / `CHAOS` | api | сторона своего игрока (activePlayer среди allPlayers) |
| `state` | `no_game` / `loading` / `in_game` / `ended` / `error` | api | состояние API (есть и без снимка игры) |

### `me` — свой игрок

`riot_id`, `champion` (Id Data Dragon), `team`, `level`, `gold` (текущее
золото), `stats` (`championStats` как есть: `abilityPower`, `armor`,
`currentHealth`, … числами) — из API. `alive` и `respawn_in` — те же факты,
что у своего чемпиона в `champions` (из контекста матча). Нет API или себя
не нашли — все отказы с этой причиной.

### `champions` — десять чемпионов

Факт: массив по составу контекста матча (ORDER, затем CHAOS), `source` —
откуда состав (`api` / `hud`); контекста нет или он не знает состава —
отказ.

| Поле записи | `value` | Источник | Смысл, отказ |
| --- | --- | --- | --- |
| `champion` | строка | api / hud | Id Data Dragon (источник состава) |
| `team` | `ORDER` / `CHAOS` | api / hud | команда в контексте матча |
| `relation` | `self` / `ally` / `enemy` | api | отношение к своему игроку по API (`relation_to_me`: его чемпион — `self`, его команда — `ally`, иначе `enemy`); своего игрока нет или состав не из API — отказ |
| `riot_id`, `level`, `items`, `scores` | строка, целое, массив id, объект | api | из `allPlayers` по Id чемпиона; без API — отказ |
| `alive` | bool | api / hud / minimap / inferred | `Champ.alive` контекста матча с его источником; не наблюдался — отказ |
| `respawn_in` | с | как у `alive` | `ChampCtx.respawn_in` у мёртвого; живой — отказ «жив (источник)»; мёртв без таймера — отказ |
| `position` | `{"u", "v"}` | minimap / screen / inferred | последнее известное место (`ChampCtx`); не наблюдалось — отказ |
| `map_state` | `visible` / `remembered` / `dead` / `unknown` | inferred | `ChampCtx.state`: виден сейчас / помним место / мёртв / нет данных |
| `on_screen` | `true` | screen | на кадре чемпион с этим именем над плашкой. **`false` не выдаётся**: «не опознан» ≠ «нет на экране». Читатель `names` выключен паспортом (вид игрока и пр.) — отказ с его причиной |
| `hp` | доля 0..1 | screen / inferred | `Champ.hp` контекста с источником (`inferred` — 1.0 после возрождения) |
| `conflicts` | целое (не факт) | — | `Champ.conflicts`: расхождений по этому чемпиону за матч |

### `enemies` — враги с давностью

Факт: массив врагов (`relation = enemy`), у каждого `champion`, `position`,
`map_state`, `alive`, `respawn_in`, `on_screen` — те же факты, что в
`champions`. Давность места — `position.age`. Своя сторона неизвестна —
отказ.

### `camera`, `scene`

| Поле | `value` | Источник | Смысл |
| --- | --- | --- | --- |
| `camera.center` | `{"u", "v"}` | screen | центр камеры, ответ модели камеры (`px, py` строки кадра); не ответила — отказ |
| `camera.spread` | число | screen | разброс ответа модели (доли карты) — неуверенность |
| `scene` | строка | screen | тип сцены: `game`, `shop`, `tab`, … (`scene_kind.CLASSES`); модели нет — отказ |

Нет обработанного кадра (пауза, окно свёрнуто, зрение не подключено) — все
три отказа; что именно с кадром — в `sources.frames`.

### `units` и `tracks` — юниты на экране

`units` — факт: массив юнитов последнего кадра, каждый — `UnitRec.to_dict()`
как есть (`kind`, `team` — сырой ответ детектора «синий = ORDER» по выучке
наблюдателя, `x, y`, `hp`, `score`, `color`), где заменены фактами:

* `name` — Id чемпиона по имени над плашкой (`screen`); читатель `names`
  выключен — отказ с причиной паспорта;
* `relation` (`self` / `ally` / `enemy`, источник `screen` — цвет полоски) и
  `side` (`ORDER` / `CHAOS`, источник `inferred` — правило sides.py) — поля
  записи; `null` → причина записи (`relation_reason`, `side_reason`:
  «вид наблюдателя: своего игрока нет», «цвет вне ожидаемого: yellow», «нет
  команды своего игрока…», «не определялось» — запись прошла мимо
  `sides.assign`). Таблица случаев — [sides.md](sides.md).

`tracks` — факт: массив треков, каждый — `TrackRec.to_dict()` (`id`, `kind`,
`team`, `state` — `seen` / `unseen` / `lost`, `u, v`, `x, y`, `hp`, `age`) с
`relation` и `side` фактами так же.

Читатель `units` выключен паспортом — `units` и `tracks` отказ с его
причиной.

### `conflicts` — расхождения источников

Факт (`source = "inferred"`): `{"total": N, "kinds": {вид: число}}` —
`MatchState.conflicts` как есть, накопительно за матч (сбрасывается с
памятью матча). Виды:

| Вид | Когда (match_state.py) |
| --- | --- |
| `roster` | состав по HUD / миникарте не совпал с составом API (или прочитанный до API — заменён) |
| `alive` | панель HUD «жив / мёртв» против API (вне окна 1,5 с после смены по API) |
| `timer` | таймер панели против API дальше 2 с |
| `minimap_dead` | иконка мёртвого по API на миникарте — назначение не тому |
| `hud_flicker` | без API: «жив» на панели посреди отсчёта таймера |

Вида нет в `kinds` — таких расхождений не было. По чемпиону — поле
`champions[].conflicts`. Растёт — читатели зрения на этой раскладке врут
(сигнал #781).

## Пример снимка

Свой ранкед (журнал `tests/fixtures/live_client/own_game.jsonl`, 5:37):
контекст матча накормлен этим API и подставной миникартой, вид наблюдателя
(паспорт). Из десяти чемпионов и пяти врагов показано по одному-два,
`stats` и пункты паспорта урезаны.

```json
{
 "schema": 1, "seq": 5, "t": 1791807815.083, "error": null,
 "sources": {
  "frames": {"state": "ok", "reason": null, "kind": "screen", "backend": "dxcam",
             "backend_note": null, "fps": 2.0, "size": [1920, 1080], "last_frame_age": 0.2},
  "vision": {"state": "ok", "reason": null, "frame": 7, "stream_t": 123.0, "age": 0.2,
             "session": "идёт"},
  "match": {"state": "ok", "reason": null, "stream_t": 1791807814.883, "age": 0.2,
            "roster": "api", "api_live": true, "api_refused": null},
  "api": {"state": "in_game", "reason": null, "age": 0.2, "game_age": 0.2}
 },
 "passport": {"source": "inferred", "age": null, "reason": null, "value": {
  "items": {
   "source": {"value": "screen", "source": "source", "how": null, "t": 0.0},
   "view": {"value": "spectator", "source": "hud", "how": "…", "t": 0.0},
   "api": {"value": {"state": "in_game", "available": true}, "source": "api", "how": null, "t": 0.0},
   "minimap": {"value": {"found": false, "box": null, "scale": null, "score": null},
               "source": "none", "how": "ещё не искали", "t": 0.0}
  },
  "pending": ["minimap"],
  "disabled": [{"reader": "sides", "title": "свои / враги",
                "reason": "нужен вид игрока, а это вид наблюдателя"}],
  "enabled": ["camera_truth", "units", "names", "hud_roster", "minimap_icons", "api_facts"]
 }},
 "match": {
  "time":  {"value": 337.35, "source": "api", "age": 0.2, "reason": null},
  "mode":  {"value": "CLASSIC", "source": "api", "age": 0.2, "reason": null},
  "map":   {"value": 11, "source": "api", "age": 0.2, "reason": null},
  "side":  {"value": "ORDER", "source": "api", "age": 0.2, "reason": null},
  "state": {"value": "in_game", "source": "api", "age": 0.2, "reason": null}
 },
 "me": {
  "riot_id":    {"value": "Player1#0000", "source": "api", "age": 0.2, "reason": null},
  "champion":   {"value": "Zilean", "source": "api", "age": 0.2, "reason": null},
  "team":       {"value": "ORDER", "source": "api", "age": 0.2, "reason": null},
  "level":      {"value": 5, "source": "api", "age": 0.2, "reason": null},
  "gold":       {"value": 443.9, "source": "api", "age": 0.2, "reason": null},
  "alive":      {"value": false, "source": "api", "age": 0.2, "reason": null},
  "respawn_in": {"value": 9.9, "source": "api", "age": 0.2, "reason": null},
  "stats":      {"value": {"abilityHaste": 10.0, "abilityPower": 36.0, "armor": 39.45},
                 "source": "api", "age": 0.2, "reason": null}
 },
 "champions": {"source": "api", "age": 0.2, "reason": null, "value": [
  {"champion":   {"value": "Darius", "source": "api", "age": 0.2, "reason": null},
   "team":       {"value": "CHAOS", "source": "api", "age": 0.2, "reason": null},
   "relation":   {"value": "enemy", "source": "api", "age": 0.2, "reason": null},
   "riot_id":    {"value": "Player6#0000", "source": "api", "age": 0.2, "reason": null},
   "level":      {"value": 6, "source": "api", "age": 0.2, "reason": null},
   "items":      {"value": [1054, 1001, 3340], "source": "api", "age": 0.2, "reason": null},
   "scores":     {"value": {"kills": 1, "deaths": 0, "assists": 0, "creep_score": 30, "ward_score": 2.25},
                  "source": "api", "age": 0.2, "reason": null},
   "alive":      {"value": true, "source": "api", "age": 0.2, "reason": null},
   "respawn_in": {"value": null, "source": null, "age": null, "reason": "жив (api)"},
   "position":   {"value": {"u": 0.5, "v": 0.5}, "source": "minimap", "age": 0.2, "reason": null},
   "map_state":  {"value": "visible", "source": "inferred", "age": 0.2, "reason": null},
   "on_screen":  {"value": true, "source": "screen", "age": 0.2, "reason": null},
   "hp":         {"value": 0.62, "source": "screen", "age": 0.2, "reason": null},
   "conflicts":  0},
  {"champion":   {"value": "Zilean", "source": "api", "age": 0.2, "reason": null},
   "relation":   {"value": "self", "source": "api", "age": 0.2, "reason": null},
   "…": "…",
   "alive":      {"value": false, "source": "api", "age": 0.2, "reason": null},
   "respawn_in": {"value": 9.9, "source": "api", "age": 0.2, "reason": null},
   "position":   {"value": null, "source": null, "age": null, "reason": "место не наблюдалось"},
   "map_state":  {"value": "dead", "source": "inferred", "age": 0.2, "reason": null},
   "on_screen":  {"value": null, "source": null, "age": null,
                  "reason": "на экране не опознан: имени над плашкой нет в кадре или оно не прочитано"},
   "hp":         {"value": null, "source": null, "age": null, "reason": "ХП не наблюдалось"},
   "conflicts":  0}
 ]},
 "enemies": {"source": "api", "age": 0.2, "reason": null, "value": [
  {"champion":   {"value": "Darius", "source": "api", "age": 0.2, "reason": null},
   "position":   {"value": {"u": 0.5, "v": 0.5}, "source": "minimap", "age": 0.2, "reason": null},
   "map_state":  {"value": "visible", "source": "inferred", "age": 0.2, "reason": null},
   "alive":      {"value": true, "source": "api", "age": 0.2, "reason": null},
   "respawn_in": {"value": null, "source": null, "age": null, "reason": "жив (api)"},
   "on_screen":  {"value": true, "source": "screen", "age": 0.2, "reason": null}}
 ]},
 "camera": {
  "center": {"value": {"u": 0.4, "v": 0.6}, "source": "screen", "age": 0.2, "reason": null},
  "spread": {"value": 0.012, "source": "screen", "age": 0.2, "reason": null}
 },
 "scene": {"value": "game", "source": "screen", "age": 0.2, "reason": null},
 "units": {"source": "screen", "age": 0.2, "reason": null, "value": [
  {"kind": "champion", "team": "CHAOS", "x": 0.5, "y": 0.4, "hp": 0.62, "score": 0.9,
   "color": null,
   "name":     {"value": "Darius", "source": "screen", "age": 0.2, "reason": null},
   "relation": {"value": null, "source": null, "age": null,
                "reason": "вид наблюдателя: своего игрока нет"},
   "side":     {"value": "CHAOS", "source": "inferred", "age": 0.2, "reason": null}}
 ]},
 "tracks": {"source": "screen", "age": 0.2, "reason": null, "value": [
  {"id": "top-minion-1", "kind": "minion", "team": "ORDER", "state": "seen",
   "u": 0.2, "v": 0.3, "x": 0.1, "y": 0.2, "hp": 1.0, "age": 0.0,
   "relation": {"value": null, "source": null, "age": null, "reason": "не определялось"},
   "side":     {"value": null, "source": null, "age": null, "reason": "не определялось"}}
 ]},
 "conflicts": {"value": {"total": 0, "kinds": {}}, "source": "inferred", "age": 0.2, "reason": null}
}
```

`"…": "…"` — пропуск в примере, не поле. Без API: в `match` и `me` — отказы «Live Client API не подключён к
сервису», состав — по HUD (`source = "hud"`), `relation` и `enemies` —
отказ «своя сторона неизвестна». Без кадра: `camera`, `scene`, `units`,
`tracks`, `on_screen` — отказ «нет обработанного кадра…», `sources.frames`
говорит почему (пауза, окно свёрнуто…). Без контекста матча: `champions`,
`enemies`, `conflicts`, `match.time`, `me.alive` — отказ «контекст матча
(MatchState) не подключён к сервису».

## Поток событий: SSE

Выбран **Server-Sent Events**, не WebSocket:

* поток односторонний (сервис → coach); обратного канала контракту не
  нужно — вопросы coach задаёт обычным `GET /state`;
* сервер — стандартная библиотека Python (`http.server`,
  `ThreadingHTTPServer`): SSE — это обычный HTTP-ответ, который пишется
  построчно; WebSocket потребовал бы своей реализации рукопожатия и кадров
  или новой зависимости;
* клиент coach — Python (httpx уже есть): SSE читается построчно из
  потокового ответа (`client.stream("GET", …)` + `aiter_lines()`),
  без библиотеки; пример на одном `http.client` — `tools/state_client_example.py`;
* loopback: прокси, балансировщиков и обрывов по простою нет — главных бед
  SSE; переподключение простое (`retry: 1000`), а свежее состояние после
  разрыва приходит в `hello`;
* в браузере SSE — встроенный `EventSource`: поток можно смотреть без кода.

Формат сообщения:

```
id: 42
event: game
data: {"schema":1,"t":1791807815.383,"data":{…}}

```

`data` — JSON одной строкой: `schema`, `t` (unix-время сервиса), `data`.

| `event` | Когда | `data.data` |
| --- | --- | --- |
| `hello` | первым сообщением после подключения (`id: 0`) | снимок целиком |
| `state` | не чаще раза в `state_every` с (по умолчанию 1 с) | снимок целиком |
| `game` | новое событие Live Client (по `EventID`; новая игра — счёт заново) | `live_client.EventRecord.to_dict()`: `event_id, name, time, data, source` |
| `vision` | новое событие контекста матча (`death` — с `src` у смерти по API, `respawn`, `roster`) | `records.Event.to_dict()` + `source: "inferred"` |
| `source` | сменилось `state` одного из `sources` (`frames`, `vision`, `match`, `api`) | `sources` снимка |

Комментарий `: ping` — раз в `keepalive` с (10 с) без событий. Пропущенное
при разрыве не досылается (`Last-Event-ID` не поддерживается): клиент
синхронизируется по `hello`. Медленный клиент, у которого очередь (256
сообщений) переполнилась, отключается — переподключится и получит `hello`.
События `game` берутся из накопленных событий снимка API, поэтому первое
подключение к уже идущей игре разошлёт и прошлые события матча — у каждого
своё `time`.

## Сборка и сервер (для интегратора)

```python
from lolstream.state_service import (FrameStatus, StateInputs, StateServer, VisionFrame,
                                     build_snapshot, match_view)

snapshot = build_snapshot(inputs: StateInputs, now: float, seq: int = 0) -> dict

Provider = Callable[[], StateInputs]          # зовётся из потока хаба раз в period
srv = StateServer(provider, host="127.0.0.1", port=8733,
                  period=0.25, state_every=1.0, keepalive=10.0).start()
srv.url; srv.hub.snapshot(); srv.close()
```

`StateInputs(session, frames, vision, match, api, passport, vision_events, error)`:

| Поле | Тип | Откуда в сеансе live |
| --- | --- | --- |
| `session` | `str \| None` | подпись состояния сеанса («идёт») |
| `frames` | `FrameStatus(kind, state, reason, backend, backend_note, fps, size, last_ok)` | последний `Frame` источника (`Status.key`, причина) и `describe()`; `last_ok` — unix-время последнего кадра |
| `vision` | `VisionFrame(i, t, at, scene, camera, spread, units, tracks)` | последняя строка кадра: номер, время потока, unix-время обработки, тип сцены, `(px, py)`, `spread`, записи `UnitRec` / `TrackRec` (после `sides.assign`) |
| `match` | `MatchView` = `match_view(session.match, at)` | **под замком сеанса**: `at` — unix-время кадра `match.t` |
| `api` | `LiveStatus` | `LiveClient.status()` |
| `passport` | `Passport` | паспорт сеанса (объект; `to_dict` и `verdict` зовёт сервис) |
| `vision_events` | `Sequence[Event]` | `MatchState.events` (можно накопительно: хаб отсеет повторы) |

Любой вход `None` — его поля отказом. Провайдер должен быть быстрым (взять
последние готовые записи, не считать модели); исключение провайдера —
снимок из отказов с причиной в `error`, хаб работает дальше. Паспорт
читается из потока хаба (`verdict`, `to_dict`), пока сеанс его меняет:
провайдеру безопаснее отдавать копию или звать сборку под замком.

Без сеанса live — `python -m lolstream.state_service`: свой `MatchState`
кормится только API (`status_observation`), зрение — отказы.

## Клиент coach

`tools/state_client_example.py` — тонкий пример на стандартной библиотеке:
`get_state()`, `events()` с переподключением и `to_source_facts(snapshot)` —
короткий список фактов `{id, text, source, observed_at}` под
`coach.analysis_types.SourceFact` (время матча, сторона, свой чемпион и
таймер, враги: мёртв или место с давностью). Отказы в факты не
превращаются. В coach его место — `Observer._analyze_once()`
(`facts=` в `analyze`) и `visual_context()`; сам coach этой задачей не
меняется.

## Не проверено

* Ручка на живом сеансе по своему экрану: провайдер из `lolstream/live.py`
  делает интегратор.
* `python -m lolstream.state_service` против настоящего порта 2999.
* Нагрузка на живом сеансе не замерена. На синтетике (журнал своей игры +
  30 юнитов и 30 треков, CPU, 2026-10-12): снимок — 43 КБ JSON, сборка
  `build_snapshot` — 0,4 мс, проверка `check_snapshot` — 9 мс (в такт
  сервиса она не входит).
