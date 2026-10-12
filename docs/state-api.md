# Сервис состояния для coach: контракт выхода (#796, задача 1C)

Код — `lolstream/state_service.py`, тесты — `tests/test_state_service.py`,
пример клиента — `tools/state_client_example.py`. ТЗ —
[mvp-state-service.md](mvp-state-service.md); входы — Live Client API
([live-client.md](live-client.md)), источник кадров
([screen-source.md](screen-source.md)), контекст матча и треки
([match-context.md](match-context.md)).

Сервис отдаёт проверенные факты о **собственной игре пользователя** на
loopback, несколько раз в секунду:

| Путь | Что | Ответ |
| --- | --- | --- |
| `GET /state` | снимок состояния | `application/json`, схема ниже |
| `GET /events` | поток событий | `text/event-stream` (SSE) |
| `GET /health` | жив ли сервис | 200 — снимки идут без сбоя провайдера, 503 — нет; тело `{"ok", "schema", "running", "seq", "age", "error", "subscribers"}` |

Адрес по умолчанию `http://127.0.0.1:8733` (смотрелка — 8732). Сервер
слушает только loopback; адрес не loopback — отказ при запуске.

## Версия схемы

Каждый снимок и каждое сообщение потока несут `"schema": 1`. Смена имени,
типа или смысла поля — новая версия; новые поля в объектах — тоже новая
версия (схема закрытая: `additionalProperties: false`). Клиент, увидев
незнакомую версию, должен отказаться читать, а не угадывать.

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
нет**: нет входа — `null` и почему; никакое поле не берёт значение «по
умолчанию».

Источники:

| `source` | Что значит |
| --- | --- |
| `api` | Live Client Data API своей игры (`127.0.0.1:2999`) |
| `minimap` | иконка на миникарте (зрение) |
| `screen` | основной вид кадра: детектор юнитов, имя над плашкой, модель камеры, тип сцены |
| `hud` | панели HUD (состав, жив/мёртв, таймер) |
| `inferred` | выведено: память контекста матча, правило игры (возрождение у фонтана), отношение по стороне |

**Значение — на момент наблюдения.** Сервис не экстраполирует: время матча,
таймер возрождения и место даны такими, какими их видели `age` секунд назад.
Сейчас ≈ `value + age` для времени матча и `value − age` для таймера —
это считает клиент, если ему нужно. Момент наблюдения в unix-времени —
`t − age`.

**Приоритет источников:** API > HUD > зрение. Где API даёт поле (состав,
команда, жив/мёртв, таймер, уровень), значение — из API; зрение — только где
API нет (где на карте, виден ли на экране, камера). Расхождение — в
`conflicts`, а не тихое перетирание.

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
| `passport` | факт | паспорт трансляции (2C) |
| `match` | объект фактов | матч |
| `me` | объект фактов | свой игрок |
| `champions` | факт: массив | десять чемпионов |
| `enemies` | факт: массив | враги с местом и давностью |
| `camera` | объект фактов | камера |
| `scene` | факт | тип сцены |
| `units` | факт: массив | юниты на экране, последний кадр |
| `tracks` | факт: массив | треки юнитов (постоянные id) |
| `conflicts` | объект | расхождения источников в этом снимке |

### `sources` — входы

Не факты, а состояния входов; `reason` непуст, когда вход не в порядке.

| Поле | Тип | Смысл |
| --- | --- | --- |
| `frames.state` | `absent` / `ok` / `paused` / `minimized` / `no_window` / `no_frame` / `ended` | последний ответ источника кадров (`frame_source.Status`); `absent` — источник не подключён |
| `frames.reason` | строка / `null` | почему кадра нет |
| `frames.kind` | `screen` / `youtube` / `file` / `null` | какой источник |
| `frames.backend`, `frames.backend_note` | строка / `null` | чем снимается кадр (dxcam / mss) и почему так |
| `frames.fps` | число / `null` | запрошенные кадры в секунду |
| `frames.size` | `[W, H]` / `null` | размер последнего кадра |
| `frames.last_frame_age` | с / `null` | давность последнего снятого кадра |
| `vision.state` | `absent` / `ok` | есть ли обработанный кадр |
| `vision.frame`, `vision.stream_t` | целое, с / `null` | номер кадра сеанса и его время по шкале потока |
| `vision.age` | с / `null` | задержка: сколько прошло с обработки кадра; входит в `age` всех фактов зрения |
| `vision.view` | `spectator` / `player` / `null` | раскладка экрана (до паспорта 2C — догадка сеанса live) |
| `vision.session` | строка / `null` | состояние сеанса live («идёт», «сбой»…) |
| `api.state` | `absent` / `no_game` / `loading` / `in_game` / `ended` / `error` | состояние Live Client (`live_client.State`); `absent` — не подключён |
| `api.reason` | строка / `null` | причина состояния |
| `api.age`, `api.game_age` | с / `null` | давность последнего опроса и последнего снимка игры |

### `passport` — паспорт трансляции

Факт с объектом паспорта (2C: источник, разрешение, прямоугольник игры,
миникарта, вид, есть ли API). **Сейчас всегда отказ**:
`reason = "паспорт трансляции ещё не построен (2C)"`. Когда 2C даст объект,
он пойдёт в `value` с `source = "inferred"`; его поля — версия схемы 2 или
отдельный раздел этого документа.

### `match` — матч (только API)

| Поле | Тип `value` | Источник | Смысл |
| --- | --- | --- | --- |
| `time` | число, с | api | время матча (`gameData.gameTime`) на момент снимка API |
| `mode` | строка | api | режим (`CLASSIC`, `ARAM`…) |
| `map` | целое | api | номер карты (11 — Ущелье) |
| `side` | `ORDER` / `CHAOS` | api | сторона своего игрока |
| `state` | `no_game` / `loading` / `in_game` / `ended` / `error` | api | состояние API (есть и без снимка игры) |

Отказы: API не подключён — «Live Client API не подключён к сервису»; API
без игры — «Live Client: no_game (причина)»; свой игрок не найден среди
десяти (наблюдатель, совпадений riotId ≠ 1) — у `side` причина из API.

### `me` — свой игрок (только API)

`riot_id` (строка), `champion` (Id Data Dragon), `team` (`ORDER`/`CHAOS`),
`level` (целое), `gold` (число, текущее золото), `alive` (bool),
`respawn_in` (с до возрождения у мёртвого; у живого — отказ «жив (api)»),
`stats` (объект `championStats` как есть: `abilityPower`, `armor`,
`attackSpeed`, `currentHealth`, … числами). Все — `source = "api"`.
Нет API или себя не нашли — все отказы с этой причиной.

### `champions` — десять чемпионов

Факт: массив до десяти записей (порядок — как в API, ORDER потом CHAOS).
Состав — из API (`source = "api"`); API нет — из контекста матча по HUD
(`source = "hud"`); нет обоих — отказ «состава нет: …». Чемпион, которого
зрение ведёт, а в составе API нет, в массив не попадает — это конфликт
`roster`.

| Поле записи | `value` | Источник | Смысл, отказ |
| --- | --- | --- | --- |
| `champion` | строка | api / hud | Id Data Dragon; API не сверил имя — отказ с `champion_reason` |
| `riot_id` | строка | api | ник; без API — отказ |
| `team` | `ORDER` / `CHAOS` | api / hud | команда |
| `relation` | `self` / `ally` / `enemy` | api | отношение к своему игроку; своя сторона неизвестна (нет API, наблюдатель) — отказ |
| `level` | целое | api | уровень; без API — отказ |
| `alive` | bool | api / hud / minimap / screen | API; без API — HUD («мёртв») или зрение («виден» → жив); иначе отказ «зрение не знает» |
| `respawn_in` | с | api / hud | до возрождения у мёртвого; у живого — отказ «жив (…)»; мёртв без таймера — отказ |
| `position` | `{"u", "v"}` | minimap / screen / inferred | последнее известное место (контекст матча); `age` = задержка кадра + возраст места; не наблюдалось — отказ |
| `map_state` | `visible` / `remembered` / `dead` / `unknown` | inferred | состояние в контексте матча: виден сейчас / помним место / мёртв / нет данных |
| `on_screen` | `true` | screen | на основном виде есть чемпион с этим именем над плашкой. **`false` не выдаётся**: «не опознан» ≠ «нет на экране» — отказ с причиной (на виде игрока имена не читаются) |
| `hp` | доля 0..1 | screen | ХП с плашки, только когда `on_screen` |
| `items` | массив целых | api | id предметов |
| `scores` | объект | api | `kills, deaths, assists, creep_score, ward_score` |

### `enemies` — враги с давностью

Факт: массив пяти врагов (`relation = enemy`), у каждого `champion`,
`position`, `map_state`, `alive`, `respawn_in`, `on_screen` — те же факты,
что в `champions`. Давность места — `position.age`: «Lux на миникарте
0.31/0.62, 14 с назад». Своя сторона неизвестна — отказ.

### `camera`, `scene`

| Поле | `value` | Источник | Смысл |
| --- | --- | --- | --- |
| `camera.center` | `{"u", "v"}` | screen | центр камеры, ответ модели камеры (`px, py` строки кадра); не ответила — отказ |
| `camera.spread` | число | screen | разброс ответа модели (доли карты) — неуверенность |
| `scene` | строка | screen | тип сцены: `game`, `shop`, `tab`, … (`scene_kind.CLASSES`); модели нет — отказ |

Нет обработанного кадра (пауза, окно свёрнуто, зрение не подключено) — все
три отказа с причиной; что именно с кадром — в `sources.frames`.

### `units` и `tracks` — юниты на экране

`units` — факт: массив юнитов детектора последнего кадра (`records.UnitRec`):
`kind` (`champion` / `minion` / `ward`), `x, y` (доли кадра), `hp`, `score`
(уверенность) — простые числа, давность — у факта массива; и факты `name`
(Id чемпиона по имени над плашкой, `screen`), `team`, `relation`.

`tracks` — факт: массив треков (`records.TrackRec`): `id`, `kind`, `state`
(`seen` / `unseen` / `lost`), `u, v`, `x, y` (`null` у невидимого), `hp`,
`age` (с от последнего наблюдения трека) и факты `team`, `relation`.

**`team` и `relation` юнитов (свои / враги / сам).** Цвет плашки значит
разное в разных раскладках:

| Вид | Цвет плашки | `team` | `relation` |
| --- | --- | --- | --- |
| наблюдатель | синий / красный → ORDER / CHAOS | `screen` | `ally`/`enemy` по `match.side` (`inferred`); без стороны — отказ |
| игрок, цвет отношения (`SELF`/`ALLY`/`ENEMY`, `unit_bars.TEAM_BY_COLOR["player"]`) | зелёный / синий / красный | по `match.side` (`inferred`); без стороны — отказ | `self`/`ally`/`enemy` (`screen`) |
| игрок, цвет прочитан как ORDER/CHAOS | — | отказ: «на виде игрока цвет плашки относителен…» | отказ: «свои/враги по экрану ещё не определяются (2A, #790)» |
| вид неизвестен | — | отказ | отказ |
| цвет вне ожидаемого (колорблайнд и пр.) | — | отказ «цвет плашки вне ожидаемого» | отказ |

Сейчас детектор live размечает команду по раскладке наблюдателя, поэтому на
виде игрока `relation` юнитов — отказ до задачи 2A; она даст цвет
отношения, и строка 2 таблицы заработает без смены схемы.

### `conflicts` — расхождения источников

`{"count": N, "items": [...]}` — по этому снимку. Элемент:
`{"field", "champion", "used", "values"}`:

| `field` | Когда |
| --- | --- |
| `team` | команда чемпиона по API ≠ по HUD |
| `alive` | API «мёртв», а зрение его видит; или API «жив», а HUD — «мёртв» |
| `respawn_in` | оба «мёртв», таймеры на сейчас расходятся больше 3 с |
| `roster` | в контексте зрения есть чемпион, которого нет в составе API (`champion = null`, в `values` оба состава) |

`used` — чей ответ ушёл в снимок (всегда `api`: у него приоритет);
`values` — значения по источникам. Это сигнал несовпадения ожиданий (#781):
растёт — читатели зрения на этой раскладке врут.

## Пример снимка

Свой ранкед (журнал `tests/fixtures/live_client/own_game.jsonl`, 5:37),
свой — Zilean, мёртв; зрение — подставное; из десяти чемпионов и пяти врагов
показано по одному-два, `stats` урезан. Факт записан в одну строку.

```json
{
 "schema": 1, "seq": 5, "t": 1791807815.383, "error": null,
 "sources": {
  "frames": {"state": "ok", "reason": null, "kind": "screen", "backend": "dxcam",
             "backend_note": null, "fps": 2.0, "size": [1920, 1080], "last_frame_age": 0.3},
  "vision": {"state": "ok", "reason": null, "frame": 7, "stream_t": 123.0, "age": 0.2,
             "view": "spectator", "session": "идёт"},
  "api": {"state": "in_game", "reason": null, "age": 0.5, "game_age": 0.5}
 },
 "passport": {"value": null, "source": null, "age": null, "reason": "паспорт трансляции ещё не построен (2C)"},
 "match": {
  "time":  {"value": 337.35, "source": "api", "age": 0.5, "reason": null},
  "mode":  {"value": "CLASSIC", "source": "api", "age": 0.5, "reason": null},
  "map":   {"value": 11, "source": "api", "age": 0.5, "reason": null},
  "side":  {"value": "ORDER", "source": "api", "age": 0.5, "reason": null},
  "state": {"value": "in_game", "source": "api", "age": 0.5, "reason": null}
 },
 "me": {
  "riot_id":    {"value": "Player1#0000", "source": "api", "age": 0.5, "reason": null},
  "champion":   {"value": "Zilean", "source": "api", "age": 0.5, "reason": null},
  "team":       {"value": "ORDER", "source": "api", "age": 0.5, "reason": null},
  "level":      {"value": 5, "source": "api", "age": 0.5, "reason": null},
  "gold":       {"value": 443.9, "source": "api", "age": 0.5, "reason": null},
  "alive":      {"value": false, "source": "api", "age": 0.5, "reason": null},
  "respawn_in": {"value": 9.85, "source": "api", "age": 0.5, "reason": null},
  "stats":      {"value": {"abilityHaste": 10.0, "abilityPower": 36.0, "armor": 39.45},
                 "source": "api", "age": 0.5, "reason": null}
 },
 "champions": {"source": "api", "age": 0.5, "reason": null, "value": [
  {"champion":   {"value": "Darius", "source": "api", "age": 0.5, "reason": null},
   "riot_id":    {"value": "Player6#0000", "source": "api", "age": 0.5, "reason": null},
   "team":       {"value": "CHAOS", "source": "api", "age": 0.5, "reason": null},
   "relation":   {"value": "enemy", "source": "api", "age": 0.5, "reason": null},
   "level":      {"value": 6, "source": "api", "age": 0.5, "reason": null},
   "alive":      {"value": true, "source": "api", "age": 0.5, "reason": null},
   "respawn_in": {"value": null, "source": null, "age": null, "reason": "жив (api)"},
   "position":   {"value": {"u": 0.7, "v": 0.3}, "source": "minimap", "age": 0.2, "reason": null},
   "map_state":  {"value": "visible", "source": "inferred", "age": 0.2, "reason": null},
   "on_screen":  {"value": true, "source": "screen", "age": 0.2, "reason": null},
   "hp":         {"value": 0.62, "source": "screen", "age": 0.2, "reason": null},
   "items":      {"value": [1054, 1001, 3340], "source": "api", "age": 0.5, "reason": null},
   "scores":     {"value": {"kills": 1, "deaths": 0, "assists": 0, "creep_score": 30, "ward_score": 2.25},
                  "source": "api", "age": 0.5, "reason": null}},
  {"champion":   {"value": "Lux", "source": "api", "age": 0.5, "reason": null},
   "…": "…",
   "position":   {"value": {"u": 0.7, "v": 0.5}, "source": "minimap", "age": 4.2, "reason": null},
   "map_state":  {"value": "remembered", "source": "inferred", "age": 0.2, "reason": null},
   "on_screen":  {"value": null, "source": null, "age": null,
                  "reason": "на экране не опознан: имени над плашкой нет в кадре или оно не прочитано"},
   "hp":         {"value": null, "source": null, "age": null,
                  "reason": "на экране не опознан: имени над плашкой нет в кадре или оно не прочитано"}}
 ]},
 "enemies": {"source": "api", "age": 0.5, "reason": null, "value": [
  {"champion": {"value": "Darius", "source": "api", "age": 0.5, "reason": null},
   "position": {"value": {"u": 0.7, "v": 0.3}, "source": "minimap", "age": 0.2, "reason": null},
   "map_state": {"value": "visible", "source": "inferred", "age": 0.2, "reason": null},
   "alive": {"value": true, "source": "api", "age": 0.5, "reason": null},
   "respawn_in": {"value": null, "source": null, "age": null, "reason": "жив (api)"},
   "on_screen": {"value": true, "source": "screen", "age": 0.2, "reason": null}}
 ]},
 "camera": {
  "center": {"value": {"u": 0.4, "v": 0.6}, "source": "screen", "age": 0.2, "reason": null},
  "spread": {"value": 0.012, "source": "screen", "age": 0.2, "reason": null}
 },
 "scene": {"value": "game", "source": "screen", "age": 0.2, "reason": null},
 "units": {"source": "screen", "age": 0.2, "reason": null, "value": [
  {"kind": "champion", "x": 0.5, "y": 0.4, "hp": 0.62, "score": 0.9,
   "name":     {"value": "Darius", "source": "screen", "age": 0.2, "reason": null},
   "team":     {"value": "CHAOS", "source": "screen", "age": 0.2, "reason": null},
   "relation": {"value": "enemy", "source": "inferred", "age": 0.2, "reason": null}}
 ]},
 "tracks": {"source": "screen", "age": 0.2, "reason": null, "value": [
  {"id": "top-minion-1", "kind": "minion", "state": "seen", "u": 0.2, "v": 0.3,
   "x": 0.1, "y": 0.2, "hp": 1.0, "age": 0.0,
   "team":     {"value": "ORDER", "source": "screen", "age": 0.2, "reason": null},
   "relation": {"value": "ally", "source": "inferred", "age": 0.2, "reason": null}}
 ]},
 "conflicts": {"count": 0, "items": []}
}
```

`"…": "…"` — пропуск в примере, не поле. Без API тот же снимок: в `match`
и `me` все отказы «Live Client API не подключён к сервису», `champions` —
состав с HUD (`source = "hud"`), `relation` и `enemies` — отказ «своя
сторона неизвестна». Без кадра: `camera`, `scene`, `units`, `tracks`,
`position`, `on_screen` — отказ «нет обработанного кадра…», а
`sources.frames` говорит почему (пауза, окно свёрнуто…).

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
* в браузере SSE — встроенный `EventSource`: поток можно смотреть в
  смотрелке или консоли без кода.

Формат сообщения:

```
id: 42
event: game
data: {"schema":1,"t":1791807815.383,"data":{…}}

```

`data` — JSON одной строкой: `schema`, `t` (unix-время сервиса), `data`.
Виды (`event`):

| `event` | Когда | `data.data` |
| --- | --- | --- |
| `hello` | первым сообщением после подключения (`id: 0`) | снимок целиком |
| `state` | не чаще раза в `state_every` с (по умолчанию 1 с) | снимок целиком |
| `game` | новое событие Live Client (по `EventID`; новая игра — счёт заново) | `live_client.EventRecord.to_dict()`: `event_id, name, time, data, source` |
| `vision` | новое событие контекста матча (`death`, `respawn`, `roster`) | `records.Event.to_dict()` + `source: "inferred"` |
| `source` | сменилось `sources.frames.state`, `sources.vision.state` или `sources.api.state` | `sources` снимка |

Комментарий `: ping` — раз в `keepalive` с (10 с) без событий. Пропущенное
при разрыве не досылается (`Last-Event-ID` не поддерживается): клиент
синхронизируется по `hello`. Медленный клиент, у которого очередь (256
сообщений) переполнилась, отключается — переподключится и получит `hello`.
События `game` берутся из накопленных событий снимка API, поэтому первое
подключение к уже идущей игре разошлёт и прошлые события матча — у каждого
своё `time`.

## Сборка и сервер (для интегратора)

```python
from lolstream.state_service import (FrameStatus, StateInputs, StateServer,
                                     VisionFrame, build_snapshot)

snapshot = build_snapshot(inputs: StateInputs, now: float, seq: int = 0) -> dict

srv = StateServer(provider: Callable[[], StateInputs], host="127.0.0.1", port=8733,
                  period=0.25, state_every=1.0, keepalive=10.0).start()
srv.url; srv.hub.snapshot(); srv.close()
```

`StateInputs(session, frames, vision, api, passport, vision_events, error)`:
`session` — подпись состояния сеанса live; `frames` — `FrameStatus`
(последний ответ источника кадров: `kind, state = Status.key, reason,
backend, backend_note, fps, size, last_ok` — unix-время последнего кадра);
`vision` — `VisionFrame` (последний обработанный кадр: `i, t, at` —
unix-время обработки, `view, scene, camera, spread, units, tracks, ctx` —
записи `records.py`; `None` у списка — читатель не работал); `api` —
`LiveStatus` от `LiveClient.status()`; `passport` — объект паспорта 2C;
`vision_events` — `MatchState.events` (можно накопительно, хаб отсеет
повторы). Любой вход `None` — соответствующие поля отказом.

Провайдер зовётся из потока хаба раз в `period` с и должен быть быстрым и
потокобезопасным (взять последние готовые записи под замком сеанса, не
считать модели). Исключение провайдера — снимок из отказов с причиной в
`error`, хаб работает дальше.

Без сеанса live — `python -m lolstream.state_service`: сервис только на Live
Client API (зрение — отказы).

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

* Ручка на живом сеансе по своему экрану: сервис ещё не подключён к
  `lolstream/live.py` (провайдер делает интегратор).
* Нагрузка на живом сеансе не замерена. На синтетике (журнал своей игры +
  30 юнитов и 30 треков, CPU, 2026-10-12): снимок — 38 КБ JSON, сборка
  `build_snapshot` — 0,33 мс, проверка `check_snapshot` — 7,5 мс (в такт
  сервиса она не входит).
