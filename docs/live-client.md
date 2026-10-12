# Live Client Data API своей игры (#795, задача 1B)

Модуль — `lolstream/live_client.py`, тесты — `tests/test_live_client.py`.
ТЗ — [mvp-state-service.md](mvp-state-service.md), ловушки источника —
[pitfalls.md](pitfalls.md) (0.8, 5.1).

## Что даёт API

Клиент игры во время матча отвечает на `https://127.0.0.1:2999/liveclientdata/`
(самоподписанный сертификат Riot; проверка отключена только для 127.0.0.1,
прокси из окружения не используется). Читается `allgamedata` — всё сразу:

| Поле | Что берём | Запись |
| --- | --- | --- |
| `gameData` | время матча (с), режим, номер карты | `GameSnapshot.game_time`, `game_mode`, `map_number` |
| `allPlayers[]` | чемпион (Id Data Dragon), команда ORDER/CHAOS, riotId, уровень, `isDead`, `respawnTimer`, предметы (id, слот, число), счёт (K/D/A, CS, варды), позиция, бот, скин, заклинания призывателя (`summonerSpells`, #804) | `PlayerRecord`, `SpellRecord` |
| `activePlayer` | riotId, уровень, золото, `championStats` как есть | `ActivePlayerRecord`; себя находим среди десяти по riotId → `GameSnapshot.me` |
| `events.Events[]` | `EventID`, имя, время и прочие поля (KillerName, VictimName, Assisters, DragonType, Stolen, Result…) | `EventRecord` |

Чего API не даёт: позиций на карте, кулдаунов умений и заклинаний, видимости
— это остаётся зрению. Все записи — `dataclass` с `to_dict()`, источник `"api"`.

**Чемпион.** `championName` локализован (у русского клиента — «Зилеан»), поэтому
берётся `rawChampionName` (`game_character_displayname_Zilean`): префикс
снимается, остаток сверяется со списком Id Data Dragon без учёта регистра
(`lolvision.ddragon.norm_name`; внутреннее имя бывает `FiddleSticks` при Id
`Fiddlesticks`). Нет в списке или списка нет — `champion = None` и
`champion_reason`. Список даёт вызывающий (`ddragon_champion_ids()` — из кэша
Data Dragon); на журнале своей игры все десять сверились с 16.19.1.

У части чемпионов `rawChampionName` старого вида — `Character_Aatrox_Name`
(вторая своя игра 2026-10-12: Aatrox среди девяти обычных); этот формат
разбирается явно (`RAW_CHAMPION_FORMATS`), любой другой — отказ с причиной. До
исправления один такой чемпион отклонял весь снимок: состав, смерти и сторона
из API в этой игре не работали ни на одном из 1843 снимков. После — обе игры:
смертей из API столько же, сколько `ChampionKill` (52 и 70), сбросов памяти
нет, своя сторона ORDER и CHAOS.

**Заклинания призывателя (#804).** `summonerSpells.summonerSpellOne/Two` —
`{displayName, rawDisplayName, rawDescription}`; `displayName` локализован
(«Телепорт»), поэтому id берётся из `rawDisplayName`
(`GeneratedTip_SummonerSpell_<id>_DisplayName`, запасной — `rawDescription`):
`spell_id` срезает сезонную приставку (`S12_SummonerTeleportUpgrade` →
вариант `SummonerTeleportUpgrade`) и относит вариант к семье
(`SPELL_FAMILIES`: Телепорт, Кара). `PlayerRecord.spells` — два
`SpellRecord(slot, family, variant, raw, name)`; поля нет (старые сборки,
обезличенные фикстуры `own_game*.jsonl`) или формат другой — `spells = None`
и `spells_reason`, остальной снимок разбирается. Все сырые значения журналов
своей игры и их нормализация — docs/state-api.md, «Заклинания призывателя».
Перезарядки API не отдаёт.

**Ники в событиях.** `KillerName`/`VictimName` в журнале своей игры — это
`riotIdGameName` (без тега), а не riotId целиком.

## Состояния

`LiveStatus(state, reason, t, game, game_t)` — явное состояние вместо исключений:

| Состояние | Когда | `game` |
| --- | --- | --- |
| `no_game` | порт закрыт / нет ответа, а игра не шла | нет |
| `loading` | ответ есть, игра не началась: полный ответ без единого события (см. ниже); HTTP-ошибка в формате Riot до игры; пустой `allPlayers` | есть, если пришёл состав |
| `in_game` | разобранный снимок с событиями, без `GameEnd` | есть |
| `ended` | событие `GameEnd` (в `reason` — Result) или API молчит дольше `lost_after` (5 с) посреди игры | последний снимок |
| `error` | не JSON, не те поля или типы (в `reason` — путь поля), HTTP-ошибка посреди игры, нет ответа посреди игры короче `lost_after` | нет |

События не повторяются: `EventID` запоминаются в пределах игры, новая игра
(время матча откатилось больше чем на 5 с) начинает счёт заново.

`LiveClient` опрашивает в своём потоке (`period=0.5` — ~2 Гц): `status()` —
последний `LiveStatus`, `events()` — новые события с прошлого вызова,
`poll_once()` — один опрос без потока. Любая ошибка транспорта или разбора
становится `error` с причиной и не выходит к вызывающему.

## В сеансе live

`LiveClient` заводит `live.start_screen` и отдаёт сеансу экрана
(`Session(api=...)`): сеанс запускает опрос вместе с собой и останавливает
вместе с собой. У ролика YouTube API нет — `api: null`, `api_reason: "нет —
источник YouTube"`. В `status()` сеанса — `api`: `LiveStatus.to_dict()` без
полного снимка игры (десять игроков с предметами — десятки КБ на каждый
опрос смотрелки), вместо него краткая сводка `brief` (`game_time`,
`game_mode`, `players`, `me` — чемпион, команда, `is_dead`; `me_reason`) и
`events` — сколько событий накоплено. Полный снимок и все события —
`GET /api/live/<id>/api` (`Session.api_full()`). События копятся в
`Session.api_events` за весь сеанс (`EventRecord`); `Session.api_status()` —
последний `LiveStatus`. Отсутствие API (`no_game`, `error`) сеанс не
роняет — это только состояние в `api`.

На каждом кадре (волна 3, #798) сеанс берёт `status_observation(client.status(),
время получения кадра)`: наблюдение идёт в `LiveChampions.set_api` (до
миникарты: иконки назначаются среди десяти из API) и в
`MatchState.update(..., api=obs)`. Одиночный сбой опроса состав не
сбрасывает: `set_api(None)` — только когда свежего наблюдения нет дольше
`API_STALE` (5 с). В строке кадра: `game_time` — `{"value", "src": "api",
"reason"}` из `MatchState.game_clock()` (нет — `null` и причина: нет API,
API не в игре, снимок не принят, API молчит), `conflicts` — счётчики
расхождений по видам за матч, `api_refused` — почему снимок игры не
принят. Смена `LiveStatus.state` — сигнал паспорту (`set_api`); отказ
состава из API в `LiveChampions` — в `note` и `status()["api_error"]`.

Сеанс своего экрана поднимает сервис состояния (state_service.py) на
`127.0.0.1:8733`; адрес — `status()["state_url"]`, порт занят — сеанс идёт
без сервиса, причина — `state_reason`.

## Наблюдение для контекста матча (#798)

`api_observation(snap, age=0)` переводит `GameSnapshot` в
`lolvision.match_state.ApiObs` — запись, которую принимает `MatchState`
(`lolvision` не импортирует `lolstream`): время матча, десять игроков
(чемпион, команда, `isDead`, `respawnTimer`) — ORDER, затем CHAOS, своя
команда. `age` — возраст снимка: время матча сдвигается вперёд, таймеры —
назад (не ниже 0). Отказ `(None, причина)`, а не половина состава: чемпион не
опознан, игроков не десять, команды не 5/5, чемпионы повторяются.
`status_observation(status, now)` — то же по последнему `LiveStatus`:
наблюдение есть в `loading` с составом (время матча стоит) и `in_game`; в
`no_game`, `error` и `ended` — `None` и причина. Порог новой игры
`NEW_GAME_BACKSTEP` один на `Tracker` и `MatchState` (задан в
`match_state.py`). Правила приоритета и сверки — в
[match-context.md](match-context.md).

## Журнал и воспроизведение

Журнал — JSONL `{"t": unix-время, "code": 200 | текст ошибки, "body": текст allgamedata | null}`.
`read_journal(path)` читает его (недописанная последняя строка растущего файла
пропускается, битая строка в середине — ошибка с номером), `replay(rows,
champion_ids)` прогоняет через тот же `Tracker`, `journal_transport(rows)` —
подставной транспорт для `LiveClient`. Фикстура `tests/fixtures/live_client/own_game.jsonl`
— 36 обезличенных снимков ранкеда 2026-10-12 (ники → Player1…Player10, тег
0000, руны, умения и описания предметов вырезаны): хвост загрузки, старт,
первая кровь и смерти до 5:37. Вторая, `own_game_end.jsonl`, — хвост той же
игры: 10 последних снимков (6 до `GameEnd`, 4 с ним), разрыв соединения и 11
таймаутов, ~290 КБ; поле `code` у ошибок — как в журнале
(`"RemoteDisconnected(...)"`, `"URLError(TimeoutError('timed out'))"`). Ники
в обеих проверены по всему журналу: riotId, riotIdGameName, summonerName,
теги и имена из событий (KillerName, VictimName, Assisters, Recipient, Acer)
не встречаются ни разу.

## Наблюдения и допущения

* **Замечено** (журнал своей игры, 427 опросов, ~1 Гц): за 5 с до старта
  API уже отдаёт полный ответ 200 с составом, `gameTime` стоит на 0,07 с,
  золото 0, событий нет; с GameStart время пошло. Ошибок за 7 минут не было.
  Несуществующий путь отвечает 404 `{"errorCode": "RESOURCE_NOT_FOUND",
  "httpStatus": 404, "message": "Invalid URI format"}`.
* **Допущение:** что отвечает API раньше, на самом экране загрузки (ошибка
  Riot или пустой `allPlayers`), документацией не описано и не поймано — оба
  варианта считаются `loading`.
* **Замечено** (тот же журнал до конца, 1849 опросов): `GameEnd` (`Result:
  "Lose"`) пришёл на 29:41, и ещё 4 опроса (~3 с) API отдавал полный ответ —
  время матча шло дальше. Затем один `RemoteDisconnected` и 60 таймаутов
  подряд за 2 минуты: **после `GameEnd` API не отказывает в соединении, а
  молчит по таймауту** (порт ещё слушает). `Tracker` держит `ended` с
  последним снимком на всех этих ошибках (`reason` — «API пропал после конца
  игры»). Прогон журнала: `loading` 5, `in_game` 1779, `ended` 65, 90
  событий без повторов.
* **Не поймано:** когда порт закрывается совсем (журнал кончился на
  таймаутах) и что API отвечает при пропаже посреди игры (вылет клиента,
  переподключение) — там действует правило `lost_after`.
* Образец событий Riot (`liveclientdata_events.json`) — перечень типов, у всех
  `EventID = 0`; для проверки «без повторов» он не годится и используется
  только для разбора.
* Режимы кроме Ущелья (ARAM, Arena) не проверены: `position` бывает пустым,
  в Arena команды могут быть не ORDER/CHAOS — тогда ответ станет `error` с
  причиной, а не тихим составом.
