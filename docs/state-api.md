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

Сервис поднимается вместе с сеансом live «свой экран» (`live.start_screen`;
адрес — `state_url` в status() сеанса), провайдер — `Session.state_inputs`:
контекст матча (`match_view`), копия паспорта и последний кадр зрения
(`VisionFrame`, юниты и треки — после `sides.assign`) снимаются под замком
сеанса. Без сеанса — `python -m lolstream.state_service` (только API).

**Сервис ничего не сводит сам — он переводит в контракт.** У каждого
сведения один хозяин:

| Что | Хозяин | Как попадает в снимок |
| --- | --- | --- |
| состав, жив / мёртв, таймер возрождения, место, ХП, время матча, расхождения источников | контекст матча `lolvision.match_state.MatchState` (приоритет API > HUD > зрение, счётчик `conflicts`) | снимок `MatchView` (`match_view(ms, at)`) |
| ник, уровень, предметы, счёт, золото и статы своего игрока, режим, карта, своя команда | Live Client API (`LiveStatus`) — без конкуренции со зрением | прямо |
| недавние события матча (убийства, драконы, башни) | Live Client API (`GameSnapshot.events`) | ники → чемпионы по составу API (`recent_events`, #803) |
| можно ли читателю зрения отвечать (имена над плашками, детектор юнитов) | паспорт трансляции `lolvision.passport.Passport` | `Passport.verdict(reader)` |
| свои / враги / сам и сторона у юнитов и треков | `lolvision.sides` (поля `relation`, `side` записей `UnitRec` / `TrackRec`) | поля записи как есть |
| камера, тип сцены, юниты, треки | кадр сеанса live | `VisionFrame` |

## Как запустить для coach

Одной командой, без смотрелки (#801):

```bash
python tools/live_screen.py                       # 2 кадра/с, захват auto — как в смотрелке
python tools/live_screen.py --fps 2 --backend dxcam --run <прогон под runs/>
```

Скрипт открывает сеанс «свой экран» (`live.start_screen`: окно игры, Live
Client API своей игры, этот сервис), печатает адрес — `http://127.0.0.1:8733/state`
(события — `/events`) — или причину, почему сервис не поднят, и раз в 10 с
строку: состояние сеанса, источника (кадр есть / пауза / окна нет), API,
вид, сколько кадров. Ctrl+C — сеанс останавливается, скрипт ждёт, пока он
отпустит захват, API и порт.

Порт 8733 один: если смотрелка уже держит сеанс «свой экран», скрипт не
запускается и говорит об этом (код выхода 2) — остановите сеанс в смотрелке
или берите состояние у неё по тому же адресу. Coach читает `/state` и
`/events` (пример клиента — `tools/state_client_example.py`, см. «Клиент
coach» ниже).

## Версия схемы

Каждый снимок и каждое сообщение потока несут `"schema": 1`. Смена имени,
типа или смысла поля, удаление поля — новая версия. Новое поле — та же
версия (то же правило, что у `records.SCHEMA`): клиент обязан пропускать
незнакомые поля, а не отказывать из-за них. Машинная схема ниже закрытая
(`additionalProperties: false`) — это проверка самого сервиса на опечатки,
а не требование к клиентам; новое поле добавляется в неё вместе с кодом
(так добавлены `me.position`, #802, `events`, #803, и `spells` /
`spell_variants`, #804). Исключение — записи
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
| `inferred` | выведено: память и правила контекста матча, паспорт |

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
| `events` | факт: массив | недавние события матча по API, без ников (#803) |
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
| `match.api_refused` | строка / `null` | почему последний снимок API не принят (не 10, не 5/5, повторы): отказ `status_observation` в сеансе (снимок до `MatchState` не дошёл, `match_view(ms, at, refused)`) или `MatchState.api_refused` |
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
| `time` | число, с | api | время матча на момент кадра — `MatchState.game_clock()`; API нет, пропал (`api_live` = false) или снимок не принят — отказ с причиной; снимок не принят (`sources.match.api_refused`, например «игроков 1, а не 10» в Practice Tool) — причина «снимок API не принят: …», она же у `champions`, `enemies`, `me.alive`, `me.respawn_in` |
| `mode` | строка | api | режим (`CLASSIC`, `ARAM`…) |
| `map` | целое | api | номер карты (11 — Ущелье) |
| `side` | `ORDER` / `CHAOS` | api | сторона своего игрока (activePlayer среди allPlayers) |
| `state` | `no_game` / `loading` / `in_game` / `ended` / `error` | api | состояние API (есть и без снимка игры) |

### `events` — недавние события матча

Чтобы ответить «что сейчас произошло»: кто меня убил, кто взял дракона или
башню (#803). Поле добавлено в schema 1 (новое поле — та же версия).

Факт (`source = "api"`, `age` — давность снимка API): массив событий Live
Client **старые → новые**, не больше `EVENTS_MAX` = 20, за последние
`EVENTS_WINDOW` = 180 с игрового времени до `match.time` (время матча
неизвестно — до времени последнего снимка API). Служебные `GameStart` и
`MinionsSpawning` не входят. Событий в окне нет — пустой массив. Отказ: API
не подключён или снимка игры нет — причина API (как у `match.mode`); снимок
есть, но игра не идёт (`loading`) — «не в игре: Live Client loading (…)».
После конца игры (`ended`) список остаётся — последний снимок, с `GameEnd`.

| Поле элемента | Тип | Смысл |
| --- | --- | --- |
| `time` | число, с | игровое время события (`EventTime`, до 0,1 с) |
| `ago` | число, с / `null` | `match.time − time` (не меньше 0); время матча неизвестно — `null` |
| `kind` | строка | `EventName` API как есть: `ChampionKill`, `FirstBlood`, `Multikill`, `Ace`, `DragonKill`, `HeraldKill`, `BaronKill`, `HordeKill` (личинки), `AtakhanKill`, `TurretKilled`, `InhibKilled`, `InhibRespawningSoon`, `InhibRespawned`, `FirstBrick`, `GameEnd`… — незнакомый вид проходит как есть |
| `killer` | who / `null` | кто: `KillerName`; у `FirstBlood` — `Recipient`, у `Ace` — `Acer`; поля нет — `null` |
| `victim` | who / `null` | кого: у `ChampionKill` — `VictimName`; у `TurretKilled` — сама башня (`other = "turret"` и чья); у `DragonKill`, `HeraldKill`, `BaronKill`, `HordeKill`, `AtakhanKill` — `{"champion": null, "relation": null, "other": "monster"}`; у остальных (`Multikill`, `Ace`, `InhibKilled`, `FirstBlood`…) — `null` |
| `assisters` | массив who | `Assisters` по порядку API; нет — `[]` |
| `detail` | строка / `null` | не-ники из события через «, » в таком порядке: `DragonType` (`Fire`, `Elder`…), `KillStreak` (`"2"`…), `TurretKilled` / `InhibKilled` / `InhibRespawned` / `InhibRespawningSoon` (имя объекта как в API: `Turret_TOrder_L1_P3_1242677625_0`), `AcingTeam` (`ORDER` / `CHAOS`), `Result` у `GameEnd` (`Win` / `Lose`), `Stolen` — слово `Stolen`, только если украден. Ничего нет — `null` |

who — участник события, **без ника**:

```json
{"champion": "Darius", "relation": "enemy", "other": null}
{"champion": null, "relation": "ally", "other": "turret"}
```

| Поле | Значения | Смысл |
| --- | --- | --- |
| `champion` | Id Data Dragon / `null` | чемпион игрока с этим ником; не чемпион или не нашёлся — `null` |
| `relation` | `self` / `ally` / `enemy` / `null` | к своему игроку: у чемпиона — `relation_to_me` (как `champions[].relation`); у башни и миньона — по команде в имени объекта (`_TOrder_`, `_T1_`, `_T100`…): своя — `ally`, чужая — `enemy`; монстр, неопознанное, своего игрока нет — `null` |
| `other` | `null` / `turret` / `minion` / `monster` / `unknown` | не чемпион: `Turret_…` — `turret`, имя с `Minion` — `minion`, `SRU_…` — `monster`; ник не нашёлся в составе или имя незнакомое — `unknown` |

**Ники не выводятся никогда.** `KillerName`, `VictimName`, `Assisters`,
`Acer`, `Recipient` — ники игроков; они сопоставляются с составом API
(`PlayerRecord.riot_id`, «Имя#тег» или summonerName): сперва точное
совпадение, иначе — единственный игрок с той же частью до `#` (в событиях
своей игры ник идёт без тега). Не нашёлся или таких игроков двое —
`{"champion": null, "relation": null, "other": "unknown"}`. Сырые имена
объектов-убийц (`Turret_T2_C_05_A`, `Minion_T200L2S29N0180`, `SRU_Baron…`)
тоже не выводятся — только метка `other`. На журналах двух своих игр
2026-10-12 (52 и 70 `ChampionKill`) сопоставлены все 378 упоминаний
игроков в убийствах, счёт убийств по чемпионам совпал со `scores.kills` API.

Поток SSE `game` (ниже) — `EventRecord.to_dict()` как есть, то есть **с
никами**; обезличенный вид — только это поле.

### `me` — свой игрок

`riot_id`, `champion` (Id Data Dragon), `team`, `level`, `gold` (текущее
золото), `stats` (`championStats` как есть: `abilityPower`, `armor`,
`currentHealth`, … числами) — из API. `alive`, `respawn_in` и `position`
(`{"u", "v"}`, место на карте) — те же факты, что у своего чемпиона в
`champions` (из контекста матча; место своего на виде игрока — по его
полоске на кадре или своей иконке миникарты, #802,
[match-context.md](match-context.md)). Нет API или себя не нашли — все
отказы с этой причиной; своего нет в контексте матча — `alive`,
`respawn_in`, `position` отказ с причиной контекста. `spells` и
`spell_variants` — заклинания призывателя, как у записи в `champions`
(ниже, «Заклинания призывателя»).

### Заклинания призывателя: `spells`, `spell_variants` (#804)

Есть у `me` и у каждой записи `champions`. Факт `api`: массив из двух id по
слотам — `summonerSpellOne`, `summonerSpellTwo` (по умолчанию клавиши D и F;
клавиши игрок может переназначить, слот — нет).

* `spells` — **семья** заклинания: на неё и опираться в правилах («есть ли
  Телепорт» — `"SummonerTeleport" in spells`). Улучшенный Телепорт и все
  стадии Кары входят в свою семью.
* `spell_variants` — точный вариант: id из API без сезонной приставки.

Id берётся из `rawDisplayName` (`GeneratedTip_SummonerSpell_<id>_DisplayName`,
запасной — `rawDescription`), не из локализованного `displayName`
(«Телепорт»). Сезонная приставка `S<n>_` срезается; семья — `SummonerTeleport`
или `SummonerSmite`, если вариант начинается с неё, иначе сам вариант.

| Сырой `<id>` в API | `spell_variants` | `spells` | Что это |
| --- | --- | --- | --- |
| `SummonerTeleport` | `SummonerTeleport` | `SummonerTeleport` | Телепорт |
| `S12_SummonerTeleportUpgrade` | `SummonerTeleportUpgrade` | `SummonerTeleport` | Усиленный телепорт: в журнале своей игры сменил обычный у обоих владельцев на 10:02 |
| `SummonerSmite` | `SummonerSmite` | `SummonerSmite` | Кара |
| `S5_SummonerSmitePlayerGanker` | `SummonerSmitePlayerGanker` | `SummonerSmite` | Усиленная кара (рост питомца, 6:37–10:22 в журналах) |
| `SummonerSmiteAvatarOffensive`, `SummonerSmiteAvatarDefensive` | как есть | `SummonerSmite` | Первобытная кара (13:26–20:40 в журналах) |
| `SummonerFlash`, `SummonerDot`, `SummonerHeal`, `SummonerHaste`, `SummonerBarrier`, `SummonerExhaust` | как есть | как есть | Скачок, Воспламенение, Исцеление, Призрак, Барьер, Изнурение |

Таблица — все значения двух журналов своей игры 2026-10-12 (Ущелье, 20
игроков); id других режимов (ARAM, Арена) не встречались и выходят как есть
по тому же правилу. Заклинание меняется посреди игры (улучшение) — снимок
показывает текущее. API не дал `summonerSpells` (старые сборки,
обезличенные журналы-фикстуры) или ключ не в этом формате — отказ с
причиной разбора; без API — отказ, как у `level`. **Готовности (перезарядки)
API не отдаёт** — `spells` говорит только, что заклинание взято.

**Готовность с HUD (`spells_ready`) не сделана** (#804, оценка). На своём
HUD у иконок D/F три вида «не нажать», а не два: перезарядка (синяя заливка
по часовой + белые цифры), запрет (иконка темнеет, поверх красный крестик —
контроль или, по-видимому, своё умение: так на кадрах с Альфа-ударом
Master Yi) и смерть (весь HUD серый, перезарядка при этом идёт). Замер по кадрам своего экрана 1920×1080
(Practice Tool, 32 кадра; ARAM, 14 игровых кадров), слоты при масштабе HUD
по умолчанию:

* на умениях Q/W/E/R перезарядка видна надёжно: «белые пиксели с тёмной
  обводкой в центре слота» — 11–52 у 63 из 64 слотов на перезарядке, 0–4 у
  всех 120 готовых; пропуск — серый HUD мёртвого (цифры тусклые);
* на заклинаниях тот же признак врёт: готовый Призрак (белая фигура с
  обводкой на иконке) — 22 на 30 из 32 кадров, выше порога. Доли синего и
  яркость тоже не годятся: в конце перезарядки синего почти нет (E: 0,03 при
  0,11 у готовой иконки), а яркость падает и без перезарядки — под
  Альфа-ударом медиана V у D/F 158 → 74 и 138 → 81, у мёртвого серый HUD;
* **на заклинаниях перезарядки на кадрах нет ни разу** — проверить не на чем.

Что нужно, чтобы сделать честно: (1) кадры своего экрана с заклинаниями на
перезарядке — Practice Tool, нажать D/F и записать 30–60 с, лучше с разными
заклинаниями (Телепорт, Кара, Скачок, Призрак) и при смерти; (2) эталон
иконки по id из `spells` — картинки заклинаний Data Dragon
(`summoner.json` + `img/spell`; улучшенных Телепорта и Кары там нет — нужен
CommunityDragon) или своя готовая иконка, запомненная в сеансе: готово ⇔
вырез похож на эталон по яркости и цвету, без синего сектора и цифр;
(3) положение слотов при масштабе HUD игрока — пункт паспорта, как у
миникарты (поиск рамки панели умений по кадру), а не постоянные координаты;
(4) читатель паспорта `hud_spells` (вид игрока, кадр, найденная панель);
красный крестик и серый HUD — `null` с причиной, а не «не готово». Секунды —
цифры белым шрифтом, образцы для ближайшего соседа есть только с умений
(6–19 и 0,3–41 с на этих кадрах).

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
| `spells`, `spell_variants` | массив из двух строк | api | заклинания призывателя по слотам (см. выше); API их не дал — отказ с причиной разбора |
| `alive` | bool | api / hud / minimap / inferred | `ChampCtx.alive` с источником `alive_src` и возрастом `alive_age`; не наблюдался — отказ |
| `respawn_in` | с | api / hud (`ChampCtx.respawn_src`) | `ChampCtx.respawn_in` у мёртвого; живой — отказ «жив (источник)»; мёртв без таймера — отказ |
| `position` | `{"u", "v"}` | minimap / screen / inferred | последнее известное место (`ChampCtx`); не наблюдалось — отказ |
| `map_state` | `visible` / `remembered` / `dead` / `unknown` | inferred | `ChampCtx.state`: виден сейчас / помним место / мёртв / нет данных |
| `on_screen` | `true` | screen | на кадре чемпион с этим именем над плашкой; свой чемпион — ещё и по своей полоске: ровно один юнит-чемпион с `relation = self` на кадре (#802; так свой виден и на виде игрока, где имена не читаются). **`false` не выдаётся**: «не опознан» ≠ «нет на экране». Читатель `names` выключен паспортом (вид игрока и пр.) — отказ с его причиной, у своего без полоски — «своей полоски (self) на кадре нет, …»; своих полосок больше одной — отказ (`conflicts` `self_many`); свой мёртв по контексту — отказ: полоска не его; детектор `units` выключен паспортом — своего по полоске не решаем |
| `hp` | доля 0..1 | screen / inferred | `ChampCtx.hp` с `hp_src` и `hp_age` (`inferred` — 1.0 после возрождения) |
| `conflicts` | целое (не факт) | — | `ChampCtx.conflicts`: расхождений по этому чемпиону за матч |

### `enemies` — враги с давностью

Факт: массив врагов (`relation = enemy`), у каждого `champion`, `position`,
`map_state`, `alive`, `respawn_in`, `on_screen` — те же факты, что в
`champions`. Давность места — `position.age`. Своя сторона неизвестна —
отказ.

### `camera`, `scene`

| Поле | `value` | Источник | Смысл |
| --- | --- | --- | --- |
| `camera.center` | `{"u", "v"}` | screen / minimap | центр камеры: на Ущелье — ответ модели камеры (`px, py` строки кадра, источник `screen`); на другой карте по API (ARAM) — центр рамки вьюпорта на миникарте (`cx, cy`, источник `minimap`), рамка не видна или миникарта не найдена — отказ с причиной |
| `camera.spread` | число | screen | разброс ответа модели (доли карты) — неуверенность; не на Ущелье — отказ |
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
  `side` (`ORDER` / `CHAOS`; источник — поле записи `side_src`: `api` —
  от команды своего игрока по API, `detector` → `screen` — ответ детектора
  на виде наблюдателя) — поля записи; `null` → причина записи (`relation_reason`, `side_reason`:
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
| `self_many` | на виде игрока на кадре больше одной своей полоски (`relation = self`): какая своя, неизвестно — ни места, ни ХП (#802) |
| `self_place` | место своего с экрана (полоска + проекция) дальше `SELF_PLACE_TOL` = 0,05 карты от своей иконки на миникарте того же кадра; место берётся с миникарты (#802) |

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
 "events": {"source": "api", "age": 0.2, "reason": null, "value": [
  {"time": 227.0, "ago": 110.4, "kind": "ChampionKill",
   "killer": {"champion": "Amumu", "relation": "ally", "other": null},
   "victim": {"champion": "Lux", "relation": "enemy", "other": null},
   "assisters": [{"champion": "Yone", "relation": "ally", "other": null}], "detail": null},
  {"time": 327.9, "ago": 9.4, "kind": "ChampionKill",
   "killer": {"champion": "Lux", "relation": "enemy", "other": null},
   "victim": {"champion": "Yone", "relation": "ally", "other": null},
   "assisters": [], "detail": null},
  {"time": 333.2, "ago": 4.1, "kind": "ChampionKill",
   "killer": {"champion": "Darius", "relation": "enemy", "other": null},
   "victim": {"champion": "Zilean", "relation": "self", "other": null},
   "assisters": [], "detail": null}
 ]},
 "me": {
  "riot_id":    {"value": "Player1#0000", "source": "api", "age": 0.2, "reason": null},
  "champion":   {"value": "Zilean", "source": "api", "age": 0.2, "reason": null},
  "team":       {"value": "ORDER", "source": "api", "age": 0.2, "reason": null},
  "level":      {"value": 5, "source": "api", "age": 0.2, "reason": null},
  "gold":       {"value": 443.9, "source": "api", "age": 0.2, "reason": null},
  "alive":      {"value": false, "source": "api", "age": 0.2, "reason": null},
  "respawn_in": {"value": 9.9, "source": "api", "age": 0.2, "reason": null},
  "position":   {"value": null, "source": null, "age": null, "reason": "место не наблюдалось"},
  "stats":      {"value": {"abilityHaste": 10.0, "abilityPower": 36.0, "armor": 39.45},
                 "source": "api", "age": 0.2, "reason": null},
  "spells":     {"value": ["SummonerTeleport", "SummonerFlash"], "source": "api", "age": 0.2,
                 "reason": null},
  "spell_variants": {"value": ["SummonerTeleportUpgrade", "SummonerFlash"], "source": "api",
                 "age": 0.2, "reason": null}
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

`"…": "…"` — пропуск в примере, не поле. Без API: в `match`, `events` и `me` — отказы «Live Client API не подключён к
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
| `game` | новое событие Live Client (по `EventID`; новая игра — счёт заново) | `live_client.EventRecord.to_dict()`: `event_id, name, time, data, source` — `data` с никами как в API; без ников — поле снимка `events` |
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
| `match` | `MatchView` = `match_view(session.match, at, refused)` | **под замком сеанса**: `at` — unix-время кадра `match.t`; `refused` — отказ снимка API на последнем кадре |
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
`get_state()`, `events()` с переподключением, `event_line(kind, msg)` —
строка печати сообщения потока (у снимка — `seq` и `conflicts.value.total`,
отказ — причиной) и `to_source_facts(snapshot)` —
короткий список фактов `{id, text, source, observed_at}` под
`coach.analysis_types.SourceFact` (время матча, сторона, свой чемпион и
таймер, враги: мёртв или место с давностью, `events.recent` — последние
пять событий одной строкой: «ChampionKill Darius (враг) → Zilean (вы), 4 с
назад»; `event_text(e)` — строка одного элемента `events`). Отказы в факты не
превращаются. В coach его место — `Observer._analyze_once()`
(`facts=` в `analyze`) и `visual_context()`; сам coach этой задачей не
меняется.

## Карты, кроме Ущелья (#803)

`u, v` — доли миникарты, они годятся для любой карты. Всё, что в единицах
Ущелья, на другой карте (`match.map` по API не 11) не выдаётся:

| Что | Ущелье (11) или карта неизвестна | Воющая бездна (12, ARAM) и прочие |
| --- | --- | --- |
| миникарта в кадре | эталон Ущелья, рамка, сохранённый прямоугольник | рамка миникарты или прямоугольник, сохранённый в игре на Ущелье (`lolvision/minimap_locate.py`, docs/screen-source.md) |
| места чемпионов с миникарты | да | да (`source: minimap`) |
| возрождение — у фонтана (`inferred`) | фонтан Ущелья | ARAM: CHAOS (0.941, 0.147) — замер по кадрам, ORDER (0.111, 0.929) — отражение через центр карты, на кадрах не проверен; другая карта — места нет (отказ), а не «где умер» |
| место с экрана (проекция, `MAP_UNITS` = 14 800) | да | нет: масштаб чужой |
| `camera.center` | модель камеры (`screen`) | рамка на миникарте (`minimap`) |
| `tracks` | да | отказ: треки в единицах Ущелья |
| «не игра» по патчевой модели (места Ущелья) | кадр — `other` | не довод: тип сцены — по DINO |

Карта неизвестна (ролик, наблюдатель — нет API) — считается Ущельем, как
обучающие ролики.

Проверка на кадрах ARAM своего экрана (15 кадров, 2 с, ARAM Mayhem —
`gameMode` KIWI, миникарта ×1.2): миникарта найдена по рамке (×1.2025,
≤ 0.4 px от прямоугольника эталона в игре на Ущелье); иконок назначено
составу API 48 из ~62 видимых глазом (~77%: поодиночке — почти все, в
кучах драки — половина; на Ущелье полнота детектора 94–98%,
docs/minimap-champions.md); рамка камеры найдена на 13 из 14 игровых
кадров (под окном выбора аугмента — нет). Патчевая модель камеры на ARAM
отвечала «не места Ущелья» на 12 из 14 игровых кадров: до #803 все они
становились `other`, и места с миникарты не шли в контекст матча.

## Снимки из журнала (#804)

Чтобы прогонять правила coach по записанным играм, а не только по живой,
`tools/replay_state.py` проигрывает журнал Live Client (`read_journal`:
`data/live_client/*.jsonl`, `data/own/<сеанс>/api.jsonl`) тем же путём, что
`python -m lolstream.state_service`: `replay` → `api_only_provider` →
`build_snapshot`, каждый снимок проходит `check_snapshot`.

```bash
python tools/replay_state.py data/live_client/20261012-171753.jsonl \
    --out runs/replay/20261012-171753.jsonl --step 5
```

* Выход — JSONL, строка на снимок: `{"t_journal": <unix-время строки>,
  "snapshot": {...}}`. Снимок — раз в `--step` с времени журнала (сетка от
  первой строки, по умолчанию 5 с; `0` — каждая строка) и всегда на
  последней строке. `seq` — номер снимка в файле.
* Часы — строка журнала: `snapshot.t = t_journal`, все `age` считаются от
  неё, как у живого сервиса в момент опроса, а не от «сейчас». Провайдер
  зовётся на каждой строке — MatchState видит все ответы API.
* Ники: по умолчанию (`--no-names`) все `riot_id` заменены на `"me"` (свой
  игрок) и `"player1"`…`"playerN"` (по порядку появления в журнале), а
  вхождения ника в любых других строках снимка — тем же псевдонимом. Такой
  файл можно отдавать в другой репозиторий как фикстуру. `--names` — оставить
  ники.
* Зрения в журнале нет: `camera`, `scene`, `units`, `tracks`, `on_screen`,
  `position` — отказы с причиной, как у сервиса без сеанса live.
* Ошибки контракта — в stderr, код выхода 1; нет кэша Data Dragon
  (`--ddragon`, по умолчанию `data/ddragon`) — код 2.

Два журнала своей игры 2026-10-12 (по ~100 МБ, ~30 мин): 383 и 394 снимка,
8,5 и 9,0 МБ; `champions` — во всех снимках, `match.time` — в 358 и 370
(остальные — загрузка и после конца игры), непустые `events` — в 361 и 370.
Ни одного из 20 ников (riotId, имя до «#», summonerName) в выходе нет.

## Не проверено

* ARAM на живом сеансе: всё выше — на сохранённых кадрах, путь по частям на
  процессоре; фонтан ORDER на ARAM; другие карты (Арена и пр.) — места у
  фонтана там не выдаются.
* Классификатор иконок на фоне ARAM: из 56 детекций у 12 лучший ответ вне
  состава (Fiddlesticks → Cassiopeia 0.1–0.36, Milio 0.8, Malphite 0.7…);
  назначение составу API спасло 4, 8 отброшены. Дообучения на ARAM не было.

* Ручка на живом сеансе по своему экрану: провайдер из `lolstream/live.py`
  делает интегратор.
* `python -m lolstream.state_service` против настоящего порта 2999.
* Нагрузка на живом сеансе не замерена. На синтетике (журнал своей игры +
  30 юнитов и 30 треков, CPU, 2026-10-12): снимок — 43 КБ JSON, сборка
  `build_snapshot` — 0,4 мс, проверка `check_snapshot` — 9 мс (в такт
  сервиса она не входит).
