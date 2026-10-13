# Паспорт выбора игры и нетипичный выбор (#807)

Зачем. Статистика «кто что берёт» по своим играм и пометки нетипичного
выбора: АДК с Телепортом, Кара не у лесника, поддержка без предмета
поддержки, редкая для чемпиона ключевая руна. Для всех десяти игроков — своего,
союзников и врагов. Пока это только записи: без голоса и без /state.

Код: `lolstream/loadouts.py` (игра → строки, запись, база частот),
`lolstream/surprises.py` (детектор), правила — `lolstream/surprise_rules.toml`,
инструмент — `tools/loadouts.py`. Роль и руны в /state — `me.role`,
`me.runes`, `champions[].role`, `champions[].runes` ([state-api.md](state-api.md),
«Роль и руны»).

## Откуда и когда пишется

Источник один — Live Client API своей игры (`allgamedata`, source `api`).

* **Живой сеанс** «свой экран» (`live.start_screen`, `tools/live_screen.py`,
  кнопка в смотрелке): `LoadoutRecorder` подключён к `LiveClient.on_status` и
  видит каждый опрос (~2 Гц). Игра пишется, когда API перешёл в `ended`
  (GameEnd или пропажа API после игры); сеанс закрыт посреди игры — пишется то,
  что есть, с `complete: false`. Выключить: `tools/live_screen.py
  --no-loadouts`, в смотрелке — `"loadouts": false` в теле
  `POST /api/live/start`. Что записано — `status()["loadouts"]` сеанса.
* **Журнал** (`data/live_client/*.jsonl`, `data/own/<сеанс>/api.jsonl`):
  `python tools/loadouts.py record <журналы…>` (`--dry-run` — только сводка).

Выход — `data/loadouts/<дата начала игры>.jsonl` (папка под `/data/`, в Git
не идёт), строка на игрока. Та же игра второй раз не пишется (тот же состав
и начало ближе 180 с — повторный прогон журнала или журнал + живой сеанс).
Тренировка (`PRACTICETOOL`) не пишется (`--all-modes` — писать).

Предметы делятся на законченные по `item.json` Data Dragon из кэша
(`data/ddragon/<версия>/item.json`; докачать — `python tools/ddragon.py
--version 16.19.1 --items`). Нет файла — `completed: null` с причиной,
остальное пишется.

## Строка (schema 1)

**Ников нет** ни в одном поле: ник — только ключ игрока в памяти.

| Поле | Что | Нет данных |
| --- | --- | --- |
| `game_id` | хэш (`game_id_source: "hash"`): режим, карта, состав (команда + rawChampionName), минута начала. Id матча API не даёт | — |
| `game_key` | хэш режима, карты и состава (без времени) — для поиска дублей | — |
| `game_start`, `game_start_unix` | начало матча: время опроса минус время матча | null |
| `mode`, `map` | `gameData.gameMode`, `mapNumber` | — |
| `patch` | **всегда null**, `patch_reason`: API версию игры не отдаёт | |
| `ddragon` | версия item.json, по которой отнесены предметы (не версия игры) | null без каталога |
| `duration`, `complete` | последнее время матча (с); был ли GameEnd | |
| `origin`, `source` | `live` / `journal:<файл>`; `api` | |
| `team`, `relation` | `ORDER`/`CHAOS`; `self`/`ally`/`enemy` относительно своего игрока | `relation: null` + `relation_reason` у наблюдателя |
| `champion`, `is_bot` | Id Data Dragon | `champion_reason` |
| `role` | `TOP`/`JUNGLE`/`MIDDLE`/`BOTTOM`/`UTILITY` (`position` драфта) | null + `role_reason` (ARAM, тренировка) |
| `spells` | `families` и `variants` (на первом снимке, по слотам), `final` (варианты в конце), `upgrades` `[{t, slot, variant}]` — улучшение Телепорта, рост Кары | null + `spells_reason` |
| `runes` | `{keystone, primary, secondary}` по `{id, name}` | null + `runes_reason` |
| `items.start` | инвентарь на последнем игровом снимке до 90 с матча | null + `start_reason` (игра подхвачена поздно) |
| `items.completed` | `[{t, item, name, kind}]` по порядку появления; `kind` — `legendary` (дальше не улучшается, от 2000 золота) или `boots` (ботинки 2-го уровня и выше) | null + `completed_reason` (нет item.json) |
| `items.purchases` | все появления предметов `[{t, item}]` (опрос 1–2 Гц: время ±1 с; превращения вроде Муроманы — тоже «появление») | |
| `items.final` | инвентарь в конце | |
| `result` | `Win`/`Lose`: GameEnd своего игрока, врагу — обратный | null + `result_reason` (нет GameEnd, наблюдатель) |
| `surprises` | `[{kind, what, expected, confidence, basis, rule}]` (ниже) | `[]` |
| `surprises_notes` | что не проверялось и почему (нет роли, мало игр в базе, нет базы) | |
| `rules` | файл и версия правил, сводка базы (`source`, `built`, `games`, `note`) | |

## Детектор: `surprises`

`kind` — `spell` / `keystone` / `item` / `role`; `what` — что взято
(`SummonerTeleport`) или чего нет (`no SummonerSmite`); `expected` — что
ожидалось; `confidence` 0..1 — насколько правило уверено, что это необычно;
`basis` — объяснение (правило и его смысл или доля в базе); `rule` — id.

Правила (`lolstream/surprise_rules.toml`, свой файл —
`tools/loadouts.py record --rules`) работают, только если у игрока есть роль:

* заклинания по роли — ожидаемые семьи (`spells.roles.<роль>.expected`), чужая
  семья — `other_confidence`; отдельные правила `has`/`lacks` заменяют общее
  для своей семьи: Кара не в лесу (0,9), лесник без Кары (0,95), стрелок с
  Телепортом (0,6), с Воспламенением (0,5), с Призраком (0,5), поддержка с
  Телепортом (0,7), без Скачка (0,5);
* ключевая руна, необычная для роли без учёта чемпиона (0,35): лес — Aery,
  Comet, Guardian, Spellbook; стрелок — Grasp, Aftershock, Guardian;
  поддержка — Lethal Tempo, Fleet Footwork, Conqueror;
* предметы: предмет поддержки не у поддержки (0,8) и поддержка без него
  (0,6, игра от 10 мин); питомец леса не у лесника (0,85) и лесник без
  питомца (0,6, от 5 мин).

Значения взяты из общего знания мета-игры, а не из статистики. Чемпионов, у
которых своя норма (Юми без Скачка, Сенна или маги на нижней линии), ролевые
правила отмечают ложно — это задача базы частот.

## База «типичного»

Файл JSON (`loadouts.Base`, `kind: "loadout_base"`, `schema: 1`):

```json
{"schema": 1, "kind": "loadout_base", "source": "own", "source_detail": "2026-10-12.jsonl",
 "built": "2026-10-13T12:52:38", "patch": null, "games": 20,
 "note": "малая выборка: 20 игроко-игр (< 100); …",
 "champions": {"Lux": {"games": 2, "roles": {"MIDDLE": 2}}},
 "entries": [{"champion": "Lux", "role": "MIDDLE", "games": 2,
              "spells": {"SummonerFlash": 2, "SummonerHeal": 1, "SummonerTeleport": 1},
              "keystones": {"ArcaneComet": 1, "DarkHarvest": 1},
              "first_item": {"6655": 2}}]}
```

`spells` — в скольких играх была семья; `keystones` — ключевые руны;
`first_item` — первый законченный легендарный предмет. Строится из своих
записей: `python tools/loadouts.py base data/loadouts/2026-*.jsonl` →
`data/loadouts/base.json` (по умолчанию её же берут `record` и живой сеанс;
нет файла — только правила, причина в `surprises_notes`). В базу идут
игроки с чемпионом и ролью, каждая игра один раз. Другой источник (свой
сбор Match-V5, ниже) пишет тот же формат с `source: "riot-match-v5"` и
патчем.

Сравнение (`[base]` в правилах): по чемпиону — роль; по чемпиону и роли —
семьи заклинаний, ключевая руна, первый предмет. Пока игр меньше
`min_games` (20) — не сравнивается (заметка в `surprises_notes`); доля ниже
`rare_share` (5 %) — сюрприз, `confidence = min(0,9, n / (n + 30))`. В базе
из своих записей есть и сама проверяемая игра — на малой выборке это
сильно сдвигает доли, поэтому порог по числу игр обязателен.

## Источники статистики: что можно брать (проверено 2026-10-13)

| Источник | Что есть | Можно ли |
| --- | --- | --- |
| Porofessor (porofessor.gg) | частоты сборок, рун, заклинаний | **Нет.** Условия использования: «Automated queries (including screen and database scraping, spiders, robots, crawlers and any other automated activity) are strictly prohibited … unless you have received express written permission». Публичного API нет |
| u.gg, op.gg, lolalytics | то же | **Нет публичного API.** Страницы условий u.gg (403) и op.gg (404 по известным адресам) автоматически не открылись, у lolalytics в подвале только политика конфиденциальности. Разрешения на автоматический сбор не найдено — не скрейпим |
| Riot API Match-V5 | матчи целиком: роль (`teamPosition`), заклинания, руны (perks), предметы, итог; патч (`gameVersion`) | **Да**, законный путь. Ключ разработчика живёт 24 ч; личный ключ (регистрация продукта, без проверки) — для личных проектов и исследований, 20 запросов/с и 100 за 2 мин на регион; политика Riot требует брать данные через их сервисы. Матч ≈ 1 запрос (+ 1 на список матчей игрока): ~3000 матчей в сутки — для частот по популярным чемпионам на патч хватит за несколько дней |
| Data Dragon / CommunityDragon | статика: предметы, руны, иконки | статистики нет; item.json используем для «законченный предмет» |
| Открытые датасеты (Kaggle и др.) | матчи старых сезонов | устарели по патчам и предметам (система предметов и рун меняется каждый сезон) — для «типичного сейчас» не годятся |

Вывод: законный и надёжный источник базы — свой сбор через Riot Match-V5 с
личным ключом (ключ — в хранилище `C:\develop\ops`, не в репозитории),
записанный в формат базы выше. Сейчас его нет: база заполняется из
собственных записей (`source: "own"`, с пометкой о малой выборке), а
детектор фактически работает на ролевых правилах.

## Прогон на двух журналах 2026-10-12

`tools/loadouts.py record data/live_client/*.jsonl`: две игры CLASSIC
(ранкед/драфт: роли у всех двадцати), обе с GameEnd, 20 строк; запись
тренировки `data/own/20261012-185628` пропущена (PRACTICETOOL).

| Игра | Чемпион, роль | Что нетипично | Правило, уверенность |
| --- | --- | --- | --- |
| 1 | Caitlyn, BOTTOM (союзник) | Призрак (Ghost + Flash) | `bottom_ghost`, 0,5 |
| 1 | Lux, MIDDLE (враг) | Исцеление в центре | `spells.roles.MIDDLE`, 0,4 |
| 1 | Yuumi, UTILITY (враг) | без Скачка (Exhaust + Heal) | `no_flash`, 0,5 — для Юми это норма: ложная пометка, которую снимет база |
| 2 | — | ничего | Lux MIDDLE с Телепортом и Dark Harvest, Tahm Kench UTILITY с Воспламенением и Grasp, Xerath UTILITY с First Strike, Tristana с Hail of Blades — в пределах правил |

База из этих 20 строк — 19 пар чемпион+роль по 1–2 игры, ни одна не
доходит до `min_games`: по базе ничего не помечено (так и должно быть).
