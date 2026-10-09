# Источники видео

## Сеть «Challenger Replays»

`champion-replay-channels.json` — по одному YouTube-каналу на каждого чемпиона
League of Legends (173 из 173). Каналы одной сети, с одинаковым описанием и
форматом: полные ранговые игры в POV с ладдеров Challenger (KR, EUW, NA).
Отсюда взят первый ролик датасета (`58w57eJ5Qks`, Volibear Challenger Replays).

## Как собрано

- Список чемпионов — Riot Data Dragon 16.20.1 (`en_US`), 173 записи.
- Хэндл проверен запросом `https://www.youtube.com/@<handle>?hl=en`:
  `200` — канал есть, `404` — нет. Название, ID канала, число видео и
  подписчиков разобраны из `ytInitialData`.
- Дата сбора — 2026-10-07.

## Схема хэндлов

Основная — `@<ChampionId>ChallengerReplays`, где `ChampionId` — идентификатор
Data Dragon. Девять исключений:

| Чемпион | Хэндл | Почему отличается |
| --- | --- | --- |
| Hwei | [`@HweiChallenger`](https://www.youtube.com/@HweiChallenger) | без суффикса `Replays` |
| Irelia | [`@IreliaChallenger`](https://www.youtube.com/@IreliaChallenger) | без суффикса `Replays` |
| Jhin | [`@JhinChallenger`](https://www.youtube.com/@JhinChallenger) | без суффикса `Replays` |
| Karthus | [`@KarthusChallenger`](https://www.youtube.com/@KarthusChallenger) | без суффикса `Replays` |
| Kassadin | [`@KassadinChallenger`](https://www.youtube.com/@KassadinChallenger) | без суффикса `Replays` |
| Katarina | [`@KatarinaChallenger`](https://www.youtube.com/@KatarinaChallenger) | без суффикса `Replays` |
| Nunu & Willump | [`@NunuWillumpChallengerReplays`](https://www.youtube.com/@NunuWillumpChallengerReplays) | полное имя вместо id `Nunu` |
| Renata Glasc | [`@RenataGlascChallengerReplays`](https://www.youtube.com/@RenataGlascChallengerReplays) | полное имя вместо id `Renata` |
| Wukong | [`@WukongChallengerReplays`](https://www.youtube.com/@WukongChallengerReplays) | игровое имя вместо id `MonkeyKing` |

## Объём

Суммарно около 252 тыс. роликов. Распределение сильно неравномерное:
от 238 у «Locke» (чемпион вышел недавно) до 4,7 тыс. у «Ekko».
Для баланса по чемпионам это ограничение снизу, а не сверху.

## Оговорки

- Числа видео и подписчиков — округлённые значения YouTube на дату сбора.
- Каналы выкладывают ещё и Shorts (вертикальные нарезки). Для полных матчей
  их нужно отфильтровывать: по длительности или по `/shorts/` в URL.
- Проверенная раскладка HUD пока одна — `spectator-volibear-challenger`.
  Сеть выглядит шаблонной, но каждому новому каналу всё равно нужна
  визуальная проверка раскладки перед добавлением кадров в датасет.
- Названия каналов местами укорочены («Jax Challenger» вместо
  «Jax Challenger Replays»); сверять канал надо по `channel_id`, не по названию.

## Топ-игроки и стримеры: записи от первого лица

`top-player-channels.json` — 76 каналов, где игру записывает сам игрок:
его клиент, его HUD, часто вебкамера, чат и оверлеи стрима. Датасет пока
собран из режима наблюдателя одной сети, поэтому эти каналы нужны для проверки,
переносится ли модель на чужой HUD, зум и разрешение. Для сравнения в каталог
включены и каналы наблюдателя и трансляций турниров: у них тип `spectator`
или `broadcast` и пометка «не годится».

### Как собрано

- Кандидаты — поиск YouTube через `yt-dlp` (`ytsearch25:` по запросам на
  английском, корейском, китайском, вьетнамском, японском и турецком) плюс
  известные стримеры. Проверено 94 хэндла, в каталог вошли 76, остальные 18
  отброшены с причиной (`rejected`: пустой канал, 404, Wild Rift, TFT, не LoL,
  давно заброшен).
- Хэндл проверен запросом `https://www.youtube.com/@<handle>?hl=en`
  (`200` — канал есть). Название, ID канала, число видео и подписчиков
  разобраны из `ytInitialData`.
- Последние 30 роликов вкладки «Видео» и 15 записей вкладки «Трансляции»
  получены через `yt-dlp --flat-playlist`. Типичная длина — медиана по ним,
  без Shorts короче трёх минут.
- До трёх свежих роликов длиннее 10 минут разобраны через
  `yt-dlp -J --skip-download`: лучшее разрешение и fps, дата загрузки.
- Вебкамеру, оверлеи и HUD оценивали на глаз по раскадровкам YouTube
  (storyboard, сетка кадров 320×180). Видео не скачивалось.
- Дата сбора — 2026-10-09.

### Поля

- `region` — сервер, на котором сыграны записи. Вьетнамские стримеры почти
  все играют на KR, поэтому регион VN стоит лишь у двух каналов, а язык `vi`
  у семи.
- `content_type`: `full_game_pov` — полная игра от первого лица, по ролику на
  игру; `stream_vod` — запись стрима целиком (лобби, очередь, чат между
  играми); `edited_pov` — игра с вырезками; `commentary_pov` — чужая POV-игра
  с комментарием поверх; `highlights` и `guide` — нарезки и гайды;
  `spectator` и `broadcast` — наблюдатель и трансляция.
- `hud`: `standard`; `cn_client` — китайский клиент Tencent (другие шрифты и
  надписи); `custom_minimap_left` — миникарта слева.
- `source_tab` — где лежат полезные длинные записи: `videos` или `streams`.
- `dataset_fit`: `yes` — полные игры, мало монтажа; `partial` — годится после
  фильтрации (многочасовые VOD, смешанный контент, канал заброшен);
  `no` — нарезки, гайды, наблюдатель, трансляция.

### Сводка

| Регион | Каналов | yes | partial | no |
| --- | --- | --- | --- | --- |
| KR | 29 | 7 | 12 | 10 |
| CN | 13 | 2 | 10 | 1 |
| EUW | 12 | 3 | 7 | 2 |
| NA | 15 | 5 | 8 | 2 |
| VN | 2 | 0 | 2 | 0 |
| JP, TR | 2 | 0 | 2 | 0 |
| не указан | 3 | 1 | 0 | 2 |
| **всего** | **76** | **18** | **41** | **17** |

По типу: `stream_vod` 23, `full_game_pov` 19, `edited_pov` 12, `spectator` 8,
`highlights` 5, `broadcast` 3, `commentary_pov` 3, `guide` 3. Почти всё —
1080p60 (59 каналов); 1440p60 у восьми, 1080p30 у шести, 720p у двух,
2160p60 у одного.

### Лучшие кандидаты для проверки обобщения

| Канал | Игрок | Регион | Чем ценен |
| --- | --- | --- | --- |
| [`@proadcarchive`](https://www.youtube.com/@proadcarchive) | Ruler, Gumayusi, Deokdam, Smash… | KR | полные игры про-игроков с SOOP-стримов, без комментария, свежие, 1080p60 |
| [`@ProJungleArchive`](https://www.youtube.com/@ProJungleArchive) | Canyon, Lucid… | KR | та же сеть, лесники, часто 2–3 игры подряд |
| [`@ProTopArchive`](https://www.youtube.com/@ProTopArchive) | Kiin, Zeus… | KR | та же сеть, топ |
| [`@LPLlive`](https://www.youtube.com/@LPLlive) | Kiin, Canyon, ShowMaker… | KR | ежедневные «proview» с KR-сервера, 1–3 игры на ролик |
| [`@Nemesis2_lol`](https://www.youtube.com/@Nemesis2_lol) | Nemesis | EUW | игра на ролик, вебкамера слева от миникарты, чат слева |
| [`@BurstBrand`](https://www.youtube.com/@BurstBrand) | Burst | NA | без вебкамеры: чистый клиентский HUD, только анимированный аватар |
| [`@TruckDriverLoL`](https://www.youtube.com/@TruckDriverLoL) | Truck Driver | NA | без вебкамеры, минимум оверлеев, 1080p30 |
| [`@hopeeuwlol`](https://www.youtube.com/@hopeeuwlol) | hope | EUW | без вебкамеры, полные игры ~27 мин |
| [`@sandevey`](https://www.youtube.com/@sandevey) | Dopa | CN | китайский клиент и вебкамера: другой интерфейс |
| [`@ShokLeague`](https://www.youtube.com/@ShokLeague) | Shok | NA | миникарта слева, HUD сдвинут: нестандартная раскладка |

Порядок проверки: сначала каналы без вебкамеры (`BurstBrand`,
`TruckDriverLoL`, `hopeeuwlol`, `Tinjus`) — у них отличаются только HUD
клиента и зум, перекрытий нет. Затем стримы с вебкамерой (`Nemesis2_lol`,
архивы про-игроков). Затем китайский клиент (`sandevey`, `CNSniper`) и
нестандартная раскладка (`ShokLeague`).

### Оговорки

- Вебкамеру, оверлеи и HUD оценивали по нескольким кадрам на канал.
  Стримеры двигают вебкамеру, а архивы перезаливов смешивают раскладки
  разных стримеров. В «Pro … Archive» вебкамера и логотипы спонсоров стоят в
  правом нижнем углу и у части игроков закрывают миникарту.
- Масштаб HUD и размер миникарты не измерялись. Как и для сети «Challenger
  Replays», каждому каналу нужна визуальная проверка раскладки до того, как
  его кадры попадут в датасет.
- Перезаливы чужих стримов могут удалить: перед сбором проверить, что ролик
  ещё доступен.
- У каналов с `source_tab: streams` полезные записи лежат на вкладке
  «Трансляции», а смотрелка каналов (`tools/channels.py`) пока листает только
  вкладку «Видео».
