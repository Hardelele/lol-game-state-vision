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
