# Dataset

Закоммиченная часть данных: раскладки HUD, проекция экрана на карту,
каталог каналов и манифесты опорных кадров нескольких матчей. Сами кадры
в репозиторий не входят (права на них у Riot Games и авторов роликов, см.
`NOTICE.md`): они пересобираются локально из публичного ролика и лежат в
`videos/<id>/frames/`, который в `.gitignore`. Обучающий набор
«кадр → координаты камеры» сюда не входит: его собирает
`tools/build_coords.py` в `data/coords/` (в Git не идёт).

## Состав

```
dataset/
  layouts/<layout>.json        маска HUD и миникарты в долях кадра, рамка миникарты
  layouts/projection.json      гомография «пиксели сцены → смещение на карте»
  sources/                     каталог YouTube-каналов (см. sources/README.md)
  label_noise/                 ручные вердикты «метка ↔ модель» по ключевым кадрам
  videos/<video_id>/
    source.json                источник, длительность, параметры выборки
    frames.csv                 frame, video_id, t_sec, timecode, note
    frames/<id>_t<сек>.jpg     полный кадр 1920×1080 без маски (локально, не в Git)
```

## Опорные кадры (`videos/`)

Полные кадры каждые 15 с, без маски: на них проверяют раскладку маски и
чтение рамки камеры с миникарты (`tools/minimap_camera.py`), их же берёт
`tools/inspect_patches.py`. Колонка `note` — свободные заметки о кадре
(чёрный кадр, оверлеи ролика поверх сцены, где стоит камера); она
необязательна.

| video_id | Источник | Длина | Кадров |
| --- | --- | --- | --- |
| `58w57eJ5Qks` | [Volibear Top vs Irelia — KR Master 26.19](https://www.youtube.com/watch?v=58w57eJ5Qks), Volibear Challenger Replays | 23:25 | 94 |
| `ibUVbSX7ARU` | [Ekko Top vs Zac — KR Master 26.19](https://www.youtube.com/watch?v=ibUVbSX7ARU), Ekko Challenger Replays | 31:02 | 125 |
| `olmTXkkUv58` | [Ekko Jungle vs Skarner — KR Master 26.19](https://www.youtube.com/watch?v=olmTXkkUv58), Ekko Challenger Replays | 23:25 | 94 |
| `zJvTSjEnKNE` | [Ekko Mid vs Zed — KR Grandmaster 26.19](https://www.youtube.com/watch?v=zJvTSjEnKNE), Ekko Challenger Replays | 25:17 | 102 |
| `0Jijr5p5gAg` | [Jinx ADC vs Draven — KR Grandmaster 26.19](https://www.youtube.com/watch?v=0Jijr5p5gAg), Jinx Challenger Replays | 23:19 | 94 |
| `4AIX8QtRid4` | [Ahri Mid vs Cassiopeia — KR Grandmaster 26.19](https://www.youtube.com/watch?v=4AIX8QtRid4), Ahri Challenger Replays | 34:14 | 137 |

Раньше в `frames.csv` была колонка `label` с ручными метками
`top`/`not_top`/`unknown` для классификатора верхней линии. Классификатор
снят с проекта, колонка удалена; прежние метки остались в истории Git.

Восстановить кадры ролика из манифеста или добавить новый ролик
(`format_id` уже собранного ролика есть в его `source.json`):

```
python -m yt_dlp -f "bv*[height<=1080]+ba/b" --merge-output-format mp4 \
  --write-info-json -o "data/videos/%(id)s.%(ext)s" <url>
python tools/extract_frames.py data/videos/<id>.mp4 --info data/videos/<id>.info.json --every 15
```

Видео лежит в `data/` (не в Git). Повторный запуск не перезаписывает
существующие кадры и сохраняет заметки.

## Игра / не игра (`not_game/`)

Разметка роликов стримеров и других игр для головы «игра / не игра»
(`tools/train_not_game.py`, подробности — `docs/not-game.md`):

```
not_game/
  split.json                 какие ролики для обучения, какие отложены
  <video_id>/source.json     источник, интервалы выборки кадров
  <video_id>/labels.csv      start_sec, end_sec, label, kind, note
```

`label` — `game`, `not_game` или `skip` (переход, в обучение и оценку не
входит); `kind` — подкласс: `lol_game_pov`, `lol_game_esports`,
`lol_client`, `lol_loading`, `webcam`, `intro`, `other_game_tft` и т. д.
Интервал — полуоткрытый `[start_sec, end_sec)`. Кадры не хранятся:
`tools/build_not_game.py --url <id>` собирает их потоком в
`data/not_game/` по интервалам из `source.json`.

## Готовность заклинаний D/F по HUD (`hud_spells/`)

Ролики вида игрока и разметка состояний слотов D/F по времени для читателя
готовности заклинаний (`lolvision/hud_spells.py`, `tools/hud_spells.py`,
подробности — `docs/hud-spells.md`, #806):

```
hud_spells/
  panel_template.png         эталон панели умений: медиана 32 кадров СВОЕЙ игры
                             (1920×1080, только панель HUD, без ников)
  <video_id>/source.json     источник ролика, лицензия, выборка (вырез низа кадра, 1 кадр/с)
  <video_id>/labels.csv      slot, start_sec, end_sec, state, note
  <video_id>/spells.csv      slot, start_sec, end_sec, spell, note — семья заклинания слота
```

`state` — `ready`, `cooldown`, `blocked` (крестик или все иконки серые),
`dead`, `hidden` (панели нет), `?` (не разобрать — в счёт не идёт).
Интервал полуоткрытый, поздние строки сильнее ранних: в начале — черновик
грубым правилом (`note` = «черновик»), в конце — правки глазами (`note`
начинается с «глазами»). Видео — `data/videos/<id>.mp4` (yt-dlp, как выше),
вырезы — `python tools/hud_spells.py extract <id>` в `data/hud_spells/`.

## Полоса статистики HUD: CS и время матча (`hud_stats*`)

Образцы для читателя CS и времени матча с полосы вида игрока
(`lolvision/hud_stats.py`, `tools/hud_stats.py`, подробности —
`docs/last-hits.md`, #816):

```
hud_stats_digits.npz            glyphs (N×12×8, 0..1), labels ('0'…'9', ':', 'x' — мусор):
                                до 80 цифр на группу k-средних по роликам подбора
hud_stats_icon.png              значок миньона полосы (серый, медиана кадров подбора)
hud_stats/cluster_labels.txt    подпись каждой из 40 групп, глазами по листу групп
```

Цифры вырезаны из кадров тех же 10 роликов вида игрока, что у
`hud_spells/` (только ролики подбора); кадры не хранятся — полоса
вырезается `python tools/hud_stats.py strips` в `data/hud_stats/`.

## Вердикты по шуму меток (`label_noise/`)

Ручная проверка ключевых кадров «метка с миникарты ↔ модель ↔ траектория по
сцене» (`tools/label_noise_page.py`, разбор — `docs/label-noise.md`):

```
label_noise/
  verdicts.csv          вердикты человека (выгрузка CSV со страницы)
  verdicts_agent.csv    предварительные вердикты агента — не ручная разметка
```

Колонки: `video_id, idx, t_sec, stratum, also_random, x, y, verdict, note`
(у агента нет `x, y`, зато есть `who`). `idx` — номер кадра плотного набора
`data/coords/<video_id>.csv` (1 кадр/с, `t_sec` — секунда ролика).
`stratum` — откуда кадр: `model_vs_label` (большое расхождение модели с
меткой), `label_vs_traj` (большой остаток метки против траектории),
`clip_vs_label` (рамка с учётом края миникарты расходится с меткой),
`random` (равномерная выборка кадров отложенных роликов с качеством ≥ 0.4;
`also_random = 1` — кадр другой страты, выпавший и в случайной). `x, y` —
центр белой рамки, поставленный человеком вслепую, в единицах меток (как
`lx, ly`); пусто, если кадр не разобрать. `verdict` выводится из `x, y`:
`label` / `model` — ближе 1.5 пикселя миникарты 256 только метка / только
модель, `both_ok` — обе, `both_wrong` — ни одна, `unclear` — не разобрать.
Набор кадров восстанавливается командой `python tools/label_noise_page.py`
(seed фиксирован).

## Раскладка `spectator-volibear-challenger`

Единственная проверенная раскладка. Закрывает верхний счёт и значки
объектов, боковые панели и вступительные карточки чемпионов, чат,
центральные объявления об убийствах/объектах, ленту убийств справа и
таймеры. Нижняя полоса закрыта целиком: миникарта, характеристики игрока,
таблица команд, порядок способностей/предметов, подписи канала и итоговая
статистика победы.

Одна и та же маска применяется ко всем кадрам. Поэтому боковые области под
редкие вступительные карточки закрыты и на обычных игровых кадрах: часть
рельефа тоже скрывается, но форма маски не служит подсказкой о фазе матча.
При 1920×1080 маска закрывает около 63% полного кадра. Полоски здоровья,
имена над персонажами и эффекты внутри сцены остаются.

Координаты округляются наружу, чтобы не оставлять край панели. Маска
масштабируется с кадром 16:9; другой формат отклоняется. Для другого
канала, масштаба HUD или обрезанного видео нужна своя проверенная раскладка.

Поле `labelling.minimap` — рамка миникарты. По ней `minimap_camera.py`,
`build_coords.py` и `live.py` снимают положение камеры — цель обучения.
Во вход модели миникарта не попадает: её закрывает маска
`bottom_overlays_and_minimap`.

### Вход модели

`build_coords.py` применяет маску, обрезает кадр по плотной незакрытой
области (`crop_box(..., "dense")` в `lolvision/mask_frames.py`) и хранит сцену
шириной 720 px; размер входа модели задаётся при обучении (`--size`, по
умолчанию 384×200). Проверить маску глазами на опорных кадрах:

```bash
python tools/mask_frames.py dataset/layouts/spectator-volibear-challenger.json \
  dataset/videos/58w57eJ5Qks/frames/*.jpg --out runs/checks/masks/58w57eJ5Qks
```

Результат — PNG с исходными именами: PNG сохраняет маску точно чёрной, без
артефактов JPEG на границах.

Именно эта раскладка и эта обрезка — слабое место текущей модели: опыт
2026-10-09 показал, что CoordNet привязан к форме маски и положению сцены
на экране (подробности — в корневом README).

## Проекция (`layouts/projection.json`)

Гомография «пиксели хранимой сцены (720×372) → смещение от центра камеры в
долях карты». Подобрана `tools/calibrate_projection.py` по движению камеры
между соседними кадрами. Её читают смотрелки: трапеция видимой области и
сетка ячеек на карте.
