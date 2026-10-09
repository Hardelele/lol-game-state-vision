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
области (`crop_box(..., "dense")` в `tools/mask_frames.py`) и хранит сцену
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
