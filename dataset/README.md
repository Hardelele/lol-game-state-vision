# Dataset

Кадры YouTube-роликов с метками «что показывает камера»: `top`, `not_top`,
`unknown` (определения — в корневом README).

## Состав

```
dataset/
  layouts/<layout>.json        маска HUD и миникарты в долях кадра
  videos/<video_id>/
    source.json                источник, длительность, параметры выборки
    frames.csv                 frame, video_id, t_sec, timecode, label, note
    frames/<id>_t<сек>.jpg     полный кадр 1920×1080 без маски
```

Кадры хранятся без маски: маска применяется при подготовке входа модели
(`tools/mask_frames.py`), чтобы её можно было уточнять без перенарезки.

## Ролики

| video_id | Источник | Длина | Кадров | top / not_top / unknown | Раскладка |
| --- | --- | --- | --- | --- | --- |
| `58w57eJ5Qks` | [Volibear Top vs Irelia — KR Master 26.19](https://www.youtube.com/watch?v=58w57eJ5Qks), Volibear Challenger Replays | 23:25 | 94 (каждые 15 с, 00:00–23:15) | 52 / 36 / 6 | `spectator-volibear-challenger` |

## Как размечено

- Метка — по рамке камеры на миникарте того же кадра: рамка на верхней
  или левой линии — `top`; база, лес, мид, бот, Барон — `not_top`.
- `unknown` — нет игры (чёрный кадр) или рамка на стыке топа с верхней
  рекой/лесом, а сцена не даёт уверенно отнести её к линии (туман войны,
  вода у логова Барона).
- Оверлеи ролика поверх сцены (табло рун, карточки чемпионов, экран
  победы) метку не меняют, но отмечены в `note`.
- Разметка первичная: сделана Claude (claude-opus-5-5) 2026-10-07 по
  листам «сцена + миникарта», человеком не проверена. Граничные кадры
  (`unknown` и `note` с «границей») — первые кандидаты на ревью.

## Как добавить ролик

```
python -m yt_dlp -f "bv*[height<=1080]+ba/b" --merge-output-format mp4 \
  --write-info-json -o "data/videos/%(id)s.%(ext)s" <url>
python tools/extract_frames.py data/videos/<id>.mp4 --info data/videos/<id>.info.json --every 15
```

Видео лежит в `data/` (не в Git). Повторный запуск не перезаписывает
существующие кадры и сохраняет уже проставленные метки.
