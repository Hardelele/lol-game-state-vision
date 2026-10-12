# Записи своей игры (2D, #800)

Проверочный набор этапа 2 (docs/mvp-state-service.md, 2D): кадры своего
экрана и журнал Live Client API той же игры, за обе стороны. Пишет
`tools/record_own_game.py`.

## Как записывать

1. Зайти в игру (Practice Tool или обычную), окно игры —
   `League of Legends (TM) Client` на основном мониторе (dxcam видит только
   его; иначе — `--backend mss`).
2. В корне репозитория:

   ```bash
   python tools/record_own_game.py              # 2 кадра/с, до 40 мин
   python tools/record_own_game.py --fps 4 --max-min 25
   ```

   Запускать можно до входа в игру: пока окна нет, идут отказы
   `no_window`, API — `no_game`.
3. Остановка: Ctrl+C (манифест дописывается), `--max-min` минут, или сама —
   через `--end-grace` секунд (120) после конца игры по API (`GameEnd` или
   пропажа API после игры — решает `live_client.Tracker`).

Раз в 10 с в консоли: кадров, отказов, состояние API, время матча. Нужно 3–5
записей: за ORDER (синие, низ слева) и за CHAOS (красные, верх справа).

## Что в записи

`data/own/<YYYYMMDD-HHMMSS>/` (занятый id — `-2`, `-3`, ничего не
перезаписывается):

| Файл | Что |
| --- | --- |
| `full/<i:06d>.jpg` | Полный кадр клиентской области окна, JPEG качество 90, размер как есть (не 1080p — тоже). `i` — номер ответа источника, на отказах файла нет — номера с пропусками |
| `frames.jsonl` | Строка на каждый ответ источника: `i`, `t` (unix-время снимка), `src_t` (время по шкале источника), `status` (`ok` / `paused` / `minimized` / `no_window` / `no_frame`), `reason` (у отказа), `file`, `size` `[W, H]` |
| `api.jsonl` | Журнал `allgamedata` ~1 Гц: `{"t": unix, "code": 200 \| текст ошибки, "body": текст \| null}` — читает `live_client.read_journal`, прогоняет `live_client.replay` |
| `manifest.json` | Сводка (атомарно, раз в ~30 с и в конце): `schema` (1), `id`, `started`/`finished` (unix и ISO), `complete`, `stop_reason`, `fps`, `capture` (бэкенд и почему он), `frames` (ответов, кадров, отказов по состоянию, размеры), `api` (строк, последнее состояние, `my_team`, `my_champion` Id Data Dragon, `game_mode`, `game_time`, `ended_reason` — причина конца: `GameEnd <Result>` из последнего снимка игры, даже если порт уже закрылся к моменту, когда запись прочла статус; иначе причина статуса API) |

`complete: false` в манифесте — запись прервана без финала (процесс убит):
кадры и журналы до этого места целы.

Кадр сопоставляется с API по времени: строка журнала с ближайшим `t` не
позже `frames.jsonl[i].t`. Сторона своего игрока — `manifest.api.my_team`
(или `allPlayers[me].team` в журнале).

## Как проверять

Как в docs/sides.md, «Как проверить на записях своей игры»:

```bash
python tools/eval_units.py --frames "data/own/<запись>/full/*.jpg" --no-mask --name own-<сторона>
python tools/eval_sides.py --preds runs/units/eval/own-<сторона>/preds.jsonl --my-team <ORDER|CHAOS> \
    --sheet runs/units/eval/own-<сторона>/sides.jpg --root <корень с data/>
```

Записи лежат в `data/` (в Git не идут); в Git — только манифест и разметка.
