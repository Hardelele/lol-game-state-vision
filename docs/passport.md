# Паспорт трансляции (#799, минимум #781)

`lolvision/passport.py` — объект ожиданий сеанса live: что за источник,
какой кадр, где миникарта, чей вид, есть ли Live Client API — и какие
читатели при этом имеют право отвечать. Не выполнено требование читателя —
его выход `null` + причина (docs/mvp-state-service.md, «Отказ лучше
выдумки»). Почему это нужно: docs/pitfalls.md, 0.3 — читатели наблюдателя на
виде игрока молча выдавали мусор (#786), истина камеры по раскладке у
стримеров с крупной миникартой съезжала (#785).

Паспорт — только данные: он ничего не меряет, не ходит в сеть и не снимает
экран. Всё мерит сеанс и сообщает паспорту сигналом.

## Пункты

Каждый пункт — `Item`: `value`, `source` (откуда), `how` (как получено или
почему значения нет), `t` (секунды потока).

| Пункт | Значение | Источник |
| --- | --- | --- |
| `source` | `screen` / `youtube` / `file` | `source` — `FrameSource.describe()` |
| `resolution` | `[W, H]` или `null` | `source` |
| `game_rect` | `[x0, y0, x1, y1]`, пиксели кадра | `source` у экрана (кадр — окно игры); `assumed` у ролика (кадр целиком, раскладка стрима не разобрана); `session`, если задан сеансом |
| `minimap` | `{found, box, scale, score}` | `locator` (найдена или потеряна по совпадению); `layout` (поиск сдался); `none` (ещё не искали) |
| `view` | `spectator` / `player` / **`unknown`** | `hud` (LiveChampions.view); `none` — не решён |
| `api` | `{state, available}` | `api`; `none` — клиента API нет |
| `scene` | тип сцены последнего кадра | `scene` |
| `roster` | `api` / `hud` / `null` | откуда известен состав |
| `assets` | `{unit_net, names_bank, minimap_model: bool}` | `session`; чего нет в словаре — «нет сведений», и читатель отказывает |

`unknown` — честное значение: пока раскладка не решена (мало чтений HUD) или
не распознана, вид неизвестен, и никто из зависящих от вида не отвечает.

## Читатели и требования

Требования проверяются по порядку; первое невыполненное — причина отказа.

| Читатель | Что | Требования |
| --- | --- | --- |
| `camera_truth` | истина камеры по рамке на миникарте | размер кадра; миникарта найдена **или** вид наблюдателя |
| `units` | детектор юнитов | размер кадра; есть `unit_net`; вид наблюдателя или игрока (маска своя у каждого вида) |
| `names` | имена над плашками | вид наблюдателя; есть `names_bank`; состав известен |
| `hud_roster` | состав по портретам HUD | размер кадра; вид наблюдателя |
| `minimap_icons` | иконки чемпионов на миникарте | размер кадра; есть `minimap_model`; миникарта найдена или вид наблюдателя; состав известен |
| `sides` | свои / враги / сам (#790) | вид игрока; API доступен (своя сторона — из API) |
| `api_facts` | факты Live Client API | API доступен |

Что выходит по раскладкам (тесты `tests/test_passport.py`):

* **вид игрока, своя игра, API в игре** — отключены `names`, `hud_roster`
  («нужен вид наблюдателя, а это вид игрока»); `camera_truth` и
  `minimap_icons` — пока миникарта не найдена;
* **наблюдатель (YouTube), API нет** — отключены `sides` («нужен вид
  игрока») и `api_facts` («Live Client API: не подключён»);
* **неизвестная раскладка** — отключены все: зависящие от вида — «вид
  неизвестен», `camera_truth` и `minimap_icons` — пока миникарта не найдена;
  после находки миникарты `camera_truth` отвечает.

## Сигналы пересчёта

Пересчёт — по сигналу, не на каждом кадре. Каждый сигнал возвращает список
`Change` (`key` = `item:<пункт>` или `reader:<читатель>`, `old`, `new`, `why`,
`t`; `text` — строка для note). Читатель попадает в изменения только при
включении или отключении; смена слов причины — нет. Пункты, которые сеансу
надо перемерить, — в `pending`.

| Сигнал | Метод | Что делает |
| --- | --- | --- |
| размер кадра сменился | `set_frame_size(size, t)` | миникарта снята, `game_rect` пересчитан; в `pending` — `minimap`, `game_rect` |
| миникарта найдена | `set_minimap(box, scale, score, t)` | `found`, снимает `minimap` из `pending` |
| перепроверка миникарты | `check_minimap(score, t)` | ниже `minimap_locate.MIN_SCORE` `LOST_AFTER` (= `AGREE`) раз подряд — потеряна; одна низкая — только `pending` |
| поиск сдался | `minimap_missing(t, how)` | остаётся раскладка (`layout`) |
| вид решён или сменился | `set_view(view, t, how)` | `None` → `unknown` |
| API появился / пропал | `set_api(state, available, t, reason)` | `available` решает сеанс (обычно `state == in_game`) |
| тип сцены сменился | `set_scene(kind, t, game)` | событие только на смене; возврат в игру после не-игры — найденную миникарту в `pending` на перепроверку |
| состав | `set_roster(source, t, how)` | `api` / `hud` / `None` |
| ресурс сеанса | `set_asset(name, ok, t)` | |

Перепроверку миникарты звать на игровом кадре: на загрузке и в клиенте
совпадение низкое у любой раскладки.

## Выход

`Passport.to_dict()` — поле `passport` снимка состояния:

```json
{"items": {"view": {"value": "player", "source": "hud", "how": "…", "t": 5.0}, "…": {}},
 "pending": ["minimap"],
 "disabled": [{"reader": "names", "title": "имена над плашками",
               "reason": "нужен вид наблюдателя, а это вид игрока"}],
 "enabled": ["units", "sides", "api_facts"]}
```

`gate(reader, value)` → `{"value": …, "reason": null}` или
`{"value": null, "reason": "…"}` — выход читателя в снимке.

## Подключение (следующая волна, live.py)

Не сделано в #799; что вызвать в `lolstream/live.py`:

* начало сеанса (после `source.open()`): `Passport.build(source.describe(), t,
  assets={"unit_net": unit_model is not None, "names_bank": champs.names is not None,
  "minimap_model": champs is not None})`;
* кадр с новым `Frame.size` — `set_frame_size`; по `pending` пересоздать
  `Locator` под новый размер;
* `locator.offer_region` вернул True — `set_minimap(locator.box,
  locator.fit.scale, locator.fit.score, t)`; `locator.tried >= LOCATE_MAX` —
  `minimap_missing`;
* `"minimap" in pending` и игровой кадр — `locator.locate(region)` в том же
  прямоугольнике и `check_minimap(fit.score, t)` (~40 мс, только по сигналу);
* смена `champs.view` — `set_view(champs.view, t, how=…)`;
* смена `LiveStatus.state` — `set_api(st.state.value, st.state is State.IN_GAME, t,
  st.reason)`;
* смена `kind` — `set_scene(kind, t, game)`; состав — `set_roster`;
* `_emit`: `truth` = `passport.verdict("camera_truth").ok`; имена, состав по HUD,
  иконки, стороны — через `gate`; `status()` — `passport.to_dict()`, а `Change.text`
  — в note и поток событий.
