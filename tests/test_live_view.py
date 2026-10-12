"""Раскладка ролика в потоке — только по положительным признакам (#799).

«Портретов наблюдателя не видно» — не вид игрока: оверлей стримера
«TEAM STATS» закрывал левую колонку портретов, и j8_-kWWh5qM шесть раз
переключался spectator ↔ player. Вид игрока — только по полосе здоровья
своей панели внизу по центру.
"""

from collections import deque

import numpy as np
import pytest

from lolstream import live_champions as lc
from lolvision.roster import FrameSlots

SPEC, PLAY = lc.SPECTATOR, lc.PLAYER


def champs(reads=()) -> lc.LiveChampions:
    """LiveChampions без моделей: только чтения раскладки."""
    obj = object.__new__(lc.LiveChampions)
    obj.reads = deque(reads, maxlen=lc.VIEW_READS)
    obj._view = None
    obj.match = None
    return obj


def feed(obj, reads) -> list:
    """Чтения по одному; вид после каждого."""
    out = []
    for r in reads:
        obj.reads.append(r)
        out.append(obj.view)
    return out


def slots(table=(0.5, 0.5), panels=(0.5, 0.5)) -> FrameSlots:
    """Корреляции кадра: по колонкам (синие, красные) таблицы и панелей."""
    sc = np.zeros((10, 2, 3), np.float32)
    for team in range(2):
        sc[5 * team:5 * team + 5, 0, 0] = table[team]
        sc[5 * team:5 * team + 5, 1, 0] = panels[team]
    return FrameSlots(sc)


def frame(bar=None) -> np.ndarray:
    """Тёмный кадр 1080p RGB; `bar` — (x, y, длина, высота) зелёной полосы."""
    f = np.full((1080, 1920, 3), 30, np.uint8)
    if bar is not None:
        x, y, n, h = bar
        f[y:y + h, x:x + n] = (40, 200, 60)
    return f


# Полоса здоровья своей панели вида игрока (замер: x 923–946, y 1037–1052,
# высота 8–13 px) и полоса выбранного чемпиона наблюдателя (x 76, 4–6 px).
PLAYER_BAR = (935, 1042, 130, 11)
SPECTATOR_BAR = (76, 865, 100, 5)


# ---------- признаки одного чтения ----------

def test_spectator_by_any_column_of_portraits():
    assert lc.spectator_hud(slots(table=(0.95, 0.95)))
    # Оверлей закрыл левую сторону, справа портреты видны — это наблюдатель.
    assert lc.spectator_hud(slots(table=(0.4, 0.95), panels=(0.3, 0.5)))
    assert lc.spectator_hud(slots(panels=(0.3, 0.82)))
    # Враги в тумане («?») — правая колонка таблицы не совпала, левая да;
    # медиана по десяти (roster.hud) этот кадр не принимает.
    fs = slots(table=(0.95, 0.5), panels=(0.75, 0.75))
    assert lc.spectator_hud(fs) and not fs.hud
    # Вид игрока: колонки не выше 0.68 / 0.64 (замер) — не наблюдатель.
    assert not lc.spectator_hud(slots(table=(0.68, 0.63), panels=(0.64, 0.6)))


def test_player_by_own_health_bar_only():
    assert lc.player_hud(frame(PLAYER_BAR))
    assert not lc.player_hud(frame())                        # нет полосы
    assert not lc.player_hud(frame(SPECTATOR_BAR))           # тонкая и в углу
    assert not lc.player_hud(frame((935, 1042, 40, 11)))     # короткая (мало здоровья)
    assert not lc.player_hud(frame((935, 1042, 130, 4)))     # тонкая: полоса юнита
    assert not lc.player_hud(frame((300, 1042, 130, 11)))    # не у центра
    assert not lc.player_hud(frame((935, 600, 130, 11)))     # не внизу


def test_player_bar_scales_with_frame():
    small = frame(PLAYER_BAR)[::2, ::2].copy()               # 960×540
    assert lc.player_hud(small)


def test_read_kind_absence_is_not_player():
    assert lc.read_kind(slots(), frame()) is None
    assert lc.read_kind(slots(), frame(PLAYER_BAR)) == PLAY
    # Портреты сильнее зелёной полосы.
    assert lc.read_kind(slots(table=(0.95, 0.95)), frame(PLAYER_BAR)) == SPEC


# ---------- раскладка по последовательности чтений ----------

def test_start_without_reads_is_unknown_not_player():
    # j8_-kWWh5qM: «0 из 4 чтений» давало player на 16 с потока.
    obj = champs()
    assert obj.view is None
    assert feed(obj, [None] * lc.VIEW_READS * 2) == [None] * lc.VIEW_READS * 2
    assert obj.view_how().startswith("раскладка не решена")


def test_spectator_survives_overlay_over_left_column():
    # Наблюдатель; оверлей «TEAM STATS» закрыл левую колонку: при правой
    # колонке чтение — всё ещё наблюдатель, при закрытой таблице — «ничего».
    obj = champs()
    views = feed(obj, [None] + [SPEC] * 3)
    assert views == [None, None, None, SPEC]
    covered = lc.read_kind(slots(table=(0.4, 0.95), panels=(0.3, 0.8)), frame())
    hidden = lc.read_kind(slots(table=(0.4, 0.5), panels=(0.3, 0.6)), frame())
    assert (covered, hidden) == (SPEC, None)
    assert set(feed(obj, [covered, hidden] * 4 + [hidden] * 20)) == {SPEC}
    assert obj.view_how().startswith("HUD:")


def test_spectator_to_player_only_by_player_signs():
    obj = champs([SPEC] * lc.VIEW_READS)
    assert obj.view == SPEC
    assert set(feed(obj, [PLAY] * (lc.VIEW_FLIP - 1))) == {SPEC}
    assert feed(obj, [PLAY])[-1] == PLAY
    # И обратно — только по портретам, «ничего» вид игрока не снимает.
    assert set(feed(obj, [None] * 20)) == {PLAY}
    assert set(feed(obj, [SPEC] * (lc.VIEW_FLIP - 1))) == {PLAY}
    assert feed(obj, [SPEC])[-1] == SPEC


def test_real_player_view_with_positive_sign():
    # Полоса здоровья видна не на каждом кадре (мало здоровья, смерть):
    # «ничего» между ними не мешает.
    obj = champs()
    views = feed(obj, [PLAY, None, PLAY, None, PLAY])
    assert views == [None, None, None, None, PLAY]


@pytest.mark.parametrize("reads", [
    [SPEC, PLAY, SPEC, PLAY, SPEC, PLAY],            # спорно — не решаем
    [SPEC, SPEC, None, None, None, None, None, None],  # мало голосов
    [PLAY, PLAY, None, None, None, None, None, None],
])
def test_mixed_or_scarce_signs_stay_unknown(reads):
    obj = champs()
    assert set(feed(obj, reads)) == {None}


def test_api_view_wins_over_reads():
    from types import SimpleNamespace
    obj = champs([PLAY] * lc.VIEW_READS)
    obj.match = SimpleNamespace(api=True, me_team=None)
    assert obj.view == SPEC
