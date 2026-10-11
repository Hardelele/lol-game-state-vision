import numpy as np

from lolvision.geometry import MAP_UNITS
from lolvision.unit_tracker import Projection, UnitTracker, lane_of

# Кадр 1000×1000 целиком — «сцена» 100×100; экран → смещение от камеры:
# 0.001 доли карты на пиксель сцены, то есть 0.1 карты на весь кадр
# (~1.5 ед. на пиксель кадра), центр кадра — центр камеры.
H = np.array([[0.001, 0, -0.05], [0, 0.001, -0.05], [0, 0, 1.0]])
U_PER_PX = 0.1 * MAP_UNITS / 1000


def tracker():
    return UnitTracker(Projection(H, (100, 100), (0, 0, 1000, 1000)), frame_size=(1000, 1000))


def minion(x, y, hp=1.0, team="ORDER"):
    return ["minion", team, x / 1000, y / 1000, hp, 0.9]


def seen(tr):
    return {t.id: t for t in tr.tracks if t.state == "seen"}


def test_projection_roundtrip():
    p = Projection(H, (100, 100), (0, 0, 1000, 1000))
    xy = np.array([[500.0, 500.0], [100.0, 900.0]])
    uv = p.to_map(xy, (0.4, 0.6))
    assert np.allclose(uv[0], (0.4, 0.6))
    assert np.allclose(p.to_frame(uv, (0.4, 0.6)), xy)


def test_same_minion_keeps_id_while_camera_moves():
    tr = tracker()
    # Миньон идёт вправо 300 ед./с, камера за 1 с сдвигается на 0.01 карты
    # вправо — на экране он сдвигается меньше, чем на карте.
    ids = []
    for k in range(6):
        cam = (0.5 + 0.01 * k, 0.5)
        world_x = 0.5 + k * 300 / MAP_UNITS
        x = 500 + (world_x - cam[0]) * MAP_UNITS / U_PER_PX
        tr.update(float(k), cam, [minion(x, 500)])
        ids.append(next(iter(seen(tr)), None))
    assert ids[0] is None                      # первая встреча — ещё не подтверждён
    assert len(set(ids[1:])) == 1 and ids[1].endswith(tuple("abcdefghijklmnopqrstuvwxyz0123456789"))


def test_low_hp_vanishing_in_view_is_death_not_loss():
    tr = tracker()
    tr.update(0.0, (0.5, 0.5), [minion(500, 500, 0.6)])
    tr.update(1.0, (0.5, 0.5), [minion(500, 500, 0.15)])
    ev = tr.update(2.0, (0.5, 0.5), [])
    assert [e["type"] for e in ev] == ["unit_death"]
    assert not tr.tracks and tr.gone[0].state == "dead"


def test_out_of_view_is_unseen_and_found_again_with_same_id():
    tr = tracker()
    tr.update(0.0, (0.5, 0.5), [minion(500, 500)])
    tr.update(1.0, (0.5, 0.5), [minion(500, 500)])
    tid = next(iter(seen(tr)))
    # Камера ушла далеко: миньон вне кадра — не мёртв, а «не виден».
    assert tr.update(2.0, (0.9, 0.9), []) == []
    assert tr.tracks[0].state == "unseen"
    # Камера вернулась, миньон рядом с прежним местом — прежний id.
    tr.update(4.0, (0.5, 0.5), [minion(520, 500)])
    assert list(seen(tr)) == [tid]


def test_hp_rise_does_not_match_minion():
    tr = tracker()
    tr.update(0.0, (0.5, 0.5), [minion(500, 500, 0.2), minion(560, 500, 1.0)])
    tr.update(1.0, (0.5, 0.5), [minion(500, 500, 0.2), minion(560, 500, 1.0)])
    before = {round(t.hp, 1): t.id for t in tr.tracks}
    # Те же двое, сместились навстречу друг другу на полпути: ХП решает, кто есть кто.
    tr.update(2.0, (0.5, 0.5), [minion(530, 500, 0.18), minion(531, 500, 0.95)])
    after = {round(t.hp, 1): t.id for t in tr.tracks}
    assert before[0.2] == after[0.2] and before[1.0] == after[0.9]


def test_champion_gets_named_id_when_name_is_read():
    tr = tracker()
    tr.update(0.0, (0.5, 0.5), [["champion", "CHAOS", 0.5, 0.5, 1.0, 0.9, None]])
    tr.update(1.0, (0.5, 0.5), [["champion", "CHAOS", 0.51, 0.5, 1.0, 0.9, "Volibear"]])
    assert [t.id for t in tr.tracks] == ["champ-CHAOS-Volibear"]


def test_lane_of():
    assert lane_of(0.5, 0.5) == "mid"
    assert lane_of(0.05, 0.4) == "top" and lane_of(0.6, 0.95) == "bot"
    assert lane_of(0.3, 0.45) == "jungle"
