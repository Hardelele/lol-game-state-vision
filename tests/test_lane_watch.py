"""Ситуации линии (#816): нажатие умения с целью и окно размена.

Без роликов: треки — словари, как их отдаёт unit_tracker (to_dict), умения —
чтения слотов hud_spells (to_dict)."""

from __future__ import annotations

from lolvision import lane_watch as lw
from lolvision.geometry import MAP_UNITS

ME = (0.30, 0.70)


def at(du: float, dv: float = 0.0) -> tuple[float, float]:
    """Место в ед. карты от своего."""
    return ME[0] + du / MAP_UNITS, ME[1] + dv / MAP_UNITS


def track(id_, kind, relation, uv, hp=1.0, rate=0.0, seen=True):
    return {"id": id_, "kind": kind, "relation": relation, "u": uv[0], "v": uv[1],
            "x": 0.5 if seen else None, "y": 0.5 if seen else None, "hp": hp, "hp_rate": rate}


def ab(**v):
    return [{"slot": k, "value": v.get(k)} for k in "QWER"]


def test_ability_into_the_wave_without_champion_nearby():
    w = lw.LaneWatch()
    wave = [track(f"m{i}", "minion", "enemy", at(300 + 50 * i)) for i in range(3)]
    assert w.update(0.0, ab(Q=True), wave, ME) == []
    [ev] = w.update(0.5, ab(Q=False), wave, ME)
    assert ev["type"] == "ability_cast" and ev["slot"] == "Q" and ev["target"] == "wave"
    assert ev["wave"] == 3 and ev["enemy_dist"] is None


def test_ability_with_enemy_champion_close_is_a_trade_not_the_wave():
    w = lw.LaneWatch()
    tr = [track("m1", "minion", "enemy", at(300)), track("m2", "minion", "enemy", at(350)),
          track("champ", "champion", "enemy", at(600))]
    w.update(0.0, ab(E=True), tr, ME)
    [ev] = w.update(0.5, ab(E=False), tr, ME)
    assert ev["target"] == "champion" and ev["enemy_dist"] == 600


def test_unknown_reading_does_not_fake_a_cast_and_unknown_place_says_so():
    w = lw.LaneWatch()
    w.update(0.0, ab(W=True), [], ME)
    w.update(0.5, ab(W=None), [], ME)              # «не знаю» между — не нажатие
    [ev] = w.update(1.0, ab(W=False), [], None)
    assert ev["target"] == "unknown" and ev["why"] == "своё место неизвестно"
    # Долгий перерыв между «готово» и «перезарядкой» — нажатие не засчитано.
    w2 = lw.LaneWatch()
    w2.update(0.0, ab(W=True), [], ME)
    assert w2.update(5.0, ab(W=False), [], ME) == []


def test_trade_window_opens_once_when_enemy_will_last_hit_our_creep():
    w = lw.LaneWatch()
    enemy = track("champ", "champion", "enemy", at(700))
    creep = track("ally-m", "minion", "ally", at(500), hp=0.6, rate=-0.2)   # 1,6 с до добивания
    [ev] = w.update(0.0, ab(Q=True, W=False, E=True), [enemy, creep], ME)
    assert ev["type"] == "trade_window" and ev["minion"] == "ally-m" and ev["enemy"] == "champ"
    assert ev["eta"] == 1.6 and ev["ready"] == ["Q", "E"] and ev["enemy_dist"] == 700
    assert w.window is not None
    # То же окно на следующем кадре — без нового события.
    assert w.update(0.25, ab(Q=True, E=True), [enemy, creep], ME) == []
    # Крип ещё долго жить будет — окна нет; враг далеко от крипа — тоже.
    slow = dict(creep, hp=0.9, rate=-0.05)
    assert w.update(0.5, ab(Q=True), [enemy, slow], ME) == [] and w.window is None
    far = track("champ", "champion", "enemy", at(2500))
    assert w.update(0.75, ab(Q=True), [far, creep], ME) == [] and w.window is None


def test_jungle_monster_taken_for_a_minion_opens_no_window():
    me = (0.50, 0.50)                              # на диагонали средней линии
    w = lw.LaneWatch()
    enemy = track("champ", "champion", "enemy", (0.50 + 400 / MAP_UNITS, 0.50))
    monster = track("jungle-minion-x", "minion", "ally", (0.50 + 300 / MAP_UNITS, 0.50), hp=0.2)
    off_lane = track("mid-minion-y", "minion", "ally", (0.475, 0.393), hp=0.2)   # лагерь пташек
    assert w.update(0.0, ab(Q=True), [enemy, monster, off_lane], me) == []
    lane = track("mid-minion-z", "minion", "ally", (0.50 + 300 / MAP_UNITS, 0.50), hp=0.2)
    [ev] = w.update(0.25, ab(Q=True), [enemy, lane], me)
    assert ev["minion"] == "mid-minion-z"
