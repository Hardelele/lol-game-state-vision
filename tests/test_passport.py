"""Паспорт трансляции: какие читатели отвечают на виде игрока, наблюдателя и
неизвестной раскладке; сигналы пересчёта; отсутствие Live Client API."""

from __future__ import annotations

import json

import pytest

from lolvision.minimap_locate import MIN_SCORE, STD_BOX
from lolvision.passport import LOST_AFTER, READERS, Passport

SCREEN = {"kind": "screen", "backend": "dxcam", "backend_note": None,
          "fps": 2.0, "size": (1920, 1080), "title": "League of Legends (TM) Client"}
YOUTUBE = {"kind": "youtube", "backend": "ffmpeg", "backend_note": None,
           "fps": 1.0, "size": (1280, 720), "title": "VOD"}
ALL_ASSETS = {"unit_net": True, "names_bank": True, "minimap_model": True}
BIG = (0.83, 0.70, 0.993, 0.984)


def keys(changes) -> set[str]:
    return {c.key for c in changes}


def player_with_api() -> Passport:
    """Своя игра: экран, вид игрока, API в игре, состав из API."""
    p = Passport.build(SCREEN, t=0.0, assets=ALL_ASSETS)
    p.set_view("player", 5.0, how="HUD: таблица не совпала")
    p.set_api("in_game", True, 5.0)
    p.set_roster("api", 5.0)
    return p


def spectator_stream() -> Passport:
    """YouTube наблюдателя Volibear: вид наблюдателя, состав по HUD, API нет."""
    p = Passport.build(YOUTUBE, t=0.0, assets=ALL_ASSETS)
    p.set_view("spectator", 20.0, how="HUD: таблица 0.97")
    p.set_roster("hud", 20.0)
    return p


# ---------------------------------------------------------------- три раскладки


def test_player_view_disables_spectator_readers():
    p = player_with_api()
    dis = p.disabled()
    assert set(dis) == {"names", "hud_roster", "camera_truth", "minimap_icons"}
    assert "нужен вид наблюдателя, а это вид игрока" in dis["names"]
    assert "нужен вид наблюдателя, а это вид игрока" in dis["hud_roster"]
    # Миникарта ещё не найдена: по раскладке наблюдателя она мимо настоящей.
    assert "миникарта не найдена" in dis["camera_truth"]
    assert "вид игрока" in dis["camera_truth"]
    assert p.verdict("sides").ok and p.verdict("api_facts").ok and p.verdict("units").ok


def test_player_view_with_found_minimap_gets_camera_and_icons():
    p = player_with_api()
    ch = p.set_minimap(BIG, 1.2, 0.61, 7.0)
    assert {"item:minimap", "reader:camera_truth"} <= keys(ch)
    assert p.verdict("camera_truth").ok and p.verdict("minimap_icons").ok
    assert "minimap" not in p.pending


def test_spectator_view_without_api():
    p = spectator_stream()
    dis = p.disabled()
    assert set(dis) == {"sides", "api_facts"}
    assert "нужен вид игрока" in dis["sides"]
    assert dis["api_facts"].startswith("Live Client API: не подключён")
    # У наблюдателя раскладка и есть миникарта: истина камеры — без находки.
    assert p.verdict("camera_truth").ok
    assert p.verdict("names").ok and p.verdict("hud_roster").ok
    assert p.verdict("minimap_icons").ok


def test_unknown_view_refuses_view_dependent_readers():
    p = Passport.build(YOUTUBE, t=0.0, assets=ALL_ASSETS)
    assert p.items["view"].value == "unknown"
    dis = p.disabled()
    assert set(dis) == set(READERS)            # без вида, состава и API — никто
    for r in ("units", "names", "hud_roster", "sides"):
        assert "вид неизвестен" in dis[r], r
    assert "вид неизвестен" in dis["camera_truth"]
    # Иконкам первым мешает миникарта, имена упираются в вид раньше состава.
    assert "миникарта не найдена" in dis["minimap_icons"]


def test_unknown_view_after_found_minimap_keeps_camera():
    p = Passport.build(YOUTUBE, t=0.0, assets=ALL_ASSETS)
    p.set_minimap(BIG, 1.2, 0.55, 3.0)
    assert p.verdict("camera_truth").ok
    assert not p.verdict("units").ok


def test_view_none_is_unknown_and_bad_view_raises():
    p = spectator_stream()
    ch = p.set_view(None, 30.0, how="HUD скрыт клавишей H")
    assert p.items["view"].value == "unknown"
    assert {"item:view", "reader:names", "reader:hud_roster", "reader:units"} <= keys(ch)
    with pytest.raises(ValueError):
        p.set_view("esports", 31.0, how="?")


def test_missing_assets_are_refusals_not_hopes():
    p = Passport.build(YOUTUBE, t=0.0, assets={"names_bank": False})
    p.set_view("spectator", 1.0, how="HUD")
    p.set_roster("hud", 1.0)
    assert p.verdict("names").reason == "нет: банк имён над плашками"
    assert p.verdict("units").reason == "нет сведений, есть ли детектор юнитов"
    ch = p.set_asset("unit_net", True, 2.0)
    assert "reader:units" in keys(ch) and p.verdict("units").ok
    with pytest.raises(ValueError):
        Passport.build(YOUTUBE, assets={"ocr": True})


def test_gate_returns_null_with_reason():
    p = player_with_api()
    assert p.gate("names", ["Ahri"]) == {
        "value": None, "reason": p.verdict("names").reason}
    assert p.gate("api_facts", 42) == {"value": 42, "reason": None}
    ok, why = p.verdict("sides")
    assert ok and why is None


# ---------------------------------------------------------------- сигналы


def test_api_appears_and_disappears():
    p = Passport.build(SCREEN, t=0.0, assets=ALL_ASSETS)
    p.set_view("player", 1.0, how="HUD")
    assert not p.verdict("sides").ok
    ch = p.set_api("in_game", True, 2.0)
    assert {"item:api", "reader:sides", "reader:api_facts"} == keys(ch)
    assert next(c for c in ch if c.key == "item:api").why == "API появился"
    assert p.set_api("in_game", True, 2.5) == []          # без смены — без событий
    ch = p.set_api("no_game", False, 3.0, reason="порт 2999 закрыт")
    assert {"reader:sides", "reader:api_facts"} <= keys(ch)
    assert next(c for c in ch if c.key == "item:api").why == "API пропал: порт 2999 закрыт"
    assert p.verdict("api_facts").reason == "Live Client API: no_game — порт 2999 закрыт"
    ch = p.set_api(None, False, 4.0)
    assert p.verdict("api_facts").reason.startswith("Live Client API: не подключён")


def test_minimap_score_drop_needs_several_checks():
    p = player_with_api()
    p.set_minimap(BIG, 1.2, 0.6, 7.0)
    low = MIN_SCORE / 2
    for _ in range(LOST_AFTER - 1):
        assert p.check_minimap(low, 8.0) == []
        assert p.items["minimap"].value["found"]
        assert "minimap" in p.pending
    ch = p.check_minimap(low, 9.0)
    assert {"item:minimap", "reader:camera_truth", "reader:minimap_icons"} <= keys(ch)
    mm = p.items["minimap"]
    assert not mm.value["found"] and "совпадение упало" in mm.how
    assert "minimap" in p.pending


def test_minimap_good_check_resets_run_and_updates_score():
    p = player_with_api()
    p.set_minimap(BIG, 1.2, 0.6, 7.0)
    for _ in range(LOST_AFTER - 1):
        p.check_minimap(0.1, 8.0)
    assert p.check_minimap(0.7, 8.5) == []
    assert p.items["minimap"].value["score"] == 0.7 and "minimap" not in p.pending
    for _ in range(LOST_AFTER - 1):
        p.check_minimap(0.1, 9.0)
    assert p.items["minimap"].value["found"]


def test_minimap_missing_falls_back_to_layout():
    p = player_with_api()
    ch = p.minimap_missing(150.0, "не найдена за 300 кадров")
    mm = p.items["minimap"]
    assert mm.source == "layout" and not mm.value["found"] and "minimap" not in p.pending
    assert ch == []          # значение то же, camera_truth и так отключён
    assert not p.verdict("camera_truth").ok


def test_frame_size_change_drops_minimap_and_game_rect():
    p = player_with_api()
    p.set_minimap(STD_BOX, 1.0, 0.8, 7.0)
    assert p.set_frame_size((1920, 1080), 8.0) == []
    ch = p.set_frame_size((2560, 1440), 9.0)
    assert {"item:resolution", "item:game_rect", "item:minimap",
            "reader:camera_truth"} <= keys(ch)
    assert p.items["game_rect"].value == (0, 0, 2560, 1440)
    assert not p.items["minimap"].value["found"]
    assert {"minimap", "game_rect"} <= p.pending
    p.set_game_rect((0, 0, 2560, 1440), 9.5, how="клиентская область окна")
    assert "game_rect" not in p.pending


def test_size_unknown_at_start_then_known():
    src = {**SCREEN, "size": None}
    p = Passport.build(src, t=0.0, assets=ALL_ASSETS)
    p.set_view("spectator", 0.0, how="HUD")
    assert p.verdict("hud_roster").reason == "размер кадра неизвестен"
    ch = p.set_frame_size((1920, 1080), 1.0)
    assert {"item:resolution", "item:game_rect", "reader:hud_roster"} <= keys(ch)
    assert p.items["game_rect"].source == "source"


def test_scene_change_back_to_game_asks_to_recheck_minimap():
    p = player_with_api()
    p.set_minimap(BIG, 1.2, 0.6, 7.0)
    assert keys(p.set_scene("game", 7.0, game=True)) == {"item:scene"}
    assert "minimap" not in p.pending
    assert p.set_scene("game", 7.5, game=True) == []        # не на каждом кадре
    p.set_scene("loading", 8.0, game=False)
    assert "minimap" not in p.pending                       # на загрузке не проверять
    p.set_scene("game", 9.0, game=True)
    assert "minimap" in p.pending


def test_roster_source_validated():
    p = spectator_stream()
    ch = p.set_roster(None, 30.0, how="новый матч")
    assert {"reader:names", "reader:minimap_icons"} <= keys(ch)
    with pytest.raises(ValueError):
        p.set_roster("ocr", 31.0)


# ---------------------------------------------------------------- выход


def test_game_rect_honesty_by_source():
    assert Passport.build(SCREEN).items["game_rect"].source == "source"
    yt = Passport.build(YOUTUBE).items["game_rect"]
    assert yt.source == "assumed" and "не разобрана" in yt.how
    own = Passport.build(YOUTUBE, game_rect=(10, 0, 1270, 710)).items["game_rect"]
    assert own.value == (10, 0, 1270, 710) and own.source == "session"


def test_to_dict_is_json_and_lists_disabled_readers():
    p = player_with_api()
    p.set_minimap(BIG, 1.2, 0.61, 7.0)
    d = json.loads(json.dumps(p.to_dict()))
    assert set(d["items"]) == {"source", "resolution", "game_rect", "minimap", "view",
                               "api", "scene", "roster", "assets"}
    for it in d["items"].values():
        assert set(it) == {"value", "source", "how", "t"}
    assert d["items"]["view"] == {"value": "player", "source": "hud",
                                  "how": "HUD: таблица не совпала", "t": 5.0}
    assert d["items"]["minimap"]["value"] == {"found": True, "box": list(BIG),
                                              "scale": 1.2, "score": 0.61}
    assert d["items"]["resolution"]["value"] == [1920, 1080]
    assert {x["reader"] for x in d["disabled"]} == {"names", "hud_roster"}
    assert all(x["reason"] for x in d["disabled"])
    assert set(d["enabled"]) | {x["reader"] for x in d["disabled"]} == set(READERS)


def test_change_to_dict_and_text():
    p = spectator_stream()
    ch = p.set_api("in_game", True, 40.0)
    d = [c.to_dict() for c in ch]
    json.dumps(d)
    api = next(c for c in ch if c.key == "item:api")
    assert api.text == "item:api: не подключён → in_game (API появился)"
    rd = next(c for c in ch if c.key == "reader:api_facts")
    assert rd.new == "on" and rd.why == "требования выполнены"
