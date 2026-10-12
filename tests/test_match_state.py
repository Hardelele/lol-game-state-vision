from dataclasses import replace

from lolvision.match_state import FOUNTAIN, MatchState
from lolvision.records import DeathEvent, MinimapChamp, UnitRec

NAMES = ["Irelia", "Evelynn", "TwistedFate", "Jhin", "Zyra",
         "Volibear", "Hecarim", "Locke", "Jinx", "Lulu"]


def champs(**over):
    """Чемпионы миникарты кадра: все живы и не видны, `over` — правки по
    имени: {"Jinx": dict(visible=True, u=.5, v=.5, alive=False, timer=10)}."""
    out = []
    for s, n in enumerate(NAMES):
        d = {"visible": False, "u": None, "v": None, "alive": True, "timer": None,
             **over.get(n, {})}
        out.append(MinimapChamp(n, "ORDER" if s < 5 else "CHAOS", f"{'LR'[s // 5]}{s % 5 + 1}",
                                **d))
    return out


def view(m, name):
    return next(r for r in m.frame_view() if r.name == name)


def test_memory_after_leaving_minimap():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)))
    m.update(5, "game", champs())
    r = view(m, "Jinx")
    assert r.state == "remembered" and (r.u, r.v, r.age, r.src) == (.6, .7, 5.0, "minimap")
    assert view(m, "Lulu").state == "unknown"


def test_non_game_frames_keep_memory():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)))
    m.update(10, "tab")
    assert view(m, "Jinx").state == "remembered" and view(m, "Jinx").age == 10.0


def test_death_and_inferred_respawn_at_fountain():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)))
    ev = m.update(1, "game", champs(Jinx=dict(alive=False, timer=10)))
    assert ev == [DeathEvent(1, "Jinx", "CHAOS", .6, .7, 10)]
    assert ev[0].to_dict() == {"t": 1, "type": "death", "name": "Jinx", "team": "CHAOS",
                               "u": .6, "v": .7, "respawn_in": 10}
    assert view(m, "Jinx").state == "dead" and view(m, "Jinx").respawn_in == 10
    m.update(5, "tab")                       # панели не видно
    ev = m.update(11, "tab")
    assert [e.type for e in ev] == ["respawn"] and ev[0].src == "inferred"
    r = view(m, "Jinx")
    assert r.state == "remembered" and (r.u, r.v) == FOUNTAIN["CHAOS"] and r.src == "inferred"


def test_hud_flicker_alive_during_countdown_is_ignored():
    m = MatchState()
    m.update(0, "game", champs(Zyra=dict(alive=False, timer=15)))
    for t in range(1, 9):                              # панель под баннером
        assert m.update(t, "game", champs()) == []
    ev = m.update(9, "game", champs(Zyra=dict(alive=False, timer=6)))
    assert ev == [] and view(m, "Zyra").state == "dead"   # та же смерть
    ev = m.update(15, "game", champs())
    assert [(e.type, e.src) for e in ev] == [("respawn", "hud")]


def test_death_without_timer_needs_two_frames():
    m = MatchState()
    m.update(0, "game", champs())
    assert m.update(1, "game", champs(Irelia=dict(alive=False))) == []
    assert m.update(2, "game", champs()) == []
    m.update(3, "game", champs(Irelia=dict(alive=False)))
    ev = m.update(4, "game", champs(Irelia=dict(alive=False)))
    assert [e.type for e in ev] == ["death"]


def test_screen_unit_with_name_is_seen_with_hp():
    m = MatchState(view=(0.2, 0.1))
    m.update(0, "game", champs(), [UnitRec("champion", "CHAOS", .75, .5, .4, .9, "Jinx"),
                                   UnitRec("minion", "ORDER", .1, .1, 1.0, .8)], (.5, .5))
    r = view(m, "Jinx")
    assert r.state == "visible" and r.src == "screen" and r.hp == .4
    assert abs(r.u - .55) < 1e-6 and abs(r.v - .5) < 1e-6


def test_roster_switch_needs_several_frames():
    m = MatchState()
    m.update(0, "game", champs())
    other = champs()
    other[0] = replace(other[0], name="Garen")
    assert m.update(1, "game", other) == []
    m.update(2, "game", other)
    ev = m.update(3, "game", other)
    assert ev[0].type == "roster" and m.order[0] == "Garen"


def test_shop_frame_still_reads_minimap():
    # Магазин — игра под оверлеем (#786): чемпион на миникарте виден.
    m = MatchState()
    m.update(0, "game", champs())
    m.update(5, "shop", champs(Jinx=dict(visible=True, u=.3, v=.4)))
    r = view(m, "Jinx")
    assert r.state == "visible" and (r.u, r.v) == (.3, .4)
