from lolvision.match_state import FOUNTAIN, MatchState

NAMES = ["Irelia", "Evelynn", "TwistedFate", "Jhin", "Zyra",
         "Volibear", "Hecarim", "Locke", "Jinx", "Lulu"]


def champs(**over):
    """Строка row["champs"]: все живы и не видны, `over` — правки по имени:
    {"Jinx": dict(vis=True, u=.5, v=.5, alive=False, timer=10)}."""
    out = []
    for s, n in enumerate(NAMES):
        d = {"vis": False, "u": None, "v": None, "alive": True, "timer": None, **over.get(n, {})}
        out.append([n, "ORDER" if s < 5 else "CHAOS", f"{'LR'[s // 5]}{s % 5 + 1}",
                    d["vis"], d["u"], d["v"], d["alive"], d["timer"]])
    return out


def view(m, name):
    return next(r for r in m.frame_view() if r[0] == name)


def test_memory_after_leaving_minimap():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs(Jinx=dict(vis=True, u=.6, v=.7))})
    m.update(5, {"kind": "game", "champs": champs()})
    r = view(m, "Jinx")
    assert r[2] == "remembered" and r[3:7] == [.6, .7, 5.0, "minimap"]
    assert view(m, "Lulu")[2] == "unknown"


def test_non_game_frames_keep_memory():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs(Jinx=dict(vis=True, u=.6, v=.7))})
    m.update(10, {"kind": "tab"})
    assert view(m, "Jinx")[2] == "remembered" and view(m, "Jinx")[5] == 10.0


def test_death_and_inferred_respawn_at_fountain():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs(Jinx=dict(vis=True, u=.6, v=.7))})
    ev = m.update(1, {"kind": "game", "champs": champs(Jinx=dict(alive=False, timer=10))})
    assert ev == [{"t": 1, "type": "death", "name": "Jinx", "team": "CHAOS",
                   "at": [.6, .7], "respawn_in": 10}]
    assert view(m, "Jinx")[2] == "dead" and view(m, "Jinx")[8] == 10
    m.update(5, {"kind": "tab"})                       # панели не видно
    ev = m.update(11, {"kind": "tab"})
    assert [e["type"] for e in ev] == ["respawn"] and ev[0]["src"] == "inferred"
    r = view(m, "Jinx")
    assert r[2] == "remembered" and tuple(r[3:5]) == FOUNTAIN["CHAOS"] and r[6] == "inferred"


def test_hud_flicker_alive_during_countdown_is_ignored():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs(Zyra=dict(alive=False, timer=15))})
    for t in range(1, 9):                              # панель под баннером
        assert m.update(t, {"kind": "game", "champs": champs()}) == []
    ev = m.update(9, {"kind": "game", "champs": champs(Zyra=dict(alive=False, timer=6))})
    assert ev == [] and view(m, "Zyra")[2] == "dead"   # та же смерть
    ev = m.update(15, {"kind": "game", "champs": champs()})
    assert [(e["type"], e["src"]) for e in ev] == [("respawn", "hud")]


def test_death_without_timer_needs_two_frames():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs()})
    assert m.update(1, {"kind": "game", "champs": champs(Irelia=dict(alive=False))}) == []
    assert m.update(2, {"kind": "game", "champs": champs()}) == []
    m.update(3, {"kind": "game", "champs": champs(Irelia=dict(alive=False))})
    ev = m.update(4, {"kind": "game", "champs": champs(Irelia=dict(alive=False))})
    assert [e["type"] for e in ev] == ["death"]


def test_screen_unit_with_name_is_seen_with_hp():
    m = MatchState(view=(0.2, 0.1))
    m.update(0, {"kind": "game", "champs": champs(), "px": .5, "py": .5,
                 "units": [["champion", "CHAOS", .75, .5, .4, .9, "Jinx"],
                           ["minion", "ORDER", .1, .1, 1.0, .8]]})
    r = view(m, "Jinx")
    assert r[2] == "visible" and r[6] == "screen" and r[7] == .4
    assert abs(r[3] - .55) < 1e-6 and abs(r[4] - .5) < 1e-6


def test_roster_switch_needs_several_frames():
    m = MatchState()
    m.update(0, {"kind": "game", "champs": champs()})
    other = champs()
    other[0][0] = "Garen"
    assert m.update(1, {"kind": "game", "champs": other}) == []
    m.update(2, {"kind": "game", "champs": other})
    ev = m.update(3, {"kind": "game", "champs": other})
    assert ev[0]["type"] == "roster" and m.order[0] == "Garen"
