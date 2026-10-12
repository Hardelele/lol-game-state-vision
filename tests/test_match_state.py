from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lolstream.live_client import read_journal, replay, status_observation
from lolvision.geometry import Projection
from lolvision.match_state import (API_STALE, FOUNTAIN, ApiObs, ApiPlayer, MatchState,
                                   SourcedDeath)
from lolvision.records import DeathEvent, MinimapChamp, RosterEvent, UnitRec

OWN = Path(__file__).parent / "fixtures" / "live_client" / "own_game.jsonl"
# Id Data Dragon чемпионов журнала своей игры (в тестах кэша Data Dragon нет).
IDS = ["Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux", "Twitch",
       "Yuumi"]

NAMES = ["Irelia", "Evelynn", "TwistedFate", "Jhin", "Zyra",
         "Volibear", "Hecarim", "Locke", "Jinx", "Lulu"]
# Кадр 1000×1000 целиком — сцена 100×100; пиксель сцены — 0.001 доли карты,
# центр кадра — центр камеры (как в test_unit_tracker).
PROJ = Projection(np.array([[0.001, 0, -0.05], [0, 0.001, -0.05], [0, 0, 1.0]]),
                  (100, 100), (0, 0, 1000, 1000), (1000, 1000))
JINX = UnitRec("champion", "CHAOS", .75, .5, .4, .9, "Jinx")


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


def test_screen_unit_with_name_is_placed_through_projection():
    m = MatchState(proj=PROJ)
    m.update(0, "game", champs(), [JINX, UnitRec("minion", "ORDER", .1, .1, 1.0, .8)], (.5, .5))
    r = view(m, "Jinx")
    assert r.state == "visible" and r.src == "screen" and r.hp == .4
    # (750, 500) px → (75, 50) px сцены → смещение (0.025, 0) от камеры.
    assert abs(r.u - .525) < 1e-6 and abs(r.v - .5) < 1e-6


def test_screen_place_comes_from_the_projection_not_view_share():
    # Наклонённая камера: внизу кадра пиксель стоит меньше карты, чем вверху.
    tilt = Projection(np.array([[0.001, 0, -0.05], [0, 0.001, -0.05], [0, 0.004, 1.0]]),
                      (100, 100), (0, 0, 1000, 1000), (1000, 1000))
    m = MatchState(proj=tilt)
    m.update(0, "game", champs(), [replace(JINX, y=.9)], (.5, .5))
    want = tilt.frac_to_map([[.75, .9]], (.5, .5))[0]
    r = view(m, "Jinx")
    assert (r.u, r.v) == (float(want[0]), float(want[1]))
    # (75, 90) px сцены: w = 1.36 → смещение (0.0184, 0.0294), а не (0.025, 0.04).
    assert abs(r.u - .5184) < 1e-3 and abs(r.v - .5294) < 1e-3


def test_no_projection_or_camera_gives_no_screen_place():
    for proj, cam in ((None, (.5, .5)), (PROJ, None), (PROJ, (None, None))):
        m = MatchState(proj=proj)
        m.update(0, "game", champs(), [JINX], cam)
        r = view(m, "Jinx")
        # Виден и ХП известно, а места нет: выдумывать его не из чего.
        assert r.state == "visible" and r.hp == .4
        assert r.u is None and r.src is None


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


# ---------- Live Client API как источник (#798, 2B) ----------

def api(game_time=60.0, dead=None, names=NAMES, me_team="ORDER"):
    """Снимок API: все живы, `dead` — {имя: таймер}."""
    dead = dead or {}
    return ApiObs(game_time, tuple(ApiPlayer(n, "ORDER" if s < 5 else "CHAOS", n in dead,
                                             float(dead.get(n, 0.0)))
                                   for s, n in enumerate(names)), me_team)


def test_api_gives_roster_deaths_and_clock_without_frames():
    m = MatchState()
    assert m.update(0, "loading", api=api(0.07)) == []      # состав уже на загрузке
    assert m.order == NAMES and m.api_live()
    ev = m.update(10, "game", api=api(70.0, {"Jinx": 12.0}))
    assert ev == [SourcedDeath(10, "Jinx", "CHAOS", None, None, 12.0, "api")]
    assert ev[0].to_dict()["src"] == "api"
    r = view(m, "Jinx")
    assert r.state == "dead" and r.respawn_in == 12.0
    assert m.game_clock().value == 70.0 and m.game_clock().src == "api"
    m.update(11.5, "game")                                   # кадр без нового снимка
    assert m.game_clock().value == 71.5 and view(m, "Jinx").respawn_in == 10.5
    ev = m.update(22, "game", api=api(82.0))
    assert [(e.type, e.src) for e in ev] == [("respawn", "api")]
    assert (view(m, "Jinx").u, view(m, "Jinx").v) == FOUNTAIN["CHAOS"]
    assert not m.conflicts


def test_api_beats_hud_and_disagreement_is_counted():
    m = MatchState()
    m.update(0, "game", champs(), api=api())
    # HUD читает Jinx мёртвой — API: жива. Значение из API, расхождение — в счётчик.
    ev = m.update(2, "game", champs(Jinx=dict(alive=False, timer=10)))
    ev += m.update(3, "game", champs(Jinx=dict(alive=False, timer=9)))
    assert ev == [] and view(m, "Jinx").state != "dead"
    assert m.conflicts == {"alive": 2}
    # API: Lulu мертва 20 с, панель — 9 с: таймер расходится.
    m.update(7, "game", api=api(67.0, {"Lulu": 20.0}))
    m.update(10, "game", champs(Lulu=dict(alive=False, timer=9)))
    assert m.conflicts["timer"] == 1 and view(m, "Lulu").respawn_in == 17.0
    # Иконка мёртвой Lulu на миникарте — назначение не тому, место не берётся.
    m.update(11, "game", champs(Lulu=dict(visible=True, u=.5, v=.5, alive=None)))
    assert m.conflicts["minimap_dead"] == 1 and view(m, "Lulu").u is None
    assert view(m, "Lulu").state == "dead"


def test_hud_lag_right_after_api_change_is_not_a_conflict():
    m = MatchState()
    m.update(0, "game", champs(), api=api())
    m.update(1, "game", champs(), api=api(61.0, {"Jinx": 10.0}))   # панель ещё «жива»
    m.update(1.5, "game", champs(Jinx=dict(alive=False, timer=9)))
    assert not m.conflicts


def test_minimap_place_is_still_taken_with_api():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)), api=api())
    r = view(m, "Jinx")
    assert r.state == "visible" and (r.u, r.v, r.src) == (.6, .7, "minimap")


def test_hud_roster_window_does_not_reset_when_api_is_there():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)), api=api())
    other = champs()
    other[0] = replace(other[0], name="Garen")
    for t in range(1, 6):                           # больше ROSTER_SWITCH кадров
        assert m.update(t, "game", other) == []
    assert m.order == NAMES and m.conflicts == {"roster": 5}
    assert view(m, "Jinx").u == .6                  # память на месте


def test_hud_roster_before_api_same_set_keeps_memory():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)))
    assert m.update(1, "game", api=api()) == []
    assert view(m, "Jinx").u == .6 and not m.conflicts


def test_hud_roster_before_api_wrong_is_replaced():
    m = MatchState()
    wrong = champs()
    wrong[0] = replace(wrong[0], name="Garen")
    m.update(0, "game", wrong)
    ev = m.update(1, "game", api=api())
    assert [e.type for e in ev] == ["roster"] and m.order == NAMES
    assert m.conflicts == {"roster": 1}


def test_new_game_in_api_resets_memory():
    m = MatchState()
    m.update(0, "game", champs(Jinx=dict(visible=True, u=.6, v=.7)), api=api(600.0))
    # Тот же состав (ремейк), но время матча пошло с нуля — новая игра.
    ev = m.update(10, "game", api=api(1.0))
    assert [e.type for e in ev] == ["roster"] and view(m, "Jinx").u is None
    # Другой состав — тоже новая игра, сразу, без окна кадров.
    names = ["Garen"] + NAMES[1:]
    ev = m.update(20, "game", api=api(2.0, names=names))
    assert [e.type for e in ev] == ["roster"] and m.order == names
    # Обычный ход времени и повтор снимка — не новая игра.
    assert m.update(21, "game", api=api(3.0, names=names)) == []
    assert m.update(22, "game", api=api(3.0, names=names)) == []


def test_refused_api_snapshot_changes_nothing():
    m = MatchState()
    half = ApiObs(60.0, api().players[:9], "ORDER")
    assert m.update(0, "game", api=half) == []
    assert m.order == [] and m.api is None and "9" in m.api_refused
    dup = ApiObs(60.0, api().players[:9] + (ApiPlayer("Jinx", "CHAOS", False, 0.0),))
    m.update(1, "game", api=dup)
    assert m.api is None and m.api_refused == "чемпионы повторяются"


def test_api_gone_hud_takes_over_but_roster_stays():
    m = MatchState()
    m.update(0, "game", champs(), api=api())
    t = API_STALE + 1                               # API молчит
    assert not m.api_live(t)
    ev = m.update(t, "game", champs(Jinx=dict(alive=False, timer=10)))
    assert ev == [DeathEvent(t, "Jinx", "CHAOS", None, None, 10)]  # снова по панели
    assert m.game_clock() is None and m.order == NAMES


def test_without_api_behaviour_is_unchanged():
    m = MatchState()
    m.update(0, "game", champs(Zyra=dict(alive=False, timer=15)))
    m.update(1, "game", champs())                   # панель под баннером
    assert m.conflicts == {"hud_flicker": 1}
    assert m.api is None and m.game_clock() is None and not m.api_live()


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_own_game_journal_through_match_state():
    """Журнал своей игры (только API, без кадров): состав верен на каждом
    снимке, смерти и возрождения — ровно переходы isDead, ложных сбросов нет."""
    m = MatchState()
    want_ev, got_ev, kills, n = [], [], [], 0
    prev: dict[str, bool] = {}
    for st, new in replay(read_journal(OWN), IDS):
        obs, why = status_observation(st, st.t)
        assert (obs is None) == (st.game is None), why
        if obs is None:
            continue
        n += 1
        got_ev += [(e.type, e.name, e.t) for e in m.update(st.t, "game", api=obs)]
        g = st.game
        assert [(r.name, r.team) for r in m.frame_view()] == \
            [(p.champion, p.team) for p in g.players]           # ORDER, затем CHAOS
        nick = {p.riot_id.split("#")[0]: p.champion for p in g.players}
        kills += [nick[e.data["VictimName"]] for e in new if e.name == "ChampionKill"]
        for p in g.players:
            r = view(m, p.champion)
            assert (r.state == "dead") == p.is_dead
            if p.is_dead:
                assert abs(r.respawn_in - p.respawn_timer) < 1e-6
            if p.champion in prev and prev[p.champion] != p.is_dead:
                want_ev.append(("death" if p.is_dead else "respawn", p.champion, st.t))
            prev[p.champion] = p.is_dead
        assert m.game_clock().value == g.game_time
    assert n == 36
    assert got_ev == want_ev
    deaths = [name for typ, name, _ in got_ev if typ == "death"]
    assert deaths == ["Yone", "Lux", "Yone", "Zilean"] == kills    # каждое убийство API
    assert not any(isinstance(e, RosterEvent) for e in m.events)    # ложных сбросов нет
    assert not m.conflicts and m.api.me_team == "ORDER"
