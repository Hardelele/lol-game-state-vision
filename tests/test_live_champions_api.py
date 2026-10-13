"""Состав из Live Client API в живом потоке (#798, 2B): портреты HUD не
читаются, иконки миникарты — только среди десяти, у вида игрока синие —
свои, имена над плашками не выдаются."""

from types import SimpleNamespace

import numpy as np
import pytest

from lolstream import live_champions as lc
from lolvision.match_state import ApiObs, ApiPlayer

ORDER = ["Zilean", "Amumu", "Yone", "Caitlyn", "Nami"]
CHAOS = ["Darius", "Diana", "Lux", "Twitch", "Yuumi"]
KNOWN = ORDER + CHAOS + ["Garen"]                    # Garen — не из этого матча


def obs(me_team="CHAOS", names=ORDER + CHAOS) -> ApiObs:
    return ApiObs(100.0, tuple(ApiPlayer(n, "ORDER" if k < 5 else "CHAOS", False, 0.0)
                               for k, n in enumerate(names)), me_team)


@pytest.fixture
def champs(monkeypatch, tmp_path):
    """LiveChampions на CPU без моделей; панели HUD при составе из API не
    читаются — их чтение роняет тест."""
    icons = {c: np.zeros((8, 8, 3), np.uint8) for c in KNOWN}
    model = SimpleNamespace(champions=KNOWN, device="cpu")
    monkeypatch.setattr(lc, "IconModel", lambda run, device: model)
    monkeypatch.setattr(lc.ddragon, "load_icons", lambda: icons)
    monkeypatch.setattr(lc, "IconBank", lambda icons: SimpleNamespace(ids=sorted(icons)))
    monkeypatch.setattr(lc.hud_panels, "DigitReader", lambda: None)

    def no_hud(*a, **k):
        raise AssertionError("HUD читается при составе из API")
    monkeypatch.setattr(lc.hud_panels, "read_panels", no_hud)
    monkeypatch.setattr(lc.hud_panels, "portrait_templates", no_hud)
    monkeypatch.setattr(lc, "read_frame", no_hud)
    obj = lc.LiveChampions("cpu", names=tmp_path / "names.npz")
    yield obj
    obj.close()


def det(name: str, color: int, x=128.0, y=64.0, p=0.99) -> dict:
    """Детекция миникарты: цвет рамки 0 — синий, 1 — красный."""
    logp = np.full(len(KNOWN), np.log((1 - p) / (len(KNOWN) - 1)))
    logp[KNOWN.index(name)] = np.log(p)
    tp = [0.95, 0.05] if color == 0 else [0.05, 0.95]
    return {"x": x, "y": y, "team": color, "score": 0.9, "team_p": tp, "logp": logp}


def test_api_roster_replaces_hud_and_player_view_colors_are_relative(champs):
    assert champs.set_api(obs("CHAOS")) is None
    m = champs.match
    assert m.api and champs.roster == CHAOS + ORDER       # синие на миникарте — свои
    assert champs.view == "player"
    champs.offer(np.zeros((1080, 1920, 3), np.uint8), 0.0)   # портреты не читаются
    assert not champs._busy and champs.read_roster(np.zeros((1080, 1920, 3), np.uint8)) is False
    # Синяя иконка Lux (CHAOS — свои) и красная Zilean (ORDER — враги);
    # Garen уверенно, но не из этого матча — без пары.
    champs.icon_net = lambda mini: [det("Lux", 0, 10, 20), det("Zilean", 1, 200, 30),
                                    det("Garen", 0, 50, 50)]
    out = champs.minimap(np.zeros((1080, 1920, 3), np.uint8), np.zeros((256, 256, 3)))
    by = {r.name: r for r in out}
    assert [r.name for r in out] == CHAOS + ORDER and "Garen" not in by
    assert (by["Lux"].team, by["Lux"].slot, by["Lux"].visible) == ("CHAOS", "R3", True)
    assert (by["Zilean"].team, by["Zilean"].slot) == ("ORDER", "L1")
    assert by["Zilean"].u == 200 / lc.MM and sum(r.visible for r in out) == 2
    # Жив / мёртв по панелям не выдумывается: его даёт API.
    assert all(r.alive is None and r.timer is None for r in out)


def test_names_over_plates_are_not_used_with_api_player_view(champs):
    champs.set_api(obs("ORDER"))
    champs.names = object()                    # банк имён есть — всё равно молчим
    units = [SimpleNamespace(kind="champion", team="ORDER", x=900, y=500)]
    assert champs.view == "player"
    assert champs.who(np.zeros((1080, 1920, 3), np.uint8), units) == [None]


def test_spectator_in_client_keeps_absolute_colors(champs):
    champs.set_api(obs(None))
    assert champs.roster == ORDER + CHAOS and champs.view == "spectator"
    assert champs.match.slot(5) == ("CHAOS", "R1")


def test_bad_api_roster_is_refused_not_half_used(champs):
    why = champs.set_api(obs(names=ORDER + CHAOS[:4] + ["Ahri"]))
    assert champs.match is None and "Ahri" in why and champs.api_error == why
    why = champs.set_api(ApiObs(1.0, obs().players[:9]))
    assert champs.match is None and "9" in why


def test_same_api_roster_keeps_match_and_none_drops_it(champs):
    champs.set_api(obs())
    m = champs.match
    champs.set_api(obs())
    assert champs.match is m                    # повтор снимка — не новый состав
    champs.set_api(None)
    assert champs.match is None and champs.view is None


# ---------- тренировка: неполный состав (#803) ----------

def practice(players, me="Zilean", me_team="ORDER", mode="PRACTICETOOL") -> ApiObs:
    return ApiObs(100.0, tuple(ApiPlayer(n, t, False, 0.0) for n, t in players), me_team, me,
                  11, mode)


def test_practice_tool_alone_assigns_only_my_icon(champs):
    assert champs.set_api(practice([("Zilean", "ORDER")])) is None
    assert champs.roster == ["Zilean"] and champs.view == "player"
    # Своя синяя иконка — своя; боты на миникарте (их API не знает) — лишние.
    champs.icon_net = lambda mini: [det("Darius", 1, 200, 30), det("Zilean", 0, 10, 20),
                                    det("Lux", 1, 90, 90)]
    out = champs.minimap(np.zeros((1080, 1920, 3), np.uint8), np.zeros((256, 256, 3)))
    assert [(r.name, r.team, r.slot, r.visible) for r in out] == [("Zilean", "ORDER", "L1",
                                                                   True)]
    assert (out[0].u, out[0].v) == (10 / lc.MM, 20 / lc.MM)
    # В обычной игре один игрок — отказ, как раньше.
    why = champs.set_api(practice([("Zilean", "ORDER")], mode="CLASSIC"))
    assert why == "игроков 1, а не 10" and champs.api_error == why


def test_practice_tool_bot_twin_is_told_apart_by_icon_color(champs):
    roster = [(n, "ORDER") for n in ORDER] + [("Zilean", "CHAOS")] + \
        [(n, "CHAOS") for n in CHAOS[1:]]
    assert champs.set_api(practice(roster)) is None
    assert champs.roster.count("Zilean") == 2
    champs.icon_net = lambda mini: [det("Zilean", 1, 200, 30), det("Zilean", 0, 10, 20)]
    out = champs.minimap(np.zeros((1080, 1920, 3), np.uint8), np.zeros((256, 256, 3)))
    at = {(r.team, r.slot): (r.u, r.v) for r in out if r.name == "Zilean"}
    assert at == {("ORDER", "L1"): (10 / lc.MM, 20 / lc.MM),
                  ("CHAOS", "R1"): (200 / lc.MM, 30 / lc.MM)}
