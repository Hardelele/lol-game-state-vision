"""Сеанс live на подставном источнике кадров и подставном Live Client (#794, #795).

Без видеокарты и без захвата: источник отдаёт заранее заданные ответы
(кадр, пауза, окна нет, конец), путь кадра собирается настоящим
`build_pipeline` на процессоре, но вместо модели камеры — заглушка, а
модели юнитов, чемпионов и типа сцены «не найдены на диске». Live Client
читает журнал своей игры через подставной транспорт — порт 2999 не трогается.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lolstream import live
from lolstream import live_champions as lc
from lolstream.frame_source import Frame, Status, gap
from lolstream.live_client import LiveClient, Response, journal_transport, read_journal
from lolstream.state_service import check_snapshot
from lolvision.minimap_locate import STD_BOX, Fit, load_fit, save_fit
from lolvision.records import MinimapChamp
from lolvision.unit_net import Unit

OWN = Path(__file__).parent / "fixtures" / "live_client" / "own_game.jsonl"
NOWHERE = Path(__file__).parent / "fixtures" / "нет-такого-файла"


class _Runner:
    """Модель камеры-заглушка: всегда одна точка, сеть не нужна."""

    kind, label, trained_on = "stub", "заглушка", []

    def describe(self, scene_size) -> str:
        return f"заглушка {scene_size[0]}x{scene_size[1]}"

    def prepare(self, scene):
        return None

    def frame_prep(self, frame_size, crop, boxes, store_size):
        return None

    def predict(self, prep):
        return ({"px": 0.4, "py": 0.6, "qx": 0.4, "qy": 0.6, "spread": 0.01},
                np.zeros((4, 4), np.float32), {"net": 0.0, "agg": 0.0})


@pytest.fixture
def cpu_models(monkeypatch, tmp_path):
    """build_pipeline без тяжёлых моделей: чекпоинт есть (пустой файл),
    грузится заглушка; детектора юнитов, чемпионов и типа сцены «нет»."""
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"")
    monkeypatch.setattr(live, "run_checkpoint", lambda run: ck)
    monkeypatch.setattr(live, "load_runner", lambda path, device: _Runner())
    monkeypatch.setattr(live, "MINI_RUN", NOWHERE)
    monkeypatch.setattr(live, "KIND_NET", NOWHERE / "net.pt")
    monkeypatch.setattr(live, "KIND_HEAD", NOWHERE / "head.pt")
    monkeypatch.setattr(live, "live_net", lambda: (None, None))
    built: list[tuple[int, int]] = []
    real = live.build_pipeline

    def build(size, run, device, fps, **kw):
        built.append(size)
        return real(size, run, device, fps, **kw)
    return build, built


def _img(w: int = 640, h: int = 360) -> np.ndarray:
    return np.full((h, w, 3), 40, np.uint8)


class _Source:
    """Источник по сценарию: список ответов, затем ENDED (или пауза без
    конца при `forever`). Запоминает потоки, в которых его открыли и читали."""

    def __init__(self, script: list[Frame], kind: str = "youtube", size=(640, 360),
                 forever: Frame | None = None, fps: float = 2.0, on_read=None):
        self.script, self.kind, self._size, self.forever = list(script), kind, size, forever
        self.on_read = on_read                 # перед каждым кадром сценария (опрос API)
        self.fps, self.backend, self.backend_note = fps, "подставной", "задан тестом"
        self.start = 0.0
        self.threads: dict[str, set[int]] = {"open": set(), "read": set()}
        self.closed = False

    @property
    def size(self):
        return self._size

    def open(self) -> None:
        self.threads["open"].add(threading.get_ident())

    def read(self) -> Frame:
        self.threads["read"].add(threading.get_ident())
        if self.script:
            if self.on_read is not None:
                self.on_read()
            return self.script.pop(0)
        if self.forever is not None:
            time.sleep(0.005)
            return self.forever
        return gap(Status.ENDED, 99.0, "ролик кончился")

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "fps": self.fps,
                "size": self._size, "title": "сценарий", "duration": 0.0}


def _ok(t: float, w: int = 640, h: int = 360) -> Frame:
    return Frame(Status.OK, t, _img(w, h))


def _wait(s: live.Session, cond, timeout: float = 20.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        st = s.status()
        if cond(st):
            return st
        time.sleep(0.01)
    raise AssertionError(f"не дождались: {s.status()['state']} · {s.status()['note']}")


def _finish(s: live.Session) -> dict:
    s.join(20.0)
    assert not s._th.is_alive()
    return s.status()


# ---------- путь кадра и состояния источника ----------

def test_rows_take_frame_time_and_gaps_write_no_rows(cpu_models):
    build, built = cpu_models
    src = _Source([_ok(0.0), gap(Status.PAUSED, 0.5), _ok(1.0),
                   gap(Status.NO_WINDOW, 1.5, "окна нет"), _ok(2.25)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert st["state"] == live.State.ENDED.label and st["active"] is False
    # Строки — только на кадры, время — Frame.t, а не номер / fps.
    assert [r["t"] for r in st["frames"]] == [0.0, 1.0, 2.25]
    assert [r["i"] for r in st["frames"]] == [0, 1, 2]
    assert st["source_gaps"] == {"paused": 1, "no_window": 1}
    assert st["source_state"]["state"] == "ended"
    assert st["source_state"]["reason"] == "ролик кончился"
    assert built == [(640, 360)]               # размер известен заранее — один путь
    assert st["kind"] == "stub" and "без юнитов" in st["note"]
    # У ролика API нет — явно, с причиной.
    assert st["api"] is None and st["api_reason"] == "нет — источник YouTube"
    assert src.closed
    json.dumps(st)


def test_open_and_read_happen_in_session_thread(cpu_models):
    build, _ = cpu_models
    src = _Source([_ok(0.0)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    _finish(s)
    assert src.threads["open"] == src.threads["read"] == {s._th.ident}
    assert threading.get_ident() not in src.threads["read"]


def test_pause_keeps_session_running_with_reason(cpu_models):
    build, _ = cpu_models
    paused = gap(Status.PAUSED, 3.0)
    src = _Source([_ok(0.0)], kind="screen", forever=paused)
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _wait(s, lambda st: st["source_state"]["state"] == "paused")
    assert st["state"] == live.State.RUNNING.label and st["active"] is True
    assert st["source_state"]["reason"] == Status.PAUSED.label
    assert st["speed"] is None                 # экран — в реальном времени
    s.stop()
    st = _finish(s)
    assert st["state"] == live.State.STOPPED.label
    assert st["source_gaps"]["paused"] >= 1
    # Остановленный сеанс прогрев не досчитывает: кадр в строки не попал.
    assert st["count"] == 0


def test_unknown_size_builds_pipeline_on_first_frame(cpu_models):
    build, built = cpu_models
    src = _Source([gap(Status.NO_WINDOW, 0.0, "окна нет"), _ok(0.5, 800, 450)],
                  kind="screen", size=None)
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert built == [(800, 450)]
    assert st["count"] == 1 and st["note"].startswith("свой экран 800x450")


def test_resize_rebuilds_pipeline_and_says_so(cpu_models):
    build, built = cpu_models
    src = _Source([_ok(0.0), _ok(0.5), _ok(1.0, 800, 450), _ok(1.5, 800, 450)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert built == [(640, 360), (800, 450)]
    assert "640x360 → 800x450: путь кадра пересобран" in st["note"]
    # Кадры прогрева старого размера отданы до пересборки, ничего не потеряно.
    assert [r["t"] for r in st["frames"]] == [0.0, 0.5, 1.0, 1.5]
    assert st["source_state"]["state"] == "ended"


def test_missing_model_fails_with_reason(monkeypatch):
    monkeypatch.setattr(live, "run_checkpoint", lambda run: NOWHERE / "model.pt")
    src = _Source([_ok(0.0)])
    s = live.Session("vid", "нет/такого", src, device="cpu")
    st = _finish(s)
    assert st["state"] == live.State.FAILED.label
    assert st["note"] == "нет модели runs/нет/такого/model.pt"
    assert st["count"] == 0 and src.closed


def test_source_open_failure_fails_session(cpu_models):
    build, _ = cpu_models

    class _Broken(_Source):
        def open(self):
            raise RuntimeError("ни один бэкенд захвата не открылся")
    src = _Broken([], kind="screen")
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert st["state"] == live.State.FAILED.label
    assert "ни один бэкенд захвата не открылся" in st["note"]


# ---------- Live Client в сеансе ----------

def _client(transport) -> LiveClient:
    return LiveClient(transport=transport, period=0.01, champion_ids=None)


def test_live_client_runs_with_screen_session(cpu_models):
    build, _ = cpu_models
    api = _client(journal_transport(read_journal(OWN)))
    src = _Source([], kind="screen", forever=gap(Status.PAUSED, 0.0))
    s = live.Session("screen", "r", src, device="cpu", build=build, api=api)
    st = _wait(s, lambda st: st["api"] is not None and st["api"]["state"] == "in_game"
               and st["api"]["events"] > 0)
    a = st["api"]
    assert st["api_reason"] is None
    # В status — сводка без полного снимка игры.
    assert "game" not in a and a["source"] == "api"
    assert a["brief"]["players"] == 10 and a["brief"]["game_time"] > 0
    assert a["reason"] is None or isinstance(a["reason"], str)
    full = s.api_full()
    assert full["api"]["game"] is not None and len(full["api"]["game"]["players"]) == 10
    assert len(full["events"]) == len(s.api_events) >= a["events"]
    ids = [e["event_id"] for e in full["events"]]
    assert len(ids) == len(set(ids))           # события без повторов
    json.dumps(s.status())
    s.stop()
    _finish(s)
    assert api._thread is None                 # опрос остановлен вместе с сеансом


def test_no_api_does_not_break_session(cpu_models):
    build, _ = cpu_models
    api = _client(lambda endpoint: Response(None, None, "нет соединения: refused"))
    src = _Source([_ok(0.0), _ok(0.5)], kind="screen", forever=gap(Status.PAUSED, 1.0))
    s = live.Session("screen", "r", src, device="cpu", build=build, api=api)
    st = _wait(s, lambda st: st["api"]["state"] == "no_game"
               and st["api"]["reason"] != "ещё не опрошен")
    assert st["active"] is True and st["api"]["brief"] is None
    assert "нет соединения" in st["api"]["reason"]
    s.stop()
    st = _finish(s)
    assert st["state"] == live.State.STOPPED.label


def test_start_screen_opens_screen_session_with_api(monkeypatch, cpu_models):
    build, _ = cpu_models
    made = {}

    def screen(fps, backend="auto"):
        made["fps"], made["backend"] = fps, backend
        return _Source([], kind="screen", forever=gap(Status.NO_WINDOW, 0.0, "окна нет"))
    monkeypatch.setattr(live, "ScreenSource", screen)
    monkeypatch.setattr(live, "build_pipeline", build)
    monkeypatch.setattr(live, "LiveClient", lambda champion_ids=None: _client(
        lambda endpoint: Response(None, None, "нет соединения")))
    monkeypatch.setattr(live, "_sessions", {})
    monkeypatch.setattr(live, "STATE_PORT", 0)        # сервис состояния — на свободном порту
    with pytest.raises(ValueError):
        live.start_screen("r", 2.0, backend="gdi")
    st = live.start_screen("r", 3.0, backend="mss")
    try:
        assert st["id"] == live.SCREEN_ID == "screen"
        assert made == {"fps": 3.0, "backend": "mss"}
        s = live.get("screen")
        assert s.api is not None
        st = _wait(s, lambda st: st["source_state"]["state"] == "no_window")
        assert st["source_state"]["reason"] == "окна нет" and st["count"] == 0
        assert live.listing()[0]["source"] == "screen"
        # Сервис состояния поднят вместе с сеансом экрана, адрес — в status().
        assert st["state_url"].startswith("http://127.0.0.1:") and st["state_reason"] is None
    finally:
        live.stop("screen")
        st = _finish(live.get("screen"))
    assert st["state_reason"] == "сеанс закрыт"


def test_start_youtube_reads_video_source_without_api(monkeypatch, cpu_models):
    build, _ = cpu_models
    made = {}

    def video(src, fps, start=0.0):
        made.update(src=src, fps=fps, start=start)
        return _Source([_ok(start)])
    monkeypatch.setattr(live, "VideoSource", video)
    monkeypatch.setattr(live, "build_pipeline", build)
    monkeypatch.setattr(live, "_sessions", {})
    live.start("dQw4w9WgXcQ", "r", 1.0, 30.0)
    s = live.get("dQw4w9WgXcQ")
    st = _finish(s)
    assert made == {"src": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "fps": 1.0,
                    "start": 30.0}
    assert s.api is None and st["api_reason"] == "нет — источник YouTube"
    assert [r["t"] for r in st["frames"]] == [30.0]


def test_serve_start_route_picks_source(monkeypatch):
    from lolstream import serve_inspect
    calls = []
    monkeypatch.setattr(live, "start_screen", lambda *a: calls.append(("screen", a)) or {})
    monkeypatch.setattr(live, "start", lambda *a: calls.append(("youtube", a)) or {})
    out = []
    h = SimpleNamespace(server=SimpleNamespace(live_run="def"), _json=out.append)
    for body in ({"source": "screen", "fps": 3, "backend": "mss"},
                 {"id": "abc", "run": "x", "fps": 2, "start": "5"}):
        h.body = body
        serve_inspect.live_start(h)
    assert calls == [("screen", ("def", 3, "mss")), ("youtube", ("abc", "x", 2, 5.0))]
    h.body = {"source": "twitch"}
    with pytest.raises(ValueError):
        serve_inspect.live_start(h)


# ---------- паспорт, API, свои / враги в пути кадра (#799, #798, #790) ----------
#
# Подставные части пути кадра: чемпионы (LiveChampions) без моделей, детектор
# юнитов с заданным ответом, поиск миникарты, который находит её сразу, и
# рамка без прогрева — строки выходят на каждом кадре.

IDS = ["Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux", "Twitch",
       "Yuumi"]
HUD_ROSTER = ["Irelia", "Evelynn", "TwistedFate", "Jhin", "Zyra",
              "Volibear", "Hecarim", "Locke", "Jinx", "Lulu"]
RGB = {"green": (40, 200, 40), "blue": (40, 120, 230), "red": (210, 40, 40)}


class _Champs:
    """LiveChampions без моделей: раскладка по HUD задаётся тестом (`hud_view`,
    `hud_roster`), состав из API — как у настоящего set_api; миникарта видит
    всех десятерых в одной точке (и мёртвых — это конфликт с API)."""

    made: list[_Champs] = []

    def __init__(self, device="cpu", hud_view=None, hud_roster=None):
        _Champs.made.append(self)
        self.names, self.notes, self.error, self.api_error = None, [], None, None
        self.hud_view, self.reads = hud_view, [hud_view] * 4
        self.match = (None if hud_roster is None else
                      SimpleNamespace(api=False, roster=list(hud_roster), me_team=None,
                                      sides=("ORDER", "CHAOS")))
        self.minimap_calls = self.who_calls = 0
        self.closed = False

    @property
    def roster(self):
        return None if self.match is None else self.match.roster

    @property
    def view(self):
        if self.match is not None and self.match.api:
            return "player" if self.match.me_team else "spectator"
        return self.hud_view

    def view_how(self):
        if self.view is None:
            return "раскладка не решена: задано тестом"
        return f"HUD: задано тестом ({self.view})"

    def set_api(self, obs):
        if obs is None:
            if self.match is not None and self.match.api:
                self.match = None
            return None
        first = obs.me_team or "ORDER"
        sides = (first, "CHAOS" if first == "ORDER" else "ORDER")
        roster = [p.name for s in sides for p in obs.players if p.team == s]
        self.match = SimpleNamespace(api=True, roster=roster, me_team=obs.me_team, sides=sides)
        return None

    def offer(self, frame, t):
        pass

    def minimap(self, frame, mini):
        self.minimap_calls += 1
        m = self.match
        if m is None:
            return None
        out = []
        for k, n in enumerate(m.roster):
            team = m.sides[k // 5]
            slot = f"{'LR'[('ORDER', 'CHAOS').index(team)]}{k % 5 + 1}"
            out.append(MinimapChamp(n, team, slot, True, 0.5, 0.5, None, None))
        return out

    def who(self, frame, units):
        self.who_calls += 1
        return [None] * len(units)

    def close(self):
        self.closed = True


class _Locator:
    """Поиск миникарты, который находит стандартную с первой попытки."""

    made: list = []

    def __init__(self, size, saved=None, manual=None):
        w, h = size
        self.size, self.region = size, (int(0.7 * w), int(0.5 * h), w, h)
        self.saved, self.manual = saved, manual
        self.box = self.fit = None
        self.tried = 0
        _Locator.made.append(self)

    def offer_region(self, region):
        self.tried += 1
        self.fit = SimpleNamespace(box=STD_BOX, scale=1.0, score=0.8, how="map")
        self.box = STD_BOX
        return True

    def locate(self, region):
        return self.fit


class _Units:
    """Детектор юнитов с заданным ответом (пиксели кадра); маску запоминает."""

    def __init__(self, units):
        self.answer, self.calls, self.zones = units, 0, []

    def set_zone(self, keep, valid):
        self.zones.append(keep.shape)

    def units(self, fr_dev):
        self.calls += 1
        return list(self.answer)


# Плашки вида игрока при 1080p (как в tests/test_sides.py): свой — зелёная,
# союзник — синяя, враг — красная; `team` детектора своего — нарочно CHAOS.
PLAQUES = [(900, 500, "green", "CHAOS"), (500, 500, "blue", "ORDER"), (1400, 500, "red", "CHAOS")]


def _player_frame() -> np.ndarray:
    rgb = np.full((1080, 1920, 3), (110, 120, 105), np.uint8)
    for x, y, color, _ in PLAQUES:
        x0, top = x - 60, y - 115
        cv2.rectangle(rgb, (x0 - 4, top - 4), (x0 + 124, top + 19), (15, 15, 20), -1)
        cv2.rectangle(rgb, (x0, top), (x0 + 99, top + 9), RGB[color], -1)
    return rgb


@pytest.fixture
def rigged(monkeypatch, cpu_models, tmp_path):
    """Путь кадра с подставными чемпионами, детектором и поиском миникарты.
    Возвращает (build, настройка): настройка — что отдать путям кадра."""
    real_build, built = cpu_models
    mini = tmp_path / "mini"
    mini.mkdir()
    (mini / "model.pt").write_bytes(b"")
    cfg = {"hud_view": None, "hud_roster": None, "units": None}
    _Champs.made = []
    monkeypatch.setattr(live, "MINI_RUN", mini)
    monkeypatch.setattr(live, "LiveChampions", lambda device: _Champs(
        device, cfg["hud_view"], cfg["hud_roster"]))
    monkeypatch.setattr(live, "Locator", _Locator)

    def build(size, run, device, fps, **kw):
        p = real_build(size, run, device, fps, **kw)
        if cfg["units"] is not None:
            w, h = size
            p.unit_model = _Units(cfg["units"])
            p.proj = live.Projection.load(p.crop, size)
            p.tracker = live.UnitTracker(p.proj, valid=live._unit_zone(w, h)[2])
        p.bw, p.bh = 70, 40                      # без прогрева рамки: строка на каждый кадр
        return p
    return build, cfg, built


def _off(row) -> dict:
    return row["off"]


def test_minimap_found_by_reference_is_saved_and_offered_next_time(rigged, tmp_path):
    """Свой экран (#803): прямоугольник, найденный эталоном, — в файл вне
    репозитория; следующий поиск того же размера получает его и масштаб из
    настройки. Без файла (ролик) ничего не пишется."""
    build, _, _ = rigged
    store = tmp_path / "lolstream" / "minimap_fit.json"
    s = live.Session("screen", "r", _Source([_ok(0.0), _ok(0.5)], kind="screen"),
                     device="cpu", build=build, minimap_store=store, minimap_scale=1.3)
    st = _finish(s)
    loc = _Locator.made[-1]
    assert loc.saved is None and loc.manual == 1.3            # файла ещё не было
    assert load_fit(store, (640, 360)).box == STD_BOX
    assert st["passport"]["items"]["minimap"]["how"] == "найдена в кадре: эталон Ущелья"
    save_fit(store, (640, 360), Fit((0.8, 0.7, 0.99, 0.98), 1.2, 0.6))
    _finish(live.Session("screen", "r", _Source([_ok(0.0)], kind="screen"), device="cpu",
                         build=build, minimap_store=store))
    assert _Locator.made[-1].saved.box == (0.8, 0.7, 0.99, 0.98)
    _finish(live.Session("v", "r", _Source([_ok(0.0)]), device="cpu", build=build))
    assert _Locator.made[-1].saved is None


def test_screen_passport_player_view_refuses_spectator_readers(rigged):
    build, cfg, _ = rigged
    src = _Source([_ok(0.0), _ok(0.5)], kind="screen")
    s = live.Session("screen", "r", src, device="cpu", build=build)
    st = _finish(s)
    p = st["passport"]
    view = p["items"]["view"]
    # Свой экран — вид игрока сразу, а не unknown на прогреве.
    assert (view["value"], view["source"], view["how"]) == ("player", "source",
                                                           "свой экран: своя игра")
    assert st["view"] == "player" and all(r["view"] == "player" for r in st["frames"])
    dis = {d["reader"]: d["reason"] for d in p["disabled"]}
    assert dis["names"] == dis["hud_roster"] == "нужен вид наблюдателя, а это вид игрока"
    assert p["items"]["minimap"]["value"]["found"] is True
    assert p["items"]["game_rect"]["value"] == [0, 0, 640, 360]
    row = st["frames"][-1]
    assert "camera_truth" not in _off(row) and "truth" not in row   # миникарта найдена
    assert _off(row)["names"] == dis["names"]
    # Изменения паспорта — в событиях и в note.
    keys = [e["key"] for e in st["passport_events"]]
    assert "item:minimap" in keys and "item:view" in keys
    assert st["passport_events_n"] == len(st["passport_events"])
    assert "паспорт:" in st["note"]
    json.dumps(st)


def test_youtube_spectator_passport_refuses_sides_and_api(rigged):
    build, cfg, _ = rigged
    cfg.update(hud_view="spectator", hud_roster=HUD_ROSTER)
    s = live.Session("vid", "r", _Source([_ok(0.0), _ok(0.5)]), device="cpu", build=build)
    st = _finish(s)
    p = st["passport"]
    assert p["items"]["view"]["value"] == "spectator" and p["items"]["roster"]["value"] == "hud"
    dis = {d["reader"]: d["reason"] for d in p["disabled"]}
    assert dis["sides"] == "нужен вид игрока, а это вид наблюдателя"
    assert dis["api_facts"] == "Live Client API: не подключён — нет — источник YouTube"
    row = st["frames"][-1]
    # Состав по HUD у наблюдателя принят: чемпионы миникарты в строке.
    assert [c["name"] for c in row["champs"]] == HUD_ROSTER and st["roster"] == HUD_ROSTER
    assert row["game_time"]["value"] is None and "YouTube" in row["game_time"]["reason"]


def test_unknown_view_runs_no_detector_and_refuses_with_reason(rigged):
    build, cfg, _ = rigged
    cfg.update(hud_view=None, hud_roster=HUD_ROSTER,
               units=[Unit("champion", "ORDER", 0.9, 0.1, 1.0, 300.0, 200.0)])
    s = live.Session("vid", "r", _Source([_ok(0.0), _ok(0.5)]), device="cpu", build=build)
    st = _finish(s)
    p = st["passport"]
    assert p["items"]["view"]["value"] == "unknown"
    assert p["items"]["view"]["how"].startswith("раскладка не решена")
    row = st["frames"][-1]
    # Вид не решён: детектор не запускался, юниты — null с причиной, а не
    # ответы по маске наблюдателя; состав по HUD не принят.
    assert s._pipe.unit_model.calls == 0
    assert row["units"] is None and "вид неизвестен" in _off(row)["units"]
    assert row["champs"] is None and st["roster"] is None
    assert p["items"]["roster"]["value"] is None
    assert "вид неизвестен" in st["note"]


class _ReadChamps(_Champs):
    """_Champs с настоящим правилом раскладки LiveChampions по чтениям `reads`."""

    view = lc.LiveChampions.view
    votes = lc.LiveChampions.votes
    view_how = lc.LiveChampions.view_how

    def __init__(self, reads, **kw):
        super().__init__(**kw)
        self.reads, self._view = deque(reads, maxlen=lc.VIEW_READS), None


@pytest.mark.parametrize("reads,want", [
    ([None] * 8, "unknown"),                                  # портретов нет — не игрок
    ([None] * 6 + ["spectator"] * 2, "unknown"),              # мало голосов
    ([None] * 4 + ["spectator"] * 4, "spectator"),
    (["player"] * 4 + [None] * 4, "player"),                  # панель игрока видна
])
def test_passport_view_only_from_positive_signs(rigged, monkeypatch, reads, want):
    # #799: без положительного признака вида игрока паспорт получает unknown
    # или spectator, а не player.
    build, cfg, _ = rigged
    monkeypatch.setattr(live, "LiveChampions", lambda device: _ReadChamps(reads))
    s = live.Session("vid", "r", _Source([_ok(0.0), _ok(0.5)]), device="cpu", build=build)
    st = _finish(s)
    view = st["passport"]["items"]["view"]
    assert view["value"] == want and st["view"] == want
    if want == "unknown":
        assert view["how"].startswith("раскладка не решена")
    else:
        assert view["source"] == "hud" and view["how"].startswith("HUD:")


def _api_session(rigged, n: int, forever=None, state_port=None):
    """Сеанс экрана 1920×1080 с плашками и Live Client по журналу своей игры:
    перед каждым кадром — один опрос журнала (без потока и порта 2999)."""
    build, cfg, _ = rigged
    cfg["units"] = [Unit("champion", team, 0.9, 0.1, 1.0, float(x), float(y))
                    for x, y, _, team in PLAQUES]
    api = LiveClient(transport=journal_transport(read_journal(OWN)), champion_ids=IDS)
    api.start = lambda: api                   # опрос — из источника, по кадру
    api.stop = lambda timeout=2.0: None
    img = _player_frame()
    src = _Source([Frame(Status.OK, float(k), img) for k in range(n)], kind="screen",
                  size=(1920, 1080), forever=forever, on_read=api.poll_once)
    return live.Session("screen", "r", src, device="cpu", build=build, api=api,
                        state_port=state_port)


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_api_feeds_context_roster_deaths_conflicts_and_sides(rigged):
    s = _api_session(rigged, 36)
    st = _finish(s)
    rows = st["frames"]
    assert len(rows) == 36
    last = rows[-1]
    # Состав — из API (ORDER, затем CHAOS), паспорт знает источник.
    assert [c["name"] for c in last["ctx"]] == IDS
    assert st["passport"]["items"]["roster"]["value"] == "api"
    assert st["passport"]["items"]["api"]["value"] == {"state": "in_game", "available": True}
    # Смерти из API — ровно переходы isDead журнала, с источником api.
    deaths = [(e["name"], e["src"]) for r in rows for e in r.get("ev", ())
              if e["type"] == "death"]
    assert deaths == [("Yone", "api"), ("Lux", "api"), ("Yone", "api"), ("Zilean", "api")]
    zil = next(c for c in last["ctx"] if c["name"] == "Zilean")
    assert (zil["state"], zil["alive"], zil["alive_src"], zil["respawn_src"]) == \
        ("dead", False, "api", "api")
    # Время матча — из API; на загрузке оно стоит на 0.1 с.
    assert last["game_time"]["src"] == "api" and last["game_time"]["value"] >= 337
    assert rows[0]["game_time"]["value"] == 0.1
    # Миникарта «видит» мёртвых по API — расхождение, а не перетирание.
    assert last["conflicts"]["minimap_dead"] > 0 and last["api_refused"] is None
    # Свои / враги / сам на виде игрока: по цвету плашки, сторона — из API.
    by_x = {u["x"]: u for u in last["units"]}
    me, ally, enemy = (by_x[round(x / 1920, 4)] for x, *_ in PLAQUES)
    assert (me["relation"], me["side"], me["side_src"]) == ("self", "ORDER", "api")
    assert (ally["relation"], ally["side"]) == ("ally", "ORDER")
    assert (enemy["relation"], enemy["side"]) == ("enemy", "CHAOS")
    assert all(u["relation_reason"] is None for u in last["units"])
    # На загрузке API ещё не «в игре»: сторона — null с причиной, отношение есть.
    first = rows[0]["units"][0]
    assert first["relation"] == "self" and first["side"] is None
    assert first["side_reason"].startswith("нет команды своего игрока: Live Client API: loading")
    # Детектор работал с маской вида игрока.
    assert s._pipe.unit_model.zones
    json.dumps(st)


@pytest.mark.skipif(not OWN.exists(), reason="нет журнала своей игры")
def test_state_service_runs_with_session(rigged):
    s = _api_session(rigged, 12, forever=gap(Status.PAUSED, 99.0), state_port=0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def state() -> dict:
        with opener.open(st["state_url"] + "/state", timeout=5) as r:
            return json.loads(r.read())
    try:
        st = _wait(s, lambda st: st["count"] >= 12 and st["state_url"] is not None)
        # /state — снимок последнего такта хаба (раз в 0,25 с), а 12 кадров
        # проходят быстрее: снимок мог быть собран до первого кадра
        # (sources.vision «absent»). Такт seq0 + 1 мог начаться раньше, такт
        # seq0 + 2 начат уже после 12-го кадра — его и ждём (#796).
        seq0 = state()["seq"]
        end = time.time() + 20
        while (snap := state())["seq"] < seq0 + 2 and time.time() < end:
            time.sleep(0.05)
        assert snap["seq"] >= seq0 + 2, "хаб состояния не сделал два такта за 20 с"
        assert check_snapshot(snap) == []
        assert snap["passport"]["value"]["items"]["view"]["value"] == "player"
        assert snap["sources"]["vision"]["state"] == "ok"
        assert snap["sources"]["vision"]["frame"] == 11          # последний из 12 кадров
        # Снимок хаба — с его такта: последний кадр или уже пауза после него.
        assert snap["sources"]["frames"]["state"] in ("ok", "paused")
        assert snap["match"]["side"]["value"] == "ORDER"
        assert [c["champion"]["value"] for c in snap["champions"]["value"]] == IDS
        rel = {u["relation"]["value"] for u in snap["units"]["value"]}
        assert rel == {"self", "ally", "enemy"}
    finally:
        s.stop()
        st = _finish(s)
    assert st["state_reason"] == "сеанс закрыт"


def test_resize_keeps_roster_in_new_pipeline(rigged):
    build, cfg, built = rigged
    cfg.update(hud_view="spectator", hud_roster=HUD_ROSTER)
    src = _Source([_ok(0.0), _ok(0.5), _ok(1.0, 800, 450), _ok(1.5, 800, 450)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert built == [(640, 360), (800, 450)]
    # Чемпионы одни на сеанс: состав, прочитанный до пересборки, остался.
    assert len(_Champs.made) == 1 and s._pipe.champs is _Champs.made[0]
    assert [c["name"] for c in st["frames"][-1]["champs"]] == HUD_ROSTER
    assert st["roster"] == HUD_ROSTER and "состав сохранён" in s._pipe.note
    # Паспорт: новый размер, миникарта найдена заново в новом пути.
    p = st["passport"]
    assert p["items"]["resolution"]["value"] == [800, 450]
    assert any(e["key"] == "item:resolution" and e["old"] == [640, 360]
               for e in st["passport_events"])
    assert p["items"]["minimap"]["value"]["found"] is True


class _BlindLocator(_Locator):
    """Первый поиск не находит ничего; каждый следующий — как _Locator."""

    def offer_region(self, region):
        if _Locator.made[0] is self:
            self.tried += 1
            return False
        return super().offer_region(region)


def test_minimap_search_gives_up_then_retries_after_pause(rigged, monkeypatch):
    """Сеанс «свой экран» переживает смену игры (ARAM → Ущелье): сдавшийся
    поиск миникарты через LOCATE_PAUSE начинает новый круг и находит её."""
    build, _, _ = rigged
    _Locator.made = []
    monkeypatch.setattr(live, "Locator", _BlindLocator)
    monkeypatch.setattr(live, "LOCATE_MAX", 2)
    monkeypatch.setattr(live, "LOCATE_PAUSE", 1.0)
    frames = [_ok(0.5 * k) for k in range(8)]
    st = _finish(live.Session("screen", "r", _Source(frames, kind="screen"),
                              device="cpu", build=build))
    assert len(_Locator.made) == 2                    # первый круг сдался, второй нашёл
    found = [e for e in st["passport_events"] if e["key"] == "item:minimap"]
    assert [e["new"]["found"] for e in found] == [True]   # отказ значение не менял
    assert st["passport"]["items"]["minimap"]["value"]["found"] is True


# ---------- готовность заклинаний D/F по HUD (#806) ----------

def _with_panel(t: float) -> Frame:
    """Кадр своего экрана 1920×1080 с панелью умений своего HUD на месте."""
    from lolvision import hud_spells as hs
    tpl = cv2.cvtColor(cv2.imread(str(hs.TEMPLATE)), cv2.COLOR_BGR2RGB)
    img = np.full((1080, 1920, 3), 60, np.uint8)
    x, y = hs.TEMPLATE_ORIGIN
    img[y:y + tpl.shape[0], x:x + tpl.shape[1]] = tpl
    return Frame(Status.OK, t, img)


def test_spell_panel_found_and_slots_read(cpu_models):
    """Свой экран — вид игрока: панель умений найдена (паспорт), эталоны
    готовых иконок набраны, оба слота «готово» в строке кадра."""
    build, _ = cpu_models
    frames = [_with_panel(2.0 * k) for k in range(16)]
    src = _Source(frames, kind="screen", size=(1920, 1080))
    st = _finish(live.Session("screen", "r", src, device="cpu", build=build))
    item = st["passport"]["items"]["spell_panel"]
    assert item["value"]["found"] is True and abs(item["value"]["scale"] - 1.0) < 0.03
    assert "hud_spells" in st["passport"]["enabled"]
    first = next(r for r in st["frames"] if r.get("spells"))
    assert {x["slot"] for x in first["spells"]} == {"D", "F"}
    last = st["frames"][-1]["spells"]
    assert [x["value"] for x in last] == [True, True]


# ---------- окно кадров своего экрана (#801) ----------

class _Reply:
    """Ответчик маршрута смотрелки: что отдал — код ошибки или тело."""

    def __init__(self):
        self.code, self.body = None, None

    def fail(self, code, text):
        self.code = code

    def _send(self, body, ctype, cache=0):
        self.code, self.body = 200, body


def _image(sid: str, kind: str, i: int) -> int:
    from lolstream import serve_inspect
    h = _Reply()
    if kind == "heat":
        serve_inspect.live_heat(h, sid, str(i))
    else:
        serve_inspect.live_image(h, sid, kind, str(i))
    return h.code


def test_screen_keeps_sliding_window_and_does_not_end(cpu_models, monkeypatch):
    """Свой экран не упирается в потолок MAX_FRAMES: держит последние
    SCREEN_WINDOW кадров, номера кадров сквозные, since — в этой нумерации."""
    build, _ = cpu_models
    monkeypatch.setattr(live, "SCREEN_WINDOW", 5)
    monkeypatch.setattr(live, "MAX_FRAMES", 4)          # у экрана не действует
    monkeypatch.setattr(live, "WARMUP_MAX", 1)          # кадр — сразу в строки
    src = _Source([_ok(0.0), _ok(0.5), _ok(1.0)], kind="screen",
                  forever=gap(Status.PAUSED, 99.0))
    s = live.Session("screen", "r", src, device="cpu", build=build)
    monkeypatch.setitem(live._sessions, "screen", s)
    try:
        # До выброса: окно не полно, всё с нуля.
        st = _wait(s, lambda st: st["count"] == 3)
        assert [r["i"] for r in st["frames"]] == [0, 1, 2]
        assert (st["window"], st["first"], st["dropped_before"]) == (5, 0, 0)
        assert [r["i"] for r in s.status(2)["frames"]] == [2]
        assert _image("screen", "mini", 0) == 200
        # Ещё девять кадров: окно сдвинулось, старые выброшены с картинками.
        src.script.extend(_ok(1.5 + k / 2) for k in range(9))
        st = _wait(s, lambda st: st["count"] == 12)
        assert st["state"] == live.State.RUNNING.label and st["active"] is True
        assert "потолок" not in st["note"]
        assert [r["i"] for r in st["frames"]] == [7, 8, 9, 10, 11]
        assert (st["first"], st["dropped_before"]) == (7, 7)
        st = s.status(9)
        assert [r["i"] for r in st["frames"]] == [9, 10, 11] and st["dropped_before"] == 0
        st = s.status(5)                       # 5 и 6 уже выброшены
        assert [r["i"] for r in st["frames"]] == [7, 8, 9, 10, 11]
        assert st["dropped_before"] == 2
        st = s.status(12)
        assert st["frames"] == [] and st["count"] == 12 and st["dropped_before"] == 0
        for store in (s.scene, s.mini, s.full, s.heat):
            assert sorted(store) == [7, 8, 9, 10, 11]
        assert s.mini.get(3) is None
        for kind in ("scene", "mini", "full", "heat"):
            assert _image("screen", kind, 3) == 404
            assert _image("screen", kind, 10) == 200
        assert next(x for x in live.listing() if x["id"] == "screen")["count"] == 12
        json.dumps(st)
    finally:
        s.stop()
        _finish(s)


def test_video_still_stops_at_the_ceiling(cpu_models, monkeypatch):
    """У ролика окна нет: потолок MAX_FRAMES, дальше сеанс не читает."""
    build, _ = cpu_models
    monkeypatch.setattr(live, "MAX_FRAMES", 4)
    monkeypatch.setattr(live, "SCREEN_WINDOW", 2)        # ролика не касается
    monkeypatch.setattr(live, "WARMUP_MAX", 1)
    src = _Source([_ok(k / 2) for k in range(200)])
    s = live.Session("vid", "r", src, device="cpu", build=build)
    st = _finish(s)
    assert st["state"] == live.State.ENDED.label
    assert "потолок памяти 4 кадров" in st["note"]
    assert 4 <= st["count"] < 200 and src.script          # ролик не дочитан
    assert st["window"] is None and st["first"] == 0 and st["dropped_before"] == 0
    assert [r["i"] for r in st["frames"]] == list(range(st["count"]))


def test_api_events_are_a_bounded_tail(monkeypatch):
    """События Live Client в памяти сеанса — хвостом, счёт — всех (#801)."""
    monkeypatch.setattr(live, "API_EVENTS_KEEP", 3)
    s = live.Session.__new__(live.Session)
    s._lock = threading.Lock()
    s.api_events = []
    batches = deque([[1, 2], [3, 4, 5], [6]])
    s.api = SimpleNamespace(events=lambda: batches.popleft() if batches else [])
    for _ in range(4):
        s._drain_api()
    assert s.api_events == [4, 5, 6] and s.api_events_n == 6
