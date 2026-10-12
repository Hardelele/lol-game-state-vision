"""Запись своей игры (#800): подставной источник кадров и подставной транспорт API.

Экрана и сети нет: кадры и отказы подаются списком, ответы allgamedata —
образцом Riot из tests/fixtures/live_client.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lolstream.frame_source import Frame, Status, gap
from lolstream.live_client import Response, State, read_journal, replay

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import record_own_game as rog  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "live_client"
IDS = ["Annie", "Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux", "Twitch"]


def game(events: list[dict], game_time: float) -> Response:
    d = json.loads((FIX / "liveclientdata_sample.json").read_text(encoding="utf-8"))
    d["events"] = {"Events": events}
    d["gameData"]["gameTime"] = game_time
    return Response(200, json.dumps(d))


START = {"EventID": 0, "EventName": "GameStart", "EventTime": 0.0}
END = {"EventID": 1, "EventName": "GameEnd", "EventTime": 600.0, "Result": "Win"}


class FakeSource:
    """Отдаёт кадры по списку с паузой; список кончился — повторяет последний."""

    kind, backend, backend_note = "screen", "fake", "подставлен"

    def __init__(self, script: list[str], fps: float = 100.0, endless: bool = True):
        self.script, self.fps, self.endless = script, fps, endless
        self.i, self.opened, self.closed = 0, False, False

    def open(self):
        self.opened = True

    def read(self) -> Frame:
        time.sleep(1 / self.fps)
        if self.i >= len(self.script) and not self.endless:
            return gap(Status.ENDED, self.i, "кончился")
        kind = self.script[min(self.i, len(self.script) - 1)]
        t = float(self.i)
        self.i += 1
        if kind == "ok":
            img = np.full((36, 64, 3), (self.i * 7) % 255, np.uint8)
            return Frame(Status.OK, t, img)
        return gap(Status.PAUSED if kind == "paused" else Status.NO_WINDOW, t)

    def close(self):
        self.closed = True

    def describe(self):
        return {"kind": self.kind, "backend": self.backend, "backend_note": self.backend_note,
                "fps": self.fps, "size": (64, 36), "title": "fake", "notes": []}


class ScriptTransport:
    """Ответы по списку; кончились — «нет соединения» (порт закрыт)."""

    def __init__(self, responses: list[Response]):
        self.responses, self.calls = list(responses), 0
        self.lock = threading.Lock()

    def __call__(self, endpoint: str) -> Response:
        assert endpoint == "allgamedata"
        with self.lock:
            self.calls += 1
            if self.responses:
                return self.responses.pop(0)
        return Response(None, None, "нет соединения: refused")


def recorder(tmp_path, source, transport, **kw) -> rog.Recorder:
    out = rog.new_record_dir(tmp_path / "own", time.time())
    kw.setdefault("api_period", 0.01)
    kw.setdefault("champion_ids", IDS)
    return rog.Recorder(out, source, transport, log=lambda s: None, **kw)


def test_new_record_dir_never_reuses(tmp_path):
    now = time.time()
    a = rog.new_record_dir(tmp_path, now)
    (a / "keep.txt").write_text("x")
    b = rog.new_record_dir(tmp_path, now)
    c = rog.new_record_dir(tmp_path, now)
    assert len({a, b, c}) == 3 and b.name == a.name + "-2" and c.name == a.name + "-3"
    assert (a / "keep.txt").read_text() == "x"


def test_frames_gaps_journal_and_manifest(tmp_path):
    src = FakeSource(["ok", "paused", "ok", "no_window", "ok"], endless=False)
    tr = ScriptTransport([Response(None, None, "нет соединения: refused"),
                          game([START], 61.0), game([START], 62.0)])
    rec = recorder(tmp_path, src, tr)
    m = rec.run()
    out = rec.out

    rows = [json.loads(s) for s in (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["status"] for r in rows] == ["ok", "paused", "ok", "no_window", "ok"]
    assert all(isinstance(r["t"], float) and r["t"] > 1e9 for r in rows)    # unix-время
    for r in rows:
        if r["status"] == "ok":
            assert r["file"] == f"full/{r['i']:06d}.jpg" and r["size"] == [64, 36]
            assert r["reason"] is None
            with Image.open(out / r["file"]) as im:
                assert im.size == (64, 36) and im.format == "JPEG"
        else:
            assert r["file"] is None and r["size"] is None and r["reason"]
    assert sorted(p.name for p in (out / "full").iterdir()) == \
        ["000000.jpg", "000002.jpg", "000004.jpg"]

    journal = read_journal(out / "api.jsonl")
    assert len(journal) >= 1 and journal[0].response.code is None
    assert "нет соединения" in journal[0].response.error
    assert [j.t for j in journal] == sorted(j.t for j in journal)
    # журнал прогоняется тем же Tracker, что и живой опрос
    states = [s.state for s, _ in replay(journal, IDS)]
    assert states[0] is State.NO_GAME

    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == m
    assert not list(out.glob("*.tmp"))
    assert m["schema"] == rog.MANIFEST_SCHEMA and m["id"] == out.name
    assert m["complete"] and m["finished"] >= m["started"]
    assert m["stop_reason"].startswith("источник")
    assert m["frames"] == {"responses": 5, "saved": 3, "gaps": 2,
                           "gaps_by_status": {"paused": 1, "no_window": 1},
                           "sizes": {"64x36": 3}, "jpeg_quality": 90}
    assert m["capture"]["backend"] == "fake" and m["fps"] == src.fps
    assert m["api"]["rows"] == len(journal)
    assert src.opened and src.closed


def test_stops_after_game_end_with_side_and_champion(tmp_path):
    src = FakeSource(["ok"])                          # бесконечный поток кадров
    tr = ScriptTransport([game([START], 500.0), game([START], 501.0),
                          game([START, END], 600.0)])
    rec = recorder(tmp_path, src, tr, end_grace=0.3, max_sec=20)
    t0 = time.monotonic()
    m = rec.run()
    assert time.monotonic() - t0 < 10
    assert m["stop_reason"].startswith("конец игры по API")
    assert m["api"]["last_state"] == "ended"
    assert m["api"]["my_team"] == "ORDER" and m["api"]["my_champion"] == "Annie"
    assert m["api"]["ended_reason"].startswith("GameEnd")
    journal = read_journal(rec.out / "api.jsonl")
    assert [j.response.code for j in journal][:3] == [200, 200, 200]
    assert State.ENDED in [s.state for s, _ in replay(journal, IDS)]
    # конец игры не останавливает сразу: после него ещё пишутся кадры
    rows = [json.loads(s) for s in
            (rec.out / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["t"] - m["api"]["game_ended_at"] >= 0.25


def test_max_length_and_periodic_manifest(tmp_path):
    src = FakeSource(["ok", "paused"])
    seen = []
    rec = recorder(tmp_path, src, ScriptTransport([]), max_sec=0.5, manifest_every=0.1)
    orig = rec.write_manifest

    def spy():
        orig()
        seen.append(json.loads((rec.out / "manifest.json").read_text(encoding="utf-8")))
    rec.write_manifest = spy
    m = rec.run()
    assert m["stop_reason"].startswith("предел длины")
    assert len(seen) >= 3 and not seen[0]["complete"] and seen[-1]["complete"]
    assert m["api"]["last_state"] == "no_game" and "my_team" not in m["api"]


def test_ctrl_c_still_writes_manifest(tmp_path):
    class Interrupting(FakeSource):
        def read(self):
            if self.i == 3:
                raise KeyboardInterrupt
            return super().read()

    rec = recorder(tmp_path, Interrupting(["ok"]), ScriptTransport([]))
    m = rec.run()
    assert m["stop_reason"] == "Ctrl+C" and m["complete"] and m["frames"]["saved"] == 3
    on_disk = json.loads((rec.out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["stop_reason"] == "Ctrl+C"


def test_record_dir_must_be_new(tmp_path):
    d = tmp_path / "x"
    (d / "full").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        rog.Recorder(d, FakeSource(["ok"]), ScriptTransport([]), log=lambda s: None)
