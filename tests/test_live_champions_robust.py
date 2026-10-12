"""Поток состава не умирает молча, сеанс честно пишет, чего не умеет (#788)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from lolstream import live_champions as lc
from lolvision.roster import FrameSlots


@pytest.fixture
def champs(monkeypatch, tmp_path):
    """LiveChampions на CPU без моделей и Data Dragon; банка имён нет."""
    icons = {c: np.zeros((8, 8, 3), np.uint8) for c in ("Ahri", "Jinx", "Lulu")}
    model = SimpleNamespace(champions=["Ahri", "Jinx", "Zed"], device="cpu")
    monkeypatch.setattr(lc, "IconModel", lambda run, device: model)
    monkeypatch.setattr(lc.ddragon, "load_icons", lambda: icons)
    monkeypatch.setattr(lc, "IconBank", lambda icons: SimpleNamespace(ids=sorted(icons)))
    monkeypatch.setattr(lc.hud_panels, "DigitReader", lambda: None)
    obj = lc.LiveChampions("cpu", names=tmp_path / "names.npz")
    yield obj
    obj.close()


def _wait(obj, timeout=5.0):
    end = time.time() + timeout
    while obj._busy and time.time() < end:
        time.sleep(0.01)
    assert not obj._busy


def test_roster_thread_survives_error(champs, monkeypatch):
    calls = []

    def read_frame(frame, bank):
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("Hwei")
        return FrameSlots(np.zeros((10, 2, 3), np.float32))     # портретов нет

    monkeypatch.setattr(lc, "read_frame", read_frame)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    champs.offer(frame, 0.0)
    _wait(champs)
    assert champs.error is not None and "KeyError" in champs.error["text"]
    assert champs.roster is None and champs._th.is_alive()
    champs.offer(frame, lc.ROSTER_EVERY)                 # следующее окно
    _wait(champs)
    assert len(calls) == 2 and champs._th.is_alive()
    assert list(champs.reads) == [None]          # признаков нет — не «игрок»


def test_no_name_bank_is_said_and_who_is_silent(champs):
    assert champs.names is None
    assert "без имён: нет банка имён" in champs.notes
    champs.match = SimpleNamespace(roster=["Ahri"] * 10)
    units = [SimpleNamespace(kind="champion", team="ORDER", x=900, y=500)]
    assert champs.who(np.zeros((1080, 1920, 3), np.uint8), units) == [None]


def test_champion_lists_mismatch_is_a_note_not_a_crash(champs):
    assert any("нет в Data Dragon: Zed" in n for n in champs.notes)
    assert any("модель миникарты не знает: Lulu" in n for n in champs.notes)
