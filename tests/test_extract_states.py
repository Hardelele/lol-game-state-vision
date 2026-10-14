"""Офлайн-прогон ролика пачками (lolstream/extract.py, #843) без сети и видеокарты.

Путь кадра собирается настоящим `build_pipeline` на процессоре, как в
test_live_session: вместо модели камеры — заглушка, моделей юнитов,
чемпионов и типа сцены «нет на диске». Источник — сценарий кадров. Главное
свойство — пакетный путь (batch > 1) даёт те же записи, что
последовательный (batch 1), и не оставляет подмен в модуле live.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from lolstream import extract as ex
from lolstream import hud_pool, limits, live
from lolstream.frame_source import Frame, Status, gap
from lolstream.state_service import check_snapshot
from lolvision import hud_stats

NOWHERE = Path(__file__).parent / "fixtures" / "нет-такого-файла"
ROSTER = {"ORDER": ["Urgot", "Briar", "Heimerdinger", "Jinx", "Lux"],
          "CHAOS": ["Volibear", "Rammus", "Yone", "MissFortune", "Lulu"], "me": "Volibear"}


class _Runner:
    """Модель камеры-заглушка: точка зависит от яркости кадра, сеть не нужна."""

    kind, label, trained_on = "stub", "заглушка", []

    def describe(self, scene_size) -> str:
        return "заглушка"

    def prepare(self, scene):
        return float(np.asarray(scene).mean())

    def frame_prep(self, frame_size, crop, boxes, store_size):
        return None

    def predict(self, prep):
        v = prep / 255
        return ({"px": v, "py": 1 - v, "qx": v, "qy": 1 - v, "spread": 0.01},
                np.zeros((4, 4), np.float32), {"net": 0.0, "agg": 0.0})


@pytest.fixture
def cpu_models(monkeypatch, tmp_path):
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"")
    monkeypatch.setattr(live, "run_checkpoint", lambda run: ck)
    monkeypatch.setattr(live, "load_runner", lambda path, device: _Runner())
    monkeypatch.setattr(live, "MINI_RUN", NOWHERE)
    monkeypatch.setattr(live, "KIND_NET", NOWHERE / "net.pt")
    monkeypatch.setattr(live, "KIND_HEAD", NOWHERE / "head.pt")
    monkeypatch.setattr(live, "live_net", lambda: (None, None))


class _Source:
    """Ролик по сценарию: кадры, затем ENDED."""

    kind, backend, backend_note, start = "file", "подставной", None, 0.0

    def __init__(self, frames: list[Frame], fps: float = 1.0, size=(640, 360)):
        self.frames, self.fps, self._size = list(frames), fps, size
        self.closed = False

    @property
    def size(self):
        return self._size

    def open(self) -> None:
        pass

    def read(self) -> Frame:
        return self.frames.pop(0) if self.frames else gap(Status.ENDED, 99.0, "ролик кончился")

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "fps": self.fps,
                "size": self._size, "title": "сценарий", "duration": 10.0}


def _frames(n: int, w: int = 640, h: int = 360) -> list[Frame]:
    rng = np.random.default_rng(7)
    return [Frame(Status.OK, float(i), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            for i in range(n)]


def _run(frames: list[Frame], batch: int, **kw) -> tuple[list[dict], ex.OfflineSession]:
    recs: list[dict] = []
    s = ex.OfflineSession(_Source(frames), device="cpu", batch=batch, on_record=recs.append,
                          keep_row=True, hud_procs=0, **kw)
    s.process()
    return recs, s


def _same(rec: dict) -> dict:
    """Запись без давностей: они от момента выдачи (прогрев выдаёт кадры пачкой)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("age", "ms")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(rec)


# ---------- стадии и состав ----------

def test_stages_parse():
    assert ex.parse_stages(None) == ex.DEFAULT_STAGES
    assert ex.parse_stages("default") == ("scene", "camera", "minimap", "hud")
    assert ex.parse_stages("default,+units") == ("scene", "camera", "minimap", "hud", "units")
    assert ex.parse_stages("all,-names") == ("scene", "camera", "minimap", "hud", "units")
    assert ex.parse_stages("camera") == ("camera",)
    with pytest.raises(ValueError, match="units"):
        ex.parse_stages("default,+names")
    with pytest.raises(ValueError, match="неизвестная"):
        ex.parse_stages("default,+radar")


def test_roster_load_and_obs():
    r = ex.Roster.load(json.dumps(ROSTER))
    assert (r.me, r.me_team) == ("Volibear", "CHAOS")
    obs = r.obs()
    assert obs.problem() is None and obs.me == "Volibear" and obs.me_team == "CHAOS"
    assert [p.team for p in obs.players] == ["ORDER"] * 5 + ["CHAOS"] * 5
    assert not any(p.dead for p in obs.players)
    assert ex.Roster.from_dict(r.to_dict()) == r


@pytest.mark.parametrize("bad, why", [
    ({**ROSTER, "ORDER": ROSTER["ORDER"][:4]}, "5 \\+ 5"),
    ({**ROSTER, "me": "Teemo"}, "не в составе"),
    ({**ROSTER, "me_team": "ORDER"}, "не в команде"),
])
def test_roster_rejects(bad, why):
    with pytest.raises(ValueError, match=why):
        ex.Roster.from_dict(bad)


def test_roster_from_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROSTER), encoding="utf-8")
    assert ex.Roster.load(str(p)).me == "Volibear"


# ---------- кадры и запись ----------

def test_chunks():
    assert list(ex.chunks(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(ex.chunks([], 4)) == []


def test_prefetch_order_and_end():
    src = _Source(_frames(5))
    got = [fr.t for fr, ms in ex.prefetch(src, 2)]
    assert got == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_prefetch_raises_reader_error():
    class Broken(_Source):
        def read(self):
            raise OSError("труба порвалась")
    with pytest.raises(OSError, match="труба"):
        list(ex.prefetch(Broken([]), 2))


def test_prefetch_close_early_frees_reader():
    src = _Source(_frames(30))
    it = ex.prefetch(src, 1)
    next(it)
    it.close()                                   # поток чтения не висит на полной очереди
    time.sleep(0.05)
    assert len(src.frames) < 30


def test_writer_gz_roundtrip(tmp_path):
    w = ex.Writer(tmp_path / "a" / "x.jsonl.gz")
    w({"video_t": 1.0, "v": np.float32(0.5), "n": np.int64(3), "s": {"a"}})
    w({"video_t": 2.0})
    w.close()
    with gzip.open(tmp_path / "a" / "x.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows == [{"video_t": 1.0, "v": 0.5, "n": 3, "s": ["a"]}, {"video_t": 2.0}]
    assert w.n == 2


def test_patch_rows_like_predict():
    class M:
        grid, has_game = 2, True
        net = type("N", (), {"frame_threshold": 0.5})()
    heat = [1.0, 2.0, 3.0, 4.0]
    out = np.array([[0.3, 0.4, 0.9, 0.31, 0.41, 0.02, 0.8, 12, 0.9, *heat],
                    [0.3, 0.4, 0.9, 0.31, 0.41, 0.02, 0.8, 12, 0.1, *heat]], np.float32)
    (r0, h0), (r1, _) = ex.patch_rows(M(), out)
    assert r0["is_game"] and r0["px"] == pytest.approx(0.3) and r0["votes"] == 12
    assert h0.shape == (2, 2)
    assert not r1["is_game"] and np.isnan(r1["px"]) and np.isnan(r1["py"])


def test_mini_key_tells_copies_apart():
    a = np.zeros((40, 50, 3), np.uint8)
    r = a[10:, 20:]
    assert ex._mini_key(r, (1, 2, 3, 4)) == ex._mini_key(a[10:, 20:], [1, 2, 3, 4])
    assert ex._mini_key(r.copy(), (1, 2, 3, 4)) != ex._mini_key(r, (1, 2, 3, 4))
    assert ex._mini_key(r, (1, 2, 3, 5)) != ex._mini_key(r, (1, 2, 3, 4))


# ---------- сеанс ----------

def test_session_records_follow_frames(cpu_models):
    recs, s = _run(_frames(6), batch=1)
    assert [r["video_t"] for r in recs] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [r["frame"] for r in recs] == list(range(6))
    for r in recs:
        assert check_snapshot(r["state"]) == []
        assert r["state"]["sources"]["vision"]["stream_t"] == r["video_t"]
        assert r["game_t"] is None                  # без HUD своего игрока часов нет
    assert s.state is live.State.ENDED and s.source.closed


@pytest.mark.parametrize("batch", [2, 4, 16])
def test_batch_equals_sequential(cpu_models, batch):
    one, _ = _run(_frames(9), batch=1)
    many, s = _run(_frames(9), batch=batch)
    assert [_same(r) for r in many] == [_same(r) for r in one]
    assert s.misses == {}                       # всё, что считала пачка, взято из пачки


def test_batch_survives_size_change(cpu_models):
    fr = _frames(5) + _frames(4, 960, 540)
    for i, f in enumerate(fr):
        fr[i] = Frame(Status.OK, float(i), f.image)
    one, _ = _run(list(fr), batch=1)
    many, s = _run(list(fr), batch=3)
    assert [r["video_t"] for r in many] == [float(i) for i in range(9)]
    assert [_same(r) for r in many] == [_same(r) for r in one]
    assert s._pipe.size == (960, 540)


def test_live_module_restored(cpu_models):
    before = (live.time, live._mini, live._full_small, live.white_mask, live.find_box_clipped)
    _run(_frames(3), batch=2)
    assert (live.time, live._mini, live._full_small, live.white_mask,
            live.find_box_clipped) == before


def test_clock_is_video_time(cpu_models):
    """Давности в снимке — по ролику: кадр выдан в момент своего t."""
    recs, _ = _run(_frames(4), batch=1)
    assert [r["state"]["t"] for r in recs][-1] == 3.0


def test_stages_off(cpu_models):
    recs, s = _run(_frames(3), batch=1, stages=("camera",))
    st = recs[-1]["state"]
    assert st["me"]["cs"]["value"] is None
    assert "тип сцены выключен" in s.note


def test_roster_only_on_player_view(cpu_models):
    class Champs:
        match = None
        sets: list = []

        def set_api(self, obs):
            self.sets.append(obs)
    s = ex.OfflineSession(_Source([]), device="cpu", roster=ex.Roster.from_dict(ROSTER),
                          hud_procs=0)
    s._pipe = type("P", (), {"champs": Champs()})()
    s.view = "spectator"
    assert s._api_step(0.0, 0.0) == (None, None) and Champs.sets == []
    s.view = "player"
    s._api_step(1.0, 1.0)
    assert len(Champs.sets) == 1 and Champs.sets[0].me == "Volibear"
    assert s._api_obs is Champs.sets[0]


# ---------- скромный режим ----------

def test_env_limits_keep_given(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    limits.env_limits(4)
    assert os.environ["OMP_NUM_THREADS"] == "2" and os.environ["MKL_NUM_THREADS"] == "4"


def test_apply_limits_without_priority():
    assert limits.apply_limits(4, priority=False) == {"threads": 4, "priority": "обычный"}


def test_hud_pool_reads_like_reader():
    rng = np.random.default_rng(3)
    for st in (np.zeros((36, 420, 3), np.uint8), rng.integers(0, 255, (36, 420, 3), np.uint8)):
        assert hud_pool.read(st) == hud_stats.StatsReader().read_strip(st)
