import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import label_units_gold as g  # noqa: E402


def test_merge_keeps_confident_and_drops_near_duplicates():
    a = [{"kind": "minion", "team": "ORDER", "x": 100, "y": 100, "score": .5}]
    b = [{"kind": "minion", "team": "ORDER", "x": 105, "y": 100, "score": .9},
         {"kind": "champion", "team": "ORDER", "x": 104, "y": 100, "score": .6},
         {"kind": "minion", "team": "CHAOS", "x": 300, "y": 100, "score": .4}]
    out = g.merge([a, b])
    assert [(u["kind"], u["x"]) for u in out] == [("minion", 105), ("champion", 104), ("minion", 300)]


def test_clean_validates_and_clamps():
    row = g.clean({"frame": "f.jpg", "status": "done", "sec": 3,
                   "units": [{"kind": "structure", "team": "CHAOS", "x": 2000, "y": -5}]}, {"f.jpg"})
    assert row["units"] == [{"kind": "structure", "team": "CHAOS", "x": 1920.0, "y": 0.0}]
    with pytest.raises(ValueError):
        g.clean({"frame": "x.jpg", "status": "done"}, {"f.jpg"})
    with pytest.raises(ValueError):
        g.clean({"frame": "f.jpg", "status": "done", "units": [{"kind": "tank", "team": "ORDER"}]}, {"f.jpg"})


def test_store_writes_labels_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LABELS", tmp_path / "labels.jsonl")
    monkeypatch.setattr(g, "HISTORY", tmp_path / "labels.history.jsonl")
    st = g.Store()
    st.save({"frame": "a", "status": "draft", "units": []})
    st.save({"frame": "a", "status": "done", "units": []})
    st.save({"frame": "b", "status": "skip", "units": []})
    labels = [json.loads(s) for s in (tmp_path / "labels.jsonl").open(encoding="utf-8")]
    hist = (tmp_path / "labels.history.jsonl").read_text(encoding="utf-8").splitlines()
    assert [(r["frame"], r["status"]) for r in labels] == [("a", "done"), ("b", "skip")]
    assert len(hist) == 3
    assert {r["frame"]: r["status"] for r in g.Store().rows.values()} == {"a": "done", "b": "skip"}
