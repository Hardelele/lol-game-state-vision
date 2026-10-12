"""tools/replay_state.py (#804): журнал Live Client → снимки `/state` по времени журнала.

Журнал — обезличенная своя игра (tests/fixtures/live_client/own_game.jsonl),
список чемпионов — тот же, что в tests/test_state_service.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import replay_state as rs  # noqa: E402
from lolstream.live_client import read_journal, replay  # noqa: E402
from lolstream.state_service import check_snapshot  # noqa: E402

OWN = Path(__file__).parent / "fixtures" / "live_client" / "own_game.jsonl"
IDS = ["Annie", "Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux",
       "Twitch", "Yuumi", "Fiddlesticks", "MonkeyKing"]


def riot_ids() -> set[str]:
    out = set()
    for st, _ in replay(read_journal(OWN), IDS):
        if st.game is not None:
            out |= {p.riot_id for p in st.game.players}
    return out


def test_snapshots_on_journal_clock_every_step():
    rows = read_journal(OWN)
    snaps = list(rs.snapshots(rows, IDS, step=5.0))
    ts = [t for t, _, _ in snaps]
    assert ts[0] == rows[0].t and ts[-1] == rows[-1].t        # первая и последняя строка
    # сетка t0 + k·step: в каждой клетке не больше одного снимка (кроме последней строки)
    cells = [int((t - ts[0]) // 5.0) for t in ts[:-1]]
    assert cells == sorted(set(cells))
    # журнал фикстуры прорежен (36 строк за 341 с): снимок — на первой строке клетки
    # плюс последняя строка, если она не первая в своей клетке
    cell = [int((r.t - rows[0].t) // 5.0) for r in rows]
    assert len(snaps) == len(set(cell)) + (cell[-1] == cell[-2])
    for n, (t, snap, _st) in enumerate(snaps, 1):
        assert check_snapshot(snap) == []
        assert snap["seq"] == n
        # часы — строка журнала, а не «сейчас»: снимок API получен в тот же момент
        assert snap["t"] == round(t, 3)
        if snap["sources"]["api"]["age"] is not None:
            assert snap["sources"]["api"]["age"] < 1.0
    last = snaps[-1][1]
    assert last["champions"]["value"] is not None and len(last["champions"]["value"]) == 10
    assert last["match"]["time"]["source"] == "api"
    assert last["match"]["time"]["age"] < 1.0


def test_step_zero_is_every_row():
    rows = read_journal(OWN)
    assert len(list(rs.snapshots(rows, IDS, step=0))) == len(rows)


def test_run_hides_names_by_default(tmp_path):
    out = tmp_path / "state.jsonl"
    n, errors = rs.run(OWN, out, 5.0, names=False, champion_ids=IDS)
    assert errors == [] and n > 0
    text = out.read_text(encoding="utf-8")
    names = riot_ids()
    assert names
    for name in names:
        assert name not in text and name.split("#", 1)[0] not in text
    rows = [json.loads(x) for x in text.splitlines()]
    assert len(rows) == n and set(rows[0]) == {"t_journal", "snapshot"}
    last = rows[-1]["snapshot"]
    assert last["me"]["riot_id"]["value"] == "me"
    aliases = {c["riot_id"]["value"] for c in last["champions"]["value"]}
    assert aliases == {"me"} | {f"player{k}" for k in range(1, 10)}
    assert check_snapshot(last) == []


def test_run_keeps_names_on_request(tmp_path):
    out = tmp_path / "state.jsonl"
    rs.run(OWN, out, 30.0, names=True, champion_ids=IDS)
    last = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])["snapshot"]
    assert {c["riot_id"]["value"] for c in last["champions"]["value"]} == riot_ids()


def test_anonymizer_replaces_names_inside_text():
    a = rs.Anonymizer()
    a.alias = {"Foo#EUW": "me", "Bar#RU1": "player1"}
    out = a.apply({"riot_id": {"value": "Bar#RU1"}, "reason": "Foo убит, Foobar жив",
                   "list": ["Bar#RU1 и Foo#EUW"]})
    assert out == {"riot_id": {"value": "player1"}, "reason": "me убит, Foobar жив",
                   "list": ["player1 и me"]}


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rs.ddragon, "champions", lambda *a, **k: IDS)
    out = tmp_path / "s.jsonl"
    assert rs.main([str(OWN), "--out", str(out)]) == 0
    monkeypatch.setattr(rs, "check_snapshot", lambda snap: ["$.x: сломано"])
    assert rs.main([str(OWN), "--out", str(out)]) == 1
    assert "$.x: сломано" in capsys.readouterr().err

    def missing(*a, **k):
        raise FileNotFoundError("нет кэша")
    monkeypatch.setattr(rs.ddragon, "champions", missing)
    assert rs.main([str(OWN), "--out", str(out)]) == 2


@pytest.mark.parametrize("step", [1.0, 60.0])
def test_last_row_always_snapshot(step):
    rows = read_journal(OWN)
    assert list(rs.snapshots(rows, IDS, step))[-1][0] == rows[-1].t
