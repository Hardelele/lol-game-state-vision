"""Подписи состояний: webapp/app.js сравнивает поле state с этими строками."""

from __future__ import annotations

from pathlib import Path

from lolstream import channels
from lolstream import live


def test_live_labels_unchanged():
    # Источник — ролик или свой экран (#794): подписи не про YouTube.
    assert {s.name: s.label for s in live.State} == {
        "PREPARING": "готовлюсь", "PROBING": "открываю источник", "RUNNING": "идёт",
        "STOPPED": "остановлено", "ENDED": "кончился", "FAILED": "сбой"}
    # app.js опрашивает дальше по полю active status(), а не по подписям.
    assert {s.label for s in live.State if s.active} == {
        "готовлюсь", "открываю источник", "идёт"}
    js = (Path(live.__file__).parent / "webapp" / "app.js").read_text(encoding="utf-8")
    assert "if (d.active) S.liveTimer" in js


def test_job_state_from_return_code():
    of = channels.JobState.of
    assert (of(None).label, of(0).label, of(1).label) == ("идёт", "готово", "сбой")
    assert channels.JobState.ALREADY_RUNNING.label == "уже идёт"
