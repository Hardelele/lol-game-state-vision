"""Подписи состояний: webapp/app.js сравнивает поле state с этими строками."""

from __future__ import annotations

from lolstream import channels
from lolstream import live


def test_live_labels_unchanged():
    assert {s.name: s.label for s in live.State} == {
        "PREPARING": "готовлюсь", "PROBING": "спрашиваю YouTube", "RUNNING": "идёт",
        "STOPPED": "остановлено", "ENDED": "ролик кончился", "FAILED": "сбой"}
    # app.js: going = ["готовлюсь", "спрашиваю YouTube", "идёт"].includes(state)
    assert {s.label for s in live.State if s.active} == {
        "готовлюсь", "спрашиваю YouTube", "идёт"}


def test_job_state_from_return_code():
    of = channels.JobState.of
    assert (of(None).label, of(0).label, of(1).label) == ("идёт", "готово", "сбой")
    assert channels.JobState.ALREADY_RUNNING.label == "уже идёт"
