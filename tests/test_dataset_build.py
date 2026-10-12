"""Сборщики датасета: оборванный поток и прерванная запись манифеста (#788)."""

import os
import subprocess
import sys

import pytest

from lolstream.video_stream import short_stream, write_atomic
from lolvision.paths import RUNS

TOOLS = RUNS.parent / "tools"


def test_short_stream():
    assert short_stream(2054, 2054.3, 1.0) is None
    assert short_stream(1960, 2054.3, 1.0) is None              # 95.4% — полный
    assert "67 кадров из ~2054" in short_stream(67, 2054.3, 1.0)
    assert short_stream(500, 1000, 0.5) is None                  # шаг 2 с
    assert short_stream(400, 1000, 0.5, skip=200) is None        # с пропуском начала
    assert short_stream(0, 0.0, 1.0) is not None                 # длительность неизвестна


def test_write_atomic_replaces_whole_file(tmp_path):
    man = tmp_path / "v.csv"
    write_atomic(man, lambda f: f.write("a,b\n1,2\n"))
    assert man.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert [p.name for p in tmp_path.iterdir()] == ["v.csv"]


def test_write_atomic_interrupted_leaves_nothing(tmp_path):
    man = tmp_path / "v.csv"

    def broken(f):
        f.write("a,b\n")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_atomic(man, broken)
    assert list(tmp_path.iterdir()) == []                         # ни пустого, ни частичного
    man.write_text("old\n", encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        write_atomic(man, broken)
    assert man.read_text(encoding="utf-8") == "old\n"             # прежний цел


def test_tools_print_in_cp1251_console():
    # Консоль Windows по умолчанию: cp1251, без PYTHONUTF8.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "cp1251"
    code = "import _bootstrap; print('кадры → манифест')"
    out = subprocess.run([sys.executable, "-c", code], cwd=TOOLS, env=env, capture_output=True)
    assert out.returncode == 0, out.stderr
    assert "→".encode() in out.stdout
    out = subprocess.run([sys.executable, str(TOOLS / "build_coords.py"), "--help"],
                         env=env, capture_output=True)
    assert out.returncode == 0, out.stderr
