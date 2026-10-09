"""Граница пакетов: модель не зависит от роликов YouTube, пакеты — от скриптов.

Разбор статический (ast), модули не импортируются: проверка не требует ни
torch, ни сети и ловит и импорты внутри функций.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {p.stem for p in (ROOT / "tools").glob("*.py")}

# Чего lolvision не касается: пакет роликов, YouTube, сеть, сервер и
# дочерние процессы (ffmpeg и yt-dlp живут в lolstream).
NOT_IN_LOLVISION = {"lolstream", "yt_dlp", "http", "socketserver", "urllib.request",
                    "subprocess"}


def imports(path: Path) -> set[str]:
    """Полные имена импортируемых модулей, включая импорты внутри функций."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: относительный импорт"
            out.add(node.module)
    return out


def hits(name: str, banned: set[str]) -> bool:
    """`http.server` попадает под `http`, `urllib.parse` под `urllib.request` — нет."""
    return any(name == b or name.startswith(b + ".") for b in banned)


def modules(pkg: str) -> list[Path]:
    return sorted((ROOT / pkg).glob("*.py"))


@pytest.mark.parametrize("path", modules("lolvision"), ids=lambda p: p.name)
def test_lolvision_stays_a_model_library(path):
    bad = sorted(m for m in imports(path) if hits(m, NOT_IN_LOLVISION))
    assert not bad, f"lolvision/{path.name} импортирует {bad}"


@pytest.mark.parametrize("path", modules("lolvision") + modules("lolstream"),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_packages_do_not_import_scripts(path):
    # Скрипты tools/ — командная строка поверх пакетов, не наоборот.
    bad = sorted(m for m in imports(path) if m.split(".")[0] in TOOLS)
    assert not bad, f"{path.parent.name}/{path.name} импортирует скрипты {bad}"


def test_banned_list_catches_submodules():
    assert hits("http.server", NOT_IN_LOLVISION)
    assert hits("lolstream.live", NOT_IN_LOLVISION)
    assert not hits("urllib.parse", NOT_IN_LOLVISION)
    assert not hits("lolvision.paths", NOT_IN_LOLVISION)
