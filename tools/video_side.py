"""Сторона записанного игрока — из описания ролика, а не из догадок.

В названии стороны нет, но в описании канал публикует timeline матча:

    21:16 Red takes Baron
    22:18 Mid Inhibitor falls (Blue)
    22:45 Victory

и в шапке результат, «(WIN)» или «(LOSS)». Отсюда сторона выводится
однозначно: ломают базу проигравшему, значит при победе записанный игрок
играет за сторону, противоположную той, чей ингибитор пал последним.

Проверено на ролике olmTXkkUv58, где сторону независимо определил человек,
посмотрев миникарту: правило даёт тот же ответ, а две эвристики по
координатам камеры — противоположный. Поэтому координаты для этого больше
не используются.

Пример:
    python tools/video_side.py                 # все ролики с info.json
    python tools/video_side.py --id 58w57eJ5Qks
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from paths import ROOT

VIDEOS = ROOT / "data" / "videos"
ROLES = ("Top", "Jungle", "Mid", "ADC", "Bot", "Support")


def fetch_info(vid: str) -> dict:
    """Описание без скачивания самого ролика."""
    out = subprocess.run(
        ["python", "-m", "yt_dlp", "--skip-download", "--dump-json",
         f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"не удалось получить описание {vid}")
    return json.loads(out.stdout)


def read_info(vid: str) -> dict:
    p = VIDEOS / f"{vid}.info.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return fetch_info(vid)


def analyse(info: dict) -> dict:
    title = info.get("title") or ""
    desc = info.get("description") or ""

    role = next((r for r in ROLES if re.search(rf"\b{r}\b", title)), None)
    won = bool(re.search(r"\(WIN\)", desc))
    lost = bool(re.search(r"\(LOSS\)", desc))

    inhib = re.findall(r"Inhibitor falls \((Blue|Red)\)", desc)
    nexus = re.findall(r"(Blue|Red) (?:destroys|takes) Nexus", desc)

    side, why, sure = None, "нет признака", False
    if nexus:
        # Прямое указание: кто снёс нексус, тот и победил.
        winner = nexus[-1]
        if won or lost:
            side = winner if won else ("Red" if winner == "Blue" else "Blue")
            why = f"нексус снесли {winner}, у нас {'победа' if won else 'поражение'}"
            sure = True
    elif inhib and (won or lost):
        loser = inhib[-1]                       # базу ломают проигравшему
        other = "Red" if loser == "Blue" else "Blue"
        side = other if won else loser
        why = (f"последним пал ингибитор {loser}, "
               f"у нас {'победа' if won else 'поражение'}")
        sure = True
    elif inhib:
        why = f"ингибитор {inhib[-1]} пал, но результат матча не указан"
    elif won or lost:
        # Запасной довод для быстрых побед, где строки про ингибитор нет:
        # команда, забравшая объекты, обычно и выигрывает. Это не строгий
        # вывод, как с ингибитором, поэтому помечается отдельно.
        obj = re.findall(r"(Blue|Red) takes (?:Baron|\dth Dragon|\dst Dragon|"
                         r"\dnd Dragon|\drd Dragon)", desc)
        if len(obj) >= 3:
            top_side = max(set(obj), key=obj.count)
            share = obj.count(top_side) / len(obj)
            if share >= 0.75:
                side = top_side if won else ("Red" if top_side == "Blue" else "Blue")
                why = (f"строки про ингибитор нет; {top_side} забрали "
                       f"{obj.count(top_side)} из {len(obj)} объектов, "
                       f"у нас {'победа' if won else 'поражение'} — вероятно")

    ru = {"Blue": "синие", "Red": "красные"}
    return {"id": info.get("id"), "title": title, "role": role,
            "result": "WIN" if won else ("LOSS" if lost else "?"),
            "side": side, "side_ru": ru.get(side, "?"),
            "why": why, "confident": sure,
            "duration": info.get("duration")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None, help="куда сложить JSON")
    args = ap.parse_args()

    ids = args.id or sorted(p.stem.replace(".info", "")
                            for p in VIDEOS.glob("*.info.json"))
    rows = []
    for vid in ids:
        try:
            rows.append(analyse(read_info(vid)))
        except Exception as exc:                              # noqa: BLE001
            print(f"  {vid}: {exc}")
    print(f"{'ролик':14s} {'роль':9s} {'итог':5s} {'сторона':9s} основание")
    for r in rows:
        mark = "" if r["confident"] else "  ← ненадёжно"
        print(f"{r['id']:14s} {str(r['role']):9s} {r['result']:5s} "
              f"{r['side_ru']:9s} {r['why']}{mark}")

    by = {}
    for r in rows:
        if r["confident"]:
            by.setdefault((r["role"], r["side_ru"]), []).append(r["id"])
    print("\nпокрытие роль × сторона:")
    for role in ROLES:
        cells = [f"{s}: {len(by.get((role, s), []))}" for s in ("синие", "красные")]
        if any(by.get((role, s)) for s in ("синие", "красные")):
            print(f"  {role:8s} " + ",  ".join(cells))

    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
