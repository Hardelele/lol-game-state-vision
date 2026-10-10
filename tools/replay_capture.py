"""Эталон камеры из Replay API клиента League: кадр + точные координаты.

Зачем. Метка с миникарты округлена до пикселя, обрезается краем карты и
сдвинута относительно точки, на которую смотрит камера (#753). Клиент при
просмотре реплея сам знает, где камера: Replay API (game.cfg, [General]
EnableReplayApi=1) отдаёт и принимает её положение. Ставим камеру куда
нужно — в том числе к краям карты — снимаем экран и записываем, где она на
самом деле. По таким кадрам меряется настоящая ошибка метки и модели.

Как. Реплей ставится на паузу в момент `--time`; камера по очереди
переносится (ключом /replay/sequence) в точки сетки (равномерно по карте, с краями), после каждого
переноса ждём, пока положение перестанет меняться, и снимаем экран. API
отдаёт положение глаза камеры (cameraPosition) и поворот; точка на земле,
куда смотрит центр экрана, считается здесь же (луч до плоскости y = 0).

Требования: открыт реплей текущего патча, окно игры на основном мониторе
и на переднем плане, 1920×1080.

Пример:
    python tools/replay_capture.py probe
    python tools/replay_capture.py grid --time 600 --n 7
    python tools/replay_capture.py random --times 300 900 1500 2100 --per-time 75
    # → data/replay/<gameId>/000.jpg … и frames.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import ssl
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from lolvision.paths import DATA

API = "https://127.0.0.1:2999"
OUT = DATA / "replay"
# Клиент слушает только localhost с самоподписанным сертификатом Riot.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

# Игровое поле Summoner's Rift в единицах игры: x и z от 0 до ~14 800.
MAP_LO, MAP_HI = 0.0, 14800.0


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_CTX, timeout=5) as r:
        return json.loads(r.read() or b"{}")


def ground_point(render: dict) -> tuple[float, float]:
    """Точка земли (y = 0) под центром экрана: луч из глаза камеры.

    Поворот в API — углы Эйлера в градусах: x — рыскание, y — тангаж
    (вниз положительный). Если тангажа нет (камера не смотрит вниз),
    возвращается положение глаза."""
    p, r = render["cameraPosition"], render["cameraRotation"]
    yaw, pitch = math.radians(r["x"]), math.radians(r["y"])
    if pitch <= 0.01:
        return p["x"], p["z"]
    # Направление взгляда: горизонтальная составляющая по рысканию.
    d = p["y"] / math.tan(pitch)
    return p["x"] + d * math.sin(yaw), p["z"] + d * math.cos(yaw)


def settle(timeout: float = 4.0, eps: float = 0.5) -> dict:
    """Ждать, пока камера доедет (в клиенте есть сглаживание прокрутки)."""
    last, t0 = None, time.time()
    while time.time() - t0 < timeout:
        r = call("/replay/render")
        p = r["cameraPosition"]
        if last and all(abs(p[k] - last[k]) < eps for k in "xyz"):
            return r
        last = p
        time.sleep(0.15)
    return call("/replay/render")


def game_window() -> int:
    import ctypes
    h = ctypes.windll.user32.FindWindowW("RiotWindowClass", None)
    if not h:
        raise SystemExit("окно игры (RiotWindowClass) не найдено — реплей открыт?")
    return h


def focus(h: int) -> None:
    """Окно игры на передний план. Windows не отдаёт фокус фоновому процессу,
    поэтому сначала «нажимаем» Alt — после этого SetForegroundWindow проходит."""
    import ctypes
    u = ctypes.windll.user32
    if u.GetForegroundWindow() == h:
        return
    u.keybd_event(0x12, 0, 0, 0)
    u.keybd_event(0x12, 0, 2, 0)
    u.ShowWindow(h, 9)                 # SW_RESTORE
    u.SetForegroundWindow(h)
    time.sleep(0.5)


def grab(path: Path, h: int) -> None:
    """Снимок экрана, только если сверху окно игры: иначе в кадре чужие окна."""
    import ctypes
    from PIL import ImageGrab
    focus(h)
    if ctypes.windll.user32.GetForegroundWindow() != h:
        raise SystemExit("не удалось вывести окно игры вперёд — переключитесь на него")
    img = ImageGrab.grab()
    if img.size != (1920, 1080):
        raise SystemExit(f"экран {img.size}, нужен 1920×1080")
    img.save(path, quality=95)


def cmd_probe(_args) -> None:
    for p in ("/replay/game", "/replay/playback", "/replay/render"):
        try:
            print(p, json.dumps(call(p), ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001 — показать, что именно не так
            print(p, "→", e)


def grid_points(n: int, margin: float) -> list[tuple[float, float]]:
    """Сетка n×n по полю, крайние ряды на `margin` за краем: проверка обрезки
    рамки краем миникарты. Змейкой, чтобы камера не прыгала через всю карту."""
    lo, hi = MAP_LO - margin, MAP_HI + margin
    vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    out = []
    for j, z in enumerate(vals):
        row = [(x, z) for x in vals]
        out += row if j % 2 == 0 else row[::-1]
    return out


def capture(name: str, plan: list[tuple[float, float, float]], wait: float) -> None:
    """Снять кадры по плану [(секунда реплея, x, z)]: пауза, камера в точку,
    снимок, запись положения. Точки одной секунды идут подряд."""
    game = call("/replay/game")
    hwnd = game_window()
    focus(hwnd)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    rows, cur, base, t0 = [], None, None, 0.0
    for n, (sec, x, z) in enumerate(plan):
        if sec != cur:
            call("/replay/sequence", {})
            call("/replay/playback", {"time": sec, "paused": True})
            time.sleep(2.0)
            t0 = call("/replay/playback")["time"]
            base, cur = call("/replay/render"), sec
        # Переносим глаз камеры так, чтобы под центром экрана оказалась (x, z):
        # сдвиг между глазом и точкой на земле постоянен при неизменном повороте.
        gx, gz = ground_point(base)
        bp = base["cameraPosition"]
        # /replay/render двигает камеру плавно (cameraMoveSpeed, ~500 ед./с);
        # ключ последовательности с blend=snap ставит её сразу.
        pos = {"x": bp["x"] + x - gx, "y": bp["y"], "z": bp["z"] + z - gz}
        call("/replay/sequence", {"cameraPosition": [
            {"time": t0 - 1, "value": pos, "blend": "snap"},
            {"time": t0 + 1000, "value": pos, "blend": "snap"}]})
        r = settle()
        time.sleep(wait)
        f = out / f"{n:03d}.jpg"
        grab(f, hwnd)
        gx2, gz2 = ground_point(r)
        p, rot = r["cameraPosition"], r["cameraRotation"]
        rows.append({"n": n, "file": f.name, "time": round(t0, 2),
                     "target_x": round(x, 1), "target_z": round(z, 1),
                     "ground_x": round(gx2, 1), "ground_z": round(gz2, 1),
                     "cam_x": round(p["x"], 1), "cam_y": round(p["y"], 1),
                     "cam_z": round(p["z"], 1), "rot_x": round(rot["x"], 3),
                     "rot_y": round(rot["y"], 3), "rot_z": round(rot["z"], 3),
                     "fov": r.get("fieldOfView", "")})
        print(f"{n:03d} t={t0:.0f} цель ({x:.0f}, {z:.0f}) → земля ({gx2:.0f}, {gz2:.0f})",
              flush=True)
    call("/replay/sequence", {})          # вернуть камеру пользователю
    with (out / "frames.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "game.json").write_text(json.dumps({"game": game, "render_base": base}, indent=1),
                                   encoding="utf-8")
    print(f"кадров {len(rows)} → {out}")


def cmd_grid(args) -> None:
    capture(args.name or "grid", [(args.time, x, z) for x, z in
                                  grid_points(args.n, args.margin)], args.wait)


def cmd_random(args) -> None:
    """Случайные точки внутри поля в нескольких моментах матча: так рамка
    встречается и в середине карты, и у краёв, как в настоящих роликах."""
    import random
    rng = random.Random(args.seed)
    plan = [(sec, rng.uniform(MAP_LO, MAP_HI), rng.uniform(MAP_LO, MAP_HI))
            for sec in args.times for _ in range(args.per_time)]
    capture(args.name or "random", plan, args.wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="что отдаёт API сейчас")
    g = sub.add_parser("grid", help="камера по сетке на паузе, кадр + координаты")
    g.add_argument("--time", type=float, default=600.0, help="секунда реплея")
    g.add_argument("--n", type=int, default=7, help="сетка n×n")
    g.add_argument("--margin", type=float, default=600.0,
                   help="на сколько крайние ряды выходят за поле, ед.")
    g.add_argument("--wait", type=float, default=0.3, help="пауза перед снимком, с")
    g.add_argument("--name", help="имя папки вместо processID")
    r = sub.add_parser("random", help="случайные точки внутри поля в нескольких моментах")
    r.add_argument("--times", type=float, nargs="+", default=[300, 900, 1500, 2100])
    r.add_argument("--per-time", type=int, default=75)
    r.add_argument("--seed", type=int, default=754)
    r.add_argument("--wait", type=float, default=0.3)
    r.add_argument("--name")
    args = ap.parse_args()
    {"probe": cmd_probe, "grid": cmd_grid, "random": cmd_random}[args.cmd](args)


if __name__ == "__main__":
    main()
