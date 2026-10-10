"""Разведка рендера Replay API для второй модели (#755, этап 0).

Зачем. План второй модели (docs/entity-model-plan.md, путь A) строит
разметку юнитов как разность рендеров одного момента: полный кадр против
кадра без плашек, без чемпионов, без миньонов… Здесь проверяется, на чём
этот план стоит: что именно выключает каждый флаг /replay/render, сколько
шума между двумя одинаковыми снимками на паузе, сколько ждать после смены
флагов и сколько стоит один снимок.

Команды:
    scout  — найти на секунде реплея точки камеры, где на экране больше
             всего чемпионов (по screenPosition* из /liveclientdata/playerlist);
    flags  — в одной точке снять полный кадр, его повтор и варианты с
             выключенными флагами (PNG без потерь) + состав и позиции;
    settle — сколько кадров после смены флага картинка ещё меняется.

Требования те же, что у replay_capture.py: открыт реплей текущего патча,
окно игры 1920×1080 на переднем плане.

Пример:
    python tools/replay_render.py scout --time 2100
    python tools/replay_render.py flags --time 2100 --x 7000 --z 7400 --name tf1
    # → data/replay/render/<name>/full.png, full2.png, noBars.png, … и meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from PIL import Image, ImageGrab

import _bootstrap  # noqa: F401
from lolvision.paths import DATA
from replay_capture import call, focus, game_window, ground_point

OUT = DATA / "replay" / "render"
# Экранная точка вне кадра — FLT_MAX.
OFFSCREEN = 1e30

BARS = {"healthBarChampions": False, "healthBarMinions": False, "healthBarPets": False,
        "healthBarStructures": False, "healthBarWards": False, "floatingText": False}
# Варианты рендера: имя → флаги поверх полного. Каждый меняет одну вещь,
# кроме noBars (все плашки сразу) — так разности читаются однозначно.
VARIANTS = {
    "full2": {},
    "clean": {"interfaceAll": False},
    "noBars": BARS,
    "noChamp": {"champions": False},
    "noMinion": {"minions": False},
    "noChar": {"characters": False},
    "noParticles": {"particles": False},
    "noFog": {"fogOfWar": False},
    "noEnv": {"environment": False},
    "noOutline": {"outlineSelect": False, "outlineHover": False},
    "noTextOnly": {"floatingText": False},
    "noChampBars": {"healthBarChampions": False},
    "noMinionBars": {"healthBarMinions": False},
}


def players() -> list[dict]:
    """Состав и экранные точки чемпионов (None — вне кадра или не виден)."""
    out = []
    for p in call("/liveclientdata/playerlist"):
        pos = {}
        for k in ("screenPositionBottom", "screenPositionCenter"):
            x, y = (float(v) for v in p[k].split(","))
            pos[k] = None if abs(x) > OFFSCREEN else [round(x, 1), round(y, 1)]
        out.append({"champion": p["rawChampionName"].removeprefix("game_character_displayname_"),
                    "name": p["riotIdGameName"], "team": p["team"], "level": p["level"],
                    "skinID": p["skinID"], "skinName": p["skinName"], "isDead": p["isDead"],
                    "position": p["position"], **pos})
    return out


def on_screen(pl: list[dict]) -> list[dict]:
    return [p for p in pl if p["screenPositionBottom"]
            and 0 <= p["screenPositionBottom"][0] < 1920
            and 0 <= p["screenPositionBottom"][1] < 1080]


def pause_at(sec: float) -> float:
    call("/replay/sequence", {})
    call("/replay/playback", {"time": sec, "paused": True})
    time.sleep(2.0)
    return call("/replay/playback")["time"]


def move_to(t0: float, x: float, z: float) -> dict:
    """Камера мгновенно (ключ snap) туда, где под центром экрана — (x, z)."""
    base = call("/replay/render")
    gx, gz = ground_point(base)
    bp = base["cameraPosition"]
    pos = {"x": bp["x"] + x - gx, "y": bp["y"], "z": bp["z"] + z - gz}
    call("/replay/sequence", {"cameraPosition": [
        {"time": t0 - 1, "value": pos, "blend": "snap"},
        {"time": t0 + 1000, "value": pos, "blend": "snap"}]})
    time.sleep(0.4)
    return call("/replay/render")


def park_cursor() -> None:
    """Курсор над полем подсвечивает юнит под ним (контур, плашка) — убрать
    его в угол панели реплея, где он ничего не подсвечивает."""
    import ctypes
    ctypes.windll.user32.SetCursorPos(1915, 5)


def camera_axes(rot: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вперёд, вправо, вверх для поворота API (x — рыскание, y — тангаж вниз)."""
    yaw, p = math.radians(rot["x"]), math.radians(rot["y"])
    f = np.array([math.sin(yaw) * math.cos(p), -math.sin(p), math.cos(yaw) * math.cos(p)])
    r = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    u = np.cross(f, r)
    return f, r, u


def ray(render: dict, sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
    """Луч из глаза через пиксель экрана 1920×1080; fieldOfView — по вертикали
    (проверено: экранные точки playerlist ложатся на эту модель с точностью ~1 ед.)."""
    f, r, u = camera_axes(render["cameraRotation"])
    fpx = 540 / math.tan(math.radians(render["fieldOfView"]) / 2)
    d = f + (sx - 960) / fpx * r - (sy - 540) / fpx * u
    p = np.array([render["cameraPosition"][k] for k in "xyz"])
    return p, d / np.linalg.norm(d)


def triangulate(a: tuple, b: tuple) -> np.ndarray:
    """Ближайшая точка двух лучей (p, d) — положение юнита в мире (x, высота, z)."""
    (p1, d1), (p2, d2) = a, b
    A = np.stack([d1, -d2], 1)
    t, *_ = np.linalg.lstsq(A, p2 - p1, rcond=None)
    return (p1 + t[0] * d1 + p2 + t[1] * d2) / 2


def world_positions(t0: float, x: float, z: float, render: dict, shift: float = 400.0) -> dict:
    """Мировые положения чемпионов по двум положениям камеры (вторая сдвинута
    на `shift` по z). Камера возвращается в (x, z)."""
    first = {p["name"]: p["screenPositionBottom"] for p in players()}
    r2 = move_to(t0, x, z + shift)
    time.sleep(0.3)
    second = {p["name"]: p["screenPositionBottom"] for p in players()}
    move_to(t0, x, z)
    out = {}
    for name, a in first.items():
        b = second.get(name)
        if a and b:
            w = triangulate(ray(render, *a), ray(r2, *b))
            out[name] = [round(float(v), 1) for v in w]
    return out


def shot(h: int) -> np.ndarray:
    focus(h)
    park_cursor()
    img = ImageGrab.grab()
    if img.size != (1920, 1080):
        raise SystemExit(f"экран {img.size}, нужен 1920×1080")
    return np.asarray(img)


def cmd_scout(args) -> None:
    t0 = pause_at(args.time)
    found = []
    vals = np.linspace(1000, 13800, args.n)
    for z in vals:
        for x in vals:
            move_to(t0, x, z)
            time.sleep(0.2)
            vis = on_screen(players())
            if vis:
                found.append((len(vis), round(x), round(z), [p["champion"] for p in vis]))
    call("/replay/sequence", {})
    for row in sorted(found, reverse=True)[:12]:
        print(*row)


def cmd_settle(args) -> None:
    """После смены флага снимаем подряд и смотрим, когда кадр перестаёт меняться."""
    h = game_window()
    t0 = pause_at(args.time)
    move_to(t0, args.x, args.z)
    time.sleep(1.0)
    flag = {args.flag: False}
    for state in (flag, {args.flag: True}):
        ref = shot(h).astype(np.int16)
        t = time.time()
        call("/replay/render", state)
        prev = ref
        for _ in range(12):
            cur = shot(h).astype(np.int16)
            print(f"{args.flag}={state[args.flag]} +{time.time() - t:.2f}s "
                  f"от исходного {(np.abs(cur - ref).max(2) > 24).mean() * 100:.2f}% "
                  f"от предыдущего {(np.abs(cur - prev).max(2) > 24).mean() * 100:.2f}%")
            prev = cur
    call("/replay/sequence", {})


def cmd_flags(args) -> None:
    h = game_window()
    t0 = pause_at(args.time)
    render = move_to(t0, args.x, args.z)
    out = OUT / (args.name or f"t{int(args.time)}_{int(args.x)}_{int(args.z)}")
    out.mkdir(parents=True, exist_ok=True)
    time.sleep(1.0)
    base_flags = {k: v for k, v in call("/replay/render").items() if isinstance(v, bool)}
    timing = {}
    t = time.time()
    Image.fromarray(shot(h)).save(out / "full.png")
    timing["full"] = round(time.time() - t, 3)
    touched = sorted({k for f in VARIANTS.values() for k in f})
    for name, flags in VARIANTS.items():
        t = time.time()
        # Каждый вариант — полное состояние: прежние флаги не нужно откатывать
        # отдельным запросом, это вдвое меньше ожиданий.
        call("/replay/render", {k: flags.get(k, base_flags[k]) for k in touched})
        time.sleep(args.wait)
        Image.fromarray(shot(h)).save(out / f"{name}.png")
        timing[name] = round(time.time() - t, 3)
    call("/replay/render", {k: base_flags[k] for k in touched})
    time.sleep(args.wait)
    pl = players()
    world = world_positions(t0, args.x, args.z, render)
    for p in pl:
        p["world"] = world.get(p["name"])
    call("/replay/sequence", {})
    meta = {"time": t0, "target": [args.x, args.z], "ground": ground_point(render),
            "camera": render["cameraPosition"], "rotation": render["cameraRotation"],
            "fov": render["fieldOfView"], "wait": args.wait, "timing_s": timing,
            "variants": VARIANTS, "players": pl, "on_screen": on_screen(pl)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"→ {out}; на экране: {[p['champion'] for p in meta['on_screen']]}; время {timing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scout", help="где на секунде больше всего чемпионов на экране")
    s.add_argument("--time", type=float, required=True)
    s.add_argument("--n", type=int, default=9, help="сетка n×n по полю")
    for name, help_ in (("flags", "снимки с выключенными флагами"),
                        ("settle", "сколько ждать после смены флага")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--time", type=float, required=True)
        p.add_argument("--x", type=float, required=True)
        p.add_argument("--z", type=float, required=True)
        if name == "flags":
            p.add_argument("--wait", type=float, default=0.35, help="пауза после смены флагов, с")
            p.add_argument("--name")
        else:
            p.add_argument("--flag", default="champions")
    args = ap.parse_args()
    {"scout": cmd_scout, "flags": cmd_flags, "settle": cmd_settle}[args.cmd](args)


if __name__ == "__main__":
    main()
