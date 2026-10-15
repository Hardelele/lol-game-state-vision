"""Готовность заклинаний D/F по HUD (#806): поиск панели умений и решение по слоту.

Без роликов: эталон панели — вырез своего HUD (dataset/hud_spells/
panel_template.png, кадр своей игры), на нём слоты D (Призрак) и F (Скачок)
готовы. Перезарядка, крестик запрета и серая иконка рисуются поверх
готовой иконки так, как их рисует игра (замеры — docs/state-api.md).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lolvision import hud_spells as hs

TPL = cv2.cvtColor(cv2.imread(str(hs.TEMPLATE)), cv2.COLOR_BGR2RGB)


def frame_with_panel(scale: float, x: int, y: int, size=(1920, 1080), seed=0) -> np.ndarray:
    """Кадр: шум сцены и панель своего HUD в масштабе `scale` в точке (x, y)."""
    rng = np.random.default_rng(seed)
    w, h = size
    fr = rng.integers(20, 120, (h, w, 3), dtype=np.uint8)
    fr = cv2.GaussianBlur(fr, (0, 0), 3)
    p = cv2.resize(TPL, (round(TPL.shape[1] * scale), round(TPL.shape[0] * scale)),
                   interpolation=cv2.INTER_LINEAR)
    ph, pw = p.shape[:2]
    ph, pw = min(ph, h - y), min(pw, w - x)
    fr[y:y + ph, x:x + pw] = p[:ph, :pw]
    return fr


@pytest.mark.parametrize("scale", [1.0, 1.24])
def test_locate_panel_finds_scale_and_place(scale):
    x, y = 960 - round(170 * scale), 1080 - round(105 * scale)
    fr = frame_with_panel(scale, x, y)
    p = hs.locate_panel(fr)
    assert p is not None and p.score > 0.9
    assert abs(p.scale - scale) <= 0.021
    assert abs(p.x - x) <= 2 and abs(p.y - y) <= 2
    assert hs.panel_score(fr, p) > 0.9
    # без панели совпадение низкое: панели «нет на кадре»
    empty = frame_with_panel(scale, x, y, seed=1)
    empty[900:] = 40
    assert hs.panel_score(empty, p) < hs.PRESENT_MIN


def test_spell_hud_needs_agreeing_finds():
    fr = frame_with_panel(1.0, 790, 975)
    hud = hs.SpellHud()
    for _ in range(hs.AGREE - 1):
        assert hud.offer(fr) is None
    panel = hud.offer(fr)
    assert panel is not None and abs(panel.scale - 1.0) < 0.03
    reads = hud.read(fr, 0.0)
    assert set(reads) == {"D", "F"}


PANEL = hs.Panel(0.0, 0.0, 1.0, 1.0)


def crop(key: str) -> np.ndarray:
    return hs.slot_crop(TPL, PANEL, key)


def cooldown(c: np.ndarray, digits: str = "45", navy: bool = True) -> np.ndarray:
    """Перезарядка: иконка темнеет, сектор «осталось» — тёмно-синий, по
    центру белые цифры с тёмной обводкой."""
    out = (c.astype(np.float32) * 0.45).astype(np.uint8)
    if navy:
        out[:, :] = (np.array([22, 52, 112]) * 0.6 + out * 0.4).astype(np.uint8)
    n = out.shape[0]
    cv2.putText(out, digits, (2, n // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(out, digits, (2, n // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def red_cross(c: np.ndarray) -> np.ndarray:
    out = (c.astype(np.float32) * 0.5).astype(np.uint8)
    n = out.shape[0]
    cv2.line(out, (4, 4), (n - 5, n - 5), (230, 20, 20), 3)
    cv2.line(out, (n - 5, 4), (4, n - 5), (230, 20, 20), 3)
    return out


def learned(key: str) -> hs.SlotRef:
    ref = hs.SlotRef()
    for k in range(hs.REF_MIN):
        r = hs.decide(key, crop(key), ref, True, False, t=k * hs.REF_EVERY)
    assert ref.ready and r.state == "ready" and r.value is True
    return ref


@pytest.mark.parametrize("key", ["D", "F"])
def test_ready_only_against_learned_reference(key):
    ref = hs.SlotRef()
    first = hs.decide(key, crop(key), ref, True, False, t=0.0)
    assert first.state == "unsure" and first.value is None     # эталона ещё нет
    ref = learned(key)
    assert hs.decide(key, crop(key), ref, True, False, t=100.0).value is True


def test_ghost_white_figure_is_not_cooldown():
    """Призрак (D своего HUD): белая фигура с обводкой даёт признак цифр
    выше порога (#804) — но и у эталона тоже, поэтому «готово»."""
    assert hs.slot_features(crop("D")).digits >= hs.CD_DIGITS
    ref = learned("D")
    r = hs.decide("D", crop("D"), ref, True, False, t=50.0)
    assert r.state == "ready", r


@pytest.mark.parametrize("key", ["D", "F"])
@pytest.mark.parametrize("navy", [True, False])
def test_cooldown_navy_or_dark_with_digits(key, navy):
    ref = learned(key)
    r = hs.decide(key, cooldown(crop(key), navy=navy), ref, True, False, t=60.0)
    assert r.state == "cooldown" and r.value is False
    # перезарядка видна и без эталона (синяя заливка)
    if navy:
        assert hs.decide(key, cooldown(crop(key)), hs.SlotRef(), True, False, 0.0).value is False


def test_red_cross_is_blocked_and_never_learned():
    ref = learned("F")
    n = len(ref.crops)
    r = hs.decide("F", red_cross(crop("F")), ref, True, False, t=80.0)
    assert r.state == "blocked" and r.value is None and "крестик" in r.reason
    assert len(ref.crops) == n


def test_dead_and_hidden_are_null_with_reason():
    ref = learned("F")
    grey = cv2.cvtColor(cv2.cvtColor(crop("F"), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    r = hs.decide("F", (grey * 0.6).astype(np.uint8), ref, True, True, t=90.0)
    assert r.state == "dead" and r.value is None
    # мёртвый, а перезарядка синяя — это перезарядка
    assert hs.decide("F", cooldown(crop("F")), ref, True, True, 91.0).value is False
    r = hs.decide("F", crop("F"), ref, False, False, t=92.0)
    assert r.state == "hidden" and r.value is None


def test_grey_icon_without_digits_is_not_ready():
    """Серая иконка при полном ХП (все умения серые) — не «готово»."""
    ref = learned("F")
    grey = cv2.cvtColor(cv2.cvtColor(crop("F"), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    r = hs.decide("F", (grey * 0.55).astype(np.uint8), ref, True, False, t=95.0)
    assert r.value is None


def test_spells_change_resets_reference():
    hud = hs.SpellHud()
    hud.refs["D"] = learned("D")
    hud.refs["F"] = learned("F")
    hud.set_spells(("SummonerHaste", "SummonerFlash"))
    hud.set_spells(("SummonerHaste", "SummonerFlash"))
    assert hud.refs["D"].ready and hud.refs["F"].ready
    hud.set_spells(("SummonerTeleportUpgrade", "SummonerFlash"))
    assert not hud.refs["D"].ready and hud.refs["F"].ready


# ---------- умения Q/W/E/R: эталон — иконка Data Dragon (#849) ----------
# Вырезы Q с HUD своих игр (10 к/с; источник и время — fixtures/hud_q/
# cases.json) и иконки умений Data Dragon 16.20.1. Готовая Q Волибира сама
# синяя (синих 0.44 при яркости 113): без иконки DD она читалась как
# перезарядка; Q Ашки (синих 0.29, яркость 124) эталоном не становилась.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

HUD_Q = Path(__file__).parent / "fixtures" / "hud_q"
CASES = json.loads((HUD_Q / "cases.json").read_text(encoding="utf-8"))


def _rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _q_ref(champ: str) -> hs.SlotRef:
    ref = hs.SlotRef()
    ref.set_icon(_rgb(HUD_Q / CASES["icons"][champ]))
    return ref


def _case(champ: str, expect: str) -> dict:
    return next(c for c in CASES["crops"] if c["champion"] == champ and c["file"].endswith(
        f"_{expect}.png"))


@pytest.mark.parametrize("c", CASES["crops"], ids=lambda c: c["file"][:-4])
def test_q_state_against_ddragon_icon(c):
    r = hs.decide("Q", _rgb(HUD_Q / c["file"]), _q_ref(c["champion"]), True, c["dead"],
                  c["t"], ring=c["ring"])
    assert r.state == c["expect"], r
    assert r.value is {"ready": True, "cooldown": False, "active": False}.get(c["expect"])


@pytest.mark.parametrize("champ", ["Volibear", "Ashe"])
def test_ready_q_was_misread_without_icon(champ):
    """Тот же вырез «готова» штатным путём (набранный эталон): не «готова» —
    у Волибира перезарядка, у Ашки эталон не набирается вовсе."""
    crop = _rgb(HUD_Q / _case(champ, "ready")["file"])
    ref = hs.SlotRef()
    reads = [hs.decide("Q", crop, ref, True, False, t=k * hs.REF_EVERY) for k in range(12)]
    assert all(r.value is not True for r in reads)
    if champ == "Volibear":
        assert reads[-1].state == "cooldown"
    assert not ref.ready


def test_active_needs_yellow_ring():
    """Активная Q и Q, погасшая под контролем, по вырезу одинаковы (яркость
    44, насыщенность 0.21) — различает только жёлтая рамка."""
    ref = _q_ref("Volibear")
    act = _case("Volibear", "active")
    crop = _rgb(HUD_Q / act["file"])
    assert hs.decide("Q", crop, ref, True, False, 1.0, ring=act["ring"]).state == "active"
    assert hs.decide("Q", crop, ref, True, False, 1.0, ring=0.0).state == "unsure"


def test_ring_and_cross_and_death_keep_old_rules():
    """Крестик сильнее иконки DD; у мёртвого готовое — null."""
    ref = _q_ref("Volibear")
    ready = _rgb(HUD_Q / _case("Volibear", "ready")["file"])
    assert hs.decide("Q", red_cross(ready), ref, True, False, 0.0).state == "blocked"
    assert hs.decide("Q", ready, ref, True, True, 0.0).state == "dead"
    assert hs.decide("Q", ready, ref, False, False, 0.0).state == "hidden"


def test_slot_ring_reads_yellow_frame():
    fr = frame_with_panel(1.0, 735, 975)
    p = hs.Panel(735.0, 975.0, 1.0, 1.0)
    assert hs.slot_ring(fr, p, "Q") < hs.ACTIVE_RING
    x0, y0, x1, y1 = (round(v) for v in p.box("Q"))
    cv2.rectangle(fr, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), (250, 200, 40), 2)
    assert hs.slot_ring(fr, p, "Q") >= hs.ACTIVE_RING


def test_set_champion_icons_only_for_abilities():
    hud = hs.SpellHud()
    icon = _rgb(HUD_Q / CASES["icons"]["Volibear"])
    hud.set_champion("Volibear", {k: icon for k in hs.ABILITY_SLOTS})
    assert all(hud.refs[k].icon is not None for k in hs.ABILITY_SLOTS)
    assert hud.refs["D"].icon is None and hud.refs["F"].icon is None
    hud.set_panel(hs.Panel(0, 0, 1.0, 1.0))           # новая панель: иконки остаются
    assert hud.refs["Q"].icon is not None
    hud.set_champion(None, None)
    assert all(hud.refs[k].icon is None for k in hs.ABILITY_SLOTS)


def test_ability_icons_from_ddragon_cache(tmp_path):
    from lolvision import ddragon
    v = "16.20.1"
    (tmp_path / v / "champion").mkdir(parents=True)
    (tmp_path / v / "img" / "spell").mkdir(parents=True)
    names = ["VolibearQ.png", "VolibearW.png", "VolibearE.png", "VolibearR.png"]
    (tmp_path / v / "champion" / "Volibear.json").write_text(json.dumps(
        {"data": {"Volibear": {"spells": [{"image": {"full": n}} for n in names]}}}))
    for n in names:
        (tmp_path / v / "img" / "spell" / n).write_bytes((HUD_Q / "VolibearQ.png").read_bytes())
    icons = hs.ability_icons(ddragon.DDragon(v, tmp_path), "Volibear")
    assert set(icons) == set(hs.ABILITY_SLOTS) and icons["Q"].shape == (64, 64, 3)


def test_presses_ready_to_active_and_active_after_cooldown():
    seq = [(0.0, "ready"), (0.1, "ready"), (0.2, "active"), (0.3, "active"), (0.4, "unsure"),
           (0.5, "active"), (1.0, "cooldown"),
           # нажата в первый же кадр готовности: перезарядка → активна
           (5.0, "cooldown"), (5.1, "active"), (5.2, "active"), (5.3, "active"), (5.5, "active"),
           (6.0, "cooldown"),
           # мигание рамки на конце действия (1–2 кадра) — не нажатие
           (9.0, "unsure"), (9.1, "active"), (9.2, "cooldown"),
           # готова → перезарядка (W/E/R)
           (12.0, "ready"), (12.1, "unsure"), (12.2, "cooldown")]
    assert hs.presses(seq) == [0.2, 5.1, 12.1]
