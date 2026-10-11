"""«Кто это»: Data Dragon, состав по HUD, имя над плашкой, объединение (без кадров роликов)."""

import json

import numpy as np
import pytest

from lolvision import ddragon, identity, name_reader as nr, roster


# --- Data Dragon ---

def test_patch_to_version_maps_year_patch_to_ddragon():
    vs = ["16.20.1", "16.19.1", "16.18.1", "15.24.1"]
    assert ddragon.patch_to_version("26.19", vs) == "16.19.1"
    assert ddragon.patch_to_version("16.18", vs) == "16.18.1"
    assert ddragon.patch_to_version("15.24.1", vs) == "15.24.1"
    with pytest.raises(KeyError):
        ddragon.patch_to_version("26.30", vs)


def test_norm_name():
    assert ddragon.norm_name("Kai'Sa") == "kaisa"
    assert ddragon.norm_name("Dr. Mundo") == "drmundo"
    assert ddragon.norm_name("Nunu & Willump") == "nunuwillump"


def test_cache_and_fetch(tmp_path):
    calls = []
    champs = {"data": {"MonkeyKing": {"id": "MonkeyKing", "key": "62", "name": "Wukong"},
                       "Kaisa": {"id": "Kaisa", "key": "145", "name": "Kai'Sa"}}}

    def fetch(url):
        calls.append(url)
        if url.endswith("versions.json"):
            return json.dumps(["16.19.1", "16.18.1"]).encode()
        if url.endswith("champion.json"):
            return json.dumps(champs).encode()
        return b"png"

    dd = ddragon.load("26.19", root=tmp_path, fetch=fetch)
    assert dd.version == "16.19.1"
    assert dd.by_name("wukong") == "MonkeyKing" and dd.by_name("KAI'SA") == "Kaisa"
    assert dd.icon("Kaisa").read_bytes() == b"png"
    n = len(calls)
    offline = ddragon.DDragon("16.19.1", root=tmp_path)      # второй раз — только кэш
    assert offline.names()["MonkeyKing"] == "Wukong"
    offline.icon("Kaisa")
    assert len(calls) == n
    with pytest.raises(FileNotFoundError):
        offline.icon("Ahri")


# --- состав по портретам HUD ---

def synthetic_icons(n=12, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        base = rng.integers(0, 255, (6, 6, 3)).astype(np.uint8)
        out[f"C{i:02d}"] = np.kron(base, np.ones((20, 20, 1), np.uint8))   # 120×120 блоками
    return out


def test_roster_from_portraits():
    import cv2
    icons = synthetic_icons()
    bank = roster.IconBank(icons)
    truth = [f"C{i:02d}" for i in (3, 1, 7, 0, 9, 2, 11, 5, 4, 8)]
    rgb = np.full((1080, 1920, 3), 30, np.uint8)
    for slot, vs in enumerate(roster.views()):
        for v in vs:
            ic = cv2.resize(icons[truth[slot]], (v.size, v.size), interpolation=cv2.INTER_AREA)
            rgb[v.y + 1:v.y + 1 + v.size, v.x - 1:v.x - 1 + v.size] = ic    # сдвиг в пределах ±3
    f = roster.read_frame(rgb, bank)
    assert f.hud
    r = roster.decide([f, roster.FrameSlots(np.zeros_like(f.scores))], bank.ids)
    assert r.champions == truth and r.frames == 1
    assert r.team("CHAOS") == truth[5:] and r.team_of("C07") == "ORDER"


def test_roster_none_without_hud():
    bank = roster.IconBank(synthetic_icons())
    f = roster.read_frame(np.zeros((1080, 1920, 3), np.uint8), bank)
    assert not f.hud
    assert roster.decide([f], bank.ids) is None


# --- имя над плашкой ---

def as_crop(w):
    """Белизна → вырез RGB, который даёт ту же белизну."""
    v = (110 + 100 * w).clip(0, 255).astype(np.uint8)
    return np.repeat(v[..., None], 3, 2)


def test_name_reader_picks_rendered_name_and_respects_candidates():
    names = {"Volibear": "Volibear", "Irelia": "Irelia", "Hecarim": "Hecarim", "Lulu": "Lulu"}
    reader = nr.NameReader.rendered(names)
    crop = np.roll(as_crop(nr.render("Irelia")), 2, axis=1)              # сдвиг на 2 px
    lab, sim, margin = reader.read(crop)
    assert lab == "Irelia" and sim > 0.9 and margin > 0
    lab, _, _ = reader.read(crop, candidates=["Volibear", "Lulu"], statuses=False)
    assert lab in ("Volibear", "Lulu")
    status = nr.render("STUNNED", px=nr.STATUS_PX, track=nr.STATUS_TRACK, dx=nr.STATUS_DX,
                       top=nr.STATUS_Y)
    lab, _, _ = reader.read(as_crop(status))
    assert lab == nr.STATUS + "STUNNED"


def test_name_reader_save_load(tmp_path):
    reader = nr.NameReader.rendered({"Zed": "Zed", "Zac": "Zac"})
    reader.save(tmp_path / "n.npz")
    back = nr.NameReader.load(tmp_path / "n.npz")
    assert back.labels == reader.labels
    assert back.read(as_crop(nr.render("Zac")))[0] == "Zac"


# --- вырезы и объединение ---

def test_crop_pad_at_frame_edge():
    rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    c = identity.crop_pad(rgb, 0, 0, (-5, -5, 5, 5))
    assert c.shape == (10, 10, 3)
    assert (c[:5] == 0).all() and (c[5:, 5:] == rgb[:5, :5]).all()


def test_name_and_body_crop_shapes():
    rgb = np.zeros((1080, 1920, 3), np.uint8)
    assert identity.name_crop(rgb, 900, 400).shape == (nr.CROP_H, nr.CROP_W, 3)
    assert identity.body_crop(rgb, 1900, 1070).shape == (identity.BODY_OUT, identity.BODY_OUT, 3)


def test_assign_distinct_resolves_conflict():
    s = np.array([[0.9, 0.8, 0.1], [0.95, 0.2, 0.1]])
    assert identity.assign_distinct(s).tolist() == [1, 0]


def test_neighbours_same_team_close_level():
    frame = np.array([0, 1, 1, 2, 5])
    x = np.array([100, 150, 900, 200, 100])
    y = np.array([100, 120, 100, 100, 100])
    team = np.array([0, 0, 0, 1, 0])
    level = np.array([5, 5, 5, 5, 5])
    nb = identity.neighbours(frame, x, y, team, level)
    assert nb[0] == [1]          # 2 — дальше, 3 — другая команда, 4 — не соседний кадр
    assert nb[1] == [0]
    assert nb[4] == []
