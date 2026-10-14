"""Индекс роликов (#843): календарь патчей, заявки названия, лента раскадровки,
итог по ролику, свой чемпион по иконкам умений, HLS и раскадровка, индекс.

Без сети и без моделей. Итоги — на сохранённых ответах моделей по четырём
настоящим роликам (tests/fixtures/video_index/verdicts.json — числа и
названия, кадров нет: NOTICE.md); иконки умений и HUD — синтетические.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from lolstream import youtube_preview as yp
from lolvision import hud_champion as hc
from lolvision import hud_spells as hs
from lolvision import patch_calendar as pc
from lolvision import video_meta as vm
from lolvision import video_verdict as vv

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "video_index"
sys.path.insert(0, str(ROOT / "tools"))


# ---------------------------------------------------------------- патчи


def test_calendar_dates_and_numbering():
    assert pc.release("26.20") == dt.date(2026, 10, 7)
    assert pc.release("16.20") == dt.date(2026, 10, 7)      # клиент 16.x = патч 26.x
    assert pc.release("26.1") == dt.date(2026, 1, 8)
    assert pc.at(dt.date(2026, 10, 6)) == "26.19"
    assert pc.at(dt.date(2026, 10, 7)) == "26.20"
    assert pc.at(dt.date(2025, 12, 31)) is None
    assert pc.current(dt.date(2026, 10, 15)) == "26.20"
    days = [pc.SCHEDULE[p] for p in sorted(pc.SCHEDULE)]
    assert days == sorted(days) and len(days) == 24


def test_since_specs():
    today = dt.date(2026, 10, 15)
    assert pc.since("patch", today) == dt.date(2026, 10, 7)
    assert pc.since("season", today) == dt.date(2026, 1, 8)
    assert pc.since("26.18", today) == dt.date(2026, 9, 10)
    assert pc.since("2026-09-01", today) == dt.date(2026, 9, 1)
    with pytest.raises(ValueError):
        pc.since("скоро", today)


def test_patch_from_text_and_upload():
    assert pc.from_text("Volibear Top vs Ornn - KR Master Patch 26.20") == "26.20"
    assert pc.from_text("Jax vs Fiora 10.5 KDA") is None
    assert pc.from_text("12:30 game, client 16.19") == "26.19"
    assert pc.patch_of("no patch", "", dt.date(2026, 9, 30)) == ("26.19", "upload")
    assert pc.patch_of("x", "Patch 26.18 notes", None) == ("26.18", "description")


# ---------------------------------------------------------------- заявки


NAMES = vm.ChampionNames({"Volibear": "Volibear", "Corki": "Corki", "MasterYi": "Master Yi",
                          "KSante": "K'Sante", "LeeSin": "Lee Sin", "Jax": "Jax",
                          "Kaisa": "Kai'Sa"})


def test_claims_from_titles():
    c = vm.claims("Kiin Stream Korean Challenger POV | Volibear vs Corki | Top Lane Proview",
                  "⚔️ Champions in this video: Volibear\n🌍 Server: Korean", NAMES)
    assert (c["rank"], c["role"], c["champion"]) == ("challenger", "top", "Volibear")
    assert c["champions"] == ["Volibear", "Corki"]
    # «Master Yi» — чемпион, а не ранг.
    c = vm.claims("Volibear Jungle vs Master Yi - EUW Patch 26.20", "", NAMES)
    assert c["rank"] is None and c["role"] == "jungle" and c["champion"] == "Volibear"
    assert vm.rank_claim("그마 탑 볼리베어") == "grandmaster"
    assert vm.rank_claim("Rank 1 Jax") == "rank1"
    assert NAMES.find("ksante vs kai'sa") == ["KSante", "Kaisa"]
    assert vm.claims("BEST MOMENTS #shorts", "", NAMES)["not_full"]


# ---------------------------------------------------------------- лента


def _tiles(spec: list[tuple[str, int]], step: float = 10.0):
    out, t = [], 0.0
    for cls, n in spec:
        for _ in range(n):
            out.append((t, cls))
            t += step
    return out


def test_timeline_one_game_with_short_gaps():
    tiles = _tiles([("loading", 6), ("game", 50), ("other", 3), ("tab", 2), ("game", 90),
                    ("client", 10)])
    segs = vv.timeline(tiles, duration=1610)
    assert len(segs) == 1
    s = segs[0]
    assert s.start == 60 and s.end == 1510 and 0.97 < s.game_share < 1.0


def test_timeline_two_games_split_by_client():
    tiles = _tiles([("game", 100), ("client", 12), ("loading", 6), ("game", 120)])
    segs = vv.timeline(tiles, duration=2400)
    assert [(s.start, s.end) for s in segs] == [(0, 1000), (1180, 2380)]


def test_pick_times_early_first_and_capped():
    segs = [vv.Segment(60, 1560, 150, 1.0), vv.Segment(1700, 1800, 10, 1.0)]
    picks = vv.pick_times(segs, max_frames=8)
    ts = [t for _, t in picks]
    assert len(picks) == 8 and all(i == 0 for i, _ in picks)      # короткий не берётся
    assert ts[:4] == [160, 250, 360, 490]
    assert ts == sorted(ts) and max(ts) < 1560


# ---------------------------------------------------------------- итог


def _fixture(vid: str):
    d = json.loads((FIX / "verdicts.json").read_text(encoding="utf-8"))[vid]
    segs = [vv.Segment(s["start"], s["end"], s["tiles"], s["game_share"]) for s in d["segments"]]
    facts = [vv.FrameFacts.from_dict(f) for f in d["facts"]]
    return d, segs, facts


def test_verdict_player_full_game_webcam_over_minimap():
    d, segs, facts = _fixture("6k5-vQ_GQ_M")
    v = vv.verdict(segs, facts, d["claims"], d["champion"], 0.99)
    assert v.usable and v.full_game and v.view == "player"
    assert (v.champion, v.champion_source) == ("Volibear", "hud")
    assert (v.role, v.role_source) == ("top", "claim")          # миникарта под вебкой
    assert v.minimap_share == 0.0 and any("миникарта" in n for n in v.notes)
    assert v.games and abs(v.games[0]["start"] + 50.7) < 1


def test_verdict_cut_video_counts_cut_seconds():
    # Nemesis: вся игра от начала до конца, но с вырезами (~300 с матча).
    d, segs, facts = _fixture("MDS-ZGLCG5k")
    v = vv.verdict(segs, facts, d["claims"], d["champion"], 0.98)
    assert v.full_game and v.cuts >= 2 and 250 < v.cut_s < 350
    assert v.champion == "Lucian" and v.role == "mid" and v.role_source == "camera"
    assert v.confidence < 0.95


def test_coverage_heavy_cut_is_not_full():
    segs = [vv.Segment(0, 1000, 100, 1.0)]
    facts = [vv.FrameFacts(t=t, segment=0, scene="game", view_read="player", clock=c)
             for t, c in ((100, 90), (300, 600), (500, 1100), (900, 1700))]
    v = vv.verdict(segs, facts, {"champion": "Jax"}, None, 1.0)
    assert not v.full_game and any(r.startswith("нарезка") for r in v.reasons)


def test_coverage_real_time_has_no_cuts():
    facts = [vv.FrameFacts(t=t, segment=0, clock=int(t) + 50) for t in (100, 400, 900)]
    c = vv.coverage(facts)
    assert c["cuts"] == 0 and c["cut_s"] == 0 and c["start"] == -50


def test_verdict_spectator_by_storyboard_and_camera():
    d, segs, facts = _fixture("AQjeX5bL644")
    v = vv.verdict(segs, facts, d["claims"], None, 1.0)
    assert v.usable and v.view == "spectator" and v.champion_source == "claim"
    assert (v.role, v.role_source) == ("top", "camera")
    assert any("часы HUD не прочитаны" in n for n in v.notes)


def test_verdict_hud_champion_without_claim():
    d, segs, facts = _fixture("ztO6wuNkO8c")
    v = vv.verdict(segs, facts, d["claims"], d["champion"], 1.0)
    assert v.champion == "Hwei" and v.champion_source == "hud"
    assert v.cuts >= 2                                           # монтаж с комментарием


def test_single_clock_misread_is_not_a_cut():
    segs = [vv.Segment(30, 1530, 150, 1.0)]
    facts = [vv.FrameFacts(t=t, segment=0, scene="game", view_read="player", clock=c)
             for t, c in ((130, 100), (230, 200), (330, 300), (530, 700), (930, 900))]
    v = vv.verdict(segs, facts, {"champion": "Jax"}, None, 1.0)
    assert v.full_game and v.usable and v.cut_s == 0
    assert any("не к месту: 1" in n for n in v.notes)


def test_verdict_not_lol():
    v = vv.verdict([], [], {"champion": None}, None, 0.0)
    assert not v.usable and v.view == "unknown"
    assert any("игры почти нет" in r for r in v.reasons)


def test_role_bot_split_by_tags():
    segs = [vv.Segment(0, 1500, 150, 1.0)]
    facts = [vv.FrameFacts(t=t, segment=0, scene="game", view_read="spectator", zone="bot")
             for t in (100, 200, 300, 450)]
    claims = {"champion": "Thresh"}
    assert vv.verdict(segs, facts, claims, None, 1.0, support_like=True).role == "support"
    assert vv.verdict(segs, facts, claims, None, 1.0, support_like=False).role == "bot"


def test_frame_facts_roundtrip():
    f = vv.FrameFacts(t=1.0, segment=0, scene="game", camera=(0.1, 0.2),
                      champion_top=[("Jax", 0.5)])
    assert vv.FrameFacts.from_dict(json.loads(json.dumps(f.to_dict()))) == f


# ---------------------------------------------------------------- свой чемпион


def _icon(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), (64, 64),
                     interpolation=cv2.INTER_CUBIC)
    return img


def _hud_with(icons: list[np.ndarray], dark: float = 1.0):
    """Эталон своего HUD (кадр своей игры) с иконками в слотах Q/W/E/R."""
    tpl = cv2.cvtColor(cv2.imread(str(hs.TEMPLATE)), cv2.COLOR_BGR2RGB)
    fr = np.full((1080, 1920, 3), 40, np.uint8)
    x0, y0 = hs.TEMPLATE_ORIGIN
    fr[y0:y0 + tpl.shape[0], x0:x0 + tpl.shape[1]] = tpl
    panel = hs.Panel(float(x0), float(y0), 1.0, 1.0)
    for key, ic in zip(hc.SLOTS, icons):
        a, b, c, d = (round(v) for v in panel.box(key))
        fr[b:d, a:c] = (cv2.resize(ic, (c - a, d - b)) * dark).astype(np.uint8)
    return fr, panel


def test_ability_bank_finds_own_champion():
    icons = {f"C{k}": [_icon(10 * k + j) for j in range(4)] for k in range(6)}
    bank = hc.AbilityBank(icons)
    frames = []
    for dark in (1.0, 0.45):                 # вторая — перезарядка: иконки тёмные
        fr, panel = _hud_with(icons["C3"], dark)
        frames.append(bank.scores(fr, panel))
    assert all(bank.ids[int(np.argmax(s))] == "C3" for s in frames)
    d = hc.decide(bank.ids, frames)
    assert d.champion == "C3" and d.margin > hc.MARGIN


def test_decide_refuses_when_close_or_empty():
    ids = ["A", "B"]
    assert hc.decide(ids, []).champion is None
    d = hc.decide(ids, [np.array([0.62, 0.60])])
    assert d.champion is None and "отрыв" in d.reason
    d = hc.decide(ids, [np.array([0.2, 0.0])])
    assert d.champion is None and "ниже" in d.reason


# ---------------------------------------------------------------- сеть без сети


PLAYLIST = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXTINF:6.0,
https://h/seg0
#EXTINF:5.5,
https://h/seg1
#EXTINF:6.0,
https://h/seg2
#EXT-X-ENDLIST
"""


def test_parse_playlist():
    segs = yp.parse_playlist(PLAYLIST)
    assert [(s.t, s.url) for s in segs] == [(0.0, "https://h/seg0"), (6.0, "https://h/seg1"),
                                            (11.5, "https://h/seg2")]


def test_pick_formats():
    fmts = [{"format_id": "sb0", "protocol": "mhtml", "width": 320, "fragments": [{}]},
            {"format_id": "sb1", "protocol": "mhtml", "width": 160, "fragments": [{}]},
            {"format_id": "299", "protocol": "https", "vcodec": "avc1", "height": 1080},
            {"format_id": "311", "protocol": "m3u8_native", "vcodec": "avc1", "height": 720},
            {"format_id": "312", "protocol": "m3u8_native", "vcodec": "avc1", "height": 1080},
            {"format_id": "616", "protocol": "m3u8_native", "vcodec": "vp09", "height": 1080},
            {"format_id": "623", "protocol": "m3u8_native", "vcodec": "vp09", "height": 1440}]
    assert yp.pick_hls(fmts)["format_id"] == "312"
    assert yp.pick_storyboard(fmts)["format_id"] == "sb0"
    assert yp._expire("https://x/videoplayback/expire/1792028437/ei/x") == 1792028437
    assert yp._expire("https://x/videoplayback?expire=17&x=1") == 17


def test_storyboard_tiles(monkeypatch):
    rng = np.random.default_rng(0)
    sheet = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    sheet[360:, 640:] = 0                    # пустая клетка в конце листа
    ok, buf = cv2.imencode(".jpg", sheet)
    monkeypatch.setattr(yp, "_get", lambda url, traffic, limit=None: buf.tobytes())
    info = yp.VideoInfo(id="x", title="", channel=None, channel_id=None, upload_date=None,
                        duration=200.0, description="", live_status=None, width=1920,
                        height=1080, hls=None, fetched=0, seconds=0,
                        storyboard={"rows": 3, "columns": 3, "width": 320, "height": 180,
                                    "fps": 0.1, "fragments": [{"url": "a", "duration": 90},
                                                              {"url": "b", "duration": 90},
                                                              {"url": "c", "duration": 90}]})
    tiles = yp.storyboard(info, yp.Traffic())
    ts = [t.t for t in tiles]
    assert ts[:3] == [0.0, 10.0, 20.0] and 80.0 not in ts       # пустые клетки отброшены
    assert max(ts) < 200 and tiles[0].image.shape == (180, 320, 3)


def test_expired_stream_refuses(monkeypatch):
    info = yp.VideoInfo(id="x", title="", channel=None, channel_id=None, upload_date=None,
                        duration=10.0, description="", live_status=None, width=None,
                        height=None, storyboard=None, fetched=0, seconds=0,
                        hls={"url": "https://m/api/manifest/expire/100/x"})
    s = yp.HlsStream(info, yp.Traffic())
    with pytest.raises(yp.ExpiredError):
        s.segments()


# ---------------------------------------------------------------- индекс


def test_index_put_merge_and_skip(tmp_path):
    vi = pytest.importorskip("video_index")
    idx = vi.Index(tmp_path / "v.sqlite")
    idx.put({"id": "a", "title": "t", "sources": "channel:x", "usable": 1,
             "reasons": ["r"], "schema": vi.SCHEMA, "status": "done"})
    idx.put({"id": "a", "sources": "search:q", "champion": "Jax"})
    r = idx.get("a")
    assert r["sources"] == "channel:x|search:q" and r["title"] == "t" and r["champion"] == "Jax"
    assert json.loads(r["reasons"]) == ["r"]
    since = dt.date(2026, 10, 7)
    assert vi._keep({"duration": 300}, since).startswith("короче")
    assert vi._keep({"duration": 1500, "url": "https://youtube.com/shorts/x"}, since) == "шорт"
    assert vi._keep({"duration": 1500, "approx_date": "2026-08-01"}, since).startswith("старый")
    assert vi._keep({"duration": 1500, "approx_date": "2026-10-01"}, since) is None
    assert vi.channel_url("@X/videos", "streams") == "https://www.youtube.com/@X/streams"


def test_three_games_in_one_segment_split_by_clock():
    # BA7OqRG-QVE: «Varus · Renekton · Jayce», игры подряд без загрузки между ними.
    d, segs, facts = _fixture("BA7OqRG-QVE")
    claims = dict(d["claims"], champions=["Varus", "Renekton", "Jayce"])
    v = vv.verdict(segs, facts, claims, d["champion"], 1.0)
    assert len(v.games) == 3 and v.usable and v.champion == "Varus"
    assert [g["champion_claim"] for g in v.games] == ["Varus", "Renekton", "Jayce"]
    assert v.games[0]["champion"] == "Varus" and v.games[0]["full"]
    assert any("3 игр подряд" in n for n in v.notes)


def test_split_ignores_single_backward_misread():
    fs = [vv.FrameFacts(t=t, segment=0, clock=c)
          for t, c in ((100, 60), (300, 260), (500, 20), (700, 660), (900, 860))]
    assert len(vv.split_games(fs)) == 1
    fs[3:] = [vv.FrameFacts(t=700, segment=0, clock=220), vv.FrameFacts(t=900, segment=0,
                                                                         clock=420)]
    assert [len(p) for p in vv.split_games(fs)] == [2, 3]


def test_cut_archive_is_full_with_cut_seconds():
    d, segs, facts = _fixture("_508S98our4")
    v = vv.verdict(segs, facts, d["claims"], d["champion"], 1.0)
    assert v.usable and v.full_game and v.cuts == 6 and 250 < v.cut_s < 300


def test_rate_bucket_limits_speed():
    b = yp._Bucket(1.0)                      # 1 МБ/с
    t0 = __import__("time").monotonic()
    for _ in range(8):
        b.take(64 * 1024)                    # 0.5 МБ, из них 0.25 МБ — ёмкость ведра
    dt_s = __import__("time").monotonic() - t0
    assert 0.2 < dt_s < 1.0
    b.set(0)
    b.take(10**9)                            # без предела — сразу


def test_title_guide_word_does_not_veto_clock_confirmed_game():
    d, segs, facts = _fixture("6k5-vQ_GQ_M")
    claims = dict(d["claims"], not_full=True)          # «... Gameplay Guide»
    v = vv.verdict(segs, facts, claims, d["champion"], 0.99)
    assert v.usable and any("гайд" in n for n in v.notes)
    d, segs, facts = _fixture("AQjeX5bL644")           # наблюдатель: часов нет
    v = vv.verdict(segs, facts, dict(d["claims"], not_full=True), None, 1.0)
    assert not v.usable and any(r.startswith("название") for r in v.reasons)


def test_role_claim_beats_camera_and_ganks_mean_jungle():
    segs = [vv.Segment(0, 1500, 150, 1.0)]
    zones = ("mid", "top", "top", "top", "river")             # xZE9zQ2SVrc
    facts = [vv.FrameFacts(t=t, segment=0, scene="game", view_read="spectator", zone=z)
             for t, z in zip((108, 196, 308, 437, 558), zones)]
    assert vv.decide_role(facts, None, 0.0)[0] == "jungle"
    v = vv.verdict(segs, facts, {"champion": "Volibear", "role": "top"}, None, 1.0)
    assert (v.role, v.role_source) == ("top", "claim≠camera")
