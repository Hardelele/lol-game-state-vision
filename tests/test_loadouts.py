"""Паспорт выбора игры и детектор нетипичного (#807): lolstream.loadouts, lolstream.surprises.

Сети и кэша Data Dragon нет: игра собирается из ответов allgamedata в
формате журнала своей игры (ники — «PlayerN#0000», в выход они не должны
попасть), каталог предметов — маленький подставной.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lolstream import surprises
from lolstream.live_client import (JournalRow, LiveClient, LiveStatus, Response, State,
                                   journal_transport, read_journal)
from lolstream.loadouts import (Base, GameLog, ItemCatalog, LoadoutRecorder, build_base,
                                read_records, records_from_journal, write_records)

FIX = Path(__file__).parent / "fixtures" / "live_client"
IDS = ["Zilean", "Amumu", "Yone", "Caitlyn", "Nami", "Darius", "Diana", "Lux", "Twitch", "Yuumi"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
# (чемпион, заклинания, ключевая руна (id, имя), ветки) по составу ORDER, затем CHAOS
LINEUP = [
    ("Zilean", ["SummonerHaste", "SummonerFlash"], (8229, "ArcaneComet"), (8200, 8300)),
    ("Amumu", ["SummonerFlash", "SummonerSmite"], (8010, "Conqueror"), (8000, 8100)),
    ("Yone", ["SummonerDot", "SummonerFlash"], (8021, "FleetFootwork"), (8000, 8400)),
    ("Caitlyn", ["SummonerFlash", "SummonerTeleport"], (8021, "FleetFootwork"), (8000, 8200)),
    ("Nami", ["SummonerFlash", "SummonerHeal"], (8214, "SummonAery"), (8200, 8400)),
    ("Darius", ["SummonerFlash", "SummonerTeleport"], (8010, "Conqueror"), (8000, 8200)),
    ("Diana", ["SummonerFlash", "SummonerDot"], (8010, "Conqueror"), (8000, 8300)),
    ("Lux", ["SummonerFlash", "SummonerHeal"], (8229, "ArcaneComet"), (8200, 8100)),
    ("Twitch", ["SummonerFlash", "SummonerBarrier"], (8008, "LethalTempo"), (8000, 8100)),
    ("Yuumi", ["SummonerExhaust", "SummonerHeal"], (8214, "SummonAery"), (8200, 8400)),
]
# Подставной item.json: стартовые, компонент, ботинки, законченные, поддержка, питомец.
ITEMS = {
    "1056": {"name": "Doran's Ring", "gold": {"total": 400}, "tags": ["Lane"]},
    "2003": {"name": "Health Potion", "gold": {"total": 50}, "tags": ["Consumable"],
             "consumed": True},
    "3340": {"name": "Stealth Ward", "gold": {"total": 0}, "tags": ["Trinket"]},
    "1001": {"name": "Boots", "gold": {"total": 300}, "tags": ["Boots"], "into": ["3020"]},
    "3020": {"name": "Sorcerer's Shoes", "gold": {"total": 1100}, "tags": ["Boots"],
             "depth": 2},
    "1082": {"name": "Dark Seal", "gold": {"total": 350}, "into": ["3041"]},
    "6655": {"name": "Luden's Echo", "gold": {"total": 2750}, "depth": 3},
    "3089": {"name": "Rabadon's Deathcap", "gold": {"total": 3600}, "depth": 3},
    "3865": {"name": "World Atlas", "gold": {"total": 400}},
    "1101": {"name": "Scorchclaw Pup", "gold": {"total": 450}, "tags": ["Jungle"]},
}
CATALOG = ItemCatalog("test", ITEMS)


def spell(raw: str) -> dict:
    return {"displayName": "x", "rawDisplayName": f"GeneratedTip_SummonerSpell_{raw}_DisplayName"}


def player(i: int, items: list[int], spells: list[str] | None = None,
           position: str | None = None) -> dict:
    champ, sp, (kid, kname), (prim, sec) = LINEUP[i]
    sp = spells or sp
    return {"championName": champ, "rawChampionName": f"game_character_displayname_{champ}",
            "riotId": f"Player{i + 1}#0000", "summonerName": f"Player{i + 1}#0000",
            "team": "ORDER" if i < 5 else "CHAOS", "level": 1, "isDead": False,
            "respawnTimer": 0.0, "isBot": False, "skinID": 0,
            "position": ROLES[i % 5] if position is None else position,
            "items": [{"itemID": it, "slot": n, "count": 1} for n, it in enumerate(items)],
            "scores": {"kills": 0, "deaths": 0, "assists": 0, "creepScore": 0, "wardScore": 0.0},
            "summonerSpells": {"summonerSpellOne": spell(sp[0]),
                               "summonerSpellTwo": spell(sp[1])},
            "runes": {"keystone": {"id": kid, "rawDisplayName": f"perk_displayname_{kname}"},
                      "primaryRuneTree": {"id": prim}, "secondaryRuneTree": {"id": sec}}}


def game(t: float, items: dict[int, list[int]] | None = None, events: list[dict] | None = None,
         me: int = 0, mode: str = "CLASSIC", spells: dict[int, list[str]] | None = None,
         position: str | None = None) -> dict:
    items, spells = items or {}, spells or {}
    return {"activePlayer": {"riotId": f"Player{me + 1}#0000", "level": 1, "currentGold": 0.0,
                             "championStats": {}},
            "allPlayers": [player(i, items.get(i, []), spells.get(i), position)
                           for i in range(10)],
            "events": {"Events": events if events is not None else
                       [{"EventID": 0, "EventName": "GameStart", "EventTime": 0.0}]},
            "gameData": {"gameMode": mode, "gameTime": t, "mapNumber": 11}}


START = [{"EventID": 0, "EventName": "GameStart", "EventTime": 0.0}]
END = START + [{"EventID": 9, "EventName": "GameEnd", "EventTime": 1800.0, "Result": "Lose"}]
T0 = 1_791_800_000.0


def rows(mode: str = "CLASSIC", position: str | None = None, me: int = 0) -> list[JournalRow]:
    """Загрузка → старт с покупкой → ботинки → Лудены → улучшение ТП → конец."""
    seq = [
        (0.07, {}, [], {}),
        (5.0, {0: [1056, 2003, 2003, 3340], 4: [3865], 1: [1101]}, START, {}),
        (300.0, {0: [1056, 1001, 3340], 4: [3865], 1: [1101]}, START, {}),
        (600.0, {0: [1056, 3020, 1082, 3340], 4: [3865]}, START,
         {5: ["SummonerFlash", "S12_SummonerTeleportUpgrade"]}),
        (900.0, {0: [6655, 3020, 1082, 3340], 4: [3865]}, START,
         {5: ["SummonerFlash", "S12_SummonerTeleportUpgrade"]}),
        (1800.0, {0: [6655, 3020, 3089, 3340], 4: [3865]}, END,
         {5: ["SummonerFlash", "S12_SummonerTeleportUpgrade"]}),
    ]
    return [JournalRow(T0 + t, Response(200, json.dumps(game(t, it, ev, me, mode, sp, position))))
            for t, it, ev, sp in seq]


def one_game(**kw) -> list[dict]:
    rec = records_from_journal(rows(**kw), IDS, CATALOG)
    assert len(rec.games) == 1
    return rec.games[0]


def by_champ(recs: list[dict]) -> dict:
    return {r["champion"]: r for r in recs}


# ---------------------------------------------------------------- записи

def test_records_all_ten_without_names():
    recs = one_game()
    assert len(recs) == 10
    text = json.dumps(recs, ensure_ascii=False)
    assert "Player" not in text and "riot" not in text            # ников нет нигде
    ch = by_champ(recs)
    z = ch["Zilean"]
    assert (z["relation"], z["team"], z["role"], z["result"]) == ("self", "ORDER", "TOP", "Lose")
    assert (ch["Nami"]["relation"], ch["Lux"]["relation"]) == ("ally", "enemy")
    assert ch["Lux"]["result"] == "Win"                            # врагу — обратный итог
    assert z["complete"] is True and z["mode"] == "CLASSIC" and z["map"] == 11
    assert z["patch"] is None and z["patch_reason"]
    assert z["game_id_source"] == "hash" and len({r["game_id"] for r in recs}) == 1
    assert z["runes"] == {"keystone": {"id": 8229, "name": "ArcaneComet"},
                          "primary": {"id": 8200, "name": "Sorcery"},
                          "secondary": {"id": 8300, "name": "Inspiration"}}
    # порядок строк: ORDER, затем CHAOS, внутри — по ролям
    assert [r["champion"] for r in recs] == [c for c, *_ in LINEUP]


def test_items_start_completed_and_purchases():
    z = by_champ(one_game())["Zilean"]["items"]
    assert z["start"] == [1056, 2003, 2003, 3340] and z["start_reason"] is None
    assert [(c["t"], c["item"], c["kind"]) for c in z["completed"]] == [
        (600.0, 3020, "boots"), (900.0, 6655, "legendary"), (1800.0, 3089, "legendary")]
    assert z["completed"][1]["name"] == "Luden's Echo"
    assert z["final"] == [3020, 3089, 3340, 6655]
    assert {"t": 300.0, "item": 1001} in z["purchases"]


def test_spell_upgrade_is_logged_with_time():
    d = by_champ(one_game())["Darius"]["spells"]
    assert d["families"] == ["SummonerFlash", "SummonerTeleport"]
    assert d["variants"] == ["SummonerFlash", "SummonerTeleport"]
    assert d["final"] == ["SummonerFlash", "SummonerTeleportUpgrade"]
    assert d["upgrades"] == [{"t": 600.0, "slot": 2, "variant": "SummonerTeleportUpgrade"}]


def test_no_catalog_and_late_start_are_reasons():
    rec = records_from_journal(rows()[3:], IDS, None, "нет item.json")
    z = by_champ(rec.games[0])["Zilean"]["items"]
    assert z["completed"] is None and z["completed_reason"] == "нет item.json"
    assert z["start"] is None and "подхвачена с 600" in z["start_reason"]


def test_spectator_has_no_relation_or_result():
    rs = rows()
    for r in rs:                                       # наблюдатель: activePlayer нет
        d = json.loads(r.response.body)
        d["activePlayer"] = {"error": "Spectator mode"}
        object.__setattr__(r, "response", Response(200, json.dumps(d)))
    z = by_champ(records_from_journal(rs, IDS, CATALOG).games[0])["Zilean"]
    assert z["relation"] is None and "Spectator" in z["relation_reason"]
    assert z["result"] is None and z["result_reason"]


def test_own_journal_fixture_without_spells_and_runes():
    # обезличенный журнал своей игры: summonerSpells и runes вырезаны
    rec = records_from_journal(read_journal(FIX / "own_game_end.jsonl"), IDS + ["Aatrox"],
                               CATALOG)
    recs = [r for g in rec.games for r in g]
    assert recs and all(r["spells"] is None and "summonerSpells" in r["spells_reason"]
                        for r in recs)
    assert all(r["runes"] is None and r["runes_reason"] for r in recs)


def test_practice_tool_is_skipped_by_default():
    rec = records_from_journal(rows(mode="PRACTICETOOL"), IDS, CATALOG)
    assert rec.games == [] and "PRACTICETOOL" in rec.skipped[0]
    rec = records_from_journal(rows(mode="PRACTICETOOL"), IDS, CATALOG, skip_modes=())
    assert len(rec.games) == 1


# ---------------------------------------------------------------- файл и живой сеанс

def test_write_records_once_per_game(tmp_path):
    recs = one_game()
    path, n = write_records(recs, tmp_path)
    assert n == 10 and path.name == f"{recs[0]['game_start'][:10]}.jsonl"
    # та же игра из живого сеанса: начало оценено на 30 с иначе — не дублируется
    again = [{**r, "game_start_unix": r["game_start_unix"] + 30, "game_id": "x"} for r in recs]
    assert write_records(again, tmp_path) == (path, 0)
    assert len(read_records([path])) == 10


def statuses(rs: list[JournalRow]) -> list[LiveStatus]:
    from lolstream.live_client import replay
    return [st for st, _ in replay(rs, IDS)]


def test_recorder_writes_on_game_end_and_once(tmp_path):
    rec = LoadoutRecorder(tmp_path, CATALOG)
    sts = statuses(rows())
    out = [rec.observe(st) for st in sts]
    assert sts[-1].state is State.ENDED
    assert out[:-1] == [None] * (len(sts) - 1) and len(out[-1]) == 10
    assert rec.observe(sts[-1]) is None and rec.close() is None   # ended повторяется
    assert [n for _, n in rec.written] == [10]
    assert read_records(tmp_path.glob("*.jsonl"))[0]["complete"] is True


def test_recorder_close_mid_game_and_new_game(tmp_path):
    rec = LoadoutRecorder(tmp_path, CATALOG)
    sts = statuses(rows())
    for st in sts[:4]:
        rec.observe(st)
    # новая игра (время пошло назад) — прошлая пишется недописанной
    done = rec.observe(statuses(rows())[1])
    assert done is not None and done[0]["complete"] is False
    assert done[0]["result"] is None and "GameEnd" in done[0]["result_reason"]
    assert rec.close() is not None                # вторая — по закрытию сеанса


def test_loading_only_is_not_written(tmp_path):
    rec = LoadoutRecorder(tmp_path, CATALOG)
    rec.observe(statuses(rows()[:1])[0])
    assert rec.close() is None and not list(tmp_path.iterdir())


def test_live_client_observer_errors_do_not_stop_polling():
    def bad(_st):
        raise RuntimeError("сломался")
    client = LiveClient(journal_transport(rows()[:2]), on_status=bad)
    assert client.poll_once().state is State.LOADING
    assert client.observer_error == "RuntimeError: сломался"
    assert client.poll_once().state is State.IN_GAME


def test_live_client_feeds_recorder(tmp_path):
    rec = LoadoutRecorder(tmp_path, CATALOG)
    client = LiveClient(journal_transport(rows()), on_status=rec.observe)
    for _ in range(6):
        client.poll_once()
    assert len(rec.games) == 1 and rec.written[0][1] == 10


# ---------------------------------------------------------------- сюрпризы по правилам

def kinds(rec: dict) -> dict[str, dict]:
    return {s["rule"]: s for s in rec["surprises"]}


def test_role_rules_flag_teleport_adc_and_no_flash():
    recs = one_game()
    rules = surprises.load_rules()
    surprises.annotate(recs, rules)
    ch = by_champ(recs)
    cait = kinds(ch["Caitlyn"])
    assert set(cait) == {"bottom_teleport"}                       # без общего «вне списка»
    s = cait["bottom_teleport"]
    assert (s["kind"], s["what"], s["confidence"]) == ("spell", "SummonerTeleport", 0.6)
    assert s["expected"].startswith("BOTTOM:") and "Телепорт" in s["basis"]
    assert "no_flash" in kinds(ch["Yuumi"])
    # лесник с Карой и питомцем, поддержка с предметом — ничего
    assert ch["Amumu"]["surprises"] == [] and ch["Nami"]["surprises"] == []
    # Диана в лесу без Кары и без питомца (в этой игре его купил только Амуму)
    assert {"jungle_without_smite", "jungle_without_pet"} <= set(kinds(ch["Diana"]))
    # поддержка без предмета поддержки — Юми
    assert "utility_without_support_item" in kinds(ch["Yuumi"])
    assert ch["Lux"]["surprises"][0]["rule"] == "spells.roles.MIDDLE"   # Исцеление в центре
    assert all("база частот не применялась" in " ".join(r["surprises_notes"]) for r in recs)
    assert ch["Zilean"]["rules"]["path"] == "surprise_rules.toml"


def test_smite_off_jungle_and_items_off_role():
    rec = {"role": "BOTTOM", "duration": 1800.0,
           "spells": {"families": ["SummonerFlash", "SummonerSmite"]},
           "runes": {"keystone": {"id": 8437, "name": "GraspOfTheUndying"}},
           "items": {"purchases": [{"t": 5.0, "item": 1101}, {"t": 6.0, "item": 3865}],
                     "final": [], "start": None}}
    got = kinds({"surprises": surprises.detect(rec, surprises.load_rules())[0]})
    assert {"smite_off_jungle", "keystones.roles.BOTTOM", "support_item_off_utility",
            "jungle_pet_off_jungle"} <= set(got)
    assert got["smite_off_jungle"]["confidence"] == 0.9


def test_no_role_means_no_role_rules():
    recs = one_game(position="")
    surprises.annotate(recs, surprises.load_rules())
    assert all(r["role"] is None and r["surprises"] == [] for r in recs)
    assert "правила по роли не применялись" in recs[0]["surprises_notes"][0]


def test_rules_file_is_configurable(tmp_path):
    p = tmp_path / "rules.toml"
    p.write_text('version = 1\n[spells.roles.BOTTOM]\nexpected = ["SummonerFlash", '
                 '"SummonerTeleport"]\nother_confidence = 0.2\n', encoding="utf-8")
    recs = one_game()
    surprises.annotate(recs, surprises.load_rules(p))
    ch = by_champ(recs)
    assert ch["Caitlyn"]["surprises"] == []
    assert kinds(ch["Twitch"])["spells.roles.BOTTOM"]["confidence"] == 0.2
    p.write_text("version = 99\n", encoding="utf-8")
    with pytest.raises(ValueError):
        surprises.load_rules(p)


# ---------------------------------------------------------------- база частот

def test_base_roundtrip_and_small_sample_note(tmp_path):
    recs = one_game() + one_game()                # одна игра дважды — считается один раз
    b = build_base(recs, "own", "test")
    assert b.games == 10 and "малая выборка" in b.note
    e = b.entry("Zilean", "TOP")
    assert e["games"] == 1 and e["keystones"]["ArcaneComet"] == 1
    assert e["spells"] == {"SummonerHaste": 1, "SummonerFlash": 1}
    assert e["first_item"] == {"6655": 1}
    b.save(tmp_path / "base.json")
    b2, why = Base.load(tmp_path / "base.json")
    assert why is None and b2.to_dict() == b.to_dict()
    assert Base.load(tmp_path / "нет.json")[0] is None
    (tmp_path / "bad.json").write_text('{"kind": "x"}', encoding="utf-8")
    assert "loadout_base" in Base.load(tmp_path / "bad.json")[1]


def test_base_flags_rare_keystone_and_role_when_sample_is_big_enough(tmp_path):
    hist = []
    for n in range(30):                             # 30 игр Зилеана на верхней с Кометой
        recs = one_game()
        for r in recs:
            r["game_id"] = f"g{n}"
        hist += recs
    b = build_base(hist, "own")
    rec = by_champ(one_game())["Zilean"]
    rec["runes"]["keystone"] = {"id": 8010, "name": "Conqueror"}
    rules = surprises.load_rules()
    got = kinds({"surprises": surprises.detect(rec, rules, b)[0]})
    s = got["base:keystone"]
    assert s["kind"] == "keystone" and s["what"] == "Conqueror"
    assert s["expected"] == "ArcaneComet 100%" and s["confidence"] == 0.5    # 30/(30+30)
    assert "база own" in s["basis"] and "30 игр" in s["basis"]
    rec["role"] = "JUNGLE"
    got = kinds({"surprises": surprises.detect(rec, rules, b)[0]})
    assert got["base:role"]["expected"] == "TOP 100%"
    # мало игр — не сравниваем, причина в заметках
    small = build_base(hist[:10], "own")
    s, notes = surprises.detect(by_champ(one_game())["Zilean"], rules, small)
    assert not [x for x in s if x["rule"].startswith("base:")]
    assert any("1 игр < 20" in n for n in notes)


def test_game_log_without_snapshots():
    g = GameLog()
    assert g.records(CATALOG) == [] and not g.playing
