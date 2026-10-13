"""Детектор нетипичного выбора (#807): правила по роли и сравнение с базой частот.

Пока только помечает строки паспорта выбора (lolstream.loadouts) — без
голоса и без /state. Каждая пометка:

    {"kind": "spell" | "keystone" | "item" | "role",
     "what": что взято / чего нет ("SummonerTeleport", "no SummonerSmite"),
     "expected": что ожидалось (строкой),
     "confidence": 0..1 — насколько правило уверено, что это необычно,
     "basis": откуда вывод ("правило bottom_teleport: …" / "база own: …"),
     "rule": id правила или "base:<измерение>"}

Правила — TOML (`RULES`, по умолчанию lolstream/surprise_rules.toml рядом
с кодом; свой файл — `load_rules(path)`). Правила по роли работают, только
когда у игрока есть роль (ранкед / драфт); почему что-то не проверялось —
`surprises_notes` в строке, а не молчание.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lolstream.loadouts import Base, first_legendary

RULES = Path(__file__).with_name("surprise_rules.toml")
RULES_VERSION = 1


@dataclass(frozen=True)
class Rules:
    data: Mapping[str, Any]
    path: str

    @property
    def base(self) -> Mapping[str, Any]:
        return self.data.get("base") or {}


def load_rules(path: Path | None = None) -> Rules:
    """TOML правил → Rules; не та версия — ValueError."""
    p = Path(path) if path is not None else RULES
    with p.open("rb") as f:
        d = tomllib.load(f)
    if d.get("version") != RULES_VERSION:
        raise ValueError(f"{p}: версия правил {d.get('version')!r}, ожидалась {RULES_VERSION}")
    return Rules(d, str(p))


def _s(kind: str, what: str, expected: str, conf: float, basis: str, rule: str) -> dict:
    return {"kind": kind, "what": what, "expected": expected,
            "confidence": round(float(conf), 2), "basis": basis, "rule": rule}


# ---------------------------------------------------------------- правила по роли

def _spell_rules(rec: Mapping, role: str, rules: Rules) -> list[dict]:
    fams = set((rec.get("spells") or {}).get("families") or [])
    sp = rules.data.get("spells") or {}
    out, covered = [], set()
    for r in sp.get("rule") or []:
        if role not in r.get("roles", []):
            continue
        basis = f"правило {r['id']}: {r.get('why', '')}".rstrip(": ")
        if "has" in r and r["has"] in fams:
            covered.add(r["has"])
            exp = (sp.get("roles") or {}).get(role, {}).get("expected")
            out.append(_s("spell", r["has"], f"{role}: " + (", ".join(exp) if exp else "другое"),
                          r["confidence"], basis, r["id"]))
        if "lacks" in r and r["lacks"] not in fams:
            out.append(_s("spell", f"no {r['lacks']}", f"{role}: {r['lacks']}",
                          r["confidence"], basis, r["id"]))
    by_role = (sp.get("roles") or {}).get(role)
    if by_role:
        expected = list(by_role.get("expected") or [])
        for fam in sorted(fams - set(expected) - covered):
            out.append(_s("spell", fam, f"{role}: {', '.join(expected)}",
                          by_role.get("other_confidence", 0.4),
                          f"правило spells.roles.{role}: семьи нет в ожидаемых для роли",
                          f"spells.roles.{role}"))
    return out


def _keystone_rules(rec: Mapping, role: str, rules: Rules) -> list[dict]:
    ks = ((rec.get("runes") or {}).get("keystone") or {}).get("name")
    r = ((rules.data.get("keystones") or {}).get("roles") or {}).get(role)
    if not ks or not r or ks not in (r.get("unusual") or []):
        return []
    return [_s("keystone", ks, f"{role}: не {', '.join(r['unusual'])}", r.get("confidence", 0.3),
               f"правило keystones.roles.{role}: ключевая руна необычна для роли "
               "(без учёта чемпиона)", f"keystones.roles.{role}")]


def _held(rec: Mapping) -> set[int]:
    it = rec.get("items") or {}
    ids = {p["item"] for p in it.get("purchases") or []}
    ids.update(it.get("final") or [])
    ids.update(it.get("start") or [])
    return ids


def _item_rules(rec: Mapping, role: str, rules: Rules) -> tuple[list[dict], list[str]]:
    items = rules.data.get("items") or {}
    sets = {k: set(v) for k, v in (items.get("sets") or {}).items()}
    held, out, notes = _held(rec), [], []
    duration = rec.get("duration") or 0.0
    for r in items.get("rule") or []:
        if role not in r.get("roles", []):
            continue
        basis = f"правило {r['id']}: {r.get('why', '')}".rstrip(": ")
        if "has_any" in r:
            hit = sorted(held & sets.get(r["has_any"], set()))
            if hit:
                out.append(_s("item", f"{r['has_any']}: {hit[0]}", f"{role}: без {r['has_any']}",
                              r["confidence"], basis, r["id"]))
        if "lacks_any" in r:
            if duration < r.get("min_game_time", 0):
                notes.append(f"{r['id']}: игра короче {r.get('min_game_time')} с — не проверялось")
            elif not held & sets.get(r["lacks_any"], set()):
                out.append(_s("item", f"no {r['lacks_any']}", f"{role}: {r['lacks_any']}",
                              r["confidence"], basis, r["id"]))
    return out, notes


# ---------------------------------------------------------------- база

def _conf(games: int, cfg: Mapping) -> float:
    return min(cfg.get("max_confidence", 0.9), games / (games + cfg.get("confidence_games", 30)))


def _base_checks(rec: Mapping, role: str | None, base: Base, cfg: Mapping) -> tuple[list[dict],
                                                                                    list[str]]:
    champ = rec.get("champion")
    dims = set(cfg.get("dimensions") or [])
    min_n, rare = cfg.get("min_games", 20), cfg.get("rare_share", 0.05)
    src = f"база {base.source}" + (f" ({base.patch})" if base.patch else "")
    out, notes = [], []
    c = base.champion(champ) if champ else None
    if "role" in dims and role and c is not None:
        n = c["games"]
        if n < min_n:
            notes.append(f"{src}: {champ} — {n} игр < {min_n}, роль не сравнивалась")
        else:
            share = c["roles"].get(role, 0) / n
            if share < rare:
                top = ", ".join(f"{r} {k / n:.0%}" for r, k in c["roles"].most_common(3))
                out.append(_s("role", role, top, _conf(n, cfg),
                              f"{src}: {champ} в роли {role} {share:.0%} из {n} игр",
                              "base:role"))
    if not role:
        return out, notes
    e = base.entry(champ, role) if champ else None
    if e is None:
        notes.append(f"{src}: нет {champ} {role}")
        return out, notes
    n = e["games"]
    if n < min_n:
        notes.append(f"{src}: {champ} {role} — {n} игр < {min_n}, не сравнивалось")
        return out, notes

    def rare_one(kind: str, what: str, counts, label: str) -> None:
        share = counts.get(what, 0) / n
        if share < rare:
            top = ", ".join(f"{k} {v / n:.0%}" for k, v in counts.most_common(3))
            out.append(_s(kind, what, top or "—", _conf(n, cfg),
                          f"{src}: {champ} {role}, {label} {share:.0%} из {n} игр",
                          f"base:{kind if kind != 'item' else 'first_item'}"))

    if "spells" in dims:
        for fam in (rec.get("spells") or {}).get("families") or []:
            rare_one("spell", fam, e["spells"], f"{fam}")
    ks = ((rec.get("runes") or {}).get("keystone") or {}).get("name")
    if "keystone" in dims and ks:
        rare_one("keystone", ks, e["keystones"], ks)
    fi = first_legendary(rec)
    if "first_item" in dims and fi is not None and sum(e["first_item"].values()):
        rare_one("item", str(fi), e["first_item"], f"первый предмет {fi}")
    return out, notes


# ---------------------------------------------------------------- всё вместе

def detect(rec: Mapping, rules: Rules, base: Base | None = None,
           base_reason: str | None = None) -> tuple[list[dict], list[str]]:
    """Строка паспорта → (сюрпризы, заметки «что не проверялось и почему»)."""
    role = rec.get("role")
    out, notes = [], []
    if not role:
        notes.append(f"роли нет ({rec.get('role_reason') or '—'}): правила по роли не применялись")
    else:
        out += _spell_rules(rec, role, rules)
        out += _keystone_rules(rec, role, rules)
        items, inotes = _item_rules(rec, role, rules)
        out += items
        notes += inotes
    if rec.get("spells") is None:
        notes.append(f"заклинаний нет: {rec.get('spells_reason')}")
    if rec.get("runes") is None:
        notes.append(f"рун нет: {rec.get('runes_reason')}")
    if base is None:
        notes.append(f"база частот не применялась: {base_reason or 'базы нет'}")
    else:
        b, bnotes = _base_checks(rec, role, base, rules.base)
        out += b
        notes += bnotes
    return out, notes


def annotate(records: Sequence[dict], rules: Rules, base: Base | None = None,
             base_reason: str | None = None) -> None:
    """Дописать в строки `surprises`, `surprises_notes`, `rules` (путь, версия) на месте."""
    for r in records:
        s, notes = detect(r, rules, base, base_reason)
        r["surprises"] = s
        r["surprises_notes"] = notes
        r["rules"] = {"path": Path(rules.path).name, "version": rules.data.get("version"),
                      "base": None if base is None else {"source": base.source,
                                                         "built": base.built,
                                                         "games": base.games,
                                                         "note": base.note}}
