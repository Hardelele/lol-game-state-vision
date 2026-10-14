"""Что ролик заявляет о себе в названии и описании (#843).

Заявка — не факт: «Challenger» в названии канала-перезалива, «Volibear» в
списке «Champions in this video» — то, что написал автор. Индекс хранит
заявку отдельно от прочитанного с кадров (lolvision/video_verdict.py) и
сравнивает их.

* ранг — высший из названных: Challenger / Grandmaster / Master / Rank 1
  (англ., кор. 챌린저 / 그랜드마스터 / 그마 / 마스터, кит. 王者 / 宗师 / 大师);
* роль — top / jungle / mid / bot / support по словам «Top Lane», «Jungle»,
  «ADC», «Support», «탑», «정글», «미드», «원딜», «서폿»;
* чемпионы — по закрытому словарю имён Data Dragon (имя целым словом), в
  порядке появления: первый в «X vs Y» — обычно свой;
* признаки «не полная игра»: #shorts, highlights, montage, guide, tier list.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

RANKS = ("challenger", "grandmaster", "master", "rank1")
_RANK_WORDS = {
    "rank1": [r"rank\s*#?\s*1\b", r"\br1\b", r"#1\s+\w+\s+(?:world|euw|na|kr)", r"1위"],
    "challenger": [r"challenger", r"\bchall\b", r"챌린저", r"王者"],
    "grandmaster": [r"grand\s*master", r"\bgm\b", r"그랜드마스터", r"그마", r"宗师"],
    "master": [r"\bmaster\b", r"마스터", r"大师"],
}
ROLES = ("top", "jungle", "mid", "bot", "support")
_ROLE_WORDS = {
    "top": [r"\btop\b", r"\btop\s*lane", r"탑", r"上单"],
    "jungle": [r"\bjungle", r"\bjgl?\b", r"정글", r"打野"],
    "mid": [r"\bmid\b", r"\bmid\s*lane", r"미드", r"中单"],
    "bot": [r"\badc\b", r"\bbot\s*lane", r"\bmarksman", r"원딜", r"下路"],
    "support": [r"\bsupport\b", r"\bsupp?\b", r"서폿", r"서포터", r"辅助"],
}
_NOT_FULL = [r"#shorts?\b", r"\bhighlights?\b", r"\bmontage\b", r"\bguide\b", r"tier\s*list",
             r"\bbest\s+(?:moments|plays)\b", r"\bclips?\b", r"\bfunny\b", r"하이라이트"]
_FULL = [r"full\s*game", r"\bfull\s*match", r"\bpov\b", r"proview", r"풀영상", r"풀게임",
         r"gameplay", r"replay", r"第一视角"]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").lower()


def _any(text: str, pats: list[str]) -> bool:
    return any(re.search(p, text) for p in pats)


def rank_claim(title: str, description: str = "") -> str | None:
    """Высший ранг, названный в названии (а если там нет — в описании)."""
    for text in (_norm(title), _norm(description)):
        text = re.sub(r"master\s*yi|마스터\s*이", " ", text)     # чемпион, не ранг
        for r in RANKS:
            if _any(text, _RANK_WORDS[r]):
                return r
    return None


def role_claim(title: str, description: str = "") -> str | None:
    """Роль из названия; в описании — только явная строка роли/линии."""
    t = _norm(title)
    hits = [r for r in ROLES if _any(t, _ROLE_WORDS[r])]
    if len(hits) == 1:
        return hits[0]
    d = _norm(description)
    m = re.search(r"(?:role|lane|position)\s*[:：]\s*(\w+)", d)
    if m:
        for r in ROLES:
            if _any(m.group(1), _ROLE_WORDS[r]):
                return r
    return None


def _name_pattern(name: str) -> str:
    """«Kai'Sa» ловит и «Kaisa», «Kai Sa»; «Dr. Mundo» — «Mundo» тоже."""
    parts = [re.escape(p) for p in re.split(r"[\s'.&]+", _norm(name)) if p]
    return r"(?<![\w])" + r"[\s'.&]*".join(parts) + r"(?![\w])"


class ChampionNames:
    """Закрытый словарь имён: Id Data Dragon → отображаемое имя."""

    ALIASES = {"MonkeyKing": ["wukong"], "DrMundo": ["mundo"], "Nunu": ["nunu"],
               "Kaisa": ["kaisa", "kai'sa"], "KSante": ["ksante", "k'sante"],
               "TwistedFate": ["tf"], "MissFortune": ["mf"], "AurelionSol": ["asol"],
               "JarvanIV": ["j4", "jarvan"], "XinZhao": ["xin"], "LeeSin": ["lee sin"]}

    def __init__(self, names: Mapping[str, str]):
        self.pats: list[tuple[str, re.Pattern]] = []
        for cid, name in names.items():
            variants = {name, cid, *self.ALIASES.get(cid, [])}
            pat = "|".join(sorted({_name_pattern(v) for v in variants}, key=len, reverse=True))
            self.pats.append((cid, re.compile(pat)))

    def find(self, text: str) -> list[str]:
        """Id чемпионов в порядке первого появления в тексте."""
        t = _norm(text)
        found = []
        for cid, p in self.pats:
            m = p.search(t)
            if m:
                found.append((m.start(), -len(m.group(0)), cid))
        found.sort()
        out: list[str] = []
        spans: list[tuple[int, int]] = []
        for start, neg, cid in found:
            end = start - neg
            # «Lee Sin» не даёт ещё и «Sin»; «Master Yi» — не ранг «master»: ранг
            # ищется отдельно, тут только имена.
            if any(a <= start < b for a, b in spans):
                continue
            spans.append((start, end))
            out.append(cid)
        return out


def own_champion_claim(title: str, description: str, names: ChampionNames) -> str | None:
    """Свой чемпион по заявке: «Champions in this video: X» в описании,
    иначе первый чемпион названия («Volibear vs Corki» → Volibear)."""
    m = re.search(r"champions?\s+in\s+this\s+video\s*[:：]\s*([^\n]+)", description or "",
                  re.IGNORECASE)
    if m:
        got = names.find(m.group(1))
        if len(got) == 1:
            return got[0]
    got = names.find(title)
    return got[0] if got else None


def claims(title: str, description: str, names: ChampionNames | None = None) -> dict:
    """Все заявки ролика одной записью."""
    t = _norm(title)
    out = {"rank": rank_claim(title, description), "role": role_claim(title, description),
           "not_full": _any(t, _NOT_FULL), "full_words": _any(t, _FULL),
           "champions": [], "champion": None}
    if names is not None:
        out["champions"] = names.find(title)
        out["champion"] = own_champion_claim(title, description, names)
    return out
