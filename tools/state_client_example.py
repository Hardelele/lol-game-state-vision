"""Пример тонкого клиента сервиса состояния (#796) — как coach берёт факты.

Только стандартная библиотека и только этот файл: coach (асинхронный, httpx)
перенесёт это к себе сам; здесь — образец контракта docs/state-api.md.

* `get_state(url)` — снимок `GET /state`;
* `events(url)` — поток `GET /events` (Server-Sent Events) с
  переподключением: после разрыва сервис снова шлёт `hello` со свежим
  снимком, пропущенные события не досылаются;
* `to_source_facts(snapshot)` — короткий список фактов для модели в духе
  `coach.analysis_types.SourceFact` (id, text, source, observed_at): только
  значения с источником, отказы не превращаются в утверждения; последние
  события матча (`events`, #803) — одной строкой `events.recent`.

Запуск: `python tools/state_client_example.py [--url http://127.0.0.1:8733]`
печатает факты снимка и затем события потока.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator

URL = "http://127.0.0.1:8733"
# Прокси из окружения loopback не касается.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_state(url: str = URL, timeout: float = 2.0) -> dict:
    with _OPENER.open(url + "/state", timeout=timeout) as r:
        snap = json.loads(r.read())
    if snap.get("schema") != 1:
        raise ValueError(f"незнакомая версия схемы: {snap.get('schema')}")
    return snap


def events(url: str = URL, retry: float = 1.0, timeout: float = 30.0) -> Iterator[tuple[str, dict]]:
    """(вид, данные) из потока; разрыв — переподключение через `retry` с."""
    u = urllib.parse.urlsplit(url)
    while True:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        try:
            conn.request("GET", "/events", headers={"Accept": "text/event-stream"})
            resp = conn.getresponse()
            kind, data = None, []
            while True:
                raw = resp.readline()
                if not raw:
                    break                                   # сервис закрыл поток
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith(":"):
                    continue                                # ping
                if line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif line == "" and kind is not None:
                    yield kind, json.loads("\n".join(data))
                    kind, data = None, []
        except OSError as e:
            print(f"поток: {e}", file=sys.stderr)
        finally:
            conn.close()
        time.sleep(retry)


def _f(fact: dict) -> bool:
    return isinstance(fact, dict) and fact.get("value") is not None


def to_source_facts(snap: dict) -> list[dict]:
    """Короткий список фактов для модели. `observed_at` — unix-время
    наблюдения: `t` снимка минус давность факта."""
    t = snap["t"]
    out = []

    def add(fid: str, text: str, fact: dict) -> None:
        age = fact["age"] or 0.0
        out.append({"id": fid, "text": text, "source": fact["source"],
                    "observed_at": round(t - age, 2)})

    m = snap["match"]
    if _f(m["time"]):
        s = m["time"]["value"]
        add("match.time", f"время матча {int(s // 60)}:{int(s % 60):02d}", m["time"])
    if _f(m["side"]):
        add("match.side", f"сторона игрока {m['side']['value']}", m["side"])
    me = snap["me"]
    if _f(me["champion"]):
        add("me.champion", f"игрок на {me['champion']['value']}", me["champion"])
    if _f(me["alive"]) and me["alive"]["value"] is False and _f(me["respawn_in"]):
        add("me.respawn", f"игрок мёртв, возрождение через {me['respawn_in']['value']:.0f} с",
            me["respawn_in"])
    if _f(snap["enemies"]):
        for e in snap["enemies"]["value"]:
            if not _f(e["champion"]):
                continue
            name = e["champion"]["value"]
            if _f(e["alive"]) and e["alive"]["value"] is False:
                add(f"enemy.{name}.dead", f"враг {name} мёртв", e["alive"])
            elif _f(e["position"]):
                p, age = e["position"]["value"], e["position"]["age"] or 0.0
                add(f"enemy.{name}.pos",
                    f"враг {name} на карте u={p['u']:.2f} v={p['v']:.2f}, {age:.0f} с назад",
                    e["position"])
    ev = snap.get("events")                      # поле с #803; старый сервис его не шлёт
    if _f(ev) and ev["value"]:
        add("events.recent", "недавние события: " +
            "; ".join(event_text(e) for e in ev["value"][-RECENT:]), ev)
    return out


RECENT = 5                                       # сколько последних событий в строке
_REL = {"self": "вы", "ally": "союзник", "enemy": "враг"}
_OTHER = {"turret": "башня", "minion": "миньоны", "monster": "монстр", "unknown": "неизвестно"}


def _who(w: dict | None) -> str:
    """Участник события: чемпион или метка объекта и отношение к игроку."""
    if w is None:
        return "—"
    name = w["champion"] or _OTHER.get(w["other"], "неизвестно")
    rel = _REL.get(w["relation"])
    return f"{name} ({rel})" if rel else name


def event_text(e: dict) -> str:
    """Элемент `events` → «ChampionKill Darius (враг) → Zilean (вы), 4 с назад»."""
    s = f"{e['kind']} {_who(e['killer'])}"
    if e["victim"] is not None:
        s += f" → {_who(e['victim'])}"
    if e["detail"]:
        s += f" [{e['detail']}]"
    if e["ago"] is not None:
        s += f", {e['ago']:.0f} с назад"
    return s


def event_line(kind: str, msg: dict) -> str:
    """Строка печати для сообщения потока. У снимка (`hello`, `state`) —
    номер и число расхождений источников: `conflicts` — факт, его значение
    `{"total", "kinds"}`; отказ печатается причиной, а не нулём."""
    if kind not in ("hello", "state"):
        return f"{kind} {json.dumps(msg['data'], ensure_ascii=False)}"
    snap = msg["data"]
    c = snap["conflicts"]
    conflicts = c["value"]["total"] if _f(c) else f"нет ({c['reason']})"
    return f"{kind} seq {snap['seq']} conflicts {conflicts}"


def main(argv: list[str] | None = None) -> None:
    # Консоль Windows в cp1251: непечатное — «?», а не исключение.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--url", default=URL)
    a = ap.parse_args(argv)
    for fact in to_source_facts(get_state(a.url)):
        print(json.dumps(fact, ensure_ascii=False))
    for kind, msg in events(a.url):
        print(event_line(kind, msg))


if __name__ == "__main__":
    main()
