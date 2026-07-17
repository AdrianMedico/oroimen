"""Sprint 4 MVP-2 T11b (minimalista): tool de RSS via feedparser.

Script standalone que parsea un feed RSS/Atom y devuelve los ultimos
N items en JSON. El wrapper agent_reach.py lo invoca via subprocess
para mantener consistencia con las otras 3 tools (read, youtube,
github) que SI usan subprocess.

Por que un script aparte (vs inline python3 -c):
- Comillas dobles en feedparser + el feed se vuelven fragiles con -c.
- Feedparser necesita parsear XML, mejor tener un file con line endings
  consistentes (LF) que un -c inline.
- Reutilizable: si en algun momento queremos invocarlo desde otro lado
  (script, tests, debug), ya existe.

JSON output (5 items max):
{
  "feed_title": "...",
  "items": [
    {
      "title": "...",
      "link": "...",
      "published": "2024-01-15T10:00:00+00:00",
      "summary": "primeros 300 chars del summary..."
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import sys
import warnings

# feedparser emite DeprecationWarnings sobre cosas internas; las ignoramos
# para no contaminar el output (que va al LLM via subprocess stdout).
warnings.filterwarnings("ignore")

import feedparser  # noqa: E402

_MAX_ITEMS = 5
_MAX_SUMMARY_CHARS = 300


def main() -> int:
    if len(sys.argv) != 2:
        print(
            json.dumps({"error": "usage: rss_read.py <feed_url>"}) + "\n",
            file=sys.stderr,
        )
        return 2

    feed_url = sys.argv[1]

    try:
        d = feedparser.parse(feed_url)
    except Exception as exc:
        print(
            json.dumps({"error": f"failed to parse feed: {exc}"}) + "\n",
            file=sys.stderr,
        )
        return 1

    if d.bozo and not d.entries:
        # bozo=True + entries vacias = feed invalido
        err = d.bozo_exception if hasattr(d, "bozo_exception") else "invalid feed"
        print(
            json.dumps({"error": f"invalid RSS/Atom feed: {err}"}) + "\n",
            file=sys.stderr,
        )
        return 1

    items = []
    for e in d.entries[:_MAX_ITEMS]:
        items.append(
            {
                "title": str(getattr(e, "title", "") or ""),
                "link": str(getattr(e, "link", "") or ""),
                "published": str(getattr(e, "published", "") or ""),
                "summary": str((getattr(e, "summary", "") or "")[:_MAX_SUMMARY_CHARS]),
            }
        )

    result = {
        "feed_title": str(getattr(d.feed, "title", "") or ""),
        "items": items,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
