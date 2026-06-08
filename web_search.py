"""
Web search via DuckDuckGo, with page-content grounding.

DDG's text search only returns marketing-y *descriptions* ("…current weather
report with temperature, feels like, wind…") — never the actual numbers. So the
model has nothing to ground on and hallucinates (e.g. wrong temps).

To fix that we don't stop at the snippet: for the top few results we FETCH the
page, strip it to readable text, and return that excerpt as `content`. The real
"62°F, partly cloudy" on the page then lands in the model's context.

Fetches run concurrently with a short timeout so the extra latency is ~one slow
page, not the sum — fine for a voice turn. Anything that fails falls back to the
snippet. No JS is executed, so values rendered purely client-side may still be
missing (that's a known limit; a plain-text source like wttr.in is the fallback
for those).
"""

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests
from ddgs import DDGS

# Page-fetch grounding (override via env).
FETCH_PAGES = os.environ.get("WEB_SEARCH_FETCH", "1") != "0"
FETCH_N = int(os.environ.get("WEB_SEARCH_FETCH_N", "3"))      # how many top pages to fetch
FETCH_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "4"))  # seconds per page
MAX_CHARS = int(os.environ.get("WEB_SEARCH_MAX_CHARS", "2000"))   # text kept per page
WEATHER_ANSWER = os.environ.get("WEB_SEARCH_WEATHER", "1") != "0"  # wttr.in live-weather inject

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

_DROP = re.compile(r"(?is)<(script|style|noscript|svg|head|nav|footer|form)[^>]*>.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_WS = re.compile(r"\s+")


def _extract_text(page_html: str) -> str:
    """Crude but dependency-free HTML -> visible text (keeps numbers/units)."""
    page_html = _DROP.sub(" ", page_html)
    text = _TAGS.sub(" ", page_html)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def _fetch(url: str) -> str:
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": _UA}, allow_redirects=True)
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return ""
        return _extract_text(r.text)[:MAX_CHARS]
    except Exception:  # noqa: BLE001
        return ""


# --- Live weather instant-answer -------------------------------------------
# The major weather sites render the temperature with JavaScript, so fetching
# their HTML doesn't surface a number. wttr.in returns live data as JSON/text;
# with no location it geolocates by the requester's IP (i.e. this machine).
_WX_WORDS = r"weather|temperatures?|temp|forecast|degrees?|humidity|rain(?:ing)?|snow(?:ing)?|sunny|cloudy|windy|wind|hot|cold|warm|cool|chilly|freezing"
_WEATHER_RE = re.compile(r"\b(?:" + _WX_WORDS + r")\b", re.I)
_FILLER = re.compile(
    r"\b(?:what|whats|what's|is|are|it|its|the|a|an|in|at|for|right|now|today|tonight|"
    r"tomorrow|current|currently|like|of|how|me|tell|please|can|could|you|get|there|to|do|outside)\b",
    re.I,
)


def _extract_location(query: str) -> str:
    loc = re.sub(r"[’'`]", "", query)        # what's -> whats (so the filler matches it)
    loc = _WEATHER_RE.sub(" ", loc)
    loc = _FILLER.sub(" ", loc)
    loc = re.sub(r"[^\w\s,.\-]", " ", loc)
    return re.sub(r"\s+", " ", loc).strip()


def _area_name(j: dict) -> str:
    na = j.get("nearest_area") or []
    if not na:
        return ""
    a, parts = na[0], []
    for key in ("areaName", "region", "country"):
        v = a.get(key)
        if isinstance(v, list) and v and v[0].get("value"):
            parts.append(v[0]["value"])
    return ", ".join(parts)


def _weather(query: str):
    """Return a plain-text live-weather line for a weather query, or None."""
    loc = _extract_location(query)
    try:
        r = requests.get(f"https://wttr.in/{quote(loc)}?format=j1",
                         timeout=FETCH_TIMEOUT, headers={"User-Agent": "curl/8"})
        j = r.json()
        c = j["current_condition"][0]
    except Exception:  # noqa: BLE001
        return None
    where = _area_name(j) or loc or "your area"
    return (f"Live current weather for {where}: {c['temp_F']}°F ({c['temp_C']}°C), "
            f"feels like {c['FeelsLikeF']}°F, {c['weatherDesc'][0]['value']}, "
            f"humidity {c['humidity']}%, wind {c['windspeedMiles']} mph. (source: wttr.in)")


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo and return a list of results. Each result has
    'title', 'url', 'snippet', and (for the top FETCH_N) a 'content' excerpt
    pulled from the actual page so the model can read real values.
    """
    raw = list(DDGS().text(query, max_results=max_results))
    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]

    if FETCH_PAGES and results:
        top = results[: max(0, FETCH_N)]
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(top))) as ex:
                texts = list(ex.map(lambda r: _fetch(r["url"]), top))
            for r, t in zip(top, texts):
                if t:
                    r["content"] = t
        except Exception:  # noqa: BLE001
            pass  # grounding is best-effort; snippets remain

    # Weather: the big sites are JS-rendered, so inject live data from wttr.in
    # as the top result (real numbers the model can read).
    if WEATHER_ANSWER and _WEATHER_RE.search(query):
        summary = _weather(query)
        if summary:
            results.insert(0, {"title": "Live weather (wttr.in)", "url": "https://wttr.in",
                               "snippet": summary, "content": summary})

    return results


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "latest news"
    out = web_search(q, max_results=5)
    for r in out:
        print(r["title"])
        print("  url:", r["url"])
        if r.get("content"):
            print("  content:", r["content"][:300])
        else:
            print("  snippet:", r["snippet"][:200])
