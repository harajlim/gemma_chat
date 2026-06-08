"""Unit tests for web_search grounding (page-fetch + weather inject), no network."""
import web_search


def _fake_ddgs(rows):
    return lambda: type("D", (), {"text": lambda self, q, max_results=5: rows})()


def test_extract_text_strips_html_keeps_words():
    out = web_search._extract_text("<div><script>junk()</script><p>hello&nbsp;62&deg;F world</p></div>")
    assert "hello" in out and "world" in out and "junk" not in out
    assert "62" in out  # numbers survive


def test_weather_intent_matches():
    assert web_search._WEATHER_RE.search("what's the temperature in Paris")
    assert web_search._WEATHER_RE.search("is it raining outside")
    assert not web_search._WEATHER_RE.search("who is the president of france")


def test_extract_location_strips_filler():
    assert web_search._extract_location("what's the weather in San Francisco right now").strip() == "San Francisco"
    assert web_search._extract_location("how hot is it in Tokyo").strip() == "Tokyo"


def test_injects_live_weather_on_top(monkeypatch):
    monkeypatch.setattr(web_search, "DDGS", _fake_ddgs([{"title": "x", "href": "http://x", "body": "b"}]))
    monkeypatch.setattr(web_search, "FETCH_PAGES", False)
    monkeypatch.setattr(web_search, "_weather", lambda q: "Live current weather for Paris: 60°F (16°C), Sunny.")
    res = web_search.web_search("weather in Paris")
    assert res[0]["title"] == "Live weather (wttr.in)"
    assert "60°F" in res[0]["content"]
    assert res[1]["title"] == "x"  # original results preserved below


def test_no_weather_inject_for_general_query(monkeypatch):
    monkeypatch.setattr(web_search, "DDGS", _fake_ddgs([{"title": "x", "href": "http://x", "body": "b"}]))
    monkeypatch.setattr(web_search, "FETCH_PAGES", False)
    called = []
    monkeypatch.setattr(web_search, "_weather", lambda q: called.append(1) or "nope")
    res = web_search.web_search("who won the 2022 world cup")
    assert res[0]["title"] == "x"
    assert not called  # _weather never invoked for non-weather queries


def test_page_fetch_adds_content(monkeypatch):
    monkeypatch.setattr(web_search, "DDGS", _fake_ddgs([{"title": "t", "href": "http://x", "body": "snip"}]))
    monkeypatch.setattr(web_search, "FETCH_PAGES", True)
    monkeypatch.setattr(web_search, "FETCH_N", 1)
    monkeypatch.setattr(web_search, "_fetch", lambda url: "real page text with the actual value 62")
    res = web_search.web_search("latest python version")   # non-weather -> no inject
    assert res[0]["content"] == "real page text with the actual value 62"
