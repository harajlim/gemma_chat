"""
Web search via DuckDuckGo. No API key required.
"""

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo and return a list of results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of dicts with 'title', 'url', and 'snippet' keys.
    """
    raw = list(DDGS().text(query, max_results=max_results))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "latest news"
    results = web_search(q, max_results=5)
    print(json.dumps(results, indent=2))
