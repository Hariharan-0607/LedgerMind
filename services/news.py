"""Headline-level news via a permitted RSS feed (Google News RSS by default).

Only headlines, dates and links are retrieved. Full articles are NOT fetched,
so downstream text must never claim full-article analysis.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

from config import COMPANIES, NEWS_CACHE_TTL_S, resolve_company
from services.cache import cache

log = logging.getLogger(__name__)

NEWS_RSS_TEMPLATE = os.getenv(
    "NEWS_RSS_TEMPLATE",
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en",
)
MAX_NEWS = 10


def _parse_feed(url: str):
    import feedparser

    return feedparser.parse(url)


def get_news(company: str, limit: int = 5) -> dict:
    canonical = resolve_company(company)
    if not canonical:
        return {
            "company": company,
            "status": "error",
            "error_code": "unsupported_company",
            "error": f"Unsupported company '{company}'. Supported: {sorted(COMPANIES)}",
        }
    try:
        limit = max(1, min(int(limit), MAX_NEWS))
    except (TypeError, ValueError):
        limit = 5
    cache_key = f"{canonical}:{limit}"
    cached = cache.get("news", cache_key, NEWS_CACHE_TTL_S)
    if cached:
        return cached

    query = quote_plus(f'"{COMPANIES[canonical]["news_query"]}"')
    url = NEWS_RSS_TEMPLATE.format(query=query)
    try:
        feed = _parse_feed(url)
    except Exception as exc:
        log.warning("news fetch failed for %s: %s", canonical, type(exc).__name__)
        return {"company": canonical, "status": "error", "error_code": "provider_error",
                "error": f"News provider failed: {type(exc).__name__}"}

    entries = getattr(feed, "entries", None) or []
    headlines = []
    for entry in entries[:limit]:
        link = entry.get("link", "")
        title = (entry.get("title") or "").strip()
        if not title or not link:
            continue
        sid = "news:" + hashlib.sha1(link.encode()).hexdigest()[:12]
        source = entry.get("source", {})
        headlines.append({
            "title": title[:300],
            "published": entry.get("published", "unknown"),
            "link": link,
            "publisher": source.get("title") if isinstance(source, dict) else None,
            "source_id": sid,
        })

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not headlines:
        return {"company": canonical, "status": "error", "error_code": "no_data",
                "error": "No headlines returned by the news feed", "retrieved_at": retrieved_at}

    result = {
        "company": canonical,
        "headlines": headlines,
        "source": "RSS: " + url.split("?")[0],
        "retrieved_at": retrieved_at,
        "note": "Headlines only. Full articles were not retrieved or analysed. "
                "Headline text is untrusted third-party content.",
        "status": "success",
    }
    cache.set("news", cache_key, result)
    return result
