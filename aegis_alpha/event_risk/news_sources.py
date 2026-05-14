from __future__ import annotations

import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
DEFAULT_RSS_SOURCES: list[dict[str, str]] = [
    {
        "name": "coindesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    },
    {
        "name": "cointelegraph",
        "url": "https://cointelegraph.com/rss",
    },
    {
        "name": "decrypt",
        "url": "https://decrypt.co/feed",
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _http_get_text(url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AegisEventRiskCollector/0.1",
            "Accept": "application/json, application/rss+xml, application/xml, text/xml, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _source_unavailable(name: str, url: str | None, error: Exception | str) -> dict[str, Any]:
    return {
        "name": name,
        "url": url,
        "status": "source_unavailable",
        "error": str(error),
        "items": [],
        "fetched_at": utc_now_iso(),
    }


def load_manual_event_risk(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {
            "name": "manual_event_risk",
            "url": str(config_path),
            "status": "missing",
            "active": False,
            "items": [],
            "fetched_at": utc_now_iso(),
        }

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _source_unavailable("manual_event_risk", str(config_path), exc)

    enabled = bool(payload.get("enabled", False))
    mode = str(payload.get("mode", "NORMAL")).upper()
    expires_at = _parse_datetime(payload.get("expires_at"))
    expired = bool(expires_at and expires_at <= datetime.now(timezone.utc))
    active = bool(enabled and not expired)
    reason = str(payload.get("reason", "manual event risk")).strip() or "manual event risk"
    item = {
        "title": f"manual_event_risk:{mode}",
        "summary": reason,
        "mode": mode,
        "expires_at": payload.get("expires_at"),
        "active": active,
    }
    return {
        "name": "manual_event_risk",
        "url": str(config_path),
        "status": "ok",
        "active": active,
        "enabled": enabled,
        "mode": mode,
        "expires_at": payload.get("expires_at"),
        "reason": reason,
        "items": [item],
        "fetched_at": utc_now_iso(),
    }


def fetch_fear_and_greed(url: str = DEFAULT_FEAR_GREED_URL, timeout_seconds: float = 4.0) -> dict[str, Any]:
    try:
        payload = json.loads(_http_get_text(url, timeout_seconds))
        data = list(payload.get("data") or [])
        latest = data[0] if data else {}
        value = latest.get("value")
        classification = latest.get("value_classification")
        timestamp = latest.get("timestamp")
        item = {
            "title": f"Fear & Greed Index: {classification or 'unknown'} {value or ''}".strip(),
            "summary": "Crypto Fear & Greed Index",
            "value": value,
            "classification": classification,
            "timestamp": timestamp,
        }
        return {
            "name": "fear_greed",
            "url": url,
            "status": "ok",
            "items": [item],
            "fetched_at": utc_now_iso(),
        }
    except (OSError, ValueError, urllib.error.URLError, TimeoutError) as exc:
        return _source_unavailable("fear_greed", url, exc)


def fetch_rss_headlines(source: dict[str, str], timeout_seconds: float = 4.0, limit: int = 12) -> dict[str, Any]:
    name = source.get("name") or "rss"
    url = source.get("url")
    if not url:
        return _source_unavailable(name, None, "missing_url")

    try:
        text = _http_get_text(url, timeout_seconds)
        root = ET.fromstring(text)
        items: list[dict[str, Any]] = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            summary = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            published_at = (item.findtext("pubDate") or item.findtext("published") or "").strip()
            if title:
                items.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published_at": published_at,
                })
        return {
            "name": name,
            "url": url,
            "status": "ok",
            "items": items,
            "fetched_at": utc_now_iso(),
        }
    except (OSError, ET.ParseError, urllib.error.URLError, TimeoutError) as exc:
        return _source_unavailable(name, url, exc)


def collect_news_sources(
    *,
    manual_config_path: str | Path,
    include_http: bool = True,
    rss_sources: list[dict[str, str]] | None = None,
    timeout_seconds: float = 4.0,
) -> list[dict[str, Any]]:
    sources = [load_manual_event_risk(manual_config_path)]
    if not include_http:
        return sources

    sources.append(fetch_fear_and_greed(timeout_seconds=timeout_seconds))
    for source in rss_sources or DEFAULT_RSS_SOURCES:
        sources.append(fetch_rss_headlines(source, timeout_seconds=timeout_seconds))
    return sources
