from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


MODE_RANK = {
    "NORMAL": 0,
    "CAUTION": 1,
    "RISK_OFF": 2,
    "MANUAL_ONLY": 3,
}
MODE_BY_RANK = {value: key for key, value in MODE_RANK.items()}

RISK_KEYWORDS: dict[str, float] = {
    "war": 0.18,
    "ceasefire": 0.10,
    "attack": 0.18,
    "tariff": 0.12,
    "sanctions": 0.15,
    "inflation": 0.12,
    "cpi": 0.14,
    "ppi": 0.12,
    "fed": 0.10,
    "rates": 0.10,
    "rate hike": 0.16,
    "etf outflow": 0.16,
    "etf outflows": 0.16,
    "hack": 0.22,
    "exploit": 0.22,
    "depeg": 0.25,
    "liquidation": 0.12,
    "liquidations": 0.12,
    "sec": 0.10,
    "exchange outage": 0.22,
    "outage": 0.14,
    "trump": 0.10,
    "lawsuit": 0.12,
    "bankruptcy": 0.20,
}

POSITIVE_KEYWORDS: dict[str, float] = {
    "etf inflow": 0.14,
    "etf inflows": 0.14,
    "ceasefire confirmed": 0.18,
    "rate cut": 0.14,
    "lower inflation": 0.16,
    "cooling inflation": 0.14,
    "institutional inflow": 0.12,
    "institutional inflows": 0.12,
    "approval": 0.08,
}

FNG_RISK_CLASSIFICATION = {
    "extreme fear": 0.34,
    "fear": 0.20,
    "neutral": 0.02,
    "greed": -0.04,
    "extreme greed": 0.08,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _keyword_hits(text: str, keywords: dict[str, float]) -> tuple[list[str], float]:
    lower = text.lower()
    hits: list[str] = []
    score = 0.0
    for keyword, weight in keywords.items():
        pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(part) for part in keyword.split()) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            hits.append(keyword)
            score += weight
    return hits, score


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _score_fear_greed(item: dict[str, Any]) -> tuple[list[str], float]:
    classification = str(item.get("classification") or "").lower()
    value = _safe_float(item.get("value"))
    hits: list[str] = []
    score = 0.0
    if classification:
        score += FNG_RISK_CLASSIFICATION.get(classification, 0.0)
        hits.append(f"fear_greed:{classification.replace(' ', '_')}")
    if value is not None:
        if value <= 20:
            score += 0.16
            hits.append("fear_greed_extreme_low")
        elif value <= 35:
            score += 0.08
            hits.append("fear_greed_low")
    return hits, score


def _mode_from_score(score: float) -> str:
    if score >= 0.88:
        return "MANUAL_ONLY"
    if score >= 0.65:
        return "RISK_OFF"
    if score >= 0.32:
        return "CAUTION"
    return "NORMAL"


def _score_floor_for_mode(mode: str) -> float:
    return {
        "NORMAL": 0.10,
        "CAUTION": 0.40,
        "RISK_OFF": 0.68,
        "MANUAL_ONLY": 0.90,
    }.get(mode, 0.10)


def _higher_mode(left: str, right: str) -> str:
    return MODE_BY_RANK[max(MODE_RANK.get(left, 0), MODE_RANK.get(right, 0))]


def classify_event_sentiment_risk(sources: list[dict[str, Any]]) -> dict[str, Any]:
    top_events: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    source_unavailable = 0
    ok_sources = 0
    external_source_unavailable = 0
    external_ok_sources = 0
    manual_mode = "NORMAL"
    manual_active = False
    manual_reason: str | None = None

    for source in sources:
        name = str(source.get("name") or "unknown")
        status = str(source.get("status") or "unknown")
        items = list(source.get("items") or [])
        source_summaries.append({
            "name": name,
            "status": status,
            "item_count": len(items),
            "fetched_at": source.get("fetched_at"),
            "url": source.get("url"),
        })
        if status == "ok":
            ok_sources += 1
        elif status == "source_unavailable":
            source_unavailable += 1

        if name == "manual_event_risk":
            manual_active = bool(source.get("active"))
            manual_mode = str(source.get("mode") or "NORMAL").upper()
            manual_reason = str(source.get("reason") or "manual_event_risk")
            if manual_active:
                top_events.append({
                    "source": name,
                    "title": f"manual override {manual_mode}",
                    "summary": manual_reason,
                    "risk_score": _score_floor_for_mode(manual_mode),
                    "matched_risk_keywords": ["manual_override"],
                    "matched_positive_keywords": [],
                    "link": None,
                })
            continue

        if status == "ok":
            external_ok_sources += 1
        elif status == "source_unavailable":
            external_source_unavailable += 1

        for item in items:
            title = _clean_text(item.get("title"))
            summary = _clean_text(item.get("summary"))
            text = f"{title} {summary}"
            risk_hits, risk_score = _keyword_hits(text, RISK_KEYWORDS)
            positive_hits, positive_score = _keyword_hits(text, POSITIVE_KEYWORDS)
            if name == "fear_greed":
                fng_hits, fng_score = _score_fear_greed(item)
                risk_hits.extend(fng_hits)
                risk_score += fng_score
            event_score = max(0.0, min(1.0, risk_score - positive_score))
            if event_score <= 0 and not risk_hits and not positive_hits:
                continue
            top_events.append({
                "source": name,
                "title": title,
                "summary": summary[:280],
                "risk_score": round(event_score, 3),
                "matched_risk_keywords": risk_hits,
                "matched_positive_keywords": positive_hits,
                "link": item.get("link"),
            })

    top_events.sort(key=lambda event: float(event.get("risk_score") or 0.0), reverse=True)
    top_events = top_events[:8]

    if top_events:
        weighted_score = sum(float(event.get("risk_score") or 0.0) for event in top_events[:5])
        risk_score = min(1.0, weighted_score / max(1.7, len(top_events[:5]) * 0.55))
    else:
        risk_score = 0.10

    suggested_mode = _mode_from_score(risk_score)
    if manual_active:
        suggested_mode = _higher_mode(suggested_mode, manual_mode)
        risk_score = max(risk_score, _score_floor_for_mode(manual_mode))

    if external_source_unavailable and external_ok_sources == 0:
        status = "source_unavailable"
        if manual_active:
            suggested_mode = _higher_mode("CAUTION", suggested_mode)
            risk_score = max(risk_score, _score_floor_for_mode(suggested_mode))
    elif source_unavailable:
        status = "partial"
    else:
        status = "ok"

    confidence = 0.40
    confidence += min(0.20, ok_sources * 0.04)
    confidence += min(0.25, len(top_events) * 0.035)
    confidence += min(0.15, risk_score * 0.15)
    if status == "source_unavailable":
        confidence = min(confidence, 0.45)
    if manual_active:
        confidence = max(confidence, 0.70)

    headline = top_events[0]["title"] if top_events else "no high-risk external event detected"
    if status == "source_unavailable":
        summary = "External sources unavailable; using manual fallback and conservative default."
    else:
        summary = f"{suggested_mode} from external event classifier; top signal: {headline}."
    if manual_active:
        summary += f" Manual override active: {manual_mode} ({manual_reason})."

    return {
        "timestamp": _utc_now_iso(),
        "mode": "SHADOW",
        "suggested_mode": suggested_mode,
        "risk_score": round(float(max(0.0, min(risk_score, 1.0))), 3),
        "confidence": round(float(max(0.0, min(confidence, 1.0))), 3),
        "top_events": top_events,
        "sources": source_summaries,
        "summary": summary,
        "status": status,
        "execute": False,
        "production_allowed": False,
        "does_not_change_trading": True,
        "does_not_change_event_risk_mode": True,
    }
