from __future__ import annotations


def signal_quality_summary(probs: list[dict[str, float]]) -> dict[str, float]:
    if not probs:
        return {"top_prob_avg": 0.0, "signals_gt_65_pct": 0.0, "long_short_gap_avg": 0.0}
    top = [max(p.values()) for p in probs]
    gaps = [abs(p.get("long", 0.0) - p.get("short", 0.0)) for p in probs]
    return {
        "top_prob_avg": sum(top) / len(top),
        "signals_gt_65_pct": sum(v >= 0.65 for v in top) / len(top),
        "long_short_gap_avg": sum(gaps) / len(gaps),
    }
