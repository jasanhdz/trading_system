from __future__ import annotations


def format_report(metrics: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(metrics.items()))
