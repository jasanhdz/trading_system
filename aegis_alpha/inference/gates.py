from __future__ import annotations

from aegis_alpha.config import SignalGateConfig


def gate_action(
    raw_action: str,
    top_prob: float,
    long_short_gap: float,
    regime_type: str,
    cfg: SignalGateConfig,
) -> tuple[str, str, bool]:
    min_top = cfg.chop_min_top_prob if regime_type == "chop" else cfg.min_top_prob
    min_gap = cfg.chop_min_gap if regime_type == "chop" else cfg.min_long_short_gap
    if raw_action == "IDLE":
        return "IDLE", "raw_idle", False
    if top_prob < min_top:
        return "IDLE", "low_top_prob", False
    if long_short_gap < min_gap:
        return "IDLE", "low_long_short_gap", False
    return raw_action, "passed", True
