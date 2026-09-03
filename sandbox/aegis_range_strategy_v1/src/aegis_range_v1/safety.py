from __future__ import annotations

from dataclasses import dataclass

from .candidates import RangeCandidate
from .models import RangePair, RegimeSnapshot


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    reason: str
    descriptive_score: float


class RangeSafetyV1:
    @staticmethod
    def descriptive_score(pair: RangePair, regime: RegimeSnapshot, candidate: RangeCandidate, episode_age_hours: float) -> float:
        clamp = lambda value: max(0.0, min(1.0, value))
        return (
            25.0 * min(1.0, min(pair.support_touches, pair.resistance_touches) / 4.0)
            + 20.0 * (1.0 - clamp(regime.adx / candidate.max_adx))
            + 20.0 * clamp((regime.chop_risk - candidate.min_chop_risk) / (1.0 - candidate.min_chop_risk))
            + 15.0 * (1.0 - clamp(regime.bollinger_width_percentile))
            + 10.0 * clamp(regime.volume_ratio / 1.5)
            + 10.0 * min(episode_age_hours / 24.0, 1.0)
        )

    @classmethod
    def evaluate(
        cls,
        pair: RangePair,
        regime: RegimeSnapshot,
        candidate: RangeCandidate,
        episode_age_hours: float,
        *,
        episode_operable: bool,
        flat: bool,
        no_pending_exit: bool,
        cooldown_ready: bool,
        quota_ready: bool,
    ) -> SafetyDecision:
        score = cls.descriptive_score(pair, regime, candidate, episode_age_hours)
        checks = (
            (regime.technical_regime in {"ACCUMULATION_RANGE", "CHOP"}, "REGIME_BLOCKED"),
            (regime.transition_risk != "HIGH", "TRANSITION_RISK_HIGH"),
            (regime.range_breakout == "NONE", "RANGE_BREAKOUT_ACTIVE"),
            (regime.adx <= candidate.max_adx, "ADX_BLOCKED"),
            (regime.chop_risk >= candidate.min_chop_risk, "CHOP_RISK_BLOCKED"),
            (regime.bollinger_width_percentile < 0.45, "BOLLINGER_WIDTH_BLOCKED"),
            (regime.atr_percentile <= 0.80, "ATR_PERCENTILE_BLOCKED"),
            (regime.volume_ratio >= candidate.min_safety_volume_ratio, "VOLUME_BLOCKED"),
            (episode_operable, "EPISODE_NOT_OPERABLE"),
            (flat, "POSITION_OPEN"),
            (no_pending_exit, "PENDING_EXIT"),
            (cooldown_ready, "COOLDOWN"),
            (quota_ready, "QUOTA"),
        )
        for passed, reason in checks:
            if not passed:
                return SafetyDecision(False, reason, score)
        return SafetyDecision(True, "ALLOWED", score)
