from __future__ import annotations


def survivor_utility(net_profit: float, p95_dd: float, regime_consistency: float, fees: float, dominance: float, signal_bonus: float) -> float:
    fee_penalty = max(0.0, fees - 500.0) / 500.0
    dominance_penalty = max(0.0, dominance - 0.90) * 5.0
    return net_profit * max(1.0 - p95_dd, 0.001) ** 3 * regime_consistency - fee_penalty - dominance_penalty + signal_bonus
