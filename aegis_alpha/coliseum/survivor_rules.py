from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurvivorVerdict:
    passed: bool
    reasons: list[str]


def survivor_rules(balance: float, p95_dd: float, worst_dd: float, fees: float, opens: int) -> SurvivorVerdict:
    reasons: list[str] = []
    if balance < 21.0:
        reasons.append("balance_below_21")
    if p95_dd > 0.65:
        reasons.append("p95_dd_above_65")
    if worst_dd > 0.75:
        reasons.append("worst_dd_above_75")
    if fees > 1000.0:
        reasons.append("fees_too_high")
    if opens > 40000:
        reasons.append("overtrade")
    return SurvivorVerdict(passed=not reasons, reasons=reasons)
