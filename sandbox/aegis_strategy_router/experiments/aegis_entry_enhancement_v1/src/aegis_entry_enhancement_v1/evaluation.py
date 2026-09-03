"""Frozen deterministic policy evaluation for Aegis Entry Enhancement V1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


POLICIES = (
    "AEGIS_ONLY", "AEGIS_OPPORTUNITY_GATE", "AEGIS_CROSS_MARKET_CONFIRMATION",
    "AEGIS_OPPORTUNITY_CONFLICT_GATE",
)


def policy_masks(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    opportunity = frame.opportunity_score.ge(config["frozen_modules"]["opportunity"]["accept_threshold"])
    margin = config["frozen_modules"]["directional"]["conflict_margin_bps"]
    no_clear_conflict = frame.predicted_aegis_advantage_bps.gt(-margin)
    strong_confirmation = (
        frame.predicted_net_aegis_side_bps.gt(config["frozen_modules"]["directional"]["minimum_aegis_side_predicted_net_bps"])
        & frame.predicted_aegis_advantage_bps.ge(margin) & ~frame.ood
    )
    return {
        "AEGIS_ONLY": pd.Series(True, index=frame.index),
        "AEGIS_OPPORTUNITY_GATE": opportunity,
        "AEGIS_CROSS_MARKET_CONFIRMATION": no_clear_conflict,
        "AEGIS_OPPORTUNITY_CONFLICT_GATE": opportunity & strong_confirmation,
    }


def classify(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series("MIXED", index=frame.index)
    good = frame.target__favorable_first.eq(1) & frame.target__net_common_payoff_bps.gt(0)
    bad = frame.target__adverse_first.eq(1) | frame.target__net_common_payoff_bps.lt(0)
    result.loc[good] = "GOOD"
    result.loc[bad] = "BAD"
    return result


def max_drawdown(values: pd.Series) -> float:
    cumulative = values.cumsum().to_numpy(float)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return float(np.max(peak - cumulative)) if len(cumulative) else 0.0


def metrics(frame: pd.DataFrame, accepted: pd.Series, policy: str) -> dict[str, Any]:
    selected = frame.loc[accepted]
    outcomes = frame.target__net_common_payoff_bps.where(accepted, 0.0)
    tail_count = max(1, math.ceil(len(selected) * 0.05)) if len(selected) else 0
    classes = classify(frame)
    rejected = ~accepted
    good, bad = classes.eq("GOOD"), classes.eq("BAD")
    return {
        "policy": policy, "signals": len(frame), "executed": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "gross_bps_per_original_signal": float(frame.target__gross_common_payoff_bps.where(accepted, 0.0).mean()),
        "net_bps_per_original_signal": float(outcomes.mean()),
        "gross_bps_per_executed": float(selected.target__gross_common_payoff_bps.mean()) if len(selected) else math.nan,
        "net_bps_per_executed": float(selected.target__net_common_payoff_bps.mean()) if len(selected) else math.nan,
        "favorable_first": float(selected.target__favorable_first.mean()) if len(selected) else math.nan,
        "mfe_bps": float(selected.target__mfe_bps.mean()) if len(selected) else math.nan,
        "mae_bps": float(selected.target__mae_bps.mean()) if len(selected) else math.nan,
        "mfe_mae_ratio": float(selected.target__mfe_bps.mean() / selected.target__mae_bps.mean()) if len(selected) and selected.target__mae_bps.mean() else math.nan,
        "mfe_minus_mae_bps": float((selected.target__mfe_bps - selected.target__mae_bps).mean()) if len(selected) else math.nan,
        "tail_mae_bps": float(selected.target__mae_bps.nlargest(tail_count).mean()) if tail_count else math.nan,
        "expected_shortfall_net_bps": float(selected.target__net_common_payoff_bps.nsmallest(tail_count).mean()) if tail_count else math.nan,
        "drawdown_proxy_bps": max_drawdown(outcomes), "turnover": int(accepted.sum()),
        "estimated_total_cost_bps": float(accepted.sum() * 20.0),
        "wins": int(selected.target__net_common_payoff_bps.gt(0).sum()),
        "losses": int(selected.target__net_common_payoff_bps.lt(0).sum()),
        "bad_rejected": int((bad & rejected).sum()), "good_rejected": int((good & rejected).sum()),
        "bad_trade_rejection_rate": float((bad & rejected).sum() / max(1, bad.sum())),
        "good_trade_destruction_rate": float((good & rejected).sum() / max(1, good.sum())),
        "rejection_precision": float((bad & rejected).sum() / max(1, rejected.sum())),
        "bad_prevalence": float(bad.mean()),
        "net_value_removed_bps": float(frame.loc[rejected, "target__net_common_payoff_bps"].sum()),
        "net_value_preserved_bps": float(frame.loc[accepted, "target__net_common_payoff_bps"].sum()),
    }


def bootstrap_delta(frame: pd.DataFrame, accepted: pd.Series, config: dict[str, Any]) -> tuple[float, float]:
    values = frame.copy()
    values["increment"] = values.target__net_common_payoff_bps.where(accepted, 0.0) - values.target__net_common_payoff_bps
    values["day"] = values.signal_timestamp.dt.floor("D")
    groups = [group.increment.to_numpy(float) for _, group in values.groupby("day")]
    rng = np.random.default_rng(config["statistics"]["seed"])
    samples = np.empty(config["statistics"]["block_bootstrap_samples"])
    for index in range(len(samples)):
        chosen = rng.integers(0, len(groups), len(groups))
        samples[index] = np.concatenate([groups[value] for value in chosen]).mean()
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def coverage_curve(frame: pd.DataFrame, config: dict[str, Any], subgroup: str) -> list[dict[str, Any]]:
    if subgroup == "AEGIS_SHORT":
        frame = frame.loc[frame.side.eq("SHORT")]
    elif subgroup == "AEGIS_LONG":
        frame = frame.loc[frame.side.eq("LONG")]
    ordered = frame.sort_values(["quality_score", "signal_timestamp", "trade_id"], ascending=[False, True, True], kind="mergesort")
    rows = []
    for coverage in config["ranking"]["coverage_levels"]:
        accepted_ids = set(ordered.head(max(1, math.ceil(len(ordered) * coverage))).index)
        accepted = pd.Series(frame.index.isin(accepted_ids), index=frame.index)
        rows.append({"subgroup": subgroup, "coverage_target": coverage, **metrics(frame, accepted, "QUALITY_RANKING")})
    return rows


def evaluate(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    frame = frame.copy()
    frame["signal_timestamp"] = pd.to_datetime(frame.signal_timestamp, utc=True)
    masks = policy_masks(frame, config)
    policy_rows = []
    for name, mask in masks.items():
        row = metrics(frame, mask, name)
        lower, upper = bootstrap_delta(frame, mask, config)
        row.update({"delta_vs_aegis_bps": row["net_bps_per_original_signal"] - metrics(frame, masks["AEGIS_ONLY"], "AEGIS_ONLY")["net_bps_per_original_signal"], "delta_ci_lower_bps": lower, "delta_ci_upper_bps": upper})
        policy_rows.append(row)
    coverage_rows = [row for subgroup in config["subgroups"] for row in coverage_curve(frame, config, subgroup)]
    return {"policy_rows": policy_rows, "coverage_rows": coverage_rows, "masks": masks}


def monotonic_ranking(rows: list[dict[str, Any]], subgroup: str = "ALL_AEGIS_SIGNALS") -> tuple[bool, float]:
    values = pd.DataFrame([row for row in rows if row["subgroup"] == subgroup]).sort_values("coverage_target", ascending=False)
    coefficient = float(spearmanr(-values.coverage_target, values.net_bps_per_executed).statistic)
    return bool(values.net_bps_per_executed.is_monotonic_increasing), coefficient
