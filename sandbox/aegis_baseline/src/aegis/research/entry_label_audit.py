"""Economic alignment diagnostics for existing entry labels.

The audit is descriptive by design. It does not select thresholds, fit models,
or access a future holdout.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


class EntryLabelAuditError(ValueError):
    """Raised when label evidence is incomplete or non-finite."""


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EntryLabelAuditError(f"non-finite label audit value: {name}")
    return result


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def normalize_entry_label_row(source: Mapping[str, Any]) -> dict[str, Any]:
    contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    diagnostics = source["v11_path_diagnostics"]
    protection = source["protection_profiles"]["CURRENT_TS"]
    event_raw = contract.get("event_bar")
    return {
        "timestamp": str(source["timestamp"]),
        "month": str(source["timestamp"])[:7],
        "symbol": str(source["symbol"]),
        "side": str(source["side"]),
        "regime": str(source["v11_causal_regime"]),
        "independent": bool(source["independent"]),
        "v11_clean": bool(source["v11_clean_entry_label"]),
        "clean_fast_success": bool(source["clean_fast_success"]),
        "target_before_stop": bool(source["target_before_stop"]),
        "positive_utility": _finite(contract["realized_utility"], "utility") > 0.0,
        "current_ts_positive": _finite(protection["worst_net_return"], "protected_return")
        > 0.0,
        "utility": _finite(contract["realized_utility"], "utility"),
        "protected_return": _finite(protection["worst_net_return"], "protected_return"),
        "mae": _finite(source["mae_fraction"], "mae"),
        "mfe": _finite(source["mfe_fraction"], "mfe"),
        "time_underwater_bars": _finite(
            source["time_underwater_bars"], "time_underwater_bars"
        ),
        "outcome": str(contract["outcome"]),
        "event_bar": int(event_raw) if event_raw is not None else None,
        "mae_barrier_ratio": _finite(
            diagnostics["pre_event_mae_as_adverse_barrier_fraction"],
            "mae_barrier_ratio",
        ),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    utility = [_finite(row["utility"], "utility") for row in rows]
    protected = [_finite(row["protected_return"], "protected_return") for row in rows]
    mae = [_finite(row["mae"], "mae") for row in rows]
    mfe = [_finite(row["mfe"], "mfe") for row in rows]
    underwater = [_finite(row["time_underwater_bars"], "time_underwater") for row in rows]
    return {
        "rows": len(rows),
        "mean_utility": mean(utility),
        "positive_utility_rate": mean(bool(row["positive_utility"]) for row in rows),
        "mean_current_ts_protected_return": mean(protected),
        "current_ts_positive_rate": mean(bool(row["current_ts_positive"]) for row in rows),
        "mean_mae": mean(mae),
        "p90_mae": _quantile(mae, 0.90),
        "mean_mfe": mean(mfe),
        "mean_time_underwater_bars": mean(underwater),
    }


def _label_alignment(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if bool(row[label])]
    positive = [row for row in rows if bool(row["positive_utility"])]
    selected_positive = [row for row in selected if bool(row["positive_utility"])]
    overall = _summary(rows)
    result = {
        "label": label,
        "prevalence": len(selected) / len(rows) if rows else 0.0,
        "selected": _summary(selected),
        "unselected": _summary([row for row in rows if not bool(row[label])]),
        "positive_utility_recall": len(selected_positive) / len(positive) if positive else None,
    }
    selected_mean = result["selected"].get("mean_utility")
    result["mean_utility_lift_vs_population"] = (
        selected_mean - overall["mean_utility"] if selected_mean is not None and rows else None
    )
    return result


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            **_summary(group_rows),
            "v11_clean_rate": mean(bool(row["v11_clean"]) for row in group_rows),
            "target_before_stop_rate": mean(
                bool(row["target_before_stop"]) for row in group_rows
            ),
        }
        for key, group_rows in sorted(groups.items())
    }


def audit_entry_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    v18_clean_average_precision: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    normalized = [normalize_entry_label_row(row) for row in rows]
    independent = [row for row in normalized if row["independent"]]
    if not independent:
        raise EntryLabelAuditError("entry label audit has no independent episodes")

    exclusions = Counter()
    for row in independent:
        favorable = row["outcome"] == "FAVORABLE_FIRST"
        timely = row["event_bar"] is not None and row["event_bar"] <= 6
        low_mae = row["mae_barrier_ratio"] <= 0.50
        if not favorable:
            exclusions["NOT_FAVORABLE_FIRST"] += 1
        if favorable and not timely:
            exclusions["FAVORABLE_BUT_AFTER_SIX_BARS"] += 1
        if favorable and timely and not low_mae:
            exclusions["TIMELY_FAVORABLE_BUT_PRE_EVENT_MAE_ABOVE_HALF_BARRIER"] += 1

    labels = [
        "v11_clean",
        "clean_fast_success",
        "target_before_stop",
        "positive_utility",
        "current_ts_positive",
    ]
    separability: dict[str, Any] = {}
    if v18_clean_average_precision:
        for side, average_precision in sorted(v18_clean_average_precision.items()):
            side_rows = [row for row in independent if row["side"] == side]
            prevalence = mean(bool(row["v11_clean"]) for row in side_rows)
            ap = _finite(average_precision, f"{side}_average_precision")
            separability[side] = {
                "prevalence": prevalence,
                "v18_average_precision": ap,
                "average_precision_lift_ratio": ap / prevalence if prevalence else None,
                "average_precision_absolute_lift": ap - prevalence,
            }

    return {
        "schema_id": "aegis-entry-label-economic-audit-v1",
        "selection_effect": "NONE",
        "holdout_accesses": 0,
        "source_rows": len(normalized),
        "independent_rows": len(independent),
        "population": _summary(independent),
        "label_alignment": {
            label: _label_alignment(independent, label) for label in labels
        },
        "v11_clean_exclusion_counts": dict(sorted(exclusions.items())),
        "by_side": _group(independent, "side"),
        "by_symbol": _group(independent, "symbol"),
        "by_regime": _group(independent, "regime"),
        "by_month": _group(independent, "month"),
        "v18_clean_label_separability": separability,
    }
