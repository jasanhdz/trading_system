"""Leakage-resistant counterfactual assessment for candidate meta-labels."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MetaEntryCounterfactual:
    event_id: str
    symbol: str
    selected: bool
    favorable: bool
    net_return: float
    mae: float
    probability: float
    percentile: float
    actual_trade: bool

    def __post_init__(self) -> None:
        if not self.event_id or not self.symbol:
            raise ValueError("meta-entry counterfactual identity is invalid")
        if not all(
            math.isfinite(value)
            for value in (
                self.net_return,
                self.mae,
                self.probability,
                self.percentile,
            )
        ):
            raise ValueError("meta-entry counterfactual contains non-finite values")
        if (
            self.mae < 0.0
            or not 0.0 <= self.probability <= 1.0
            or not 0.0 <= self.percentile <= 1.0
        ):
            raise ValueError("meta-entry counterfactual values are invalid")


def favorable_entry(
    *,
    net_return: float,
    mae: float,
    bad_entry: bool,
    maximum_mae: float,
) -> bool:
    if not all(math.isfinite(value) for value in (net_return, mae, maximum_mae)):
        raise ValueError("meta-entry label contains non-finite values")
    if mae < 0.0 or maximum_mae <= 0.0:
        raise ValueError("meta-entry MAE contract is invalid")
    return net_return > 0.0 and mae <= maximum_mae and not bad_entry


def _subset_metrics(
    rows: Sequence[MetaEntryCounterfactual],
) -> Mapping[str, float | int | None]:
    returns = [row.net_return for row in rows]
    return {
        "rows": len(rows),
        "favorable": sum(row.favorable for row in rows),
        "unfavorable": sum(not row.favorable for row in rows),
        "win_rate": (
            sum(value > 0.0 for value in returns) / len(returns)
            if returns
            else None
        ),
        "mean_net_return": (
            statistics.fmean(returns) if returns else None
        ),
        "mean_mae": (
            statistics.fmean(row.mae for row in rows) if rows else None
        ),
    }


def _confusion_metrics(
    rows: Sequence[MetaEntryCounterfactual],
) -> Mapping[str, float | int | None]:
    selected = [row for row in rows if row.selected]
    true_positive = sum(row.selected and row.favorable for row in rows)
    false_positive = sum(row.selected and not row.favorable for row in rows)
    false_negative = sum(not row.selected and row.favorable for row in rows)
    true_negative = sum(not row.selected and not row.favorable for row in rows)
    favorable = sum(row.favorable for row in rows)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (
            true_positive / len(selected) if selected else None
        ),
        "recall": true_positive / favorable if favorable else None,
    }


def assess_counterfactual_predictions(
    rows: Sequence[MetaEntryCounterfactual],
) -> Mapping[str, object]:
    selected = [row for row in rows if row.selected]
    rejected = [row for row in rows if not row.selected]
    actual = [row for row in rows if row.actual_trade]
    return {
        "population": _subset_metrics(rows),
        "selected": _subset_metrics(selected),
        "rejected": _subset_metrics(rejected),
        "confusion": _confusion_metrics(rows),
        "actual_trade_assessment": {
            "population": _subset_metrics(actual),
            "selected": _subset_metrics(
                [row for row in actual if row.selected]
            ),
            "rejected": _subset_metrics(
                [row for row in actual if not row.selected]
            ),
            "confusion": _confusion_metrics(actual),
        },
        "selected_symbol_counts": {
            symbol: sum(row.symbol == symbol for row in selected)
            for symbol in sorted({row.symbol for row in selected})
        },
        "exchange_mutations": 0,
    }


def counterfactual_mapping(
    row: MetaEntryCounterfactual,
) -> Mapping[str, object]:
    classification = (
        "TRUE_POSITIVE"
        if row.selected and row.favorable
        else "FALSE_POSITIVE"
        if row.selected
        else "FALSE_NEGATIVE"
        if row.favorable
        else "TRUE_NEGATIVE"
    )
    return {**asdict(row), "classification": classification}
