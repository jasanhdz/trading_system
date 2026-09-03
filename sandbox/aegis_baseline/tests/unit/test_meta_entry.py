from datetime import datetime, timezone

import pytest

from aegis.research.directional_challenger import (
    DirectionalEvidenceRow,
    within_symbol_percentiles_with_global_fallback,
)
from aegis.research.meta_entry import (
    MetaEntryCounterfactual,
    assess_counterfactual_predictions,
    counterfactual_mapping,
    favorable_entry,
)


def _evidence(symbol: str, score: float) -> DirectionalEvidenceRow:
    return DirectionalEvidenceRow(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol=symbol,
        score=score,
        net_return=0.01,
        mae=0.001,
        bad_entry=False,
        regime="BEAR_TREND",
    )


def _prediction(
    identity: str,
    *,
    selected: bool,
    favorable: bool,
) -> MetaEntryCounterfactual:
    return MetaEntryCounterfactual(
        event_id=identity,
        symbol="BTCUSDT",
        selected=selected,
        favorable=favorable,
        net_return=0.01 if favorable else -0.01,
        mae=0.001,
        probability=0.8,
        percentile=0.9,
        actual_trade=identity == "actual",
    )


def test_favorable_entry_requires_profit_low_mae_and_no_bad_label() -> None:
    assert favorable_entry(
        net_return=0.01,
        mae=0.002,
        bad_entry=False,
        maximum_mae=0.00275,
    )
    assert not favorable_entry(
        net_return=-0.001,
        mae=0.001,
        bad_entry=False,
        maximum_mae=0.00275,
    )
    assert not favorable_entry(
        net_return=0.01,
        mae=0.003,
        bad_entry=False,
        maximum_mae=0.00275,
    )
    assert not favorable_entry(
        net_return=0.01,
        mae=0.001,
        bad_entry=True,
        maximum_mae=0.00275,
    )


def test_counterfactual_assessment_reports_all_confusion_classes() -> None:
    rows = (
        _prediction("tp", selected=True, favorable=True),
        _prediction("fp", selected=True, favorable=False),
        _prediction("fn", selected=False, favorable=True),
        _prediction("actual", selected=False, favorable=False),
    )
    report = assess_counterfactual_predictions(rows)
    assert report["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
    }
    assert report["actual_trade_assessment"]["population"]["rows"] == 1
    assert report["actual_trade_assessment"]["selected"]["rows"] == 0
    assert report["actual_trade_assessment"]["confusion"]["true_negative"] == 1
    assert report["exchange_mutations"] == 0
    assert counterfactual_mapping(rows[0])["classification"] == "TRUE_POSITIVE"


def test_missing_symbol_uses_held_out_global_percentile_reference() -> None:
    percentiles, fallbacks = within_symbol_percentiles_with_global_fallback(
        (_evidence("BTCUSDT", 0.2), _evidence("BTCUSDT", 0.8)),
        (_evidence("ETHUSDT", 0.5),),
    )
    assert percentiles == (0.5,)
    assert fallbacks == ("ETHUSDT",)


def test_counterfactual_rejects_non_finite_probability() -> None:
    with pytest.raises(ValueError):
        MetaEntryCounterfactual(
            event_id="bad",
            symbol="BTCUSDT",
            selected=False,
            favorable=False,
            net_return=0.0,
            mae=0.0,
            probability=float("nan"),
            percentile=0.0,
            actual_trade=False,
        )
