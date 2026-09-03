from datetime import datetime, timedelta, timezone

from aegis.research.directional_challenger import (
    DirectionalEvidenceRow,
    DirectionalSelectionContract,
    derive_selection_policy,
    scoring_policy_passes,
    select_one_per_timestamp,
    selection_metrics,
    within_symbol_percentiles,
)


def _row(
    minute: int,
    symbol: str,
    score: float,
    net: float,
    *,
    mae: float = 0.001,
    regime: str = "BEAR_TREND",
) -> DirectionalEvidenceRow:
    return DirectionalEvidenceRow(
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute),
        symbol,
        score,
        net,
        mae,
        net < 0.0,
        regime,
    )


def _contract() -> DirectionalSelectionContract:
    return DirectionalSelectionContract(
        schema_version="aegis-directional-selection-contract-v1",
        probability_quantiles=(0.5, 0.75),
        minimum_calibration_selections=2,
        minimum_scoring_selections=2,
        maximum_mean_mae=0.01,
        maximum_symbol_concentration=0.75,
        bootstrap_resamples=200,
        bootstrap_seed=7,
        bootstrap_block_minutes=1,
        minimum_calibration_blocks=2,
        minimum_scoring_blocks=2,
    )


def test_percentiles_are_fitted_within_symbol_on_reference_only() -> None:
    reference = (
        _row(0, "BTCUSDT", 0.1, 0.0),
        _row(1, "BTCUSDT", 0.9, 0.0),
        _row(0, "ETHUSDT", 0.4, 0.0),
        _row(1, "ETHUSDT", 0.6, 0.0),
    )
    target = (
        _row(2, "BTCUSDT", 0.5, 0.0),
        _row(2, "ETHUSDT", 0.5, 0.0),
    )
    assert within_symbol_percentiles(reference, target) == (0.5, 0.5)


def test_selection_is_bounded_to_one_symbol_per_timestamp() -> None:
    rows = (
        _row(0, "BTCUSDT", 0.8, 0.01),
        _row(0, "ETHUSDT", 0.9, 0.02),
        _row(1, "BTCUSDT", 0.95, 0.03),
    )
    selected = select_one_per_timestamp(
        rows,
        (0.8, 0.9, 0.95),
        threshold=0.75,
    )
    assert selected == (1, 2)


def test_regime_variant_filters_without_changing_model_score() -> None:
    rows = (
        _row(0, "BTCUSDT", 0.9, 0.01, regime="BEAR_TREND"),
        _row(1, "ETHUSDT", 0.9, 0.01, regime="RANGE"),
    )
    selected = select_one_per_timestamp(
        rows,
        (0.9, 0.9),
        threshold=0.75,
        allowed_regimes=("BEAR_TREND",),
    )
    assert selected == (0,)


def test_threshold_derivation_rejects_abstention_and_uses_economics() -> None:
    rows = (
        _row(0, "BTCUSDT", 0.5, -0.01),
        _row(1, "ETHUSDT", 0.6, -0.01),
        _row(2, "BTCUSDT", 0.8, 0.02),
        _row(3, "ETHUSDT", 0.9, 0.02),
    )
    policy = derive_selection_policy(
        rows,
        (0.5, 0.5, 0.9, 0.9),
        _contract(),
    )
    assert policy.calibration_valid
    assert policy.threshold == 0.75
    assert policy.calibration_metrics.signals == 2
    assert policy.calibration_metrics.mean_net_expectancy == 0.02


def test_scoring_requires_positive_confidence_interval_not_only_mean() -> None:
    rows = tuple(
        _row(index, ("BTCUSDT", "ETHUSDT")[index % 2], 0.9, 0.01)
        for index in range(20)
    )
    metrics = selection_metrics(
        rows,
        tuple(range(len(rows))),
        bootstrap_resamples=200,
        bootstrap_seed=7,
        bootstrap_block_minutes=1,
    )
    assert scoring_policy_passes(metrics, _contract())

    mixed = tuple(
        _row(index, ("BTCUSDT", "ETHUSDT")[index % 2], 0.9, (-1) ** index * 0.01)
        for index in range(20)
    )
    mixed_metrics = selection_metrics(
        mixed,
        tuple(range(len(mixed))),
        bootstrap_resamples=200,
        bootstrap_seed=7,
        bootstrap_block_minutes=1,
    )
    assert not scoring_policy_passes(mixed_metrics, _contract())


def test_scoring_does_not_treat_overlapping_signals_as_independent() -> None:
    rows = tuple(
        _row(index, ("BTCUSDT", "ETHUSDT")[index % 2], 0.9, 0.01)
        for index in range(20)
    )
    metrics = selection_metrics(
        rows,
        tuple(range(len(rows))),
        bootstrap_resamples=200,
        bootstrap_seed=7,
        bootstrap_block_minutes=720,
    )

    assert metrics.signals == 20
    assert metrics.independent_blocks == 1
    assert metrics.expectancy_ci95_low is None
    assert not scoring_policy_passes(metrics, _contract())
