from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from aegis.training.dataset import TrainingDataset, TrainingRow, TrainingTarget
from aegis.training.hybrid_directional import (
    DirectionalSide,
    HybridDirectionalRow,
    fit_hybrid_directional_committee,
    hybrid_shadow_rank_score,
    load_hybrid_directional_artifact,
    paired_directional_rows,
    write_hybrid_directional_artifact,
)


def _dataset(side: DirectionalSide) -> TrainingDataset:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    terminal = 0.006 if side is DirectionalSide.LONG else -0.006
    row = TrainingRow(
        timestamp=timestamp,
        symbol="ADAUSDT",
        features=tuple(float(index == 0) for index in range(len(FEATURE_NAMES))),
        target=TrainingTarget(
            direction=side.sign,
            expected_return=terminal,
            tail_event=0.0,
            qmae=0.001,
            clean_quality=1.0,
            net_quality_after_costs=0.003,
            bad_entry=0.0,
        ),
    )
    return TrainingDataset(
        dataset_id=f"test-{side.value.lower()}",
        schema_version="aegis-training-dataset-v2",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        symbols=CANONICAL_SYMBOLS,
        timeframe="5m",
        rows=(row,),
        artifact_hash=side.value.lower() * 8,
    )


def test_pairing_preserves_directional_semantics_without_hindsight() -> None:
    rows = paired_directional_rows(
        _dataset(DirectionalSide.LONG),
        _dataset(DirectionalSide.SHORT),
        round_trip_cost_fraction=0.001,
    )

    assert len(rows) == 2
    assert {row.side for row in rows} == set(DirectionalSide)
    assert all(row.opportunity and row.clean_entry for row in rows)
    assert all(row.net_return_after_costs == pytest.approx(0.005) for row in rows)
    assert all(row.mfe_fraction == pytest.approx(0.005) for row in rows)


def _rows(start: datetime, count: int) -> tuple[HybridDirectionalRow, ...]:
    rows = []
    for index in range(count):
        timestamp = start + timedelta(minutes=5 * index)
        for side in DirectionalSide:
            positive = index % 2 == 0
            directional_signal = side.sign * (2.0 if positive else -2.0)
            features = [0.0] * len(FEATURE_NAMES)
            features[0] = directional_signal
            features[1] = float(index % 5) / 5.0
            rows.append(
                HybridDirectionalRow(
                    timestamp_ns=int(timestamp.timestamp() * 1_000_000_000),
                    symbol=CANONICAL_SYMBOLS[index % len(CANONICAL_SYMBOLS)],
                    side=side,
                    features=tuple(features),
                    opportunity=positive,
                    clean_entry=positive,
                    danger=not positive,
                    mae_fraction=0.001 if positive else 0.006,
                    mfe_fraction=0.008 if positive else 0.001,
                    net_return_after_costs=0.004 if positive else -0.004,
                )
            )
    return tuple(rows)


def _fit():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    train = _rows(start, 48)
    calibration = _rows(start + timedelta(days=2), 24)
    scoring = _rows(start + timedelta(days=4), 24)
    artifact = fit_hybrid_directional_committee(
        train,
        calibration,
        scoring,
        seed=17,
        embargo_minutes=120,
        round_trip_cost_fraction=0.001,
        classifier_parameters={
            "max_iter": 30,
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 2,
            "l2_regularization": 0.1,
            "early_stopping": False,
        },
        regressor_parameters={
            "max_iter": 30,
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 2,
            "l2_regularization": 0.1,
            "early_stopping": False,
        },
    )
    return artifact, scoring


def test_hybrid_fit_has_specialists_and_shared_directional_heads() -> None:
    artifact, scoring = _fit()
    positive_long = next(
        row for row in scoring if row.side is DirectionalSide.LONG and row.opportunity
    )
    negative_long = next(
        row
        for row in scoring
        if row.side is DirectionalSide.LONG and not row.opportunity
    )
    positive = artifact.predict(DirectionalSide.LONG, positive_long.features)
    negative = artifact.predict(DirectionalSide.LONG, negative_long.features)

    assert set(artifact.opportunity_heads) == set(DirectionalSide)
    assert positive.opportunity_probability > negative.opportunity_probability
    assert positive.danger_probability < negative.danger_probability
    assert positive.mae_q90 >= positive.mae_q50 >= 0.0
    assert positive.mfe_q50 >= 0.0
    assert positive.shadow_rank_score >= 0.0
    assert positive.selection_effect == "NONE"
    assert positive.exchange_authority is False
    assert positive.exchange_mutations == 0


def test_artifact_round_trip_preserves_predictions(tmp_path: Path) -> None:
    artifact, scoring = _fit()
    path = tmp_path / "hybrid.json"
    write_hybrid_directional_artifact(path, artifact)
    restored = load_hybrid_directional_artifact(path)
    row = scoring[0]

    assert restored.predict(row.side, row.features) == artifact.predict(
        row.side, row.features
    )


def test_fit_can_emit_an_observational_selection_trace() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    trace = []
    scoring = _rows(start + timedelta(days=4), 24)
    fit_hybrid_directional_committee(
        _rows(start, 48),
        _rows(start + timedelta(days=2), 24),
        scoring,
        seed=17,
        embargo_minutes=120,
        round_trip_cost_fraction=0.001,
        classifier_parameters={"max_iter": 10, "min_samples_leaf": 2},
        regressor_parameters={"max_iter": 10, "min_samples_leaf": 2},
        selection_trace=trace,
    )

    assert len(trace) == len({row.timestamp_ns for row in scoring}) * 2
    assert {item.side for item in trace} == set(DirectionalSide)
    assert all(item.symbol in CANONICAL_SYMBOLS for item in trace)


def test_temporal_overlap_is_rejected() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = _rows(start, 24)
    with pytest.raises(ValueError, match="embargo"):
        fit_hybrid_directional_committee(
            rows,
            rows,
            rows,
            seed=17,
            embargo_minutes=120,
            round_trip_cost_fraction=0.001,
            classifier_parameters={"max_iter": 2, "min_samples_leaf": 2},
            regressor_parameters={"max_iter": 2, "min_samples_leaf": 2},
        )


def test_shadow_rank_orders_better_path_without_becoming_a_guard() -> None:
    strong = hybrid_shadow_rank_score(
        opportunity_probability=0.7,
        danger_probability=0.2,
        mae_q90=0.002,
        mfe_q50=0.008,
        net_return_mean=0.002,
        round_trip_cost_fraction=0.001,
    )
    weak = hybrid_shadow_rank_score(
        opportunity_probability=0.7,
        danger_probability=0.6,
        mae_q90=0.008,
        mfe_q50=0.002,
        net_return_mean=-0.002,
        round_trip_cost_fraction=0.001,
    )

    assert strong > weak > 0.0


def test_module_has_no_exchange_mutation_surface() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/aegis/training/hybrid_directional.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "create_order",
        "cancel_order",
        "modify_order",
        "close_position",
        "BinanceAdapter",
        "api_secret",
    ):
        assert forbidden not in source
