from datetime import datetime, timedelta, timezone

import pytest

from aegis.models import CalibrationMethod, CalibratorSpec
from aegis.research.entry_quality import (
    EntryQualityEvidenceRow,
    EntryQualityInputs,
    EntryQualityPromotionCriteria,
    HierarchicalProbabilityCalibrator,
    MaeAwareScoreContract,
    assess_entry_quality,
    score_entry_quality,
)


def _identity(sample_count: int = 100) -> CalibratorSpec:
    return CalibratorSpec(CalibrationMethod.IDENTITY, 0.1, 0.1, sample_count)


def _contract() -> MaeAwareScoreContract:
    return MaeAwareScoreContract(
        schema_version="aegis-mae-aware-score-contract-v1",
        qmae_penalty=0.5,
        tail_risk_penalty=0.001,
        maximum_qmae_fraction=0.02,
        maximum_tail_probability=0.8,
        require_bearish_trend_context=False,
    )


def _inputs(qmae: float) -> EntryQualityInputs:
    return EntryQualityInputs(
        symbol="XRPUSDT",
        expected_short_return=0.02,
        opportunity_probability=0.8,
        qmae_q90=qmae,
        tail_risk_probability=0.2,
        qmae_valid=True,
        calibration_valid=True,
    )


def test_mae_aware_score_penalizes_forecast_adverse_excursion() -> None:
    low = score_entry_quality(_inputs(0.005), _contract())
    high = score_entry_quality(_inputs(0.015), _contract())

    assert low.eligible and high.eligible
    assert low.score > high.score
    assert low.qmae_penalty < high.qmae_penalty


def test_mae_aware_score_fails_closed_above_explicit_contract() -> None:
    result = score_entry_quality(_inputs(0.03), _contract())
    assert not result.eligible
    assert "QMAE_LIMIT_EXCEEDED" in result.reason_codes


def test_symbol_calibration_is_shrunk_toward_global_probability() -> None:
    local = CalibratorSpec(
        CalibrationMethod.PLATT,
        0.1,
        0.1,
        20,
        parameters=(1.0, 1.0),
    )
    calibrator = HierarchicalProbabilityCalibrator(
        schema_version="aegis-hierarchical-symbol-calibration-v1",
        global_calibrator=_identity(),
        symbol_calibrators={"XRPUSDT": local},
        symbol_sample_counts={"XRPUSDT": 20},
        shrinkage_sample_count=80,
    )

    global_value = calibrator.apply("ETHUSDT", 0.5)
    local_value = local.apply(0.5)
    blended = calibrator.apply("XRPUSDT", 0.5)
    assert global_value < blended < local_value


def test_promotion_assessment_requires_monotonic_score_and_qmae() -> None:
    start = datetime(2026, 7, 24, tzinfo=timezone.utc)
    rows = []
    for index in range(40):
        score = index / 1000
        rows.append(
            EntryQualityEvidenceRow(
                start + timedelta(minutes=index),
                "XRPUSDT" if index % 2 else "SUIUSDT",
                score,
                0.02 - score / 2,
                score - 0.01,
                0.02 - score / 2,
            )
        )
    criteria = EntryQualityPromotionCriteria(
        schema_version="aegis-entry-quality-promotion-criteria-v1",
        minimum_rows=40,
        minimum_symbols=2,
        minimum_rows_per_evaluated_symbol=10,
        minimum_score_return_correlation=0.5,
        minimum_top_bottom_return_spread=0.01,
        minimum_qmae_mae_correlation=0.5,
        maximum_mean_mae=0.02,
        maximum_negative_symbol_fraction=0.5,
    )

    assessment = assess_entry_quality(rows, criteria)
    assert assessment.passed
    assert assessment.score_return_correlation == pytest.approx(1.0)
    assert assessment.qmae_mae_correlation == pytest.approx(1.0)
