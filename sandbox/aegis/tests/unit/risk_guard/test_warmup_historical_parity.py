"""Warmup phase coverage and full-history E4 parity validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.live_api import _build_e4_config
from aegis.risk_guard.domain import FROZEN_TAIL_RISK_THRESHOLD
from aegis.risk_guard.e4_tail_risk_guard import E4TailRiskGuard
from aegis.risk_guard.feature_bridge import (
    ANCHOR_CADENCE_MINUTES,
    ANCHOR_COUNT,
    FROZEN_E4_TIMEFROZEN,
    FROZEN_E4_UNIVERSE,
)
from aegis.risk_guard.market_snapshot import (
    MAX_INDICATOR_WARMUP_BARS,
    WARMUP_MINUTES,
    _derive_minimum_warmup_minutes,
)
from sandbox.aegis_strategy_router.experiments.aegis_e4_robust_training.src.aegis_e4.features import (
    add_cross_market,
    assert_causal_availability,
    build_neutral_symbol_panel,
    orient_sides,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDLE_ROOT = REPO_ROOT / "data/independent_entry_quality_discovery_v1/candles_1m"
REFERENCE_PATH = REPO_ROOT / (
    "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/"
    "artifacts/dataset_v1/development_labeled.parquet"
)


def _load_candles() -> dict[str, pd.DataFrame]:
    return {
        symbol: pd.read_parquet(CANDLE_ROOT / f"{symbol}_1m.parquet")
        for symbol in sorted(FROZEN_E4_UNIVERSE)
    }


@pytest.fixture(scope="module")
def historical_inputs():
    guard = E4TailRiskGuard(_build_e4_config())
    guard.load()
    columns = ["decision_at", "symbol", "side", *guard.bridge.feature_names]
    return guard, _load_candles(), pd.read_parquet(REFERENCE_PATH, columns=columns)


def test_minimum_warmup_covers_all_48_phases():
    assert _derive_minimum_warmup_minutes() == 24_020
    assert WARMUP_MINUTES == 24_020

    timeframe = max(FROZEN_E4_TIMEFROZEN)
    lookback = ANCHOR_COUNT * ANCHOR_CADENCE_MINUTES
    base = pd.Timestamp("2023-11-07T12:00:00Z")
    counts = []
    for phase in range(0, timeframe, ANCHOR_CADENCE_MINUTES):
        decision_at = base + pd.Timedelta(minutes=phase)
        earliest_anchor = decision_at - pd.Timedelta(minutes=lookback)
        first_open = (decision_at - pd.Timedelta(minutes=WARMUP_MINUTES)).ceil(
            f"{timeframe}min"
        )
        last_close = earliest_anchor.floor(f"{timeframe}min")
        counts.append(int((last_close - first_open) / pd.Timedelta(minutes=timeframe)))
    assert len(counts) == 48
    assert min(counts) == MAX_INDICATOR_WARMUP_BARS


def test_full_history_phase_matrix_exact_parity(historical_inputs):
    guard, candles, reference = historical_inputs
    bases = [
        pd.Timestamp(f"{day}T12:00:00Z")
        for day in ("2023-09-15", "2023-11-07", "2023-12-05")
    ]
    decisions = pd.DatetimeIndex(
        [
            timestamp
            for base in bases
            for timestamp in pd.date_range(base, periods=48, freq="5min")
        ]
    )
    anchors = pd.DatetimeIndex(
        sorted(
            {
                timestamp
                for base in bases
                for timestamp in pd.date_range(
                    base - pd.Timedelta(minutes=25),
                    base + pd.Timedelta(minutes=235),
                    freq="5min",
                )
            }
        )
    )

    panels = []
    families: dict[str, str] = {}
    for symbol, frame in candles.items():
        causal = frame[frame.open_time_ms.lt(int(decisions.max().timestamp() * 1000))]
        panel, symbol_families = build_neutral_symbol_panel(
            causal, anchors, FROZEN_E4_TIMEFROZEN
        )
        panel["symbol"] = symbol
        panels.append(panel)
        families.update(symbol_families)
    combined, cross_families = add_cross_market(pd.concat(panels, ignore_index=True))
    families.update(cross_families)
    oriented, _ = orient_sides(combined, families)
    assert_causal_availability(oriented)
    actual = oriented[oriented.decision_at.isin(decisions)].sort_values(
        ["decision_at", "symbol", "side"]
    )
    expected = reference[reference.decision_at.isin(decisions)].sort_values(
        ["decision_at", "symbol", "side"]
    )

    assert len(actual) == len(expected) == 3 * 48 * 22
    actual_features = actual[guard.bridge.feature_names].to_numpy(float)
    expected_features = expected[guard.bridge.feature_names].to_numpy(float)
    assert actual_features.shape[1] == 146
    assert np.isfinite(actual_features).all()
    assert np.array_equal(actual_features, expected_features)

    model = guard._tail_bundle["model"]
    calibrator = guard._tail_bundle["calibrator"]
    actual_frame = pd.DataFrame(actual_features, columns=guard.bridge.feature_names)
    expected_frame = pd.DataFrame(expected_features, columns=guard.bridge.feature_names)
    actual_scores = calibrator.predict_proba(
        model.decision_function(actual_frame).reshape(-1, 1)
    )[:, 1]
    expected_scores = calibrator.predict_proba(
        model.decision_function(expected_frame).reshape(-1, 1)
    )[:, 1]
    print(
        {
            "cases": 3 * 48,
            "rows": len(actual),
            "max_abs_feature_error": float(
                np.max(np.abs(actual_features - expected_features))
            ),
            "features_differing": int(
                np.any(actual_features != expected_features, axis=0).sum()
            ),
            "max_abs_score_error": float(
                np.max(np.abs(actual_scores - expected_scores))
            ),
            "decision_matches": int(
                np.sum(
                    (actual_scores >= FROZEN_TAIL_RISK_THRESHOLD)
                    == (expected_scores >= FROZEN_TAIL_RISK_THRESHOLD)
                )
            ),
        }
    )
    assert np.array_equal(actual_scores, expected_scores)
    assert np.array_equal(
        actual_scores >= FROZEN_TAIL_RISK_THRESHOLD,
        expected_scores >= FROZEN_TAIL_RISK_THRESHOLD,
    )
