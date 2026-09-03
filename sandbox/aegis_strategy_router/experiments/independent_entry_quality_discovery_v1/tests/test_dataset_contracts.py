from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aegis_strategy_router.domain.types import Side, Timeframe
from independent_entry_quality_v1.dataset import _targets, split_for
from independent_entry_quality_v1.features import assert_feature_allowlist, feature_hash


EXPERIMENT = Path(__file__).resolve().parents[1]
DATASET = EXPERIMENT / "artifacts/dataset_v1"


def test_split_boundaries_preserve_embargo_and_holdout() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    assert split_for(pd.Timestamp("2023-10-16T12:00:00Z"), config["splits"]) == "EMBARGO_1"
    assert split_for(pd.Timestamp("2023-11-07T00:00:00Z"), config["splits"]) == "VALIDATION"
    assert split_for(pd.Timestamp("2023-12-07T00:00:00Z"), config["splits"]) == "FINAL_HOLDOUT"
    contaminated = pd.Timestamp(config["contamination"]["strategy_router_v1_discovery_contaminated"][0])
    assert pd.Timestamp(config["source"]["end_exclusive"]) <= contaminated


def test_allowlist_rejects_future_or_target_columns() -> None:
    assert_feature_allowlist(["feature__tf15m__directional_return_1_bps"])
    with pytest.raises(ValueError, match="LEAKAGE_FEATURES"):
        assert_feature_allowlist(["feature__future_mfe"])


def test_built_dataset_is_symmetric_causal_and_holdout_is_unlabeled() -> None:
    development = pd.read_parquet(DATASET / "development_labeled.parquet")
    holdout = pd.read_parquet(DATASET / "final_holdout_features_sealed.parquet")
    assert not any(column.startswith("target__") for column in holdout)
    assert holdout.label_state.eq("SEALED").all()
    side_sets = set(
        development.groupby("market_state_group_id").side.apply(lambda values: frozenset(values))
    )
    assert side_sets == {frozenset({"LONG", "SHORT"})}
    assert (pd.to_datetime(development.max_feature_available_at, utc=True) <= pd.to_datetime(development.decision_at, utc=True)).all()
    forbidden = ("aegis", "candidate_strategy", "committee", "confidence")
    assert not [column for column in development if any(token in column.lower() for token in forbidden)]


def test_feature_hash_is_reproducible() -> None:
    development = pd.read_parquet(DATASET / "development_labeled.parquet").head(50)
    feature_columns = [column for column in development if column.startswith("feature__")]
    expected = [feature_hash(row) for row in development[feature_columns].to_dict("records")]
    assert expected == development.feature_values_hash.tolist()


def test_same_bar_barrier_ambiguity_is_adverse_first_for_both_sides() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    decision = pd.Timestamp("2023-09-01T00:00:00Z")
    frame = _future_frame(decision, high=np.full(60, 101.0), low=np.full(60, 99.0))
    snapshot = _snapshot(decision)
    assert _targets(frame, snapshot, Side.LONG, config)["target__path_label"] == "ADVERSE_FIRST"
    assert _targets(frame, snapshot, Side.SHORT, config)["target__path_label"] == "ADVERSE_FIRST"


def test_mirrored_paths_produce_symmetric_long_short_targets() -> None:
    config = json.loads((EXPERIMENT / "config/preregistration_v1.json").read_text())
    decision = pd.Timestamp("2023-09-01T00:00:00Z")
    up = np.linspace(100.1, 101.0, 60)
    long_frame = _future_frame(decision, high=up, low=np.full(60, 99.9), close=up)
    short_frame = _future_frame(decision, high=np.full(60, 100.1), low=200.0 - up, close=200.0 - up)
    long = _targets(long_frame, _snapshot(decision), Side.LONG, config)
    short = _targets(short_frame, _snapshot(decision), Side.SHORT, config)
    comparable = (
        "target__path_label", "target__favorable_first", "target__adverse_first",
        "target__mfe_bps", "target__mae_bps", "target__gross_common_payoff_bps",
        "target__net_common_payoff_bps",
    )
    for name in comparable:
        assert short[name] == pytest.approx(long[name]) if isinstance(long[name], float) else short[name] == long[name]


def _snapshot(decision: pd.Timestamp) -> SimpleNamespace:
    state = SimpleNamespace(timeframe=Timeframe.M15, structural=SimpleNamespace(atr14=1.0))
    return SimpleNamespace(decision_at=decision.to_pydatetime(), reference_price=100.0, timeframes=(state,))


def _future_frame(
    decision: pd.Timestamp, *, high: np.ndarray, low: np.ndarray, close: np.ndarray | None = None
) -> pd.DataFrame:
    close = np.full(60, 100.0) if close is None else close
    return pd.DataFrame({
        "open_time_ms": [int((decision + pd.Timedelta(minutes=index)).timestamp() * 1_000) for index in range(60)],
        "open": np.full(60, 100.0), "high": high, "low": low, "close": close,
    })
