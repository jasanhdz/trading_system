import json
import math
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from aegis.features import (
    DeterministicFeaturePipeline, FEATURE_HASH, FEATURE_NAMES, FEATURE_NAMES_V1,
    FEATURE_NAMES_V2_ADDITIONS, FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V1,
    feature_contract,
)


GOLDEN = Path(__file__).parents[1] / "fixtures" / "aegis_features_v2_golden.json"


def test_v1_and_v2_are_separate_hash_pinned_contracts() -> None:
    v1_names, v1_hash = feature_contract(FEATURE_SCHEMA_VERSION_V1)
    assert v1_names == FEATURE_NAMES_V1 and len(v1_names) == 39
    assert len(FEATURE_NAMES) == 83 and len(FEATURE_NAMES_V2_ADDITIONS) == 44
    assert FEATURE_HASH != v1_hash
    assert DeterministicFeaturePipeline(schema_version=FEATURE_SCHEMA_VERSION).feature_hash == FEATURE_HASH
    assert DeterministicFeaturePipeline(schema_version=FEATURE_SCHEMA_VERSION_V1).feature_hash == v1_hash


def test_every_ported_v2_feature_matches_versioned_golden(snapshot_factory) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    batch = DeterministicFeaturePipeline().transform(snapshot_factory())
    assert batch.schema_version == golden["feature_schema_version"]
    assert batch.feature_hash == golden["feature_hash"]
    values = dict(zip(batch.feature_names, batch.row_for(golden["symbol"]).raw_values))
    assert set(golden["values"]) == set(FEATURE_NAMES_V2_ADDITIONS)
    for name, expected in golden["values"].items():
        assert math.isclose(values[name], expected, rel_tol=0.0, abs_tol=golden["tolerance"]), name


def test_v2_uses_historical_momentum_acceleration_semantics(snapshot_factory) -> None:
    batch = DeterministicFeaturePipeline().transform(snapshot_factory())
    row = batch.row_for("BTCUSDT")
    values = dict(zip(batch.feature_names, row.raw_values))
    assert values["momentum_acceleration_3_12"] == values["ret_3"] - values["ret_12"]


def test_future_candle_cannot_change_features_at_previous_cut(snapshot_factory) -> None:
    current = snapshot_factory()
    extended = snapshot_factory(bars=61, closed_at=current.closed_at + timedelta(minutes=5))
    truncated_series = tuple(
        replace(series, candles=series.candles[:-1], last_confirmed_close=current.closed_at)
        for series in extended.series
    )
    reconstructed = replace(extended, closed_at=current.closed_at, series=truncated_series)
    pipeline = DeterministicFeaturePipeline()
    assert pipeline.transform(current) == pipeline.transform(reconstructed)
