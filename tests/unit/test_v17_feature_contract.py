from dataclasses import replace
from datetime import timedelta

import pytest

from aegis.features import FEATURE_NAMES, DeterministicFeaturePipeline
from aegis.v17_feature_contract import (
    V17_DTYPE,
    V17_FEATURE_SCHEMA,
    V17FeatureContractError,
    build_v17_runtime_features,
    contract_for_side,
)


def test_contract_freezes_directional_names_hashes_and_normalizers() -> None:
    long = contract_for_side("LONG")
    short = contract_for_side("SHORT")
    assert long["feature_count"] == 129
    assert short["feature_count"] == 168
    assert len(long["schema_hash"]) == len(short["schema_hash"]) == 64
    assert long["schema_hash"] != short["schema_hash"]
    assert long["normalization"]["pairwise_ranker"] == "FITTED_STANDARD_SCALER"


def test_runtime_reconstructs_the_exact_causal_v17_contract(snapshot_factory) -> None:
    snapshot = snapshot_factory(bars=576)
    batch = DeterministicFeaturePipeline().transform(snapshot)
    series = {item.symbol: item for item in snapshot.series}
    for row in batch.rows:
        base = dict(zip(batch.feature_names, row.raw_values))
        vector = build_v17_runtime_features(
            side="LONG",
            base_features=base,
            history=series[row.symbol].candles,
        )
        assert vector.names == contract_for_side("LONG")["feature_names"]
        assert len(vector.values) == 129
        with pytest.raises(V17FeatureContractError, match="VERSION"):
            replace(vector, schema_version="wrong").validate()
        with pytest.raises(V17FeatureContractError, match="HASH"):
            replace(vector, schema_hash="0" * 64).validate()
        with pytest.raises(V17FeatureContractError, match="ORDER"):
            replace(vector, names=tuple(reversed(vector.names))).validate()
        with pytest.raises(V17FeatureContractError, match="DTYPE"):
            replace(vector, dtype="float32").validate()
        assert vector.schema_version == V17_FEATURE_SCHEMA
        assert vector.dtype == V17_DTYPE
        break


def test_runtime_fails_closed_on_missing_extra_order_dtype_and_history(snapshot_factory) -> None:
    snapshot = snapshot_factory(bars=576)
    batch = DeterministicFeaturePipeline().transform(snapshot)
    row = batch.rows[0]
    history = snapshot.series[0].candles
    base = dict(zip(batch.feature_names, row.raw_values))

    missing = dict(base)
    missing.pop(FEATURE_NAMES[0])
    with pytest.raises(V17FeatureContractError, match="MISSING"):
        build_v17_runtime_features(side="SHORT", base_features=missing, history=history)

    extra = {**base, "unexpected": 1.0}
    with pytest.raises(V17FeatureContractError, match="EXTRA"):
        build_v17_runtime_features(side="SHORT", base_features=extra, history=history)

    reordered = {name: base[name] for name in reversed(FEATURE_NAMES)}
    with pytest.raises(V17FeatureContractError, match="ORDER"):
        build_v17_runtime_features(side="SHORT", base_features=reordered, history=history)

    invalid = dict(base)
    invalid[FEATURE_NAMES[0]] = "not-a-number"
    with pytest.raises(V17FeatureContractError, match="DTYPE"):
        build_v17_runtime_features(side="SHORT", base_features=invalid, history=history)

    with pytest.raises(V17FeatureContractError, match="HISTORY"):
        build_v17_runtime_features(side="SHORT", base_features=base, history=history[-575:])

    open_history = (*history[:-1], replace(history[-1], is_closed=False))
    with pytest.raises(V17FeatureContractError, match="HISTORY"):
        build_v17_runtime_features(side="SHORT", base_features=base, history=open_history)
