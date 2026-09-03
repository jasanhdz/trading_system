from dataclasses import replace
from datetime import timedelta

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import DeterministicFeaturePipeline, FEATURE_HASH, FEATURE_NAMES
from aegis.training import CausalDatasetBuilder, TrainingTarget, walk_forward_splits
from aegis.utils import Sha256HashProvider


def test_training_and_inference_use_identical_feature_values(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    pipeline = DeterministicFeaturePipeline()
    inference = pipeline.transform(snapshot)
    targets = {(snapshot.closed_at, symbol): TrainingTarget(0, 0, 0, 0, 0) for symbol in CANONICAL_SYMBOLS}
    dataset = CausalDatasetBuilder(pipeline, Sha256HashProvider(), CANONICAL_SYMBOLS, "5m").build("parity", (snapshot,), targets)
    by_symbol = {row.symbol: row.features for row in dataset.rows}
    assert inference.feature_hash == dataset.feature_hash == FEATURE_HASH
    assert inference.feature_names == FEATURE_NAMES
    for row in inference.rows:
        assert by_symbol[row.symbol] == row.normalized_values


def test_labels_and_future_outcomes_cannot_change_features(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    pipeline = DeterministicFeaturePipeline()
    before = pipeline.transform(snapshot)
    first_targets = {(snapshot.closed_at, symbol): TrainingTarget(-1, -0.5, 1, 0.9, 0) for symbol in CANONICAL_SYMBOLS}
    second_targets = {(snapshot.closed_at, symbol): TrainingTarget(1, 0.5, 0, 0.0, 1) for symbol in CANONICAL_SYMBOLS}
    builder = CausalDatasetBuilder(pipeline, Sha256HashProvider(), CANONICAL_SYMBOLS, "5m")
    first = builder.build("first", (snapshot,), first_targets)
    second = builder.build("second", (snapshot,), second_targets)
    assert tuple(row.features for row in first.rows) == tuple(row.features for row in second.rows)
    assert before.rows == pipeline.transform(snapshot).rows


def test_cross_section_uses_one_coordinated_cut_and_ties_receive_equal_rank(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    constant_series = []
    for index, series in enumerate(snapshot.series):
        price = 10.0 + index
        candles = tuple(replace(candle, open=price, high=price, low=price, close=price) for candle in series.candles)
        constant_series.append(replace(series, candles=candles))
    batch = DeterministicFeaturePipeline().transform(replace(snapshot, series=tuple(constant_series)))
    rank_index = batch.feature_names.index("cross_rank_return_6")
    assert {row.raw_values[rank_index] for row in batch.rows} == {0.5}
    assert tuple(row.symbol for row in batch.rows) == CANONICAL_SYMBOLS


def test_walk_forward_has_explicit_embargo_and_no_overlap(snapshot_factory) -> None:
    pipeline = DeterministicFeaturePipeline()
    start = snapshot_factory().closed_at
    snapshots = tuple(snapshot_factory(closed_at=start + timedelta(hours=index)) for index in range(20))
    targets = {(snapshot.closed_at, symbol): TrainingTarget(0, 0, 0, 0, 0)
               for snapshot in snapshots for symbol in CANONICAL_SYMBOLS}
    dataset = CausalDatasetBuilder(pipeline, Sha256HashProvider(), CANONICAL_SYMBOLS, "5m").build("embargo", snapshots, targets)
    for train, validation in walk_forward_splits(dataset, fold_count=4, embargo=timedelta(hours=2)):
        train_times = {dataset.rows[index].timestamp for index in train}
        validation_times = {dataset.rows[index].timestamp for index in validation}
        assert train_times.isdisjoint(validation_times)
        assert min(validation_times) - max(train_times) > timedelta(hours=2)
