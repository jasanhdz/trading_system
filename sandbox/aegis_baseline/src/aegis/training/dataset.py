"""Causal offline datasets built with the production feature pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence

from ..config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from ..data import CanonicalBar, CanonicalSeriesSource
from ..domain import Candle, FeedQuality, MarketSnapshot, PortfolioContext, Regime, SymbolSeries
from ..features import DeterministicFeaturePipeline, FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeaturePipeline
from ..layers import classify_market_regime
from ..utils import HashProvider, canonical_json
from .labels import (
    LONG_LABEL_SCHEMA_VERSION,
    SHORT_LABEL_SCHEMA_VERSION,
    LongLabelConfig,
    ShortLabelConfig,
    build_long_path_label,
    build_short_path_label,
)


@dataclass(frozen=True)
class TrainingTarget:
    direction: float
    expected_return: float
    tail_event: float
    qmae: float
    clean_quality: float
    net_quality_after_costs: float = 0.0
    bad_entry: float = 0.0
    label_valid: bool = True


@dataclass(frozen=True)
class TrainingRow:
    timestamp: datetime
    symbol: str
    features: tuple[float, ...]
    target: TrainingTarget
    regime: Regime = Regime.UNKNOWN


@dataclass(frozen=True)
class TrainingDataset:
    dataset_id: str
    schema_version: str
    feature_schema_version: str
    feature_hash: str
    symbols: tuple[str, ...]
    timeframe: str
    rows: tuple[TrainingRow, ...]
    artifact_hash: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


class DatasetBuilder(Protocol):
    def build(self, dataset_id: str, snapshots: Sequence[MarketSnapshot], targets: Mapping[tuple[datetime, str], TrainingTarget]) -> TrainingDataset: ...


@dataclass(frozen=True)
class CausalDatasetBuilder:
    feature_pipeline: FeaturePipeline
    hashing: HashProvider
    symbols: tuple[str, ...]
    timeframe: str

    def build(self, dataset_id: str, snapshots: Sequence[MarketSnapshot], targets: Mapping[tuple[datetime, str], TrainingTarget]) -> TrainingDataset:
        rows: list[TrainingRow] = []
        for snapshot in sorted(snapshots, key=lambda item: item.closed_at):
            batch = self.feature_pipeline.transform(snapshot)
            for feature_row in batch.rows:
                key = (snapshot.closed_at, feature_row.symbol)
                if key not in targets:
                    continue
                rows.append(TrainingRow(snapshot.closed_at, feature_row.symbol, feature_row.normalized_values, targets[key]))
        if not rows:
            raise ValueError("training dataset has no aligned causal targets")
        payload = {"dataset_id": dataset_id, "feature_hash": self.feature_pipeline.feature_hash, "rows": rows}
        return TrainingDataset(dataset_id, "aegis-training-dataset-v1", self.feature_pipeline.schema_version,
                               self.feature_pipeline.feature_hash, self.symbols, self.timeframe,
                               tuple(rows), self.hashing.digest_value(payload))


def walk_forward_splits(dataset: TrainingDataset, fold_count: int = 4, embargo: timedelta = timedelta(minutes=120)) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Expanding temporal folds; no row after validation enters training."""
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    timestamps = sorted({row.timestamp for row in dataset.rows})
    if len(timestamps) < fold_count + 2:
        raise ValueError("insufficient timestamps for walk-forward evaluation")
    folds = []
    for fold in range(fold_count):
        train_fraction = 0.50 + fold * (0.30 / max(1, fold_count - 1))
        validation_fraction = min(0.90, train_fraction + 0.10)
        train_end = timestamps[min(len(timestamps) - 2, int(len(timestamps) * train_fraction))]
        validation_end = timestamps[min(len(timestamps) - 1, int(len(timestamps) * validation_fraction))]
        train = tuple(index for index, row in enumerate(dataset.rows) if row.timestamp <= train_end - embargo)
        validation = tuple(index for index, row in enumerate(dataset.rows) if train_end < row.timestamp <= validation_end)
        if not train or not validation:
            raise ValueError("empty temporal fold")
        folds.append((train, validation))
    return tuple(folds)


@dataclass(frozen=True)
class ExplicitFoldWindow:
    fold_id: int
    train_start: datetime
    train_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    scoring_start: datetime
    scoring_end: datetime


@dataclass(frozen=True)
class ExplicitFoldSplit:
    window: ExplicitFoldWindow
    train: tuple[int, ...]
    calibration: tuple[int, ...]
    scoring: tuple[int, ...]


def explicit_temporal_folds(
    dataset: TrainingDataset, windows: Sequence[ExplicitFoldWindow], *, embargo: timedelta,
) -> tuple[ExplicitFoldSplit, ...]:
    """Single E2 fold authority with disjoint TRAIN/CALIBRATION/SCORING blocks."""
    if embargo <= timedelta(0) or not windows:
        raise ValueError("explicit fold embargo/windows are invalid")
    results = []
    previous_train_end: datetime | None = None
    for expected_id, window in enumerate(windows, start=1):
        values = (
            window.train_start, window.train_end, window.calibration_start,
            window.calibration_end, window.scoring_start, window.scoring_end,
        )
        if window.fold_id != expected_id or any(value.tzinfo is None for value in values):
            raise ValueError("explicit fold identity/timestamps are invalid")
        if not (
            window.train_start < window.train_end
            and window.train_end + embargo <= window.calibration_start < window.calibration_end
            and window.calibration_end + embargo == window.scoring_start < window.scoring_end
        ):
            raise ValueError("explicit fold chronology or embargo is invalid")
        if previous_train_end is not None and window.train_end <= previous_train_end:
            raise ValueError("explicit training folds are not expanding")
        train = tuple(index for index, row in enumerate(dataset.rows) if window.train_start <= row.timestamp <= window.train_end)
        calibration = tuple(index for index, row in enumerate(dataset.rows) if window.calibration_start <= row.timestamp <= window.calibration_end)
        scoring = tuple(index for index, row in enumerate(dataset.rows) if window.scoring_start <= row.timestamp <= window.scoring_end)
        if not train or not calibration or not scoring:
            raise ValueError("explicit temporal fold contains an empty block")
        if set(train) & set(calibration) or set(train) & set(scoring) or set(calibration) & set(scoring):
            raise ValueError("explicit temporal fold blocks overlap")
        results.append(ExplicitFoldSplit(window, train, calibration, scoring))
        previous_train_end = window.train_end
    return tuple(results)


@dataclass(frozen=True)
class HourlyDatasetBuild:
    dataset: TrainingDataset
    expected_anchor_count: int
    found_anchor_count: int
    valid_cycle_count: int
    skipped_history_cycles: int
    quarantined_label_cycles: int
    rows_by_symbol: Mapping[str, int]
    first_anchor: datetime
    last_anchor: datetime


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _candle(bar: CanonicalBar) -> Candle:
    return Candle(
        bar.timestamp, bar.timestamp + timedelta(minutes=5), bar.open, bar.high,
        bar.low, bar.close, bar.volume, True, "CANONICAL_D3_READ_ONLY",
    )


def build_e2_hourly_short_dataset(
    series: Mapping[str, Sequence[CanonicalBar]], sampling: Mapping[str, Any], *,
    dataset_id: str, source_finality_verified: bool,
) -> HourlyDatasetBuild:
    return _build_e2_hourly_directional_dataset(
        series,
        sampling,
        dataset_id=dataset_id,
        source_finality_verified=source_finality_verified,
        side="SHORT",
    )


def build_e2_hourly_long_dataset(
    series: Mapping[str, Sequence[CanonicalBar]],
    sampling: Mapping[str, Any],
    *,
    dataset_id: str,
    source_finality_verified: bool,
) -> HourlyDatasetBuild:
    return _build_e2_hourly_directional_dataset(
        series,
        sampling,
        dataset_id=dataset_id,
        source_finality_verified=source_finality_verified,
        side="LONG",
    )


def _build_e2_hourly_directional_dataset(
    series: Mapping[str, Sequence[CanonicalBar]],
    sampling: Mapping[str, Any],
    *,
    dataset_id: str,
    source_finality_verified: bool,
    side: str,
) -> HourlyDatasetBuild:
    """Build causal E2 rows at exact hourly close anchors without overlap or interpolation."""
    if side not in {"LONG", "SHORT"}:
        raise ValueError("directional dataset side is invalid")
    if not source_finality_verified:
        raise ValueError("E2 requires final canonical candles")
    if set(series) != set(CANONICAL_SYMBOLS) or int(sampling["coordinated_symbols_required"]) != len(CANONICAL_SYMBOLS):
        raise ValueError("E2 requires the coordinated eleven-symbol universe")
    history_bars = int(sampling["history_bars"]); horizon_bars = int(sampling["horizon_bars"])
    if int(sampling["stride_bars"]) != horizon_bars or horizon_bars != 12:
        raise ValueError("E2 H12 sampling overlap contract is invalid")
    expected = sampling["expected_rows"]
    first_anchor, last_anchor = _utc(expected["first_anchor_utc"]), _utc(expected["last_dev_anchor_utc"])
    interval = timedelta(minutes=5)
    expected_anchor_count = int((last_anchor - first_anchor).total_seconds() // 3600) + 1
    indexes = {symbol: {bar.timestamp: index for index, bar in enumerate(rows)} for symbol, rows in series.items()}
    anchors = []
    for bar in series[CANONICAL_SYMBOLS[0]]:
        close_time = bar.timestamp + interval
        if first_anchor <= close_time <= last_anchor and close_time.minute == close_time.second == close_time.microsecond == 0:
            anchors.append(close_time)
    if any(right - left < timedelta(minutes=5 * horizon_bars) for left, right in zip(anchors, anchors[1:])):
        raise ValueError("E2 H12 anchors overlap")
    pipeline = DeterministicFeaturePipeline()
    label_config = (
        LongLabelConfig(horizon_bars=horizon_bars, entry_rule="NEXT_BAR_OPEN")
        if side == "LONG"
        else ShortLabelConfig(horizon_bars=horizon_bars, entry_rule="NEXT_BAR_OPEN")
    )
    training_rows: list[TrainingRow] = []
    skipped_history = quarantined = valid_cycles = 0
    rows_by_symbol = {symbol: 0 for symbol in CANONICAL_SYMBOLS}
    hash_stream = hashlib.sha256()
    for anchor in anchors:
        anchor_open = anchor - interval
        selected: dict[str, tuple[Sequence[CanonicalBar], Sequence[CanonicalBar]]] = {}
        history_invalid = False
        for symbol in CANONICAL_SYMBOLS:
            position = indexes[symbol].get(anchor_open)
            if position is None or position - history_bars + 1 < 0:
                history_invalid = True; break
            history = series[symbol][position - history_bars + 1:position + 1]
            future = series[symbol][position + 1:position + horizon_bars + 1]
            if len(history) != history_bars or any(
                history[index].timestamp - history[index - 1].timestamp != interval
                for index in range(1, len(history))
            ):
                history_invalid = True; break
            selected[symbol] = (history, future)
        if history_invalid or len(selected) != len(CANONICAL_SYMBOLS):
            skipped_history += 1; continue
        symbol_series = tuple(SymbolSeries(
            symbol, tuple(_candle(bar) for bar in selected[symbol][0]), anchor, FeedQuality(),
        ) for symbol in CANONICAL_SYMBOLS)
        snapshot = MarketSnapshot(
            anchor, "5m", CANONICAL_SYMBOL_SET_HASH, symbol_series,
            PortfolioContext(available_slots=1, operational_time=anchor),
        )
        batch = pipeline.transform(snapshot)
        pending: list[TrainingRow] = []
        cycle_quarantined = False
        for feature_row in batch.rows:
            history, future = selected[feature_row.symbol]
            signal_candle = _candle(history[-1])
            future_candles = tuple(_candle(bar) for bar in future)
            label = (
                build_long_path_label(
                    signal_candle,
                    future_candles,
                    label_config,
                )
                if isinstance(label_config, LongLabelConfig)
                else build_short_path_label(
                    signal_candle,
                    future_candles,
                    label_config,
                )
            )
            if not label.valid:
                cycle_quarantined = True; break
            terminal_return = (
                label.terminal_long_return
                if isinstance(label_config, LongLabelConfig)
                else label.terminal_short_return
            )
            assert terminal_return is not None and label.mae_fraction is not None
            assert label.net_quality_after_costs is not None
            direction = (
                1.0 if terminal_return > label_config.round_trip_cost_fraction else 0.0
            ) if side == "LONG" else (
                -1.0 if terminal_return > label_config.round_trip_cost_fraction else 0.0
            )
            target = TrainingTarget(
                direction,
                terminal_return if side == "LONG" else -terminal_return,
                float(label.tail_event),
                label.mae_fraction,
                float(label.clean_entry), label.net_quality_after_costs, float(label.bad_entry), True,
            )
            regime, _ = classify_market_regime(dict(zip(FEATURE_NAMES, feature_row.raw_values)))
            pending.append(TrainingRow(anchor, feature_row.symbol, feature_row.raw_values, target, regime))
        if cycle_quarantined:
            quarantined += 1; continue
        for row in pending:
            training_rows.append(row); rows_by_symbol[row.symbol] += 1
            hash_stream.update(canonical_json(row).encode("utf-8")); hash_stream.update(b"\n")
        valid_cycles += 1
    minimum_rows = int(expected["approximate_maximum_dev_rows"] * float(expected["hard_stop_if_valid_rows_below_fraction"]))
    if len(training_rows) < minimum_rows:
        raise ValueError("E2 valid rows are below the pre-registered 90% hard stop")
    dataset_hash = hashlib.sha256(
        canonical_json({
            "dataset_id": dataset_id, "feature_hash": FEATURE_HASH,
            "label_schema": (
                LONG_LABEL_SCHEMA_VERSION
                if side == "LONG"
                else SHORT_LABEL_SCHEMA_VERSION
            ),
            "sampling": sampling,
            "rows_sha256": hash_stream.hexdigest(), "row_count": len(training_rows),
        }).encode("utf-8")
    ).hexdigest()
    dataset = TrainingDataset(
        dataset_id, "aegis-training-dataset-v2", FEATURE_SCHEMA_VERSION, FEATURE_HASH,
        CANONICAL_SYMBOLS, "5m", tuple(training_rows), dataset_hash,
    )
    return HourlyDatasetBuild(
        dataset, expected_anchor_count, len(anchors), valid_cycles, skipped_history,
        quarantined, rows_by_symbol, first_anchor, last_anchor,
    )


def load_and_build_e2_hourly_dataset(
    source: CanonicalSeriesSource, preregistration: Mapping[str, Any],
) -> HourlyDatasetBuild:
    sampling = preregistration["sampling"]
    first_anchor = _utc(sampling["expected_rows"]["first_anchor_utc"])
    last_anchor = _utc(sampling["expected_rows"]["last_dev_anchor_utc"])
    semi_blind = _utc(preregistration["lockbox"]["semi_blind_start"])
    if last_anchor >= semi_blind:
        raise ValueError("SEMI_BLIND_ACCESS_FORBIDDEN_PRE_LOCKBOX")
    start = first_anchor - timedelta(minutes=5 * int(sampling["history_bars"]))
    # Half-open load: the final H12 bar opens at t+55m, so t+60m is sufficient.
    end = last_anchor + timedelta(minutes=5 * int(sampling["horizon_bars"]))
    if end > semi_blind:
        raise ValueError("SEMI_BLIND_ACCESS_FORBIDDEN_PRE_LOCKBOX")
    audit = source.audit(verify_content=True)
    loaded = source.load(start=start, end=end)
    return build_e2_hourly_short_dataset(
        loaded, sampling, dataset_id=str(preregistration["experiment_id"]),
        source_finality_verified=audit.finality_verified,
    )


def load_and_build_e2_hourly_long_dataset(
    source: CanonicalSeriesSource,
    preregistration: Mapping[str, Any],
) -> HourlyDatasetBuild:
    sampling = preregistration["sampling"]
    first_anchor = _utc(sampling["expected_rows"]["first_anchor_utc"])
    last_anchor = _utc(sampling["expected_rows"]["last_dev_anchor_utc"])
    semi_blind = _utc(preregistration["lockbox"]["semi_blind_start"])
    if last_anchor >= semi_blind:
        raise ValueError("SEMI_BLIND_ACCESS_FORBIDDEN_PRE_LOCKBOX")
    start = first_anchor - timedelta(minutes=5 * int(sampling["history_bars"]))
    end = last_anchor + timedelta(minutes=5 * int(sampling["horizon_bars"]))
    if end > semi_blind:
        raise ValueError("SEMI_BLIND_ACCESS_FORBIDDEN_PRE_LOCKBOX")
    audit = source.audit(verify_content=True)
    loaded = source.load(start=start, end=end)
    return build_e2_hourly_long_dataset(
        loaded,
        sampling,
        dataset_id=str(preregistration["experiment_id"]),
        source_finality_verified=audit.finality_verified,
    )
