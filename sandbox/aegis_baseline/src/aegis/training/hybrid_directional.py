"""Inspectable hybrid LONG/SHORT committee for offline and Shadow research."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, MutableSequence, Sequence

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import average_precision_score, mean_absolute_error

from ..features import (
    FEATURE_HASH,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FrozenNormalizer,
)
from ..models import CalibrationMethod, CalibratorSpec
from ..tree_models import TreeEnsemble
from .competition import export_hist_gradient_boosting, pinball_loss
from .dataset import TrainingDataset
from .train import calibration_metrics, fit_platt_calibrator

HYBRID_SCHEMA_VERSION = "aegis-hybrid-directional-committee-v1"
HYBRID_FEATURE_NAMES = (
    *FEATURE_NAMES,
    *(f"directional__{name}" for name in FEATURE_NAMES),
    "hypothesis_direction",
)


class DirectionalSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> float:
        return 1.0 if self is DirectionalSide.LONG else -1.0


@dataclass(frozen=True)
class HybridDirectionalRow:
    timestamp_ns: int
    symbol: str
    side: DirectionalSide
    features: tuple[float, ...]
    opportunity: bool
    clean_entry: bool
    danger: bool
    mae_fraction: float
    mfe_fraction: float
    net_return_after_costs: float

    def __post_init__(self) -> None:
        values = (
            self.mae_fraction,
            self.mfe_fraction,
            self.net_return_after_costs,
            *self.features,
        )
        if (
            self.timestamp_ns <= 0
            or not self.symbol
            or len(self.features) != len(FEATURE_NAMES)
            or not all(math.isfinite(value) for value in values)
            or self.mae_fraction < 0.0
            or self.mfe_fraction < 0.0
        ):
            raise ValueError("hybrid directional row is invalid")


@dataclass(frozen=True)
class HybridDirectionalPrediction:
    side: DirectionalSide
    opportunity_probability: float
    danger_probability: float
    mae_q50: float
    mae_q90: float
    mfe_q50: float
    net_return_mean: float
    shadow_rank_score: float
    selection_effect: str = "NONE"
    exchange_authority: bool = False
    exchange_mutations: int = 0


@dataclass(frozen=True)
class HybridDirectionalSelection:
    timestamp_ns: int
    symbol: str
    side: DirectionalSide
    shadow_rank_score: float
    opportunity_probability: float
    danger_probability: float
    mae_q90: float
    mfe_q50: float
    net_return_mean: float
    observed_net_return_after_costs: float


@dataclass(frozen=True)
class HybridDirectionalMetrics:
    row_count: int
    side_rows: Mapping[str, int]
    opportunity_average_precision: Mapping[str, float]
    opportunity_prevalence: Mapping[str, float]
    opportunity_ece: Mapping[str, float]
    opportunity_brier: Mapping[str, float]
    danger_average_precision: float
    danger_prevalence: float
    danger_ece: float
    danger_brier: float
    mae_q50_mae: float
    mae_q90_pinball: float
    mae_q90_baseline_pinball: float
    mae_q90_coverage: float
    mfe_q50_mae: float
    net_return_mae: float
    shadow_selection: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class HybridDirectionalArtifact:
    schema_version: str
    feature_schema: str
    feature_hash: str
    normalizer: FrozenNormalizer
    opportunity_heads: Mapping[DirectionalSide, TreeEnsemble]
    opportunity_calibrators: Mapping[DirectionalSide, CalibratorSpec]
    danger_head: TreeEnsemble
    danger_calibrator: CalibratorSpec
    mae_q50_head: TreeEnsemble
    mae_q90_head: TreeEnsemble
    mae_q90_conformal_adjustment: float
    mfe_q50_head: TreeEnsemble
    net_return_head: TreeEnsemble
    round_trip_cost_fraction: float
    metrics: HybridDirectionalMetrics
    runtime_authority: str = "SHADOW_ONLY"

    def __post_init__(self) -> None:
        if (
            self.schema_version != HYBRID_SCHEMA_VERSION
            or self.feature_schema != FEATURE_SCHEMA_VERSION
            or self.feature_hash != FEATURE_HASH
            or set(self.opportunity_heads) != set(DirectionalSide)
            or set(self.opportunity_calibrators) != set(DirectionalSide)
            or not 0.0 <= self.round_trip_cost_fraction < 1.0
            or self.mae_q90_conformal_adjustment < 0.0
            or self.runtime_authority != "SHADOW_ONLY"
        ):
            raise ValueError("hybrid directional artifact authority is invalid")

    def _base_vector(self, features: Sequence[float]) -> tuple[float, ...]:
        if len(features) != len(FEATURE_NAMES) or not all(
            math.isfinite(float(value)) for value in features
        ):
            raise ValueError("hybrid inference features are invalid")
        return tuple(
            self.normalizer.normalize(name, float(value))[0]
            for name, value in zip(FEATURE_NAMES, features)
        )

    def predict(
        self,
        side: DirectionalSide,
        features: Sequence[float],
    ) -> HybridDirectionalPrediction:
        base = self._base_vector(features)
        shared = (*base, *(value * side.sign for value in base), side.sign)
        opportunity = self.opportunity_calibrators[side].apply(
            self.opportunity_heads[side].evaluate(base)
        )
        danger = self.danger_calibrator.apply(self.danger_head.evaluate(shared))
        mae_q50 = max(0.0, self.mae_q50_head.evaluate(shared))
        mae_q90 = max(
            mae_q50,
            self.mae_q90_head.evaluate(shared) + self.mae_q90_conformal_adjustment,
        )
        mfe_q50 = max(0.0, self.mfe_q50_head.evaluate(shared))
        net_return = self.net_return_head.evaluate(shared)
        rank_score = hybrid_shadow_rank_score(
            opportunity_probability=opportunity,
            danger_probability=danger,
            mae_q90=mae_q90,
            mfe_q50=mfe_q50,
            net_return_mean=net_return,
            round_trip_cost_fraction=self.round_trip_cost_fraction,
        )
        return HybridDirectionalPrediction(
            side=side,
            opportunity_probability=opportunity,
            danger_probability=danger,
            mae_q50=mae_q50,
            mae_q90=mae_q90,
            mfe_q50=mfe_q50,
            net_return_mean=net_return,
            shadow_rank_score=rank_score,
        )


def hybrid_shadow_rank_score(
    *,
    opportunity_probability: float,
    danger_probability: float,
    mae_q90: float,
    mfe_q50: float,
    net_return_mean: float,
    round_trip_cost_fraction: float,
) -> float:
    """Rank every hypothesis without turning the Shadow score into a guard."""

    values = (
        opportunity_probability,
        danger_probability,
        mae_q90,
        mfe_q50,
        net_return_mean,
        round_trip_cost_fraction,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or not 0.0 <= opportunity_probability <= 1.0
        or not 0.0 <= danger_probability <= 1.0
        or min(mae_q90, mfe_q50, round_trip_cost_fraction) < 0.0
        or round_trip_cost_fraction >= 1.0
    ):
        raise ValueError("hybrid Shadow ranking inputs are invalid")
    path_denominator = mfe_q50 + mae_q90 + round_trip_cost_fraction
    path_efficiency = mfe_q50 / path_denominator if path_denominator > 0.0 else 0.0
    cost_scale = max(round_trip_cost_fraction, 1e-9)
    net_factor = 1.0 / (
        1.0 + math.exp(-max(-40.0, min(40.0, net_return_mean / cost_scale)))
    )
    return (
        opportunity_probability
        * (1.0 - danger_probability)
        * path_efficiency
        * net_factor
    )


def paired_directional_rows(
    long_dataset: TrainingDataset,
    short_dataset: TrainingDataset,
    *,
    round_trip_cost_fraction: float,
) -> tuple[HybridDirectionalRow, ...]:
    """Pair causal directional labels without changing either source dataset."""

    if (
        long_dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or short_dataset.feature_schema_version != FEATURE_SCHEMA_VERSION
        or long_dataset.feature_hash != FEATURE_HASH
        or short_dataset.feature_hash != FEATURE_HASH
        or long_dataset.symbols != short_dataset.symbols
        or long_dataset.timeframe != short_dataset.timeframe
        or not 0.0 <= round_trip_cost_fraction < 1.0
    ):
        raise ValueError("hybrid source dataset authority mismatch")
    long_index = {(row.timestamp, row.symbol): row for row in long_dataset.rows}
    short_index = {(row.timestamp, row.symbol): row for row in short_dataset.rows}
    if set(long_index) != set(short_index):
        raise ValueError("directional source rows do not align")
    result: list[HybridDirectionalRow] = []
    for identity in sorted(long_index):
        directional = (
            (DirectionalSide.LONG, long_index[identity]),
            (DirectionalSide.SHORT, short_index[identity]),
        )
        if directional[0][1].features != directional[1][1].features:
            raise ValueError("directional source feature vectors differ")
        for side, row in directional:
            oriented_terminal_return = (
                row.target.expected_return
                if side is DirectionalSide.LONG
                else -row.target.expected_return
            )
            mfe = max(
                0.0,
                row.target.net_quality_after_costs
                + row.target.qmae
                + round_trip_cost_fraction,
            )
            result.append(
                HybridDirectionalRow(
                    timestamp_ns=int(row.timestamp.timestamp() * 1_000_000_000),
                    symbol=row.symbol,
                    side=side,
                    features=row.features,
                    opportunity=oriented_terminal_return > round_trip_cost_fraction,
                    clean_entry=row.target.clean_quality >= 0.5,
                    danger=row.target.bad_entry >= 0.5,
                    mae_fraction=row.target.qmae,
                    mfe_fraction=mfe,
                    net_return_after_costs=oriented_terminal_return
                    - round_trip_cost_fraction,
                )
            )
    return tuple(result)


def fit_hybrid_normalizer(rows: Sequence[HybridDirectionalRow]) -> FrozenNormalizer:
    matrix = np.asarray([row.features for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("hybrid normalizer matrix is invalid")
    means = matrix.mean(axis=0)
    scales = np.where(matrix.std(axis=0) <= 1e-12, 1.0, matrix.std(axis=0))
    return FrozenNormalizer(
        dict(zip(FEATURE_NAMES, means.tolist())),
        dict(zip(FEATURE_NAMES, scales.tolist())),
    )


def _base_matrix(
    rows: Sequence[HybridDirectionalRow], normalizer: FrozenNormalizer
) -> np.ndarray:
    return np.asarray(
        [
            [
                normalizer.normalize(name, value)[0]
                for name, value in zip(FEATURE_NAMES, row.features)
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _shared_matrix(
    rows: Sequence[HybridDirectionalRow], normalizer: FrozenNormalizer
) -> np.ndarray:
    base = _base_matrix(rows, normalizer)
    signs = np.asarray([row.side.sign for row in rows], dtype=np.float64)
    return np.column_stack([base, base * signs[:, None], signs])


def _validate_blocks(
    train: Sequence[HybridDirectionalRow],
    calibration: Sequence[HybridDirectionalRow],
    scoring: Sequence[HybridDirectionalRow],
    *,
    embargo_minutes: int,
) -> None:
    if not train or not calibration or not scoring or embargo_minutes <= 0:
        raise ValueError("hybrid temporal blocks are invalid")
    minute_ns = 60 * 1_000_000_000
    if max(row.timestamp_ns for row in train) + embargo_minutes * minute_ns > min(
        row.timestamp_ns for row in calibration
    ) or max(
        row.timestamp_ns for row in calibration
    ) + embargo_minutes * minute_ns > min(
        row.timestamp_ns for row in scoring
    ):
        raise ValueError("hybrid temporal embargo is violated")
    for block in (train, calibration, scoring):
        if {row.side for row in block} != set(DirectionalSide):
            raise ValueError("hybrid temporal block lacks a direction")


def _fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    *,
    seed: int,
    parameters: Mapping[str, object],
) -> tuple[HistGradientBoostingClassifier, CalibratorSpec]:
    if any(len(np.unique(values)) != 2 for values in (train_y, calibration_y)):
        raise ValueError("classifier blocks require both classes")
    model = HistGradientBoostingClassifier(**dict(parameters), random_state=seed).fit(
        train_x, train_y
    )
    raw = np.asarray(model.predict_proba(calibration_x)[:, 1], dtype=np.float64)
    return model, fit_platt_calibrator(raw, calibration_y)


def fit_hybrid_directional_committee(
    train: Sequence[HybridDirectionalRow],
    calibration: Sequence[HybridDirectionalRow],
    scoring: Sequence[HybridDirectionalRow],
    *,
    seed: int,
    embargo_minutes: int,
    round_trip_cost_fraction: float,
    classifier_parameters: Mapping[str, object],
    regressor_parameters: Mapping[str, object],
    selection_trace: MutableSequence[HybridDirectionalSelection] | None = None,
) -> HybridDirectionalArtifact:
    """Fit specialists plus shared directional heads on disjoint time blocks."""

    _validate_blocks(train, calibration, scoring, embargo_minutes=embargo_minutes)
    normalizer = fit_hybrid_normalizer(train)
    opportunity_heads: dict[DirectionalSide, TreeEnsemble] = {}
    opportunity_calibrators: dict[DirectionalSide, CalibratorSpec] = {}
    opportunity_metrics: dict[str, dict[str, float]] = {}
    opportunity_predictions: dict[tuple[int, str, DirectionalSide], float] = {}
    for offset, side in enumerate(DirectionalSide):
        side_train = tuple(row for row in train if row.side is side)
        side_calibration = tuple(row for row in calibration if row.side is side)
        side_scoring = tuple(row for row in scoring if row.side is side)
        model, calibrator = _fit_classifier(
            _base_matrix(side_train, normalizer),
            np.asarray([row.opportunity for row in side_train], dtype=np.int8),
            _base_matrix(side_calibration, normalizer),
            np.asarray([row.opportunity for row in side_calibration], dtype=np.int8),
            seed=seed + offset,
            parameters=classifier_parameters,
        )
        head = export_hist_gradient_boosting(
            model,
            f"hybrid-{side.value.lower()}-opportunity",
            FEATURE_NAMES,
            classifier=True,
        )
        raw = np.asarray(
            [head.evaluate(row) for row in _base_matrix(side_scoring, normalizer)],
            dtype=np.float64,
        )
        probabilities = np.asarray(
            [calibrator.apply(value) for value in raw], dtype=np.float64
        )
        actual = np.asarray([row.opportunity for row in side_scoring], dtype=np.int8)
        ece, brier = calibration_metrics(probabilities, actual)
        opportunity_metrics[side.value] = {
            "average_precision": float(average_precision_score(actual, probabilities)),
            "prevalence": float(np.mean(actual)),
            "ece": ece,
            "brier": brier,
        }
        opportunity_predictions.update(
            {
                (row.timestamp_ns, row.symbol, row.side): float(probability)
                for row, probability in zip(side_scoring, probabilities)
            }
        )
        opportunity_heads[side] = head
        opportunity_calibrators[side] = calibrator

    train_shared = _shared_matrix(train, normalizer)
    calibration_shared = _shared_matrix(calibration, normalizer)
    scoring_shared = _shared_matrix(scoring, normalizer)
    danger_model, danger_calibrator = _fit_classifier(
        train_shared,
        np.asarray([row.danger for row in train], dtype=np.int8),
        calibration_shared,
        np.asarray([row.danger for row in calibration], dtype=np.int8),
        seed=seed + 10,
        parameters=classifier_parameters,
    )
    danger_head = export_hist_gradient_boosting(
        danger_model,
        "hybrid-shared-danger",
        HYBRID_FEATURE_NAMES,
        classifier=True,
    )
    danger_actual = np.asarray([row.danger for row in scoring], dtype=np.int8)
    danger_probabilities = np.asarray(
        [danger_calibrator.apply(danger_head.evaluate(row)) for row in scoring_shared],
        dtype=np.float64,
    )
    danger_ece, danger_brier = calibration_metrics(danger_probabilities, danger_actual)

    def quantile_model(target: np.ndarray, quantile: float, offset: int):
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            **dict(regressor_parameters),
            random_state=seed + offset,
        ).fit(train_shared, target)

    train_mae = np.asarray([row.mae_fraction for row in train], dtype=np.float64)
    calibration_mae = np.asarray(
        [row.mae_fraction for row in calibration], dtype=np.float64
    )
    scoring_mae = np.asarray([row.mae_fraction for row in scoring], dtype=np.float64)
    mae_q50_model = quantile_model(train_mae, 0.50, 20)
    mae_q90_model = quantile_model(train_mae, 0.90, 21)
    residuals = calibration_mae - mae_q90_model.predict(calibration_shared)
    rank = min(
        len(residuals) - 1,
        math.ceil((len(residuals) + 1) * 0.90) - 1,
    )
    adjustment = max(0.0, float(np.sort(residuals)[rank]))
    scoring_mae_q50 = mae_q50_model.predict(scoring_shared)
    scoring_mae_q90 = mae_q90_model.predict(scoring_shared) + adjustment
    baseline_q90 = float(np.quantile(train_mae, 0.90, method="higher"))

    train_mfe = np.asarray([row.mfe_fraction for row in train], dtype=np.float64)
    scoring_mfe = np.asarray([row.mfe_fraction for row in scoring], dtype=np.float64)
    mfe_q50_model = quantile_model(train_mfe, 0.50, 22)
    scoring_mfe_q50 = mfe_q50_model.predict(scoring_shared)

    train_net = np.asarray(
        [row.net_return_after_costs for row in train], dtype=np.float64
    )
    scoring_net = np.asarray(
        [row.net_return_after_costs for row in scoring], dtype=np.float64
    )
    net_model = HistGradientBoostingRegressor(
        loss="squared_error",
        **dict(regressor_parameters),
        random_state=seed + 23,
    ).fit(train_shared, train_net)
    scoring_net_predictions = net_model.predict(scoring_shared)

    shadow_selection: dict[str, Mapping[str, object]] = {}
    for side in DirectionalSide:
        side_indices = [index for index, row in enumerate(scoring) if row.side is side]
        by_timestamp: dict[int, list[int]] = {}
        for index in side_indices:
            by_timestamp.setdefault(scoring[index].timestamp_ns, []).append(index)
        selected_indices: list[int] = []
        for indices in by_timestamp.values():
            ranked = []
            for index in indices:
                row = scoring[index]
                opportunity = opportunity_predictions[
                    (row.timestamp_ns, row.symbol, row.side)
                ]
                score = hybrid_shadow_rank_score(
                    opportunity_probability=opportunity,
                    danger_probability=float(danger_probabilities[index]),
                    mae_q90=float(scoring_mae_q90[index]),
                    mfe_q50=float(scoring_mfe_q50[index]),
                    net_return_mean=float(scoring_net_predictions[index]),
                    round_trip_cost_fraction=round_trip_cost_fraction,
                )
                ranked.append((score, row.symbol, index))
            score, _, selected = max(ranked, key=lambda item: (item[0], item[1]))
            selected_indices.append(selected)
            if selection_trace is not None:
                row = scoring[selected]
                selection_trace.append(
                    HybridDirectionalSelection(
                        timestamp_ns=row.timestamp_ns,
                        symbol=row.symbol,
                        side=row.side,
                        shadow_rank_score=float(score),
                        opportunity_probability=opportunity_predictions[
                            (row.timestamp_ns, row.symbol, row.side)
                        ],
                        danger_probability=float(danger_probabilities[selected]),
                        mae_q90=float(scoring_mae_q90[selected]),
                        mfe_q50=float(scoring_mfe_q50[selected]),
                        net_return_mean=float(scoring_net_predictions[selected]),
                        observed_net_return_after_costs=row.net_return_after_costs,
                    )
                )
        selected_rows = [scoring[index] for index in selected_indices]
        symbol_counts = {
            symbol: sum(row.symbol == symbol for row in selected_rows)
            for symbol in sorted({row.symbol for row in selected_rows})
        }
        shadow_selection[side.value] = {
            "signals": len(selected_rows),
            "mean_net_expectancy": (
                float(np.mean([row.net_return_after_costs for row in selected_rows]))
                if selected_rows
                else 0.0
            ),
            "win_rate": (
                sum(row.net_return_after_costs > 0.0 for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else 0.0
            ),
            "mean_mae": (
                float(np.mean([row.mae_fraction for row in selected_rows]))
                if selected_rows
                else 0.0
            ),
            "symbol_concentration": (
                max(symbol_counts.values()) / len(selected_rows)
                if selected_rows
                else 1.0
            ),
            "symbol_counts": symbol_counts,
        }

    metrics = HybridDirectionalMetrics(
        row_count=len(scoring),
        side_rows={
            side.value: sum(row.side is side for row in scoring)
            for side in DirectionalSide
        },
        opportunity_average_precision={
            side: values["average_precision"]
            for side, values in opportunity_metrics.items()
        },
        opportunity_prevalence={
            side: values["prevalence"] for side, values in opportunity_metrics.items()
        },
        opportunity_ece={
            side: values["ece"] for side, values in opportunity_metrics.items()
        },
        opportunity_brier={
            side: values["brier"] for side, values in opportunity_metrics.items()
        },
        danger_average_precision=float(
            average_precision_score(danger_actual, danger_probabilities)
        ),
        danger_prevalence=float(np.mean(danger_actual)),
        danger_ece=danger_ece,
        danger_brier=danger_brier,
        mae_q50_mae=float(mean_absolute_error(scoring_mae, scoring_mae_q50)),
        mae_q90_pinball=pinball_loss(scoring_mae, scoring_mae_q90, 0.90),
        mae_q90_baseline_pinball=pinball_loss(
            scoring_mae,
            np.full(len(scoring_mae), baseline_q90),
            0.90,
        ),
        mae_q90_coverage=float(np.mean(scoring_mae <= scoring_mae_q90)),
        mfe_q50_mae=float(mean_absolute_error(scoring_mfe, scoring_mfe_q50)),
        net_return_mae=float(mean_absolute_error(scoring_net, scoring_net_predictions)),
        shadow_selection=shadow_selection,
    )
    return HybridDirectionalArtifact(
        schema_version=HYBRID_SCHEMA_VERSION,
        feature_schema=FEATURE_SCHEMA_VERSION,
        feature_hash=FEATURE_HASH,
        normalizer=normalizer,
        opportunity_heads=opportunity_heads,
        opportunity_calibrators=opportunity_calibrators,
        danger_head=danger_head,
        danger_calibrator=danger_calibrator,
        mae_q50_head=export_hist_gradient_boosting(
            mae_q50_model,
            "hybrid-shared-mae-q50",
            HYBRID_FEATURE_NAMES,
            classifier=False,
        ),
        mae_q90_head=export_hist_gradient_boosting(
            mae_q90_model,
            "hybrid-shared-mae-q90",
            HYBRID_FEATURE_NAMES,
            classifier=False,
        ),
        mae_q90_conformal_adjustment=adjustment,
        mfe_q50_head=export_hist_gradient_boosting(
            mfe_q50_model,
            "hybrid-shared-mfe-q50",
            HYBRID_FEATURE_NAMES,
            classifier=False,
        ),
        net_return_head=export_hist_gradient_boosting(
            net_model,
            "hybrid-shared-net-return",
            HYBRID_FEATURE_NAMES,
            classifier=False,
        ),
        round_trip_cost_fraction=round_trip_cost_fraction,
        metrics=metrics,
    )


def calibrator_mapping(value: CalibratorSpec) -> Mapping[str, object]:
    return {
        "method": value.method.value,
        "ece": value.ece,
        "brier": value.brier,
        "sample_count": value.sample_count,
        "parameters": list(value.parameters),
        "x": list(value.x),
        "y": list(value.y),
    }


def calibrator_from_mapping(value: Mapping[str, object]) -> CalibratorSpec:
    return CalibratorSpec(
        method=CalibrationMethod(str(value["method"])),
        ece=float(value["ece"]),
        brier=float(value["brier"]),
        sample_count=int(value["sample_count"]),
        parameters=tuple(float(item) for item in value.get("parameters", ())),
        x=tuple(float(item) for item in value.get("x", ())),
        y=tuple(float(item) for item in value.get("y", ())),
    )


def hybrid_artifact_mapping(
    artifact: HybridDirectionalArtifact,
) -> Mapping[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "feature_schema": artifact.feature_schema,
        "feature_hash": artifact.feature_hash,
        "feature_names": list(FEATURE_NAMES),
        "hybrid_feature_names": list(HYBRID_FEATURE_NAMES),
        "normalizer": {
            "means": dict(artifact.normalizer.means),
            "scales": dict(artifact.normalizer.scales),
            "clip_absolute": artifact.normalizer.clip_absolute,
        },
        "opportunity_heads": {
            side.value: artifact.opportunity_heads[side].to_payload()
            for side in DirectionalSide
        },
        "opportunity_calibrators": {
            side.value: calibrator_mapping(artifact.opportunity_calibrators[side])
            for side in DirectionalSide
        },
        "danger_head": artifact.danger_head.to_payload(),
        "danger_calibrator": calibrator_mapping(artifact.danger_calibrator),
        "mae_q50_head": artifact.mae_q50_head.to_payload(),
        "mae_q90_head": artifact.mae_q90_head.to_payload(),
        "mae_q90_conformal_adjustment": artifact.mae_q90_conformal_adjustment,
        "mfe_q50_head": artifact.mfe_q50_head.to_payload(),
        "net_return_head": artifact.net_return_head.to_payload(),
        "round_trip_cost_fraction": artifact.round_trip_cost_fraction,
        "metrics": {
            "row_count": artifact.metrics.row_count,
            "side_rows": dict(artifact.metrics.side_rows),
            "opportunity_average_precision": dict(
                artifact.metrics.opportunity_average_precision
            ),
            "opportunity_prevalence": dict(artifact.metrics.opportunity_prevalence),
            "opportunity_ece": dict(artifact.metrics.opportunity_ece),
            "opportunity_brier": dict(artifact.metrics.opportunity_brier),
            "danger_average_precision": artifact.metrics.danger_average_precision,
            "danger_prevalence": artifact.metrics.danger_prevalence,
            "danger_ece": artifact.metrics.danger_ece,
            "danger_brier": artifact.metrics.danger_brier,
            "mae_q50_mae": artifact.metrics.mae_q50_mae,
            "mae_q90_pinball": artifact.metrics.mae_q90_pinball,
            "mae_q90_baseline_pinball": artifact.metrics.mae_q90_baseline_pinball,
            "mae_q90_coverage": artifact.metrics.mae_q90_coverage,
            "mfe_q50_mae": artifact.metrics.mfe_q50_mae,
            "net_return_mae": artifact.metrics.net_return_mae,
            "shadow_selection": artifact.metrics.shadow_selection,
        },
        "runtime_authority": artifact.runtime_authority,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def write_hybrid_directional_artifact(
    path: Path, artifact: HybridDirectionalArtifact
) -> None:
    from .run_state import atomic_write_json

    atomic_write_json(path, hybrid_artifact_mapping(artifact))


def load_hybrid_directional_artifact(path: Path) -> HybridDirectionalArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            tuple(payload["feature_names"]) != FEATURE_NAMES
            or tuple(payload["hybrid_feature_names"]) != HYBRID_FEATURE_NAMES
            or payload.get("exchange_authority") is not False
            or int(payload.get("exchange_mutations", -1)) != 0
        ):
            raise ValueError("hybrid artifact safety metadata is invalid")
        normalizer_payload = payload["normalizer"]
        metrics_payload = payload["metrics"]
        return HybridDirectionalArtifact(
            schema_version=str(payload["schema_version"]),
            feature_schema=str(payload["feature_schema"]),
            feature_hash=str(payload["feature_hash"]),
            normalizer=FrozenNormalizer(
                means={
                    str(name): float(value)
                    for name, value in normalizer_payload["means"].items()
                },
                scales={
                    str(name): float(value)
                    for name, value in normalizer_payload["scales"].items()
                },
                clip_absolute=float(normalizer_payload["clip_absolute"]),
            ),
            opportunity_heads={
                side: TreeEnsemble.from_payload(
                    payload["opportunity_heads"][side.value]
                )
                for side in DirectionalSide
            },
            opportunity_calibrators={
                side: calibrator_from_mapping(
                    payload["opportunity_calibrators"][side.value]
                )
                for side in DirectionalSide
            },
            danger_head=TreeEnsemble.from_payload(payload["danger_head"]),
            danger_calibrator=calibrator_from_mapping(payload["danger_calibrator"]),
            mae_q50_head=TreeEnsemble.from_payload(payload["mae_q50_head"]),
            mae_q90_head=TreeEnsemble.from_payload(payload["mae_q90_head"]),
            mae_q90_conformal_adjustment=float(payload["mae_q90_conformal_adjustment"]),
            mfe_q50_head=TreeEnsemble.from_payload(payload["mfe_q50_head"]),
            net_return_head=TreeEnsemble.from_payload(payload["net_return_head"]),
            round_trip_cost_fraction=float(payload["round_trip_cost_fraction"]),
            metrics=HybridDirectionalMetrics(
                row_count=int(metrics_payload["row_count"]),
                side_rows={
                    str(name): int(value)
                    for name, value in metrics_payload["side_rows"].items()
                },
                opportunity_average_precision={
                    str(name): float(value)
                    for name, value in metrics_payload[
                        "opportunity_average_precision"
                    ].items()
                },
                opportunity_prevalence={
                    str(name): float(value)
                    for name, value in metrics_payload["opportunity_prevalence"].items()
                },
                opportunity_ece={
                    str(name): float(value)
                    for name, value in metrics_payload["opportunity_ece"].items()
                },
                opportunity_brier={
                    str(name): float(value)
                    for name, value in metrics_payload["opportunity_brier"].items()
                },
                danger_average_precision=float(
                    metrics_payload["danger_average_precision"]
                ),
                danger_prevalence=float(metrics_payload["danger_prevalence"]),
                danger_ece=float(metrics_payload["danger_ece"]),
                danger_brier=float(metrics_payload["danger_brier"]),
                mae_q50_mae=float(metrics_payload["mae_q50_mae"]),
                mae_q90_pinball=float(metrics_payload["mae_q90_pinball"]),
                mae_q90_baseline_pinball=float(
                    metrics_payload["mae_q90_baseline_pinball"]
                ),
                mae_q90_coverage=float(metrics_payload["mae_q90_coverage"]),
                mfe_q50_mae=float(metrics_payload["mfe_q50_mae"]),
                net_return_mae=float(metrics_payload["net_return_mae"]),
                shadow_selection={
                    str(side): dict(values)
                    for side, values in metrics_payload["shadow_selection"].items()
                },
            ),
            runtime_authority=str(payload["runtime_authority"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("hybrid directional artifact is invalid") from exc
