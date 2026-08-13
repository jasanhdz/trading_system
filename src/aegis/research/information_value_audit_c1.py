"""Incremental causal information-family audit for C1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SIDES = ("LONG", "SHORT")
PRIMARY_COST = 0.0014
STRESS_COST = 0.0020
FAMILIES: Mapping[str, tuple[str, ...]] = {
    "PRICE_STATE": (
        "side_return_1h", "side_return_4h", "side_return_24h",
        "realized_volatility_4h", "realized_volatility_24h",
        "side_distance_sma_4h", "side_distance_sma_24h",
        "side_extension_z_24h", "side_breakout_acceptance", "side_wick_rejection",
    ),
    "FLOW_ACTIVITY": (
        "side_taker_flow_15m", "side_taker_flow_1h",
        "side_prior_taker_flow_1h", "volume_persistence_1h",
    ),
    "DERIVATIVES_CARRY": (
        "side_mark_spot_basis", "side_basis_z_7d", "basis_convergence_1h",
        "side_funding_rate", "funding_age_hours",
    ),
    "CROSS_MARKET": (
        "side_btc_state_return", "side_relative_strength_btc_4h",
        "side_cross_sectional_rank_4h", "side_breadth_4h", "beta_btc",
        "side_common_alt_state",
    ),
    "CALENDAR_CONTROL": ("utc_hour_sin", "utc_hour_cos", "weekday_sin", "weekday_cos"),
}
CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "PRICE_STATE": ("PRICE_STATE",),
    "PRICE_STATE_PLUS_FLOW_ACTIVITY": ("PRICE_STATE", "FLOW_ACTIVITY"),
    "PRICE_STATE_PLUS_DERIVATIVES_CARRY": ("PRICE_STATE", "DERIVATIVES_CARRY"),
    "PRICE_STATE_PLUS_CROSS_MARKET": ("PRICE_STATE", "CROSS_MARKET"),
    "PRICE_STATE_PLUS_CALENDAR_CONTROL": ("PRICE_STATE", "CALENDAR_CONTROL"),
    "PRICE_STATE_PLUS_ALL_AVAILABLE_NON_CALENDAR": (
        "PRICE_STATE", "FLOW_ACTIVITY", "DERIVATIVES_CARRY", "CROSS_MARKET",
    ),
}


class C1ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    value: Any
    barrier: Any
    mae: Any
    selection_threshold: float
    features: tuple[str, ...]


def feature_names(candidate: str) -> tuple[str, ...]:
    try:
        groups = CANDIDATES[candidate]
    except KeyError as error:
        raise C1ContractError("AEGIS_C1_CANDIDATE_UNKNOWN") from error
    return tuple(name for group in groups for name in FAMILIES[group])


def contract_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("ascii")).hexdigest()


def canonical_features(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "side", "return_1h", "return_4h", "return_24h", "realized_volatility_4h",
        "realized_volatility_24h", "distance_sma_4h", "distance_sma_24h",
        "extension_z_24h", "breakout_acceptance_long", "breakout_acceptance_short",
        "wick_rejection_long", "wick_rejection_short", "taker_flow_15m",
        "taker_flow_1h", "prior_taker_flow_1h", "volume_persistence_1h",
        "mark_spot_basis", "basis_z_7d", "basis_convergence_1h", "funding_rate",
        "funding_age_hours", "btc_state_return", "relative_strength_btc_4h",
        "cross_sectional_return_rank_4h", "breadth_4h", "beta_btc",
        "common_alt_state", "utc_hour_sin", "utc_hour_cos", "weekday_sin", "weekday_cos",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise C1ContractError(f"AEGIS_C1_SOURCE_COLUMNS_MISSING:{','.join(sorted(missing))}")
    sign = np.where(rows.side.eq("LONG"), 1.0, -1.0)
    lower = rows.side.str.lower()
    result = pd.DataFrame(index=rows.index)
    for source, target in (
        ("return_1h", "side_return_1h"), ("return_4h", "side_return_4h"),
        ("return_24h", "side_return_24h"), ("distance_sma_4h", "side_distance_sma_4h"),
        ("distance_sma_24h", "side_distance_sma_24h"), ("extension_z_24h", "side_extension_z_24h"),
        ("taker_flow_15m", "side_taker_flow_15m"), ("taker_flow_1h", "side_taker_flow_1h"),
        ("prior_taker_flow_1h", "side_prior_taker_flow_1h"),
        ("mark_spot_basis", "side_mark_spot_basis"), ("basis_z_7d", "side_basis_z_7d"),
        ("funding_rate", "side_funding_rate"), ("btc_state_return", "side_btc_state_return"),
        ("relative_strength_btc_4h", "side_relative_strength_btc_4h"),
        ("common_alt_state", "side_common_alt_state"),
    ):
        result[target] = sign * rows[source]
    result["side_cross_sectional_rank_4h"] = np.where(
        rows.side.eq("LONG"), rows.cross_sectional_return_rank_4h,
        1.0 - rows.cross_sectional_return_rank_4h,
    )
    result["side_breadth_4h"] = sign * rows.breadth_4h
    result["side_breakout_acceptance"] = np.where(
        lower.eq("long"), rows.breakout_acceptance_long, rows.breakout_acceptance_short,
    )
    result["side_wick_rejection"] = np.where(
        lower.eq("long"), rows.wick_rejection_long, rows.wick_rejection_short,
    )
    for name in (
        "realized_volatility_4h", "realized_volatility_24h", "volume_persistence_1h",
        "basis_convergence_1h", "funding_age_hours", "beta_btc", "utc_hour_sin",
        "utc_hour_cos", "weekday_sin", "weekday_cos",
    ):
        result[name] = rows[name]
    expected = tuple(name for names in FAMILIES.values() for name in names)
    if set(result.columns) != set(expected) or len(result.columns) != len(expected):
        raise C1ContractError("AEGIS_C1_CANONICAL_FEATURE_ORDER_MISMATCH")
    result = result.loc[:, list(expected)]
    values = result.to_numpy(float)
    if not np.isfinite(values).all():
        raise C1ContractError("AEGIS_C1_CANONICAL_FEATURE_NONFINITE")
    return result


def _linear(estimator: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def fit_bundle(
    rows: pd.DataFrame,
    train_timestamps: set[int],
    calibration_timestamps: set[int],
    candidate: str,
) -> ModelBundle:
    names = feature_names(candidate)
    train = rows.timestamp_ms.isin(train_timestamps)
    calibration = rows.timestamp_ms.isin(calibration_timestamps)
    if train.sum() < 500 or calibration.sum() < 100:
        raise C1ContractError("AEGIS_C1_TRAINING_ROWS_INSUFFICIENT")
    value = _linear(Ridge(alpha=10.0)).fit(rows.loc[train, list(names)], rows.loc[train, "residual_utility"])
    barrier = _linear(LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=1813,
    )).fit(rows.loc[train, list(names)], rows.loc[train, "favorable_first"])
    mae = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=1.0, random_state=1813,
    ).fit(rows.loc[train, list(names)], rows.loc[train, "mae"])
    threshold = float(np.quantile(value.predict(rows.loc[calibration, list(names)]), 0.90))
    return ModelBundle(value, barrier, mae, threshold, names)


def _safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    value = float(spearmanr(actual, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def evaluate_bundle(bundle: ModelBundle, rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    features = rows.loc[:, list(bundle.features)]
    value_score = bundle.value.predict(features)
    barrier_probability = bundle.barrier.predict_proba(features)[:, 1]
    predicted_mae = bundle.mae.predict(features)
    scored = rows.assign(
        predicted_value=value_score,
        barrier_probability=barrier_probability,
        predicted_mae=predicted_mae,
    )
    group_correlations = [
        _safe_spearman(group.residual_utility, group.predicted_value)
        for _, group in scored.groupby("timestamp_ms", sort=True)
    ]
    eligible = scored.loc[scored.predicted_value.ge(bundle.selection_threshold)]
    selected = (
        eligible.sort_values(["timestamp_ms", "predicted_value", "symbol"], ascending=[True, False, True])
        .drop_duplicates("timestamp_ms", keep="first")
        .copy()
    )
    selected["net_primary"] = selected.gross_return - PRIMARY_COST
    selected["net_stress"] = selected.gross_return - STRESS_COST
    labels = rows.favorable_first.astype(int)
    return {
        "rows": len(rows), "events": rows.timestamp_ms.nunique(),
        "grouped_spearman": float(np.mean(group_correlations)),
        "barrier_log_loss": float(log_loss(labels, barrier_probability, labels=[0, 1])),
        "barrier_average_precision": float(average_precision_score(labels, barrier_probability)),
        "barrier_brier": float(brier_score_loss(labels, barrier_probability)),
        "mae_spearman": _safe_spearman(rows.mae, predicted_mae),
        "mae_absolute_error": float(mean_absolute_error(rows.mae, predicted_mae)),
        "selected_events": len(selected),
        "selected_primary_net": float(selected.net_primary.mean()) if len(selected) else -math.inf,
        "selected_stress_net": float(selected.net_stress.mean()) if len(selected) else -math.inf,
        "selected_win_rate": float(selected.net_primary.gt(0.0).mean()) if len(selected) else 0.0,
    }, selected


def day_cluster_bootstrap(selected: pd.DataFrame, repetitions: int = 1000) -> dict[str, float] | None:
    if selected.empty:
        return None
    days = list(
        selected.assign(day=pd.to_datetime(selected.timestamp_ms, unit="ms", utc=True).dt.floor("1D"))
        .groupby("day")["net_primary"].apply(np.asarray)
    )
    random = np.random.default_rng(1814)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([days[index] for index in random.integers(0, len(days), len(days))])
        values.append(float(sample.mean()))
    return {"lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}


def incremental_metrics(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "grouped_spearman_lift": float(candidate["grouped_spearman"] - baseline["grouped_spearman"]),
        "barrier_log_loss_improvement": float(baseline["barrier_log_loss"] - candidate["barrier_log_loss"]),
        "barrier_average_precision_improvement": float(candidate["barrier_average_precision"] - baseline["barrier_average_precision"]),
        "mae_spearman_lift": float(candidate["mae_spearman"] - baseline["mae_spearman"]),
    }
