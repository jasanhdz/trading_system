"""Causal features, protected outcomes, and simple selectors for M1B."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.data import CanonicalBar
from aegis.training.hybrid_directional import DirectionalSide

from .hybrid_ts_protection_replay import (
    IntrabarPath,
    TsProtectionConfig,
    replay_ts_price_protection,
)


FEATURE_NAMES = (
    "side_return_3m",
    "side_return_12m",
    "side_return_60m",
    "side_taker_flow_1m",
    "side_taker_flow_3m",
    "volume_ratio_24m",
    "compression_ratio",
    "breakout_distance",
    "mark_spot_basis",
    "basis_change_15m",
    "basis_zscore_7d",
    "latest_funding_rate",
    "side_adjusted_funding_rate",
    "funding_age_hours",
    "direction_score",
    "realized_volatility_1h",
    "liquidity_ratio_1h",
    "btc_return_1h",
    "cross_symbol_breadth_1h",
    "utc_hour_sin",
    "utc_hour_cos",
    "weekday_sin",
    "weekday_cos",
)

MARK_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


class M1BContractError(ValueError):
    pass


def _read_single_csv(path: Path, names: Sequence[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise M1BContractError("AEGIS_M1B_ARCHIVE_MEMBER_INVALID")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=names)
    first = pd.to_numeric(frame[names[0]], errors="coerce")
    frame = frame.loc[first.notna()].copy()
    frame[names[0]] = first.loc[first.notna()].astype("int64")
    return frame


def load_mark_prices(root: Path, symbol: str) -> pd.DataFrame:
    paths = sorted(
        (root / "futures/um/monthly/markPriceKlines" / symbol / "1m").glob("*.zip")
    )
    if not paths:
        raise M1BContractError("AEGIS_M1B_MARK_PRICE_MISSING")
    frame = pd.concat((_read_single_csv(path, MARK_COLUMNS) for path in paths), ignore_index=True)
    frame["mark_close"] = pd.to_numeric(frame["close"], errors="raise")
    if frame["open_time"].duplicated().any():
        raise M1BContractError("AEGIS_M1B_MARK_PRICE_DUPLICATE")
    return frame[["open_time", "mark_close"]].sort_values("open_time", ignore_index=True)


def load_funding(root: Path, symbol: str) -> pd.DataFrame:
    paths = sorted((root / "futures/um/monthly/fundingRate" / symbol).glob("*.zip"))
    if not paths:
        raise M1BContractError("AEGIS_M1B_FUNDING_MISSING")
    frames = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise M1BContractError("AEGIS_M1B_FUNDING_ARCHIVE_INVALID")
            with archive.open(members[0]) as handle:
                frame = pd.read_csv(handle)
        expected = {"calc_time", "last_funding_rate"}
        if not expected.issubset(frame.columns):
            raise M1BContractError("AEGIS_M1B_FUNDING_SCHEMA_INVALID")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["funding_time"] = pd.to_numeric(result["calc_time"], errors="raise").astype("int64")
    result["funding_rate"] = pd.to_numeric(result["last_funding_rate"], errors="raise")
    result = result[["funding_time", "funding_rate"]].drop_duplicates("funding_time")
    return result.sort_values("funding_time", ignore_index=True)


def enrich_symbol_frame(
    frame: pd.DataFrame,
    mark: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    regime_hourly: pd.DataFrame,
    btc_hourly: pd.DataFrame,
    breadth_hourly: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.sort_values("open_time").merge(
        mark, on="open_time", how="left", validate="one_to_one"
    )
    if len(result) != len(frame):
        raise M1BContractError("AEGIS_M1B_MARK_PRICE_IDENTITY_MISMATCH")
    result["mark_spot_basis"] = result["mark_close"] / result["spot_close"] - 1.0
    result["basis_change_15m"] = result["mark_spot_basis"] - result["mark_spot_basis"].shift(15)
    rolling = result["mark_spot_basis"].rolling(7 * 24 * 60, min_periods=24 * 60)
    result["basis_zscore_7d"] = (
        result["mark_spot_basis"] - rolling.mean()
    ) / rolling.std().replace(0.0, np.nan)
    result = pd.merge_asof(
        result,
        funding,
        left_on="open_time",
        right_on="funding_time",
        direction="backward",
    )
    result["funding_age_hours"] = (
        result["open_time"] - result["funding_time"]
    ) / 3_600_000.0
    result = pd.merge_asof(
        result,
        regime_hourly.sort_values("timestamp_ms"),
        left_on="open_time",
        right_on="timestamp_ms",
        direction="backward",
    ).drop(columns="timestamp_ms")
    result = pd.merge_asof(
        result,
        btc_hourly.sort_values("timestamp_ms"),
        left_on="open_time",
        right_on="timestamp_ms",
        direction="backward",
    ).drop(columns="timestamp_ms")
    result = pd.merge_asof(
        result,
        breadth_hourly.sort_values("timestamp_ms"),
        left_on="open_time",
        right_on="timestamp_ms",
        direction="backward",
    ).drop(columns="timestamp_ms")
    timestamp = pd.to_datetime(result["open_time"], unit="ms", utc=True)
    result["utc_hour_sin"] = np.sin(2.0 * np.pi * timestamp.dt.hour / 24.0)
    result["utc_hour_cos"] = np.cos(2.0 * np.pi * timestamp.dt.hour / 24.0)
    result["weekday_sin"] = np.sin(2.0 * np.pi * timestamp.dt.dayofweek / 7.0)
    result["weekday_cos"] = np.cos(2.0 * np.pi * timestamp.dt.dayofweek / 7.0)
    return result


def feature_row(row: Mapping[str, Any], side: str) -> tuple[float, ...]:
    if side not in {"LONG", "SHORT"}:
        raise M1BContractError("AEGIS_M1B_SIDE_INVALID")
    sign = 1.0 if side == "LONG" else -1.0
    breakout = max(float(row["breakout_up"]), float(row["breakout_down"]))
    values = (
        sign * float(row["ret_3"]),
        sign * float(row["ret_12"]),
        sign * float(row["ret_60"]),
        sign * float(row["flow_1"]),
        sign * float(row["flow_3"]),
        float(row["volume_ratio"]),
        float(row["compression"]),
        breakout,
        float(row["mark_spot_basis"]),
        float(row["basis_change_15m"]),
        float(row["basis_zscore_7d"]),
        float(row["funding_rate"]),
        sign * float(row["funding_rate"]),
        float(row["funding_age_hours"]),
        sign * float(row["direction_score"]),
        float(row["realized_volatility_1h"]),
        float(row["liquidity_ratio_1h"]),
        sign * float(row["btc_return_1h"]),
        sign * float(row["cross_symbol_breadth_1h"]),
        float(row["utc_hour_sin"]),
        float(row["utc_hour_cos"]),
        float(row["weekday_sin"]),
        float(row["weekday_cos"]),
    )
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise M1BContractError("AEGIS_M1B_FEATURE_CONTRACT_INVALID")
    if float(row["funding_age_hours"]) > 12.0:
        raise M1BContractError("AEGIS_M1B_FUNDING_STALE")
    return values


def _bars(frame: pd.DataFrame, start: int, end: int) -> tuple[CanonicalBar, ...]:
    rows = frame.iloc[start:end]
    return tuple(
        CanonicalBar(
            datetime.fromtimestamp(int(row.open_time) / 1000.0, tz=timezone.utc),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for row in rows.itertuples(index=False)
    )


def protected_outcome(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    entry_time: int,
    side: str,
    config: TsProtectionConfig,
    horizon: int = 240,
) -> Mapping[str, Any]:
    locations = pd.Index(frame["open_time"]).get_indexer([entry_time])
    location = int(locations[0])
    if location < config.atr_period + 1 or location + horizon > len(frame):
        raise M1BContractError("AEGIS_M1B_PATH_INCOMPLETE")
    history = _bars(frame, location - config.atr_period - 1, location)
    future = _bars(frame, location, location + horizon)
    direction = DirectionalSide.LONG if side == "LONG" else DirectionalSide.SHORT
    results = [
        replay_ts_price_protection(
            side=direction,
            history=history,
            future=future,
            path=path,
            config=config,
        )
        for path in IntrabarPath
    ]
    worst = min(results, key=lambda item: item.net_return_after_costs)
    exit_time = entry_time + (worst.bars_held - 1) * 60_000 + 59_999
    paid = funding.loc[
        funding["funding_time"].ge(entry_time) & funding["funding_time"].le(exit_time),
        "funding_rate",
    ].sum()
    funding_return = (-1.0 if side == "LONG" else 1.0) * float(paid)
    net = worst.net_return_after_costs + funding_return
    first_positive = None
    for offset, bar in enumerate(future, start=1):
        favorable = bar.high if side == "LONG" else bar.low
        gross = (
            (favorable - future[0].open) / future[0].open
            if side == "LONG"
            else (future[0].open - favorable) / future[0].open
        )
        if gross > config.round_trip_cost_fraction:
            first_positive = offset
            break
    return {
        "entry_price": worst.entry_price,
        "exit_price": worst.exit_price,
        "bars_held": worst.bars_held,
        "gross_return_fraction": worst.gross_return_fraction,
        "net_return_after_costs": worst.net_return_after_costs,
        "peak_roe": worst.peak_roe,
        "lowest_roe": worst.lowest_roe,
        "break_even_armed": worst.break_even_armed,
        "trailing_armed": worst.trailing_armed,
        "atr_available": worst.atr_available,
        "exit_reason": worst.exit_reason.value,
        "path": worst.path.value,
        "funding_return_fraction": funding_return,
        "protected_net_return": net,
        "positive_protected_net": net > 0.0,
        "mae_fraction": max(0.0, -worst.lowest_roe / config.leverage),
        "mfe_fraction": max(0.0, worst.peak_roe / config.leverage),
        "time_to_first_positive_net": first_positive,
        "target_before_stop": worst.peak_roe >= config.break_even_trigger_roe,
    }


@dataclass(frozen=True)
class M1BPolicy:
    minimum_positive_probability: float
    maximum_mae_q90: float
    minimum_net_utility: float
    calibration_score: float
    calibration_events: int


@dataclass
class M1BModels:
    probability: Any
    mae: Any
    utility: Any
    calibrator: Any | None = None


def train_models(rows: pd.DataFrame, feature_names: Sequence[str] = FEATURE_NAMES) -> M1BModels:
    if tuple(feature_names) != FEATURE_NAMES:
        raise M1BContractError("AEGIS_M1B_FEATURE_ORDER_INVALID")
    x = rows.loc[:, feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or len(rows) < 200:
        raise M1BContractError("AEGIS_M1B_TRAINING_DATA_INVALID")
    probability = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=2000, random_state=181001)),
    ]).fit(x, rows["positive_protected_net"].astype(int))
    mae = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=80, max_depth=2,
        learning_rate=0.04, random_state=181001,
    ).fit(x, rows["mae_fraction"])
    utility = GradientBoostingRegressor(
        loss="huber", alpha=0.90, n_estimators=80, max_depth=2,
        learning_rate=0.04, random_state=181001,
    ).fit(x, rows["protected_net_return"])
    return M1BModels(probability, mae, utility)


def calibrate_probability(models: M1BModels, calibration: pd.DataFrame) -> None:
    """Fit Platt scaling on the frozen calibration partition only."""

    if models.calibrator is not None:
        raise M1BContractError("AEGIS_M1B_CALIBRATOR_ALREADY_FITTED")
    x = calibration.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = calibration["positive_protected_net"].astype(int)
    if len(calibration) < 100 or labels.nunique() != 2 or not np.isfinite(x).all():
        raise M1BContractError("AEGIS_M1B_CALIBRATION_DATA_INVALID")
    raw = models.probability.predict_proba(x)[:, 1]
    logits = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / np.clip(1 - raw, 1e-9, 1))
    models.calibrator = LogisticRegression(
        C=1.0, max_iter=2000, random_state=181001
    ).fit(logits.reshape(-1, 1), labels)


def predict_models(models: M1BModels, rows: pd.DataFrame) -> pd.DataFrame:
    if models.calibrator is None:
        raise M1BContractError("AEGIS_M1B_CALIBRATOR_NOT_FITTED")
    x = rows.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    raw = models.probability.predict_proba(x)[:, 1]
    logits = np.log(np.clip(raw, 1e-9, 1 - 1e-9) / np.clip(1 - raw, 1e-9, 1))
    result = rows.copy()
    result["predicted_positive_probability"] = models.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    result["predicted_mae_q90"] = np.maximum(0.0, models.mae.predict(x))
    result["predicted_net_utility"] = models.utility.predict(x)
    return result


def fit_policy(calibration: pd.DataFrame) -> M1BPolicy:
    best: M1BPolicy | None = None
    minimum_selected = max(30, math.ceil(len(calibration) * 0.05))
    quantiles = (0.50, 0.60, 0.70, 0.80, 0.90)
    for p_quantile in quantiles:
        for mae_quantile in quantiles:
            for utility_quantile in quantiles:
                minimum_p = float(calibration["predicted_positive_probability"].quantile(p_quantile))
                maximum_mae = float(calibration["predicted_mae_q90"].quantile(1.0 - mae_quantile))
                minimum_utility = float(calibration["predicted_net_utility"].quantile(utility_quantile))
                selected = calibration.loc[
                    calibration["predicted_positive_probability"].ge(minimum_p)
                    & calibration["predicted_mae_q90"].le(maximum_mae)
                    & calibration["predicted_net_utility"].ge(minimum_utility)
                ]
                if len(selected) < minimum_selected:
                    continue
                values = selected["protected_net_return"].to_numpy()
                score = float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values)))
                candidate = M1BPolicy(minimum_p, maximum_mae, minimum_utility, score, len(selected))
                if best is None or (candidate.calibration_score, candidate.calibration_events) > (
                    best.calibration_score, best.calibration_events
                ):
                    best = candidate
    if best is None:
        raise M1BContractError("AEGIS_M1B_CALIBRATION_POLICY_UNAVAILABLE")
    return best


def apply_policy(rows: pd.DataFrame, policy: M1BPolicy) -> pd.DataFrame:
    selected = rows.loc[
        rows["predicted_positive_probability"].ge(policy.minimum_positive_probability)
        & rows["predicted_mae_q90"].le(policy.maximum_mae_q90)
        & rows["predicted_net_utility"].ge(policy.minimum_net_utility)
    ].copy()
    if "timestamp_ms" not in selected or "symbol" not in selected:
        raise M1BContractError("AEGIS_M1B_EVENT_IDENTITY_MISSING")
    return (
        selected.sort_values(
            ["timestamp_ms", "predicted_net_utility", "symbol"],
            ascending=[True, False, True],
        )
        .drop_duplicates("timestamp_ms", keep="first")
        .sort_values(["timestamp_ms", "symbol"], ignore_index=True)
    )
