"""Transparent causal mechanisms for Economic Alpha Discovery A1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MECHANISMS = ("TREND_ACCEPTANCE", "EXTREME_REVERSAL", "CARRY_CONVERGENCE")
SIDES = ("LONG", "SHORT")
HORIZONS = (60, 240, 480, 720, 1440)


class AlphaDiscoveryContractError(ValueError):
    pass


@dataclass(frozen=True)
class RobustScale:
    median: float
    iqr: float
    lower: float = -math.inf
    upper: float = math.inf

    def apply(self, values: pd.Series) -> pd.Series:
        clipped = values.clip(self.lower, self.upper)
        return ((clipped - self.median) / self.iqr).clip(-8.0, 8.0)


def aggregate_completed_15m(frame: pd.DataFrame, mark: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(mark, on="open_time", how="left", validate="one_to_one")
    merged["bucket"] = merged["open_time"] // 900_000 * 900_000
    grouped = merged.groupby("bucket", sort=True)
    result = grouped.agg(
        first_time=("open_time", "first"),
        last_time=("open_time", "last"),
        minute_count=("open_time", "size"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        quote_volume=("quote_volume", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
        spot_close=("spot_close", "last"),
        mark_close=("mark_close", "last"),
    ).reset_index(names="timestamp_ms")
    result = result.loc[
        result["minute_count"].eq(15)
        & result["first_time"].eq(result["timestamp_ms"])
        & result["last_time"].eq(result["timestamp_ms"] + 14 * 60_000)
    ].copy()
    result["state_close_ms"] = result["timestamp_ms"] + 900_000 - 1
    result["taker_flow_15m"] = (
        2.0 * result["taker_buy_quote"] / result["quote_volume"].replace(0.0, np.nan)
        - 1.0
    )
    return result.drop(columns=["first_time", "last_time", "minute_count"])


def add_causal_features(state: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    result = state.sort_values("timestamp_ms", ignore_index=True).copy()
    close = result["close"]
    ret15 = close.pct_change()
    result["return_1h"] = close / close.shift(4) - 1.0
    result["return_4h"] = close / close.shift(16) - 1.0
    result["return_24h"] = close / close.shift(96) - 1.0
    result["realized_volatility_4h"] = ret15.rolling(16, min_periods=16).std(ddof=0)
    result["realized_volatility_24h"] = ret15.rolling(96, min_periods=96).std(ddof=0)
    result["taker_flow_1h"] = result["taker_flow_15m"].rolling(4, min_periods=4).mean()
    result["prior_taker_flow_1h"] = result["taker_flow_1h"].shift(4)
    baseline_volume = result["quote_volume"].shift(4).rolling(672, min_periods=96).median()
    result["volume_persistence_1h"] = (
        result["quote_volume"].rolling(4, min_periods=4).mean()
        / baseline_volume.replace(0.0, np.nan)
    )
    prior_high = result["high"].shift(4).rolling(16, min_periods=16).max()
    prior_low = result["low"].shift(4).rolling(16, min_periods=16).min()
    long_accept = (
        result["close"].gt(prior_high).astype(float).rolling(4, min_periods=4).mean()
        - 0.5
    )
    short_accept = (
        result["close"].lt(prior_low).astype(float).rolling(4, min_periods=4).mean()
        - 0.5
    )
    result["breakout_acceptance_long"] = long_accept
    result["breakout_acceptance_short"] = short_accept
    result["distance_sma_4h"] = close / close.rolling(16, min_periods=16).mean() - 1.0
    result["distance_sma_24h"] = close / close.rolling(96, min_periods=96).mean() - 1.0
    rolling_return = result["return_4h"].rolling(30 * 24 * 4, min_periods=7 * 24 * 4)
    result["extension_z_24h"] = (
        result["return_4h"] - rolling_return.mean()
    ) / rolling_return.std(ddof=0).replace(0.0, np.nan)
    high_1h = result["high"].rolling(4, min_periods=4).max()
    low_1h = result["low"].rolling(4, min_periods=4).min()
    open_1h = result["open"].shift(3)
    upper = (high_1h - pd.concat([open_1h, close], axis=1).max(axis=1)) / close
    lower = (pd.concat([open_1h, close], axis=1).min(axis=1) - low_1h) / close
    result["wick_rejection_long"] = lower - upper
    result["wick_rejection_short"] = upper - lower
    result["mark_spot_basis"] = result["mark_close"] / result["spot_close"] - 1.0
    basis_rolling = result["mark_spot_basis"].rolling(7 * 24 * 4, min_periods=24 * 4)
    result["basis_z_7d"] = (
        result["mark_spot_basis"] - basis_rolling.mean()
    ) / basis_rolling.std(ddof=0).replace(0.0, np.nan)
    result["basis_convergence_1h"] = (
        result["mark_spot_basis"].shift(4).abs() - result["mark_spot_basis"].abs()
    )
    result = pd.merge_asof(
        result.sort_values("state_close_ms"),
        funding.sort_values("funding_time"),
        left_on="state_close_ms",
        right_on="funding_time",
        direction="backward",
    )
    result["funding_age_hours"] = (
        result["state_close_ms"] - result["funding_time"]
    ) / 3_600_000.0
    timestamp = pd.to_datetime(result["state_close_ms"], unit="ms", utc=True)
    result["utc_hour_sin"] = np.sin(2.0 * np.pi * timestamp.dt.hour / 24.0)
    result["utc_hour_cos"] = np.cos(2.0 * np.pi * timestamp.dt.hour / 24.0)
    result["weekday_sin"] = np.sin(2.0 * np.pi * timestamp.dt.dayofweek / 7.0)
    result["weekday_cos"] = np.cos(2.0 * np.pi * timestamp.dt.dayofweek / 7.0)
    return result.sort_values("timestamp_ms", ignore_index=True)


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    btc = result.loc[result["symbol"].eq("BTCUSDT"), ["timestamp_ms", "return_4h"]].rename(
        columns={"return_4h": "btc_return_4h"}
    )
    result = result.merge(btc, on="timestamp_ms", how="left", validate="many_to_one")
    result["relative_strength_btc_4h"] = result["return_4h"] - result["btc_return_4h"]
    result["cross_sectional_return_rank_4h"] = result.groupby("timestamp_ms")[
        "return_4h"
    ].rank(pct=True, method="average")
    breadth = (
        result.assign(positive=result["return_4h"].gt(0.0).astype(float))
        .groupby("timestamp_ms")["positive"]
        .transform("mean")
    )
    result["breadth_4h"] = 2.0 * breadth - 1.0
    return result


def robust_scale(values: pd.Series) -> RobustScale:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 100:
        raise AlphaDiscoveryContractError("AEGIS_A1_SCALE_DATA_INSUFFICIENT")
    q25, q75 = finite.quantile([0.25, 0.75])
    iqr = float(q75 - q25)
    if not math.isfinite(iqr) or iqr <= 1e-12:
        raise AlphaDiscoveryContractError("AEGIS_A1_SCALE_DEGENERATE")
    lower, upper = finite.quantile([0.001, 0.999])
    return RobustScale(float(finite.median()), iqr, float(lower), float(upper))


def side_components(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    if side not in SIDES:
        raise AlphaDiscoveryContractError("AEGIS_A1_SIDE_INVALID")
    sign = 1.0 if side == "LONG" else -1.0
    acceptance = panel[f"breakout_acceptance_{side.lower()}"]
    wick = panel[f"wick_rejection_{side.lower()}"]
    result = pd.DataFrame(index=panel.index)
    result["trend_return_4h"] = sign * panel["return_4h"]
    result["trend_return_24h"] = sign * panel["return_24h"]
    result["trend_volume"] = panel["volume_persistence_1h"]
    result["trend_flow"] = sign * panel["taker_flow_1h"]
    result["trend_acceptance"] = acceptance
    result["trend_relative"] = sign * panel["relative_strength_btc_4h"]
    result["reversal_extension"] = -sign * panel["extension_z_24h"]
    result["reversal_flow_exhaustion"] = sign * (
        panel["taker_flow_1h"] - panel["prior_taker_flow_1h"]
    )
    result["reversal_reclaim"] = sign * panel["return_1h"]
    result["reversal_wick"] = wick
    result["reversal_relative"] = sign * (
        panel["return_1h"] - panel["btc_return_4h"] / 4.0
    )
    result["carry_basis_extremity"] = -sign * panel["basis_z_7d"]
    result["carry_funding"] = -sign * panel["funding_rate"]
    result["carry_convergence"] = panel["basis_convergence_1h"]
    result["carry_liquidity"] = panel["volume_persistence_1h"]
    return result


COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "TREND_ACCEPTANCE": (
        "trend_return_4h", "trend_return_24h", "trend_volume", "trend_flow",
        "trend_acceptance", "trend_relative",
    ),
    "EXTREME_REVERSAL": (
        "reversal_extension", "reversal_flow_exhaustion", "reversal_reclaim",
        "reversal_wick", "reversal_relative",
    ),
    "CARRY_CONVERGENCE": (
        "carry_basis_extremity", "carry_funding", "carry_convergence", "carry_liquidity",
    ),
}


def fit_scales_and_thresholds(
    panel: pd.DataFrame, train_mask: pd.Series
) -> tuple[dict[str, dict[str, RobustScale]], dict[str, dict[str, float]]]:
    scales: dict[str, dict[str, RobustScale]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for side in SIDES:
        values = side_components(panel, side)
        training = values.loc[train_mask]
        scales[side] = {
            name: robust_scale(training[name])
            for name in sorted({item for names in COMPONENTS.values() for item in names})
        }
        thresholds[side] = {
            "trend_volume_q60": float(training["trend_volume"].quantile(0.60)),
            "reversal_extension_q90": float(training["reversal_extension"].quantile(0.90)),
            "carry_basis_q90": float(training["carry_basis_extremity"].quantile(0.90)),
        }
    return scales, thresholds


def mechanism_rows(
    panel: pd.DataFrame,
    *,
    side: str,
    mechanism: str,
    scales: Mapping[str, RobustScale],
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    if mechanism not in MECHANISMS:
        raise AlphaDiscoveryContractError("AEGIS_A1_MECHANISM_INVALID")
    values = side_components(panel, side)
    score = sum(scales[name].apply(values[name]) for name in COMPONENTS[mechanism]) / len(
        COMPONENTS[mechanism]
    )
    if mechanism == "TREND_ACCEPTANCE":
        eligible = (
            values["trend_acceptance"].gt(0.0)
            & values["trend_volume"].ge(thresholds["trend_volume_q60"])
            & values["trend_flow"].gt(0.0)
        )
    elif mechanism == "EXTREME_REVERSAL":
        eligible = (
            values["reversal_extension"].ge(thresholds["reversal_extension_q90"])
            & values["reversal_reclaim"].gt(0.0)
            & values["reversal_flow_exhaustion"].gt(0.0)
        )
    else:
        eligible = (
            values["carry_basis_extremity"].ge(thresholds["carry_basis_q90"])
            & values["carry_funding"].gt(0.0)
            & values["carry_convergence"].gt(0.0)
            & panel["funding_age_hours"].le(12.0)
        )
    columns = [
        "timestamp_ms", "state_close_ms", "symbol", "return_4h",
        "realized_volatility_24h", "cross_sectional_return_rank_4h",
    ]
    result = panel.loc[eligible & score.notna(), columns].copy()
    result["side"] = side
    result["mechanism"] = mechanism
    result["score"] = score.loc[result.index]
    return result


def cross_sectional_winners(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return (
        rows.sort_values(["timestamp_ms", "score", "symbol"], ascending=[True, False, True])
        .drop_duplicates("timestamp_ms", keep="first")
        .sort_values(["timestamp_ms", "symbol"], ignore_index=True)
    )


def daily_space(rows: pd.DataFrame, minimum_minutes: int = 1440) -> pd.DataFrame:
    minimum = minimum_minutes * 60_000
    accepted = []
    latest: dict[str, int] = {}
    for row in rows.sort_values(["timestamp_ms", "symbol"]).itertuples(index=False):
        timestamp = int(row.timestamp_ms)
        if timestamp - latest.get(str(row.symbol), -minimum) < minimum:
            continue
        accepted.append(row._asdict())
        latest[str(row.symbol)] = timestamp
    return pd.DataFrame(accepted, columns=rows.columns)


def deterministic_random_symbol(rows: pd.DataFrame, identity: str) -> str:
    symbols = sorted(set(str(value) for value in rows["symbol"]))
    if not symbols:
        raise AlphaDiscoveryContractError("AEGIS_A1_RANDOM_POOL_EMPTY")
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    return symbols[int.from_bytes(digest[:8], "big") % len(symbols)]


def finite_contract(frame: pd.DataFrame, names: Sequence[str]) -> None:
    if tuple(frame.loc[:, names].columns) != tuple(names):
        raise AlphaDiscoveryContractError("AEGIS_A1_FEATURE_ORDER_INVALID")
    if not np.isfinite(frame.loc[:, names].to_numpy(dtype=float)).all():
        raise AlphaDiscoveryContractError("AEGIS_A1_FEATURE_NONFINITE")


def positive_count(values: Iterable[float]) -> int:
    return sum(float(value) > 0.0 for value in values)
