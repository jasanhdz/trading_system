"""Causal closed-candle volume-wave research primitives for W1."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution


SCHEMA_VERSION = "aegis-volume-wave-w1-dataset-v1"
SIDES = ("LONG", "SHORT")
ENTRY_VARIANTS = (
    "A_IMMEDIATE",
    "B_ONE_BAR_CONFIRMATION",
    "C_CAUSAL_PULLBACK",
    "D_EXTREME_BREAK_CONFIRMATION",
)
LADDERS = (
    "VOLUME_DIRECTION",
    "CLEAN_IMPULSE",
    "FLOW_ALIGNED",
    "TREND_5M_ALIGNED",
    "TREND_15M_ALIGNED",
    "BTC_NOT_OPPOSING",
    "SPACE_REMAINING",
)
SOURCE_COLUMNS = (
    "symbol", "open_time_ms", "open", "high", "low", "close",
    "quote_volume", "taker_buy_quote", "taker_sell_quote", "agg_trade_count",
)


class VolumeWaveContractError(ValueError):
    pass


def aggregate_closed_bars(minutes: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Aggregate exact complete one-minute buckets without filling gaps."""

    if interval_minutes <= 0 or any(column not in minutes for column in SOURCE_COLUMNS):
        raise VolumeWaveContractError("AEGIS_W1_SOURCE_CONTRACT_INVALID")
    values = minutes.loc[:, SOURCE_COLUMNS].copy()
    numeric = values.loc[:, SOURCE_COLUMNS[1:]].to_numpy(dtype=np.float64)
    if (
        values.empty
        or values["symbol"].nunique() != 1
        or not np.isfinite(numeric).all()
        or (values[["open", "high", "low", "close"]] <= 0.0).any().any()
        or values["open_time_ms"].duplicated().any()
    ):
        raise VolumeWaveContractError("AEGIS_W1_SOURCE_VALUES_INVALID")
    values.sort_values("open_time_ms", inplace=True)
    interval_ms = interval_minutes * 60_000
    values["bucket"] = values["open_time_ms"] // interval_ms * interval_ms
    grouped = values.groupby("bucket", sort=True)
    result = grouped.agg(
        symbol=("symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        quote_volume=("quote_volume", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
        taker_sell_quote=("taker_sell_quote", "sum"),
        agg_trade_count=("agg_trade_count", "sum"),
        minute_count=("open_time_ms", "size"),
        first_minute=("open_time_ms", "min"),
        last_minute=("open_time_ms", "max"),
    ).reset_index(names="open_time_ms")
    complete = (
        result["minute_count"].eq(interval_minutes)
        & result["first_minute"].eq(result["open_time_ms"])
        & result["last_minute"].eq(
            result["open_time_ms"] + (interval_minutes - 1) * 60_000
        )
    )
    result = result.loc[complete].drop(
        columns=["minute_count", "first_minute", "last_minute"]
    )
    result["close_time_ms"] = result["open_time_ms"] + interval_ms - 1
    result.reset_index(drop=True, inplace=True)
    return result


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _technical_features(bars: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    result = bars.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ], axis=1,
    ).max(axis=1)
    atr_period = int(config["features"]["atr_period"])
    result["atr"] = _wilder_average(true_range, atr_period)
    result["atr_fraction"] = result["atr"] / result["close"]
    candle_range = (result["high"] - result["low"]).replace(0.0, np.nan)
    result["body"] = result["close"] - result["open"]
    result["body_ratio"] = result["body"].abs() / candle_range
    result["body_atr"] = result["body"].abs() / result["atr"]
    result["clv"] = (result["close"] - result["low"]) / candle_range
    total_flow = result["taker_buy_quote"] + result["taker_sell_quote"]
    result["taker_buy_ratio"] = result["taker_buy_quote"] / total_flow.replace(0.0, np.nan)
    result["taker_sell_ratio"] = result["taker_sell_quote"] / total_flow.replace(0.0, np.nan)
    result["taker_imbalance"] = (
        result["taker_buy_quote"] - result["taker_sell_quote"]
    ) / total_flow.replace(0.0, np.nan)
    result["delta_quote"] = result["taker_buy_quote"] - result["taker_sell_quote"]
    result["delta_velocity"] = result["delta_quote"].diff() / total_flow.replace(0.0, np.nan)
    result["delta_acceleration"] = result["delta_velocity"].diff()
    log_volume = np.log(result["quote_volume"].where(result["quote_volume"] > 0.0))
    for lookback in config["features"]["volume_median_lookbacks"]:
        period = int(lookback)
        lagged = result["quote_volume"].shift(1)
        median = lagged.rolling(period, min_periods=period).median()
        result[f"volume_ratio_{period}"] = result["quote_volume"] / median.replace(0.0, np.nan)
    for lookback in config["features"]["volume_log_z_lookbacks"]:
        period = int(lookback)
        lagged = log_volume.shift(1)
        mean = lagged.rolling(period, min_periods=period).mean()
        std = lagged.rolling(period, min_periods=period).std(ddof=0)
        result[f"volume_z_{period}"] = (log_volume - mean) / std.replace(0.0, np.nan)
    result["return_1"] = result["close"].pct_change()
    result["return_3"] = result["close"].pct_change(3)
    result["velocity_atr_1"] = result["close"].diff() / result["atr"]
    result["velocity_atr_2"] = result["close"].diff(2) / result["atr"]
    result["acceleration_atr"] = result["velocity_atr_1"].diff()
    for period in config["features"]["moving_average_periods"]:
        size = int(period)
        average = result["close"].rolling(size, min_periods=size).mean()
        result[f"ma_{size}"] = average
        result[f"price_vs_ma_{size}_atr"] = (result["close"] - average) / result["atr"]
        result[f"ma_{size}_slope_atr"] = (average - average.shift(3)) / result["atr"]
    delta = result["close"].diff()
    for period in config["features"]["rsi_periods"]:
        size = int(period)
        gain = _wilder_average(delta.clip(lower=0.0), size)
        loss = _wilder_average((-delta.clip(upper=0.0)), size)
        relative = gain / loss.replace(0.0, np.nan)
        result[f"rsi_{size}"] = 100.0 - 100.0 / (1.0 + relative)
    result["higher_high"] = result["high"].gt(result["high"].shift(1))
    result["higher_low"] = result["low"].gt(result["low"].shift(1))
    result["lower_high"] = result["high"].lt(result["high"].shift(1))
    result["lower_low"] = result["low"].lt(result["low"].shift(1))
    absolute_step = result["close"].diff().abs()
    for period in (3, 6):
        path = absolute_step.rolling(period, min_periods=period).sum()
        displacement = result["close"] - result["close"].shift(period)
        result[f"directional_persistence_{period}"] = displacement / path.replace(0.0, np.nan)
        result[f"path_efficiency_{period}"] = displacement.abs() / path.replace(0.0, np.nan)
    return result


def build_causal_feature_frame(
    minutes: pd.DataFrame,
    btc_minutes: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build 5m decision features with completed 15m and BTC context."""

    five = _technical_features(aggregate_closed_bars(minutes, 5), config)
    fifteen = _technical_features(aggregate_closed_bars(minutes, 15), config)
    btc_five = _technical_features(aggregate_closed_bars(btc_minutes, 5), config)
    btc_fifteen = _technical_features(aggregate_closed_bars(btc_minutes, 15), config)
    context_columns = (
        "close_time_ms", "return_1", "return_3", "atr_fraction", "rsi_6",
        "price_vs_ma_25_atr", "ma_25_slope_atr", "volume_ratio_20",
        "taker_imbalance",
    )
    fifteen_context = fifteen.loc[:, context_columns].rename(
        columns={name: f"context_15m_{name}" for name in context_columns if name != "close_time_ms"}
    )
    result = pd.merge_asof(
        five.sort_values("close_time_ms"),
        fifteen_context.sort_values("close_time_ms"),
        on="close_time_ms", direction="backward", allow_exact_matches=True,
    )
    btc_five_context = btc_five.loc[:, context_columns].rename(
        columns={name: f"btc_5m_{name}" for name in context_columns if name != "close_time_ms"}
    )
    result = pd.merge_asof(
        result.sort_values("close_time_ms"),
        btc_five_context.sort_values("close_time_ms"),
        on="close_time_ms", direction="backward", allow_exact_matches=True,
    )
    btc_fifteen_context = btc_fifteen.loc[:, context_columns].rename(
        columns={name: f"btc_15m_{name}" for name in context_columns if name != "close_time_ms"}
    )
    result = pd.merge_asof(
        result.sort_values("close_time_ms"),
        btc_fifteen_context.sort_values("close_time_ms"),
        on="close_time_ms", direction="backward", allow_exact_matches=True,
    )
    correlation_bars = int(config["features"]["rolling_btc_correlation_bars"])
    result["btc_return_aligned"] = result["btc_5m_return_1"]
    result["btc_correlation"] = result["return_1"].rolling(
        correlation_bars, min_periods=correlation_bars
    ).corr(result["btc_return_aligned"])
    return result


def _variant_decision_index(
    frame: pd.DataFrame, index: int, side: str, variant: str
) -> int | None:
    sign = 1.0 if side == "LONG" else -1.0
    impulse = frame.iloc[index]
    midpoint = (float(impulse.open) + float(impulse.close)) / 2.0
    if variant == "A_IMMEDIATE":
        return index
    if variant == "B_ONE_BAR_CONFIRMATION":
        if index + 1 >= len(frame):
            return None
        confirmation = frame.iloc[index + 1]
        return (
            index + 1
            if sign * (float(confirmation.close) - float(impulse.close)) > 0.0
            and sign * (float(confirmation.close) - midpoint) > 0.0
            else None
        )
    for candidate_index in (index + 1, index + 2):
        if candidate_index >= len(frame):
            break
        candidate = frame.iloc[candidate_index]
        if variant == "C_CAUSAL_PULLBACK":
            body = abs(float(impulse.close) - float(impulse.open))
            retracement = sign * (float(impulse.close) - float(candidate.close)) / body
            if 0.10 <= retracement <= 0.40 and sign * (float(candidate.close) - midpoint) > 0.0:
                return candidate_index
        elif variant == "D_EXTREME_BREAK_CONFIRMATION":
            extreme = float(impulse.high) if side == "LONG" else float(impulse.low)
            if sign * (float(candidate.close) - extreme) > 0.0:
                return candidate_index
        else:
            raise VolumeWaveContractError("AEGIS_W1_ENTRY_VARIANT_INVALID")
    return None


def _variant_decision_indices(
    frame: pd.DataFrame, indices: np.ndarray, side: str, variant: str
) -> np.ndarray:
    sign = 1.0 if side == "LONG" else -1.0
    decision = np.full(len(indices), -1, dtype=np.int64)
    if variant == "A_IMMEDIATE":
        return indices.copy()
    opens = frame["open"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)
    midpoint = (opens[indices] + closes[indices]) / 2.0
    if variant == "B_ONE_BAR_CONFIRMATION":
        valid = indices + 1 < len(frame)
        candidates = indices[valid] + 1
        accepted = (
            sign * (closes[candidates] - closes[indices[valid]]) > 0.0
        ) & (sign * (closes[candidates] - midpoint[valid]) > 0.0)
        decision[np.flatnonzero(valid)[accepted]] = candidates[accepted]
        return decision
    if variant not in {"C_CAUSAL_PULLBACK", "D_EXTREME_BREAK_CONFIRMATION"}:
        raise VolumeWaveContractError("AEGIS_W1_ENTRY_VARIANT_INVALID")
    body = np.abs(closes[indices] - opens[indices])
    for offset in (1, 2):
        unresolved = decision < 0
        valid = unresolved & (indices + offset < len(frame))
        positions = np.flatnonzero(valid)
        candidates = indices[positions] + offset
        if variant == "C_CAUSAL_PULLBACK":
            retracement = sign * (
                closes[indices[positions]] - closes[candidates]
            ) / body[positions]
            accepted = (
                (retracement >= 0.10) & (retracement <= 0.40)
                & (sign * (closes[candidates] - midpoint[positions]) > 0.0)
            )
        else:
            extreme = highs[indices[positions]] if side == "LONG" else lows[indices[positions]]
            accepted = sign * (closes[candidates] - extreme) > 0.0
        decision[positions[accepted]] = candidates[accepted]
    return decision


def _side_features(row: pd.Series, side: str) -> dict[str, float | bool]:
    sign = 1.0 if side == "LONG" else -1.0
    side_clv = float(row.clv) if side == "LONG" else 1.0 - float(row.clv)
    side_rsi_space = 100.0 - float(row.rsi_6) if side == "LONG" else float(row.rsi_6)
    values: dict[str, float | bool] = {
        "side_clv": side_clv,
        "side_taker_imbalance": sign * float(row.taker_imbalance),
        "side_price_vs_ma25_atr": sign * float(row.price_vs_ma_25_atr),
        "side_ma25_slope_atr": sign * float(row.ma_25_slope_atr),
        "side_15m_return": sign * float(row.context_15m_return_1),
        "side_15m_ma25_slope_atr": sign * float(row.context_15m_ma_25_slope_atr),
        "side_btc_15m_return_atr": sign * float(row.btc_15m_return_1) / max(
            float(row.btc_15m_atr_fraction), 1e-12
        ),
        "side_rsi_space": side_rsi_space,
        "side_extension_ma25_atr": sign * float(row.price_vs_ma_25_atr),
        "side_directional_persistence_3": sign * float(row.directional_persistence_3),
        "side_directional_persistence_6": sign * float(row.directional_persistence_6),
    }
    ladder = True
    values["ladder_VOLUME_DIRECTION"] = ladder
    ladder = ladder and float(row.body_ratio) >= 0.60 and side_clv >= 0.70
    values["ladder_CLEAN_IMPULSE"] = ladder
    ladder = ladder and float(values["side_taker_imbalance"]) >= 0.10
    values["ladder_FLOW_ALIGNED"] = ladder
    ladder = (
        ladder and float(values["side_price_vs_ma25_atr"]) > 0.0
        and float(values["side_ma25_slope_atr"]) > 0.0
    )
    values["ladder_TREND_5M_ALIGNED"] = ladder
    ladder = (
        ladder and float(values["side_15m_return"]) > 0.0
        and float(values["side_15m_ma25_slope_atr"]) > 0.0
    )
    values["ladder_TREND_15M_ALIGNED"] = ladder
    ladder = ladder and float(values["side_btc_15m_return_atr"]) >= -0.10
    values["ladder_BTC_NOT_OPPOSING"] = ladder
    ladder = (
        ladder and float(values["side_rsi_space"]) >= 25.0
        and float(values["side_extension_ma25_atr"]) <= 2.0
    )
    values["ladder_SPACE_REMAINING"] = ladder
    return values


def build_wave_events(
    features: pd.DataFrame,
    config: Mapping[str, object],
    *,
    maximum_future_bars: int = 6,
    minimum_volume_ratio: float | None = None,
    candidate_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Create causal W1 events and keep future path strictly as label columns."""

    if maximum_future_bars < 1 or features.empty:
        raise VolumeWaveContractError("AEGIS_W1_EVENT_INPUT_INVALID")
    required = {
        "symbol", "open_time_ms", "close_time_ms", "open", "high", "low", "close",
        "atr", "body", "body_ratio", "body_atr", "clv", "volume_ratio_20",
        "taker_imbalance", "rsi_6", "price_vs_ma_25_atr", "ma_25_slope_atr",
        "context_15m_return_1", "context_15m_ma_25_slope_atr",
        "btc_15m_return_1", "btc_15m_atr_fraction", "directional_persistence_3",
        "directional_persistence_6", "path_efficiency_3", "path_efficiency_6",
    }
    if not required.issubset(features.columns):
        raise VolumeWaveContractError("AEGIS_W1_FEATURE_CONTRACT_MISSING")
    frame = features.reset_index(drop=True)
    records = []
    minimum_volume = (
        float(config["candidate_population"]["minimum_volume_ratio_20"])
        if minimum_volume_ratio is None else float(minimum_volume_ratio)
    )
    if not math.isfinite(minimum_volume) or minimum_volume < 0.0:
        raise VolumeWaveContractError("AEGIS_W1_MINIMUM_VOLUME_INVALID")
    minimum_body = float(config["candidate_population"]["minimum_absolute_body_atr"])
    allowed = np.arange(len(frame), dtype=np.int64)
    if candidate_indices is not None:
        allowed = np.asarray(candidate_indices, dtype=np.int64)
        if len(allowed) and (allowed.min() < 0 or allowed.max() >= len(frame)):
            raise VolumeWaveContractError("AEGIS_W1_CANDIDATE_INDEX_INVALID")
    eligible = (
        frame["body_atr"].ge(minimum_body)
        & frame["volume_ratio_20"].ge(minimum_volume)
        & frame["body"].ne(0.0)
        & np.isfinite(frame["body_atr"])
        & np.isfinite(frame["volume_ratio_20"])
    ).to_numpy()
    allowed = allowed[eligible[allowed]]
    open_times = frame["open_time_ms"].to_numpy(dtype=np.int64)
    output_frames = []
    for side in SIDES:
        sign = 1.0 if side == "LONG" else -1.0
        side_indices = allowed[(sign * frame["body"].to_numpy()[allowed]) > 0.0]
        for variant in ENTRY_VARIANTS:
            decisions = _variant_decision_indices(frame, side_indices, side, variant)
            valid = decisions >= 0
            impulses = side_indices[valid]
            decisions = decisions[valid]
            entries = decisions + 1
            valid = entries + maximum_future_bars <= len(frame)
            impulses, decisions, entries = impulses[valid], decisions[valid], entries[valid]
            if not len(impulses):
                continue
            contiguous = (
                open_times[entries + maximum_future_bars - 1] - open_times[entries]
                == (maximum_future_bars - 1) * 300_000
            )
            impulses = impulses[contiguous]
            decisions = decisions[contiguous]
            entries = entries[contiguous]
            decision_atr = frame["atr"].to_numpy(dtype=np.float64)[decisions]
            finite = np.isfinite(decision_atr) & (decision_atr > 0.0)
            impulses, decisions, entries = impulses[finite], decisions[finite], entries[finite]
            decision_atr = decision_atr[finite]
            if not len(impulses):
                continue
            result = frame.iloc[impulses].copy().reset_index(drop=True)
            result["schema_version"] = SCHEMA_VERSION
            result["side"] = side
            result["entry_variant"] = variant
            result["event_timestamp_ms"] = result["close_time_ms"].astype(np.int64)
            result["decision_timestamp_ms"] = frame["close_time_ms"].to_numpy()[decisions]
            result["entry_timestamp_ms"] = open_times[entries]
            result["entry_price"] = frame["open"].to_numpy(dtype=np.float64)[entries]
            result["entry_atr"] = decision_atr
            result["confirmation_bars"] = decisions - impulses
            side_clv = result["clv"] if side == "LONG" else 1.0 - result["clv"]
            result["side_clv"] = side_clv
            result["side_taker_imbalance"] = sign * result["taker_imbalance"]
            result["side_price_vs_ma25_atr"] = sign * result["price_vs_ma_25_atr"]
            result["side_ma25_slope_atr"] = sign * result["ma_25_slope_atr"]
            result["side_15m_return"] = sign * result["context_15m_return_1"]
            result["side_15m_ma25_slope_atr"] = sign * result["context_15m_ma_25_slope_atr"]
            result["side_btc_15m_return_atr"] = (
                sign * result["btc_15m_return_1"]
                / result["btc_15m_atr_fraction"].clip(lower=1e-12)
            )
            result["side_rsi_space"] = (
                100.0 - result["rsi_6"] if side == "LONG" else result["rsi_6"]
            )
            result["side_extension_ma25_atr"] = sign * result["price_vs_ma_25_atr"]
            result["side_directional_persistence_3"] = sign * result["directional_persistence_3"]
            result["side_directional_persistence_6"] = sign * result["directional_persistence_6"]
            ladder = pd.Series(True, index=result.index)
            result["ladder_VOLUME_DIRECTION"] = ladder
            ladder &= result["body_ratio"].ge(0.60) & side_clv.ge(0.70)
            result["ladder_CLEAN_IMPULSE"] = ladder
            ladder &= result["side_taker_imbalance"].ge(0.10)
            result["ladder_FLOW_ALIGNED"] = ladder
            ladder &= result["side_price_vs_ma25_atr"].gt(0.0) & result["side_ma25_slope_atr"].gt(0.0)
            result["ladder_TREND_5M_ALIGNED"] = ladder
            ladder &= result["side_15m_return"].gt(0.0) & result["side_15m_ma25_slope_atr"].gt(0.0)
            result["ladder_TREND_15M_ALIGNED"] = ladder
            ladder &= result["side_btc_15m_return_atr"].ge(-0.10)
            result["ladder_BTC_NOT_OPPOSING"] = ladder
            ladder &= result["side_rsi_space"].ge(25.0) & result["side_extension_ma25_atr"].le(2.0)
            result["ladder_SPACE_REMAINING"] = ladder
            highs = frame["high"].to_numpy(dtype=np.float64)
            lows = frame["low"].to_numpy(dtype=np.float64)
            closes = frame["close"].to_numpy(dtype=np.float64)
            future_fields = {
                "taker_imbalance": frame["taker_imbalance"].to_numpy(dtype=np.float64),
                "quote_volume": frame["quote_volume"].to_numpy(dtype=np.float64),
                "velocity_atr_1": frame["velocity_atr_1"].to_numpy(dtype=np.float64),
                "delta_velocity": frame["delta_velocity"].to_numpy(dtype=np.float64),
                "ma25_slope_atr": frame["ma_25_slope_atr"].to_numpy(dtype=np.float64),
                "rsi_6": frame["rsi_6"].to_numpy(dtype=np.float64),
            }
            for offset in range(1, maximum_future_bars + 1):
                future = entries + offset - 1
                result[f"future_high_{offset}"] = highs[future]
                result[f"future_low_{offset}"] = lows[future]
                result[f"future_close_{offset}"] = closes[future]
                for name, values in future_fields.items():
                    result[f"future_{name}_{offset}"] = values[future]
            output_frames.append(result)
    result = pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame()
    if result.empty:
        return result
    result.sort_values(
        ["event_timestamp_ms", "symbol", "side", "entry_variant"], inplace=True
    )
    result.reset_index(drop=True, inplace=True)
    return result


def collapse_event_cooldown(events: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    if cooldown_bars < 1 or events.empty:
        raise VolumeWaveContractError("AEGIS_W1_COOLDOWN_INPUT_INVALID")
    keep = []
    last: dict[tuple[str, str, str], int] = {}
    for row in events.sort_values(
        ["event_timestamp_ms", "symbol", "side", "entry_variant"]
    ).itertuples(index=False):
        key = (str(row.symbol), str(row.side), str(row.entry_variant))
        timestamp = int(row.event_timestamp_ms)
        if key in last and timestamp - last[key] < cooldown_bars * 300_000:
            continue
        last[key] = timestamp
        keep.append(row._asdict())
    return pd.DataFrame(keep, columns=events.columns)


def deterministic_matched_controls(
    broad_events: pd.DataFrame,
    wave_events: pd.DataFrame,
    *,
    minimum_volume_ratio: float,
) -> pd.DataFrame:
    """Match non-wave controls by symbol, side, entry variant and UTC month."""

    if broad_events.empty or wave_events.empty or minimum_volume_ratio <= 0.0:
        raise VolumeWaveContractError("AEGIS_W1_MATCHED_CONTROL_INPUT_INVALID")
    broad = broad_events.copy()
    selected = wave_events.copy()
    for values in (broad, selected):
        values["event_month"] = pd.to_datetime(
            values["event_timestamp_ms"], unit="ms", utc=True
        ).dt.strftime("%Y-%m")
    samples = []
    keys = ["symbol", "side", "entry_variant", "event_month"]
    for identity, events in selected.groupby(keys, sort=True):
        mask = np.ones(len(broad), dtype=bool)
        for column, value in zip(keys, identity, strict=True):
            mask &= broad[column].eq(value).to_numpy()
        candidates = broad.loc[mask].copy()
        if len(candidates) < len(events):
            raise VolumeWaveContractError("AEGIS_W1_MATCHED_CONTROL_INSUFFICIENT")
        stable_identity = (
            candidates["event_timestamp_ms"].astype(str) + ":"
            + candidates["symbol"] + ":" + candidates["side"] + ":"
            + candidates["entry_variant"]
        )
        candidates["control_order"] = pd.util.hash_pandas_object(
            stable_identity, index=False, hash_key="1814011814011814"
        ).to_numpy(dtype=np.uint64)
        samples.append(
            candidates.sort_values(["control_order", "event_timestamp_ms"])
            .head(len(events)).drop(columns="control_order")
        )
    if not samples:
        raise VolumeWaveContractError("AEGIS_W1_MATCHED_CONTROL_UNAVAILABLE")
    result = pd.concat(samples, ignore_index=True)
    result["sample_source"] = "MATCHED_PRICE_ONLY_CONTROL"
    return result


def path_outcomes(
    events: pd.DataFrame,
    *,
    horizon_bars: int,
    favorable_atr: float,
    adverse_atr: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Evaluate one ATR contract with adverse-first same-bar semantics."""

    values = (favorable_atr, adverse_atr, cost_bps)
    if horizon_bars < 1 or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise VolumeWaveContractError("AEGIS_W1_PATH_CONTRACT_INVALID")
    if favorable_atr <= 0.0 or adverse_atr <= 0.0:
        raise VolumeWaveContractError("AEGIS_W1_PATH_BARRIER_INVALID")
    required = {"side", "entry_price", "entry_atr"}
    required.update(f"future_{field}_{offset}" for field in ("high", "low", "close") for offset in range(1, horizon_bars + 1))
    if not required.issubset(events.columns):
        raise VolumeWaveContractError("AEGIS_W1_PATH_COLUMN_MISSING")
    entry = events["entry_price"].to_numpy(dtype=np.float64)
    atr = events["entry_atr"].to_numpy(dtype=np.float64)
    sign = np.where(events["side"].eq("LONG"), 1.0, -1.0)
    highs = events[[f"future_high_{index}" for index in range(1, horizon_bars + 1)]].to_numpy(dtype=np.float64)
    lows = events[[f"future_low_{index}" for index in range(1, horizon_bars + 1)]].to_numpy(dtype=np.float64)
    closes = events[[f"future_close_{index}" for index in range(1, horizon_bars + 1)]].to_numpy(dtype=np.float64)
    favorable = np.where(sign[:, None] > 0.0, highs - entry[:, None], entry[:, None] - lows) / atr[:, None]
    adverse = np.where(sign[:, None] > 0.0, entry[:, None] - lows, highs - entry[:, None]) / atr[:, None]
    favorable_hit = favorable >= favorable_atr
    adverse_hit = adverse >= adverse_atr
    favorable_any = favorable_hit.any(axis=1)
    adverse_any = adverse_hit.any(axis=1)
    favorable_time = np.where(favorable_any, favorable_hit.argmax(axis=1) + 1, 0)
    adverse_time = np.where(adverse_any, adverse_hit.argmax(axis=1) + 1, 0)
    favorable_first = favorable_any & (~adverse_any | (favorable_time < adverse_time))
    adverse_first = adverse_any & (~favorable_any | (adverse_time <= favorable_time))
    terminal = sign * (closes[:, -1] - entry) / entry
    close_path = np.column_stack([entry, closes])
    path_length = np.abs(np.diff(close_path, axis=1)).sum(axis=1)
    net_displacement = sign * (closes[:, -1] - entry)
    directional_persistence = np.divide(
        net_displacement, path_length,
        out=np.zeros_like(net_displacement), where=path_length > 0.0,
    )
    path_efficiency = np.divide(
        np.abs(net_displacement), path_length,
        out=np.zeros_like(net_displacement), where=path_length > 0.0,
    )
    favorable_fraction = favorable_atr * atr / entry
    adverse_fraction = adverse_atr * atr / entry
    cost = cost_bps / 10_000.0
    utility = np.where(
        favorable_first, favorable_fraction - cost,
        np.where(adverse_first, -adverse_fraction - cost, terminal - cost),
    )
    favorable_net = favorable * atr[:, None] / entry[:, None] > cost
    favorable_net_any = favorable_net.any(axis=1)
    time_to_positive = np.where(
        favorable_net_any, favorable_net.argmax(axis=1) + 1, 0
    )
    return pd.DataFrame({
        "favorable_before_adverse": favorable_first,
        "adverse_before_or_same": adverse_first,
        "mfe_atr": favorable.max(axis=1),
        "mae_atr": adverse.max(axis=1),
        "mfe_fraction": favorable.max(axis=1) * atr / entry,
        "mae_fraction": adverse.max(axis=1) * atr / entry,
        "time_to_mfe_bars": favorable.argmax(axis=1) + 1,
        "time_to_mae_bars": adverse.argmax(axis=1) + 1,
        "time_to_first_positive_net_bars": time_to_positive,
        "mfe_before_mae": favorable.argmax(axis=1) < adverse.argmax(axis=1),
        "directional_persistence": directional_persistence,
        "path_efficiency": path_efficiency,
        "terminal_side_return": terminal,
        "net_utility": utility,
    }, index=events.index)


def contract_identity(horizon: int, favorable: float, adverse: float) -> str:
    return f"H{horizon}_F{round(favorable * 100):03d}_A{round(adverse * 100):03d}"


def registered_contracts(config: Mapping[str, object]) -> tuple[tuple[str, int, float, float], ...]:
    contracts = []
    for horizon in config["targets"]["horizons_5m_bars"]:
        for favorable, adverse in config["targets"]["atr_barrier_pairs"]:
            contracts.append((
                contract_identity(int(horizon), float(favorable), float(adverse)),
                int(horizon), float(favorable), float(adverse),
            ))
    if len({item[0] for item in contracts}) != len(contracts):
        raise VolumeWaveContractError("AEGIS_W1_CONTRACT_ID_DUPLICATE")
    return tuple(contracts)


def clustered_economic_metrics(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    repetitions: int = 10_000,
    seed: int = 181401,
) -> dict[str, object]:
    if len(events) != len(outcomes) or events.empty or repetitions < 100:
        raise VolumeWaveContractError("AEGIS_W1_METRICS_INPUT_INVALID")
    utility = outcomes["net_utility"].to_numpy(dtype=np.float64)
    if not np.isfinite(utility).all():
        raise VolumeWaveContractError("AEGIS_W1_UTILITY_NON_FINITE")
    day = pd.to_datetime(events["event_timestamp_ms"], unit="ms", utc=True).dt.floor("1D")
    daily = pd.DataFrame({
        "day": day, "utility": utility,
        "gain": np.clip(utility, 0.0, None),
        "loss": np.clip(-utility, 0.0, None),
    }).groupby("day", sort=True).agg(
        total=("utility", "sum"), count=("utility", "size"),
        gain=("gain", "sum"), loss=("loss", "sum"),
    )
    random = np.random.default_rng(seed)
    draws = random.integers(0, len(daily), size=(repetitions, len(daily)))
    sampled_total = daily["total"].to_numpy()[draws].sum(axis=1)
    sampled_count = daily["count"].to_numpy()[draws].sum(axis=1)
    sampled_expectancy = sampled_total / sampled_count
    sampled_gain = daily["gain"].to_numpy()[draws].sum(axis=1)
    sampled_loss = daily["loss"].to_numpy()[draws].sum(axis=1)
    sampled_pf = np.divide(
        sampled_gain, sampled_loss,
        out=np.full_like(sampled_gain, np.inf), where=sampled_loss > 0.0,
    )
    ordered_pf = np.sort(sampled_pf)
    pf_lower = ordered_pf[int((repetitions - 1) * 0.025)]
    pf_upper = ordered_pf[int((repetitions - 1) * 0.975)]
    gains = utility[utility > 0.0].sum()
    losses = -utility[utility < 0.0].sum()
    ordered = events.assign(utility=utility).sort_values(
        ["event_timestamp_ms", "symbol", "side"]
    )
    equity = ordered["utility"].cumsum().to_numpy()
    drawdown = np.maximum.accumulate(np.maximum(equity, 0.0)) - equity
    tail = max(1, math.ceil(len(utility) * 0.05))
    wins = int(outcomes["favorable_before_adverse"].sum())
    trials = int(len(outcomes))
    posterior = beta_distribution(wins + 1, trials - wins + 1)
    monthly = ordered.assign(
        month=pd.to_datetime(ordered["event_timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
    ).groupby("month", sort=True)["utility"].mean()
    symbol_expectancy = ordered.groupby("symbol", sort=True)["utility"].mean()
    symbol_share = ordered["symbol"].value_counts(normalize=True)
    return {
        "events": len(utility),
        "net_expectancy": float(utility.mean()),
        "expectancy_ci_95": [
            float(np.quantile(sampled_expectancy, 0.025)),
            float(np.quantile(sampled_expectancy, 0.975)),
        ],
        "bootstrap_probability_expectancy_le_zero": float(
            (np.count_nonzero(sampled_expectancy <= 0.0) + 1) / (repetitions + 1)
        ),
        "profit_factor": float(gains / losses) if losses > 0.0 else "INF",
        "profit_factor_ci_95": [
            float(pf_lower), float(pf_upper),
        ],
        "win_rate_net": float((utility > 0.0).mean()),
        "favorable_before_adverse_rate": wins / trials,
        "continuation_posterior_mean": float(posterior.mean()),
        "continuation_credible_interval_95": [
            float(posterior.ppf(0.025)), float(posterior.ppf(0.975)),
        ],
        "mean_mfe_fraction": float(outcomes["mfe_fraction"].mean()),
        "median_mfe_fraction": float(outcomes["mfe_fraction"].median()),
        "mean_mae_fraction": float(outcomes["mae_fraction"].mean()),
        "median_mae_fraction": float(outcomes["mae_fraction"].median()),
        "mean_mfe_atr": float(outcomes["mfe_atr"].mean()),
        "mean_mae_atr": float(outcomes["mae_atr"].mean()),
        "mean_directional_persistence": float(outcomes["directional_persistence"].mean()),
        "mean_path_efficiency": float(outcomes["path_efficiency"].mean()),
        "mfe_before_mae_rate": float(outcomes["mfe_before_mae"].mean()),
        "maximum_additive_drawdown": float(drawdown.max(initial=0.0)),
        "cvar_05": float(np.sort(utility)[:tail].mean()),
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "positive_symbols": int(symbol_expectancy.gt(0.0).sum()),
        "maximum_symbol_share": float(symbol_share.max()),
        "symbol_expectancy": {
            str(symbol): float(value) for symbol, value in symbol_expectancy.items()
        },
    }


def benjamini_hochberg(
    pvalues: Mapping[str, float], *, false_discovery_rate: float = 0.05
) -> dict[str, bool]:
    if (
        not pvalues or not 0.0 < false_discovery_rate < 1.0
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in pvalues.values())
    ):
        raise VolumeWaveContractError("AEGIS_W1_FDR_INPUT_INVALID")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    maximum_rank = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= false_discovery_rate * rank / len(ordered):
            maximum_rank = rank
    accepted = {key for key, _ in ordered[:maximum_rank]}
    return {key: key in accepted for key in pvalues}
