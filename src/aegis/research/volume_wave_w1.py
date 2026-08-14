"""Causal closed-candle volume-wave research primitives for W1."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


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
    feature_columns = [
        column for column in frame.columns
        if column not in {"symbol"} and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for index, impulse in frame.iterrows():
        if (
            not math.isfinite(float(impulse.body_atr))
            or float(impulse.body_atr) < minimum_body
            or not math.isfinite(float(impulse.volume_ratio_20))
            or float(impulse.volume_ratio_20) < minimum_volume
            or float(impulse.body) == 0.0
        ):
            continue
        side = "LONG" if float(impulse.body) > 0.0 else "SHORT"
        side_values = _side_features(impulse, side)
        for variant in ENTRY_VARIANTS:
            decision_index = _variant_decision_index(frame, index, side, variant)
            if decision_index is None:
                continue
            entry_index = decision_index + 1
            if entry_index + maximum_future_bars > len(frame):
                continue
            path = frame.iloc[entry_index : entry_index + maximum_future_bars]
            expected = int(frame.iloc[entry_index].open_time_ms) + np.arange(
                maximum_future_bars
            ) * 300_000
            if not np.array_equal(path["open_time_ms"].to_numpy(dtype=np.int64), expected):
                continue
            decision = frame.iloc[decision_index]
            if not math.isfinite(float(decision.atr)) or float(decision.atr) <= 0.0:
                continue
            record: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "symbol": str(impulse.symbol),
                "side": side,
                "entry_variant": variant,
                "event_timestamp_ms": int(impulse.close_time_ms),
                "decision_timestamp_ms": int(decision.close_time_ms),
                "entry_timestamp_ms": int(path.iloc[0].open_time_ms),
                "entry_price": float(path.iloc[0].open),
                "entry_atr": float(decision.atr),
                "confirmation_bars": decision_index - index,
                **side_values,
            }
            for column in feature_columns:
                value = impulse[column]
                if isinstance(value, (bool, np.bool_)):
                    record[column] = bool(value)
                elif pd.notna(value):
                    record[column] = float(value)
            for offset, future in enumerate(path.itertuples(index=False), start=1):
                record[f"future_high_{offset}"] = float(future.high)
                record[f"future_low_{offset}"] = float(future.low)
                record[f"future_close_{offset}"] = float(future.close)
            records.append(record)
    result = pd.DataFrame.from_records(records)
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
    broad = broad_events.loc[
        broad_events["volume_ratio_20"].lt(minimum_volume_ratio)
    ].copy()
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
        if candidates.empty:
            continue
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
    favorable_fraction = favorable_atr * atr / entry
    adverse_fraction = adverse_atr * atr / entry
    cost = cost_bps / 10_000.0
    utility = np.where(
        favorable_first, favorable_fraction - cost,
        np.where(adverse_first, -adverse_fraction - cost, terminal - cost),
    )
    return pd.DataFrame({
        "favorable_before_adverse": favorable_first,
        "adverse_before_or_same": adverse_first,
        "mfe_atr": favorable.max(axis=1),
        "mae_atr": adverse.max(axis=1),
        "mfe_fraction": favorable.max(axis=1) * atr / entry,
        "mae_fraction": adverse.max(axis=1) * atr / entry,
        "time_to_mfe_bars": favorable.argmax(axis=1) + 1,
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
