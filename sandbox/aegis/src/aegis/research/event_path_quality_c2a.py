"""Causal taker-flow features and future-only path outcomes for C2A."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .market_event_fast_track_m1a import FlowBucket, MinuteBar


SCHEMA_VERSION = "aegis-event-path-quality-c2a-dataset-v1"
FLOW_WINDOWS = (1, 3, 5, 15, 60)
ZSCORE_LOOKBACK = 1_440


class C2AContractError(ValueError):
    pass


AGG_TRADE_COLUMNS = (
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
)


def read_agg_trade_archives_chunked(
    paths: Sequence[Path], symbol: str, *, chunk_size: int = 1_000_000
) -> tuple[FlowBucket, ...]:
    """Aggregate verified archives by minute without materializing every trade."""

    if not paths or not symbol or chunk_size <= 0:
        raise C2AContractError("AEGIS_C2A_AGG_ARCHIVE_INPUT_INVALID")
    grouped_parts = []
    for path in paths:
        try:
            with zipfile.ZipFile(path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) != 1:
                    raise C2AContractError("AEGIS_C2A_AGG_ARCHIVE_MEMBER_INVALID")
                with archive.open(members[0]) as source:
                    chunks = pd.read_csv(source, chunksize=chunk_size)
                    for chunk in chunks:
                        if tuple(chunk.columns) != AGG_TRADE_COLUMNS:
                            raise C2AContractError("AEGIS_C2A_AGG_ARCHIVE_HEADER_INVALID")
                        numeric = chunk[["price", "quantity", "transact_time"]].apply(
                            pd.to_numeric, errors="coerce"
                        )
                        maker = chunk["is_buyer_maker"].astype(str).str.lower()
                        timestamps = numeric["transact_time"].to_numpy(dtype=np.float64)
                        timestamps = np.where(timestamps >= 100_000_000_000_000, timestamps // 1000, timestamps)
                        if (
                            not np.isfinite(numeric.to_numpy()).all()
                            or (numeric[["price", "quantity"]].to_numpy() <= 0.0).any()
                            or (timestamps < 1_000_000_000_000).any()
                            or (timestamps >= 10_000_000_000_000).any()
                            or not maker.isin(("true", "false")).all()
                        ):
                            raise C2AContractError("AEGIS_C2A_AGG_ARCHIVE_ROW_INVALID")
                        notional = numeric["price"] * numeric["quantity"]
                        normalized = pd.DataFrame({
                            "open_time_ms": (timestamps.astype(np.int64) // 60_000) * 60_000,
                            "aggressive_buy_quote": np.where(maker.eq("false"), notional, 0.0),
                            "aggressive_sell_quote": np.where(maker.eq("true"), notional, 0.0),
                            "trade_count": 1,
                        })
                        grouped_parts.append(
                            normalized.groupby("open_time_ms", as_index=False, sort=True).sum()
                        )
        except zipfile.BadZipFile as error:
            raise C2AContractError("AEGIS_C2A_AGG_ARCHIVE_ZIP_INVALID") from error
    combined = (
        pd.concat(grouped_parts, ignore_index=True)
        .groupby("open_time_ms", as_index=False, sort=True).sum()
    )
    return tuple(
        FlowBucket(
            symbol, int(row.open_time_ms), float(row.aggressive_buy_quote),
            float(row.aggressive_sell_quote), int(row.trade_count),
        )
        for row in combined.itertuples(index=False)
    )


@dataclass(frozen=True)
class PathContract:
    horizon_minutes: int
    favorable_fraction: float
    adverse_fraction: float
    cost_fraction: float

    def __post_init__(self) -> None:
        values = (self.favorable_fraction, self.adverse_fraction, self.cost_fraction)
        if (
            self.horizon_minutes <= 0
            or not all(math.isfinite(value) and value >= 0.0 for value in values)
            or min(self.favorable_fraction, self.adverse_fraction) <= 0.0
        ):
            raise C2AContractError("AEGIS_C2A_PATH_CONTRACT_INVALID")

    @property
    def identity(self) -> str:
        favorable = round(self.favorable_fraction * 10_000)
        adverse = round(self.adverse_fraction * 10_000)
        return f"H{self.horizon_minutes}_F{favorable:03d}_A{adverse:03d}"


def _side_path_outcomes(
    frame: pd.DataFrame, side: str, contract: PathContract
) -> pd.DataFrame:
    if side not in {"LONG", "SHORT"}:
        raise C2AContractError("AEGIS_C2A_SIDE_INVALID")
    required = ("open", "high", "low", "close")
    if any(name not in frame for name in required):
        raise C2AContractError("AEGIS_C2A_PRICE_COLUMN_MISSING")
    prices = frame.loc[:, required].to_numpy(dtype=np.float64)
    if (
        len(prices) <= contract.horizon_minutes
        or not np.isfinite(prices).all()
        or np.min(prices) <= 0.0
    ):
        raise C2AContractError("AEGIS_C2A_PRICE_PATH_INVALID")
    horizon = contract.horizon_minutes
    entry = prices[1 : len(prices) - horizon + 1, 0]
    high = np.lib.stride_tricks.sliding_window_view(prices[1:, 1], horizon)
    low = np.lib.stride_tricks.sliding_window_view(prices[1:, 2], horizon)
    close = np.lib.stride_tricks.sliding_window_view(prices[1:, 3], horizon)
    if side == "LONG":
        favorable_path = high / entry[:, None] - 1.0
        adverse_path = 1.0 - low / entry[:, None]
        terminal = close[:, -1] / entry - 1.0
    else:
        favorable_path = 1.0 - low / entry[:, None]
        adverse_path = high / entry[:, None] - 1.0
        terminal = 1.0 - close[:, -1] / entry

    favorable_hit = favorable_path >= contract.favorable_fraction
    adverse_hit = adverse_path >= contract.adverse_fraction
    favorable_any = favorable_hit.any(axis=1)
    adverse_any = adverse_hit.any(axis=1)
    favorable_time = np.where(favorable_any, favorable_hit.argmax(axis=1) + 1, 0)
    adverse_time = np.where(adverse_any, adverse_hit.argmax(axis=1) + 1, 0)
    favorable_first = favorable_any & (~adverse_any | (favorable_time < adverse_time))
    adverse_first = adverse_any & (~favorable_any | (adverse_time <= favorable_time))
    outcome = np.full(len(entry), "NEITHER_REACHED", dtype=object)
    outcome[favorable_first] = "FAVORABLE_FIRST"
    outcome[adverse_first] = "ADVERSE_FIRST_OR_SAME"

    mfe = favorable_path.max(axis=1)
    mae = adverse_path.max(axis=1)
    time_to_mfe = favorable_path.argmax(axis=1) + 1
    time_to_mae = adverse_path.argmax(axis=1) + 1
    first_positive = favorable_path > contract.cost_fraction
    first_positive_any = first_positive.any(axis=1)
    time_to_positive = np.where(
        first_positive_any, first_positive.argmax(axis=1) + 1, 0
    )
    utility = np.where(
        favorable_first,
        contract.favorable_fraction - contract.cost_fraction,
        np.where(
            adverse_first,
            -contract.adverse_fraction - contract.cost_fraction,
            terminal - contract.cost_fraction,
        ),
    )
    return pd.DataFrame({
        "entry_price": entry,
        "barrier_outcome": outcome,
        "favorable_before_adverse": favorable_first,
        "mae_fraction": mae,
        "mfe_fraction": mfe,
        "mfe_before_mae": time_to_mfe < time_to_mae,
        "time_to_first_favorable_minutes": np.where(
            first_positive_any, time_to_positive, np.nan
        ),
        "time_to_mfe_minutes": time_to_mfe,
        "terminal_side_return": terminal,
        "net_utility": utility,
    })


def _base_frame(
    bars: Sequence[MinuteBar], flow: Sequence[FlowBucket]
) -> pd.DataFrame:
    if not bars or not flow:
        raise C2AContractError("AEGIS_C2A_SOURCE_EMPTY")
    symbols = {row.symbol for row in bars} | {row.symbol for row in flow}
    if len(symbols) != 1:
        raise C2AContractError("AEGIS_C2A_SYMBOL_MISMATCH")
    price = pd.DataFrame({
        "symbol": [row.symbol for row in bars],
        "open_time_ms": [row.open_time_ms for row in bars],
        "open": [row.open for row in bars],
        "high": [row.high for row in bars],
        "low": [row.low for row in bars],
        "close": [row.close for row in bars],
        "quote_volume": [row.quote_volume for row in bars],
        "kline_trade_count": [row.trade_count for row in bars],
    })
    micro = pd.DataFrame({
        "open_time_ms": [row.open_time_ms for row in flow],
        "taker_buy_quote": [row.aggressive_buy_quote for row in flow],
        "taker_sell_quote": [row.aggressive_sell_quote for row in flow],
        "agg_trade_count": [row.trade_count for row in flow],
    })
    if price.open_time_ms.duplicated().any() or micro.open_time_ms.duplicated().any():
        raise C2AContractError("AEGIS_C2A_SOURCE_DUPLICATE")
    frame = price.merge(micro, on="open_time_ms", how="inner", validate="one_to_one")
    frame.sort_values("open_time_ms", inplace=True, ignore_index=True)
    if frame.empty:
        raise C2AContractError("AEGIS_C2A_SOURCE_ALIGNMENT_EMPTY")
    frame["event_timestamp_ms"] = frame["open_time_ms"] + 60_000 - 1
    frame["flow_quote"] = frame["taker_buy_quote"] + frame["taker_sell_quote"]
    frame["signed_flow_quote"] = frame["taker_buy_quote"] - frame["taker_sell_quote"]
    frame["taker_imbalance"] = frame["signed_flow_quote"] / frame["flow_quote"]
    if not np.isfinite(frame[["flow_quote", "taker_imbalance"]].to_numpy()).all():
        raise C2AContractError("AEGIS_C2A_FLOW_INVALID")
    return frame


def _causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    flow = result["signed_flow_quote"]
    count = result["agg_trade_count"].astype(float)
    lagged_flow = flow.shift(1)
    lagged_count = count.shift(1)
    flow_mean = lagged_flow.rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK).mean()
    flow_std = lagged_flow.rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK).std(ddof=0)
    count_mean = lagged_count.rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK).mean()
    count_std = lagged_count.rolling(ZSCORE_LOOKBACK, min_periods=ZSCORE_LOOKBACK).std(ddof=0)
    result["flow_z"] = (flow - flow_mean) / flow_std.replace(0.0, np.nan)
    result["trade_count_z"] = (count - count_mean) / count_std.replace(0.0, np.nan)
    for window in FLOW_WINDOWS:
        buy = result["taker_buy_quote"].rolling(window, min_periods=window).sum()
        sell = result["taker_sell_quote"].rolling(window, min_periods=window).sum()
        total = buy + sell
        result[f"flow_imbalance_{window}m"] = (buy - sell) / total.replace(0.0, np.nan)
        result[f"flow_persistence_{window}m"] = (
            np.sign(flow).rolling(window, min_periods=window).mean()
        )
    result["price_response_1m"] = result["close"] / result["open"] - 1.0
    result["flow_price_efficiency"] = (
        result["price_response_1m"] / result["taker_imbalance"].abs().replace(0.0, np.nan)
    )
    result["history_contiguous"] = (
        result["open_time_ms"]
        - result["open_time_ms"].shift(ZSCORE_LOOKBACK)
        == ZSCORE_LOOKBACK * 60_000
    )
    return result


def build_path_dataset(
    bars: Sequence[MinuteBar],
    flow: Sequence[FlowBucket],
    contracts: Sequence[PathContract],
) -> pd.DataFrame:
    if not contracts or len({contract.identity for contract in contracts}) != len(contracts):
        raise C2AContractError("AEGIS_C2A_CONTRACT_SET_INVALID")
    base = _causal_features(_base_frame(bars, flow))
    maximum_horizon = max(contract.horizon_minutes for contract in contracts)
    base["future_contiguous"] = (
        base["open_time_ms"].shift(-maximum_horizon)
        - base["open_time_ms"]
        == maximum_horizon * 60_000
    )
    feature_names = (
        "flow_z", "trade_count_z", "price_response_1m", "flow_price_efficiency",
        *(f"flow_imbalance_{window}m" for window in FLOW_WINDOWS),
        *(f"flow_persistence_{window}m" for window in FLOW_WINDOWS),
    )
    outputs = []
    for side in ("LONG", "SHORT"):
        result = base.copy()
        result["side"] = side
        multiplier = 1.0 if side == "LONG" else -1.0
        for name in feature_names:
            if name.startswith(("flow_", "price_response")):
                result[f"side_{name}"] = result[name] * multiplier
        for contract in contracts:
            outcomes = _side_path_outcomes(base, side, contract)
            prefix = contract.identity.lower()
            for name in outcomes:
                result.loc[: len(outcomes) - 1, f"{prefix}_{name}"] = outcomes[name].to_numpy()
        outputs.append(result)
    combined = pd.concat(outputs, ignore_index=True)
    required_features = [
        "flow_z", "trade_count_z", "price_response_1m", "flow_price_efficiency",
        *(f"flow_imbalance_{window}m" for window in FLOW_WINDOWS),
        *(f"flow_persistence_{window}m" for window in FLOW_WINDOWS),
    ]
    primary_outcome = f"{contracts[0].identity.lower()}_net_utility"
    combined = combined.loc[
        combined["history_contiguous"]
        & combined["future_contiguous"]
        & combined[required_features].notna().all(axis=1)
        & combined[primary_outcome].notna()
    ].copy()
    combined["schema_version"] = SCHEMA_VERSION
    combined.sort_values(["event_timestamp_ms", "symbol", "side"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def contracts_from_preregistration(config: Mapping[str, object]) -> tuple[PathContract, ...]:
    targets = config["targets"]
    economics = config["economics"]
    if not isinstance(targets, Mapping) or not isinstance(economics, Mapping):
        raise C2AContractError("AEGIS_C2A_CONFIG_INVALID")
    horizons = tuple(int(value) for value in targets["horizons_minutes"])
    favorable = tuple(float(value) / 10_000.0 for value in targets["favorable_barrier_bps"])
    adverse = tuple(float(value) / 10_000.0 for value in targets["adverse_barrier_bps"])
    if len(favorable) != len(adverse):
        raise C2AContractError("AEGIS_C2A_BARRIER_PAIR_INVALID")
    cost = float(economics["primary_cost_bps"]) / 10_000.0
    return tuple(
        PathContract(horizon, favorable_value, adverse_value, cost)
        for horizon in horizons
        for favorable_value, adverse_value in zip(favorable, adverse, strict=True)
    )


def detect_registered_events(
    rows: pd.DataFrame, config: Mapping[str, object]
) -> pd.DataFrame:
    detectors = config.get("event_detectors")
    if not isinstance(detectors, Mapping):
        raise C2AContractError("AEGIS_C2A_DETECTOR_CONFIG_MISSING")
    impulse = detectors.get("FLOW_IMPULSE_CONTINUATION")
    absorption = detectors.get("FLOW_ABSORPTION_REVERSAL")
    if not isinstance(impulse, Mapping) or not isinstance(absorption, Mapping):
        raise C2AContractError("AEGIS_C2A_DETECTOR_CONFIG_INVALID")
    required = {
        "event_timestamp_ms", "symbol", "side", "side_flow_z",
        "trade_count_z", "side_flow_imbalance_3m",
        "side_flow_persistence_5m", "side_price_response_1m",
    }
    if not required.issubset(rows.columns):
        raise C2AContractError("AEGIS_C2A_DETECTOR_FEATURE_MISSING")
    values = rows.copy()
    impulse_mask = (
        values["side_flow_z"].ge(float(impulse["side_flow_z_minimum"]))
        & values["trade_count_z"].ge(float(impulse["side_trade_count_z_minimum"]))
        & values["side_flow_imbalance_3m"].ge(float(impulse["side_flow_imbalance_3m_minimum"]))
        & values["side_flow_persistence_5m"].ge(float(impulse["side_flow_persistence_5m_minimum"]))
        & values["side_price_response_1m"].ge(float(impulse["side_price_response_1m_minimum"]))
    )
    absorption_mask = (
        values["side_flow_z"].le(-float(absorption["opposing_flow_z_minimum"]))
        & values["trade_count_z"].ge(float(absorption["opposing_trade_count_z_minimum"]))
        & values["side_price_response_1m"].ge(float(absorption["side_price_response_1m_minimum"]))
        & values["side_flow_imbalance_3m"].ge(float(absorption["side_flow_imbalance_3m_minimum"]))
    )
    parts = []
    if impulse_mask.any():
        parts.append(values.loc[impulse_mask].assign(event_family="FLOW_IMPULSE_CONTINUATION"))
    if absorption_mask.any():
        parts.append(values.loc[absorption_mask].assign(event_family="FLOW_ABSORPTION_REVERSAL"))
    if not parts:
        return values.iloc[0:0].assign(event_family=pd.Series(dtype="object"))
    return pd.concat(parts, ignore_index=True).sort_values(
        ["event_timestamp_ms", "symbol", "side", "event_family"], ignore_index=True
    )


def collapse_registered_events(rows: pd.DataFrame, cooldown_minutes: int) -> pd.DataFrame:
    if cooldown_minutes <= 0:
        raise C2AContractError("AEGIS_C2A_COOLDOWN_INVALID")
    selected = []
    last: dict[tuple[str, str, str], int] = {}
    for row in rows.sort_values(
        ["event_timestamp_ms", "symbol", "side", "event_family"]
    ).itertuples(index=False):
        key = (str(row.symbol), str(row.side), str(row.event_family))
        timestamp = int(row.event_timestamp_ms)
        if key in last and timestamp - last[key] < cooldown_minutes * 60_000:
            continue
        last[key] = timestamp
        selected.append(row._asdict())
    return pd.DataFrame(selected, columns=rows.columns)


def economic_summary(rows: pd.DataFrame, utility_column: str) -> Mapping[str, object]:
    if utility_column not in rows:
        raise C2AContractError("AEGIS_C2A_UTILITY_COLUMN_MISSING")
    values = rows[utility_column].to_numpy(dtype=np.float64)
    if not len(values):
        return {
            "events": 0, "net_expectancy": None, "profit_factor": None,
            "win_rate": None, "cvar_10": None,
        }
    if not np.isfinite(values).all():
        raise C2AContractError("AEGIS_C2A_UTILITY_NON_FINITE")
    wins, losses = values[values > 0.0], values[values < 0.0]
    tail_count = max(1, math.ceil(len(values) * 0.10))
    return {
        "events": len(values),
        "net_expectancy": float(values.mean()),
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if len(losses)
            else math.inf if len(wins) else 0.0
        ),
        "win_rate": float((values > 0.0).mean()),
        "cvar_10": float(np.sort(values)[:tail_count].mean()),
    }


def day_cluster_bootstrap(
    rows: pd.DataFrame, utility_column: str, *, repetitions: int = 1_000,
    seed: int = 181001,
) -> Mapping[str, float]:
    if repetitions <= 0 or not len(rows):
        raise C2AContractError("AEGIS_C2A_BOOTSTRAP_INPUT_INVALID")
    values = rows.assign(
        day=pd.to_datetime(rows["event_timestamp_ms"], unit="ms", utc=True).dt.floor("1D")
    )
    clusters = [
        group[utility_column].to_numpy(dtype=np.float64)
        for _, group in values.groupby("day", sort=True)
    ]
    if not clusters or any(not np.isfinite(cluster).all() for cluster in clusters):
        raise C2AContractError("AEGIS_C2A_BOOTSTRAP_CLUSTER_INVALID")
    random = np.random.default_rng(seed)
    expectancy = []
    for _ in range(repetitions):
        sample = np.concatenate([
            clusters[index] for index in random.integers(0, len(clusters), len(clusters))
        ])
        expectancy.append(float(sample.mean()))
    return {
        "expectancy_lower_95": float(np.quantile(expectancy, 0.025)),
        "expectancy_upper_95": float(np.quantile(expectancy, 0.975)),
    }


def deterministic_matched_control(
    population: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    """Match event counts by symbol and side without consulting outcomes."""

    if selected.empty:
        return population.iloc[0:0].copy()
    keys = set(zip(
        selected["event_timestamp_ms"], selected["symbol"], selected["side"], strict=True
    ))
    pool = population.loc[[
        (timestamp, symbol, side) not in keys
        for timestamp, symbol, side in zip(
            population["event_timestamp_ms"], population["symbol"],
            population["side"], strict=True,
        )
    ]].copy()
    samples = []
    for (symbol, side), events in selected.groupby(["symbol", "side"], sort=True):
        candidates = pool.loc[pool.symbol.eq(symbol) & pool.side.eq(side)].copy()
        if len(candidates) < len(events):
            raise C2AContractError("AEGIS_C2A_MATCHED_CONTROL_INSUFFICIENT")
        identity = (
            candidates["event_timestamp_ms"].astype(str) + ":"
            + candidates["symbol"] + ":" + candidates["side"]
        )
        candidates["control_order"] = pd.util.hash_pandas_object(
            identity, index=False, hash_key="1810011810011810"
        ).to_numpy(dtype=np.uint64)
        samples.append(
            candidates.sort_values(["control_order", "event_timestamp_ms"])
            .head(len(events)).drop(columns="control_order")
        )
    return pd.concat(samples, ignore_index=True)
