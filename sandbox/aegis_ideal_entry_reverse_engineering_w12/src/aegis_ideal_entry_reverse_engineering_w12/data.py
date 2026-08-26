"""Hash-verified source loading and causal feature/teacher panel construction."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SANDBOX = Path(__file__).resolve().parents[2]
REPOSITORY = Path(__file__).resolve().parents[4]
CONFIG_PATH = SANDBOX / "config" / "w12_frozen.json"
RAW_COLUMNS = (
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume",
)
FORBIDDEN_FEATURE_TOKENS = (
    "future", "label", "teacher", "mfe", "mae", "barrier", "quality_score",
    "ideal", "gross", "net", "outcome", "terminal", "time_to",
)


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def verify_source(config: Mapping[str, Any], repository: str | Path = REPOSITORY) -> dict[str, Any]:
    root = Path(repository).resolve()
    manifest_path = (root / config["source"]["manifest"]).resolve()
    candle_dir = (root / config["source"]["candle_dir"]).resolve()
    if manifest_path.parent != candle_dir:
        raise ValueError("source manifest must be inside the configured candle directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for symbol in config["source"]["symbols"]:
        source = manifest["symbols"][symbol]
        path = (root / source["parquet"]).resolve()
        if path != candle_dir / f"{symbol}_1m.parquet":
            raise ValueError(f"unexpected source path for {symbol}")
        actual = sha256_file(path)
        if actual != source["parquet_sha256"]:
            raise ValueError(f"source hash mismatch for {symbol}")
        if source["gaps"] != 0 or source["duplicates"] != 0:
            raise ValueError(f"configured interval source is not clean for {symbol}")
        records[symbol] = {
            "path": path.as_posix(), "sha256": actual, "rows": int(source["rows"]),
            "first_open_ms": int(source["first_open_ms"]),
            "last_open_ms": int(source["last_open_ms"]),
        }
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "symbols": records,
    }


def _rolling_efficiency(close: pd.Series, window: int) -> pd.Series:
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return (close - close.shift(window)).abs() / path.replace(0.0, np.nan)


def _rolling_persistence(close: pd.Series, window: int) -> pd.Series:
    direction = np.sign(close.diff())
    return direction.rolling(window, min_periods=window).mean()


def causal_symbol_features(raw: pd.DataFrame, symbol: str, config: Mapping[str, Any]) -> pd.DataFrame:
    """Build symbol-local features using only candles closed by each decision."""
    frame = raw.sort_values("open_time_ms", kind="mergesort").reset_index(drop=True).copy()
    if frame["open_time_ms"].duplicated().any() or not frame["open_time_ms"].diff().iloc[1:].eq(60_000).all():
        raise ValueError(f"non-contiguous source for {symbol}")
    open_time = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame["available_at"] = pd.to_datetime(frame["close_time_ms"] + 1, unit="ms", utc=True)
    close = frame["close"].astype(float)
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    volume = frame["volume"].astype(float)
    taker = frame["taker_buy_volume"].astype(float)
    output = pd.DataFrame(index=frame.index)
    output["decision_at"] = frame["available_at"].dt.floor("min")
    output["feature_available_at"] = frame["available_at"]
    output["symbol"] = symbol
    for minutes in config["features"]["lookbacks_minutes"]:
        output[f"return_{minutes}m_bps"] = (close / close.shift(int(minutes)) - 1.0) * 10_000.0
    output["return_acceleration_1m_bps"] = output["return_1m_bps"].diff()
    output["return_acceleration_5m_bps"] = output["return_5m_bps"] - output["return_5m_bps"].shift(5)
    for window in (5, 15, 30, 60):
        output[f"persistence_{window}m"] = _rolling_persistence(close, window)
        output[f"efficiency_{window}m"] = _rolling_efficiency(close, window)
    prior = close.shift(1)
    true_range = pd.concat((high - low, (high - prior).abs(), (low - prior).abs()), axis=1).max(axis=1)
    output["atr_15m_bps"] = true_range.rolling(15, min_periods=15).mean() / close * 10_000.0
    output["atr_60m_bps"] = true_range.rolling(60, min_periods=60).mean() / close * 10_000.0
    returns = close.pct_change(fill_method=None)
    output["realized_vol_15m_bps"] = returns.rolling(15, min_periods=15).std(ddof=0) * math.sqrt(15) * 10_000.0
    output["realized_vol_60m_bps"] = returns.rolling(60, min_periods=60).std(ddof=0) * math.sqrt(60) * 10_000.0
    output["vol_expansion_ratio"] = output["realized_vol_15m_bps"] / output["realized_vol_60m_bps"].replace(0.0, np.nan)
    range15 = high.rolling(15, min_periods=15).max() - low.rolling(15, min_periods=15).min()
    range60 = high.rolling(60, min_periods=60).max() - low.rolling(60, min_periods=60).min()
    output["range_15m_bps"] = range15 / close * 10_000.0
    output["range_60m_bps"] = range60 / close * 10_000.0
    output["compression_ratio"] = output["range_15m_bps"] / output["range_60m_bps"].replace(0.0, np.nan)
    prior_high = high.shift(1).rolling(60, min_periods=60).max()
    prior_low = low.shift(1).rolling(60, min_periods=60).min()
    output["distance_high_60m_bps"] = (prior_high - close) / close * 10_000.0
    output["distance_low_60m_bps"] = (close - prior_low) / close * 10_000.0
    output["range_position_60m"] = (close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
    output["breakout_up_60m"] = close.gt(prior_high).astype(float)
    output["breakout_down_60m"] = close.lt(prior_low).astype(float)
    for span in (7, 25, 60):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        output[f"ema{span}_distance_bps"] = (close / ema - 1.0) * 10_000.0
        output[f"ema{span}_slope_5m_bps"] = (ema / ema.shift(5) - 1.0) * 10_000.0
    volume_mean20 = volume.shift(1).rolling(20, min_periods=20).mean()
    volume_mean60 = volume.shift(1).rolling(60, min_periods=60).mean()
    output["relative_volume_20m"] = volume / volume_mean20
    output["relative_volume_60m"] = volume / volume_mean60
    output["volume_acceleration"] = output["relative_volume_20m"] - output["relative_volume_20m"].shift(5)
    output["volume_percentile_240m"] = volume.rolling(240, min_periods=120).rank(pct=True)
    imbalance = 2.0 * taker / volume.replace(0.0, np.nan) - 1.0
    output["taker_imbalance_1m"] = imbalance
    output["taker_imbalance_5m"] = taker.rolling(5, min_periods=5).sum() * 2.0 / volume.rolling(5, min_periods=5).sum() - 1.0
    output["taker_imbalance_15m"] = taker.rolling(15, min_periods=15).sum() * 2.0 / volume.rolling(15, min_periods=15).sum() - 1.0
    output["taker_persistence_15m"] = np.sign(imbalance).rolling(15, min_periods=15).mean()
    for lag in (1, 5, 15, 30, 60):
        output[f"return_1m_lag_{lag}m_bps"] = output["return_1m_bps"].shift(lag)
        output[f"taker_lag_{lag}m"] = output["taker_imbalance_1m"].shift(lag)
        output[f"volume_lag_{lag}m"] = output["relative_volume_20m"].shift(lag)
    cadence = int(config["source"]["decision_cadence_minutes"])
    start = pd.Timestamp(config["source"]["start_inclusive"])
    end = pd.Timestamp(config["source"]["end_exclusive"])
    mask = output["decision_at"].dt.minute.mod(cadence).eq(0) & output["decision_at"].ge(start) & output["decision_at"].lt(end)
    result = output.loc[mask].copy()
    result["source_row_index"] = result.index.astype(np.int64)
    if not result["feature_available_at"].le(result["decision_at"]).all():
        raise AssertionError("feature source was not closed by decision time")
    return result.reset_index(drop=True)


def _first_hit(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    any_hit = mask.any(axis=1)
    first = np.where(any_hit, mask.argmax(axis=1), -1)
    return any_hit, first


def teacher_labels(raw: pd.DataFrame, snapshots: pd.DataFrame, horizon: int, config: Mapping[str, Any]) -> pd.DataFrame:
    """Use future one-minute paths only to create side-specific labels and outcomes."""
    starts = snapshots["source_row_index"].to_numpy(np.int64) + 1
    valid = starts + horizon <= len(raw)
    rows = snapshots.loc[valid, ["decision_at", "symbol", "source_row_index"]].copy().reset_index(drop=True)
    starts = starts[valid]
    entry = raw["open"].to_numpy(float)[starts]
    offsets = np.arange(horizon, dtype=np.int64)
    positions = starts[:, None] + offsets[None, :]
    highs = raw["high"].to_numpy(float)[positions]
    lows = raw["low"].to_numpy(float)[positions]
    closes = raw["close"].to_numpy(float)[positions]
    teacher = config["teachers"]
    cost = float(config["costs_bps"]["baseline_total"])
    outputs = []
    for side, sign in (("LONG", 1.0), ("SHORT", -1.0)):
        favorable_path = (highs / entry[:, None] - 1.0) * 10_000.0 if sign > 0 else (1.0 - lows / entry[:, None]) * 10_000.0
        adverse_path = (1.0 - lows / entry[:, None]) * 10_000.0 if sign > 0 else (highs / entry[:, None] - 1.0) * 10_000.0
        mfe = favorable_path.max(axis=1)
        mae = adverse_path.max(axis=1)
        time_mfe = favorable_path.argmax(axis=1) + 1
        time_mae = adverse_path.argmax(axis=1) + 1
        favorable_hit, favorable_first_minute = _first_hit(favorable_path >= float(teacher["barrier"]["favorable_bps"]))
        adverse_hit, adverse_first_minute = _first_hit(adverse_path >= float(teacher["barrier"]["adverse_bps"]))
        favorable_first = favorable_hit & (~adverse_hit | (favorable_first_minute < adverse_first_minute))
        adverse_first = adverse_hit & (~favorable_hit | (adverse_first_minute <= favorable_first_minute))
        neither = ~(favorable_first | adverse_first)
        rr = mfe / (mae + float(teacher["risk_reward"]["mae_epsilon_bps"]))
        close_path = np.column_stack((entry, closes))
        path_length = np.abs(np.diff(close_path, axis=1)).sum(axis=1)
        directional_terminal = sign * (closes[:, -1] - entry) / entry * 10_000.0
        efficiency = np.divide(np.abs(closes[:, -1] - entry), path_length, out=np.zeros(len(entry)), where=path_length > 0)
        persistence = np.divide(sign * (closes[:, -1] - entry), path_length, out=np.zeros(len(entry)), where=path_length > 0)
        pre_mfe_adverse = np.array([adverse_path[index, : time_mfe[index]].max() for index in range(len(entry))])
        a = (mfe >= teacher["mfe_mae"]["minimum_mfe_bps"]) & (mae <= teacher["mfe_mae"]["maximum_mae_bps"]) & (rr >= teacher["mfe_mae"]["minimum_ratio"])
        b = favorable_first
        c = (rr >= teacher["risk_reward"]["minimum_ratio"]) & ((mfe - cost) >= teacher["risk_reward"]["minimum_net_room_bps"])
        d = (efficiency >= teacher["path"]["minimum_efficiency"]) & (time_mfe <= horizon * teacher["path"]["maximum_time_fraction_to_mfe"]) & (pre_mfe_adverse <= teacher["path"]["maximum_pre_mfe_adverse_bps"]) & (persistence >= teacher["path"]["minimum_directional_persistence"])
        e = (mfe >= teacher["economic"]["minimum_mfe_bps"]) & favorable_first
        votes = np.column_stack((a, b, c, d, e))
        weights = np.array([teacher["consensus"]["teacher_weights"][key] for key in "ABCDE"])
        economic_component = np.clip((mfe - cost) / 46.0, 0.0, 1.0)
        rr_component = np.clip(rr / 4.0, 0.0, 1.0)
        barrier_component = np.where(favorable_first, 1.0, np.where(neither, 0.25, 0.0))
        speed = np.clip(1.0 - (time_mfe - 1) / horizon, 0.0, 1.0)
        path_component = np.clip((efficiency + np.clip(persistence, 0, 1) + speed + np.clip(1.0 - pre_mfe_adverse / 30.0, 0, 1)) / 4.0, 0, 1)
        quality = 100.0 * (0.30 * economic_component + 0.25 * rr_component + 0.20 * barrier_component + 0.25 * path_component)
        realized_gross = np.where(favorable_first, teacher["barrier"]["favorable_bps"], np.where(adverse_first, -teacher["barrier"]["adverse_bps"], directional_terminal))
        result = rows.copy()
        result["side"] = side
        result["horizon_minutes"] = horizon
        result["mfe_bps"] = mfe
        result["mae_bps"] = mae
        result["time_to_mfe_minutes"] = time_mfe
        result["time_to_mae_minutes"] = time_mae
        result["pre_mfe_adverse_bps"] = pre_mfe_adverse
        result["path_efficiency"] = efficiency
        result["directional_persistence"] = persistence
        result["teacher_a_good"] = a
        result["teacher_b_good"] = b
        result["teacher_c_good"] = c
        result["teacher_d_good"] = d
        result["teacher_e_good"] = e
        result["majority_ideal"] = votes.sum(axis=1) >= int(teacher["consensus"]["majority_minimum"])
        result["strict_ideal"] = votes.sum(axis=1) >= int(teacher["consensus"]["strict_minimum"])
        result["weighted_ideal"] = (votes * weights).sum(axis=1) >= float(teacher["consensus"]["weighted_minimum"])
        result["entry_quality_score"] = quality
        result["barrier_outcome"] = np.where(favorable_first, "FAVORABLE_FIRST", np.where(adverse_first, "ADVERSE_FIRST", "NEITHER"))
        result["terminal_gross_bps"] = directional_terminal
        result["policy_gross_bps"] = realized_gross
        result["policy_net14_bps"] = realized_gross - 14.0
        result["policy_net20_bps"] = realized_gross - 20.0
        result["policy_net30_bps"] = realized_gross - 30.0
        result["outcome_available_at"] = result["decision_at"] + pd.Timedelta(minutes=horizon)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True).sort_values(["decision_at", "side"], kind="mergesort", ignore_index=True)


def assign_zones(labels: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    result = labels.sort_values(["symbol", "side", "horizon_minutes", "decision_at"], kind="mergesort").copy()
    result["zone_id"] = None
    result["zone_start"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["zone_end"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["zone_best"] = False
    max_gap = pd.Timedelta(minutes=int(config["zones"]["maximum_gap_minutes"]))
    for keys, group in result[result["majority_ideal"]].groupby(["symbol", "side", "horizon_minutes"], sort=True):
        ordered = group.sort_values("decision_at")
        sequence = ordered["decision_at"].diff().gt(max_gap).fillna(True).cumsum()
        for number, (_, zone) in enumerate(ordered.groupby(sequence, sort=True), 1):
            symbol, side, horizon = keys
            zone_id = f"W12_{symbol}_{side}_{horizon}M_{number:06d}"
            indices = zone.index
            best_index = zone.sort_values(["entry_quality_score", "decision_at"], ascending=[False, True], kind="mergesort").index[0]
            result.loc[indices, "zone_id"] = zone_id
            result.loc[indices, "zone_start"] = zone["decision_at"].min()
            result.loc[indices, "zone_end"] = zone["decision_at"].max()
            result.loc[best_index, "zone_best"] = True
    return result.sort_values(["decision_at", "symbol", "horizon_minutes", "side"], kind="mergesort", ignore_index=True)


def _symbol_task(arguments: tuple[str, str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol, path, config = arguments
    raw = pd.read_parquet(path, columns=list(RAW_COLUMNS))
    start_ms = int(pd.Timestamp(config["source"]["start_inclusive"]).timestamp() * 1000)
    end_ms = int(pd.Timestamp(config["source"]["end_exclusive"]).timestamp() * 1000)
    warmup_ms = start_ms - int(config["features"]["maximum_lookback_minutes"]) * 60_000
    raw = raw[raw["open_time_ms"].ge(warmup_ms) & raw["open_time_ms"].lt(end_ms)].reset_index(drop=True)
    features = causal_symbol_features(raw, symbol, config)
    labels = pd.concat(
        [teacher_labels(raw, features, int(horizon), config) for horizon in config["teachers"]["horizons_minutes"]],
        ignore_index=True,
    )
    return features, assign_zones(labels, config)


def add_cross_market_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.sort_values(["decision_at", "symbol"], kind="mergesort").copy()
    return15 = result.pivot(index="decision_at", columns="symbol", values="return_15m_bps")
    return1 = result.pivot(index="decision_at", columns="symbol", values="return_1m_bps")
    btc15, eth15 = return15["BTCUSDT"], return15["ETHUSDT"]
    breadth = (return15 > 0).mean(axis=1)
    dispersion = return15.std(axis=1, ddof=0)
    basket = return15.drop(columns=["BTCUSDT"]).mean(axis=1)
    rank = return15.rank(axis=1, pct=True)
    records = []
    for symbol, group in result.groupby("symbol", sort=True):
        working = group.set_index("decision_at").copy()
        local1 = return1[symbol]
        btc1 = return1["BTCUSDT"]
        covariance = local1.rolling(120, min_periods=60).cov(btc1)
        variance = btc1.rolling(120, min_periods=60).var()
        working["btc_return_15m_bps"] = btc15
        working["eth_return_15m_bps"] = eth15
        working["relative_to_btc_15m_bps"] = return15[symbol] - btc15
        working["relative_to_eth_15m_bps"] = return15[symbol] - eth15
        working["alt_basket_return_15m_bps"] = basket
        working["market_breadth_15m"] = breadth
        working["cross_sectional_dispersion_15m_bps"] = dispersion
        working["cross_sectional_rank_15m"] = rank[symbol]
        working["btc_beta_120m"] = covariance / variance.replace(0.0, np.nan)
        working["btc_correlation_120m"] = local1.rolling(120, min_periods=60).corr(btc1)
        records.append(working.reset_index())
    return pd.concat(records, ignore_index=True).sort_values(["decision_at", "symbol"], kind="mergesort", ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"decision_at", "feature_available_at", "symbol", "source_row_index"}
    columns = [column for column in frame.columns if column not in metadata]
    forbidden = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(f"future/label columns entered feature schema: {forbidden}")
    return columns


def build_panels(config: Mapping[str, Any], repository: str | Path = REPOSITORY, workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    authority = verify_source(config, repository)
    tasks = [(symbol, authority["symbols"][symbol]["path"], dict(config)) for symbol in config["source"]["symbols"]]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        results = list(executor.map(_symbol_task, tasks))
    features = add_cross_market_features(pd.concat([item[0] for item in results], ignore_index=True))
    labels = pd.concat([item[1] for item in results], ignore_index=True).sort_values(
        ["decision_at", "symbol", "horizon_minutes", "side"], kind="mergesort", ignore_index=True
    )
    names = feature_columns(features)
    if not features["feature_available_at"].le(features["decision_at"]).all():
        raise AssertionError("causal feature availability audit failed")
    schema = {"feature_version": config["features"]["version"], "feature_names": names, "feature_schema_sha256": canonical_hash(names)}
    return features, labels, {**authority, **schema}
