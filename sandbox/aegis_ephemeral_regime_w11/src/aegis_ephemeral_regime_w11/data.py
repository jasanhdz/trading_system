"""Manifest-verified, causal market panel construction for W11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_DIR = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "w11_frozen.json"

RAW_COLUMNS = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)


def load_frozen_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the preregistered W11 configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_candles(
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    repository_dir: str | Path = REPOSITORY_DIR,
    verify_hashes: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load only frozen symbols and verify each parquet against its manifest."""
    cfg = load_frozen_config(config) if isinstance(config, (str, Path)) else dict(config or load_frozen_config())
    root = Path(repository_dir).resolve()
    manifest_path = (root / cfg["source"]["manifest"]).resolve()
    candle_dir = (root / cfg["source"]["candle_dir"]).resolve()
    if manifest_path.parent != candle_dir:
        raise ValueError("manifest must be inside the configured candle directory")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    selected: dict[str, pd.DataFrame] = {}
    for symbol in cfg["source"]["symbols"]:
        try:
            record = manifest["symbols"][symbol]
        except KeyError as exc:
            raise ValueError(f"manifest has no entry for {symbol}") from exc
        parquet_path = (root / record["parquet"]).resolve()
        expected_path = candle_dir / f"{symbol}_1m.parquet"
        if parquet_path != expected_path or parquet_path.parent != candle_dir:
            raise ValueError(f"unexpected parquet path for {symbol}: {parquet_path}")
        if verify_hashes and _sha256(parquet_path) != record["parquet_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {symbol}")

        frame = pd.read_parquet(parquet_path, columns=list(RAW_COLUMNS))
        opens = frame["open_time_ms"]
        if len(frame) != record["rows"]:
            raise ValueError(f"row-count mismatch for {symbol}")
        if opens.duplicated().any() or record.get("duplicates") != 0:
            raise ValueError(f"duplicate candle timestamps for {symbol}")
        ordered = frame.sort_values("open_time_ms", kind="mergesort").reset_index(drop=True)
        if (
            ordered["open_time_ms"].iloc[0] != record["first_open_ms"]
            or ordered["open_time_ms"].iloc[-1] != record["last_open_ms"]
            or not ordered["open_time_ms"].diff().iloc[1:].eq(60_000).all()
            or record.get("gaps") != 0
        ):
            raise ValueError(f"timestamp integrity mismatch for {symbol}")
        close_delay = ordered["close_time_ms"] - ordered["open_time_ms"]
        if not close_delay.between(0, 60_000, inclusive="left").all():
            raise ValueError(f"invalid close timestamp for {symbol}")
        selected[symbol] = ordered
    return selected


def aggregate_complete_5m_bars(candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate exact half-open 1m sets [t, t+5m) and label them by t+5m."""
    pieces: list[pd.DataFrame] = []
    minute_ns = 60_000_000_000
    for symbol in sorted(candles):
        raw = candles[symbol].copy()
        missing = set(RAW_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"{symbol} is missing columns: {sorted(missing)}")
        if raw["open_time_ms"].duplicated().any():
            raise ValueError(f"duplicate 1m timestamps for {symbol}")
        raw["open_time"] = pd.to_datetime(raw["open_time_ms"], unit="ms", utc=True)
        raw = raw.sort_values("open_time", kind="mergesort")
        raw["bar_start"] = raw["open_time"].dt.floor("5min")
        raw["offset"] = (raw["open_time"].astype("int64") - raw["bar_start"].astype("int64")) // minute_ns

        grouped = raw.groupby("bar_start", sort=True, observed=True)
        complete = grouped["offset"].agg(
            lambda values: len(values) == 5 and np.array_equal(np.sort(values.to_numpy()), np.arange(5))
        )
        valid_starts = complete.index[complete]
        if valid_starts.empty:
            continue
        valid = raw[raw["bar_start"].isin(valid_starts)]
        bars = valid.groupby("bar_start", sort=True, observed=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            trade_count=("trade_count", "sum"),
            taker_buy_volume=("taker_buy_volume", "sum"),
            taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
        )
        bars.index = bars.index + pd.Timedelta(minutes=5)
        bars.index.name = "close_time"
        bars.insert(0, "symbol", symbol)
        pieces.append(bars.reset_index())

    columns = ["close_time", "symbol", *RAW_COLUMNS[1:6], *RAW_COLUMNS[7:]]
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["close_time", "symbol"], kind="mergesort", ignore_index=True
    )


def build_snapshot_panel(
    bars: pd.DataFrame,
    config: Mapping[str, Any] | str | Path | None = None,
) -> pd.DataFrame:
    """Build the long 15m causal feature and forward-outcome panel."""
    cfg = load_frozen_config(config) if isinstance(config, (str, Path)) else dict(config or load_frozen_config())
    feature_names = list(cfg["features"]["names"])
    horizons = list(cfg["experts"]["horizons_minutes"])
    threshold = float(cfg["targets"]["opportunity_min_gross_bps"])
    symbols = list(cfg["source"]["symbols"])
    if bars.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=["decision_at", "symbol"])
        return pd.DataFrame(index=index, columns=[*feature_names, "feature_available_at", "outcome_available_at"])

    required = {"close_time", "symbol", "open", "high", "low", "close", "volume", "taker_buy_volume"}
    if missing := required - set(bars.columns):
        raise ValueError(f"5m bars are missing columns: {sorted(missing)}")
    work = bars.copy()
    work["close_time"] = pd.to_datetime(work["close_time"], utc=True)
    work = work[work["symbol"].isin(symbols)].sort_values(["close_time", "symbol"], kind="mergesort")
    if work.duplicated(["close_time", "symbol"]).any():
        raise ValueError("duplicate 5m bars")

    full_index = pd.date_range(work["close_time"].min(), work["close_time"].max(), freq="5min", tz="UTC")

    def wide(column: str) -> pd.DataFrame:
        return work.pivot(index="close_time", columns="symbol", values=column).reindex(
            index=full_index, columns=symbols
        )

    close, open_, high, low = (wide(name) for name in ("close", "open", "high", "low"))
    volume, taker = wide("volume"), wide("taker_buy_volume")
    ret5 = close.pct_change(fill_method=None)
    returns = {minutes: (close / close.shift(minutes // 5) - 1.0) * 10_000 for minutes in (5, 15, 30, 60)}

    previous_close = close.shift(1)
    true_range = np.maximum(
        high - low,
        np.maximum((high - previous_close).abs(), (low - previous_close).abs()),
    )
    ema = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema = ema.where(close.rolling(12, min_periods=12).count().eq(12))
    path_length = close.diff().abs().rolling(12, min_periods=12).sum()
    rolling_low = low.rolling(12, min_periods=12).min()
    rolling_high = high.rolling(12, min_periods=12).max()
    volume_mean = volume.rolling(12, min_periods=12).mean()
    taker_sum = taker.rolling(3, min_periods=3).sum()
    volume_sum = volume.rolling(3, min_periods=3).sum()

    features: dict[str, pd.DataFrame] = {
        "return_5m_bps": returns[5],
        "return_15m_bps": returns[15],
        "return_30m_bps": returns[30],
        "return_60m_bps": returns[60],
        "realized_vol_60m_bps": ret5.rolling(12, min_periods=12).std(ddof=0) * np.sqrt(12) * 10_000,
        "atr_60m_bps": true_range.rolling(12, min_periods=12).mean() / close * 10_000,
        "ema_distance_60m_bps": (close / ema - 1.0) * 10_000,
        "ema_slope_60m_bps": (ema / ema.shift(12) - 1.0) * 10_000,
        "efficiency_60m": (close - close.shift(12)).abs() / path_length,
        "range_position_60m": (close - rolling_low) / (rolling_high - rolling_low),
        "relative_volume_60m": volume / volume_mean,
        "taker_imbalance_15m": 2.0 * taker_sum / volume_sum - 1.0,
    }

    btc15 = returns[15]["BTCUSDT"]
    btc5 = ret5["BTCUSDT"]
    features["btc_return_15m_bps"] = pd.DataFrame(
        np.broadcast_to(btc15.to_numpy()[:, None], close.shape), index=close.index, columns=close.columns
    )
    features["relative_to_btc_15m_bps"] = returns[15].sub(btc15, axis=0)
    covariance = ret5.rolling(24, min_periods=24).cov(btc5)
    btc_variance = btc5.rolling(24, min_periods=24).var()
    features["btc_beta_120m"] = covariance.div(btc_variance, axis=0)
    features["btc_correlation_120m"] = ret5.rolling(24, min_periods=24).corr(btc5)
    breadth = (returns[15] > 0).where(returns[15].notna()).mean(axis=1, skipna=True)
    dispersion = returns[15].std(axis=1, ddof=0, skipna=True)
    alt_columns = [symbol for symbol in symbols if symbol != "BTCUSDT"]
    alt_return = returns[15][alt_columns].mean(axis=1, skipna=True)
    eth_btc = returns[15]["ETHUSDT"] - btc15
    for name, series in (
        ("market_breadth_15m", breadth),
        ("cross_sectional_dispersion_15m_bps", dispersion),
        ("alt_basket_return_15m_bps", alt_return),
        ("eth_btc_relative_15m_bps", eth_btc),
    ):
        features[name] = pd.DataFrame(
            np.broadcast_to(series.to_numpy()[:, None], close.shape), index=close.index, columns=close.columns
        )

    if set(features) != set(feature_names):
        raise ValueError("implemented feature set differs from frozen configuration")
    snapshot_mask = full_index.minute % int(cfg["source"]["snapshot_minutes"]) == 0
    snapshot_index = full_index[snapshot_mask]
    current_bar = close.loc[snapshot_index].notna()
    records: list[pd.DataFrame] = []
    max_horizon = max(horizons)
    for symbol in symbols:
        present = current_bar.index[current_bar[symbol]]
        if present.empty:
            continue
        frame = pd.DataFrame(index=present)
        for name in feature_names:
            frame[name] = features[name].loc[present, symbol]
        frame["feature_available_at"] = present
        entry = open_[symbol].shift(-1)
        for horizon in horizons:
            exit_close = close[symbol].shift(-(horizon // 5))
            target = (exit_close / entry - 1.0) * 10_000
            target_name = f"gross_target_{horizon}m_bps"
            frame[target_name] = target.loc[present]
            frame[f"opportunity_{horizon}m"] = frame[target_name].abs().ge(threshold).where(frame[target_name].notna())
        frame["outcome_available_at"] = present + pd.Timedelta(minutes=max_horizon)
        frame["symbol"] = symbol
        frame.index.name = "decision_at"
        records.append(frame.reset_index())

    panel = pd.concat(records, ignore_index=True).sort_values(
        ["decision_at", "symbol"], kind="mergesort", ignore_index=True
    )
    panel = panel.set_index(["decision_at", "symbol"])
    decisions = panel.index.get_level_values("decision_at")
    if not panel["feature_available_at"].le(decisions).all():
        raise AssertionError("feature availability exceeds decision time")
    panel.attrs["feature_names"] = feature_names
    return panel


def build_data_panel(
    candles: Mapping[str, pd.DataFrame] | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    repository_dir: str | Path = REPOSITORY_DIR,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Load (when needed), aggregate, and construct the frozen W11 panel."""
    cfg = load_frozen_config(config) if isinstance(config, (str, Path)) else dict(config or load_frozen_config())
    raw = candles if candles is not None else load_selected_candles(
        cfg, repository_dir=repository_dir, verify_hashes=verify_hashes
    )
    return build_snapshot_panel(aggregate_complete_5m_bars(raw), cfg)
