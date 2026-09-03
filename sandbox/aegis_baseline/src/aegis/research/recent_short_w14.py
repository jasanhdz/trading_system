"""Recent-regime SHORT selector and isolated volume-navigation replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "ret_1", "ret_3", "ret_6", "ret_12", "atr_bps", "atr_percentile",
    "ema7_distance_atr", "ema25_distance_atr", "ema99_distance_atr",
    "ema7_slope_atr", "ema25_slope_atr", "rsi6", "rsi12", "rsi24",
    "rsi_short_space", "volume_ratio", "volume_zscore", "taker_imbalance",
    "body_ratio", "clv", "range_atr", "prior_move_atr", "trend_age_short",
    "btc_ret_1", "btc_ret_3", "btc_ret_12",
)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(20, window // 4)).apply(
        lambda values: float(np.mean(values[:-1] <= values[-1])) if len(values) > 1 else np.nan,
        raw=True,
    )


def load_five_minute(path: Path, symbol: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw.index = pd.to_datetime(raw["open_time_ms"], unit="ms", utc=True)
    frame = raw.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"), trade_count=("trade_count", "sum"),
    ).dropna()
    frame["symbol"] = symbol
    close = frame["close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [(frame.high - frame.low), (frame.high - previous).abs(), (frame.low - previous).abs()],
        axis=1,
    ).max(axis=1)
    frame["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean().shift(1)
    frame["atr_bps"] = frame.atr / close * 10_000
    frame["atr_percentile"] = _rolling_percentile(frame.atr_bps, 288)
    for horizon in (1, 3, 6, 12):
        frame[f"ret_{horizon}"] = close.pct_change(horizon) * 10_000
    for period in (7, 25, 99):
        ema = close.ewm(span=period, adjust=False).mean()
        frame[f"ema{period}_distance_atr"] = (close - ema) / frame.atr
        if period in (7, 25):
            frame[f"ema{period}_slope_atr"] = ema.diff(3) / frame.atr
    for period in (6, 12, 24):
        frame[f"rsi{period}"] = _rsi(close, period)
    frame["rsi_short_space"] = frame.rsi6
    median_volume = frame.volume.shift(1).rolling(20, min_periods=10).median()
    log_volume = np.log1p(frame.volume)
    frame["volume_ratio"] = frame.volume / median_volume.replace(0, np.nan)
    frame["volume_zscore"] = (
        (log_volume - log_volume.rolling(20, min_periods=10).mean())
        / log_volume.rolling(20, min_periods=10).std().replace(0, np.nan)
    )
    total = frame.volume
    buy = frame.taker_buy_volume
    frame["taker_imbalance"] = (2 * buy - total) / total.replace(0, np.nan)
    prior_range = (frame.high - frame.low).replace(0, np.nan)
    frame["body_ratio"] = (frame.close - frame.open).abs() / prior_range
    frame["clv"] = (frame.close - frame.low) / prior_range
    frame["range_atr"] = prior_range / frame.atr
    frame["prior_move_atr"] = (close - close.shift(12)) / frame.atr
    below = (close < close.ewm(span=25, adjust=False).mean()).astype(int)
    groups = (below != below.shift()).cumsum()
    frame["trend_age_short"] = below.groupby(groups).cumsum()
    return frame


def build_dataset(root: Path, symbols: list[str]) -> pd.DataFrame:
    base = root / "data/live_entry_quality_audit_20260815/candles_1m"
    frames = {symbol: load_five_minute(base / f"{symbol}_1m.parquet", symbol) for symbol in symbols}
    btc = frames["BTCUSDT"][["ret_1", "ret_3", "ret_12"]].rename(
        columns={name: f"btc_{name}" for name in ("ret_1", "ret_3", "ret_12")}
    )
    rows = []
    for symbol, frame in frames.items():
        frame = frame.join(btc, how="left")
        entry = frame.open.shift(-1)
        for minutes in (15, 30, 60):
            bars = minutes // 5
            frame[f"short_gross_{minutes}m_bps"] = (entry / frame.close.shift(-bars) - 1) * 10_000
        rows.append(frame)
    return pd.concat(rows).sort_index()


def _mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))


def _model(name: str, seed: int):
    if name == "regularized_logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced", random_state=seed),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.05,
                                       l2_regularization=2.0, random_state=seed),
    )


def _economic_metrics(gross: np.ndarray, selected: np.ndarray, cost: float) -> dict[str, Any]:
    net = gross[selected] - cost
    losses = -net[net < 0].sum()
    return {
        "signals": int(len(gross)), "trades": int(selected.sum()),
        "coverage": float(selected.mean()),
        "gross_bps_per_trade": float(np.mean(gross[selected])) if selected.any() else 0.0,
        "net_bps_per_trade": float(np.mean(net)) if selected.any() else 0.0,
        "net_bps_per_signal": float(net.sum() / len(gross)) if len(gross) else 0.0,
        "profit_factor": float(net[net > 0].sum() / losses) if losses > 0 else 0.0,
        "win_rate": float(np.mean(net > 0)) if len(net) else 0.0,
    }


def _block_bootstrap(frame: pd.DataFrame, gross_column: str, selected_column: str,
                     cost: float, repetitions: int, seed: int) -> dict[str, float]:
    work = frame[[gross_column, selected_column]].copy()
    work["block"] = work.index.floor("6h")
    blocks = [group for _, group in work.groupby("block")]
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        sampled = [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        gross = np.concatenate([part[gross_column].to_numpy(float) for part in sampled])
        selected = np.concatenate([part[selected_column].to_numpy(bool) for part in sampled])
        net = gross[selected] - cost
        values[iteration] = float(net.mean()) if len(net) else 0.0
    return {
        "mean_net_bps_per_trade": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def evaluate_short_selector(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    split = config["splits"]
    hourly = frame.loc[frame.index.minute == 0].dropna(subset=list(FEATURES))
    train = hourly[_mask(hourly, *split["train"])]
    calibration = hourly[_mask(hourly, *split["calibration"])]
    validation = hourly[_mask(hourly, *split["validation"])]
    candidates = []
    fitted: dict[tuple[str, int], Any] = {}
    for model_name in config["w14a"]["models"]:
        for horizon in config["w14a"]["horizons_minutes"]:
            target = f"short_gross_{horizon}m_bps"
            usable = train.dropna(subset=[target])
            model = _model(model_name, 20260816)
            model.fit(usable[list(FEATURES)], (usable[target] > 14.0).astype(int))
            fitted[(model_name, horizon)] = model
            calibration_usable = calibration.dropna(subset=[target])
            probabilities = model.predict_proba(calibration_usable[list(FEATURES)])[:, 1]
            for quantile in config["w14a"]["calibration_selection_quantiles"]:
                threshold = float(np.quantile(probabilities, quantile))
                selected = probabilities >= threshold
                metrics = _economic_metrics(calibration_usable[target].to_numpy(), selected, 14.0)
                if metrics["trades"] >= config["w14a"]["minimum_calibration_trades"]:
                    candidates.append((metrics["net_bps_per_signal"], model_name, horizon, quantile, threshold, metrics))
    candidates.sort(reverse=True, key=lambda row: row[0])
    if not candidates:
        raise RuntimeError("W14A_NO_CALIBRATION_CANDIDATE")
    _, model_name, horizon, quantile, threshold, calibration_metrics = candidates[0]
    target = f"short_gross_{horizon}m_bps"
    validation = validation.dropna(subset=[target])
    probabilities = fitted[(model_name, horizon)].predict_proba(validation[list(FEATURES)])[:, 1]
    selected = probabilities >= threshold
    validation_rows = validation[["symbol", target]].copy()
    validation_rows["probability"] = probabilities
    validation_rows["selected"] = selected
    validation_rows["net_bps"] = np.where(selected, validation_rows[target] - 14.0, 0.0)
    per_symbol = {}
    for symbol, group in validation_rows.groupby("symbol"):
        per_symbol[symbol] = _economic_metrics(group[target].to_numpy(), group.selected.to_numpy(), 14.0)
    per_day = {
        str(day): _economic_metrics(group[target].to_numpy(), group.selected.to_numpy(), 14.0)
        for day, group in validation_rows.groupby(validation_rows.index.date)
    }
    primary = _economic_metrics(validation[target].to_numpy(), selected, 14.0)
    stress = _economic_metrics(validation[target].to_numpy(), selected, 20.0)
    baseline = _economic_metrics(validation[target].to_numpy(), np.ones(len(validation), dtype=bool), 14.0)
    bootstrap = _block_bootstrap(
        validation_rows, target, "selected", 14.0,
        config["statistics"]["bootstrap_repetitions"], config["statistics"]["bootstrap_seed"],
    )
    positive_symbols = sum(values["net_bps_per_trade"] > 0 for values in per_symbol.values())
    passed = all((
        primary["trades"] >= config["w14a"]["minimum_validation_trades"],
        primary["net_bps_per_trade"] >= config["w14a"]["minimum_net_expectancy_bps"],
        stress["net_bps_per_trade"] > 0,
        positive_symbols >= config["w14a"]["minimum_positive_symbols"],
        bootstrap["ci95_low"] > 0,
    ))
    return {
        "selected_model": model_name, "horizon_minutes": horizon,
        "calibration_quantile": quantile, "frozen_threshold": threshold,
        "calibration": calibration_metrics, "validation": primary,
        "validation_stress_20bps": stress, "validation_enter_all": baseline,
        "positive_symbols": positive_symbols, "per_symbol": per_symbol,
        "per_day": per_day,
        "block_bootstrap": bootstrap,
        "W14A_RECENT_SHORT_EDGE_FOUND": passed,
    }, validation_rows


def _volume_trade_return(group: pd.DataFrame, index: int, side: int, exit_name: str) -> float:
    entry_index = index + 1
    if entry_index >= len(group):
        return np.nan
    entry = float(group.open.iloc[entry_index])
    if exit_name.startswith("fixed_"):
        bars = int(exit_name.split("_")[1][:-1]) // 5
        exit_index = min(entry_index + bars - 1, len(group) - 1)
        exit_price = float(group.close.iloc[exit_index])
    else:
        peak = 0.0
        adverse_closes = 0
        exit_price = float(group.close.iloc[entry_index])
        for offset in range(0, 12):
            cursor = entry_index + offset
            if cursor >= len(group):
                break
            close = float(group.close.iloc[cursor])
            favorable = side * (close / entry - 1) * 10_000
            peak = max(peak, favorable)
            adverse_closes = adverse_closes + 1 if favorable < 0 else 0
            exit_price = close
            if adverse_closes >= 2 or (peak >= 14.0 and peak - favorable >= 0.40 * peak):
                break
    return side * (exit_price / entry - 1) * 10_000


def evaluate_volume_navigation(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    split = config["splits"]
    candidates = []
    all_events: dict[tuple[int, float, str], pd.DataFrame] = {}
    for length in config["w14b"]["sequence_lengths"]:
        for volume_min in config["w14b"]["volume_ratio_minimums"]:
            for exit_name in config["w14b"]["exits"]:
                rows = []
                for symbol, group in frame.groupby("symbol"):
                    group = group.sort_index().copy()
                    direction = np.sign(group.close - group.open)
                    aligned = direction.ne(0)
                    for lag in range(1, length):
                        aligned &= direction.eq(direction.shift(lag))
                    event = aligned & group.volume_ratio.ge(volume_min)
                    last_accepted = -10_000
                    for idx in np.flatnonzero(event.to_numpy()):
                        if idx - last_accepted < config["w14b"]["minimum_separation_bars"]:
                            continue
                        gross = _volume_trade_return(group, int(idx), int(direction.iloc[idx]), exit_name)
                        if np.isfinite(gross):
                            rows.append({"timestamp": group.index[idx], "symbol": symbol,
                                         "side": "LONG" if direction.iloc[idx] > 0 else "SHORT", "gross_bps": gross})
                            last_accepted = int(idx)
                events = pd.DataFrame(rows)
                if events.empty:
                    continue
                events.index = pd.DatetimeIndex(events.pop("timestamp"))
                all_events[(length, volume_min, exit_name)] = events
                calibration = events[_mask(events, *split["calibration"])]
                metrics = _economic_metrics(calibration.gross_bps.to_numpy(), np.ones(len(calibration), bool), 14.0)
                if metrics["trades"] >= 50:
                    candidates.append((metrics["net_bps_per_trade"], length, volume_min, exit_name, metrics))
    candidates.sort(reverse=True, key=lambda row: row[0])
    if not candidates:
        raise RuntimeError("W14B_NO_CALIBRATION_CANDIDATE")
    _, length, volume_min, exit_name, calibration_metrics = candidates[0]
    validation = all_events[(length, volume_min, exit_name)]
    validation = validation[_mask(validation, *split["validation"])]
    primary = _economic_metrics(validation.gross_bps.to_numpy(), np.ones(len(validation), bool), 14.0)
    stress = _economic_metrics(validation.gross_bps.to_numpy(), np.ones(len(validation), bool), 20.0)
    per_symbol = {
        symbol: _economic_metrics(group.gross_bps.to_numpy(), np.ones(len(group), bool), 14.0)
        for symbol, group in validation.groupby("symbol")
    }
    per_day = {
        str(day): _economic_metrics(group.gross_bps.to_numpy(), np.ones(len(group), bool), 14.0)
        for day, group in validation.groupby(validation.index.date)
    }
    per_side = {
        side: _economic_metrics(group.gross_bps.to_numpy(), np.ones(len(group), bool), 14.0)
        for side, group in validation.groupby("side")
    }
    positive_symbols = sum(values["net_bps_per_trade"] > 0 for values in per_symbol.values())
    bootstrap_frame = validation.copy()
    bootstrap_frame["selected"] = True
    bootstrap = _block_bootstrap(
        bootstrap_frame, "gross_bps", "selected", 14.0,
        config["statistics"]["bootstrap_repetitions"], config["statistics"]["bootstrap_seed"] + 1,
    )
    passed = all((
        primary["trades"] >= config["w14b"]["minimum_validation_trades"],
        primary["net_bps_per_trade"] >= config["w14b"]["minimum_net_expectancy_bps"],
        stress["net_bps_per_trade"] > 0,
        positive_symbols >= config["w14b"]["minimum_positive_symbols"],
        bootstrap["ci95_low"] > 0,
    ))
    return {
        "sequence_length": length, "volume_ratio_minimum": volume_min,
        "exit": exit_name, "calibration": calibration_metrics,
        "validation": primary, "validation_stress_20bps": stress,
        "positive_symbols": positive_symbols, "per_symbol": per_symbol,
        "per_day": per_day, "per_side": per_side,
        "block_bootstrap": bootstrap,
        "W14B_VOLUME_NAVIGATION_EDGE_FOUND": passed,
    }


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())
