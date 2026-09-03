"""Read-only audit of same-color candle continuation on finalized OHLCV."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..config import CANONICAL_SYMBOLS
from ..utils import canonical_json, sha256_file


class CandleMomentumAuditError(ValueError):
    pass


@dataclass(frozen=True)
class CandleMomentumAuditConfig:
    audit_id: str
    database: Path
    table: str
    timeframe: str
    symbols: tuple[str, ...]
    lookback_days: int
    exact_run_lengths: tuple[int, ...]
    volume_baseline_bars: int
    trend_ema_bars: int
    trend_slope_bars: int
    breakout_lookback_bars: int
    strong_body_minimum_fraction: float
    discovery_fraction: float
    round_trip_cost_fraction: float
    minimum_validation_events: int
    minimum_continuation_lift: float
    require_positive_net_ci95: bool
    conditions: tuple[str, ...]
    json_report: Path
    markdown_report: Path


CONDITIONS = (
    "ALL_EXACT_RUN",
    "VOLUME_ALL_ABOVE_MEDIAN",
    "VOLUME_ALL_ABOVE_MEDIAN_RISING",
    "STRONG_BODY_VOLUME",
    "TREND_ALIGNED_VOLUME",
    "BREAKOUT_TREND_VOLUME",
)


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandleMomentumAuditError(f"{identity} must be a mapping")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_candle_momentum_audit_config(
    path: Path,
    *,
    repo_root: Path,
) -> CandleMomentumAuditConfig:
    payload = _mapping(yaml.safe_load(path.read_text()), "audit")
    universe = _mapping(payload["universe"], "universe")
    source = _mapping(payload["source"], "source")
    pattern = _mapping(payload["pattern"], "pattern")
    evaluation = _mapping(payload["evaluation"], "evaluation")
    outputs = _mapping(payload["outputs"], "outputs")
    authority = _mapping(payload["authority"], "authority")
    config = CandleMomentumAuditConfig(
        audit_id=str(payload["audit_id"]),
        database=_resolve(repo_root, source["database"]),
        table=str(source["table"]),
        timeframe=str(universe["timeframe"]),
        symbols=tuple(str(item) for item in universe["symbols"]),
        lookback_days=int(source["lookback_days"]),
        exact_run_lengths=tuple(
            int(item) for item in pattern["exact_run_lengths"]
        ),
        volume_baseline_bars=int(pattern["volume_baseline_bars"]),
        trend_ema_bars=int(pattern["trend_ema_bars"]),
        trend_slope_bars=int(pattern["trend_slope_bars"]),
        breakout_lookback_bars=int(pattern["breakout_lookback_bars"]),
        strong_body_minimum_fraction=float(
            pattern["strong_body_minimum_fraction"]
        ),
        discovery_fraction=float(evaluation["discovery_fraction"]),
        round_trip_cost_fraction=float(
            evaluation["round_trip_cost_fraction"]
        ),
        minimum_validation_events=int(
            evaluation["minimum_validation_events"]
        ),
        minimum_continuation_lift=float(
            evaluation["minimum_continuation_lift"]
        ),
        require_positive_net_ci95=bool(
            evaluation["require_positive_net_ci95"]
        ),
        conditions=tuple(str(item) for item in payload["conditions"]),
        json_report=_resolve(repo_root, outputs["json_report"]),
        markdown_report=_resolve(repo_root, outputs["markdown_report"]),
    )
    if (
        payload.get("schema_version") != "aegis-candle-momentum-audit-v1"
        or payload.get("mode") != "RESEARCH_ONLY"
        or config.symbols != CANONICAL_SYMBOLS
        or config.timeframe != "5m"
        or config.table != "ohlcv_data"
        or config.conditions != CONDITIONS
        or config.exact_run_lengths != (2, 3, 4)
        or config.lookback_days < 365
        or config.volume_baseline_bars < 10
        or config.trend_ema_bars < 10
        or not 0.5 <= config.discovery_fraction < 0.9
        or not 0.0 <= config.round_trip_cost_fraction < 0.01
        or not 0.0 < config.strong_body_minimum_fraction <= 1.0
        or bool(authority.get("live_changes"))
        or bool(authority.get("automatic_promotion"))
        or bool(authority.get("automatic_training"))
        or not bool(authority.get("owner_review_required"))
    ):
        raise CandleMomentumAuditError(
            "AEGIS_CANDLE_MOMENTUM_CONFIG_INVALID"
        )
    return config


def _database_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _read_symbol(
    connection: sqlite3.Connection,
    config: CandleMomentumAuditConfig,
    symbol: str,
) -> pd.DataFrame:
    maximum = connection.execute(
        f"SELECT MAX(timestamp) FROM {config.table} "
        "WHERE symbol = ? AND timeframe = ?",
        (_database_symbol(symbol), config.timeframe),
    ).fetchone()[0]
    if maximum is None:
        raise CandleMomentumAuditError(
            f"AEGIS_CANDLE_MOMENTUM_SYMBOL_MISSING:{symbol}"
        )
    end = pd.Timestamp(maximum, tz="UTC")
    start = end - pd.Timedelta(days=config.lookback_days)
    frame = pd.read_sql_query(
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM {config.table} "
        "WHERE symbol = ? AND timeframe = ? AND timestamp >= ? "
        "ORDER BY timestamp",
        connection,
        params=(
            _database_symbol(symbol),
            config.timeframe,
            start.tz_localize(None).isoformat(sep=" "),
        ),
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _quality(frame: pd.DataFrame) -> Mapping[str, Any]:
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    ordered = frame.drop_duplicates("timestamp", keep="last").sort_values(
        "timestamp"
    )
    intervals = ordered["timestamp"].diff()
    expected = pd.Timedelta(minutes=5)
    gaps = intervals > expected
    invalid = (
        ~np.isfinite(
            ordered[["open", "high", "low", "close", "volume"]]
        ).all(axis=1)
        | (ordered[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (ordered["volume"] < 0)
        | (ordered["high"] < ordered[["open", "close", "low"]].max(axis=1))
        | (ordered["low"] > ordered[["open", "close", "high"]].min(axis=1))
    )
    missing = int(
        sum(
            max(0, round(delta / expected) - 1)
            for delta in intervals[gaps]
        )
    )
    return {
        "rows": len(frame),
        "first_timestamp": ordered["timestamp"].iloc[0].isoformat(),
        "last_timestamp": ordered["timestamp"].iloc[-1].isoformat(),
        "duplicate_timestamps": duplicate_count,
        "gap_count": int(gaps.sum()),
        "estimated_missing_bars": missing,
        "invalid_rows": int(invalid.sum()),
    }


def _prepare_events(
    frame: pd.DataFrame,
    config: CandleMomentumAuditConfig,
    *,
    run_length: int,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    if run_length not in config.exact_run_lengths:
        raise CandleMomentumAuditError(
            "AEGIS_CANDLE_MOMENTUM_RUN_LENGTH_INVALID"
        )
    data = (
        frame.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )
    color = np.sign(data["close"] - data["open"]).astype("int8")
    step = data["timestamp"].diff()
    next_step = data["timestamp"].shift(-1) - data["timestamp"]
    next_two_step = data["timestamp"].shift(-2) - data["timestamp"].shift(-1)
    continuous = next_step.eq(
        pd.Timedelta(minutes=5)
    ) & next_two_step.eq(pd.Timedelta(minutes=5))
    for lag in range(run_length):
        continuous &= step.shift(lag).eq(pd.Timedelta(minutes=5))
    rolling_volume = (
        data["volume"]
        .shift(1)
        .rolling(config.volume_baseline_bars, min_periods=config.volume_baseline_bars)
        .median()
    )
    volume_ratio = data["volume"] / rolling_volume.replace(0.0, np.nan)
    candle_range = (data["high"] - data["low"]).replace(0.0, np.nan)
    body_fraction = (data["close"] - data["open"]).abs() / candle_range
    ema = data["close"].ewm(span=config.trend_ema_bars, adjust=False).mean()
    ema_slope = ema - ema.shift(config.trend_slope_bars)
    prior_high = (
        data["high"]
        .shift(1)
        .rolling(config.breakout_lookback_bars)
        .max()
    )
    prior_low = (
        data["low"]
        .shift(1)
        .rolling(config.breakout_lookback_bars)
        .min()
    )
    exact_run = color.ne(0) & color.ne(color.shift(run_length)) & continuous
    for lag in range(1, run_length):
        exact_run &= color.eq(color.shift(lag))
    events = data.loc[
        exact_run,
        ["timestamp", "open", "high", "low", "close"],
    ].copy()
    indexes = events.index
    direction = color.loc[indexes].astype("int8")
    entry = data.loc[indexes, "close"]
    close_3 = data["close"].shift(-1).loc[indexes]
    close_4 = data["close"].shift(-2).loc[indexes]
    high_3 = data["high"].shift(-1).loc[indexes]
    high_4 = data["high"].shift(-2).loc[indexes]
    low_3 = data["low"].shift(-1).loc[indexes]
    low_4 = data["low"].shift(-2).loc[indexes]
    events["direction"] = np.where(direction > 0, "GREEN", "RED")
    next_color = color.shift(-1).loc[indexes]
    events["third_same"] = next_color.eq(direction).to_numpy()
    events["next_opposite"] = next_color.eq(-direction).to_numpy()
    events["next_doji"] = next_color.eq(0).to_numpy()
    events["fourth_same"] = color.shift(-2).loc[indexes].eq(direction).to_numpy()
    events["run_reaches_four"] = (
        events["third_same"] & events["fourth_same"]
    )
    events["gross_return_h1"] = (
        direction.to_numpy() * (close_3.to_numpy() / entry.to_numpy() - 1.0)
    )
    events["gross_return_h2"] = (
        direction.to_numpy() * (close_4.to_numpy() / entry.to_numpy() - 1.0)
    )
    events["net_return_h1"] = (
        events["gross_return_h1"] - config.round_trip_cost_fraction
    )
    events["net_return_h2"] = (
        events["gross_return_h2"] - config.round_trip_cost_fraction
    )
    high_h2 = np.maximum(high_3.to_numpy(), high_4.to_numpy())
    low_h2 = np.minimum(low_3.to_numpy(), low_4.to_numpy())
    events["mae_h2"] = np.where(
        direction.to_numpy() > 0,
        np.maximum(0.0, (entry.to_numpy() - low_h2) / entry.to_numpy()),
        np.maximum(0.0, (high_h2 - entry.to_numpy()) / entry.to_numpy()),
    )
    events["mfe_h2"] = np.where(
        direction.to_numpy() > 0,
        np.maximum(0.0, (high_h2 - entry.to_numpy()) / entry.to_numpy()),
        np.maximum(0.0, (entry.to_numpy() - low_h2) / entry.to_numpy()),
    )
    all_volume = np.ones(len(events), dtype=bool)
    rising_volume = np.ones(len(events), dtype=bool)
    strong_body = np.ones(len(events), dtype=bool)
    for lag in range(run_length):
        all_volume &= (
            volume_ratio.shift(lag).loc[indexes].ge(1.0).to_numpy()
        )
        strong_body &= (
            body_fraction.shift(lag)
            .loc[indexes]
            .ge(config.strong_body_minimum_fraction)
            .to_numpy()
        )
        if lag:
            rising_volume &= (
                data["volume"]
                .shift(lag - 1)
                .loc[indexes]
                .to_numpy()
                > data["volume"].shift(lag).loc[indexes].to_numpy()
            )
    trend_aligned = np.where(
        direction.to_numpy() > 0,
        (entry.to_numpy() > ema.loc[indexes].to_numpy())
        & (ema_slope.loc[indexes].to_numpy() > 0.0),
        (entry.to_numpy() < ema.loc[indexes].to_numpy())
        & (ema_slope.loc[indexes].to_numpy() < 0.0),
    )
    breakout = np.where(
        direction.to_numpy() > 0,
        entry.to_numpy() > prior_high.loc[indexes].to_numpy(),
        entry.to_numpy() < prior_low.loc[indexes].to_numpy(),
    )
    events["ALL_EXACT_RUN"] = True
    events["VOLUME_ALL_ABOVE_MEDIAN"] = all_volume
    events["VOLUME_ALL_ABOVE_MEDIAN_RISING"] = all_volume & rising_volume
    events["STRONG_BODY_VOLUME"] = all_volume & rising_volume & strong_body
    events["TREND_ALIGNED_VOLUME"] = all_volume & rising_volume & trend_aligned
    events["BREAKOUT_TREND_VOLUME"] = (
        all_volume & rising_volume & trend_aligned & breakout
    )
    base_rates = {}
    for label, value in (("GREEN", 1), ("RED", -1)):
        eligible = (
            color.eq(value)
            & color.shift(-1).ne(0)
            & next_step.eq(pd.Timedelta(minutes=5))
        )
        base_rates[label] = float(
            color.shift(-1).loc[eligible].eq(value).mean()
        )
    return events.reset_index(drop=True), base_rates


def _wilson(successes: int, count: int) -> tuple[float | None, float | None]:
    if count == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - margin, center + margin


def _daily_mean_ci(
    rows: pd.DataFrame,
    field: str,
) -> tuple[float | None, float | None]:
    if rows.empty:
        return None, None
    daily = rows.assign(day=rows["timestamp"].dt.date).groupby("day")[field].mean()
    if len(daily) < 2:
        return None, None
    mean = float(daily.mean())
    margin = 1.959963984540054 * float(daily.std(ddof=1)) / math.sqrt(len(daily))
    return mean - margin, mean + margin


def _metrics(
    rows: pd.DataFrame,
    *,
    base_rate: float,
    config: CandleMomentumAuditConfig,
) -> Mapping[str, Any]:
    count = len(rows)
    third_count = int(rows["third_same"].sum()) if count else 0
    opposite_count = int(rows["next_opposite"].sum()) if count else 0
    doji_count = int(rows["next_doji"].sum()) if count else 0
    run4_count = int(rows["run_reaches_four"].sum()) if count else 0
    third_rate = third_count / count if count else None
    third_ci = _wilson(third_count, count)
    opposite_ci = _wilson(opposite_count, count)
    third_rows = rows.loc[rows["third_same"]] if count else rows
    fourth_given_third = (
        float(third_rows["fourth_same"].mean()) if len(third_rows) else None
    )
    net_h1_ci = _daily_mean_ci(rows, "net_return_h1")
    net_h2_ci = _daily_mean_ci(rows, "net_return_h2")
    gains = float(rows.loc[rows["net_return_h2"] > 0, "net_return_h2"].sum())
    losses = -float(rows.loc[rows["net_return_h2"] < 0, "net_return_h2"].sum())
    profit_factor = gains / losses if losses else ("INFINITE" if gains else None)
    opportunity = bool(
        count >= config.minimum_validation_events
        and third_rate is not None
        and third_rate - base_rate >= config.minimum_continuation_lift
        and (
            not config.require_positive_net_ci95
            or (
                net_h1_ci[0] is not None
                and net_h2_ci[0] is not None
                and max(net_h1_ci[0], net_h2_ci[0]) > 0.0
            )
        )
    )
    return {
        "events": count,
        "baseline_next_same_color_rate": base_rate,
        "next_same_color_rate": third_rate,
        "next_opposite_color_rate": (
            opposite_count / count if count else None
        ),
        "next_opposite_color_ci95": {
            "low": opposite_ci[0],
            "high": opposite_ci[1],
        },
        "opposite_minus_same_rate": (
            (opposite_count - third_count) / count if count else None
        ),
        "next_doji_rate": doji_count / count if count else None,
        "next_same_color_ci95": {"low": third_ci[0], "high": third_ci[1]},
        "continuation_lift": (
            third_rate - base_rate if third_rate is not None else None
        ),
        "run_extends_two_more_rate": run4_count / count if count else None,
        "second_next_same_given_next_rate": fourth_given_third,
        "mean_gross_return_h1": (
            float(rows["gross_return_h1"].mean()) if count else None
        ),
        "mean_net_return_h1": (
            float(rows["net_return_h1"].mean()) if count else None
        ),
        "net_return_h1_daily_cluster_ci95": {
            "low": net_h1_ci[0],
            "high": net_h1_ci[1],
        },
        "mean_net_return_h2": (
            float(rows["net_return_h2"].mean()) if count else None
        ),
        "net_return_h2_daily_cluster_ci95": {
            "low": net_h2_ci[0],
            "high": net_h2_ci[1],
        },
        "win_rate_h2_after_cost": (
            float((rows["net_return_h2"] > 0.0).mean()) if count else None
        ),
        "profit_factor_h2_after_cost": profit_factor,
        "mean_mae_h2": float(rows["mae_h2"].mean()) if count else None,
        "p90_mae_h2": (
            float(rows["mae_h2"].quantile(0.90)) if count else None
        ),
        "mean_mfe_h2": float(rows["mfe_h2"].mean()) if count else None,
        "opportunity_gate_passed": opportunity,
    }


def _period_metrics(
    events: pd.DataFrame,
    base_rates: Mapping[str, float],
    config: CandleMomentumAuditConfig,
) -> Mapping[str, Any]:
    result = {}
    for condition in config.conditions:
        selected = events.loc[events[condition]]
        result[condition] = {
            direction: _metrics(
                selected.loc[selected["direction"] == direction],
                base_rate=base_rates[direction],
                config=config,
            )
            for direction in ("GREEN", "RED")
        }
    return result


def _run_length_audit(
    frames: Mapping[str, pd.DataFrame],
    config: CandleMomentumAuditConfig,
    run_length: int,
) -> Mapping[str, Any]:
    per_symbol = {}
    discovery_events = []
    validation_events = []
    for symbol in config.symbols:
        events, base_rates = _prepare_events(
            frames[symbol],
            config,
            run_length=run_length,
        )
        split_index = int(len(events) * config.discovery_fraction)
        discovery = events.iloc[:split_index]
        validation = events.iloc[split_index:]
        per_symbol[symbol] = {
            "base_rates": base_rates,
            "event_count": len(events),
            "discovery": _period_metrics(discovery, base_rates, config),
            "validation": _period_metrics(validation, base_rates, config),
        }
        tagged = events.copy()
        tagged["symbol"] = symbol
        discovery_events.append(tagged.iloc[:split_index])
        validation_events.append(tagged.iloc[split_index:])
    pooled_discovery = pd.concat(discovery_events, ignore_index=True)
    pooled_validation = pd.concat(validation_events, ignore_index=True)
    pooled_base = {}
    for direction in ("GREEN", "RED"):
        rates = [
            per_symbol[symbol]["base_rates"][direction]
            for symbol in config.symbols
        ]
        weights = [
            sum(
                per_symbol[symbol][period]["ALL_EXACT_RUN"][direction][
                    "events"
                ]
                for period in ("discovery", "validation")
            )
            for symbol in config.symbols
        ]
        finite = [
            (rate, weight)
            for rate, weight in zip(rates, weights)
            if math.isfinite(rate)
        ]
        if not finite:
            raise CandleMomentumAuditError(
                "AEGIS_CANDLE_MOMENTUM_BASE_RATE_UNAVAILABLE"
            )
        total_weight = sum(weight for _, weight in finite)
        pooled_base[direction] = float(
            np.average(
                [rate for rate, _ in finite],
                weights=(
                    [weight for _, weight in finite]
                    if total_weight
                    else None
                ),
            )
        )
    validation_passes = []
    for symbol in config.symbols:
        for condition in config.conditions:
            for direction in ("GREEN", "RED"):
                if per_symbol[symbol]["validation"][condition][direction][
                    "opportunity_gate_passed"
                ]:
                    validation_passes.append(
                        {
                            "run_length": run_length,
                            "symbol": symbol,
                            "condition": condition,
                            "direction": direction,
                        }
                    )
    return {
        "run_length": run_length,
        "per_symbol": per_symbol,
        "pooled": {
            "base_rates": pooled_base,
            "discovery": _period_metrics(
                pooled_discovery,
                pooled_base,
                config,
            ),
            "validation": _period_metrics(
                pooled_validation,
                pooled_base,
                config,
            ),
        },
        "validated_opportunities": validation_passes,
    }


def run_candle_momentum_audit(
    config: CandleMomentumAuditConfig,
    *,
    generated_at: datetime | None = None,
) -> Mapping[str, Any]:
    if not config.database.exists():
        raise CandleMomentumAuditError(
            "AEGIS_CANDLE_MOMENTUM_DATABASE_MISSING"
        )
    connection = sqlite3.connect(
        f"file:{config.database}?mode=ro",
        uri=True,
    )
    try:
        frames = {
            symbol: _read_symbol(connection, config, symbol)
            for symbol in config.symbols
        }
    finally:
        connection.close()
    quality = {
        symbol: _quality(frame) for symbol, frame in frames.items()
    }
    run_lengths = {
        str(run_length): _run_length_audit(
            frames,
            config,
            run_length,
        )
        for run_length in config.exact_run_lengths
    }
    validation_passes = [
        item
        for result in run_lengths.values()
        for item in result["validated_opportunities"]
    ]
    exhaustion_checks = []
    for run_length in (3, 4):
        result = run_lengths[str(run_length)]["pooled"]
        for direction in ("GREEN", "RED"):
            discovery = result["discovery"]["VOLUME_ALL_ABOVE_MEDIAN"][
                direction
            ]
            validation = result["validation"]["VOLUME_ALL_ABOVE_MEDIAN"][
                direction
            ]
            discovery_difference = discovery[
                "opposite_minus_same_rate"
            ]
            validation_difference = validation[
                "opposite_minus_same_rate"
            ]
            exhaustion_checks.append(
                {
                    "run_length": run_length,
                    "direction": direction,
                    "discovery_opposite_minus_same_rate": (
                        discovery_difference
                    ),
                    "validation_opposite_minus_same_rate": (
                        validation_difference
                    ),
                    "repeats_across_periods": bool(
                        discovery_difference is not None
                        and validation_difference is not None
                        and discovery_difference > 0.0
                        and validation_difference > 0.0
                    ),
                    "validation_net_momentum_h2": validation[
                        "mean_net_return_h2"
                    ],
                }
            )
    repeated_exhaustion = all(
        item["repeats_across_periods"] for item in exhaustion_checks
    )
    current = generated_at or datetime.now(timezone.utc)
    return {
        "schema_id": "aegis-candle-momentum-audit-report-v1",
        "audit_id": config.audit_id,
        "generated_at": current.astimezone(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "source": {
            "database": str(config.database),
            "database_sha256": sha256_file(config.database),
            "timeframe": config.timeframe,
            "lookback_days": config.lookback_days,
            "closed_final_candles_only": True,
        },
        "method": {
            "pattern": "exact same-color run before evaluation",
            "exact_run_lengths": list(config.exact_run_lengths),
            "doji_handling": "excluded",
            "volume_definition": (
                "each setup candle at or above its trailing 20-bar median"
            ),
            "cost_fraction": config.round_trip_cost_fraction,
            "discovery_fraction": config.discovery_fraction,
            "validation_fraction": 1.0 - config.discovery_fraction,
            "opportunity_gate": {
                "minimum_validation_events": config.minimum_validation_events,
                "minimum_continuation_lift": (
                    config.minimum_continuation_lift
                ),
                "positive_daily_cluster_net_ci95_required": (
                    config.require_positive_net_ci95
                ),
            },
        },
        "quality": quality,
        "run_lengths": run_lengths,
        "descriptive_patterns": {
            "long_run_exhaustion": {
                "classification": (
                    "REPEATED_COLOR_REVERSAL_NOT_TRADABLE_AFTER_COSTS"
                    if repeated_exhaustion
                    else "NOT_CONSISTENTLY_REPEATED"
                ),
                "checks": exhaustion_checks,
                "live_authority": False,
            }
        },
        "validated_opportunities": validation_passes,
        "conclusion": (
            "VALIDATED_CONDITIONAL_MOMENTUM_OPPORTUNITIES_FOUND"
            if validation_passes
            else "NO_VALIDATED_TRADABLE_MOMENTUM_OPPORTUNITY"
        ),
        "live_changes_authorized": False,
        "automatic_promotion": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def render_candle_momentum_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Aegis 5m Same-Color Candle Momentum Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Source SHA-256: `{report['source']['database_sha256']}`",
        "- Trading or Live changes: `NONE`",
        "",
        "## Data quality",
        "",
        "| Symbol | Rows | First | Last | Duplicates removed | Gaps | Invalid |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for symbol, quality in report["quality"].items():
        lines.append(
            "| {symbol} | {rows} | {first} | {last} | {duplicates} | "
            "{gaps} | {invalid} |".format(
                symbol=symbol,
                rows=quality["rows"],
                first=quality["first_timestamp"],
                last=quality["last_timestamp"],
                duplicates=quality["duplicate_timestamps"],
                gaps=quality["gap_count"],
                invalid=quality["invalid_rows"],
            )
        )
    for run_length, run_payload in report["run_lengths"].items():
        lines.extend(
            [
                "",
                f"## Exact run of {run_length} candles: pooled validation",
                "",
                "| Condition | Side | Events | Next same | Opposite | Lift | "
                "Extends 2 more | Net H2 | MAE H2 | Opportunity |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for condition, directions in run_payload["pooled"][
            "validation"
        ].items():
            for direction in ("GREEN", "RED"):
                metric = directions[direction]
                lines.append(
                    "| {condition} | {direction} | {events} | {next_rate} | "
                    "{opposite} | {lift} | {extend} | {net} | {mae} | "
                    "{gate} |".format(
                        condition=condition,
                        direction=direction,
                        events=metric["events"],
                        next_rate=_percent(
                            metric["next_same_color_rate"]
                        ),
                        opposite=_percent(
                            metric["next_opposite_color_rate"]
                        ),
                        lift=_percent(metric["continuation_lift"]),
                        extend=_percent(
                            metric["run_extends_two_more_rate"]
                        ),
                        net=_percent(metric["mean_net_return_h2"]),
                        mae=_percent(metric["mean_mae_h2"]),
                        gate=(
                            "PASS"
                            if metric["opportunity_gate_passed"]
                            else "NO"
                        ),
                    )
                )
        lines.extend(
            [
                "",
                f"## Exact run of {run_length}: per-symbol validation",
                "",
                "Volume requires every candle in the run to be at or above "
                "its trailing 20-bar median.",
                "",
                "| Symbol | Side | Exact events | Exact next | Exact +2 | "
                "Volume events | Volume next | Volume +2 | Net H2 volume |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for symbol, payload in run_payload["per_symbol"].items():
            for direction in ("GREEN", "RED"):
                exact = payload["validation"]["ALL_EXACT_RUN"][direction]
                volume = payload["validation"][
                    "VOLUME_ALL_ABOVE_MEDIAN"
                ][direction]
                lines.append(
                    "| {symbol} | {direction} | {exact_events} | "
                    "{exact_next} | {exact_extend} | {volume_events} | "
                    "{volume_next} | {volume_extend} | {net} |".format(
                        symbol=symbol,
                        direction=direction,
                        exact_events=exact["events"],
                        exact_next=_percent(
                            exact["next_same_color_rate"]
                        ),
                        exact_extend=_percent(
                            exact["run_extends_two_more_rate"]
                        ),
                        volume_events=volume["events"],
                        volume_next=_percent(
                            volume["next_same_color_rate"]
                        ),
                        volume_extend=_percent(
                            volume["run_extends_two_more_rate"]
                        ),
                        net=_percent(volume["mean_net_return_h2"]),
                    )
                )
    exhaustion = report["descriptive_patterns"]["long_run_exhaustion"]
    lines.extend(
        [
            "",
            "## Repeated descriptive pattern",
            "",
            f"- Long-run exhaustion: `{exhaustion['classification']}`",
            "- This is a repeated color-reversal tendency, not a validated "
            "contrarian trading strategy.",
            "",
            "## Interpretation",
            "",
            "A higher same-color probability is not sufficient for trading. "
            "The opportunity gate also requires enough validation events and "
            "a positive daily-clustered net-return confidence bound after the "
            "frozen 0.10% round-trip cost.",
            "",
            "Historical discovery and validation are reported separately. "
            "No condition is promoted automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.3f}%"


def write_candle_momentum_reports(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> Mapping[str, str]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(json_path.parent, 0o700)
    payloads = {
        json_path: canonical_json(report) + "\n",
        markdown_path: render_candle_momentum_markdown(report),
    }
    digests = {}
    for path, payload in payloads.items():
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        digests[str(path)] = sha256_file(path)
    return digests
