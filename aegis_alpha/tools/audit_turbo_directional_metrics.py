from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)

DEFAULT_LOG_DIR = REPO_ROOT / "binance-futures-bot-ts" / "logs" / "aegis"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "binance_candles.db"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: datetime
    action: str
    score: float
    votes_long: int = 0
    votes_short: int = 0
    votes_neutral: int = 0
    reason: str | None = None
    source_path: str | None = None


@dataclass
class SignalOutcome:
    symbol: str
    timestamp: str
    action: str
    score: float
    entryPrice: float
    horizonMinutes: int
    forwardReturn: float | None
    netForwardReturn: float | None
    mfe: float | None
    mae: float | None
    netMfe: float | None
    maeAdverse: float | None
    hit5BeforeMinus5: bool | None
    hit8BeforeMinus5: bool | None
    hit10BeforeMinus8: bool | None
    timeToHit8Minutes: float | None
    timeToMinus5Minutes: float | None
    realizedClass: str | None = None


@dataclass
class SideMetrics:
    count: int = 0
    winRate60m: float | None = None
    avgForwardReturn60m: float | None = None
    medianForwardReturn60m: float | None = None
    avgMfe60m: float | None = None
    avgMae60m: float | None = None
    mfeMaeRatio: float | None = None
    hit5BeforeMinus5: float | None = None
    hit8BeforeMinus5: float | None = None
    hit10BeforeMinus8: float | None = None
    falseSignalRate: float | None = None
    avgTimeToHit8: float | None = None
    avgTimeToMinus5: float | None = None
    p90Mae: float | None = None
    netExpectancy60m: float | None = None


@dataclass
class NeutralMetrics:
    count: int = 0
    largeMoveRate: float | None = None
    missedLongMoveRate: float | None = None
    missedShortMoveRate: float | None = None
    avgAbsReturn60m: float | None = None
    neutralQualityScore: float | None = None


@dataclass
class ClassMetrics:
    precision_LONG: float | None = None
    recall_LONG: float | None = None
    f1_LONG: float | None = None
    precision_SHORT: float | None = None
    recall_SHORT: float | None = None
    f1_SHORT: float | None = None
    precision_NEUTRAL: float | None = None
    recall_NEUTRAL: float | None = None
    f1_NEUTRAL: float | None = None


@dataclass
class SymbolDirectionalSummary:
    symbol: str
    directionalStatus: str
    directionalConfidence: float
    sampleCount: int
    long: SideMetrics
    short: SideMetrics
    neutral: NeutralMetrics
    classification: ClassMetrics
    confusionMatrix: dict[str, dict[str, int]]
    scoreCalibration: str
    directionalWarnings: list[str] = field(default_factory=list)
    recommendedAction: list[str] = field(default_factory=list)


def parse_dt(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        text = text + "T00:00:00+00:00"
    text = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[int(idx)]
    return ordered[lower] * (upper - idx) + ordered[upper] * (idx - lower)


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(values: list[bool]) -> float | None:
    return sum(1 for item in values if item) / len(values) if values else None


def assign_score_bucket(score: float, bucket_edges: list[float]) -> str:
    edges = sorted(bucket_edges)
    if not edges:
        return "all"
    for index, edge in enumerate(edges):
        if score < edge:
            lower = 0.0 if index == 0 else edges[index - 1]
            return f"{lower:.2f}-{edge:.2f}"
    return f"{edges[-1]:.2f}-1.00"


def load_signals(
    *,
    log_dir: Path,
    symbols: set[str],
    start: datetime,
    end: datetime,
    include_neutral: bool,
    sample_every: int,
) -> list[Signal]:
    paths = sorted(log_dir.glob("turbo_signals_*.jsonl"))
    signals: list[Signal] = []
    seen_by_symbol: dict[str, int] = {}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    symbol = str(raw.get("symbol", "")).upper()
                    if symbol not in symbols:
                        continue
                    timestamp_raw = raw.get("timestamp")
                    if not timestamp_raw:
                        continue
                    try:
                        timestamp = parse_dt(str(timestamp_raw))
                    except ValueError:
                        continue
                    if timestamp < start or timestamp > end:
                        continue
                    action = str(raw.get("final_action") or raw.get("raw_action") or "HOLD").upper()
                    if action not in {"LONG", "SHORT", "HOLD"}:
                        action = "HOLD"
                    if action == "HOLD" and not include_neutral:
                        continue
                    seen_by_symbol[symbol] = seen_by_symbol.get(symbol, 0) + 1
                    if sample_every > 1 and (seen_by_symbol[symbol] - 1) % sample_every != 0:
                        continue
                    votes = raw.get("votes") if isinstance(raw.get("votes"), dict) else {}
                    signals.append(
                        Signal(
                            symbol=symbol,
                            timestamp=timestamp,
                            action=action,
                            score=safe_float(raw.get("turbo_score")),
                            votes_long=int(votes.get("long") or 0),
                            votes_short=int(votes.get("short") or 0),
                            votes_neutral=int(votes.get("neutral") or 0),
                            reason=str(raw.get("reason")) if raw.get("reason") is not None else None,
                            source_path=str(path),
                        )
                    )
        except OSError:
            continue
    return signals


def load_candles(db_path: Path, symbol: str, start: datetime, end: datetime) -> list[Candle]:
    symbol = symbol.upper().replace("/", "")
    symbol_variants = [symbol]
    if symbol.endswith("USDT"):
        symbol_variants.append(f"{symbol[:-4]}/USDT")
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol IN (?, ?) AND timeframe = '5m' AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (
                symbol_variants[0],
                symbol_variants[1] if len(symbol_variants) > 1 else symbol_variants[0],
                start.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                end.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ).fetchall()
    finally:
        con.close()
    candles: list[Candle] = []
    for row in rows:
        candles.append(
            Candle(
                timestamp=parse_dt(str(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5] or 0),
            )
        )
    return candles


def find_entry_candle(candles: list[Candle], timestamp: datetime) -> Candle | None:
    previous: Candle | None = None
    for candle in candles:
        if candle.timestamp <= timestamp:
            previous = candle
            continue
        break
    return previous


def future_candles(candles: list[Candle], entry_time: datetime, horizon_minutes: int) -> list[Candle]:
    end_time = entry_time + timedelta(minutes=horizon_minutes)
    return [candle for candle in candles if entry_time < candle.timestamp <= end_time]


def hit_before(
    *,
    side: str,
    entry_price: float,
    candles: list[Candle],
    target_return: float,
    stop_return: float,
) -> tuple[bool | None, float | None, float | None]:
    target_time: float | None = None
    stop_time: float | None = None
    for candle in candles:
        elapsed = (candle.timestamp - candles[0].timestamp).total_seconds() / 60 if candles else None
        if side == "LONG":
            target_hit = candle.high >= entry_price * (1 + target_return)
            stop_hit = candle.low <= entry_price * (1 - stop_return)
        else:
            target_hit = candle.low <= entry_price * (1 - target_return)
            stop_hit = candle.high >= entry_price * (1 + stop_return)
        if target_hit and stop_hit:
            return False, elapsed, elapsed
        if target_hit and target_time is None:
            target_time = elapsed
        if stop_hit and stop_time is None:
            stop_time = elapsed
        if target_time is not None or stop_time is not None:
            break
    if target_time is None and stop_time is None:
        return None, None, None
    if target_time is not None and stop_time is None:
        return True, target_time, None
    if target_time is None and stop_time is not None:
        return False, None, stop_time
    return bool(target_time < stop_time), target_time, stop_time


def realized_class_from_forward(
    *,
    long_hit8: bool | None,
    short_hit8: bool | None,
    close_return: float | None,
    threshold_return: float,
) -> str:
    if long_hit8 is True or (close_return is not None and close_return > threshold_return):
        return "LONG"
    if short_hit8 is True or (close_return is not None and close_return < -threshold_return):
        return "SHORT"
    return "NEUTRAL"


def compute_signal_outcome(
    signal: Signal,
    candles: list[Candle],
    horizon_minutes: int,
    *,
    fee_bps: float,
    slippage_bps: float,
    leverage_proxy: float,
) -> SignalOutcome | None:
    entry = find_entry_candle(candles, signal.timestamp)
    if entry is None:
        return None
    future = future_candles(candles, entry.timestamp, horizon_minutes)
    if not future:
        return None
    entry_price = entry.close
    future_close = future[-1].close
    max_high = max(candle.high for candle in future)
    min_low = min(candle.low for candle in future)
    cost_return = (fee_bps + slippage_bps) / 10000.0
    target5 = 0.05 / leverage_proxy
    target8 = 0.08 / leverage_proxy
    target10 = 0.10 / leverage_proxy
    stop5 = 0.05 / leverage_proxy
    stop8 = 0.08 / leverage_proxy

    if signal.action == "LONG":
        forward_return = (future_close - entry_price) / entry_price
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low - entry_price) / entry_price
        hit5, _, minus5_time = hit_before(side="LONG", entry_price=entry_price, candles=future, target_return=target5, stop_return=stop5)
        hit8, hit8_time, _ = hit_before(side="LONG", entry_price=entry_price, candles=future, target_return=target8, stop_return=stop5)
        hit10, _, _ = hit_before(side="LONG", entry_price=entry_price, candles=future, target_return=target10, stop_return=stop8)
    elif signal.action == "SHORT":
        forward_return = (entry_price - future_close) / entry_price
        mfe = (entry_price - min_low) / entry_price
        mae = (entry_price - max_high) / entry_price
        hit5, _, minus5_time = hit_before(side="SHORT", entry_price=entry_price, candles=future, target_return=target5, stop_return=stop5)
        hit8, hit8_time, _ = hit_before(side="SHORT", entry_price=entry_price, candles=future, target_return=target8, stop_return=stop5)
        hit10, _, _ = hit_before(side="SHORT", entry_price=entry_price, candles=future, target_return=target10, stop_return=stop8)
    else:
        raw_close_return = (future_close - entry_price) / entry_price
        long_hit8, _, _ = hit_before(side="LONG", entry_price=entry_price, candles=future, target_return=target8, stop_return=stop5)
        short_hit8, _, _ = hit_before(side="SHORT", entry_price=entry_price, candles=future, target_return=target8, stop_return=stop5)
        return SignalOutcome(
            symbol=signal.symbol,
            timestamp=signal.timestamp.isoformat(),
            action=signal.action,
            score=signal.score,
            entryPrice=entry_price,
            horizonMinutes=horizon_minutes,
            forwardReturn=raw_close_return,
            netForwardReturn=abs(raw_close_return) - cost_return,
            mfe=(max_high - entry_price) / entry_price,
            mae=(min_low - entry_price) / entry_price,
            netMfe=max(abs((max_high - entry_price) / entry_price), abs((entry_price - min_low) / entry_price)) - cost_return,
            maeAdverse=None,
            hit5BeforeMinus5=None,
            hit8BeforeMinus5=None,
            hit10BeforeMinus8=None,
            timeToHit8Minutes=None,
            timeToMinus5Minutes=None,
            realizedClass=realized_class_from_forward(
                long_hit8=long_hit8,
                short_hit8=short_hit8,
                close_return=raw_close_return,
                threshold_return=target8,
            ),
        )

    net_forward = forward_return - cost_return
    return SignalOutcome(
        symbol=signal.symbol,
        timestamp=signal.timestamp.isoformat(),
        action=signal.action,
        score=signal.score,
        entryPrice=entry_price,
        horizonMinutes=horizon_minutes,
        forwardReturn=forward_return,
        netForwardReturn=net_forward,
        mfe=mfe,
        mae=mae,
        netMfe=mfe - cost_return,
        maeAdverse=max(0.0, -mae),
        hit5BeforeMinus5=hit5,
        hit8BeforeMinus5=hit8,
        hit10BeforeMinus8=hit10,
        timeToHit8Minutes=hit8_time,
        timeToMinus5Minutes=minus5_time,
        realizedClass=realized_class_from_forward(
            long_hit8=hit8 if signal.action == "LONG" else None,
            short_hit8=hit8 if signal.action == "SHORT" else None,
            close_return=forward_return if signal.action == "LONG" else -forward_return,
            threshold_return=target8,
        ),
    )


def side_metrics(outcomes: list[SignalOutcome]) -> SideMetrics:
    if not outcomes:
        return SideMetrics()
    forward = [item.forwardReturn for item in outcomes if item.forwardReturn is not None]
    net = [item.netForwardReturn for item in outcomes if item.netForwardReturn is not None]
    mfe = [item.mfe for item in outcomes if item.mfe is not None]
    mae = [item.mae for item in outcomes if item.mae is not None]
    adverse = [item.maeAdverse for item in outcomes if item.maeAdverse is not None]
    hit5 = [item.hit5BeforeMinus5 for item in outcomes if item.hit5BeforeMinus5 is not None]
    hit8 = [item.hit8BeforeMinus5 for item in outcomes if item.hit8BeforeMinus5 is not None]
    hit10 = [item.hit10BeforeMinus8 for item in outcomes if item.hit10BeforeMinus8 is not None]
    t_hit8 = [item.timeToHit8Minutes for item in outcomes if item.timeToHit8Minutes is not None]
    t_minus5 = [item.timeToMinus5Minutes for item in outcomes if item.timeToMinus5Minutes is not None]
    avg_mfe = avg(mfe)
    avg_adverse = avg(adverse)
    return SideMetrics(
        count=len(outcomes),
        winRate60m=rate([value > 0 for value in net]),
        avgForwardReturn60m=avg(forward),
        medianForwardReturn60m=median(forward) if forward else None,
        avgMfe60m=avg_mfe,
        avgMae60m=avg(mae),
        mfeMaeRatio=(avg_mfe / avg_adverse) if avg_mfe is not None and avg_adverse not in (None, 0) else None,
        hit5BeforeMinus5=rate(hit5),
        hit8BeforeMinus5=rate(hit8),
        hit10BeforeMinus8=rate(hit10),
        falseSignalRate=rate([value < 0 for value in net]),
        avgTimeToHit8=avg(t_hit8),
        avgTimeToMinus5=avg(t_minus5),
        p90Mae=percentile(adverse, 0.90),
        netExpectancy60m=avg(net),
    )


def neutral_metrics(outcomes: list[SignalOutcome], threshold_return: float) -> NeutralMetrics:
    if not outcomes:
        return NeutralMetrics()
    abs_returns = [abs(item.forwardReturn or 0.0) for item in outcomes]
    large = [value >= threshold_return for value in abs_returns]
    missed_long = [item.realizedClass == "LONG" for item in outcomes]
    missed_short = [item.realizedClass == "SHORT" for item in outcomes]
    large_rate = rate(large)
    return NeutralMetrics(
        count=len(outcomes),
        largeMoveRate=large_rate,
        missedLongMoveRate=rate(missed_long),
        missedShortMoveRate=rate(missed_short),
        avgAbsReturn60m=avg(abs_returns),
        neutralQualityScore=(1.0 - large_rate) if large_rate is not None else None,
    )


def build_confusion(outcomes: list[SignalOutcome]) -> dict[str, dict[str, int]]:
    labels = ("LONG", "SHORT", "NEUTRAL")
    matrix = {predicted: {actual: 0 for actual in labels} for predicted in labels}
    for outcome in outcomes:
        predicted = "NEUTRAL" if outcome.action == "HOLD" else outcome.action
        actual = outcome.realizedClass or "NEUTRAL"
        if predicted in matrix and actual in matrix[predicted]:
            matrix[predicted][actual] += 1
    return matrix


def class_metrics(matrix: dict[str, dict[str, int]]) -> ClassMetrics:
    labels = ("LONG", "SHORT", "NEUTRAL")
    result: dict[str, float | None] = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        predicted_total = sum(matrix.get(label, {}).values())
        actual_total = sum(matrix.get(predicted, {}).get(label, 0) for predicted in labels)
        precision = tp / predicted_total if predicted_total else None
        recall = tp / actual_total if actual_total else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall > 0 else None
        result[f"precision_{label}"] = precision
        result[f"recall_{label}"] = recall
        result[f"f1_{label}"] = f1
    return ClassMetrics(**result)


def score_bucket_rows(outcomes: list[SignalOutcome], bucket_edges: list[float]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[SignalOutcome]] = {}
    for outcome in outcomes:
        if outcome.action not in {"LONG", "SHORT"}:
            continue
        key = (outcome.symbol, outcome.action, assign_score_bucket(outcome.score, bucket_edges))
        groups.setdefault(key, []).append(outcome)
    rows: list[dict[str, Any]] = []
    for (symbol, side, bucket), items in sorted(groups.items()):
        metrics = side_metrics(items)
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "bucket": bucket,
                "count": metrics.count,
                "hit8BeforeMinus5": metrics.hit8BeforeMinus5,
                "avgMfe": metrics.avgMfe60m,
                "avgMae": metrics.avgMae60m,
                "mfeMaeRatio": metrics.mfeMaeRatio,
                "expectancy": metrics.netExpectancy60m,
                "p90Mae": metrics.p90Mae,
            }
        )
    return rows


def score_calibration_status(bucket_rows_for_symbol: list[dict[str, Any]], min_bucket_count: int = 5) -> str:
    usable = [row for row in bucket_rows_for_symbol if int(row.get("count") or 0) >= min_bucket_count and row.get("expectancy") is not None]
    if len(usable) < 2:
        return "UNKNOWN"
    usable.sort(key=lambda row: str(row.get("bucket")))
    low = float(usable[0]["expectancy"])
    high = float(usable[-1]["expectancy"])
    if high > low:
        return "IMPROVES"
    return "NOT_CALIBRATED"


def classify_symbol(
    *,
    symbol: str,
    long_metrics: SideMetrics,
    short_metrics: SideMetrics,
    score_calibration: str,
    min_samples: int,
    leverage_proxy: float,
) -> tuple[str, float, list[str], list[str]]:
    warnings: list[str] = []
    actions: list[str] = []
    directional_samples = long_metrics.count + short_metrics.count
    if directional_samples < min_samples:
        warnings.append("insufficient_directional_samples")
        actions.append("insufficient_directional_samples")
        return "UNKNOWN", 0.0, warnings, actions

    sides = [metrics for metrics in (long_metrics, short_metrics) if metrics.count >= max(5, min_samples // 4)]
    positive = [metrics for metrics in sides if (metrics.netExpectancy60m or 0.0) > 0]
    negative = [metrics for metrics in sides if (metrics.netExpectancy60m or 0.0) < 0]
    high_mae = any((metrics.p90Mae or 0.0) * leverage_proxy > 0.10 for metrics in sides)
    if high_mae:
        warnings.append("high_p90_mae")
    if score_calibration == "NOT_CALIBRATED":
        warnings.append("score_not_calibrated")
        actions.append("score_not_calibrated")

    if positive and not warnings:
        actions.append("directional_edge_ok")
        return "GREEN", 0.8, warnings, actions
    if positive:
        actions.append("mixed_edge_require_extra_confirmation")
        return "YELLOW", 0.55, warnings, actions
    if negative and len(negative) == len(sides):
        warnings.append("negative_directional_expectancy")
        actions.append("reduce_confidence_until_directional_metrics")
        actions.append("require_extra_confirmation_for_symbol_side")
        return "RED", 0.25, warnings, actions
    actions.append("directional_edge_mixed")
    return "YELLOW", 0.5, warnings, actions


def summarize_symbol(
    symbol: str,
    outcomes60: list[SignalOutcome],
    bucket_rows: list[dict[str, Any]],
    *,
    min_samples: int,
    leverage_proxy: float,
) -> SymbolDirectionalSummary:
    long_outcomes = [item for item in outcomes60 if item.action == "LONG"]
    short_outcomes = [item for item in outcomes60 if item.action == "SHORT"]
    neutral_outcomes = [item for item in outcomes60 if item.action == "HOLD"]
    long_metrics = side_metrics(long_outcomes)
    short_metrics = side_metrics(short_outcomes)
    neutral = neutral_metrics(neutral_outcomes, threshold_return=0.08 / leverage_proxy)
    matrix = build_confusion(outcomes60)
    classes = class_metrics(matrix)
    calibration = score_calibration_status([row for row in bucket_rows if row.get("symbol") == symbol])
    status, confidence, warnings, actions = classify_symbol(
        symbol=symbol,
        long_metrics=long_metrics,
        short_metrics=short_metrics,
        score_calibration=calibration,
        min_samples=min_samples,
        leverage_proxy=leverage_proxy,
    )
    if long_metrics.count >= 5 and short_metrics.count >= 5:
        long_exp = long_metrics.netExpectancy60m or 0.0
        short_exp = short_metrics.netExpectancy60m or 0.0
        if long_exp > 0 and short_exp <= 0:
            actions.append("long_edge_good_short_edge_bad")
        if short_exp > 0 and long_exp <= 0:
            actions.append("short_edge_good_long_edge_bad")
    return SymbolDirectionalSummary(
        symbol=symbol,
        directionalStatus=status,
        directionalConfidence=confidence,
        sampleCount=len(outcomes60),
        long=long_metrics,
        short=short_metrics,
        neutral=neutral,
        classification=classes,
        confusionMatrix=matrix,
        scoreCalibration=calibration,
        directionalWarnings=warnings,
        recommendedAction=sorted(set(actions)),
    )


def build_outcomes(
    signals: list[Signal],
    candles_by_symbol: dict[str, list[Candle]],
    horizons: list[int],
    *,
    fee_bps: float,
    slippage_bps: float,
    leverage_proxy: float,
) -> dict[int, list[SignalOutcome]]:
    by_horizon: dict[int, list[SignalOutcome]] = {horizon: [] for horizon in horizons}
    for signal in signals:
        candles = candles_by_symbol.get(signal.symbol, [])
        for horizon in horizons:
            outcome = compute_signal_outcome(
                signal,
                candles,
                horizon,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                leverage_proxy=leverage_proxy,
            )
            if outcome is not None:
                by_horizon[horizon].append(outcome)
    return by_horizon


def audit_directional_metrics(
    *,
    symbols: list[str],
    start: datetime,
    end: datetime,
    horizons: list[int],
    fee_bps: float,
    slippage_bps: float,
    leverage_proxy: float,
    sample_every: int,
    score_buckets: list[float],
    include_neutral: bool,
    max_samples_per_symbol: int | None,
    db_path: Path = DEFAULT_DB_PATH,
    log_dir: Path = DEFAULT_LOG_DIR,
    min_samples: int = 20,
) -> dict[str, Any]:
    normalized_symbols = [symbol.replace("/", "").upper() for symbol in symbols]
    signals = load_signals(
        log_dir=log_dir,
        symbols=set(normalized_symbols),
        start=start,
        end=end,
        include_neutral=include_neutral,
        sample_every=max(1, sample_every),
    )
    if max_samples_per_symbol is not None:
        limited: list[Signal] = []
        counts: dict[str, int] = {}
        for signal in signals:
            count = counts.get(signal.symbol, 0)
            if count >= max_samples_per_symbol:
                continue
            counts[signal.symbol] = count + 1
            limited.append(signal)
        signals = limited

    max_horizon = max(horizons) if horizons else 60
    candles_by_symbol = {
        symbol: load_candles(db_path, symbol, start - timedelta(minutes=10), end + timedelta(minutes=max_horizon + 10))
        for symbol in normalized_symbols
    }
    outcomes_by_horizon = build_outcomes(
        signals,
        candles_by_symbol,
        horizons,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        leverage_proxy=leverage_proxy,
    )
    classification_horizon = 60 if 60 in horizons else horizons[0]
    outcomes60 = outcomes_by_horizon.get(classification_horizon, [])
    buckets = score_bucket_rows(outcomes60, score_buckets)
    summaries = [
        summarize_symbol(
            symbol,
            [outcome for outcome in outcomes60 if outcome.symbol == symbol],
            buckets,
            min_samples=min_samples,
            leverage_proxy=leverage_proxy,
        )
        for symbol in normalized_symbols
    ]
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "turbo_signals_jsonl_plus_sqlite_ohlcv",
        "signalsPath": str(log_dir / "turbo_signals_*.jsonl"),
        "ohlcvPath": str(db_path),
        "from": start.isoformat(),
        "to": end.isoformat(),
        "symbols": normalized_symbols,
        "horizonsMinutes": horizons,
        "classificationHorizonMinutes": classification_horizon,
        "feeBps": fee_bps,
        "slippageBps": slippage_bps,
        "leverageProxy": leverage_proxy,
        "includeNeutral": include_neutral,
        "sampleEvery": sample_every,
        "sampleCount": len(signals),
        "outcomeCount": len(outcomes60),
        "symbolSummaries": [asdict(item) for item in summaries],
        "scoreBuckets": buckets,
        "confusionRows": confusion_rows(summaries),
        "notes": [
            "Directional metrics are offline proxy metrics from persisted Aegis Turbo signals and future OHLCV.",
            "They are not training metrics and should not be treated as live trade guarantees.",
        ],
    }


def confusion_rows(summaries: list[SymbolDirectionalSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for predicted, actuals in summary.confusionMatrix.items():
            for actual, count in actuals.items():
                rows.append({"symbol": summary.symbol, "predicted": predicted, "actual": actual, "count": count})
    return rows


def write_json(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def summary_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report["symbolSummaries"]:
        rows.append(
            {
                "symbol": item["symbol"],
                "directionalStatus": item["directionalStatus"],
                "directionalConfidence": item["directionalConfidence"],
                "sampleCount": item["sampleCount"],
                "longCount": item["long"]["count"],
                "longExpectancy60m": item["long"]["netExpectancy60m"],
                "longHit8BeforeMinus5": item["long"]["hit8BeforeMinus5"],
                "shortCount": item["short"]["count"],
                "shortExpectancy60m": item["short"]["netExpectancy60m"],
                "shortHit8BeforeMinus5": item["short"]["hit8BeforeMinus5"],
                "neutralCount": item["neutral"]["count"],
                "scoreCalibration": item["scoreCalibration"],
                "warnings": "|".join(item["directionalWarnings"]),
                "recommendedAction": "|".join(item["recommendedAction"]),
            }
        )
    return rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summaries = report["symbolSummaries"]
    worst = sorted(summaries, key=lambda item: min(item["long"]["netExpectancy60m"] or 0, item["short"]["netExpectancy60m"] or 0))[:5]
    best = sorted(summaries, key=lambda item: max(item["long"]["netExpectancy60m"] or -999, item["short"]["netExpectancy60m"] or -999), reverse=True)[:5]
    lines = [
        "# Aegis Turbo Directional Metrics",
        "",
        f"Generated: {report['generatedAt']}",
        f"Source: {report['source']}",
        f"Window: {report['from']} -> {report['to']}",
        f"Signals: {report['sampleCount']} | Outcomes @{report['classificationHorizonMinutes']}m: {report['outcomeCount']}",
        "",
        "These metrics are proxy validation from persisted signals plus future OHLCV, not live guarantees.",
        "",
        "## Symbol Directional Status",
        "",
        "| Symbol | Status | Confidence | Long count | Long expectancy | Short count | Short expectancy | Score calibration | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        lines.append(
            "| {symbol} | {status} | {confidence:.2f} | {lc} | {le} | {sc} | {se} | {cal} | {warn} |".format(
                symbol=item["symbol"],
                status=item["directionalStatus"],
                confidence=float(item["directionalConfidence"]),
                lc=item["long"]["count"],
                le=item["long"]["netExpectancy60m"],
                sc=item["short"]["count"],
                se=item["short"]["netExpectancy60m"],
                cal=item["scoreCalibration"],
                warn=", ".join(item["directionalWarnings"]),
            )
        )
    lines.extend(["", "## Worst Symbols/Sides", ""])
    for item in worst:
        lines.append(f"- {item['symbol']}: LONG {item['long']['netExpectancy60m']} / SHORT {item['short']['netExpectancy60m']} warnings={item['directionalWarnings']}")
    lines.extend(["", "## Best Symbols/Sides", ""])
    for item in best:
        lines.append(f"- {item['symbol']}: LONG {item['long']['netExpectancy60m']} / SHORT {item['short']['netExpectancy60m']} calibration={item['scoreCalibration']}")
    lines.extend(["", "## Recommended Actions", ""])
    for item in summaries:
        lines.append(f"- {item['symbol']}: {', '.join(item['recommendedAction'])}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(report: dict[str, Any], out_dir: Path, timestamp: str, *, write_md: bool, write_json_flag: bool, write_csv_flag: bool) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if write_md:
        md = out_dir / f"aegis_turbo_directional_metrics_{timestamp}.md"
        write_markdown(report, md)
        paths["md"] = str(md)
    if write_json_flag:
        json_path = out_dir / f"aegis_turbo_directional_metrics_{timestamp}.json"
        write_json(report, json_path)
        paths["json"] = str(json_path)
    if write_csv_flag:
        summary = out_dir / f"aegis_turbo_directional_summary_{timestamp}.csv"
        write_csv(
            summary_csv_rows(report),
            summary,
            [
                "symbol",
                "directionalStatus",
                "directionalConfidence",
                "sampleCount",
                "longCount",
                "longExpectancy60m",
                "longHit8BeforeMinus5",
                "shortCount",
                "shortExpectancy60m",
                "shortHit8BeforeMinus5",
                "neutralCount",
                "scoreCalibration",
                "warnings",
                "recommendedAction",
            ],
        )
        paths["summary_csv"] = str(summary)
        bucket_path = out_dir / f"aegis_turbo_directional_score_buckets_{timestamp}.csv"
        write_csv(report["scoreBuckets"], bucket_path, ["symbol", "side", "bucket", "count", "hit8BeforeMinus5", "avgMfe", "avgMae", "mfeMaeRatio", "expectancy", "p90Mae"])
        paths["score_buckets_csv"] = str(bucket_path)
        confusion_path = out_dir / f"aegis_turbo_directional_confusion_{timestamp}.csv"
        write_csv(report["confusionRows"], confusion_path, ["symbol", "predicted", "actual", "count"])
        paths["confusion_csv"] = str(confusion_path)
    return paths


def parse_csv_list(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Aegis Turbo directional metrics from persisted signals and OHLCV.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--horizons-minutes", default="15,30,60,120")
    parser.add_argument("--fee-bps", type=float, default=8.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--leverage-proxy", type=float, default=20.0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--score-buckets", default="0.55,0.60,0.70,0.80,0.90")
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--max-samples-per-symbol", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-md", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_directional_metrics(
        symbols=parse_csv_list(args.symbols, str),
        start=parse_dt(args.from_date),
        end=parse_dt(args.to_date),
        horizons=parse_csv_list(args.horizons_minutes, int),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        leverage_proxy=float(args.leverage_proxy),
        sample_every=int(args.sample_every),
        score_buckets=parse_csv_list(args.score_buckets, float),
        include_neutral=bool(args.include_neutral),
        max_samples_per_symbol=args.max_samples_per_symbol,
        min_samples=int(args.min_samples),
        db_path=Path(args.db_path),
        log_dir=Path(args.log_dir),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = write_reports(
        report,
        Path(args.out_dir),
        timestamp,
        write_md=not args.no_md,
        write_json_flag=not args.no_json,
        write_csv_flag=not args.no_csv,
    )
    print(json.dumps({"paths": paths, "sampleCount": report["sampleCount"], "outcomeCount": report["outcomeCount"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
