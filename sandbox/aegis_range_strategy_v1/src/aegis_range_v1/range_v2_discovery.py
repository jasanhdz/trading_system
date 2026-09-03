from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .costs import BASELINE, STRESS_20, STRESS_30, adverse_fill, fee_return, funding_return, gross_return
from .models import Candle5m
from .numeric import canonical_decimal_12dp, iso_utc_millis
from .readiness import SYMBOLS, SealedPartitionGuard, SourceIntegrityError
from .train_backtest import TRAIN_END, TRAIN_START, _funding_slice, load_train_candles, load_train_funding

SCHEMA = "aegis-range-v2-canonical-opportunity-v1"
STATUS = "AEGIS_RANGE_V2_DISCOVERY_READY_FOR_REVIEW"
FLAGS = {"TRAIN": True, "CALIBRATION": False, "VALIDATION": False, "HOLDOUT": False}
RUN_A_HASHES = {
    "run_manifest.json": "5f62022f35fb38de174e6f7c573397d1c1ceebc75d76f7d848260c35456012b8",
    "candidate_metrics.json": "12f72be45420099d7ab0a56524ca934e791dfbaa9da0c0add87277d7939b656f",
    "episodes.jsonl.gz": "82989a83a68935ed44866afb2f5904e703c81e27b50acde5fe1c2fabd6af5270",
    "trades.jsonl.gz": "125f31dcb1bf27e6f183bbbb02da901a5133847e577196a9bd6a59be42cd4537",
}
REGIME_CACHE_MANIFEST_SHA256 = "a9699e874537bcdf14042e3d811594448e886b6811f48741b9b6ce5ad7e9c22b"
SCENARIOS = (BASELINE, STRESS_20, STRESS_30)
HORIZONS = (15, 30, 60, 120)


def _parse(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return result.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False)


def canonical_opportunity_id(trade: dict[str, Any]) -> str:
    """Identify the frozen setup, deliberately excluding contract and outcome fields."""
    value = "|".join(
        (
            SCHEMA,
            trade["symbol"],
            trade["side"],
            iso_utc_millis(_parse(trade["decision_at"])),
            iso_utc_millis(_parse(trade["entry_at"])),
            canonical_decimal_12dp(trade["support_at_entry"]),
            canonical_decimal_12dp(trade["resistance_at_entry"]),
            canonical_decimal_12dp(trade["midpoint_at_entry"]),
        )
    )
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def assign_unique_weights(trades: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    identifiers = [canonical_opportunity_id(row) for row in trades]
    multiplicity: dict[str, int] = defaultdict(int)
    for identifier in identifiers:
        multiplicity[identifier] += 1
    return [
        {**row, "canonical_opportunity_id": identifier, "group_multiplicity": multiplicity[identifier], "row_weight": 1.0, "unique_weight": 1.0 / multiplicity[identifier]}
        for row, identifier in zip(trades, identifiers)
    ]


def _side_extrema(side: str, prices: Sequence[float], entry: float) -> tuple[float, float, float, float]:
    low, high = min(*prices, entry), max(*prices, entry)
    favorable, adverse = (high, low) if side == "LONG" else (low, high)
    direction = 1.0 if side == "LONG" else -1.0
    return favorable, adverse, direction * (favorable - entry) / entry, -direction * (adverse - entry) / entry


def _extrema_context(prefix: str, price: float, entry: float, support: float, resistance: float, midpoint: float) -> dict[str, float]:
    result = {
        f"{prefix}_price": price,
        f"{prefix}_signed_return_from_entry": (price - entry) / entry,
        f"{prefix}_absolute_return_from_entry": abs(price - entry) / entry,
        f"{prefix}_frozen_range_position": (price - support) / (resistance - support),
    }
    for name, reference in (("entry", entry), ("support", support), ("resistance", resistance), ("midpoint", midpoint)):
        result[f"{prefix}_signed_distance_to_{name}"] = (price - reference) / reference
        result[f"{prefix}_absolute_distance_to_{name}"] = abs(price - reference) / reference
    return result


def opportunity_path(trade: dict[str, Any], candles: Sequence[Candle5m]) -> dict[str, Any]:
    entry_at, exit_at = _parse(trade["entry_at"]), _parse(trade["exit_at"])
    held = [bar for bar in candles if entry_at <= bar.open_time <= exit_at]
    if not held or held[0].open_time != entry_at or held[-1].open_time != exit_at:
        raise ValueError("TRADE_PATH_DATA_GAP")
    if any(b.open_time != a.open_time + timedelta(minutes=5) or b.segment_id != a.segment_id for a, b in zip(held, held[1:])):
        raise ValueError("TRADE_PATH_DATA_GAP")
    terminal, reason = held[-1], trade["exit_reason"]
    complete = held[:-1]
    terminal_model: list[float]
    if reason in {"TRADE_BREAKOUT", "MAX_HOLD", "STOP_GAP", "TARGET_GAP"}:
        terminal_model = [terminal.open]
    elif reason == "STOP":
        terminal_model = [terminal.open, float(trade["stop_at_entry"])]
    elif reason == "TARGET":
        terminal_model = [terminal.open, float(trade["target_at_entry"])]
    else:
        raise ValueError("UNKNOWN_EXIT_REASON")
    model_prices = [price for bar in complete for price in (bar.low, bar.high)] + terminal_model
    full_prices = [price for bar in held for price in (bar.low, bar.high)]
    entry = float(trade["entry_fill"])
    favorable, adverse, mfe, mae = _side_extrema(trade["side"], model_prices, entry)
    full_favorable, full_adverse, full_mfe, full_mae = _side_extrema(trade["side"], full_prices, entry)

    def first_time(price: float, favorable_side: bool) -> tuple[int, float]:
        direction = 1 if trade["side"] == "LONG" else -1
        for index, bar in enumerate(held):
            values = (bar.high, bar.low) if direction == 1 else (bar.low, bar.high)
            observed = values[0 if favorable_side else 1]
            if (observed >= price if direction * (1 if favorable_side else -1) == 1 else observed <= price):
                return index, index * 5.0
        return len(held) - 1, (len(held) - 1) * 5.0

    mfe_bars, mfe_minutes = first_time(favorable, True)
    mae_bars, mae_minutes = first_time(adverse, False)
    support, resistance, midpoint = (float(trade[key]) for key in ("support_at_entry", "resistance_at_entry", "midpoint_at_entry"))
    amplitude = resistance - support
    midpoint_hit = any(bar.low <= midpoint <= bar.high for bar in complete)
    midpoint_hit = midpoint_hit or (bool(terminal_model) and min(terminal_model) <= midpoint <= max(terminal_model))
    official_base = float(trade["exit_base"])
    if reason == "STOP":
        reconstructed = float(trade["stop_at_entry"])
    elif reason in {"TARGET", "TARGET_GAP"}:
        reconstructed = float(trade["target_at_entry"])
    else:
        reconstructed = terminal.open
    if not math.isclose(reconstructed, official_base, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("OFFICIAL_EXIT_RECONSTRUCTION_MISMATCH")
    exit_side = "SHORT" if trade["side"] == "LONG" else "LONG"
    reconstructed_fill = adverse_fill(reconstructed, exit_side, BASELINE.slippage_bps_per_side)
    if not math.isclose(reconstructed_fill, float(trade["exit_fill"]), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("OFFICIAL_EXIT_RECONSTRUCTION_MISMATCH")
    stop_touched = (adverse <= float(trade["stop_at_entry"])) if trade["side"] == "LONG" else (adverse >= float(trade["stop_at_entry"]))
    target_touched = (favorable >= float(trade["target_at_entry"])) if trade["side"] == "LONG" else (favorable <= float(trade["target_at_entry"]))
    return {
        **trade,
        "mfe": mfe,
        "mae": mae,
        **_extrema_context("mfe_extremum", favorable, entry, support, resistance, midpoint),
        **_extrema_context("mae_extremum", adverse, entry, support, resistance, midpoint),
        "mfe_bars": mfe_bars,
        "mfe_minutes": mfe_minutes,
        "mae_bars": mae_bars,
        "mae_minutes": mae_minutes,
        "full_terminal_mfe": full_mfe,
        "full_terminal_mae": full_mae,
        "full_terminal_favorable_price": full_favorable,
        "full_terminal_adverse_price": full_adverse,
        **_extrema_context("full_terminal_mfe_extremum", full_favorable, entry, support, resistance, midpoint),
        **_extrema_context("full_terminal_mae_extremum", full_adverse, entry, support, resistance, midpoint),
        "entry_position_in_range": (entry - support) / amplitude,
        "distance_to_entry": 0.0,
        "distance_to_support": abs(entry - support) / entry,
        "distance_to_resistance": abs(resistance - entry) / entry,
        "distance_to_midpoint": abs(midpoint - entry) / entry,
        "midpoint_hit_while_model_open": midpoint_hit,
        "stop_first": reason in {"STOP", "STOP_GAP"} and stop_touched,
        "target_first": reason in {"TARGET", "TARGET_GAP"} and target_touched,
    }


def stop_recovery(trade: dict[str, Any], candles: Sequence[Candle5m]) -> dict[str, Any]:
    if trade["exit_reason"] != "STOP":
        raise ValueError("STOP recovery requires an intrabar STOP")
    exit_at, side = _parse(trade["exit_at"]), trade["side"]
    bars = [bar for bar in candles if exit_at + timedelta(minutes=5) <= bar.open_time <= exit_at + timedelta(minutes=120)]
    contiguous = []
    expected = exit_at + timedelta(minutes=5)
    for bar in bars:
        if bar.open_time != expected or (contiguous and bar.segment_id != contiguous[-1].segment_id):
            break
        contiguous.append(bar)
        expected += timedelta(minutes=5)
    entry, midpoint, stop_fill = float(trade["entry_fill"]), float(trade["midpoint_at_entry"]), float(trade["exit_fill"])
    result: dict[str, Any] = {key: trade[key] for key in ("candidate_id", "canonical_opportunity_id", "unique_weight", "symbol", "side") if key in trade}
    entry_by_60 = entry_by_120 = midpoint_by_120 = adverse_120 = mature_120 = False
    for minutes in HORIZONS:
        selected = contiguous[: minutes // 5]
        highs, lows = [bar.high for bar in selected], [bar.low for bar in selected]
        entry_hit = bool(selected) and ((max(highs) >= entry) if side == "LONG" else (min(lows) <= entry))
        midpoint_hit = bool(selected) and ((max(highs) >= midpoint) if side == "LONG" else (min(lows) <= midpoint))
        favorable = None if not selected else max(0.0, (max(highs) - stop_fill) / stop_fill if side == "LONG" else (stop_fill - min(lows)) / stop_fill)
        adverse = None if not selected else max(0.0, (stop_fill - min(lows)) / stop_fill if side == "LONG" else (max(highs) - stop_fill) / stop_fill)
        result[f"horizon_{minutes}"] = {
            "complete_bars": len(selected),
            "mature": len(selected) == minutes // 5,
            "entry_recovered": entry_hit,
            "midpoint_reached": midpoint_hit,
            "favorable_excursion_from_stop": favorable,
            "adverse_continuation_from_stop": adverse,
        }
        if minutes == 60:
            entry_by_60 = entry_hit
        if minutes == 120:
            entry_by_120, midpoint_by_120, adverse_120 = entry_hit, midpoint_hit, bool(adverse and adverse > 0)
            mature_120 = len(selected) == minutes // 5
    if midpoint_by_120:
        category = "STOP_THEN_MIDPOINT_RECOVERY"
    elif entry_by_60:
        category = "STOP_THEN_ENTRY_RECOVERY"
    elif mature_120 and not entry_by_120 and adverse_120:
        category = "STOP_TRUE_FAILURE"
    else:
        category = "STOP_AMBIGUOUS"
    result["category"] = category
    return result


def _cancel_reason(trade: dict[str, Any], episode: dict[str, Any], bar: Candle5m, previous: Candle5m | None) -> str | None:
    if previous and (bar.open_time != previous.open_time + timedelta(minutes=5) or bar.segment_id != previous.segment_id):
        return "NONCONTIGUOUS_DATA"
    end = episode.get("episode_end_at")
    if end and _parse(end) <= bar.available_at:
        return "EPISODE_ENDED"
    midpoint, stop = float(trade["midpoint_at_entry"]), float(trade["stop_at_entry"])
    if bar.low <= midpoint <= bar.high:
        return "MIDPOINT_TOUCHED"
    if (trade["side"] == "LONG" and bar.low <= stop) or (trade["side"] == "SHORT" and bar.high >= stop):
        return "THESIS_FULLY_CROSSED"
    return None


def confirmation_entry(
    trade: dict[str, Any], episode: dict[str, Any], candles: Sequence[Candle5m], method: str
) -> dict[str, Any]:
    decision = _parse(trade["decision_at"])
    by_time = {bar.open_time: bar for bar in candles}
    rejection = by_time.get(decision - timedelta(minutes=5))
    if rejection is None:
        return {"status": "NO_TRADE", "reason": "MISSING_REJECTION_BAR", "method": method}
    count = 1 if method == "NEXT_CLOSE_PROGRESS" else 3 if method == "REJECTION_EXTREME_RECLAIM" else 0
    if not count:
        raise ValueError("unknown confirmation method")
    window = [by_time.get(decision + timedelta(minutes=5 * index)) for index in range(count)]
    previous = rejection
    qualifying: Candle5m | None = None
    for bar in window:
        if bar is None:
            return {"status": "NO_TRADE", "reason": "NONCONTIGUOUS_DATA", "method": method}
        cancelled = _cancel_reason(trade, episode, bar, previous)
        if cancelled:
            return {"status": "NO_TRADE", "reason": cancelled, "method": method}
        support, resistance, midpoint = (float(trade[key]) for key in ("support_at_entry", "resistance_at_entry", "midpoint_at_entry"))
        inside = support < bar.close < resistance
        if method == "NEXT_CLOSE_PROGRESS":
            condition = inside and ((bar.close < midpoint and bar.close > rejection.close) if trade["side"] == "LONG" else (bar.close > midpoint and bar.close < rejection.close))
        else:
            condition = (bar.close > rejection.high and bar.close < midpoint) if trade["side"] == "LONG" else (bar.close < rejection.low and bar.close > midpoint)
        if condition:
            qualifying = bar
            break
        previous = bar
    if qualifying is None:
        return {"status": "NO_TRADE", "reason": "WINDOW_EXPIRED", "method": method}
    entry_bar = by_time.get(qualifying.open_time + timedelta(minutes=5))
    if entry_bar is None or entry_bar.segment_id != qualifying.segment_id:
        return {"status": "NO_TRADE", "reason": "MISSING_NEXT_BAR", "method": method}
    if episode.get("episode_end_at") and _parse(episode["episode_end_at"]) <= entry_bar.open_time:
        return {"status": "NO_TRADE", "reason": "EPISODE_ENDED", "method": method}
    support, resistance = float(trade["support_at_entry"]), float(trade["resistance_at_entry"])
    if not support < entry_bar.open < resistance:
        return {"status": "NO_TRADE", "reason": "OPEN_OUTSIDE_RANGE", "method": method}
    fill = adverse_fill(entry_bar.open, trade["side"], BASELINE.slippage_bps_per_side)
    stop, target = float(trade["stop_at_entry"]), float(trade["target_at_entry"])
    reward = target - fill if trade["side"] == "LONG" else fill - target
    risk = fill - stop if trade["side"] == "LONG" else stop - fill
    if reward <= 0:
        reason = "TARGET_NOT_FAVORABLE"
    elif reward / fill < 0.0042:
        reason = "TARGET_DISTANCE_LT_42_BPS"
    elif risk <= 0 or reward / risk < 1.0:
        reason = "REWARD_RISK_LT_1"
    else:
        return {"status": "ENTERED", "reason": None, "method": method, "confirmation_at": iso_utc_millis(qualifying.available_at), "entry_at": iso_utc_millis(entry_bar.open_time), "entry_base": entry_bar.open, "entry_fill": fill}
    return {"status": "NO_TRADE", "reason": reason, "method": method}


def counterfactual_exit(
    trade: dict[str, Any], entry: dict[str, Any], candles: Sequence[Candle5m], funding: Sequence[tuple[datetime, float, float]] = ()
) -> dict[str, Any]:
    if entry["status"] != "ENTERED":
        return {**entry, "censored": False, "purged": False, "scenarios": None}
    entry_at = _parse(entry["entry_at"])
    bars = [bar for bar in candles if bar.open_time >= entry_at]
    adverse_closes = closed = 0
    pending: str | None = None
    exit_bar = None
    exit_base = None
    reason = None
    for index, bar in enumerate(bars):
        if index and (bar.open_time != bars[index - 1].open_time + timedelta(minutes=5) or bar.segment_id != bars[index - 1].segment_id):
            break
        if pending:
            exit_bar, exit_base, reason = bar, bar.open, pending
            break
        side, stop, target = trade["side"], float(trade["stop_at_entry"]), float(trade["target_at_entry"])
        if side == "LONG":
            if bar.open <= stop:
                exit_bar, exit_base, reason = bar, bar.open, "STOP_GAP"
            elif bar.open >= target:
                exit_bar, exit_base, reason = bar, target, "TARGET_GAP"
            elif bar.low <= stop:
                exit_bar, exit_base, reason = bar, stop, "STOP"
            elif bar.high >= target:
                exit_bar, exit_base, reason = bar, target, "TARGET"
        else:
            if bar.open >= stop:
                exit_bar, exit_base, reason = bar, bar.open, "STOP_GAP"
            elif bar.open <= target:
                exit_bar, exit_base, reason = bar, target, "TARGET_GAP"
            elif bar.high >= stop:
                exit_bar, exit_base, reason = bar, stop, "STOP"
            elif bar.low <= target:
                exit_bar, exit_base, reason = bar, target, "TARGET"
        if exit_bar:
            break
        closed += 1
        adverse = bar.close < float(trade["support_at_entry"]) - 0.10 * float(trade["ATR_entry"]) if side == "LONG" else bar.close > float(trade["resistance_at_entry"]) + 0.10 * float(trade["ATR_entry"])
        adverse_closes = adverse_closes + 1 if adverse else 0
        if adverse_closes == 2:
            pending = "TRADE_BREAKOUT"
        elif closed == 144:
            pending = "MAX_HOLD"
    if exit_bar is None or exit_base is None:
        return {**entry, "exit_reason": "CENSORED_DATA_BOUNDARY", "censored": True, "purged": True, "scenarios": None}
    events = _funding_slice(list(funding), entry_at, exit_bar.open_time)
    scenarios = {}
    exit_side = "SHORT" if trade["side"] == "LONG" else "LONG"
    for scenario in SCENARIOS:
        entry_fill = adverse_fill(float(entry["entry_base"]), trade["side"], scenario.slippage_bps_per_side)
        exit_fill = adverse_fill(exit_base, exit_side, scenario.slippage_bps_per_side)
        gross = gross_return(trade["side"], entry_fill, exit_fill)
        fees = fee_return(entry_fill, exit_fill, scenario.fee_bps_per_side)
        funding_value = funding_return(trade["side"], entry_fill, events)
        scenarios[scenario.name] = {"entry_fill": entry_fill, "exit_fill": exit_fill, "gross_return": gross, "fees": fees, "funding_return": funding_value, "net_return": gross - fees + funding_value}
    return {**entry, "exit_at": iso_utc_millis(exit_bar.open_time), "exit_base": exit_base, "exit_reason": reason, "holding_bars": closed, "censored": False, "purged": False, "scenarios": scenarios}


def suitability_observation(trade: dict[str, Any], episode: dict[str, Any], candles: Sequence[Candle5m]) -> dict[str, Any]:
    decision = _parse(trade["decision_at"])
    maturity = decision + timedelta(hours=12)
    end = min(maturity, _parse(episode["episode_end_at"])) if episode.get("episode_end_at") else maturity
    bars = [bar for bar in candles if decision <= bar.open_time and bar.available_at <= end]
    midpoint = float(trade["midpoint_at_entry"])
    reference = next((bar.close for bar in candles if bar.open_time == decision - timedelta(minutes=5)), None)
    touched = [index for index, bar in enumerate(bars) if bar.low <= midpoint <= bar.high]
    if reference is None:
        raise ValueError("MISSING_REJECTION_BAR")
    if trade["side"] == "LONG":
        mfe = max(0.0, *((bar.high - reference) / reference for bar in bars))
        mae = max(0.0, *((reference - bar.low) / reference for bar in bars))
    else:
        mfe = max(0.0, *((reference - bar.low) / reference for bar in bars))
        mae = max(0.0, *((bar.high - reference) / reference for bar in bars))
    breakout = bool(episode.get("episode_end_reason") == "CONFIRMED_BREAKOUT" and episode.get("episode_end_at") and _parse(episode["episode_end_at"]) <= maturity)
    return {
        "candidate_id": trade["candidate_id"],
        "canonical_opportunity_id": trade["canonical_opportunity_id"],
        "symbol": trade["symbol"],
        "decision_at": iso_utc_millis(decision),
        "maturity_at": iso_utc_millis(maturity),
        "mature_in_train": maturity <= TRAIN_END,
        "midpoint_hit": bool(touched),
        "bars_to_midpoint": None if not touched else touched[0] + 1,
        "minutes_to_midpoint": None if not touched else (touched[0] + 1) * 5,
        "structural_mfe_after_boundary_touch": mfe,
        "structural_mae_after_boundary_touch": mae,
        "breakout_after_boundary_touch": breakout,
        "false_range": bool(episode.get("false_range")),
    }


def prior_symbol_metrics(opportunities: Sequence[dict[str, Any]], decision_at: datetime, symbol: str) -> dict[str, Any]:
    start = decision_at - timedelta(days=60)
    rows = [row for row in opportunities if row["symbol"] == symbol and start <= _parse(row["decision_at"]) < decision_at and _parse(row["maturity_at"]) <= decision_at]
    unique: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        unique[row["canonical_opportunity_id"]].append(row)
    means = []
    for identifier, group in unique.items():
        means.append({"id": identifier, **{key: statistics.fmean(float(row[key]) for row in group if row[key] is not None) for key in ("midpoint_hit", "false_range", "breakout_after_boundary_touch")}, "bars": statistics.median([row["bars_to_midpoint"] for row in group if row["bars_to_midpoint"] is not None]) if any(row["bars_to_midpoint"] is not None for row in group) else None, "mae": statistics.median(row["structural_mae_after_boundary_touch"] for row in group), "mfe": statistics.median(row["structural_mfe_after_boundary_touch"] for row in group), "month": _parse(group[0]["decision_at"]).strftime("%Y-%m")})
    result: dict[str, Any] = {"symbol": symbol, "sample_count": len(means), "window_start": iso_utc_millis(start), "window_end_exclusive": iso_utc_millis(decision_at)}
    if len(means) < 30:
        return {**result, "status": "INSUFFICIENT_HISTORY", "score": None}
    monthly: dict[str, list[float]] = defaultdict(list)
    for row in means:
        monthly[row["month"]].append(row["midpoint_hit"])
    maes = [row["mae"] for row in means if row["mae"] is not None]
    mfes = [row["mfe"] for row in means if row["mfe"] is not None]
    result.update({
        "status": "ELIGIBLE",
        "score": statistics.fmean(row["midpoint_hit"] for row in means),
        "boundary_to_midpoint_reversion_rate": statistics.fmean(row["midpoint_hit"] for row in means),
        "false_range_rate": statistics.fmean(row["false_range"] for row in means),
        "median_bars_to_midpoint": statistics.median(row["bars"] for row in means if row["bars"] is not None) if any(row["bars"] is not None for row in means) else None,
        "median_structural_mae_after_boundary_touch": statistics.median(maes) if maes else None,
        "median_structural_mfe_after_boundary_touch": statistics.median(mfes) if mfes else None,
        "MFE_to_MAE_ratio": None if not maes or not mfes or statistics.median(maes) == 0 else statistics.median(mfes) / statistics.median(maes),
        "breakout_rate": statistics.fmean(row["breakout_after_boundary_touch"] for row in means),
        "utc_monthly_reversion_rates": {month: statistics.fmean(values) for month, values in sorted(monthly.items())},
        "utc_monthly_consistency": statistics.fmean(statistics.fmean(values) for values in monthly.values()),
        "utc_month_sample_count": len(monthly),
    })
    return result


def causal_terciles(metrics: Sequence[dict[str, Any]]) -> dict[str, str]:
    eligible = sorted((row for row in metrics if row["status"] == "ELIGIBLE"), key=lambda row: (row["boundary_to_midpoint_reversion_rate"], row["symbol"]))
    if len(eligible) < 3:
        return {
            row["symbol"]: "INSUFFICIENT_HISTORY" if row["status"] == "INSUFFICIENT_HISTORY" else "INSUFFICIENT_CROSS_SECTION"
            for row in metrics
        }
    result = {row["symbol"]: "INSUFFICIENT_HISTORY" for row in metrics}
    for rank, row in enumerate(eligible):
        result[row["symbol"]] = ("LOW", "MEDIUM", "HIGH")[min(2, 3 * rank // len(eligible))]
    return result


def deterministic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="ascii", newline="") as text:
                for row in rows:
                    text.write(_json(row) + "\n")
                    count += 1
    return count, _sha256_file(path)


def verify_authority(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    flags = SealedPartitionGuard.access_flags(environment)
    if flags != FLAGS:
        raise PermissionError("AEGIS_RANGE_V2_TRAIN_PARTITION_VIOLATION")
    run_a = (repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a").resolve()
    output = output_root.resolve()
    if output == run_a or run_a in output.parents:
        raise PermissionError("AEGIS_RANGE_V2_OUTPUT_INSIDE_RUN_A")
    for name, expected in RUN_A_HASHES.items():
        if _sha256_file(run_a / name) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_V2_BLOCKED_BY_RUN_A_DRIFT")
    manifest = json.loads((run_a / "run_manifest.json").read_text(encoding="ascii"))
    if manifest["partition_flags"] != FLAGS or manifest["artifacts"]["trades"]["rows"] != 22016:
        raise SourceIntegrityError("AEGIS_RANGE_V2_BLOCKED_BY_RUN_A_DRIFT")
    regime_manifest_path = run_a / "regime_cache_manifest.json"
    if _sha256_file(regime_manifest_path) != REGIME_CACHE_MANIFEST_SHA256:
        raise SourceIntegrityError("AEGIS_RANGE_V2_BLOCKED_BY_REGIME_CACHE_DRIFT")
    regime_manifest = json.loads(regime_manifest_path.read_text(encoding="ascii"))
    if set(regime_manifest["caches"]) != set(SYMBOLS):
        raise SourceIntegrityError("AEGIS_RANGE_V2_BLOCKED_BY_REGIME_CACHE_DRIFT")
    for symbol, item in regime_manifest["caches"].items():
        cache_path = run_a / "regime_cache" / f"{symbol}.csv.gz"
        if item["symbol"] != symbol or _sha256_file(cache_path) != item["sha256"]:
            raise SourceIntegrityError("AEGIS_RANGE_V2_BLOCKED_BY_REGIME_CACHE_DRIFT")
    return {"flags": flags, "run_a": run_a, "manifest": manifest, "regime_manifest": regime_manifest}


def _weighted_metrics(rows: Sequence[dict[str, Any]], weight_key: str, scenario: str) -> dict[str, Any]:
    active = [row for row in rows if not row.get("censored") and row.get("scenarios")]
    total = sum(float(row[weight_key]) for row in rows)
    retained = sum(float(row[weight_key]) for row in active)
    if not retained:
        return {"retained_rows": 0, "effective_unique_opportunities": 0.0, "abstention_rate": 1.0, "gross_expectancy": None, "net_expectancy": None, "profit_factor": None}
    values = [(float(row[weight_key]), row["scenarios"][scenario]) for row in active]
    mean = lambda key: sum(weight * value[key] for weight, value in values) / retained
    net_positive = sum(weight * max(value["net_return"], 0.0) for weight, value in values)
    net_negative = sum(weight * min(value["net_return"], 0.0) for weight, value in values)
    fee_factor = sum(weight * (1.0 + value["exit_fill"] / value["entry_fill"]) for weight, value in values) / retained
    return {
        "retained_rows": len(active),
        "effective_unique_opportunities": retained,
        "abstention_rate": None if not total else 1.0 - retained / total,
        "incremental_abstention_rate_relative_to_accepted_v1_rows": None if not total else 1.0 - retained / total,
        "gross_expectancy": mean("gross_return"),
        "net_expectancy": mean("net_return"),
        "profit_factor": "Infinity" if net_negative == 0 else net_positive / abs(net_negative),
        "win_rate": sum(weight * (value["net_return"] > 0) for weight, value in values) / retained,
        "stop_rate": sum(float(row[weight_key]) * row["exit_reason"].startswith("STOP") for row in active) / retained,
        "target_rate": sum(float(row[weight_key]) * row["exit_reason"].startswith("TARGET") for row in active) / retained,
        "break_even_round_trip_fee_bps": (mean("gross_return") + mean("funding_return")) * 10000.0,
        "break_even_fee_bps_per_side": (mean("gross_return") + mean("funding_return")) * 10000.0 / fee_factor,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return [json.loads(line) for line in handle]


def _load_rejection_context(run_a: Path, symbol: str, decisions: set[datetime]) -> dict[datetime, dict[str, Any]]:
    wanted = {decision - timedelta(minutes=5) for decision in decisions}
    result: dict[datetime, dict[str, Any]] = {}
    with gzip.open(run_a / "regime_cache" / f"{symbol}.csv.gz", "rt", encoding="ascii", newline="") as handle:
        for row in csv.DictReader(handle):
            opened = _parse(row["open_time"])
            if opened in wanted:
                percentile = float(row["atr_percentile"])
                result[opened + timedelta(minutes=5)] = {
                    "atr_percentile_at_rejection": percentile,
                    "technical_regime_at_rejection": row["technical_regime"],
                    "atr_regime": "LOW" if percentile <= 1 / 3 else "MEDIUM" if percentile <= 2 / 3 else "HIGH",
                }
    if set(result) != decisions:
        raise SourceIntegrityError("AEGIS_RANGE_V2_REGIME_CONTEXT_JOIN_INVARIANT")
    return result


def _weighted_mean(rows: Sequence[dict[str, Any]], weight: str, key: str) -> float | None:
    selected = [(float(row[weight]), row[key]) for row in rows if row.get(key) is not None]
    denominator = sum(item[0] for item in selected)
    return None if not denominator else sum(item[0] * float(item[1]) for item in selected) / denominator


def _tercile_cutpoints(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("terciles require observations")
    if len(values) == 1:
        return values[0], values[0]
    first, second = statistics.quantiles(values, n=3, method="inclusive")
    return first, second


def _category_distribution(rows: Sequence[dict[str, Any]], weight: str) -> dict[str, Any]:
    total = sum(float(row[weight]) for row in rows)
    categories = sorted({row["category"] for row in rows})
    return {
        "candidate_rows": len(rows),
        "effective_weight": total,
        "categories": {category: {"candidate_rows": sum(row["category"] == category for row in rows), "effective_weight": sum(float(row[weight]) for row in rows if row["category"] == category), "rate": None if not total else sum(float(row[weight]) for row in rows if row["category"] == category) / total} for category in categories},
    }


def failure_anatomy_summary(paths: Sequence[dict[str, Any]], recoveries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique_amplitudes: dict[str, list[float]] = defaultdict(list)
    for row in paths:
        unique_amplitudes[row["canonical_opportunity_id"]].append(float(row["range_amplitude_pct"]))
    amplitude_means = [statistics.fmean(values) for values in unique_amplitudes.values()]
    low_cut, high_cut = _tercile_cutpoints(amplitude_means)

    def decorate(row: dict[str, Any]) -> dict[str, Any]:
        age = float(row["episode_age_minutes"])
        amplitude = float(row["range_amplitude_pct"])
        return {
            **row,
            "episode_age_bin": "LT_12H" if age < 720 else "12_TO_24H" if age < 1440 else "24_TO_48H",
            "range_amplitude_pct_tercile": "LOW" if amplitude <= low_cut else "MEDIUM" if amplitude <= high_cut else "HIGH",
        }

    decorated = [decorate(row) for row in recoveries]
    result: dict[str, Any] = {"range_amplitude_pct_tercile_cutpoints": {"low_medium": low_cut, "medium_high": high_cut}, "views": {}}
    for view, weight in (("candidate_weighted", "row_weight"), ("unique_opportunity_weighted", "unique_weight")):
        path_weight = sum(float(row[weight]) for row in paths)
        failure = {
            "candidate_rows": len(paths),
            "effective_weight": path_weight,
            "weighted_mfe": _weighted_mean(paths, weight, "mfe"),
            "weighted_mae": _weighted_mean(paths, weight, "mae"),
            "midpoint_hit_rate": _weighted_mean(paths, weight, "midpoint_hit_while_model_open"),
            "stop_first_rate": _weighted_mean(paths, weight, "stop_first"),
            "target_first_rate": _weighted_mean(paths, weight, "target_first"),
        }
        recovery: dict[str, Any] = _category_distribution(decorated, weight)
        recovery["horizons"] = {}
        for minutes in HORIZONS:
            key = f"horizon_{minutes}"
            all_rows = [{**row, **row[key]} for row in decorated]
            rows = [row for row in all_rows if row["mature"]]
            recovery["horizons"][str(minutes)] = {
                "candidate_rows": len(rows),
                "effective_weight": sum(float(row[weight]) for row in rows),
                "entry_recovery_rate": _weighted_mean(rows, weight, "entry_recovered"),
                "midpoint_recovery_rate": _weighted_mean(rows, weight, "midpoint_reached"),
                "favorable_excursion_from_stop": _weighted_mean(rows, weight, "favorable_excursion_from_stop"),
                "adverse_continuation_from_stop": _weighted_mean(rows, weight, "adverse_continuation_from_stop"),
                "mature_rate": _weighted_mean(all_rows, weight, "mature"),
            }
        recovery["category_breakdowns"] = {}
        for dimension in ("side", "symbol", "month", "atr_regime", "episode_age_bin", "range_amplitude_pct_tercile"):
            values = sorted({row[dimension] for row in decorated})
            recovery["category_breakdowns"][dimension] = {value: _category_distribution([row for row in decorated if row[dimension] == value], weight) for value in values}
        result["views"][view] = {"v1_failure_anatomy": failure, "stop_recovery": recovery}
    return result


def confirmation_q_summary(rows: Sequence[dict[str, Any]], originals: dict[tuple[str, str], dict[str, Any]], weight: str) -> dict[str, Any]:
    censored_stopped = [row for row in rows if row.get("censored") and originals[(row["candidate_id"], row["canonical_opportunity_id"])]["exit_reason"].startswith("STOP")]
    censored_targeted = [row for row in rows if row.get("censored") and originals[(row["candidate_id"], row["canonical_opportunity_id"])]["exit_reason"].startswith("TARGET")]
    stopped = [row for row in rows if not row.get("censored") and originals[(row["candidate_id"], row["canonical_opportunity_id"])]["exit_reason"].startswith("STOP")]
    targeted = [row for row in rows if not row.get("censored") and originals[(row["candidate_id"], row["canonical_opportunity_id"])]["exit_reason"].startswith("TARGET")]
    stop_weight = sum(float(row[weight]) for row in stopped)
    target_weight = sum(float(row[weight]) for row in targeted)
    filtered_stop = [row for row in stopped if row["status"] == "NO_TRADE"]
    changed_stop = [row for row in stopped if row["status"] == "ENTERED" and not row["exit_reason"].startswith("STOP")]
    filtered_target = [row for row in targeted if row["status"] == "NO_TRADE"]
    lost_target = [row for row in targeted if row["status"] == "NO_TRADE" or not row.get("exit_reason", "").startswith("TARGET")]

    def amount(selected: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        return {"candidate_rows": len(selected), "effective_weight": sum(float(row[weight]) for row in selected)}

    return {
        "original_stop": amount(stopped),
        "original_target": amount(targeted),
        "censored_original_stop_excluded": amount(censored_stopped),
        "censored_original_target_excluded": amount(censored_targeted),
        "original_stop_filtered": {**amount(filtered_stop), "rate": None if not stop_weight else amount(filtered_stop)["effective_weight"] / stop_weight},
        "original_stop_no_longer_stop": {**amount(changed_stop), "rate": None if not stop_weight else amount(changed_stop)["effective_weight"] / stop_weight},
        "original_stop_avoided_rate": None if not stop_weight else (amount(filtered_stop)["effective_weight"] + amount(changed_stop)["effective_weight"]) / stop_weight,
        "original_target_filtered": {**amount(filtered_target), "rate": None if not target_weight else amount(filtered_target)["effective_weight"] / target_weight},
        "original_target_lost": {**amount(lost_target), "rate": None if not target_weight else amount(lost_target)["effective_weight"] / target_weight},
    }


def execute_discovery(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    authority = verify_authority(repo_root, output_root, environment)
    run_a = authority["run_a"]
    trades = assign_unique_weights(_load_jsonl(run_a / "trades.jsonl.gz"))
    if len(trades) != 22016 or len({row["canonical_opportunity_id"] for row in trades}) != 382:
        raise SourceIntegrityError("AEGIS_RANGE_V2_CANONICAL_OPPORTUNITY_INVARIANT")
    required_episodes = {(row["candidate_id"], row["range_episode_id"]) for row in trades}
    episodes: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(run_a / "episodes.jsonl.gz", "rt", encoding="ascii") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["candidate_id"], row["range_episode_id"])
            if key in required_episodes:
                episodes[key] = row
    if set(episodes) != required_episodes:
        raise SourceIntegrityError("AEGIS_RANGE_V2_EPISODE_JOIN_INVARIANT")

    paths: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    suitability: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_symbol[trade["symbol"]].append(trade)
    for symbol in SYMBOLS:
        candles = load_train_candles(repo_root, symbol)
        candle_times = [bar.open_time for bar in candles]
        funding = load_train_funding(repo_root, symbol)
        rejection_context = _load_rejection_context(run_a, symbol, {_parse(row["decision_at"]) for row in by_symbol[symbol]})
        for trade in by_symbol[symbol]:
            episode = episodes[(trade["candidate_id"], trade["range_episode_id"])]
            decision_at = _parse(trade["decision_at"])
            context = rejection_context[decision_at]
            window_start = min(_parse(trade["entry_at"]), decision_at - timedelta(minutes=5))
            window_end = max(_parse(trade["exit_at"]) + timedelta(minutes=120), decision_at + timedelta(hours=13))
            local_candles = candles[bisect_left(candle_times, window_start):bisect_right(candle_times, window_end)]
            amplitude = trade["resistance_at_entry"] - trade["support_at_entry"]
            amplitude_pct = amplitude / trade["midpoint_at_entry"]
            path = opportunity_path(trade, local_candles)
            path.update({"month": trade["decision_at"][:7], "range_amplitude": amplitude, "range_amplitude_pct": amplitude_pct, "atr_range_ratio": trade["ATR_entry"] / amplitude, "episode_age_minutes": (_parse(trade["decision_at"]) - _parse(episode["range_confirmed_at"])).total_seconds() / 60, **context})
            paths.append(path)
            if trade["exit_reason"] == "STOP":
                recovery = stop_recovery(trade, local_candles)
                recovery.update({"row_weight": 1.0, "month": trade["decision_at"][:7], "range_amplitude": amplitude, "range_amplitude_pct": amplitude_pct, "ATR_entry": trade["ATR_entry"], "atr_range_ratio": trade["ATR_entry"] / amplitude, "episode_age_minutes": (_parse(trade["decision_at"]) - _parse(episode["range_confirmed_at"])).total_seconds() / 60, **context})
                recoveries.append(recovery)
            for method in ("NEXT_CLOSE_PROGRESS", "REJECTION_EXTREME_RECLAIM"):
                entry = confirmation_entry(trade, episode, local_candles, method)
                counterfactuals.append({**{key: trade[key] for key in ("candidate_id", "canonical_opportunity_id", "unique_weight", "symbol", "side", "decision_at")}, "row_weight": 1.0, **counterfactual_exit(trade, entry, local_candles, funding)})
            suitability.append(suitability_observation(trade, episode, local_candles))

    paths.sort(key=lambda row: (row["canonical_opportunity_id"], row["candidate_id"]))
    recoveries.sort(key=lambda row: (row["canonical_opportunity_id"], row["candidate_id"]))
    counterfactuals.sort(key=lambda row: (row["canonical_opportunity_id"], row["method"], row["candidate_id"]))
    suitability.sort(key=lambda row: (row["decision_at"], row["canonical_opportunity_id"], row["candidate_id"]))
    suitability_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in suitability:
        suitability_groups[row["canonical_opportunity_id"]].append(row)
    opportunity_suitability = []
    for identifier, group in suitability_groups.items():
        first = group[0]
        if not first["mature_in_train"]:
            continue
        touched = [row for row in group if row["midpoint_hit"]]
        opportunity_suitability.append({
            "canonical_opportunity_id": identifier,
            "symbol": first["symbol"],
            "decision_at": first["decision_at"],
            "maturity_at": first["maturity_at"],
            "mature_in_train": first["mature_in_train"],
            "candidate_rows": len(group),
            "candidate_observations": group,
            "midpoint_hit": statistics.fmean(row["midpoint_hit"] for row in group),
            "false_range": statistics.fmean(row["false_range"] for row in group),
            "breakout_after_boundary_touch": statistics.fmean(row["breakout_after_boundary_touch"] for row in group),
            "bars_to_midpoint": statistics.fmean(row["bars_to_midpoint"] for row in touched) if touched else None,
            "minutes_to_midpoint": statistics.fmean(row["minutes_to_midpoint"] for row in touched) if touched else None,
            "structural_mfe_after_boundary_touch": statistics.fmean(row["structural_mfe_after_boundary_touch"] for row in group),
            "structural_mae_after_boundary_touch": statistics.fmean(row["structural_mae_after_boundary_touch"] for row in group),
        })
    opportunity_suitability.sort(key=lambda row: (row["decision_at"], row["canonical_opportunity_id"]))
    for row in opportunity_suitability:
        decision = _parse(row["decision_at"])
        prior = [prior_symbol_metrics(opportunity_suitability, decision, symbol) for symbol in SYMBOLS]
        row["prior_symbol_metrics"] = next(item for item in prior if item["symbol"] == row["symbol"])
        row["prior_tercile"] = causal_terciles(prior)[row["symbol"]]
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in (("opportunity_paths.jsonl.gz", paths), ("stop_recovery.jsonl.gz", recoveries), ("confirmation_counterfactuals.jsonl.gz", counterfactuals), ("symbol_suitability.jsonl.gz", opportunity_suitability)):
        count, digest = deterministic_gzip_jsonl(output_root / name, rows)
        artifacts[name] = {"rows": count, "sha256": digest}

    original = [{**row, "row_weight": 1.0, "censored": False} for row in trades]
    for row in original:
        row["scenarios"] = row["scenarios"]
    methodology_contract = {
        "opportunity_path_extrema": "CONSERVATIVE_TERMINAL_MODEL_WITH_FULL_TERMINAL_OHLC_AMBIGUITY_DISCLOSURE",
        "post_stop_start": "FIRST_COMPLETE_5M_BAR_STRICTLY_AFTER_STOP_EXIT_BAR",
        "suitability_maturity_lag_hours": 12,
        "prior_window_days": 60,
        "prior_minimum_unique_opportunities": 30,
        "unique_weighting": "EACH_CANDIDATE_ROW_WEIGHTED_BY_INVERSE_CANONICAL_OPPORTUNITY_MULTIPLICITY",
    }
    summary: dict[str, Any] = {"label": "DISCOVERY_ONLY", "authority": "NO_WHITELIST_AUTHORITY", "methodology": "HYPOTHESIS_GENERATION", "methodology_contract": methodology_contract, "v1_official_aggregate_abstention_rate": 0.9848864285400466, "system_abstention_definition": "INCREMENTAL_RELATIVE_TO_ACCEPTED_V1_ROWS", "views": {}, "q_metrics": {}}
    systems = {"V1_ORIGINAL": original, "CONFIRMATION_A": [row for row in counterfactuals if row["method"] == "NEXT_CLOSE_PROGRESS"], "CONFIRMATION_B": [row for row in counterfactuals if row["method"] == "REJECTION_EXTREME_RECLAIM"]}
    for system, rows in systems.items():
        summary["views"][system] = {view: {scenario.name: _weighted_metrics(rows, weight, scenario.name) for scenario in SCENARIOS} for view, weight in (("candidate_weighted", "row_weight"), ("unique_opportunity_weighted", "unique_weight"))}
    original_by_key = {(row["candidate_id"], row["canonical_opportunity_id"]): row for row in trades}
    for method, label in (("NEXT_CLOSE_PROGRESS", "CONFIRMATION_A"), ("REJECTION_EXTREME_RECLAIM", "CONFIRMATION_B")):
        rows = [row for row in counterfactuals if row["method"] == method]
        q = {}
        for view, weight in (("candidate_weighted", "row_weight"), ("unique_opportunity_weighted", "unique_weight")):
            q[view] = confirmation_q_summary(rows, original_by_key, weight)
        summary["q_metrics"][label] = q
    summary["v1_failure_anatomy_and_stop_recovery"] = failure_anatomy_summary(paths, recoveries)
    path_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paths:
        path_groups[row["canonical_opportunity_id"]].append(row)
    unique_economics = []
    for identifier, group in path_groups.items():
        unique_economics.append({
            "canonical_opportunity_id": identifier,
            "symbol": group[0]["symbol"],
            "month": group[0]["decision_at"][:7],
            "stop": statistics.fmean(row["exit_reason"].startswith("STOP") for row in group),
            "target": statistics.fmean(row["exit_reason"].startswith("TARGET") for row in group),
            "mfe": statistics.fmean(row["mfe"] for row in group),
            "mae": statistics.fmean(row["mae"] for row in group),
            "gross": statistics.fmean(row["gross_return"] for row in group),
            "net": statistics.fmean(row["net_return"] for row in group),
        })
    suitability_by_id = {row["canonical_opportunity_id"]: row for row in opportunity_suitability}
    tercile_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_economics = [row for row in unique_economics if row["canonical_opportunity_id"] in suitability_by_id]
    for row in unique_economics:
        observation = suitability_by_id[row["canonical_opportunity_id"]]
        if observation["prior_tercile"] in {"LOW", "MEDIUM", "HIGH"}:
            tercile_rows[observation["prior_tercile"]].append({**row, "midpoint_hit": observation["midpoint_hit"]})
    summary["future_by_prior_tercile"] = {tercile: {"unique_opportunities": len(rows), "stop_rate": statistics.fmean(row["stop"] for row in rows), "midpoint_hit_rate": statistics.fmean(row["midpoint_hit"] for row in rows), "gross_expectancy": statistics.fmean(row["gross"] for row in rows), "net_expectancy": statistics.fmean(row["net"] for row in rows)} for tercile, rows in sorted(tercile_rows.items())}
    symbol_table = []
    for symbol in SYMBOLS:
        economic = [row for row in unique_economics if row["symbol"] == symbol]
        observed = [suitability_by_id[row["canonical_opportunity_id"]] for row in economic]
        months = sorted({row["month"] for row in economic})
        monthly = {}
        for month in months:
            selected = [row for row in economic if row["month"] == month]
            selected_observations = [suitability_by_id[row["canonical_opportunity_id"]] for row in selected]
            monthly[month] = {
                "unique_opportunities": len(selected),
                "reversion_rate": statistics.fmean(row["midpoint_hit"] for row in selected_observations),
                "midpoint_hit_rate": statistics.fmean(row["midpoint_hit"] for row in selected_observations),
                "stop_rate": statistics.fmean(row["stop"] for row in selected),
                "false_range_rate": statistics.fmean(row["false_range"] for row in selected_observations),
                "structural_mfe_after_boundary_touch": statistics.fmean(row["structural_mfe_after_boundary_touch"] for row in selected_observations),
                "structural_mae_after_boundary_touch": statistics.fmean(row["structural_mae_after_boundary_touch"] for row in selected_observations),
                "gross_expectancy": statistics.fmean(row["gross"] for row in selected),
                "net_expectancy": statistics.fmean(row["net"] for row in selected),
            }
        symbol_table.append({
            "symbol": symbol,
            "unique_opportunities": len(economic),
            "reversion_rate": statistics.fmean(row["midpoint_hit"] for row in observed),
            "stop_rate": statistics.fmean(row["stop"] for row in economic),
            "midpoint_hit_rate": statistics.fmean(row["midpoint_hit"] for row in observed),
            "false_range_rate": statistics.fmean(row["false_range"] for row in observed),
            "structural_mfe_after_boundary_touch": statistics.fmean(row["structural_mfe_after_boundary_touch"] for row in observed),
            "structural_mae_after_boundary_touch": statistics.fmean(row["structural_mae_after_boundary_touch"] for row in observed),
            "gross_expectancy": statistics.fmean(row["gross"] for row in economic),
            "net_expectancy": statistics.fmean(row["net"] for row in economic),
            "monthly": monthly,
            "label": "DISCOVERY_ONLY",
            "authority": "NO_WHITELIST_AUTHORITY",
        })
    summary["symbol_table"] = symbol_table
    summary_path = output_root / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    artifacts[summary_path.name] = {"rows": 1, "sha256": _sha256_file(summary_path)}
    source_sha = _sha256_file(Path(__file__))
    manifest = {"schema_version": "aegis-range-v2-discovery-v1", "status": STATUS, "flags": FLAGS, "train": {"start_inclusive": iso_utc_millis(TRAIN_START), "end_exclusive": iso_utc_millis(TRAIN_END)}, "input_hashes": {**RUN_A_HASHES, "regime_cache_manifest.json": REGIME_CACHE_MANIFEST_SHA256}, "regime_cache_hashes": {symbol: authority["regime_manifest"]["caches"][symbol]["sha256"] for symbol in SYMBOLS}, "canonical_opportunities": 382, "candidate_rows": 22016, "source_methodology": "TRAIN_2024_ONLY_HYPOTHESIS_GENERATION", "methodology_contract": methodology_contract, "source_sha": source_sha, "labels": ["DISCOVERY_ONLY", "NO_WHITELIST_AUTHORITY"], "artifacts": artifacts}
    manifest_path = output_root / "diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return {**manifest, "diagnostics_manifest_sha256": _sha256_file(manifest_path)}
