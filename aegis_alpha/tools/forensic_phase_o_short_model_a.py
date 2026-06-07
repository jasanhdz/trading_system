#!/usr/bin/env python3
"""Read-only forensic audit for Phase O SHORT model behavior."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_live_entries import (  # noqa: E402
    ALL_SYMBOLS,
    REPO,
    json_safe,
    parse_dt,
    to_float,
    utc_stamp,
    write_csv,
)
from aegis_alpha.tools.audit_phase_o_short_live_quality import (  # noqa: E402
    build_quality_audit,
    candles_between,
    compute_short_mae_mfe,
    first_return_short,
    hit_before_stop_short,
    load_candles,
    pearson,
    safe_div,
    short_pnl,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], value: float | None) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if value is None or not vals:
        return None
    below = sum(1 for v in vals if v <= value)
    return below / len(vals)


def zscore(values: list[float], value: float | None) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if value is None or len(vals) < 2:
        return None
    mu = mean(vals)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
    if sigma == 0:
        return None
    return (value - mu) / sigma


def effect_size(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((va + vb) / 2)
    if pooled == 0:
        return None
    return (ma - mb) / pooled


def candle_features(candles: list[dict[str, Any]], idx: int) -> dict[str, float]:
    if idx <= 0 or idx >= len(candles):
        return {}
    c = candles[idx]
    close = float(c["close"])
    high = float(c["high"])
    low = float(c["low"])
    opn = float(c["open"])
    def ret(n: int) -> float | None:
        j = idx - n
        if j < 0:
            return None
        prev = float(candles[j]["close"])
        return safe_div(close - prev, prev)
    prev = candles[max(0, idx - 24): idx + 1]
    highs = [float(x["high"]) for x in prev]
    lows = [float(x["low"]) for x in prev]
    closes = [float(x["close"]) for x in prev]
    ranges = [(float(x["high"]) - float(x["low"])) / max(float(x["close"]), 1e-12) for x in prev]
    ema25_proxy = mean(closes[-25:]) if closes else close
    ema99_proxy = mean(closes[-99:]) if closes else close
    recent_high = max(highs) if highs else high
    recent_low = min(lows) if lows else low
    rng = max(high - low, 1e-12)
    range24 = max(recent_high - recent_low, 1e-12)
    lower_wick = min(opn, close) - low
    upper_wick = high - max(opn, close)
    return {
        "return_15m": ret(3) or 0.0,
        "return_30m": ret(6) or 0.0,
        "return_60m": ret(12) or 0.0,
        "range_expansion": ranges[-1] / max(mean(ranges[:-1] or ranges), 1e-12),
        "realized_vol_24": math.sqrt(mean([(r - mean(ranges)) ** 2 for r in ranges])) if len(ranges) > 1 else 0.0,
        "close_location": (close - low) / rng,
        "distance_ema25": safe_div(close - ema25_proxy, ema25_proxy) or 0.0,
        "distance_ema99": safe_div(close - ema99_proxy, ema99_proxy) or 0.0,
        "distance_to_recent_low": safe_div(close - recent_low, close) or 0.0,
        "distance_to_recent_high": safe_div(recent_high - close, close) or 0.0,
        "breakdown_strength": safe_div(recent_low - close, close) or 0.0,
        "lower_wick_ratio": lower_wick / rng,
        "upper_wick_ratio": upper_wick / rng,
        "room_to_fall": safe_div(close - recent_low, close) or 0.0,
        "close_location_24": (close - recent_low) / range24,
    }


def find_candle_index(candles: list[dict[str, Any]], ts: datetime) -> int | None:
    candidates = [i for i, c in enumerate(candles) if c["timestamp"] <= ts]
    return candidates[-1] if candidates else None


def reconstruct_features(trades: list[dict[str, Any]], candles_by_symbol: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        symbol = str(trade.get("symbol"))
        opened = parse_dt(trade.get("opened_at"))
        candles = candles_by_symbol.get(symbol, [])
        idx = find_candle_index(candles, opened) if opened else None
        feats = candle_features(candles, idx) if idx is not None else {}
        prior = candles[max(0, (idx or 0) - 288 * 30): idx or 0]
        dist: dict[str, list[float]] = defaultdict(list)
        for j in range(1, len(prior)):
            for k, v in candle_features(prior, j).items():
                dist[k].append(v)
        row = {"trade_id": trade.get("trade_id"), "symbol": symbol, "opened_at": trade.get("opened_at")}
        for k, v in feats.items():
            row[k] = v
            row[f"{k}_percentile"] = percentile(dist.get(k, []), v)
            row[f"{k}_zscore"] = zscore(dist.get(k, []), v)
            if row[f"{k}_zscore"] is not None:
                row[f"{k}_ood"] = abs(row[f"{k}_zscore"]) > 3
        rows.append(row)
    return rows


def classify_live_trade(row: dict[str, Any]) -> str:
    pnl = to_float(row.get("net_pnl_estimated"))
    mfe = to_float(row.get("mfe_roe")) or 0.0
    mae = to_float(row.get("mae_roe")) or 0.0
    ratio = to_float(row.get("mfe_mae_ratio")) or 0.0
    eff = to_float(row.get("close_efficiency"))
    if pnl is not None and pnl < -1.0 and mae > 0.25:
        return "LIVE_TRADE_BIG_LOSS"
    if pnl is not None and pnl > 0 and mfe > mae and ratio >= 1.2:
        return "LIVE_TRADE_GOOD_ENTRY"
    if pnl is not None and pnl >= 0 and (ratio < 1.0 or mae > 0.15):
        return "LIVE_TRADE_SAVED_BY_MANAGEMENT"
    if pnl is not None and pnl < 0 and (mfe < mae or (eff is not None and eff < 0.25)):
        return "LIVE_TRADE_BAD_ENTRY"
    return "LIVE_TRADE_UNKNOWN"


def winners_losers_features(trades: list[dict[str, Any]], features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["trade_id"]: r for r in trades}
    feature_names = [k for k in features[0].keys() if k not in {"trade_id", "symbol", "opened_at"} and not k.endswith(("_percentile", "_zscore", "_ood"))] if features else []
    out = []
    for feat in feature_names:
        winners, losers = [], []
        for row in features:
            trade = by_id.get(row["trade_id"], {})
            value = to_float(row.get(feat))
            if value is None:
                continue
            if trade.get("winner") is True:
                winners.append(value)
            elif trade.get("winner") is False:
                losers.append(value)
        if winners and losers:
            wm, lm = mean(winners), mean(losers)
            direction = "higher_in_winners" if wm > lm else "higher_in_losers"
            out.append({
                "feature": feat,
                "winner_mean": wm,
                "loser_mean": lm,
                "diff": wm - lm,
                "effect_size": effect_size(winners, losers),
                "direction": direction,
                "interpretation": interpret_feature(feat, direction),
            })
    return sorted(out, key=lambda r: abs(r.get("effect_size") or 0), reverse=True)


def interpret_feature(feature: str, direction: str) -> str:
    if feature in {"lower_wick_ratio", "distance_to_recent_low", "room_to_fall"} and direction == "higher_in_losers":
        return "possible_reclaim_or_low_room_to_fall_risk"
    if feature in {"range_expansion", "realized_vol_24"} and direction == "higher_in_losers":
        return "volatility_or_extension_risk"
    if feature in {"return_15m", "return_30m", "return_60m"} and direction == "higher_in_losers":
        return "late_short_or_rebound_risk"
    return "candidate_separator"


def score_calibration(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [(to_float(r.get("model_score")), to_float(r.get("net_pnl_estimated")), to_float(r.get("mfe_roe")), to_float(r.get("mae_roe"))) for r in trades]
    valid = [v for v in valid if all(x is not None for x in v)]
    rows = []
    if len(valid) < 5:
        return [{"scope": "global", "status": "SCORE_CALIBRATION_INSUFFICIENT_DATA", "sample_count": len(valid)}]
    scores = [v[0] for v in valid]
    corr_pnl = pearson(scores, [v[1] for v in valid])
    corr_mfe = pearson(scores, [v[2] for v in valid])
    corr_mae = pearson(scores, [v[3] for v in valid])
    status = "SCORE_CALIBRATION_OK" if (corr_pnl or 0) > 0.25 and (corr_mfe or 0) > 0.20 else "SCORE_CALIBRATION_WEAK"
    if (corr_pnl or 0) < -0.1:
        status = "SCORE_CALIBRATION_INVERTED"
    rows.append({"scope": "global", "sample_count": len(valid), "score_vs_pnl": corr_pnl, "score_vs_mfe": corr_mfe, "score_vs_mae": corr_mae, "status": status})
    for symbol in sorted({r["symbol"] for r in trades}):
        group = [r for r in trades if r["symbol"] == symbol and to_float(r.get("model_score")) is not None and to_float(r.get("net_pnl_estimated")) is not None]
        if len(group) < 3:
            rows.append({"scope": "symbol", "symbol": symbol, "sample_count": len(group), "status": "SCORE_CALIBRATION_INSUFFICIENT_DATA"})
            continue
        corr = pearson([to_float(r["model_score"]) for r in group], [to_float(r["net_pnl_estimated"]) for r in group])
        rows.append({"scope": "symbol", "symbol": symbol, "sample_count": len(group), "score_vs_pnl": corr, "status": "SCORE_CALIBRATION_OK" if (corr or 0) > 0.25 else "SCORE_CALIBRATION_WEAK"})
    return rows


def random_entry_quality(symbol: str, entry_idx: int, candles: list[dict[str, Any]], leverage: float, horizon_bars: int) -> dict[str, Any] | None:
    if entry_idx < 0 or entry_idx >= len(candles) - 2:
        return None
    entry = float(candles[entry_idx]["close"])
    window = candles[entry_idx:min(len(candles), entry_idx + max(2, horizon_bars))]
    mm = compute_short_mae_mfe(entry, leverage, window)
    exit_price = float(window[-1]["close"])
    pnl_proxy = (entry - exit_price) / entry * leverage
    return {"mfe_roe": mm.get("mfe_roe"), "mae_roe": mm.get("mae_roe"), "quality": (mm.get("mfe_roe") or 0) - (mm.get("mae_roe") or 0), "pnl_proxy_roe": pnl_proxy}


def random_baseline(trades: list[dict[str, Any]], candles_by_symbol: dict[str, list[dict[str, Any]]], n: int = 100) -> list[dict[str, Any]]:
    rng = random.Random(73071)
    rows = []
    for trade in trades:
        symbol = str(trade["symbol"])
        candles = candles_by_symbol.get(symbol, [])
        opened = parse_dt(trade.get("opened_at"))
        if not opened or len(candles) < 50:
            rows.append({"trade_id": trade["trade_id"], "symbol": symbol, "status": "INSUFFICIENT_RANDOM_BASELINE"})
            continue
        idx = find_candle_index(candles, opened)
        horizon = max(3, min(24, int((to_float(trade.get("time_in_trade_seconds")) or 3600) // 300)))
        candidates = [i for i, c in enumerate(candles[:-horizon]) if c["timestamp"].date() == opened.date() and (idx is None or abs(i - idx) > 6)]
        if not candidates:
            candidates = list(range(10, max(10, len(candles) - horizon)))
        sample = rng.sample(candidates, min(n, len(candidates)))
        randoms = [r for i in sample if (r := random_entry_quality(symbol, i, candles, to_float(trade.get("leverage")) or 20.0, horizon))]
        if len(randoms) < 10:
            rows.append({"trade_id": trade["trade_id"], "symbol": symbol, "status": "INSUFFICIENT_RANDOM_BASELINE"})
            continue
        live_quality = (to_float(trade.get("mfe_roe")) or 0) - (to_float(trade.get("mae_roe")) or 0)
        live_pnl = to_float(trade.get("roe")) or to_float(trade.get("pnl_proxy_roe")) or 0
        q_vals = [r["quality"] for r in randoms]
        pnl_vals = [r["pnl_proxy_roe"] for r in randoms]
        mfe_vals = [r["mfe_roe"] for r in randoms if r["mfe_roe"] is not None]
        mae_vals = [r["mae_roe"] for r in randoms if r["mae_roe"] is not None]
        rows.append({
            "trade_id": trade["trade_id"],
            "symbol": symbol,
            "random_count": len(randoms),
            "live_quality": live_quality,
            "live_quality_percentile": percentile(q_vals, live_quality),
            "live_pnl_percentile": percentile(pnl_vals, live_pnl),
            "live_mfe_percentile": percentile(mfe_vals, to_float(trade.get("mfe_roe"))),
            "live_mae_percentile": percentile(mae_vals, to_float(trade.get("mae_roe"))),
            "live_better_than_random_median": (percentile(q_vals, live_quality) or 0) >= 0.5,
            "status": "OK",
        })
    return rows


def classify_random_global(rows: list[dict[str, Any]]) -> str:
    ok = [r for r in rows if r.get("status") == "OK"]
    if len(ok) < 5:
        return "INSUFFICIENT_RANDOM_BASELINE"
    above = sum(1 for r in ok if r.get("live_better_than_random_median"))
    avg_pct = mean([r.get("live_quality_percentile") or 0 for r in ok])
    if avg_pct >= 0.6 and above / len(ok) >= 0.6:
        return "MODEL_BEATS_RANDOM"
    if avg_pct < 0.45 and above / len(ok) < 0.5:
        return "MODEL_NOT_BETTER_THAN_RANDOM"
    return "MODEL_MIXED_VS_RANDOM"


def management_comparison(trades: list[dict[str, Any]], candles_by_symbol: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        symbol = str(trade["symbol"])
        candles = candles_by_symbol.get(symbol, [])
        opened = parse_dt(trade.get("opened_at"))
        idx = find_candle_index(candles, opened) if opened else None
        entry = to_float(trade.get("entry_price"))
        leverage = to_float(trade.get("leverage")) or 20.0
        real = to_float(trade.get("roe"))
        if idx is None or entry is None:
            rows.append({"trade_id": trade["trade_id"], "symbol": symbol, "classification": "UNKNOWN"})
            continue
        out = {"trade_id": trade["trade_id"], "symbol": symbol, "real_roe": real}
        for bars in (3, 6, 12, 24):
            if idx + bars < len(candles):
                close = float(candles[idx + bars]["close"])
                out[f"simple_exit_roe_{bars}"] = (entry - close) / entry * leverage
        simple_vals = [v for k, v in out.items() if k.startswith("simple_exit_roe_") and v is not None]
        best_simple = max(simple_vals) if simple_vals else None
        out["best_simple_roe"] = best_simple
        out["management_value_added"] = real - best_simple if real is not None and best_simple is not None else None
        mfe = to_float(trade.get("mfe_roe")) or 0
        mae = to_float(trade.get("mae_roe")) or 0
        pnl = to_float(trade.get("net_pnl_estimated"))
        if pnl is not None and pnl > 0 and mfe > mae:
            cls = "ENTRY_HAS_EDGE_WITHOUT_MANAGEMENT"
        elif pnl is not None and pnl >= 0 and mfe <= mae:
            cls = "ENTRY_DEPENDS_ON_MANAGEMENT"
        elif pnl is not None and pnl > -0.2 and mae > mfe:
            cls = "MANAGEMENT_MASKS_BAD_ENTRY"
        else:
            cls = "UNKNOWN"
        out["classification"] = cls
        rows.append(out)
    return rows


def symbol_diagnostics(trades: list[dict[str, Any]], random_rows: list[dict[str, Any]], management_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_random = defaultdict(list)
    for row in random_rows:
        by_random[row["symbol"]].append(row)
    by_mgmt = defaultdict(list)
    for row in management_rows:
        by_mgmt[row["symbol"]].append(row)
    out = []
    for symbol in sorted({r["symbol"] for r in trades}):
        group = [r for r in trades if r["symbol"] == symbol]
        pnl = sum(to_float(r.get("net_pnl_estimated")) or 0 for r in group)
        avg_mae = mean([to_float(r.get("mae_roe")) or 0 for r in group]) if group else 0
        avg_random = mean([r.get("live_quality_percentile") or 0 for r in by_random[symbol] if r.get("status") == "OK"]) if by_random[symbol] else None
        mgmt_dep = sum(1 for r in by_mgmt[symbol] if r.get("classification") in {"ENTRY_DEPENDS_ON_MANAGEMENT", "MANAGEMENT_MASKS_BAD_ENTRY"})
        if len(group) < 2:
            mode = "SAMPLE_TOO_SMALL"
            rec = "TOO_EARLY"
        elif pnl < 0 and avg_mae > 0.12:
            mode = "MAE_DANGER_UNDERPREDICTED"
            rec = "REBUILD_WITH_STRONGER_MAE_TARGET"
        elif avg_random is not None and avg_random < 0.45:
            mode = "MODEL_OVERTRADES_SYMBOL"
            rec = "DISABLE_LIVE_UNTIL_RETRAIN"
        elif mgmt_dep >= max(1, len(group) // 2):
            mode = "MANAGEMENT_DEPENDENT"
            rec = "REBUILD_WITH_STRONGER_MAE_TARGET"
        elif pnl > 0:
            mode = "SYMBOL_OK"
            rec = "KEEP_FOR_MODEL_REBUILD"
        else:
            mode = "SCORE_NOT_CALIBRATED"
            rec = "REBUILD_SYMBOL_SPECIFIC_ONLY"
        out.append({
            "symbol": symbol,
            "trade_count": len(group),
            "pnl_total": pnl,
            "avg_mae_roe": avg_mae,
            "avg_random_quality_percentile": avg_random,
            "management_dependent_count": mgmt_dep,
            "failure_mode": mode,
            "recommendation": rec,
        })
    return out


def global_diagnosis(trades: list[dict[str, Any]], calibration: list[dict[str, Any]], random_status: str, symbol_rows: list[dict[str, Any]]) -> str:
    if len(trades) < 10:
        return "MODEL_TOO_EARLY"
    pnl = sum(to_float(r.get("net_pnl_estimated")) or 0 for r in trades)
    cal_status = calibration[0].get("status") if calibration else None
    bad_symbols = [r for r in symbol_rows if r.get("failure_mode") not in {"SYMBOL_OK", "SAMPLE_TOO_SMALL"}]
    if pnl < 0 and random_status == "MODEL_NOT_BETTER_THAN_RANDOM":
        return "MODEL_OVERTRADING_NO_EDGE"
    if cal_status in {"SCORE_CALIBRATION_WEAK", "SCORE_CALIBRATION_INVERTED"}:
        return "MODEL_SCORE_NOT_CALIBRATED"
    if bad_symbols and pnl < 0:
        return "MODEL_SYMBOL_MIXED"
    if pnl < 0:
        return "MODEL_ENTRY_EDGE_WEAK"
    return "MODEL_ENTRY_EDGE_CONFIRMED"


def proposed_model_design(symbol_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep = [r["symbol"] for r in symbol_rows if r.get("recommendation") == "KEEP_FOR_MODEL_REBUILD"]
    exclude = [r["symbol"] for r in symbol_rows if r.get("recommendation") in {"DISABLE_LIVE_UNTIL_RETRAIN", "REBUILD_WITH_STRONGER_MAE_TARGET"}]
    return {
        "name": "operable_short_quality_v4",
        "labels": [
            "short_clean_entry_v4: hit5/hit8 before minus3/minus5, MAE capped, time_to_mfe fast, MFE/MAE > threshold, fees included",
            "short_bad_entry_v4: early MAE, low MFE, management_saved_trade, high score but low realized quality",
            "short_premium_allowed_v4: expected net quality high, danger low, symbol calibration positive",
        ],
        "initial_symbols": keep or ["SUIUSDT", "ADAUSDT", "AVAXUSDT", "SOLUSDT"],
        "exclude_or_repair": exclude,
        "promotion_metrics": ["positive net expectancy after fees", "p90 MAE controlled", "beats random baseline", "score calibration positive", "premium outperforms normal"],
    }


def build_forensic(args: argparse.Namespace) -> dict[str, Any]:
    quality_args = argparse.Namespace(
        from_ts=args.from_ts,
        to=args.to,
        symbols=args.symbols,
        include_open=True,
        include_closed=True,
        machine_gun_window_seconds=300,
        max_log_bytes_per_file=250_000_000,
    )
    quality = build_quality_audit(quality_args)
    trades = quality["trades"]
    start = parse_dt(args.from_ts) or datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = parse_dt(args.to) if args.to != "now" else datetime.now(timezone.utc)
    end = end or datetime.now(timezone.utc)
    candles_by_symbol = {s: load_candles(s, start, end) for s in sorted({r["symbol"] for r in trades})}
    features = reconstruct_features(trades, candles_by_symbol) if args.include_feature_reconstruction else []
    for row in trades:
        row["live_trade_classification"] = classify_live_trade(row)
        row["big_loss"] = row["live_trade_classification"] == "LIVE_TRADE_BIG_LOSS"
        row["big_win"] = (to_float(row.get("net_pnl_estimated")) or 0) > 1.0
    winners_losers = winners_losers_features(trades, features) if features else []
    calibration = score_calibration(trades) if args.include_score_calibration else []
    random_rows = random_baseline(trades, candles_by_symbol, n=args.random_n) if args.include_random_baseline else []
    random_status = classify_random_global(random_rows) if random_rows else "INSUFFICIENT_RANDOM_BASELINE"
    management = management_comparison(trades, candles_by_symbol) if args.include_management_comparison else []
    symbols = symbol_diagnostics(trades, random_rows, management) if args.include_symbol_diagnostics else []
    diagnosis = global_diagnosis(trades, calibration, random_status, symbols)
    design = proposed_model_design(symbols)
    return {
        "schema_version": "short_model_forensic_a_v1",
        "created_at": now_iso(),
        "mode": "READ_ONLY",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "global_diagnosis": diagnosis,
        "random_baseline_status": random_status,
        "summary": {
            "trade_count": len(trades),
            "closed_count": sum(1 for r in trades if r.get("status") == "CLOSED"),
            "pnl_total": sum(to_float(r.get("net_pnl_estimated")) or 0 for r in trades),
            "classifications": dict(Counter(r.get("live_trade_classification") for r in trades)),
            "score_calibration": calibration[0].get("status") if calibration else None,
        },
        "live_trades": trades,
        "features": features,
        "winners_losers": winners_losers,
        "symbols": symbols,
        "buckets": quality.get("buckets", []),
        "score_calibration": calibration,
        "random_baseline": random_rows,
        "management": management,
        "research_distribution": research_distribution_stub(),
        "proposed_model": design,
        "recommendations": build_recommendations(diagnosis, random_status, symbols, design),
        "confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2": True,
            "no_orders": True,
            "no_env": True,
            "no_push": True,
            "no_commit": True,
        },
    }


def research_distribution_stub() -> list[dict[str, Any]]:
    root = Path("/home/jasan/Develop")
    reports = sorted(root.glob("aegis_phase_o*.json"))[-10:]
    return [{"source": str(p), "classification": "LIVE_TOO_EARLY", "note": "Research metric extraction deferred; live comparison used current realized distribution."} for p in reports]


def build_recommendations(diagnosis: str, random_status: str, symbols: list[dict[str, Any]], design: dict[str, Any]) -> list[dict[str, Any]]:
    recs = [
        {"area": "global", "recommendation": "DO_NOT_SCALE_CAPITAL", "reason": f"diagnosis={diagnosis}, random_baseline={random_status}"},
        {"area": "model", "recommendation": "REBUILD_OPERABLE_SHORT_QUALITY_V4", "reason": "Current model opens technically valid trades but weak live net quality."},
        {"area": "live", "recommendation": "NO_FILTER_CHANGES_IN_THIS_PHASE", "reason": "Forensic phase only; do not patch bot filters from this run."},
    ]
    for row in symbols:
        recs.append({"area": "symbol", "symbol": row["symbol"], "recommendation": row["recommendation"], "reason": row["failure_mode"]})
    recs.append({"area": "target", "recommendation": design["name"], "reason": "; ".join(design["labels"])})
    return recs


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# SHORT Model Forensic A",
        "",
        "## Safety",
        "- read-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Global diagnosis: `{payload['global_diagnosis']}`",
        f"- Random baseline: `{payload['random_baseline_status']}`",
        f"- Score calibration: `{s.get('score_calibration')}`",
        f"- Trades: `{s['trade_count']}` closed=`{s['closed_count']}` pnl_total=`{s['pnl_total']}`",
        f"- Live classifications: `{s['classifications']}`",
        "",
        "## Live Trades",
        "| symbol | bucket | score | pnl | mfe | mae | class |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["live_trades"]:
        lines.append(f"| {row.get('symbol')} | {row.get('bucket')} | {row.get('model_score')} | {row.get('net_pnl_estimated')} | {row.get('mfe_roe')} | {row.get('mae_roe')} | {row.get('live_trade_classification')} |")
    lines += ["", "## Winners vs Losers Top Features"]
    for row in payload["winners_losers"][:15]:
        lines.append(f"- {row.get('feature')}: winner_mean={row.get('winner_mean')} loser_mean={row.get('loser_mean')} effect={row.get('effect_size')} interpretation={row.get('interpretation')}")
    lines += ["", "## Score Calibration"]
    for row in payload["score_calibration"]:
        lines.append(f"- {row}")
    lines += ["", "## Random Baseline"]
    ok = [r for r in payload["random_baseline"] if r.get("status") == "OK"]
    if ok:
        avg = mean([r.get("live_quality_percentile") or 0 for r in ok])
        above = sum(1 for r in ok if r.get("live_better_than_random_median"))
        lines.append(f"- avg_live_quality_percentile={avg} above_random_median={above}/{len(ok)}")
    lines += ["", "## Entry vs Management"]
    for row in payload["management"][:25]:
        lines.append(f"- {row.get('symbol')} {row.get('trade_id')}: {row.get('classification')} real_roe={row.get('real_roe')} best_simple={row.get('best_simple_roe')} value_added={row.get('management_value_added')}")
    lines += ["", "## Symbol Diagnosis"]
    for row in payload["symbols"]:
        lines.append(f"- {row.get('symbol')}: failure={row.get('failure_mode')} recommendation={row.get('recommendation')} pnl={row.get('pnl_total')} random_pct={row.get('avg_random_quality_percentile')}")
    lines += ["", "## Proposed Model"]
    lines.append(f"- Name: `{payload['proposed_model']['name']}`")
    for label in payload["proposed_model"]["labels"]:
        lines.append(f"- {label}")
    lines.append(f"- Initial symbols: `{payload['proposed_model']['initial_symbols']}`")
    lines.append(f"- Exclude/repair: `{payload['proposed_model']['exclude_or_repair']}`")
    lines += ["", "## Recommendations"]
    for row in payload["recommendations"]:
        lines.append(f"- {row.get('area')} {row.get('symbol','')}: `{row.get('recommendation')}` - {row.get('reason')}")
    lines += ["", "## Confirmations"]
    for key, value in payload["confirmations"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    base = out_dir / f"aegis_short_model_forensic_a_{ts}"
    paths = {
        "md": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "live_trades_csv": str(out_dir / f"aegis_short_model_forensic_live_trades_{ts}.csv"),
        "features_csv": str(out_dir / f"aegis_short_model_forensic_features_{ts}.csv"),
        "winners_losers_csv": str(out_dir / f"aegis_short_model_forensic_winners_losers_{ts}.csv"),
        "symbols_csv": str(out_dir / f"aegis_short_model_forensic_symbols_{ts}.csv"),
        "buckets_csv": str(out_dir / f"aegis_short_model_forensic_buckets_{ts}.csv"),
        "score_calibration_csv": str(out_dir / f"aegis_short_model_forensic_score_calibration_{ts}.csv"),
        "random_baseline_csv": str(out_dir / f"aegis_short_model_forensic_random_baseline_{ts}.csv"),
        "management_csv": str(out_dir / f"aegis_short_model_forensic_management_{ts}.csv"),
        "recommendations_csv": str(out_dir / f"aegis_short_model_forensic_recommendations_{ts}.csv"),
    }
    payload["reports"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(paths["md"]), payload)
    write_csv(Path(paths["live_trades_csv"]), payload["live_trades"])
    write_csv(Path(paths["features_csv"]), payload["features"])
    write_csv(Path(paths["winners_losers_csv"]), payload["winners_losers"])
    write_csv(Path(paths["symbols_csv"]), payload["symbols"])
    write_csv(Path(paths["buckets_csv"]), payload["buckets"])
    write_csv(Path(paths["score_calibration_csv"]), payload["score_calibration"])
    write_csv(Path(paths["random_baseline_csv"]), payload["random_baseline"])
    write_csv(Path(paths["management_csv"]), payload["management"])
    write_csv(Path(paths["recommendations_csv"]), payload["recommendations"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_ts", default="2026-06-01T00:00:00Z")
    p.add_argument("--to", default="now")
    p.add_argument("--out-dir", default="/home/jasan/Develop")
    p.add_argument("--symbols", default=",".join([s for s in ALL_SYMBOLS if s != "LINKUSDT"]))
    p.add_argument("--include-live-trades", action="store_true")
    p.add_argument("--include-feature-reconstruction", action="store_true")
    p.add_argument("--include-score-calibration", action="store_true")
    p.add_argument("--include-random-baseline", action="store_true")
    p.add_argument("--include-training-distribution", action="store_true")
    p.add_argument("--include-symbol-diagnostics", action="store_true")
    p.add_argument("--include-bucket-diagnostics", action="store_true")
    p.add_argument("--include-management-comparison", action="store_true")
    p.add_argument("--random-n", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_forensic(args)
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({
        "global_diagnosis": payload["global_diagnosis"],
        "random_baseline_status": payload["random_baseline_status"],
        "summary": payload["summary"],
        "reports": paths,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
