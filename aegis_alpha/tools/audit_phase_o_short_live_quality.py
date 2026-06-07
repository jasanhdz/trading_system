#!/usr/bin/env python3
"""Read-only quality audit for real Phase O SHORT live trades."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
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
    collect_phase_o_trades,
    date_strings,
    detect_machine_gun,
    events_by_trade,
    infer_bucket,
    json_safe,
    parse_dt,
    read_log_rows,
    to_float,
    utc_stamp,
    write_csv,
)

DB_PATH = REPO / "data" / "binance_candles.db"
FEE_RATE_PER_SIDE = 0.0004


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def symbol_db_name(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def dt_to_db(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def load_candles(symbol: str, start: datetime, end: datetime, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    start_q = dt_to_db(start - timedelta(minutes=10))
    end_q = dt_to_db(end + timedelta(minutes=10))
    con = sqlite3.connect(db_path, timeout=3)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select timestamp, open, high, low, close, volume
            from ohlcv_data
            where symbol = ? and timeframe = '5m' and timestamp >= ? and timestamp <= ?
            order by timestamp
            """,
            (symbol_db_name(symbol), start_q, end_q),
        ).fetchall()
    finally:
        con.close()
    out = []
    for row in rows:
        ts = parse_dt(str(row["timestamp"]).replace(" ", "T") + "Z")
        if ts is None:
            continue
        out.append({
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return out


def candles_between(candles: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [c for c in candles if start <= c["timestamp"] <= end]


def short_pnl(entry: float, exit_price: float, qty: float) -> float:
    return (entry - exit_price) * qty


def roe_from_pnl(pnl: float | None, margin: float | None) -> float | None:
    return safe_div(pnl, margin)


def compute_short_mae_mfe(entry: float, leverage: float, candles: list[dict[str, Any]]) -> dict[str, Any]:
    if entry <= 0 or not candles:
        return {
            "mfe_price_move": None,
            "mae_price_move": None,
            "mfe_roe": None,
            "mae_roe": None,
            "time_to_mfe": None,
            "time_to_mae": None,
            "mfe_mae_ratio": None,
        }
    best_fav = -1e9
    worst_adv = -1e9
    best_ts = None
    worst_ts = None
    for c in candles:
        fav = (entry - float(c["low"])) / entry
        adv = (float(c["high"]) - entry) / entry
        if fav > best_fav:
            best_fav = fav
            best_ts = c["timestamp"]
        if adv > worst_adv:
            worst_adv = adv
            worst_ts = c["timestamp"]
    mfe = max(0.0, best_fav)
    mae = max(0.0, worst_adv)
    return {
        "mfe_price_move": mfe,
        "mae_price_move": mae,
        "mfe_roe": mfe * leverage,
        "mae_roe": mae * leverage,
        "time_to_mfe": best_ts.isoformat() if best_ts else None,
        "time_to_mae": worst_ts.isoformat() if worst_ts else None,
        "mfe_mae_ratio": safe_div(mfe, mae) if mae > 0 else None,
    }


def hit_before_stop_short(entry: float, leverage: float, candles: list[dict[str, Any]], target_roe: float, stop_roe: float) -> dict[str, Any]:
    target_move = target_roe / leverage
    stop_move = stop_roe / leverage
    target_price = entry * (1 - target_move)
    stop_price = entry * (1 + stop_move)
    for idx, c in enumerate(candles, start=1):
        hit = float(c["low"]) <= target_price
        stop = float(c["high"]) >= stop_price
        if hit and stop:
            return {"hit": False, "stopped": True, "ambiguous_hit_stop": True, "time_to_hit": None, "time_to_stop": idx}
        if stop:
            return {"hit": False, "stopped": True, "ambiguous_hit_stop": False, "time_to_hit": None, "time_to_stop": idx}
        if hit:
            return {"hit": True, "stopped": False, "ambiguous_hit_stop": False, "time_to_hit": idx, "time_to_stop": None}
    return {"hit": False, "stopped": False, "ambiguous_hit_stop": False, "time_to_hit": None, "time_to_stop": None}


def first_return_short(entry: float, candles: list[dict[str, Any]], bars: int) -> float | None:
    if entry <= 0 or len(candles) < bars:
        return None
    close = float(candles[bars - 1]["close"])
    return (entry - close) / entry


def close_event_for_trade(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    closes = [e for e in events if str(e.get("event") or "").upper() == "TRADE_CLOSED"]
    return closes[-1] if closes else None


def current_price_from_candles(candles: list[dict[str, Any]]) -> float | None:
    return float(candles[-1]["close"]) if candles else None


def closed_by(close_event: dict[str, Any] | None) -> str:
    if not close_event:
        return "OPEN"
    meta = close_event.get("metadata") if isinstance(close_event.get("metadata"), dict) else {}
    label = str(meta.get("canonicalExitType") or meta.get("exitType") or close_event.get("reason") or "unknown").upper()
    if "TRAIL" in label:
        return "trailing"
    if "TAKE" in label or "TP" in label:
        return "TP"
    if "STOP" in label and "TRAIL" not in label:
        return "SL"
    if "BREAKEVEN" in label or "BREAK_EVEN" in label:
        return "BE"
    if "MANUAL" in label:
        return "manual"
    return "unknown"


def classify_winner(pnl: float | None) -> bool | None:
    if pnl is None:
        return None
    return pnl > 0


def trade_quality_row(trade: dict[str, Any], events: list[dict[str, Any]], candles: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    entry_dt = parse_dt(trade.get("opened_at") or trade.get("timestamp"))
    close_event = close_event_for_trade(events)
    close_dt = parse_dt(close_event.get("timestamp")) if close_event else None
    end_dt = close_dt or now
    entry = to_float(trade.get("entry_price"))
    qty = to_float(trade.get("quantity"))
    leverage = to_float(trade.get("leverage")) or 1.0
    margin = to_float(trade.get("margin_estimated"))
    notional = to_float(trade.get("notional_estimated"))
    if entry is None or qty is None or entry_dt is None:
        return {"trade_id": trade.get("trade_id"), "symbol": trade.get("symbol"), "data_status": "DATA_INSUFFICIENT"}

    window = candles_between(candles, entry_dt, end_dt)
    exit_price = to_float(close_event.get("price")) if close_event else current_price_from_candles(window)
    gross_pnl = to_float((close_event.get("metadata") or {}).get("pnl")) if close_event and isinstance(close_event.get("metadata"), dict) else None
    if gross_pnl is None and exit_price is not None:
        gross_pnl = short_pnl(entry, exit_price, qty)
    fee_estimate = (notional or abs(entry * qty)) * FEE_RATE_PER_SIDE * (2 if close_event else 1)
    net_pnl = gross_pnl - fee_estimate if gross_pnl is not None else None
    roe = to_float(close_event.get("roe")) if close_event else roe_from_pnl(gross_pnl, margin)
    mae_mfe_full = compute_short_mae_mfe(entry, leverage, window)
    by_bars = {}
    for bars in (6, 12, 24):
        fixed = window[:bars]
        mm = compute_short_mae_mfe(entry, leverage, fixed)
        by_bars[f"mfe_{bars}_roe"] = mm.get("mfe_roe")
        by_bars[f"mae_{bars}_roe"] = mm.get("mae_roe")
    hit5 = hit_before_stop_short(entry, leverage, window[:24], 0.05, 0.05)
    hit8 = hit_before_stop_short(entry, leverage, window[:24], 0.08, 0.05)
    hit10 = hit_before_stop_short(entry, leverage, window[:24], 0.10, 0.08)
    max_possible_mfe_pnl = (mae_mfe_full.get("mfe_price_move") or 0) * (notional or abs(entry * qty))
    close_efficiency = safe_div(gross_pnl, max_possible_mfe_pnl)
    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "side": "SHORT",
        "status": "CLOSED" if close_event else "OPEN",
        "opened_at": entry_dt.isoformat(),
        "closed_at": close_dt.isoformat() if close_dt else None,
        "entry_price": entry,
        "exit_price": exit_price,
        "qty": qty,
        "leverage": leverage,
        "position_fraction": trade.get("position_fraction"),
        "bucket": infer_bucket(to_float(trade.get("position_fraction"))),
        "model_score": trade.get("turbo_score"),
        "votes": trade.get("votes"),
        "margin": margin,
        "notional": notional,
        "gross_pnl": gross_pnl,
        "fee_estimate": fee_estimate,
        "net_pnl_estimated": net_pnl,
        "roe": roe,
        "roe_pct": roe * 100 if roe is not None else None,
        "winner": classify_winner(net_pnl if net_pnl is not None else gross_pnl),
        "closed_by": closed_by(close_event),
        "time_in_trade_seconds": (end_dt - entry_dt).total_seconds(),
        "brackets_confirmed": trade.get("brackets_confirmed"),
        "sl": trade.get("sl_price"),
        "tp": trade.get("tp_price"),
        "entry_to_first_5m_return": first_return_short(entry, window, 1),
        "entry_to_first_15m_return": first_return_short(entry, window, 3),
        "entry_to_first_30m_return": first_return_short(entry, window, 6),
        "entry_to_first_60m_return": first_return_short(entry, window, 12),
        "close_efficiency": close_efficiency,
        "gave_back_mfe_pct": 1 - close_efficiency if close_efficiency is not None else None,
        "hit5_before_minus5": hit5["hit"],
        "hit8_before_minus5": hit8["hit"],
        "hit10_before_minus8": hit10["hit"],
        "time_to_hit5": hit5["time_to_hit"],
        "time_to_hit8": hit8["time_to_hit"],
        "time_to_minus5": hit5["time_to_stop"],
        "ambiguous_hit_stop": hit5["ambiguous_hit_stop"] or hit8["ambiguous_hit_stop"] or hit10["ambiguous_hit_stop"],
        "data_status": "OK" if window else "DATA_INSUFFICIENT",
        **mae_mfe_full,
        **by_bars,
    }


def aggregate_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for bucket, group in sorted(defaultdict(list, {b: [r for r in rows if r.get("bucket") == b] for b in {r.get("bucket") for r in rows}}).items()):
        if bucket is None:
            bucket = "unknown"
        pnl = [to_float(r.get("net_pnl_estimated")) for r in group if to_float(r.get("net_pnl_estimated")) is not None]
        winners = [r for r in group if r.get("winner") is True]
        out.append({
            "bucket": bucket,
            "trades_count": len(group),
            "closed_count": sum(1 for r in group if r.get("status") == "CLOSED"),
            "win_rate": len(winners) / len(group) if group else None,
            "pnl_total": sum(pnl) if pnl else 0.0,
            "pnl_avg": mean(pnl) if pnl else None,
            "mfe_roe_avg": mean([r["mfe_roe"] for r in group if r.get("mfe_roe") is not None]) if group else None,
            "mae_roe_avg": mean([r["mae_roe"] for r in group if r.get("mae_roe") is not None]) if group else None,
        })
    return out


def classify_symbol(row: dict[str, Any]) -> str:
    if row["closed_count"] < 2:
        return "SYMBOL_INSUFFICIENT_DATA"
    if row["pnl_total"] > 0 and (row.get("mfe_mae_ratio_avg") or 0) >= 1 and (row.get("p90_mae_roe") or 0) <= 0.40:
        return "SYMBOL_KEEP_ACTIVE"
    if row["pnl_total"] > 0:
        return "SYMBOL_WATCH"
    if (row.get("mfe_roe_avg") or 0) > 0 and (row.get("p90_mae_roe") or 0) > 0.40:
        return "SYMBOL_REDUCE_SIZE"
    if row["pnl_total"] < 0:
        return "SYMBOL_DISABLE_TEMPORARILY"
    return "SYMBOL_WATCH"


def aggregate_symbols(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for symbol in sorted({r["symbol"] for r in rows}):
        group = [r for r in rows if r["symbol"] == symbol]
        closed = [r for r in group if r.get("status") == "CLOSED"]
        pnl = [to_float(r.get("net_pnl_estimated")) for r in group if to_float(r.get("net_pnl_estimated")) is not None]
        roes = [to_float(r.get("roe")) for r in group if to_float(r.get("roe")) is not None]
        maes = [to_float(r.get("mae_roe")) for r in group if to_float(r.get("mae_roe")) is not None]
        mfes = [to_float(r.get("mfe_roe")) for r in group if to_float(r.get("mfe_roe")) is not None]
        ratios = [to_float(r.get("mfe_mae_ratio")) for r in group if to_float(r.get("mfe_mae_ratio")) is not None]
        row = {
            "symbol": symbol,
            "trades_count": len(group),
            "open_count": sum(1 for r in group if r.get("status") == "OPEN"),
            "closed_count": len(closed),
            "win_rate": sum(1 for r in group if r.get("winner") is True) / len(group) if group else None,
            "pnl_total": sum(pnl) if pnl else 0.0,
            "pnl_avg": mean(pnl) if pnl else None,
            "roe_avg": mean(roes) if roes else None,
            "roe_median": median(roes) if roes else None,
            "mfe_roe_avg": mean(mfes) if mfes else None,
            "mae_roe_avg": mean(maes) if maes else None,
            "p90_mae_roe": sorted(maes)[int(0.9 * (len(maes) - 1))] if maes else None,
            "mfe_mae_ratio_avg": mean(ratios) if ratios else None,
            "avg_time_in_trade": mean([r["time_in_trade_seconds"] for r in group if r.get("time_in_trade_seconds") is not None]) if group else None,
            "trailing_count": sum(1 for r in group if r.get("closed_by") == "trailing"),
            "model_score_avg": mean([r["model_score"] for r in group if to_float(r.get("model_score")) is not None]) if group else None,
            "bucket_distribution": dict(Counter(str(r.get("bucket")) for r in group)),
        }
        row["entry_quality_grade"] = classify_symbol(row)
        out.append(row)
    return out


def score_correlation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [(to_float(r.get("model_score")), to_float(r.get("net_pnl_estimated")), to_float(r.get("mfe_roe")), to_float(r.get("mae_roe"))) for r in rows]
    valid = [(s, p, m, a) for s, p, m, a in pairs if s is not None and p is not None and m is not None and a is not None]
    if len(valid) < 3:
        return [{"metric": "score_correlation", "sample_count": len(valid), "status": "INSUFFICIENT_SAMPLE"}]
    scores = [v[0] for v in valid]
    return [
        {"metric": "score_vs_net_pnl", "sample_count": len(valid), "correlation": pearson(scores, [v[1] for v in valid]), "status": "OK"},
        {"metric": "score_vs_mfe_roe", "sample_count": len(valid), "correlation": pearson(scores, [v[2] for v in valid]), "status": "OK"},
        {"metric": "score_vs_mae_roe", "sample_count": len(valid), "correlation": pearson(scores, [v[3] for v in valid]), "status": "OK"},
    ]


def classify_burst(machine_gun: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for win in machine_gun.get("multisymbol_windows", []):
        symbols = set(win.get("symbols") or [])
        start = parse_dt(win.get("start"))
        end = parse_dt(win.get("end"))
        group = []
        for r in rows:
            ts = parse_dt(r.get("opened_at"))
            if ts and start and end and start <= ts <= end and r.get("symbol") in symbols:
                group.append(r)
        pnl = sum(to_float(r.get("net_pnl_estimated")) or 0 for r in group)
        mae = sum(to_float(r.get("mae_roe")) or 0 for r in group)
        if len(group) < 2:
            status = "BURST_INSUFFICIENT_DATA"
        elif pnl > 0 and mae <= 2.0:
            status = "BURST_VALID_MARKET_EVENT"
        elif pnl < 0 and mae > 1.0:
            status = "BURST_CORRELATED_OVEREXPOSURE"
        else:
            status = "BURST_MIXED"
        out.append({"start": win.get("start"), "end": win.get("end"), "symbols": sorted(symbols), "trade_count": len(group), "pnl_total": pnl, "mae_roe_sum": mae, "classification": status})
    if not out:
        out.append({"classification": "BURST_INSUFFICIENT_DATA", "trade_count": 0})
    return out


def classify_global(rows: list[dict[str, Any]], symbol_rows: list[dict[str, Any]]) -> str:
    closed = [r for r in rows if r.get("status") == "CLOSED"]
    if len(closed) < 5:
        return "PHASE_O_SHORT_LIVE_TOO_EARLY"
    pnl_total = sum(to_float(r.get("net_pnl_estimated")) or 0 for r in rows)
    maes = [to_float(r.get("mae_roe")) for r in rows if to_float(r.get("mae_roe")) is not None]
    mfes = [to_float(r.get("mfe_roe")) for r in rows if to_float(r.get("mfe_roe")) is not None]
    bad_symbols = [r for r in symbol_rows if r.get("entry_quality_grade") in {"SYMBOL_DISABLE_TEMPORARILY", "SYMBOL_REDUCE_SIZE"}]
    if pnl_total > 0 and len(rows) >= 10 and mean(mfes or [0]) > mean(maes or [0]) and not bad_symbols:
        return "PHASE_O_SHORT_LIVE_STRONG_INITIAL"
    if pnl_total > 0 and mean(mfes or [0]) >= mean(maes or [0]) * 0.8:
        return "PHASE_O_SHORT_LIVE_PROMISING"
    if pnl_total < 0:
        return "PHASE_O_SHORT_LIVE_WEAK"
    return "PHASE_O_SHORT_LIVE_MIXED"


def build_quality_audit(args: argparse.Namespace) -> dict[str, Any]:
    start = parse_dt(args.from_ts) or datetime(2026, 6, 1, tzinfo=timezone.utc)
    end = parse_dt(args.to) if args.to != "now" else datetime.now(timezone.utc)
    end = end or datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    raw_trades = read_log_rows("trades", start, end, args.max_log_bytes_per_file)
    raw_events = read_log_rows("events", start, end, args.max_log_bytes_per_file)
    symbol_set = set(symbols)
    raw_trades = [r for r in raw_trades if str(r.get("symbol") or "").upper() in symbol_set]
    raw_events = [r for r in raw_events if str(r.get("symbol") or "").upper() in symbol_set]
    trades = collect_phase_o_trades(raw_trades, args.include_open, args.include_closed)
    ev_by_trade = events_by_trade(raw_events)

    candles_by_symbol: dict[str, list[dict[str, Any]]] = {}
    quality_rows = []
    for trade in trades:
        trade_id = str(trade.get("trade_id") or "")
        entry_dt = parse_dt(trade.get("opened_at") or trade.get("timestamp"))
        if entry_dt is None:
            continue
        events = ev_by_trade.get(trade_id, [])
        close_event = close_event_for_trade(events)
        if close_event:
            trade = dict(trade)
            trade["status"] = "CLOSED"
            trade["closed_at"] = close_event.get("timestamp")
        symbol = str(trade.get("symbol") or "").upper()
        if symbol not in candles_by_symbol:
            candles_by_symbol[symbol] = load_candles(symbol, start, end)
        quality_rows.append(trade_quality_row(trade, events, candles_by_symbol[symbol], end))

    symbol_rows = aggregate_symbols(quality_rows)
    bucket_rows = aggregate_bucket(quality_rows)
    corr_rows = score_correlation(quality_rows)
    mg = detect_machine_gun(quality_rows, args.machine_gun_window_seconds)
    burst_rows = classify_burst(mg, quality_rows)
    global_status = classify_global(quality_rows, symbol_rows)
    pnl_values = [to_float(r.get("net_pnl_estimated")) for r in quality_rows if to_float(r.get("net_pnl_estimated")) is not None]
    closed = [r for r in quality_rows if r.get("status") == "CLOSED"]
    summary = {
        "global_classification": global_status,
        "trades_count": len(quality_rows),
        "open_count": sum(1 for r in quality_rows if r.get("status") == "OPEN"),
        "closed_count": len(closed),
        "pnl_total_estimated": sum(pnl_values) if pnl_values else 0.0,
        "pnl_avg_estimated": mean(pnl_values) if pnl_values else None,
        "win_rate": sum(1 for r in quality_rows if r.get("winner") is True) / len(quality_rows) if quality_rows else None,
        "machine_gun": mg.get("classification"),
        "burst_classifications": dict(Counter(r.get("classification") for r in burst_rows)),
        "symbol_recommendations": dict(Counter(r.get("entry_quality_grade") for r in symbol_rows)),
    }
    return {
        "schema_version": "phase_o_short_live_quality_v1",
        "created_at": now_iso(),
        "mode": "READ_ONLY",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "summary": summary,
        "trades": quality_rows,
        "symbols": symbol_rows,
        "buckets": bucket_rows,
        "score_correlation": corr_rows,
        "bursts": burst_rows,
        "recommendations": build_recommendations(summary, symbol_rows, bucket_rows, corr_rows, burst_rows),
        "confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_push": True,
            "no_commit": True,
        },
    }


def build_recommendations(summary: dict[str, Any], symbols: list[dict[str, Any]], buckets: list[dict[str, Any]], corr: list[dict[str, Any]], bursts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recs = []
    if summary["global_classification"] in {"PHASE_O_SHORT_LIVE_STRONG_INITIAL", "PHASE_O_SHORT_LIVE_PROMISING"}:
        recs.append({"area": "global", "symbol": "", "recommendation": "KEEP_RUNNING_WATCH_SAMPLE", "reason": "Positive initial quality, but sample is still early."})
    elif summary["global_classification"] == "PHASE_O_SHORT_LIVE_TOO_EARLY":
        recs.append({"area": "global", "symbol": "", "recommendation": "WAIT_MORE_SAMPLE", "reason": "Too few closed trades for a strong conclusion."})
    else:
        recs.append({"area": "global", "symbol": "", "recommendation": "WATCH_CLOSELY_BEFORE_SCALING", "reason": "Mixed/weak early quality."})
    for row in symbols:
        if row.get("entry_quality_grade") == "SYMBOL_DISABLE_TEMPORARILY":
            recs.append({"area": "symbol", "symbol": row["symbol"], "recommendation": "DISABLE_TEMPORARILY_REVIEW", "reason": "Negative quality metrics."})
        elif row.get("entry_quality_grade") == "SYMBOL_REDUCE_SIZE":
            recs.append({"area": "symbol", "symbol": row["symbol"], "recommendation": "REDUCE_SIZE_REVIEW", "reason": "MAE/p90 risk high versus output."})
        elif row.get("entry_quality_grade") == "SYMBOL_INSUFFICIENT_DATA":
            recs.append({"area": "symbol", "symbol": row["symbol"], "recommendation": "INSUFFICIENT_DATA_KEEP_WATCH", "reason": "Need more closed trades."})
    for row in bursts:
        if row.get("classification") == "BURST_CORRELATED_OVEREXPOSURE":
            recs.append({"area": "burst", "symbol": "", "recommendation": "ADD_PER_SCAN_CAP_OR_REDUCE_BURST_SIZE", "reason": "Burst showed correlated overexposure."})
        elif row.get("classification") == "BURST_VALID_MARKET_EVENT":
            recs.append({"area": "burst", "symbol": "", "recommendation": "NO_BURST_LIMIT_REQUIRED_YET", "reason": "Burst was positive and MAE controlled."})
    if any(r.get("status") == "INSUFFICIENT_SAMPLE" for r in corr):
        recs.append({"area": "score", "symbol": "", "recommendation": "DO_NOT_TUNE_SCORE_YET", "reason": "Score correlation sample is insufficient."})
    return recs


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Phase O SHORT Live Quality Audit",
        "",
        "## Safety",
        "- read-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2 restart",
        "- no orders",
        "",
        "## Executive Summary",
        f"- Global classification: `{s['global_classification']}`",
        f"- Trades: `{s['trades_count']}` open=`{s['open_count']}` closed=`{s['closed_count']}`",
        f"- Estimated net PnL: `{s['pnl_total_estimated']}`",
        f"- Win rate: `{s['win_rate']}`",
        f"- Machine gun: `{s['machine_gun']}`",
        "",
        "## Trades",
        "| symbol | status | pnl | roe | MFE ROE | MAE ROE | bucket | score | close reason | entry quality |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in payload["trades"]:
        lines.append(f"| {row.get('symbol')} | {row.get('status')} | {row.get('net_pnl_estimated')} | {row.get('roe')} | {row.get('mfe_roe')} | {row.get('mae_roe')} | {row.get('bucket')} | {row.get('model_score')} | {row.get('closed_by')} | {row.get('data_status')} |")
    lines += ["", "## Symbols", "| symbol | recommendation | trades | closed | pnl | win_rate | MFE/MAE | p90 MAE ROE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["symbols"]:
        lines.append(f"| {row.get('symbol')} | {row.get('entry_quality_grade')} | {row.get('trades_count')} | {row.get('closed_count')} | {row.get('pnl_total')} | {row.get('win_rate')} | {row.get('mfe_mae_ratio_avg')} | {row.get('p90_mae_roe')} |")
    lines += ["", "## Bucket Analysis"]
    for row in payload["buckets"]:
        lines.append(f"- {row.get('bucket')}: trades={row.get('trades_count')} pnl={row.get('pnl_total')} win_rate={row.get('win_rate')} mfe_roe_avg={row.get('mfe_roe_avg')} mae_roe_avg={row.get('mae_roe_avg')}")
    lines += ["", "## Score Correlation"]
    for row in payload["score_correlation"]:
        lines.append(f"- {row.get('metric')}: sample={row.get('sample_count')} corr={row.get('correlation')} status={row.get('status')}")
    lines += ["", "## Burst Analysis"]
    for row in payload["bursts"]:
        lines.append(f"- {row.get('classification')}: symbols={row.get('symbols')} pnl={row.get('pnl_total')} mae_sum={row.get('mae_roe_sum')}")
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
    base = out_dir / f"aegis_phase_o_short_live_quality_{ts}"
    paths = {
        "md": str(base.with_suffix(".md")),
        "json": str(base.with_suffix(".json")),
        "trades_csv": str(out_dir / f"aegis_phase_o_short_live_quality_trades_{ts}.csv"),
        "symbols_csv": str(out_dir / f"aegis_phase_o_short_live_quality_symbols_{ts}.csv"),
        "buckets_csv": str(out_dir / f"aegis_phase_o_short_live_quality_buckets_{ts}.csv"),
        "score_corr_csv": str(out_dir / f"aegis_phase_o_short_live_quality_score_corr_{ts}.csv"),
        "bursts_csv": str(out_dir / f"aegis_phase_o_short_live_quality_bursts_{ts}.csv"),
        "recommendations_csv": str(out_dir / f"aegis_phase_o_short_live_quality_recommendations_{ts}.csv"),
    }
    payload["reports"] = paths
    Path(paths["json"]).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(paths["md"]), payload)
    write_csv(Path(paths["trades_csv"]), payload["trades"])
    write_csv(Path(paths["symbols_csv"]), payload["symbols"])
    write_csv(Path(paths["buckets_csv"]), payload["buckets"])
    write_csv(Path(paths["score_corr_csv"]), payload["score_correlation"])
    write_csv(Path(paths["bursts_csv"]), payload["bursts"])
    write_csv(Path(paths["recommendations_csv"]), payload["recommendations"])
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_ts", default="2026-06-01T00:00:00Z")
    p.add_argument("--to", default="now")
    p.add_argument("--out-dir", default="/home/jasan/Develop")
    p.add_argument("--symbols", default=",".join(ALL_SYMBOLS))
    p.add_argument("--include-open", action="store_true")
    p.add_argument("--include-closed", action="store_true")
    p.add_argument("--include-mae-mfe", action="store_true")
    p.add_argument("--include-score-correlation", action="store_true")
    p.add_argument("--include-symbol-ranking", action="store_true")
    p.add_argument("--include-bucket-analysis", action="store_true")
    p.add_argument("--include-burst-analysis", action="store_true")
    p.add_argument("--machine-gun-window-seconds", type=int, default=300)
    p.add_argument("--max-log-bytes-per-file", type=int, default=250_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_quality_audit(args)
    paths = write_reports(payload, Path(args.out_dir))
    print(json.dumps({"summary": payload["summary"], "reports": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
