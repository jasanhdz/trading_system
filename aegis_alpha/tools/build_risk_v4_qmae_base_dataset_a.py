#!/usr/bin/env python3
"""Research-only base dataset for Risk Model V4 / QMAE."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.turbo.short_quality_v4_labels import (
    ShortV4Config,
    build_operable_short_quality_v4_labels,
)


DB_PATH = REPO / "data" / "binance_candles.db"
DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
SAMPLE_FIELDS = [
    "timestamp",
    "symbol",
    "timeframe",
    "horizon",
    "close",
    "future_mfe_roe_proxy",
    "future_mae_roe_proxy",
    "time_to_mfe",
    "time_to_mae",
    "mfe_before_mae",
    "mfe_mae_ratio",
    "net_quality_after_costs",
    "clean_entry_v4",
    "bad_entry_v4",
    "premium_allowed_v4",
    "management_dependent_v4",
    "no_trade_v4",
    "tail_risk_v4",
    "early_mae_v4",
    "squeeze_risk_proxy_v4",
    "qmae_target",
    "q95_mae_target",
    "volatility_features",
    "trend_features",
    "wick_reclaim_proxies",
    "btc_eth_context",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def db_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "")
    return s[:-4] + "/USDT" if s.endswith("USDT") else s


def load_ohlcv_from_sqlite(db_path: Path, symbol: str, timeframe: str, lookback_days: int, limit: int = 5000) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    sql = """
        select timestamp, open, high, low, close, volume, buy_volume
        from ohlcv_data
        where symbol = ? and timeframe = ?
        order by timestamp desc
        limit ?
    """
    with sqlite3.connect(db_path, timeout=5) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, (db_symbol(symbol), timeframe, limit)).fetchall()]
    return list(reversed(rows))


def pct_change(values: list[float]) -> float:
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return (values[-1] - values[0]) / values[0]


def build_samples(candles: list[dict[str, Any]], symbol: str, timeframe: str, horizons: list[int], max_rows_per_combo: int = 1200) -> list[dict[str, Any]]:
    if len(candles) < max(horizons, default=1) + 20:
        return []
    high = [float(c["high"]) for c in candles]
    low = [float(c["low"]) for c in candles]
    close = [float(c["close"]) for c in candles]
    timestamps = [str(c["timestamp"]) for c in candles]
    rows: list[dict[str, Any]] = []
    step = max(1, (len(close) - max(horizons)) // max_rows_per_combo)
    for horizon in horizons:
        labels = build_operable_short_quality_v4_labels(
            symbol=symbol,
            high=high,
            low=low,
            close=close,
            steps=range(20, len(close) - horizon, step),
            horizon=horizon,
            config=ShortV4Config(horizon=horizon),
        )
        mae_values = [float(r.get("mae_roe_proxy") or 0.0) for r in labels]
        q95 = float(np.quantile(mae_values, 0.95)) if mae_values else 0.0
        for r in labels:
            idx = int(r["entry_index"])
            prior = close[max(0, idx - 12):idx + 1]
            vol_window = close[max(0, idx - 20):idx + 1]
            returns = np.diff(vol_window) / np.asarray(vol_window[:-1]) if len(vol_window) > 2 else np.array([0.0])
            mae = float(r.get("mae_roe_proxy") or 0.0)
            early_mae = int((r.get("time_to_mae") or horizon + 1) <= 3 and mae >= 0.045)
            bad = int(r.get("short_bad_entry_v4") or 0)
            tail = int(mae >= 0.10 or early_mae or (bad and mae >= 0.08))
            trend = pct_change(prior)
            wick_proxy = max(0.0, (float(candles[idx]["close"]) - float(candles[idx]["low"])) / max(float(candles[idx]["close"]), 1e-9))
            squeeze = int(trend < -0.015 and (early_mae or bool(r.get("mae_before_mfe"))) and wick_proxy > 0.002)
            rows.append({
                "timestamp": timestamps[idx],
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon,
                "close": close[idx],
                "future_mfe_roe_proxy": r.get("mfe_roe_proxy"),
                "future_mae_roe_proxy": mae,
                "time_to_mfe": r.get("time_to_mfe"),
                "time_to_mae": r.get("time_to_mae"),
                "mfe_before_mae": r.get("mfe_before_mae"),
                "mfe_mae_ratio": r.get("mfe_mae_ratio"),
                "net_quality_after_costs": r.get("net_quality_after_costs"),
                "clean_entry_v4": r.get("short_clean_entry_v4"),
                "bad_entry_v4": bad,
                "premium_allowed_v4": r.get("short_premium_allowed_v4"),
                "management_dependent_v4": r.get("short_management_dependent_v4"),
                "no_trade_v4": r.get("short_no_trade_v4"),
                "tail_risk_v4": tail,
                "early_mae_v4": early_mae,
                "squeeze_risk_proxy_v4": squeeze,
                "qmae_target": mae,
                "q95_mae_target": q95,
                "volatility_features": json.dumps({"ret_std_20": float(np.std(returns))}),
                "trend_features": json.dumps({"ret_12": trend}),
                "wick_reclaim_proxies": json.dumps({"lower_wick_close_pct": wick_proxy}),
                "btc_eth_context": "{}",
            })
    return rows


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(1 for r in rows if r.get(field)) / len(rows) if rows else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(values, q))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["symbol"], r["timeframe"], int(r["horizon"])), []).append(r)
    out: list[dict[str, Any]] = []
    for (symbol, timeframe, horizon), data in sorted(groups.items()):
        maes = [float(r["qmae_target"]) for r in data]
        out.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": horizon,
            "samples": len(data),
            "clean_rate": rate(data, "clean_entry_v4"),
            "bad_rate": rate(data, "bad_entry_v4"),
            "tail_risk_rate": rate(data, "tail_risk_v4"),
            "early_mae_rate": rate(data, "early_mae_v4"),
            "squeeze_proxy_rate": rate(data, "squeeze_risk_proxy_v4"),
            "q50_mae": median(maes) if maes else 0.0,
            "q90_mae": quantile(maes, 0.90),
            "q95_mae": quantile(maes, 0.95),
        })
    return out


def label_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": f, "rate": rate(rows, f), "count": sum(1 for r in rows if r.get(f)), "total": len(rows)} for f in [
        "clean_entry_v4",
        "bad_entry_v4",
        "premium_allowed_v4",
        "management_dependent_v4",
        "no_trade_v4",
        "tail_risk_v4",
        "early_mae_v4",
        "squeeze_risk_proxy_v4",
    ]]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for symbol in symbols:
        for timeframe in timeframes:
            candles = load_ohlcv_from_sqlite(Path(args.db_path), symbol, timeframe, args.lookback_days)
            if not candles:
                missing.append(f"{symbol}:{timeframe}")
                continue
            rows.extend(build_samples(candles, symbol, timeframe, horizons))
    status = "OK" if rows else "INSUFFICIENT_OHLCV_DATA"
    summary = summarize_rows(rows)
    distribution = label_distribution(rows)
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    ready = len(rows) >= 5000 and len(summary) >= 6
    result = {
        "schema_version": "risk_v4_qmae_base_dataset_a_v1",
        "status": status,
        "generated_at": timestamp,
        "sample_count": len(rows),
        "groups": len(summary),
        "missing_symbol_timeframes": missing[:50],
        "training_readiness": "ready_for_next_phase_review" if ready else "not_ready_or_needs_review",
        "recommendation": "Train TRRM/QMAE next only after reviewing label distribution and leakage controls." if ready else "Gather/verify more OHLCV coverage before training.",
        "no_train": True,
    }
    json_path = out_dir / f"aegis_risk_v4_qmae_base_dataset_a_{timestamp}.json"
    md_path = out_dir / f"aegis_risk_v4_qmae_base_dataset_a_{timestamp}.md"
    samples_path = out_dir / f"aegis_risk_v4_qmae_base_dataset_a_samples_{timestamp}.csv"
    summary_path = out_dir / f"aegis_risk_v4_qmae_base_dataset_a_summary_{timestamp}.csv"
    dist_path = out_dir / f"aegis_risk_v4_qmae_base_dataset_a_label_distribution_{timestamp}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Aegis Risk V4 / QMAE Base Dataset A",
            "",
            f"- status: {status}",
            f"- samples: {len(rows)}",
            f"- groups: {len(summary)}",
            f"- training_readiness: {result['training_readiness']}",
            "- no_train: true",
            "",
            "Initial threshold candidates: tail_risk_v4 at MAE >= 0.10 ROE, early_mae_v4 in first 3 candles at MAE >= 0.045 ROE.",
        ]) + "\n",
        encoding="utf-8",
    )
    write_csv(samples_path, rows, SAMPLE_FIELDS)
    write_csv(summary_path, summary)
    write_csv(dist_path, distribution)
    result["outputs"] = {"json": str(json_path), "md": str(md_path), "samples_csv": str(samples_path), "summary_csv": str(summary_path), "label_distribution_csv": str(dist_path)}
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build research-only Risk V4 / QMAE base dataset.")
    p.add_argument("--symbols", required=True)
    p.add_argument("--timeframes", required=True)
    p.add_argument("--horizons", required=True)
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--db-path", default=str(DB_PATH))
    p.add_argument("--research-only", action="store_true")
    p.add_argument("--no-train", action="store_true")
    return p


def main() -> int:
    result = run_builder(build_parser().parse_args())
    print(json.dumps({k: result[k] for k in ("status", "sample_count", "groups", "training_readiness", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
