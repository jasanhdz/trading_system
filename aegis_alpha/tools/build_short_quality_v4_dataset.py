#!/usr/bin/env python3
"""Research-only builder for operable_short_quality_v4 labels and summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_live_entries import json_safe, parse_dt, to_float, utc_stamp  # noqa: E402
from aegis_alpha.tools.audit_phase_o_short_live_quality import build_quality_audit  # noqa: E402
from aegis_alpha.turbo.operable_feature_builder_v3 import apply_feature_set  # noqa: E402
from aegis_alpha.turbo.short_quality_v4_labels import (  # noqa: E402
    SHORT_V4_SCHEMA_VERSION,
    ShortV4Config,
    build_operable_short_quality_v4_labels,
    summarize_short_v4_labels,
)


REPO = Path("/home/jasan/Develop/trading_system")
DB_PATH = REPO / "data" / "binance_candles.db"
PRIMARY_SYMBOLS = ("ADAUSDT", "SUIUSDT", "SOLUSDT", "AVAXUSDT")
SECONDARY_SYMBOLS = ("BTCUSDT", "BNBUSDT", "XRPUSDT")


@dataclass
class MarketFrame:
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    features: np.ndarray | None = None


def symbol_db_name(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT" if symbol.endswith("USDT") else symbol


def dt_to_db(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_csv(value: str) -> list[str]:
    return [v.strip().upper() for v in value.split(",") if v.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def safe_mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
    return mean(vals) if vals else None


def percentile(values: list[float], value: float | None) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if value is None or not vals:
        return None
    return sum(1 for v in vals if v <= value) / len(vals)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or (list(rows[0].keys()) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def assert_research_only_output(path: Path) -> None:
    text = str(path.resolve())
    forbidden = ("/active/", "active_manifest.json", "phase_o_short_manifest.json")
    if any(token in text for token in forbidden):
        raise ValueError(f"refusing active/live output path: {path}")


def load_market(symbol: str, lookback_days: int, db_path: Path = DB_PATH, end: datetime | None = None) -> MarketFrame:
    end = end or now_utc()
    start = end - timedelta(days=lookback_days)
    con = sqlite3.connect(db_path, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select timestamp, open, high, low, close, volume
            from ohlcv_data
            where symbol = ? and timeframe = '5m' and timestamp >= ? and timestamp <= ?
            order by timestamp
            """,
            (symbol_db_name(symbol), dt_to_db(start), dt_to_db(end)),
        ).fetchall()
    finally:
        con.close()
    timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for row in rows:
        timestamps.append(str(row["timestamp"]).replace(" ", "T"))
        opens.append(float(row["open"]))
        highs.append(float(row["high"]))
        lows.append(float(row["low"]))
        closes.append(float(row["close"]))
        volumes.append(float(row["volume"]))
    return MarketFrame(
        symbol=symbol,
        timestamps=np.asarray(timestamps, dtype=str),
        open=np.asarray(opens, dtype=np.float64),
        high=np.asarray(highs, dtype=np.float64),
        low=np.asarray(lows, dtype=np.float64),
        close=np.asarray(closes, dtype=np.float64),
        volume=np.asarray(volumes, dtype=np.float64),
    )


def build_feature_diagnostics(market: MarketFrame, steps: np.ndarray, context: dict[str, MarketFrame]) -> dict[str, Any]:
    if len(steps) == 0:
        return {"feature_count": 0, "feature_schema_hash": None, "feature_set": "combined_v3"}
    dataset = {
        "X": np.zeros((len(steps), 1), dtype=np.float32),
        "feature_names": np.asarray(["bias"], dtype=str),
        "step": steps.astype(np.int64),
    }
    try:
        built = apply_feature_set(dataset, market, "combined_v3", context_markets=context)
        return {
            "feature_count": int(len(built.get("feature_names", []))),
            "feature_schema_hash": built.get("feature_schema_hash"),
            "feature_set": built.get("feature_set"),
            "diagnostics": built.get("feature_diagnostics"),
        }
    except Exception as exc:
        return {"feature_count": 0, "feature_schema_hash": None, "feature_set": "combined_v3", "feature_error": repr(exc)}


def classify_v4_label(summary: dict[str, Any]) -> str:
    samples = int(summary.get("sample_count") or 0)
    if samples < 500:
        return "INSUFFICIENT_DATA"
    clean_rate = float(summary.get("clean_rate") or 0.0)
    premium_rate = float(summary.get("premium_allowed_rate") or 0.0)
    baseline_quality = summary.get("baseline_net_quality")
    clean_quality = summary.get("clean_net_quality")
    baseline_mae = summary.get("baseline_mae_roe")
    clean_mae = summary.get("clean_mae_roe")
    baseline_ratio = summary.get("baseline_mfe_mae_ratio")
    clean_ratio = summary.get("clean_mfe_mae_ratio")
    if clean_rate < 0.005:
        return "V4_LABEL_TOO_STRICT"
    if clean_rate > 0.15 or premium_rate > 0.05:
        return "V4_LABEL_TOO_LOOSE"
    improves_quality = clean_quality is not None and baseline_quality is not None and clean_quality > baseline_quality
    improves_mae = clean_mae is not None and baseline_mae is not None and clean_mae <= baseline_mae
    improves_ratio = clean_ratio is not None and baseline_ratio is not None and clean_ratio > baseline_ratio
    if improves_quality and improves_mae and improves_ratio and 0.01 <= clean_rate <= 0.10 and premium_rate <= 0.03:
        return "V4_LABEL_PROMISING"
    if improves_quality or improves_mae or improves_ratio:
        return "V4_LABEL_MIXED"
    return "V4_LABEL_BAD"


def label_quality_rows(rows: list[dict[str, Any]], symbol: str, horizon: int) -> dict[str, Any]:
    summary = summarize_short_v4_labels(rows)
    baseline_maes = [float(r["mae_roe_proxy"]) for r in rows if r.get("mae_roe_proxy") is not None]
    clean_rows = [r for r in rows if r.get("short_clean_entry_v4")]
    clean_maes = [float(r["mae_roe_proxy"]) for r in clean_rows if r.get("mae_roe_proxy") is not None]
    summary.update({
        "symbol": symbol,
        "horizon": horizon,
        "baseline_p90_mae": float(np.quantile(baseline_maes, 0.90)) if baseline_maes else None,
        "clean_p90_mae": float(np.quantile(clean_maes, 0.90)) if clean_maes else None,
    })
    summary["classification"] = classify_v4_label(summary)
    return summary


def random_baseline_for_rows(rows: list[dict[str, Any]], n: int = 1000) -> dict[str, Any]:
    rng = random.Random(44017)
    if not rows:
        return {"random_count": 0}
    sample = rng.sample(rows, min(n, len(rows)))
    clean = [r for r in rows if r.get("short_clean_entry_v4")]
    random_quality = [float(r.get("net_quality_after_costs") or 0.0) for r in sample]
    random_mfe = [float(r.get("mfe_roe_proxy") or 0.0) for r in sample]
    random_mae = [float(r.get("mae_roe_proxy") or 0.0) for r in sample]
    clean_quality = safe_mean([r.get("net_quality_after_costs") for r in clean])
    clean_mfe = safe_mean([r.get("mfe_roe_proxy") for r in clean])
    clean_mae = safe_mean([r.get("mae_roe_proxy") for r in clean])
    return {
        "random_count": len(sample),
        "random_quality_avg": safe_mean(random_quality),
        "random_mfe_avg": safe_mean(random_mfe),
        "random_mae_avg": safe_mean(random_mae),
        "clean_quality_percentile_vs_random": percentile(random_quality, clean_quality),
        "clean_mfe_percentile_vs_random": percentile(random_mfe, clean_mfe),
        "clean_mae_percentile_vs_random": percentile(random_mae, clean_mae),
        "v4_beats_random": (percentile(random_quality, clean_quality) or 0.0) >= 0.60,
    }


def load_live_quality(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_live_trade_overlap:
        return []
    quality_args = argparse.Namespace(
        from_ts="2026-06-01T00:00:00Z",
        to="now",
        symbols=",".join(sorted(set(parse_csv(args.symbols) + list(SECONDARY_SYMBOLS) + ["ETHUSDT", "DOGEUSDT"]))),
        include_open=True,
        include_closed=True,
        machine_gun_window_seconds=300,
        max_log_bytes_per_file=250_000_000,
    )
    try:
        return list(build_quality_audit(quality_args).get("trades", []))
    except Exception:
        return []


def nearest_label_for_trade(trade: dict[str, Any], rows: list[dict[str, Any]], timestamps: np.ndarray) -> dict[str, Any] | None:
    opened = parse_dt(trade.get("opened_at"))
    if opened is None or len(rows) == 0:
        return None
    opened_key = opened.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="minutes")
    best_idx = None
    best_delta = 10**12
    for row in rows:
        idx = int(row.get("entry_index") or 0)
        if idx >= len(timestamps):
            continue
        ts = parse_dt(str(timestamps[idx]) + "Z")
        if ts is None:
            continue
        delta = abs((opened - ts).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_idx = row
        if str(timestamps[idx]).startswith(opened_key):
            return row
    return best_idx if best_delta <= 900 else None


def live_overlap_rows(live_trades: list[dict[str, Any]], labels_by_key: dict[tuple[str, int], list[dict[str, Any]]], markets: dict[str, MarketFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in live_trades:
        symbol = str(trade.get("symbol") or "")
        for (row_symbol, horizon), rows in labels_by_key.items():
            if row_symbol != symbol:
                continue
            label = nearest_label_for_trade(trade, rows, markets[symbol].timestamps)
            out.append({
                "trade_id": trade.get("trade_id"),
                "symbol": symbol,
                "horizon": horizon,
                "opened_at": trade.get("opened_at"),
                "net_pnl_estimated": trade.get("net_pnl_estimated"),
                "winner": trade.get("winner"),
                "big_loss": bool(to_float(trade.get("net_pnl_estimated")) is not None and (to_float(trade.get("net_pnl_estimated")) or 0.0) < -1.0),
                "live_bucket": trade.get("bucket"),
                "matched_label": label is not None,
                "short_clean_entry_v4": label.get("short_clean_entry_v4") if label else None,
                "short_bad_entry_v4": label.get("short_bad_entry_v4") if label else None,
                "short_premium_allowed_v4": label.get("short_premium_allowed_v4") if label else None,
                "short_management_dependent_v4": label.get("short_management_dependent_v4") if label else None,
                "v4_would_block": bool(label and not label.get("short_clean_entry_v4")),
                "premium_loser_blocked": bool(label and trade.get("bucket") == "premium" and (to_float(trade.get("net_pnl_estimated")) or 0.0) < 0 and not label.get("short_premium_allowed_v4")),
            })
    return out


def aggregate_live_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [r for r in rows if r.get("matched_label")]
    big_losses = [r for r in matched if r.get("big_loss")]
    winners = [r for r in matched if r.get("winner") is True]
    premium_losers = [r for r in matched if r.get("live_bucket") == "premium" and (to_float(r.get("net_pnl_estimated")) or 0.0) < 0]
    return {
        "matched_live_rows": len(matched),
        "live_winners_conserved_clean": sum(1 for r in winners if r.get("short_clean_entry_v4")),
        "live_winners_total": len(winners),
        "live_big_losses_blocked": sum(1 for r in big_losses if r.get("v4_would_block") or r.get("short_bad_entry_v4")),
        "live_big_losses_total": len(big_losses),
        "premium_losers_blocked": sum(1 for r in premium_losers if r.get("premium_loser_blocked")),
        "premium_losers_total": len(premium_losers),
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    assert_research_only_output(out_dir)
    symbols = parse_csv(args.symbols)
    horizons = parse_int_csv(args.horizons)
    end = now_utc()
    markets = {s: load_market(s, args.lookback_days, end=end) for s in sorted(set(symbols) | {"BTCUSDT", "ETHUSDT"})}
    context = {k: v for k, v in markets.items() if k in {"BTCUSDT", "ETHUSDT"}}
    live_trades = load_live_quality(args)

    all_summary: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    random_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for symbol in symbols:
        market = markets[symbol]
        if len(market.close) < 100:
            continue
        for horizon in horizons:
            max_step = max(0, len(market.close) - horizon - 1)
            steps = np.arange(64, max_step, dtype=np.int64)
            cfg = ShortV4Config(horizon=horizon)
            rows = build_operable_short_quality_v4_labels(
                symbol=symbol,
                high=market.high,
                low=market.low,
                close=market.close,
                steps=steps,
                horizon=horizon,
                config=cfg,
            )
            labels_by_key[(symbol, horizon)] = rows
            summary = label_quality_rows(rows, symbol, horizon)
            features = build_feature_diagnostics(market, steps[: min(len(steps), 5000)] if args.fast else steps, context)
            summary.update({
                "feature_count": features.get("feature_count"),
                "feature_schema_hash": features.get("feature_schema_hash"),
                "feature_set": features.get("feature_set"),
            })
            random_summary = random_baseline_for_rows(rows, n=1000 if args.fast else 5000)
            random_summary.update({"symbol": symbol, "horizon": horizon})
            summary.update({
                "random_quality_avg": random_summary.get("random_quality_avg"),
                "clean_quality_percentile_vs_random": random_summary.get("clean_quality_percentile_vs_random"),
                "v4_beats_random": random_summary.get("v4_beats_random"),
            })
            all_summary.append(summary)
            by_symbol.setdefault(symbol, []).append(summary)
            by_horizon.setdefault(horizon, []).append(summary)
            random_rows.append(random_summary)
            feature_rows.append({"symbol": symbol, "horizon": horizon, **features})

    symbol_rows = [aggregate_group(symbol, rows, "symbol") for symbol, rows in by_symbol.items()]
    horizon_rows = [aggregate_group(str(horizon), rows, "horizon") for horizon, rows in by_horizon.items()]
    overlap = live_overlap_rows(live_trades, labels_by_key, markets) if args.include_live_trade_overlap else []
    overlap_summary = aggregate_live_overlap(overlap)
    recommendations = build_recommendations(all_summary, overlap_summary)
    return {
        "schema_version": "short_v4_dataset_labels_a_v1",
        "label_schema_version": SHORT_V4_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "RESEARCH_ONLY",
        "symbols": symbols,
        "horizons": horizons,
        "lookback_days": args.lookback_days,
        "summary": all_summary,
        "by_symbol": symbol_rows,
        "by_horizon": horizon_rows,
        "feature_diagnostics": feature_rows,
        "live_overlap": overlap,
        "live_overlap_summary": overlap_summary,
        "random_baseline": random_rows,
        "recommendations": recommendations,
        "confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2": True,
            "no_orders": True,
            "no_env": True,
            "no_push": True,
            "no_commit": True,
            "no_matrix_saved": bool(args.no_save_matrix),
        },
    }


def aggregate_group(name: str, rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    classifications = {}
    for row in rows:
        classifications[row.get("classification")] = classifications.get(row.get("classification"), 0) + 1
    return {
        key: name,
        "configs": len(rows),
        "avg_clean_rate": safe_mean([r.get("clean_rate") for r in rows]),
        "avg_bad_rate": safe_mean([r.get("bad_rate") for r in rows]),
        "avg_premium_allowed_rate": safe_mean([r.get("premium_allowed_rate") for r in rows]),
        "avg_clean_net_quality": safe_mean([r.get("clean_net_quality") for r in rows]),
        "avg_baseline_net_quality": safe_mean([r.get("baseline_net_quality") for r in rows]),
        "avg_clean_p90_mae": safe_mean([r.get("clean_p90_mae") for r in rows]),
        "avg_baseline_p90_mae": safe_mean([r.get("baseline_p90_mae") for r in rows]),
        "classifications": classifications,
    }


def build_recommendations(summary_rows: list[dict[str, Any]], overlap: dict[str, Any]) -> list[dict[str, Any]]:
    promising = [r for r in summary_rows if r.get("classification") == "V4_LABEL_PROMISING"]
    mixed = [r for r in summary_rows if r.get("classification") == "V4_LABEL_MIXED"]
    too_strict = [r for r in summary_rows if r.get("classification") == "V4_LABEL_TOO_STRICT"]
    recs = [
        {
            "area": "global",
            "recommendation": "SHORT_V4_B_TRAINING_READY" if promising else "ADJUST_LABEL_BEFORE_TRAINING",
            "reason": f"promising={len(promising)} mixed={len(mixed)} too_strict={len(too_strict)}",
        },
        {
            "area": "live_overlap",
            "recommendation": "USE_V4_AS_TARGET_NOT_LIVE_FILTER",
            "reason": f"big_losses_blocked={overlap.get('live_big_losses_blocked')}/{overlap.get('live_big_losses_total')} premium_losers_blocked={overlap.get('premium_losers_blocked')}/{overlap.get('premium_losers_total')}",
        },
    ]
    for row in sorted(summary_rows, key=lambda r: (r.get("classification") != "V4_LABEL_PROMISING", r.get("symbol"), r.get("horizon"))):
        recs.append({
            "area": "config",
            "symbol": row.get("symbol"),
            "horizon": row.get("horizon"),
            "recommendation": row.get("classification"),
            "reason": f"clean_rate={row.get('clean_rate'):.4f} premium={row.get('premium_allowed_rate'):.4f} clean_quality={row.get('clean_net_quality')}",
        })
    return recs


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    promising = [r for r in payload["summary"] if r.get("classification") == "V4_LABEL_PROMISING"]
    lines = [
        "# SHORT V4 Dataset Labels A",
        "",
        "## Safety",
        "- research-only",
        "- no live changes",
        "- no active_manifest",
        "- no YAML",
        "- no PM2",
        "- no orders",
        "",
        "## Label Design",
        "- short_clean_entry_v4: fast MFE, controlled MAE, positive net quality after fees/slippage, MFE before MAE",
        "- short_bad_entry_v4: early MAE, low MFE, poor MFE/MAE, negative net quality or management dependency proxy",
        "- short_premium_allowed_v4: clean, fast, low danger, high MFE/MAE, allowed initial rebuild symbols only",
        "- short_management_dependent_v4: MAE/late MFE profile that can look acceptable only with management",
        "- short_no_trade_v4: no clean/premium edge or high-risk near-zero quality",
        "",
        "## Executive Summary",
        f"- Symbols: `{payload['symbols']}`",
        f"- Horizons: `{payload['horizons']}`",
        f"- Promising configs: `{len(promising)}`",
        f"- Live overlap: `{payload['live_overlap_summary']}`",
        "",
        "## By Config",
        "| symbol | horizon | status | clean_rate | premium_rate | clean_quality | clean_p90_mae | beats_random |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row.get('symbol')} | {row.get('horizon')} | {row.get('classification')} | "
            f"{row.get('clean_rate'):.4f} | {row.get('premium_allowed_rate'):.4f} | "
            f"{row.get('clean_net_quality')} | {row.get('clean_p90_mae')} | {row.get('v4_beats_random')} |"
        )
    lines.extend([
        "",
        "## Recommendations",
    ])
    for rec in payload["recommendations"][:20]:
        lines.append(f"- {rec.get('area')}: {rec.get('recommendation')} - {rec.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    stamp = utc_stamp()
    paths = {
        "md": out_dir / f"aegis_short_v4_dataset_labels_{stamp}.md",
        "json": out_dir / f"aegis_short_v4_dataset_labels_{stamp}.json",
        "summary": out_dir / f"aegis_short_v4_dataset_summary_{stamp}.csv",
        "by_symbol": out_dir / f"aegis_short_v4_dataset_by_symbol_{stamp}.csv",
        "by_horizon": out_dir / f"aegis_short_v4_dataset_by_horizon_{stamp}.csv",
        "live_overlap": out_dir / f"aegis_short_v4_dataset_live_overlap_{stamp}.csv",
        "random_baseline": out_dir / f"aegis_short_v4_dataset_random_baseline_{stamp}.csv",
        "recommendations": out_dir / f"aegis_short_v4_dataset_recommendations_{stamp}.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(paths["md"], payload)
    paths["json"].write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(paths["summary"], payload["summary"])
    write_csv(paths["by_symbol"], payload["by_symbol"])
    write_csv(paths["by_horizon"], payload["by_horizon"])
    write_csv(paths["live_overlap"], payload["live_overlap"])
    write_csv(paths["random_baseline"], payload["random_baseline"])
    write_csv(paths["recommendations"], payload["recommendations"])
    return {k: str(v) for k, v in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    normalized = [(a.replace("–", "--", 1) if a.startswith("–") else a) for a in (argv if argv is not None else sys.argv[1:])]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="ADAUSDT,SUIUSDT,SOLUSDT,AVAXUSDT")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--horizons", default="6,12,24")
    parser.add_argument("--out-dir", default="/home/jasan/Develop")
    parser.add_argument("--no-save-matrix", action="store_true", default=False)
    parser.add_argument("--save-summary", action="store_true", default=False)
    parser.add_argument("--include-phase-o-comparison", action="store_true")
    parser.add_argument("--include-live-trade-overlap", action="store_true")
    parser.add_argument("--include-random-baseline", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args(normalized)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_dataset(args)
    paths = write_outputs(payload, Path(args.out_dir))
    print(json.dumps({"status": "OK", "paths": paths, "promising": sum(1 for r in payload["summary"] if r.get("classification") == "V4_LABEL_PROMISING")}, indent=2))


if __name__ == "__main__":
    main()
