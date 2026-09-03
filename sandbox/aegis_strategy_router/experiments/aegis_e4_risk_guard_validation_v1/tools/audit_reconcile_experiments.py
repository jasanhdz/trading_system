#!/usr/bin/env python3
"""Reconcile AEGIS_ENTRY_ENHANCEMENT_V1 vs AEGIS_E4_RISK_GUARD_VALIDATION_V1.

Compares the two experiments trade-by-trade on the VALIDATION split to explain
why AEGIS_ONLY changed from -20.64 bps to +17.01 bps.

Outputs:
  - artifacts/audit/reconciliation_result.json
  - artifacts/audit/per_trade_reconciliation.csv
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "sandbox/aegis_strategy_router/src"))

TRADE_ID_RE = re.compile(r"^AEGIS-TURBO-[A-Z0-9]+-(\d{8})-(\d{6})-(\d{3})$")
SOURCE_CSV_REL = "reports/governance/aegis_prospective_validation/live/live_entry_quality_audit_20260815/live_entry_classification.csv"
SOURCE_CSV_SHA = "1d53b18171c97195be1f73a8c8d76966cb8383e24cc5f779c7a09c8f61214a38"

SPLITS = {
    "DISCOVERY": ["2026-05-07T00:00:00Z", "2026-07-01T00:00:00Z"],
    "EMBARGO_1": ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"],
    "CALIBRATION": ["2026-07-01T01:00:00Z", "2026-07-15T00:00:00Z"],
    "EMBARGO_2": ["2026-07-15T00:00:00Z", "2026-07-15T01:00:00Z"],
    "VALIDATION": ["2026-07-15T01:00:00Z", "2026-08-01T00:00:00Z"],
    "EMBARGO_3": ["2026-08-01T00:00:00Z", "2026-08-01T01:00:00Z"],
    "FINAL_HOLDOUT": ["2026-08-01T01:00:00Z", "2026-08-15T00:00:00Z"],
}
MAX_OPEN_MINUS_SIGNAL_MS = 60_000
BARRIER_ATR = 0.5
HORIZON_MINUTES = 60
CONSERVATIVE_COST_BPS = 20.0
E4_COST_BPS = 14.0
CANDLE_ROOT = REPO_ROOT / "data/aegis_entry_enhancement_v1/candles_1m"
LOG_ROOT = REPO_ROOT / "binance-futures-bot-ts/logs/aegis"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_timestamp(trade_id):
    m = TRADE_ID_RE.match(trade_id)
    if m is None:
        raise ValueError(f"UNPARSABLE:{trade_id}")
    day, clock, ms = m.groups()
    return pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}.{ms}Z")


def split_for(ts):
    val = pd.to_datetime(ts, utc=True)
    for name, bounds in SPLITS.items():
        if pd.Timestamp(bounds[0]) <= val < pd.Timestamp(bounds[1]):
            return name
    return "OUTSIDE"


def load_source_csv():
    csv_path = REPO_ROOT / SOURCE_CSV_REL
    actual = sha256_file(csv_path)
    if actual != SOURCE_CSV_SHA:
        raise RuntimeError(f"HASH_MISMATCH: expected {SOURCE_CSV_SHA}, got {actual}")
    df = pd.read_csv(csv_path)
    df["signal_timestamp"] = [signal_timestamp(t) for t in df["trade_id"]]
    df["opened_at_dt"] = pd.to_datetime(df["opened_at"], utc=True, format="mixed")
    df["split"] = [split_for(ts) for ts in df["signal_timestamp"]]
    df["opened_minus_signal_ms"] = (df["opened_at_dt"] - df["signal_timestamp"]).dt.total_seconds() * 1000.0
    return df


def load_open_events(required):
    events = {}
    for path in sorted(LOG_ROOT.glob("turbo_trades_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row.get("trade_id")
                if tid in required and row.get("status") == "OPEN":
                    events[tid] = row
    return events


def compute_barrier_outcome(candle_path, entry_price, signal_ts, side):
    frame = pd.read_parquet(candle_path).sort_values("open_time_ms", kind="mergesort")
    start_ms = int(signal_ts.floor("min").timestamp() * 1000)
    horizon_ms = HORIZON_MINUTES * 60_000
    future = frame.loc[
        frame.open_time_ms.ge(start_ms) & frame.open_time_ms.lt(start_ms + horizon_ms)
    ].sort_values("open_time_ms", kind="mergesort")
    if len(future) < HORIZON_MINUTES:
        return {"barrier_gross_bps": np.nan, "barrier_label": "INCOMPLETE",
                "barrier_net_bps": np.nan, "barrier_bps": np.nan,
                "barrier_mfe_bps": np.nan, "barrier_mae_bps": np.nan}

    # ATR14 from 15m bars
    tf15_start = int(signal_ts.floor("15min").timestamp() * 1000)
    tf15_frame = frame.loc[frame.open_time_ms.le(tf15_start)].tail(15)
    if len(tf15_frame) < 14:
        tr = future["high"].to_numpy(float) - future["low"].to_numpy(float)
        atr14 = float(np.mean(tr[:14])) if len(tr) >= 14 else float(np.mean(tr))
    else:
        tr_15m = tf15_frame["high"].to_numpy(float) - tf15_frame["low"].to_numpy(float)
        atr14 = float(np.mean(tr_15m[-14:]))

    barrier_abs = BARRIER_ATR * atr14
    barrier_bps = barrier_abs / entry_price * 10_000.0
    high = future["high"].to_numpy(float)
    low = future["low"].to_numpy(float)
    close = future["close"].to_numpy(float)

    if side == "LONG":
        favorable = high >= entry_price + barrier_abs
        adverse = low <= entry_price - barrier_abs
        fav_exc = (high - entry_price) / entry_price * 10_000.0
        adv_exc = (entry_price - low) / entry_price * 10_000.0
    else:
        favorable = low <= entry_price - barrier_abs
        adverse = high >= entry_price + barrier_abs
        fav_exc = (entry_price - low) / entry_price * 10_000.0
        adv_exc = (high - entry_price) / entry_price * 10_000.0

    f_idx = int(np.argmax(favorable)) if favorable.any() else None
    a_idx = int(np.argmax(adverse)) if adverse.any() else None

    if a_idx is not None and (f_idx is None or a_idx <= f_idx):
        label, gross = "ADVERSE_FIRST", -barrier_bps
    elif f_idx is not None:
        label, gross = "FAVORABLE_FIRST", barrier_bps
    else:
        label = "NEITHER"
        d = 1.0 if side == "LONG" else -1.0
        gross = d * (close[-1] - entry_price) / entry_price * 10_000.0

    mfe = max(0.0, float(np.max(fav_exc)))
    mae = max(0.0, float(np.max(adv_exc)))
    return {
        "barrier_gross_bps": float(gross),
        "barrier_net_bps": float(gross - CONSERVATIVE_COST_BPS),
        "barrier_label": label,
        "barrier_bps": float(barrier_bps),
        "barrier_mfe_bps": float(mfe),
        "barrier_mae_bps": float(mae),
        "atr14_15m": float(atr14),
    }


def main():
    print("=" * 80)
    print("AUDIT: AEGIS_ENTRY_ENHANCEMENT_V1 vs AEGIS_E4_RISK_GUARD_VALIDATION_V1")
    print("=" * 80)

    # Step 1: Load source CSV
    print("\n[1/6] Loading source CSV...")
    source = load_source_csv()
    print(f"      Total rows in CSV: {len(source)}")

    # Step 2: Timestamp-integrity filter
    print("\n[2/6] Applying timestamp-integrity filter (60s)...")
    invalid = source["opened_minus_signal_ms"].gt(MAX_OPEN_MINUS_SIGNAL_MS)
    excluded_ids = source.loc[invalid, "trade_id"].tolist()
    eligible = source.loc[~invalid].copy()
    print(f"      Excluded: {len(excluded_ids)}")
    print(f"      Eligible: {len(eligible)}")

    # Step 3: Split distribution
    print("\n[3/6] Split distribution:")
    for name, count in eligible["split"].value_counts().items():
        print(f"      {name}: {count}")

    val = eligible.loc[eligible["split"] == "VALIDATION"].copy()
    print(f"\n      VALIDATION: {len(val)} signals")
    sides = val["side"].value_counts()
    for s, c in sides.items():
        print(f"        {s}: {c}")

    # Step 4: Load open events
    print("\n[4/6] Loading open events...")
    val_ids = set(val["trade_id"])
    events = load_open_events(val_ids)
    missing = val_ids - set(events.keys())
    if missing:
        print(f"      MISSING: {len(missing)}")
    else:
        print(f"      All {len(val_ids)} found.")

    val["entry_price_event"] = val["trade_id"].map(
        lambda t: float(events[t]["entry_price"]) if t in events else np.nan
    )
    val["side_event"] = val["trade_id"].map(
        lambda t: events[t].get("side") if t in events else None
    )
    side_mismatch = val[val["side"] != val["side_event"]]
    if len(side_mismatch) > 0:
        print(f"      SIDE MISMATCHES: {len(side_mismatch)}")

    # Step 5: Compute barrier outcomes
    print("\n[5/6] Computing barrier-based outcomes (60min ATR barrier)...")
    barrier_results = []
    n = len(val)
    for i, (_, row) in enumerate(val.iterrows()):
        tid = row["trade_id"]
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"      Processing {i+1}/{n}...")
        if tid not in events or np.isnan(row["entry_price_event"]):
            barrier_results.append({"trade_id": tid, "barrier_gross_bps": np.nan,
                                    "barrier_net_bps": np.nan, "barrier_label": "NO_EVENT"})
            continue
        cp = CANDLE_ROOT / f"{row['symbol']}_1m.parquet"
        if not cp.exists():
            barrier_results.append({"trade_id": tid, "barrier_gross_bps": np.nan,
                                    "barrier_net_bps": np.nan, "barrier_label": "NO_CANDLES"})
            continue
        r = compute_barrier_outcome(cp, row["entry_price_event"], row["signal_timestamp"], row["side"])
        r["trade_id"] = tid
        barrier_results.append(r)

    bdf = pd.DataFrame(barrier_results)
    val = val.merge(bdf, on="trade_id", how="left")

    # Realized PnL
    val["realized_gross_bps"] = (val["roe"] / val["leverage"].clip(lower=1.0)) * 10_000.0
    val["realized_net_bps"] = val["realized_gross_bps"] - E4_COST_BPS

    # Step 6: Analysis
    print("\n[6/6] Analysis...")

    valid_both = val.dropna(subset=["barrier_gross_bps", "realized_gross_bps"])
    n_valid = len(valid_both)

    barrier_gross_mean = float(valid_both["barrier_gross_bps"].mean())
    barrier_net_mean = float(valid_both["barrier_net_bps"].mean())
    realized_gross_mean = float(valid_both["realized_gross_bps"].mean())
    realized_net_mean = float(valid_both["realized_net_bps"].mean())

    # Side breakdown
    side_comp = {}
    for sv in ["LONG", "SHORT"]:
        sdf = valid_both.loc[valid_both["side"] == sv]
        if len(sdf) == 0:
            continue
        side_comp[sv] = {
            "count": len(sdf),
            "barrier_gross_bps": float(sdf["barrier_gross_bps"].mean()),
            "barrier_net_bps": float(sdf["barrier_net_bps"].mean()),
            "realized_gross_bps": float(sdf["realized_gross_bps"].mean()),
            "realized_net_bps": float(sdf["realized_net_bps"].mean()),
            "gross_diff_bps": float(sdf["realized_gross_bps"].mean() - sdf["barrier_gross_bps"].mean()),
        }

    # Barrier label distribution
    label_dist = valid_both["barrier_label"].value_counts().to_dict()

    # Per-trade CSV
    per_trade = valid_both[[
        "trade_id", "symbol", "side", "signal_timestamp", "roe", "leverage",
        "entry_price_event", "barrier_gross_bps", "barrier_net_bps", "barrier_label",
        "barrier_bps", "barrier_mfe_bps", "barrier_mae_bps",
        "realized_gross_bps", "realized_net_bps",
        "mfe_bps_underlying", "mae_bps_underlying",
    ]].copy()
    per_trade["gross_diff_bps"] = per_trade["realized_gross_bps"] - per_trade["barrier_gross_bps"]
    per_trade = per_trade.rename(columns={
        "mfe_bps_underlying": "realized_mfe_bps",
        "mae_bps_underlying": "realized_mae_bps",
    })
    per_trade = per_trade.sort_values("trade_id")

    # Result JSON
    result = {
        "schema": "reconciliation-v1",
        "population": {
            "total_csv_rows": len(source),
            "timestamp_excluded": len(excluded_ids),
            "excluded_trade_ids": excluded_ids,
            "eligible_signals": len(eligible),
            "splits": {k: int(v) for k, v in eligible["split"].value_counts().items()},
            "validation_signals": len(val),
            "validation_sides": val["side"].value_counts().to_dict(),
        },
        "aegis_only_comparison": {
            "description": "Same VALIDATION signals, two different outcome definitions",
            "signals_compared": n_valid,
            "signals_with_nan": len(val) - n_valid,
            "entry_enhancement_v1": {
                "outcome_definition": "FIXED_60MIN_ATR_BARRIER",
                "cost_bps": CONSERVATIVE_COST_BPS,
                "gross_bps_per_signal": barrier_gross_mean,
                "net_bps_per_signal": barrier_net_mean,
            },
            "e4_risk_guard_v1": {
                "outcome_definition": "REALIZED_PNL_FROM_BOT",
                "cost_bps": E4_COST_BPS,
                "gross_bps_per_signal": realized_gross_mean,
                "net_bps_per_signal": realized_net_mean,
            },
            "gross_difference_bps": realized_gross_mean - barrier_gross_mean,
            "net_difference_bps": realized_net_mean - barrier_net_mean,
            "cost_difference_bps": CONSERVATIVE_COST_BPS - E4_COST_BPS,
            "unexplained_by_cost_bps": (realized_net_mean - barrier_net_mean) - (CONSERVATIVE_COST_BPS - E4_COST_BPS),
        },
        "side_breakdown": side_comp,
        "barrier_label_distribution": label_dist,
        "entry_enhancement_claimed_side_dist": "100_PERCENT_SHORT",
        "actual_side_dist_in_csv": val["side"].value_counts().to_dict(),
        "side_discrepancy_explanation": "PENDING_MANUAL_REVIEW",
    }

    # Save outputs
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "reconciliation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    per_trade.to_csv(out_dir / "per_trade_reconciliation.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RECONCILIATION SUMMARY")
    print("=" * 80)
    print(f"\nPopulation: {len(val)} VALIDATION signals")
    print(f"  Sides: {val['side'].value_counts().to_dict()}")
    print(f"  Entry Enhancement claimed: 100% SHORT")
    print(f"\nMetrics on same {n_valid} signals (both computed):")
    print(f"  {'':30s} {'Barrier (EE)':>15s} {'Realized (E4)':>15s} {'Diff':>10s}")
    print(f"  {'gross_bps':30s} {barrier_gross_mean:>15.2f} {realized_gross_mean:>15.2f} {realized_gross_mean - barrier_gross_mean:>+10.2f}")
    print(f"  {'net_bps':30s} {barrier_net_mean:>15.2f} {realized_net_mean:>15.2f} {realized_net_mean - barrier_net_mean:>+10.2f}")
    print(f"\nCost difference: {CONSERVATIVE_COST_BPS} vs {E4_COST_BPS} = {CONSERVATIVE_COST_BPS - E4_COST_BPS} bps")
    unexplained = (realized_net_mean - barrier_net_mean) - (CONSERVATIVE_COST_BPS - E4_COST_BPS)
    print(f"Unexplained by cost: {unexplained:+.2f} bps")
    print(f"\nBarrier label distribution: {label_dist}")
    if side_comp:
        print(f"\nSide breakdown:")
        for s, d in side_comp.items():
            print(f"  {s} ({d['count']} trades): barrier={d['barrier_gross_bps']:.2f} bps, realized={d['realized_gross_bps']:.2f} bps, diff={d['gross_diff_bps']:+.2f} bps")

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  reconciliation_result.json")
    print(f"  per_trade_reconciliation.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
