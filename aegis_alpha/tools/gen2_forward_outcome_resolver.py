#!/usr/bin/env python3
"""GEN2 forward outcome resolver — real implementation (replaces the stub).

Reads paper/live decision streams, downloads canonical FINAL candles for the
mature window, rebuilds H12 outcomes (terminal short return, MAE, tail label,
net quality after assumed costs), appends immutable evidence records exactly
once per decision, and produces an aggregated paper-vs-live report. It never
mutates historical evidence, never changes policy, never retrains, never
submits orders, and never reads immature futures.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, utc_now, validate_gen2_path  # noqa: E402
from aegis_alpha.tools.gen2_d3_snapshot import fetch_range  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
FORWARD_DECISIONS = GEN2_ROOT / "forward" / "forward_decisions.jsonl"
H12_BARS = 12
BAR_MS = 5 * 60_000
LEVERAGE_PROXY = 20.0
TAIL_ROE = 0.30
ASSUMED_COST_PCT_ROUND_TRIP = 0.0014  # 5bps fee + 2bps slip per side (ECON1 base scenario)
MATURITY_MINUTES = 65  # 12 bars + closing margin


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mature(ts: str, now: pd.Timestamp | None = None) -> bool:
    now = now or pd.Timestamp(utc_now()).tz_localize(None)
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return now >= t + timedelta(minutes=MATURITY_MINUTES)


def default_fetch_candles(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    cutoff = int((pd.Timestamp(utc_now()).tz_localize(None) - timedelta(seconds=120)).timestamp() * 1000)
    return fetch_range(symbol, start_ms, end_ms, cutoff)


def compute_outcome(candles: pd.DataFrame, decision_ts: pd.Timestamp) -> dict[str, Any] | None:
    ts = pd.to_datetime(candles["open_time"], unit="ms")
    idx_map = {t: i for i, t in enumerate(ts)}
    i = idx_map.get(decision_ts)
    if i is None or i + 1 + H12_BARS >= len(candles):
        return None
    entry = float(candles.iloc[i + 1]["open"])
    window = candles.iloc[i + 1: i + 1 + H12_BARS + 1]
    exit_ = float(window.iloc[-1]["close"])
    max_high = float(window["high"].max())
    gross_short_return = (entry - exit_) / entry
    mae_pct = max(0.0, (max_high - entry) / entry)
    mae_roe = mae_pct * LEVERAGE_PROXY
    return {
        "entry_price": entry,
        "exit_price": exit_,
        "gross_short_return_pct": gross_short_return,
        "net_short_return_pct": gross_short_return - ASSUMED_COST_PCT_ROUND_TRIP,
        "mae_pct": mae_pct,
        "mae_roe_proxy": mae_roe,
        "tail_risk_roe_030": int(mae_roe >= TAIL_ROE),
        "final_candles_used": int(len(window)),
    }


def decision_key(record: dict[str, Any]) -> str:
    return f"{record.get('candidate_id')}|{record.get('symbol')}|{record.get('ts') or record.get('timestamp')}"


def signal_id_of(record: dict[str, Any]) -> str:
    """Same derivation the decision loop uses when building the DecisionOrder."""
    ts = str(record.get("ts") or record.get("timestamp") or "")
    return f"{record.get('symbol')}-{ts.replace(' ', 'T')}"


def build_live_fill_index(cdir: Path) -> dict[str, dict[str, Any]]:
    """signal_id -> fill facts, joined through live_orders.jsonl (decisions carry
    no client_order_id; the submitted order is the only place both ids meet)."""
    id_by_signal: dict[str, str] = {}
    for row in read_jsonl(cdir / "live_orders.jsonl"):
        order = row.get("order") or {}
        if order.get("signal_id") and order.get("client_order_id"):
            id_by_signal[str(order["signal_id"])] = str(order["client_order_id"])
    fills_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(cdir / "fills.jsonl"):
        coid = row.get("client_order_id")
        if not coid or row.get("type") not in (None, "FILL"):
            continue
        payload = row.get("payload") or {}
        fills_by_id[str(coid)] = {k: payload.get(k, row.get(k)) for k in ("fill_price", "fill_qty", "fees", "latency_ms")}
    return {sig: {"client_order_id": coid, **fills_by_id[coid]}
            for sig, coid in id_by_signal.items() if coid in fills_by_id}


def resolve(candidate_id: str, decisions_path: Path = FORWARD_DECISIONS,
            fetch_fn: Callable[[str, int, int], pd.DataFrame] = default_fetch_candles,
            now: pd.Timestamp | None = None) -> dict[str, Any]:
    cdir = core.canary_dir(candidate_id)
    out_path = cdir / "forward_outcomes.jsonl"
    already = {r.get("key") for r in read_jsonl(out_path)}
    decisions = [r for r in read_jsonl(decisions_path) if r.get("candidate_id") == candidate_id]
    live_fills = build_live_fill_index(cdir)
    resolved = skipped_immature = skipped_done = failed = 0
    by_symbol_window: dict[str, list[pd.Timestamp]] = {}
    pending: list[dict[str, Any]] = []
    for rec in decisions:
        ts = rec.get("ts") or rec.get("timestamp")
        if not ts or rec.get("decision") == "NO_DECISION":
            continue
        if decision_key(rec) in already:
            skipped_done += 1
            continue
        if not mature(str(ts), now):
            skipped_immature += 1
            continue
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        by_symbol_window.setdefault(str(rec["symbol"]), []).append(t)
        pending.append(rec)
    candle_cache: dict[str, pd.DataFrame] = {}
    for symbol, stamps in by_symbol_window.items():
        start_ms = int(min(stamps).timestamp() * 1000) - BAR_MS
        end_ms = int(max(stamps).timestamp() * 1000) + (H12_BARS + 3) * BAR_MS
        try:
            candle_cache[symbol] = fetch_fn(symbol, start_ms, end_ms)
        except Exception as exc:
            candle_cache[symbol] = pd.DataFrame()
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl", {"type": "OUTCOME_FETCH_FAILED", "symbol": symbol, "error": repr(exc)})
    for rec in pending:
        t = pd.Timestamp(rec.get("ts") or rec.get("timestamp"))
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        candles = candle_cache.get(str(rec["symbol"]), pd.DataFrame())
        outcome = compute_outcome(candles, t) if len(candles) else None
        if outcome is None:
            failed += 1
            continue
        fill = live_fills.get(signal_id_of(rec))
        evidence = {
            "schema": "gen2_forward_outcome_v2",
            "key": decision_key(rec),
            "candidate_id": candidate_id,
            "symbol": rec["symbol"],
            "decision_ts": str(t),
            "stream": "PAPER_AND_LIVE" if fill else "PAPER",
            "hypothetical_action": rec.get("hypothetical_action"),
            "vetoed_by_trrm": rec.get("vetoed_by_trrm"),
            "eqm_score": rec.get("eqm_score"),
            "tail_score": rec.get("tail_score"),
            "qmae_q90": rec.get("qmae_q90"),
            **outcome,
            "live_fill": {k: fill.get(k) for k in ("client_order_id", "fill_price", "fill_qty", "fees", "latency_ms")} if fill else None,
            "paper_vs_live_entry_delta_pct": (
                (float(fill["fill_price"]) - outcome["entry_price"]) / outcome["entry_price"]
                if fill and fill.get("fill_price") else None
            ),
        }
        core.append_jsonl(out_path, evidence)
        resolved += 1
    rows = read_jsonl(out_path)
    taken = [r for r in rows if r.get("hypothetical_action") == "CANDIDATE_SHORT"]
    vetoed = [r for r in rows if r.get("vetoed_by_trrm")]
    summary = {
        "schema": "gen2_forward_outcome_resolution_v2",
        "candidate_id": candidate_id,
        "resolved_new": resolved,
        "skipped_already_resolved": skipped_done,
        "skipped_immature": skipped_immature,
        "failed_no_candles": failed,
        "total_evidence_records": len(rows),
        "candidate_short_count": len(taken),
        "candidate_short_mean_net_return_pct": float(np.mean([r["net_short_return_pct"] for r in taken])) if taken else None,
        "candidate_short_tail_rate": float(np.mean([r["tail_risk_roe_030"] for r in taken])) if taken else None,
        "vetoed_mean_net_return_pct": float(np.mean([r["net_short_return_pct"] for r in vetoed])) if vetoed else None,
        "vetoed_tail_rate": float(np.mean([r["tail_risk_roe_030"] for r in vetoed])) if vetoed else None,
        "live_fills_matched": sum(1 for r in rows if r.get("live_fill")),
        "policy_changed": False,
        "models_retrained": False,
        "orders_submitted": 0,
        "generated_at_utc": utc_now().isoformat(),
    }
    out = cdir / "outcome_resolution_latest.json"
    validate_gen2_path(out)
    out.write_text(json.dumps(summary, indent=2, default=json_default))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve mature Gen2 forward outcomes from canonical final candles")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--decisions-path", default=str(FORWARD_DECISIONS))
    args = parser.parse_args(argv)
    print(json.dumps(resolve(args.candidate_id, Path(args.decisions_path)), indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
