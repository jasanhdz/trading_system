#!/usr/bin/env python3
"""GEN2 decision loop — the complete brain cycle (architecture review C4).

closed candle -> canonical features -> TRRM -> QMAE -> EQM -> policy ->
paper decision (ALWAYS) -> gates -> TS execution request (when enabled) ->
event ingestion -> (outcome resolver runs separately on mature records).

Fail-closed: any uncertainty produces NO_DECISION / NO_TRADE with the exact
reason recorded. Paper stream never depends on the live path. Scientific
models and hashes are loaded from the frozen candidate and verified.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_bridge_client as bridge  # noqa: E402
import aegis_alpha.tools.gen2_canary_core as core  # noqa: E402
import aegis_alpha.tools.gen2_operational_contract as oc  # noqa: E402
from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import add_market_context, compute_causal_features  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, sha256_file, utc_now, validate_gen2_path  # noqa: E402
from aegis_alpha.tools.gen2_d3_snapshot import build_snapshot, load_snapshot_symbol  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import MedianImputer, score_of  # noqa: E402

DEFAULT_CANDIDATE_ID = "gen2-20260711T202935Z"
FORWARD_ROOT = GEN2_ROOT / "forward"
SERIES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT", "LINKUSDT"]
WARMUP_DAYS = 15
PRIMARY_HORIZON = 12
STOP_DISTANCE_PCT = oc.STOP_DISTANCE_PCT


def load_bundles(trrm_dir: Path, eqm_dir: Path, freeze: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.modules["__main__"].MedianImputer = MedianImputer
    trrm_pkl = trrm_dir / "rv2_candidate.pkl"
    eqm_pkl = eqm_dir / "eqm1_candidate.pkl"
    if sha256_file(trrm_pkl) != freeze["trrm_v2_sha256"]:
        raise ValueError("TRRM_HASH_MISMATCH_VS_FREEZE")
    if sha256_file(eqm_pkl) != freeze["eqm1_sha256"]:
        raise ValueError("EQM_HASH_MISMATCH_VS_FREEZE")
    with trrm_pkl.open("rb") as f:
        trrm = pickle.load(f)
    with eqm_pkl.open("rb") as f:
        eqm = pickle.load(f)
    return trrm, eqm


def evaluate_symbol(symbol: str, candles: pd.DataFrame, context: dict[str, pd.DataFrame],
                    trrm: dict[str, Any], eqm: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    """Frozen evaluator: features -> TRRM -> QMAE -> EQM -> hypothetical action."""
    feats = add_market_context(compute_causal_features(candles), context)
    last = feats.iloc[[-1]]
    ts = str(last.iloc[0]["timestamp"])
    row = {f"feature.{c}": last.iloc[0][c] for c in last.columns if c not in {"timestamp", "open", "high", "low", "volume"}}
    frame = pd.DataFrame([row])
    for h in (6, 12, 24):
        frame[f"horizon_{h}"] = 1.0 if h == PRIMARY_HORIZON else 0.0
    missing = [c for c in trrm["features"] if c not in frame.columns]
    if missing:
        return {"symbol": symbol, "ts": ts, "decision": "NO_DECISION", "reason": f"MISSING_FEATURES:{missing[:3]}"}
    x_t = trrm["imputer"].transform(frame[trrm["features"]].apply(pd.to_numeric, errors="coerce"))
    tail_raw = score_of(trrm["trrm_model"], x_t)
    tail = float(trrm["calibrator"].predict(tail_raw)[0]) if trrm.get("calibrator_kind") == "isotonic" and trrm.get("calibrator") is not None else float(tail_raw[0])
    qmae90 = float(trrm["qmae_models"]["q90"].predict(x_t)[0] + trrm.get("qmae_q90_conformal_adjustment", 0.0)) if trrm.get("qmae_models") else None
    x_e = eqm["imputer"].transform(frame[eqm["features"]].apply(pd.to_numeric, errors="coerce"))
    s_reg = float(eqm["reg_model"].predict(x_e)[0])
    s_clf = float(score_of(eqm["clf_model"], x_e)[0])
    s = s_clf * s_reg if eqm.get("score_kind") == "composite_ev" else s_reg
    if not np.isfinite(s) or not np.isfinite(tail):
        return {"symbol": symbol, "ts": ts, "decision": "NO_DECISION", "reason": "SCORE_NOT_FINITE"}
    vetoed = bool(tail >= freeze["veto"]["threshold_full_dev_informational"])
    return {
        "symbol": symbol, "ts": ts, "candidate_id": freeze["candidate_id"],
        "tail_score": tail, "qmae_q90": qmae90, "eqm_reg": s_reg, "eqm_clf": s_clf, "eqm_score": s,
        "vetoed_by_trrm": vetoed,
        "hypothetical_action": "ABSTAIN" if vetoed else "CANDIDATE_SHORT",
        "signal_price": float(candles.iloc[-1]["close"]),
        "enforcement_action": "NONE",
    }


def default_symbol_filters(symbol: str) -> dict[str, Any] | None:
    """Real exchangeInfo filters; None (fail-closed) when unavailable."""
    from aegis_alpha.tools.gen2_canary_exec import load_public_exchange_info

    info = load_public_exchange_info((symbol,))
    f = info.get(symbol)
    if f is None:
        return None
    return {"step_size": f.step_size, "min_notional": f.min_notional, "min_qty": f.min_qty}


def build_decision_order(decision: dict[str, Any], sizing: dict[str, Any], token: dict[str, Any]) -> dict[str, Any]:
    from aegis_alpha.tools.gen2_canary_exec import deterministic_client_order_id

    signal_id = f"{decision['symbol']}-{decision['ts'].replace(' ', 'T')}"
    client_order_id = deterministic_client_order_id(decision["candidate_id"], signal_id, decision["symbol"], "SHORT")
    entry_price = decision["signal_price"]
    time_exit = str(pd.Timestamp(decision["ts"]) + pd.Timedelta(minutes=5 * (PRIMARY_HORIZON + 1)))
    return {
        "schema": "gen2_decision_order_v1",
        "candidate_id": decision["candidate_id"],
        "client_order_id": client_order_id,
        "signal_id": signal_id,
        "opportunity_ts": decision["ts"],
        "symbol": decision["symbol"],
        "side": "SHORT",
        "quantity": sizing["quantity"],
        "leverage": sizing["leverage"],
        "margin_type": "ISOLATED",
        "entry": {"type": "MARKET"},
        "brackets": {"stop_price": round(entry_price * (1 + STOP_DISTANCE_PCT), 8), "time_exit_at": time_exit, "reduce_only": True},
        "risk_context": {"max_loss_usd": sizing["per_stop_loss"], "arm_token_checksum": token.get("checksum")},
        "expires_at": str(pd.Timestamp(utc_now().isoformat()).tz_localize(None) + pd.Timedelta(seconds=60)),
    }


def gates_for_live(candidate_id: str, decision: dict[str, Any], token_symbols: list[str], bridge_status: dict[str, Any] | None) -> tuple[bool, str]:
    if decision.get("decision") == "NO_DECISION":
        return False, str(decision.get("reason"))
    if decision.get("hypothetical_action") != "CANDIDATE_SHORT":
        return False, "VETOED_OR_ABSTAIN"
    if decision["symbol"] not in token_symbols:
        return False, "SYMBOL_NOT_IN_ARM_TOKEN"
    ok, reason = oc.risk_gate(candidate_id)
    if not ok:
        return False, reason
    ok, reason, _ = oc.verify_arm_token(candidate_id)
    if not ok:
        return False, reason
    if bridge_status is None:
        return False, "BRIDGE_STATUS_UNAVAILABLE"
    if bridge_status.get("phase_o_allow_orders") is not False:
        return False, "PHASE_O_NOT_PAUSED_PER_BRIDGE"
    if bridge_status.get("gen2_enabled") is not True:
        return False, "BRIDGE_GEN2_DISABLED"
    if bridge_status.get("open_positions"):
        return False, "MAX_CONCURRENT_POSITIONS_REACHED"
    return True, "ELIGIBLE"


def run_cycle(candidate_id: str = DEFAULT_CANDIDATE_ID,
              trrm_dir: Path | None = None, eqm_dir: Path | None = None,
              live_enabled: bool = False,
              snapshot_fn: Any = None, status_fn: Any = None, execute_fn: Any = None,
              filters_fn: Any = None) -> dict[str, Any]:
    freeze = json.loads(core.FREEZE_PATH.read_text())
    if freeze["candidate_id"] != candidate_id:
        raise ValueError("CANDIDATE_MISMATCH")
    trrm_dir = trrm_dir or GEN2_ROOT / "rv2" / "20260711T171832Z"
    eqm_dir = eqm_dir or GEN2_ROOT / "eqm1" / "20260711T201456Z"
    trrm, eqm = load_bundles(trrm_dir, eqm_dir, freeze)
    cdir = core.canary_dir(candidate_id)
    state_path = cdir / "loop_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"last_processed": {}}

    snapshot_fn = snapshot_fn or (lambda: build_snapshot(symbols=SYMBOLS, range_days=WARMUP_DAYS, snapshot_root=FORWARD_ROOT / "snapshots"))
    snap = snapshot_fn()
    snap_dir = Path(snap["directory"])
    context: dict[str, pd.DataFrame] = {}
    for ctx in ("BTCUSDT", "ETHUSDT"):
        if ctx in snap["symbols"]:
            raw = load_snapshot_symbol(snap_dir, ctx).rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS]
            context[ctx] = compute_causal_features(raw)

    bridge_status = None
    if live_enabled:
        try:
            bridge_status = (status_fn or bridge.get_status)()
        except Exception:
            bridge_status = None

    token_ok, _, token = oc.verify_arm_token(candidate_id)
    contract = None
    try:
        contract = oc.load_contract(candidate_id)
    except Exception:
        pass

    decisions: list[dict[str, Any]] = []
    live_attempts: list[dict[str, Any]] = []
    for symbol in snap["symbols"]:
        raw = load_snapshot_symbol(snap_dir, symbol).rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS]
        decision = evaluate_symbol(symbol, raw, context, trrm, eqm, freeze)
        if state["last_processed"].get(symbol) == decision.get("ts"):
            continue  # dedup: one decision per closed candle
        state["last_processed"][symbol] = decision.get("ts")
        core.append_jsonl(FORWARD_ROOT / "forward_decisions.jsonl", {"collected_at": utc_now().isoformat(), **decision})
        decisions.append(decision)
        eligible, reason = gates_for_live(candidate_id, decision, token.get("allowed_symbols", []) if token_ok else [], bridge_status)
        attempt: dict[str, Any] = {"symbol": symbol, "ts": decision.get("ts"), "eligible": eligible, "reason": reason}
        if eligible and live_enabled and contract is not None:
            try:
                filters = (filters_fn or default_symbol_filters)(symbol)
            except Exception:
                filters = None
            if filters is None:
                attempt["reason"] = "FILTERS_UNAVAILABLE"  # fail-closed: never size with guessed filters
                live_attempts.append(attempt)
                continue
            sizing = oc.compute_sizing(contract, decision["signal_price"], float(bridge_status.get("available_balance", 0) or 0), filters["step_size"], filters["min_notional"], filters.get("min_qty", 0.0))
            attempt["sizing"] = sizing
            if sizing["decision"] != "SIZED":
                attempt["reason"] = sizing["reason"]
            else:
                order = build_decision_order(decision, sizing, token)
                ack = (execute_fn or bridge.post_execute)(order)
                attempt["ack"] = ack
                core.append_jsonl(cdir / "live_orders.jsonl", {"order": order, "ack": ack})
                if ack.get("status") == "ACCEPTED":
                    oc.consume_order(candidate_id)
                elif ack.get("status") == "BRIDGE_UNAVAILABLE":
                    core.append_jsonl(cdir / "incidents" / "incidents.jsonl", {"type": "BRIDGE_UNAVAILABLE_ON_SUBMIT", "order": order["client_order_id"]})
        elif eligible and not live_enabled:
            attempt["reason"] = "WOULD_SUBMIT_LIVE_DISABLED"
        live_attempts.append(attempt)
    core.atomic_write(state_path, json.dumps(state, indent=2))
    summary = {
        "schema": "gen2_decision_loop_cycle_v1",
        "candidate_id": candidate_id,
        "snapshot": snap.get("snapshot_id"),
        "decisions": len(decisions),
        "candidate_short": sum(1 for d in decisions if d.get("hypothetical_action") == "CANDIDATE_SHORT"),
        "vetoed": sum(1 for d in decisions if d.get("vetoed_by_trrm")),
        "no_decision": sum(1 for d in decisions if d.get("decision") == "NO_DECISION"),
        "live_enabled": live_enabled,
        "live_attempts": live_attempts,
        "orders_submitted": sum(1 for a in live_attempts if a.get("ack", {}).get("status") == "ACCEPTED"),
        "generated_at_utc": utc_now().isoformat(),
    }
    core.atomic_write(cdir / "loop_last_cycle.json", json.dumps(summary, indent=2, default=json_default))
    return summary


def ingest_events(candidate_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """At-least-once ingestion with sequence dedup; wires PnL back into the risk gate."""
    cdir = core.canary_dir(candidate_id)
    seen_path = cdir / "events_seen.json"
    seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
    ingested = 0
    for ev in events:
        key = f"{ev.get('client_order_id')}|{ev.get('ts_sequence')}"
        if key in seen:
            continue
        seen.add(key)
        kind = ev.get("type")
        target = {"FILL": "fills.jsonl", "BRACKET_CONFIRMED": "brackets.jsonl", "BRACKET_FAILED": "brackets.jsonl",
                  "POSITION_CLOSED": "fills.jsonl", "ORDER_REJECTED": "live_orders.jsonl",
                  "TIME_EXIT_EXECUTED": "fills.jsonl", "TIME_EXIT_SKIPPED_NO_POSITION": "fills.jsonl",
                  "RECONCILIATION": "reconciliations.jsonl", "INCIDENT": "incidents/incidents.jsonl"}.get(kind, "live_orders.jsonl")
        core.append_jsonl(cdir / target, ev)
        if kind == "BRACKET_FAILED":
            core.engage_kill_switch(candidate_id, "CRITICAL_EXECUTION_FAILURE_BRACKET")
        if kind == "POSITION_CLOSED":
            pnl = ev.get("payload", {}).get("realized_pnl")
            if pnl is not None:
                core.record_trade_result(candidate_id, float(pnl))
            else:
                # never guess PnL: unknown result requires the operator (runbook §3.5)
                core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                                  {"type": "POSITION_CLOSED_UNKNOWN_PNL_REQUIRES_OPERATOR", "event": ev})
        ingested += 1
    core.atomic_write(seen_path, json.dumps(sorted(seen)))
    return {"ingested": ingested, "total_seen": len(seen)}


def pull_events(candidate_id: str, fetch_fn: Any = None) -> dict[str, Any]:
    """Drain execution events from the TS bridge into the canary streams.

    The consumed-sequence checkpoint persists across runs; ingest_events dedups
    by (client_order_id, ts_sequence) so at-least-once delivery is safe.
    """
    cdir = core.canary_dir(candidate_id)
    ckpt_path = cdir / "events_checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {"last_sequence": 0}
    try:
        events = (fetch_fn or bridge.get_events)(int(ckpt["last_sequence"]))
    except Exception as exc:
        core.append_jsonl(cdir / "incidents" / "incidents.jsonl", {"type": "EVENTS_PULL_FAILED", "error": repr(exc)})
        return {"pulled": 0, "ingested": 0, "last_sequence": ckpt["last_sequence"], "error": "BRIDGE_UNAVAILABLE"}
    result = ingest_events(candidate_id, events)
    if events:
        # never regress on re-delivery of already-seen sequences
        ckpt["last_sequence"] = max(int(ckpt["last_sequence"]), max(int(e.get("ts_sequence", 0)) for e in events))
    core.atomic_write(ckpt_path, json.dumps(ckpt, indent=2))
    return {"pulled": len(events), **result, "last_sequence": ckpt["last_sequence"]}


# Scientific-integrity failures must stop the runner cold; anything else is an
# operational incident the watcher records and survives (fail-closed per cycle).
HARD_STOP_MARKERS = ("TRRM_HASH_MISMATCH_VS_FREEZE", "EQM_HASH_MISMATCH_VS_FREEZE", "CANDIDATE_MISMATCH")


def run_watch(candidate_id: str = DEFAULT_CANDIDATE_ID, interval_seconds: float = 300.0,
              max_cycles: int | None = None, live_enabled: bool = False,
              cycle_fn: Any = None, pull_fn: Any = None, resolve_fn: Any = None,
              sleep_fn: Any = time.sleep) -> dict[str, Any]:
    """Autonomous forward-evidence runner: events -> decisions -> outcomes -> heartbeat.

    Paper evidence collection never stops for live-path problems: the kill
    switch / risk gates silently block submissions while decisions and
    outcomes keep accruing. Only a scientific-integrity failure (hash or
    candidate mismatch) aborts the watcher.
    """
    cdir = core.canary_dir(candidate_id)
    counters = {"cycles": 0, "cycle_errors": 0, "resolve_errors": 0, "decisions": 0,
                "orders_submitted": 0, "outcomes_resolved": 0, "events_ingested": 0}
    heartbeat: dict[str, Any] = {}
    while max_cycles is None or counters["cycles"] < max_cycles:
        cycle_started = utc_now().isoformat()
        if live_enabled:
            ev = (pull_fn or pull_events)(candidate_id)
            counters["events_ingested"] += int(ev.get("ingested", 0))
        try:
            summary = (cycle_fn or run_cycle)(candidate_id, live_enabled=live_enabled)
            counters["decisions"] += int(summary.get("decisions", 0))
            counters["orders_submitted"] += int(summary.get("orders_submitted", 0))
        except Exception as exc:
            if any(m in str(exc) for m in HARD_STOP_MARKERS):
                core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                                  {"type": "WATCH_HARD_STOP_INTEGRITY", "error": repr(exc)})
                raise
            counters["cycle_errors"] += 1
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                              {"type": "WATCH_CYCLE_FAILED", "error": repr(exc)})
        try:
            if resolve_fn is None:
                import aegis_alpha.tools.gen2_forward_outcome_resolver as resolver

                resolve_fn_actual = resolver.resolve
            else:
                resolve_fn_actual = resolve_fn
            res = resolve_fn_actual(candidate_id)
            counters["outcomes_resolved"] += int(res.get("resolved_new", 0))
        except Exception as exc:
            counters["resolve_errors"] += 1
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                              {"type": "WATCH_RESOLVE_FAILED", "error": repr(exc)})
        counters["cycles"] += 1
        heartbeat = {
            "schema": "gen2_watch_heartbeat_v1",
            "candidate_id": candidate_id,
            "live_enabled": live_enabled,
            "interval_seconds": interval_seconds,
            "cycle_started_utc": cycle_started,
            "cycle_finished_utc": utc_now().isoformat(),
            "kill_switch": core.kill_switch_engaged(candidate_id),
            **counters,
        }
        core.atomic_write(cdir / "heartbeat.json", json.dumps(heartbeat, indent=2, default=json_default))
        if max_cycles is None or counters["cycles"] < max_cycles:
            sleep_fn(interval_seconds)
    return heartbeat


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GEN2 decision loop (manual; no PM2)")
    p.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    p.add_argument("--live", action="store_true", help="enable TS bridge submission path (still gated by token/contract/kill)")
    p.add_argument("--pull-events", action="store_true", help="drain bridge execution events before the cycle")
    p.add_argument("--events-only", action="store_true", help="only drain bridge events; skip the decision cycle")
    p.add_argument("--watch", action="store_true", help="autonomous runner: events -> cycle -> outcomes -> heartbeat, repeat")
    p.add_argument("--interval-seconds", type=float, default=300.0)
    p.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles (default: run until interrupted)")
    args = p.parse_args(argv)
    if args.watch:
        out: dict[str, Any] = {"watch": run_watch(args.candidate_id, args.interval_seconds, args.max_cycles, live_enabled=args.live)}
    else:
        out = {}
        if args.pull_events or args.events_only:
            out["events"] = pull_events(args.candidate_id)
        if not args.events_only:
            out["cycle"] = run_cycle(args.candidate_id, live_enabled=args.live)
    print(json.dumps(out, indent=2, default=json_default)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
