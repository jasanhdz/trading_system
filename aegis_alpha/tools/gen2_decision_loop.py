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


def check_environment_vs_freeze(freeze: dict[str, Any]) -> None:
    """Fail-closed BEFORE unpickling: the frozen candidates are only valid in the
    environment they were trained in (two venvs named rocm62 exist on this host;
    pandas-3 pickles do not load under pandas 2)."""
    env = freeze.get("environment") or {}
    expected = {k: env.get(k) for k in ("pandas", "sklearn", "numpy") if env.get(k)}
    if not expected:
        return
    import numpy
    import sklearn

    current = {"pandas": pd.__version__, "sklearn": sklearn.__version__, "numpy": numpy.__version__}
    mismatched = {k: (v, current[k]) for k, v in expected.items() if current[k] != v}
    if mismatched:
        raise ValueError(
            f"ENVIRONMENT_MISMATCH_VS_FREEZE: {mismatched} — run with the freeze interpreter "
            f"{env.get('executable', '(unrecorded)')}"
        )


def load_bundles(trrm_dir: Path, eqm_dir: Path, freeze: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    check_environment_vs_freeze(freeze)
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
    return {"step_size": f.step_size, "min_notional": f.min_notional, "min_qty": f.min_qty, "tick_size": f.tick_size}


def round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the symbol's tick size. Binance rejects orders whose
    price is not a multiple of the tick (this caused a real bracket failure on
    DOGEUSDT when the stop was rounded to 8 decimals instead of the 5-decimal
    tick), so the stop price MUST snap to the tick or the whole order aborts."""
    if not tick_size or tick_size <= 0:
        return round(price, 8)
    import math

    decimals = max(0, -int(round(math.log10(tick_size)))) if tick_size < 1 else 0
    return round(round(price / tick_size) * tick_size, decimals + 2)


def build_decision_order(decision: dict[str, Any], sizing: dict[str, Any], contract: dict[str, Any],
                         tick_size: float | None = None) -> dict[str, Any]:
    from aegis_alpha.tools.gen2_canary_exec import deterministic_client_order_id

    signal_id = f"{decision['symbol']}-{decision['ts'].replace(' ', 'T')}"
    client_order_id = deterministic_client_order_id(decision["candidate_id"], signal_id, decision["symbol"], "SHORT")
    entry_price = decision["signal_price"]
    raw_stop = entry_price * (1 + STOP_DISTANCE_PCT)
    stop_price = round_to_tick(raw_stop, tick_size) if tick_size else round(raw_stop, 8)
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
        "brackets": {"stop_price": stop_price, "time_exit_at": time_exit, "reduce_only": True},
        "risk_context": {"max_loss_usd": sizing["per_stop_loss"], "config_checksum": (contract or {}).get("config_checksum")},
        # Explicit ownership so the TS executor never manages this position with
        # another strategy's policy (Phase O adopting a Gen2 short is the bug
        # this closes). The bridge persists this into the ownership registry.
        "strategy_context": {
            "owner": "GEN2",
            "strategy": "GEN2_EQM_TRRM",
            "exit_policy": "GEN2_H12",
            "risk_policy": f"GEN2_{(contract or {}).get('mode', 'EXPERIMENTAL')}",
            "notification_policy": "GEN2",
        },
        "expires_at": str(pd.Timestamp(utc_now().isoformat()).tz_localize(None) + pd.Timedelta(seconds=60)),
    }


def gates_for_live(candidate_id: str, decision: dict[str, Any], allowed_symbols: list[str], bridge_status: dict[str, Any] | None) -> tuple[bool, str]:
    if decision.get("decision") == "NO_DECISION":
        return False, str(decision.get("reason"))
    if decision.get("hypothetical_action") != "CANDIDATE_SHORT":
        return False, "VETOED_OR_ABSTAIN"
    if decision["symbol"] not in allowed_symbols:
        return False, "SYMBOL_NOT_ALLOWED"
    # Autonomous authorization: no arm token. A running watcher + a coherent
    # config + kill switch off IS the authorization (PM2 is the arm mechanism).
    # execution_enabled is checked at the cycle level; risk_gate covers caps/kill.
    ok, reason = oc.risk_gate(candidate_id)
    if not ok:
        return False, reason
    if bridge_status is None:
        return False, "BRIDGE_STATUS_UNAVAILABLE"
    if bridge_status.get("phase_o_allow_orders") is not False:
        return False, "PHASE_O_NOT_PAUSED_PER_BRIDGE"
    if bridge_status.get("gen2_enabled") is not True:
        return False, "BRIDGE_GEN2_DISABLED"
    # NOTE: the max-concurrent-positions cap is enforced in the execution phase
    # of run_cycle (count-based, against the live open-position count), not here.
    # gates_for_live now answers only "is this a valid, armed opportunity?".
    return True, "ELIGIBLE"


def run_cycle(candidate_id: str = DEFAULT_CANDIDATE_ID,
              trrm_dir: Path | None = None, eqm_dir: Path | None = None,
              live_enabled: bool = False,
              snapshot_fn: Any = None, status_fn: Any = None, execute_fn: Any = None,
              filters_fn: Any = None, selection_threshold: float | None = None) -> dict[str, Any]:
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

    # Autonomous authorization: the operational config (gen2_config.yaml) is the
    # single source of truth; a running watcher IS the arm. No arm token.
    contract = None
    try:
        contract = oc.load_contract(candidate_id)
    except Exception:
        pass

    # Frozen selection threshold — part of the scientific freeze. Verified against
    # the freeze hashes on every load (cheap; no recompute). A missing or drifted
    # policy is a scientific-integrity hard stop, exactly like a model-hash
    # mismatch: operating without it would be a different, unvalidated strategy.
    # (Injectable for tests; production always loads the frozen artifact.)
    if selection_threshold is None:
        from aegis_alpha.tools.gen2_selection_policy import load_threshold_verified

        selection_threshold = load_threshold_verified(freeze)

    decisions: list[dict[str, Any]] = []
    candidate_opps: list[dict[str, Any]] = []
    live_attempts: list[dict[str, Any]] = []
    # Symbol allowlist and the live master switch come from the config, not a token.
    allowed_symbols = list(contract.get("symbols", [])) if contract else []
    execution_enabled = bool(contract.get("execution_enabled", False)) if contract else False

    # Phase 1 — evaluate EVERY active symbol, record paper (always, independent of
    # the live path), and collect every CANDIDATE_SHORT with a finite score. These
    # are the rankable opportunities regardless of whether the live path is armed.
    def _score(d: dict[str, Any]) -> float:
        s = d.get("eqm_score")
        return s if isinstance(s, (int, float)) and s == s else float("-inf")

    for symbol in snap["symbols"]:
        raw = load_snapshot_symbol(snap_dir, symbol).rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS]
        decision = evaluate_symbol(symbol, raw, context, trrm, eqm, freeze)
        if state["last_processed"].get(symbol) == decision.get("ts"):
            continue  # dedup: one decision per closed candle
        state["last_processed"][symbol] = decision.get("ts")
        core.append_jsonl(FORWARD_ROOT / "forward_decisions.jsonl", {"collected_at": utc_now().isoformat(), **decision})
        # persist per symbol: a crash mid-cycle must not re-emit already-appended candles
        core.atomic_write(state_path, json.dumps(state, indent=2))
        decisions.append(decision)
        if decision.get("hypothetical_action") == "CANDIDATE_SHORT" and _score(decision) != float("-inf"):
            candidate_opps.append(decision)

    # Phase 2 — GLOBAL ranking by score, best first. Not iteration order, not
    # per-symbol: the whole cross-symbol candidate field competes on quality.
    candidate_opps.sort(key=_score, reverse=True)

    # Phase 3 — sequential execution over the ranked field. Apply the live gates to
    # each opportunity in quality order; submit at most max_orders_per_cycle (1),
    # never exceeding max_concurrent open positions. Everything below the top is
    # DEFERRED and re-evaluated fresh next cycle, so the system always opens the
    # best ELIGIBLE opportunity available at that instant — never a batch of
    # mediocre ones, never the next-in-list just because it was queued.
    max_concurrent = int(contract.get("max_concurrent_positions", 1)) if contract else 1
    max_per_cycle = int(contract.get("max_orders_per_cycle", 1)) if contract else 1
    open_count = len(bridge_status.get("open_positions") or []) if bridge_status else 0
    submitted_this_cycle = 0
    for rank_idx, decision in enumerate(candidate_opps):
        symbol = decision["symbol"]
        attempt: dict[str, Any] = {"symbol": symbol, "ts": decision.get("ts"),
                                   "rank": rank_idx + 1, "score": decision.get("eqm_score")}
        # FROZEN SELECTION GATE (scientific, applies in paper and live): the best
        # candidate must clear the frozen top-decile threshold or it is an
        # abstention. Ranked descending, so once one is below, all below it are too.
        if _score(decision) < selection_threshold:
            attempt["eligible"] = False
            attempt["reason"] = "BELOW_FROZEN_EQM_THRESHOLD"
            live_attempts.append(attempt)
            continue
        eligible, reason = gates_for_live(candidate_id, decision, allowed_symbols, bridge_status)
        attempt["eligible"] = eligible
        if not eligible:
            attempt["reason"] = reason
            live_attempts.append(attempt)
            continue
        # live_enabled = the runner enabled the submission path; execution_enabled
        # = the config master switch. Both must be on to place a real order.
        if not (live_enabled and contract is not None):
            attempt["reason"] = "WOULD_SUBMIT_LIVE_DISABLED"
            live_attempts.append(attempt)
            continue
        if not execution_enabled:
            attempt["reason"] = "EXECUTION_DISABLED_IN_CONFIG"
            live_attempts.append(attempt)
            continue
        if submitted_this_cycle >= max_per_cycle:
            attempt["reason"] = "DEFERRED_ONE_ORDER_PER_CYCLE"
            live_attempts.append(attempt)
            continue
        if open_count + submitted_this_cycle >= max_concurrent:
            attempt["reason"] = "MAX_CONCURRENT_POSITIONS_REACHED"
            live_attempts.append(attempt)
            continue
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
            attempt["reason"] = sizing["reason"]  # not an order attempt -> next-best may still submit
            live_attempts.append(attempt)
            continue
        order = build_decision_order(decision, sizing, contract, tick_size=filters.get("tick_size"))
        # Fail-closed ordering: record the order against the daily cap BEFORE
        # wiring it. A crash between submit and record would otherwise let the
        # DAILY_ORDER_CAP under-count and permit an extra order.
        core.record_order_opened(candidate_id)  # daily order counter -> DAILY_ORDER_CAP gate
        submitted_this_cycle += 1  # this cycle's single-order budget is now spent
        ack = (execute_fn or bridge.post_execute)(order)
        attempt["ack"] = ack
        attempt["reason"] = "SUBMITTED"
        core.append_jsonl(cdir / "live_orders.jsonl", {"order": order, "ack": ack})
        if ack.get("status") not in {"ACCEPTED", "ACCEPTED_DRYRUN", "DUPLICATE"}:
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                              {"type": "TOKEN_CONSUMED_WITHOUT_CONFIRMED_ACK", "ack": ack,
                               "order": order["client_order_id"],
                               "operator_action": "verify on exchange before re-arming"})
        live_attempts.append(attempt)
    core.atomic_write(state_path, json.dumps(state, indent=2))

    # Per-cycle selection outcome — always recorded (paper), documenting exactly
    # why the cycle did or did not open. Distinguishes the SCIENTIFIC abstention
    # (below the frozen threshold, all vetoed) from OPERATIONAL blocks.
    orders_submitted = sum(1 for a in live_attempts if a.get("ack", {}).get("status") == "ACCEPTED")
    best = candidate_opps[0] if candidate_opps else None
    best_score = _score(best) if best else None
    if not candidate_opps:
        no_decision_reason = "NO_CANDIDATES_ALL_TRRM_REJECTED_OR_NODECISION"
    elif best_score is not None and best_score < selection_threshold:
        no_decision_reason = "BELOW_FROZEN_EQM_THRESHOLD"
    elif orders_submitted > 0:
        no_decision_reason = None
    else:
        # best cleared the threshold but an operational gate blocked it; surface
        # that gate's reason (token/kill/daily-cap/max-concurrent/execution-disabled)
        top = next((a for a in live_attempts if a.get("rank") == 1), None)
        no_decision_reason = (top or {}).get("reason", "OPERATIONAL_BLOCK")
    selection_outcome = {
        "frozen_threshold": selection_threshold,
        "best_symbol": best["symbol"] if best else None,
        "best_score": best_score,
        "best_clears_threshold": bool(best_score is not None and best_score >= selection_threshold),
        "opened": orders_submitted > 0,
        "no_decision_reason": no_decision_reason,
    }
    core.append_jsonl(FORWARD_ROOT / "selection_outcomes.jsonl",
                      {"collected_at": utc_now().isoformat(), "candidate_id": candidate_id,
                       "snapshot": snap.get("snapshot_id"), **selection_outcome})
    summary = {
        "schema": "gen2_decision_loop_cycle_v3",
        "candidate_id": candidate_id,
        "snapshot": snap.get("snapshot_id"),
        "decisions": len(decisions),
        "candidate_short": sum(1 for d in decisions if d.get("hypothetical_action") == "CANDIDATE_SHORT"),
        "vetoed": sum(1 for d in decisions if d.get("vetoed_by_trrm")),
        "no_decision": sum(1 for d in decisions if d.get("decision") == "NO_DECISION"),
        "ranked_candidates": len(candidate_opps),
        "live_eligible": sum(1 for a in live_attempts if a.get("eligible")),
        "ranking": [{"rank": i + 1, "symbol": d["symbol"], "score": d.get("eqm_score")} for i, d in enumerate(candidate_opps)],
        "selection_outcome": selection_outcome,
        "max_concurrent_positions": max_concurrent,
        "max_orders_per_cycle": max_per_cycle,
        "open_positions_at_cycle_start": open_count,
        "live_enabled": live_enabled,
        "live_attempts": live_attempts,
        "orders_submitted": orders_submitted,
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
HARD_STOP_MARKERS = ("TRRM_HASH_MISMATCH_VS_FREEZE", "EQM_HASH_MISMATCH_VS_FREEZE", "CANDIDATE_MISMATCH",
                     "ENVIRONMENT_MISMATCH_VS_FREEZE", "SELECTION_POLICY_MISSING",
                     "SELECTION_POLICY_CANDIDATE_MISMATCH", "SELECTION_POLICY_HASH_MISMATCH",
                     "SELECTION_POLICY_INVALID_SCHEMA")
SNAPSHOT_RETENTION = 12  # forward snapshots kept on disk (~1 hour at 5-min cadence)
# Transient/connectivity incidents notify at WARNING (no exposure implied); every
# other new incident type is CRITICAL. WATCH_CYCLE_FAILED/PRUNE are notified
# separately as WARNING at the point they happen, so they are excluded here.
WARNING_INCIDENT_TYPES = {"EVENTS_PULL_FAILED", "OUTCOME_FETCH_FAILED", "BRIDGE_UNAVAILABLE_ON_SUBMIT"}
SUPPRESS_INCIDENT_TYPES = {"WATCH_CYCLE_FAILED", "WATCH_PRUNE_FAILED"}


def acquire_watch_lock(candidate_id: str) -> Path:
    """Singleton guard: two watchers would double-append decisions and race the
    risk state. Stale locks (dead pid) are taken over automatically (PM2 restart)."""
    import os

    lock = core.canary_dir(candidate_id) / "watcher.lock"
    if lock.exists():
        try:
            pid = int(json.loads(lock.read_text())["pid"])
            os.kill(pid, 0)  # raises if the pid is gone
            raise RuntimeError(f"WATCHER_ALREADY_RUNNING pid={pid}")
        except (ProcessLookupError, PermissionError, ValueError, KeyError):
            pass  # stale or unreadable -> take over
    core.atomic_write(lock, json.dumps({"pid": os.getpid(), "started_utc": utc_now().isoformat()}))
    return lock


def prune_forward_snapshots(keep: int = SNAPSHOT_RETENTION) -> int:
    """Cap disk growth: the watcher creates one snapshot per cycle (~288/day)."""
    import shutil

    root = FORWARD_ROOT / "snapshots"
    if not root.exists():
        return 0
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    removed = 0
    for old in snaps[:-keep] if keep > 0 else []:
        validate_gen2_path(old)
        shutil.rmtree(old, ignore_errors=True)
        removed += 1
    return removed


def collect_startup_state(candidate_id: str, status_fn: Any = None) -> dict[str, Any]:
    """Real system state for the Telegram startup message. Read-only, never raises."""
    state: dict[str, Any] = {}
    freeze = json.loads(core.FREEZE_PATH.read_text())
    try:
        state["contract"] = oc.load_contract(candidate_id)
    except Exception:
        state["contract"] = None
    # Autonomous: "armed" = config valid + execution enabled + kill off (PM2 alive).
    contract = state["contract"]
    killed = core.kill_switch_engaged(candidate_id)
    exec_on = bool(contract.get("execution_enabled")) if contract else False
    state["armed"] = bool(contract) and exec_on and not killed
    state["arm_reason"] = ("ARMED_VIA_CONFIG" if state["armed"]
                           else "KILL_SWITCH" if killed
                           else "EXECUTION_DISABLED_IN_CONFIG" if contract and not exec_on
                           else "NO_CONFIG")
    _, state["risk_reason"] = oc.risk_gate(candidate_id, mutate=False)
    state["kill_switch"] = killed
    state["phase_o_paused"] = core.phase_o_new_entries_paused()[0]
    state["symbols_analyzed"] = list(SYMBOLS)
    state["symbols_executable"] = list(contract.get("symbols", [])) if (contract and exec_on) else []
    try:
        state["bridge"] = (status_fn or bridge.get_status)()
    except Exception:
        state["bridge"] = None
    trrm_dir = GEN2_ROOT / "rv2" / "20260711T171832Z"
    eqm_dir = GEN2_ROOT / "eqm1" / "20260711T201456Z"
    models: dict[str, Any] = {}
    try:
        check_environment_vs_freeze(freeze)
        models["environment"] = "valid"
    except Exception as exc:
        models["environment"] = str(exc)[:120]
    try:
        trrm_ok = sha256_file(trrm_dir / "rv2_candidate.pkl") == freeze["trrm_v2_sha256"]
        eqm_ok = sha256_file(eqm_dir / "eqm1_candidate.pkl") == freeze["eqm1_sha256"]
        models.update({"trrm": "hash-verified" if trrm_ok else "HASH MISMATCH",
                       "qmae": "hash-verified" if trrm_ok else "HASH MISMATCH",  # QMAE lives in the TRRM bundle
                       "eqm": "hash-verified" if eqm_ok else "HASH MISMATCH",
                       "freeze_valid": trrm_ok and eqm_ok})
    except Exception as exc:
        models.update({"freeze_valid": False, "trrm": f"unreadable: {exc}"[:80]})
    state["models"] = models
    cdir = core.canary_dir(candidate_id)

    def _count(path: Path) -> int:
        return sum(1 for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()) if path.exists() else 0

    def _rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        for l in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if l.strip():
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
        return out

    # live_orders.jsonl mixes legacy dry-run attempts and decision-loop submissions;
    # separate the four honestly (never count a dry-run as a real order).
    lo = _rows(cdir / "live_orders.jsonl")
    real_submissions = sum(1 for r in lo if (r.get("ack") or {}).get("status") == "ACCEPTED")
    dryrun_requests = sum(1 for r in lo if (r.get("ack") or {}).get("status") == "ACCEPTED_DRYRUN"
                          or r.get("dry_run") is True or r.get("order_action") in ("NO_ORDER", "SUBMIT_ORDER"))
    real_fills = sum(1 for r in _rows(cdir / "fills.jsonl") if r.get("type") in (None, "FILL"))
    state["evidence"] = {"paper_decisions": _count(FORWARD_ROOT / "forward_decisions.jsonl"),
                         "outcomes": _count(cdir / "forward_outcomes.jsonl"),
                         "dryrun_requests": dryrun_requests,
                         "real_order_submissions": real_submissions,
                         "real_fills": real_fills,
                         "incidents": _count(cdir / "incidents" / "incidents.jsonl")}
    return state


def startup_gauntlet(candidate_id: str, status_fn: Any = None) -> tuple[bool, list[str]]:
    """Automatable safety gauntlet run at watcher start. Returns (ok, failures).
    Used to decide whether a persisted kill switch may clear on `pm2 restart`:
    the kill clears ONLY if this passes — so a restart can never resume into a
    broken/unsafe state (drifted science, open position, Phase O active)."""
    failures: list[str] = []
    freeze = json.loads(core.FREEZE_PATH.read_text())
    # science: environment + model hashes + selection policy vs freeze
    try:
        check_environment_vs_freeze(freeze)
    except Exception as exc:
        failures.append(str(exc)[:80])
    try:
        trrm_dir = GEN2_ROOT / "rv2" / "20260711T171832Z"
        eqm_dir = GEN2_ROOT / "eqm1" / "20260711T201456Z"
        if sha256_file(trrm_dir / "rv2_candidate.pkl") != freeze["trrm_v2_sha256"]:
            failures.append("TRRM_HASH_MISMATCH_VS_FREEZE")
        if sha256_file(eqm_dir / "eqm1_candidate.pkl") != freeze["eqm1_sha256"]:
            failures.append("EQM_HASH_MISMATCH_VS_FREEZE")
    except Exception as exc:
        failures.append(f"MODEL_HASH_UNREADABLE:{exc}"[:80])
    try:
        from aegis_alpha.tools.gen2_selection_policy import load_threshold_verified

        load_threshold_verified(freeze)
    except Exception as exc:
        failures.append(str(exc)[:80])
    # config coherent
    try:
        oc.load_contract(candidate_id)
    except Exception as exc:
        failures.append(f"CONFIG:{exc}"[:80])
    # Phase O paused
    if not core.phase_o_new_entries_paused()[0]:
        failures.append("PHASE_O_NOT_PAUSED")
    # bridge reachable + NO open position (a kill from an orphan/unprotected
    # position must not clear while the position still exists)
    try:
        st = (status_fn or bridge.get_status)()
        if st.get("open_positions"):
            failures.append("OPEN_POSITION_PRESENT")
    except Exception:
        pass  # bridge unreachable at boot is not itself a reason to keep a kill
    return (len(failures) == 0, failures)


def startup_arm(candidate_id: str, status_fn: Any = None, notify_fn: Any = None) -> dict[str, Any]:
    """PM2 = arm. On watcher start: write the startup audit, and if a kill switch
    is engaged, clear it ONLY if the gauntlet passes (smart re-arm)."""
    import aegis_alpha.tools.gen2_config as cfg

    audit = None
    try:
        audit = cfg.write_startup_audit(candidate_id)
        cfg.snapshot_for_change_tracking(candidate_id)
    except Exception as exc:
        core.append_jsonl(core.canary_dir(candidate_id) / "incidents" / "incidents.jsonl",
                          {"type": "STARTUP_CONFIG_INVALID", "error": str(exc)[:160]})
    ok, failures = startup_gauntlet(candidate_id, status_fn)
    result: dict[str, Any] = {"gauntlet_ok": ok, "failures": failures, "kill_cleared": False, "audit": audit}
    if core.kill_switch_engaged(candidate_id):
        if ok:
            kpath = core.canary_dir(candidate_id) / "KILL_SWITCH"
            kpath.unlink(missing_ok=True)
            result["kill_cleared"] = True
            core.append_jsonl(core.canary_dir(candidate_id) / "incidents" / "incidents.jsonl",
                              {"type": "KILL_SWITCH_CLEARED_ON_RESTART", "gauntlet": "PASSED"})
            if notify_fn is not None:
                notify_fn(candidate_id, "INFO", "Kill switch limpiado en restart (gauntlet OK)",
                          "El gauntlet de arranque pasó limpio; el kill switch se limpió y el sistema queda armado.",
                          fingerprint=f"kill-cleared-{utc_now().isoformat()}")
        elif notify_fn is not None:
            notify_fn(candidate_id, "CRITICAL", "Kill switch NO se limpió en restart",
                      f"El gauntlet falló: {', '.join(failures)}. El sistema queda DESARMADO hasta resolver la causa.",
                      fingerprint=f"kill-held-{'|'.join(failures)}")
    return result


def run_watch(candidate_id: str = DEFAULT_CANDIDATE_ID, interval_seconds: float = 300.0,
              max_cycles: int | None = None, live_enabled: bool = False,
              cycle_fn: Any = None, pull_fn: Any = None, resolve_fn: Any = None,
              sleep_fn: Any = time.sleep, notify_fn: Any = None,
              arm_on_start: bool = True) -> dict[str, Any]:
    """Autonomous forward-evidence runner: events -> decisions -> outcomes -> heartbeat.

    Paper evidence collection never stops for live-path problems: the kill
    switch / risk gates silently block submissions while decisions and
    outcomes keep accruing. Only a scientific-integrity failure (hash or
    candidate mismatch) aborts the watcher.
    """
    cdir = core.canary_dir(candidate_id)
    lock = acquire_watch_lock(candidate_id)
    counters = {"cycles": 0, "cycle_errors": 0, "resolve_errors": 0, "decisions": 0,
                "orders_submitted": 0, "outcomes_resolved": 0, "events_ingested": 0,
                "snapshots_pruned": 0}
    heartbeat: dict[str, Any] = {}
    prev_kill = core.kill_switch_engaged(candidate_id)
    incidents_path = cdir / "incidents" / "incidents.jsonl"

    def _incident_count() -> int:
        return sum(1 for l in incidents_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()) if incidents_path.exists() else 0

    prev_incidents = _incident_count()
    # PM2 = arm: startup audit + smart (gauntlet-gated) kill clear.
    if arm_on_start:
        try:
            startup_arm(candidate_id, notify_fn=notify_fn)
        except Exception as exc:
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl", {"type": "STARTUP_ARM_FAILED", "error": repr(exc)})
    prev_kill = core.kill_switch_engaged(candidate_id)  # re-read after a possible clear
    import aegis_alpha.tools.gen2_config as _cfg

    prev_config_checksum = _cfg.config_checksum()
    if notify_fn is not None:
        try:
            import aegis_alpha.tools.gen2_telegram as tg

            startup_state = collect_startup_state(candidate_id)
            notify_fn(candidate_id, "INFO", "GEN2 CANARY ONLINE",
                      tg.build_startup_message(candidate_id, startup_state),
                      fingerprint=f"startup-{utc_now().isoformat()}")
        except Exception:
            pass
    while max_cycles is None or counters["cycles"] < max_cycles:
        cycle_started = utc_now().isoformat()
        # detect a live config edit (CONFIGURATION_CHANGED) before acting on it
        try:
            change = _cfg.detect_config_change(prev_config_checksum, candidate_id)
            if change is not None:
                prev_config_checksum = change["new_checksum"]
                if notify_fn is not None:
                    sev = "WARNING" if change["new_config_valid"] else "CRITICAL"
                    body = ("cambios: " + json.dumps(change["diff"])[:400]) if change["new_config_valid"] \
                        else f"CONFIG INVÁLIDA, se mantiene la anterior: {change['invalid_reason']}"
                    notify_fn(candidate_id, sev, "CONFIGURATION_CHANGED", body,
                              fingerprint=f"cfgchg-{change['new_checksum'][:12]}")
        except Exception:
            pass
        if live_enabled:
            ev = (pull_fn or pull_events)(candidate_id)
            counters["events_ingested"] += int(ev.get("ingested", 0))
        try:
            summary = (cycle_fn or run_cycle)(candidate_id, live_enabled=live_enabled)
            counters["decisions"] += int(summary.get("decisions", 0))
            new_orders = int(summary.get("orders_submitted", 0))
            counters["orders_submitted"] += new_orders
            if new_orders and notify_fn is not None:
                acks = [a.get("ack", {}) for a in summary.get("live_attempts", []) if a.get("ack")]
                notify_fn(candidate_id, "INFO", f"ORDEN ENVIADA ({new_orders})",
                          json.dumps(acks)[:600], fingerprint=f"order-{acks[0].get('client_order_id', counters['cycles'])}" if acks else None)
        except Exception as exc:
            if any(m in str(exc) for m in HARD_STOP_MARKERS):
                core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                                  {"type": "WATCH_HARD_STOP_INTEGRITY", "error": repr(exc)})
                if notify_fn is not None:
                    notify_fn(candidate_id, "CRITICAL", "WATCHER HARD STOP — integridad científica",
                              repr(exc)[:400], fingerprint="hard-stop")
                raise
            counters["cycle_errors"] += 1
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                              {"type": "WATCH_CYCLE_FAILED", "error": repr(exc)})
            if notify_fn is not None:
                notify_fn(candidate_id, "WARNING", "Ciclo del watcher falló (continúo)",
                          repr(exc)[:300], fingerprint=f"cycle-fail-{type(exc).__name__}")
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
        try:
            counters["snapshots_pruned"] += prune_forward_snapshots()
        except Exception as exc:
            core.append_jsonl(cdir / "incidents" / "incidents.jsonl",
                              {"type": "WATCH_PRUNE_FAILED", "error": repr(exc)})
        if notify_fn is not None:
            kill_now = core.kill_switch_engaged(candidate_id)
            if kill_now and not prev_kill:
                notify_fn(candidate_id, "CRITICAL", "KILL SWITCH ENGANCHADO",
                          "El kill switch se activó durante este ciclo. Live bloqueado; paper continúa. Revisar incidentes y exchange.",
                          fingerprint="kill-engaged")
            prev_kill = kill_now
            # Soft consecutive-loss signal (EXPERIMENTAL_CONTINUOUS): alert but do
            # not halt — capital caps still bind. Hard modes are blocked in risk_gate.
            try:
                if contract is not None and contract.get("consecutive_loss_policy") == "soft":
                    streak = int(json.loads((cdir / "risk_state.json").read_text()).get("consecutive_losses", 0))
                    if streak >= 3:
                        notify_fn(candidate_id, "WARNING", f"Racha de {streak} pérdidas (modo continuo)",
                                  "Continúo operando (política soft). Caps de pérdida diaria/total siguen activos.",
                                  fingerprint=f"loss-streak-{streak}")
            except Exception:
                pass
            inc_now = _incident_count()
            if inc_now > prev_incidents:
                tail = incidents_path.read_text(encoding="utf-8", errors="replace").splitlines()[-(inc_now - prev_incidents):]
                new_types = {json.loads(l).get("type", "?") for l in tail if l.strip()} - SUPPRESS_INCIDENT_TYPES
                warn_types = sorted(new_types & WARNING_INCIDENT_TYPES)
                crit_types = sorted(new_types - WARNING_INCIDENT_TYPES)
                if crit_types:
                    notify_fn(candidate_id, "CRITICAL", f"Incidentes críticos nuevos ({len(crit_types)})",
                              ", ".join(crit_types)[:400], fingerprint=f"incidents-crit-{'|'.join(crit_types)}")
                if warn_types:
                    notify_fn(candidate_id, "WARNING", "Incidentes de conectividad",
                              ", ".join(warn_types)[:400], fingerprint=f"incidents-warn-{'|'.join(warn_types)}")
            prev_incidents = inc_now
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
    lock.unlink(missing_ok=True)
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
    p.add_argument("--no-telegram", action="store_true", help="disable Telegram notifications")
    p.add_argument("--dry-run-cycle", action="store_true",
                   help="run ONE paper cycle and print ranking/scores/threshold/winner/decision (no orders)")
    args = p.parse_args(argv)
    if args.dry_run_cycle:
        summary = run_cycle(args.candidate_id, live_enabled=False)
        so = summary["selection_outcome"]
        print("=== GEN2 DRY-RUN CYCLE (paper, no orders) ===")
        print(f"snapshot: {summary['snapshot']} | símbolos analizados: {summary['decisions']} | "
              f"vetados TRRM: {summary['vetoed']} | candidatos: {summary['ranked_candidates']}")
        print(f"FROZEN THRESHOLD: {so['frozen_threshold']:.8f}")
        print("ranking global (EQM score, mejor primero):")
        for r in summary["ranking"]:
            mark = "✓" if (r.get("score") is not None and r["score"] >= so["frozen_threshold"]) else "·"
            print(f"  {mark} #{r['rank']:2} {r['symbol']:9} {r['score']:+.6f}")
        print(f"ganador: {so['best_symbol']} score={so['best_score']} | supera umbral: {so['best_clears_threshold']}")
        print(f"DECISIÓN: {'ABRIRÍA (live)' if so['best_clears_threshold'] else 'NO_DECISION'} | "
              f"motivo: {so['no_decision_reason'] or 'OPEN_CANDIDATE'}")
        print("(dry-run: ninguna orden enviada)")
        return 0
    if args.watch:
        notify_fn = None
        if not args.no_telegram:
            from aegis_alpha.tools.gen2_telegram import notify as notify_fn  # noqa: F811
        out: dict[str, Any] = {"watch": run_watch(args.candidate_id, args.interval_seconds, args.max_cycles,
                                                  live_enabled=args.live, notify_fn=notify_fn)}
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
