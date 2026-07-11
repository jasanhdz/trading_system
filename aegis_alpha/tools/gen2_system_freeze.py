#!/usr/bin/env python3
"""GEN2 system freeze + passive forward/paper collector (no enforcement).

Freeze: single manifest binding every hash of the approved stack. Collector:
manual, append-only, canonical final candles only, enforcement_action=NONE.
No PM2/cron/systemd. No live paths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import add_market_context, compute_causal_features  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, environment_info, git_commit, sha256_file, sha256_text, utc_now, utc_stamp, validate_gen2_path  # noqa: E402
from aegis_alpha.tools.gen2_d3_snapshot import build_snapshot, load_snapshot_symbol  # noqa: E402
import aegis_alpha.tools.gen2_eqm1_train as eqm  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import MedianImputer, score_of  # noqa: E402

FORWARD_ROOT = GEN2_ROOT / "forward"
FREEZE_PATH = GEN2_ROOT / "GEN2_SYSTEM_FREEZE.json"
SERIES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "buy_volume"]


def freeze_system(args: argparse.Namespace) -> dict[str, Any]:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"freeze exists (immutability): {FREEZE_PATH}")
    trrm_dir, eqm_dir, econ_dir = Path(args.trrm_dir), Path(args.eqm_dir), Path(args.econ_dir)
    eqm_training = json.loads((eqm_dir / "training_report.json").read_text())
    gate2 = json.loads(Path(args.gate2_json).read_text())
    if gate2.get("econ_decision") not in {"GEN2_ECONOMIC_EDGE_READY", "RULES_PLUS_TRRM_CHAMPION"}:
        raise ValueError(f"freeze not allowed: econ_decision={gate2.get('econ_decision')}")
    candidate_id = f"gen2-{utc_stamp()}"
    manifest = {
        "schema": "gen2_system_freeze_v1", "candidate_id": candidate_id, "created_at_utc": utc_now().isoformat(),
        "decision_basis": {"eqm": gate2.get("eqm_decision"), "econ": gate2.get("econ_decision")},
        "champion_kind": "eqm_plus_trrm" if gate2.get("econ_decision") == "GEN2_ECONOMIC_EDGE_READY" else "rules_plus_trrm",
        "d3_dataset_sha256": eqm_training["dataset_sha256"],
        "trrm_v2_sha256": sha256_file(trrm_dir / "rv2_candidate.pkl"),
        "eqm1_sha256": sha256_file(eqm_dir / "eqm1_candidate.pkl"),
        "econ1_report_sha256": sha256_file(econ_dir / "econ1_report.json"),
        "feature_hash": eqm_training["feature_hash"],
        "opportunity_semantics": {"primary_horizon": 12, "corr_window_min": 30, "one_opportunity_per_signal": True},
        "veto": {"budget": 0.30, "rank_based": True, "threshold_full_dev_informational": eqm_training["frozen_candidate"]["veto_threshold_full_dev"]},
        "eqm_policy": {"score_kind": eqm_training["final_score_kind"], "topk": 0.10, "ensemble_adopted": eqm_training["ensemble_adopted"]},
        "economics": {"notional": 100.0, "hold_bars": 12, "cost_scenarios": "econ1-spec-v1.0 table"},
        "fallback_semantics": {"missing_features_or_history": "NO_DECISION (fail-closed: no hypothetical entry recorded as taken)"},
        "code_commit": git_commit(), "environment": environment_info(),
        "gen1_candidate_ids_reused": False,
    }
    validate_gen2_path(FREEZE_PATH)
    FREEZE_PATH.write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return manifest


def collect_forward(args: argparse.Namespace) -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text())
    sys.modules["__main__"].MedianImputer = MedianImputer
    with (Path(args.trrm_dir) / "rv2_candidate.pkl").open("rb") as f:
        trrm = pickle.load(f)
    with (Path(args.eqm_dir) / "eqm1_candidate.pkl").open("rb") as f:
        bundle = pickle.load(f)
    assert sha256_file(Path(args.eqm_dir) / "eqm1_candidate.pkl") == freeze["eqm1_sha256"], "EQM hash mismatch vs freeze"
    assert sha256_file(Path(args.trrm_dir) / "rv2_candidate.pkl") == freeze["trrm_v2_sha256"], "TRRM hash mismatch vs freeze"
    FORWARD_ROOT.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot(symbols=list(json.loads(Path(args.symbols_json).read_text()) if args.symbols_json else
                                       ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT", "LINKUSDT"]),
                          range_days=args.warmup_days, snapshot_root=FORWARD_ROOT / "snapshots")
    snap_dir = Path(snap["directory"])
    context: dict[str, pd.DataFrame] = {}
    for ctx in ("BTCUSDT", "ETHUSDT"):
        raw = load_snapshot_symbol(snap_dir, ctx).rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS]
        context[ctx] = compute_causal_features(raw)
    decisions = []
    for symbol in snap["symbols"]:
        raw = load_snapshot_symbol(snap_dir, symbol).rename(columns={"taker_buy_base_volume": "buy_volume"})[SERIES_COLUMNS]
        feats = add_market_context(compute_causal_features(raw), context)
        last = feats.iloc[[-1]].copy()
        row = {f"feature.{c}": last.iloc[0][c] for c in last.columns if c not in {"timestamp", "open", "high", "low", "close", "volume"}}
        frame = pd.DataFrame([row])
        for h in (6, 12, 24):
            frame[f"horizon_{h}"] = 1.0 if h == 12 else 0.0
        missing = [c for c in bundle["features"] if c not in frame.columns]
        if missing:
            decision = {"symbol": symbol, "ts": str(last.iloc[0]["timestamp"]), "decision": "NO_DECISION", "reason": f"missing features {missing[:3]}"}
        else:
            x_t = trrm["imputer"].transform(frame[trrm["features"]].apply(pd.to_numeric, errors="coerce"))
            tail_raw = score_of(trrm["trrm_model"], x_t)
            tail = float(trrm["calibrator"].predict(tail_raw)[0]) if trrm.get("calibrator_kind") == "isotonic" else float(tail_raw[0])
            qmae90 = float(trrm["qmae_models"]["q90"].predict(x_t)[0] + trrm["qmae_q90_conformal_adjustment"]) if trrm.get("qmae_models") else None
            x_e = bundle["imputer"].transform(frame[bundle["features"]].apply(pd.to_numeric, errors="coerce"))
            s_reg = float(bundle["reg_model"].predict(x_e)[0])
            s_clf = float(score_of(bundle["clf_model"], x_e)[0])
            s = s_clf * s_reg if bundle["score_kind"] == "composite_ev" else s_reg
            vetoed = bool(tail >= freeze["veto"]["threshold_full_dev_informational"])
            decision = {
                "symbol": symbol, "ts": str(last.iloc[0]["timestamp"]), "candidate_id": freeze["candidate_id"],
                "tail_score": tail, "qmae_q90": qmae90, "eqm_reg": s_reg, "eqm_clf": s_clf, "eqm_score": s,
                "vetoed_by_trrm": vetoed, "hypothetical_action": "ABSTAIN" if vetoed else "CANDIDATE_SHORT",
                "enforcement_action": "NONE",
            }
        decisions.append(decision)
    log = FORWARD_ROOT / "forward_decisions.jsonl"
    with log.open("a", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps({"collected_at": utc_now().isoformat(), **d}, default=json_default) + "\n")
    return {"snapshot": snap["snapshot_id"], "decisions": len(decisions), "log": str(log),
            "vetoed": sum(1 for d in decisions if d.get("vetoed_by_trrm")), "enforcement": "NONE"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["freeze", "collect"], required=True)
    p.add_argument("--trrm-dir", default=str(eqm.DEFAULT_TRRM_DIR))
    p.add_argument("--eqm-dir", default="")
    p.add_argument("--econ-dir", default="")
    p.add_argument("--gate2-json", default="")
    p.add_argument("--warmup-days", type=int, default=15)
    p.add_argument("--symbols-json", default="")
    args = p.parse_args(argv)
    result = freeze_system(args) if args.mode == "freeze" else collect_forward(args)
    print(json.dumps(result, indent=2, default=json_default)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
