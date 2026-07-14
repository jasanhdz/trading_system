#!/usr/bin/env python3
"""GEN2 system freeze + passive forward/paper collector (no enforcement).

Freeze: single manifest binding every hash of the approved stack. Collector:
manual, append-only, canonical final candles only, enforcement_action=NONE.
No PM2/cron/systemd. No live paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, environment_info, git_commit, sha256_file, utc_now, utc_stamp, validate_gen2_path  # noqa: E402
import aegis_alpha.tools.gen2_eqm1_train as eqm  # noqa: E402

FORWARD_ROOT = GEN2_ROOT / "forward"
FREEZE_PATH = GEN2_ROOT / "GEN2_SYSTEM_FREEZE.json"


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
    """Delegates to the decision loop's paper path (single evaluator implementation).

    The previous inline evaluator duplicated gen2_decision_loop.evaluate_symbol
    without per-candle dedup or signal_price, so running both could pollute
    forward_decisions.jsonl with double-counted candles.
    """
    from aegis_alpha.tools.gen2_decision_loop import run_cycle

    freeze = json.loads(FREEZE_PATH.read_text())
    kwargs: dict[str, Any] = {"live_enabled": False}
    if args.trrm_dir:
        kwargs["trrm_dir"] = Path(args.trrm_dir)
    if args.eqm_dir:
        kwargs["eqm_dir"] = Path(args.eqm_dir)
    summary = run_cycle(freeze["candidate_id"], **kwargs)
    return {"delegated_to": "gen2_decision_loop.run_cycle", "enforcement": "NONE", **summary}


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
