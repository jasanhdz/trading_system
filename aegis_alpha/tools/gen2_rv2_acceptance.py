#!/usr/bin/env python3
"""GEN2-RV2 acceptance audit — emits the single reproducible phase decision.

Verifies training artifact integrity, evaluates hypotheses H1/H2/H3 from the
frozen spec, performs the ONE registered semi-blind lockbox query with the
frozen candidate (non-contradiction check, never selection), appends the query
to GEN2_LOCKBOX_MANIFEST, and decides GEN2_TRRM_READY / PROMISING / PARTIAL /
REJECTED.
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
from aegis_alpha.tools.gen2_d3_common import (  # noqa: E402
    LOCKBOX_MANIFEST_PATH,
    read_csv_exact,
    sha256_file,
    utc_now,
    utc_stamp,
    verify_manifest_checksums,
)
import aegis_alpha.tools.gen2_rv2_train as rv2train  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import TARGET, MedianImputer, fold_metrics, score_of  # noqa: E402


def load_candidate(artifact_dir: Path) -> dict[str, Any]:
    sys.modules["__main__"].MedianImputer = MedianImputer
    with (artifact_dir / "rv2_candidate.pkl").open("rb") as f:
        return pickle.load(f)


def semi_blind_query(artifact_dir: Path, training: dict[str, Any], lockbox_path: Path) -> dict[str, Any]:
    lockbox = json.loads(lockbox_path.read_text())
    allowed = int(lockbox.get("allowed_query_count_per_candidate", 1))
    candidate_hash = training["frozen_candidate"]["pickle_sha256"]
    already = [q for q in lockbox.get("query_log", []) if q.get("candidate_hash") == candidate_hash]
    if len(already) >= allowed:
        return {"status": "QUERY_BUDGET_EXHAUSTED", "existing": already[0]}
    bundle = load_candidate(artifact_dir)
    df = read_csv_exact(training["dataset_csv"])
    df["_ts"] = pd.to_datetime(df["id.timestamp"], errors="coerce")
    dev_end = pd.Timestamp(lockbox["historical_development_period"]["end"])
    semi = df[df["_ts"] > dev_end].reset_index(drop=True)
    for horizon in (6, 12, 24):
        semi[f"horizon_{horizon}"] = (pd.to_numeric(semi["id.horizon"], errors="coerce") == horizon).astype(float)
    x = bundle["imputer"].transform(semi[bundle["features"]].apply(pd.to_numeric, errors="coerce"))
    score = score_of(bundle["trrm_model"], x)
    y = semi[TARGET].astype(int).to_numpy()
    metrics = fold_metrics(y, score)
    qmae90 = bundle["qmae_models"]["q90"].predict(x) + bundle["qmae_q90_conformal_adjustment"]
    y_mae = pd.to_numeric(semi[rv2train.QMAE_TARGET], errors="coerce").fillna(0.0).to_numpy()
    metrics["qmae_q90_conformal_coverage"] = float((y_mae <= qmae90).mean())
    entry = {
        "opened_at_utc": utc_now().isoformat(),
        "candidate_hash": candidate_hash,
        "phase": "GEN2-RV2 acceptance",
        "purpose": "single registered non-contradiction check (never selection)",
        "window": {"start": str(semi["_ts"].min()), "end": str(semi["_ts"].max()), "rows": int(len(semi))},
    }
    lockbox["query_log"].append(entry)
    lockbox["current_query_count"] = int(lockbox.get("current_query_count", 0)) + 1
    lockbox_path.write_text(json.dumps(lockbox, indent=2), encoding="utf-8")
    return {"status": "QUERY_EXECUTED_AND_LOGGED", "metrics": metrics, "log_entry": entry}


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    artifact_dir = Path(args.artifact_dir)
    training = json.loads((artifact_dir / "training_report.json").read_text())
    integrity_ok, integrity_errors = verify_manifest_checksums(artifact_dir, "rv2_manifest")

    winner = training.get("trrm_winner")
    trrm_agg = training["trrm_results"].get(winner, {}).get("aggregate", {}) if winner else {}
    qmae_agg = training["qmae_results"]["aggregate"]
    h1 = bool(winner and trrm_agg.get("h1_all_folds_lift_ge_2x"))
    h2 = bool(winner and trrm_agg.get("h2_stability_ge_080"))
    h3 = bool(qmae_agg.get("h3_pass"))

    semi = semi_blind_query(artifact_dir, training, Path(args.lockbox_path)) if winner and not args.skip_semi_blind else {"status": "SKIPPED"}
    contradiction = False
    if semi.get("metrics"):
        pr = semi["metrics"].get("pr_auc") or 0.0
        contradiction = pr < 0.5 * trrm_agg.get("mean_pr_auc", 0.0)

    if not winner or not h1 or contradiction or not integrity_ok:
        decision = "GEN2_TRRM_REJECTED"
        reason = "no candidate rejected H0" if not winner or not h1 else ("semi-blind contradiction" if contradiction else f"artifact integrity: {integrity_errors}")
    elif h2 and h3:
        decision = "GEN2_TRRM_READY"
        reason = "H1+H2+H3 pass; semi-blind does not contradict; artifacts intact"
    elif h2 != h3 and (h2 or h3):
        failed = "H3 (QMAE coverage/pinball)" if h2 else "H2 (stability)"
        decision = "GEN2_TRRM_PROMISING"
        reason = f"H1 holds; exactly one criterion fails: {failed}"
    else:
        decision = "GEN2_TRRM_PARTIAL"
        reason = "H1 holds but both H2 and H3 fail"

    payload = {
        "schema": "gen2_rv2_acceptance_v1",
        "generated_at": stamp,
        "spec": "GEN2_RV2_SPEC.md rv2-spec-v1.0 (FROZEN)",
        "decision": decision,
        "reason": reason,
        "training_artifact": str(artifact_dir),
        "winner": winner,
        "hypotheses": {
            "H1_signal": h1,
            "H2_stability": h2,
            "H3_qmae_conformal": h3,
            "detail": {"trrm": trrm_agg, "qmae": qmae_agg},
        },
        "semi_blind": semi,
        "semi_blind_contradiction": contradiction,
        "artifact_integrity": {"ok": integrity_ok, "errors": integrity_errors},
        "next_phase": (
            "GEN2-EQM1 specification (await owner approval)" if decision in {"GEN2_TRRM_READY", "GEN2_TRRM_PROMISING"}
            else "RV2.1 targeted repair" if decision == "GEN2_TRRM_PARTIAL" else "STOP — research halt condition"
        ),
        "safety": {"no_live": True, "no_shadow": True, "no_promotion": True, "lockbox_query_registered": semi.get("status") == "QUERY_EXECUTED_AND_LOGGED", "research_only": True},
    }
    reports = artifact_dir / f"acceptance_{stamp}.json"
    reports.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    md = artifact_dir / f"acceptance_{stamp}.md"
    md.write_text("\n".join([
        "# GEN2-RV2 Acceptance",
        "",
        f"- decision: **{decision}**",
        f"- reason: {reason}",
        f"- winner: {winner}",
        f"- H1={h1} H2={h2} H3={h3}",
        f"- semi-blind: {semi.get('status')} metrics={json.dumps(semi.get('metrics'), default=json_default) if semi.get('metrics') else 'n/a'}",
        f"- next: {payload['next_phase']}",
        "",
    ]), encoding="utf-8")
    payload["outputs"] = {"json": str(reports), "md": str(md)}
    print(json.dumps({"decision": decision, "reason": reason, "winner": winner, "hypotheses": {"H1": h1, "H2": h2, "H3": h3}, "semi_blind": semi.get("status"), "md": str(md)}, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-RV2 acceptance audit")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--lockbox-path", default=str(LOCKBOX_MANIFEST_PATH))
    p.add_argument("--skip-semi-blind", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_acceptance(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
