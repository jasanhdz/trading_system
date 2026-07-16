#!/usr/bin/env python3
"""GEN2 frozen selection policy — the absolute EQM entry threshold.

This converts ECON1's validated top-decile RELATIVE ranking into a single
ABSOLUTE, market-independent, freeze-bound threshold, so the live canary opens
only the trades ECON1 measured (never score>0, never best-of-cycle-regardless).

Definition (reproducible, deterministic):
  threshold = quantile_{0.90}( s_eqm over TRRM-veto survivors of the frozen dev set )

where survivors = rows whose TRRM tail score < the frozen veto threshold
(threshold_full_dev_informational), and s_eqm is the frozen EQM score
(reg_component). This is exactly the top-10% cutoff ECON1 used
(budget_n = round(0.10 * keep_n)), expressed as an absolute score value.

No models, hashes, datasets or thresholds are recomputed at run time — the loop
loads the frozen value and verifies the artifact is bound to the current freeze.
`validate` recomputes it from scratch and hard-stops on any drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aegis_alpha.tools.gen2_eqm1_train as eqm  # noqa: E402
import aegis_alpha.tools.gen2_rv2_train as rv2  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, environment_info, git_commit, sha256_file, utc_now, validate_gen2_path  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import MedianImputer, score_of  # noqa: E402

FREEZE_PATH = GEN2_ROOT / "GEN2_SYSTEM_FREEZE.json"
POLICY_PATH = GEN2_ROOT / "GEN2_SELECTION_POLICY.json"
DATASET_CSV = GEN2_ROOT / "d3" / "datasets" / "d3-v1-build_a" / "dense.csv"
TRRM_DIR = GEN2_ROOT / "rv2" / "20260711T171832Z"
EQM_DIR = GEN2_ROOT / "eqm1" / "20260711T201456Z"
PERCENTILE = 0.90  # top-decile cutoff (ECON1 TOPK = 0.10)
RECOMPUTE_TOLERANCE = 1e-9


def _load_freeze() -> dict[str, Any]:
    return json.loads(FREEZE_PATH.read_text())


def compute_threshold(freeze: dict[str, Any], dataset_csv: Path = DATASET_CSV,
                      trrm_dir: Path = TRRM_DIR, eqm_dir: Path = EQM_DIR) -> dict[str, Any]:
    """Deterministically recompute the frozen threshold. Verifies every hash
    against the freeze BEFORE touching the data (fail-closed)."""
    ds_hash = sha256_file(dataset_csv)
    trrm_hash = sha256_file(trrm_dir / "rv2_candidate.pkl")
    eqm_hash = sha256_file(eqm_dir / "eqm1_candidate.pkl")
    if ds_hash != freeze["d3_dataset_sha256"]:
        raise ValueError("DATASET_HASH_MISMATCH_VS_FREEZE")
    if trrm_hash != freeze["trrm_v2_sha256"]:
        raise ValueError("TRRM_HASH_MISMATCH_VS_FREEZE")
    if eqm_hash != freeze["eqm1_sha256"]:
        raise ValueError("EQM_HASH_MISMATCH_VS_FREEZE")

    dev_all, _, feature_cols, _ = rv2.load_development(dataset_csv)
    dev = dev_all[pd.to_numeric(dev_all["id.horizon"], errors="coerce") == eqm.PRIMARY_HORIZON].reset_index(drop=True)
    trrm = eqm.load_trrm(trrm_dir)
    tail = eqm.tail_scores(trrm, dev)
    sys.modules["__main__"].MedianImputer = MedianImputer
    with (eqm_dir / "eqm1_candidate.pkl").open("rb") as f:
        bundle = pickle.load(f)
    x = bundle["imputer"].transform(dev[feature_cols].apply(pd.to_numeric, errors="coerce"))
    s_reg = np.asarray(bundle["reg_model"].predict(x), dtype=float)
    s_eqm = score_of(bundle["clf_model"], x) * s_reg if bundle["score_kind"] == "composite_ev" else s_reg
    veto_threshold = float(freeze["veto"]["threshold_full_dev_informational"])
    survivors = tail < veto_threshold
    n_surv = int(survivors.sum())
    if n_surv < 100:
        raise ValueError(f"TOO_FEW_SURVIVORS:{n_surv}")
    threshold = float(np.quantile(s_eqm[survivors], PERCENTILE))
    return {
        "threshold_value": threshold,
        "percentile": PERCENTILE,
        "veto_threshold": veto_threshold,
        "score_kind": bundle["score_kind"],
        "dev_rows_h12": int(len(dev)),
        "survivors": n_surv,
        "survivor_fraction": round(n_surv / len(dev), 6),
        "d3_dataset_sha256": ds_hash,
        "trrm_v2_sha256": trrm_hash,
        "eqm1_sha256": eqm_hash,
    }


def freeze_policy(force: bool = False) -> dict[str, Any]:
    freeze = _load_freeze()
    if POLICY_PATH.exists() and not force:
        raise FileExistsError(f"selection policy already frozen (immutability): {POLICY_PATH}")
    computed = compute_threshold(freeze)
    policy = {
        "schema": "gen2_selection_policy_v1",
        "candidate_id": freeze["candidate_id"],
        "training_commit": freeze.get("code_commit"),
        "feature_hash": freeze["feature_hash"],
        "method": "absolute EQM score cutoff = quantile_0.90(s_eqm | TRRM-veto survivors of frozen dev H12); "
                  "equivalent to ECON1 top-decile (TOPK=0.10) as a single frozen number",
        "created_at_utc": utc_now().isoformat(),
        "environment": environment_info(),
        "explanation": (
            "Live entry rule: rank the non-vetoed candidates by EQM score, take the best, and open ONLY if "
            "best_score >= threshold_value. Below it -> NO_DECISION (BELOW_FROZEN_EQM_THRESHOLD). This preserves "
            "the ECON1-validated economics: it selects the same high-score trades the backtest measured, is "
            "market-independent, and never adapts."
        ),
        **computed,
    }
    validate_gen2_path(POLICY_PATH)
    tmp = POLICY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(policy, indent=2))
    tmp.replace(POLICY_PATH)
    return policy


def load_policy() -> dict[str, Any]:
    if not POLICY_PATH.exists():
        raise FileNotFoundError("SELECTION_POLICY_MISSING")
    policy = json.loads(POLICY_PATH.read_text())
    if policy.get("schema") != "gen2_selection_policy_v1":
        raise ValueError("SELECTION_POLICY_INVALID_SCHEMA")
    return policy


def load_threshold_verified(freeze: dict[str, Any] | None = None) -> float:
    """Cheap runtime load for the loop: verify the frozen policy is bound to the
    current freeze (candidate + all hashes) WITHOUT recomputing. Hard-stops on
    any mismatch so a swapped model/dataset can never silently change entries."""
    freeze = freeze or _load_freeze()
    policy = load_policy()
    if policy["candidate_id"] != freeze["candidate_id"]:
        raise ValueError("SELECTION_POLICY_CANDIDATE_MISMATCH")
    for k in ("d3_dataset_sha256", "trrm_v2_sha256", "eqm1_sha256", "feature_hash"):
        if policy.get(k) != freeze.get(k):
            raise ValueError(f"SELECTION_POLICY_HASH_MISMATCH:{k}")
    return float(policy["threshold_value"])


def validate_policy() -> dict[str, Any]:
    """Full reproducibility check: recompute from scratch and compare to the
    frozen value. Any drift (threshold, hash, candidate) is a hard stop."""
    freeze = _load_freeze()
    policy = load_policy()
    result: dict[str, Any] = {"schema": "gen2_selection_policy_validation_v1", "candidate_id": freeze["candidate_id"]}
    problems: list[str] = []
    if policy["candidate_id"] != freeze["candidate_id"]:
        problems.append("CANDIDATE_MISMATCH")
    for k in ("d3_dataset_sha256", "trrm_v2_sha256", "eqm1_sha256", "feature_hash"):
        if policy.get(k) != freeze.get(k):
            problems.append(f"HASH_MISMATCH:{k}")
    try:
        recomputed = compute_threshold(freeze)
        drift = abs(recomputed["threshold_value"] - float(policy["threshold_value"]))
        result["recomputed_threshold"] = recomputed["threshold_value"]
        result["frozen_threshold"] = float(policy["threshold_value"])
        result["drift"] = drift
        if drift > RECOMPUTE_TOLERANCE:
            problems.append(f"THRESHOLD_DRIFT:{drift:.3e}")
    except Exception as exc:
        problems.append(f"RECOMPUTE_FAILED:{exc}")
    result["problems"] = problems
    result["status"] = "SELECTION_POLICY_VALID" if not problems else "SELECTION_POLICY_INVALID"
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GEN2 frozen selection policy")
    p.add_argument("--mode", choices=["freeze", "validate-selection-policy", "show"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.mode == "freeze":
        out = freeze_policy(force=args.force)
    elif args.mode == "show":
        out = load_policy()
    else:
        out = validate_policy()
        print(json.dumps(out, indent=2))
        return 0 if out["status"] == "SELECTION_POLICY_VALID" else 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
