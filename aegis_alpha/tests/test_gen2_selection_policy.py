#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_selection_policy as sp

FREEZE = {
    "candidate_id": "gen2-test",
    "d3_dataset_sha256": "DS", "trrm_v2_sha256": "TR", "eqm1_sha256": "EQ", "feature_hash": "FH",
    "code_commit": "abc123",
    "veto": {"threshold_full_dev_informational": 0.1017},
}


def frozen_policy(threshold: float = 0.0143) -> dict:
    return {
        "schema": "gen2_selection_policy_v1", "candidate_id": "gen2-test",
        "training_commit": "abc123", "feature_hash": "FH",
        "method": "x", "created_at_utc": "t", "environment": {}, "explanation": "x",
        "threshold_value": threshold, "percentile": 0.90, "veto_threshold": 0.1017,
        "score_kind": "reg_component", "dev_rows_h12": 49313, "survivors": 32405, "survivor_fraction": 0.657,
        "d3_dataset_sha256": "DS", "trrm_v2_sha256": "TR", "eqm1_sha256": "EQ",
    }


def setup(tmp: Path, freeze: dict | None = None, policy: dict | None = None) -> None:
    sp.FREEZE_PATH = tmp / "freeze.json"
    sp.POLICY_PATH = tmp / "policy.json"
    sp.FREEZE_PATH.write_text(json.dumps(freeze or FREEZE))
    if policy is not None:
        sp.POLICY_PATH.write_text(json.dumps(policy))


def test_load_threshold_verified_ok() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), policy=frozen_policy(0.0143))
        assert abs(sp.load_threshold_verified() - 0.0143) < 1e-12


def test_missing_policy_is_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))  # no policy written
        try:
            sp.load_threshold_verified()
            raise AssertionError("must fail closed when policy missing")
        except FileNotFoundError as e:
            assert "SELECTION_POLICY_MISSING" in str(e)


def test_hash_mismatch_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as t:
        pol = frozen_policy()
        pol["eqm1_sha256"] = "DIFFERENT"  # a swapped model
        setup(Path(t), policy=pol)
        try:
            sp.load_threshold_verified()
            raise AssertionError("must reject a policy bound to a different EQM")
        except ValueError as e:
            assert "SELECTION_POLICY_HASH_MISMATCH" in str(e)


def test_candidate_mismatch_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as t:
        pol = frozen_policy()
        pol["candidate_id"] = "gen2-other"
        setup(Path(t), policy=pol)
        try:
            sp.load_threshold_verified()
            raise AssertionError("must reject a policy for a different candidate")
        except ValueError as e:
            assert "SELECTION_POLICY_CANDIDATE_MISMATCH" in str(e)


def test_dataset_mismatch_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as t:
        freeze = {**FREEZE, "d3_dataset_sha256": "OTHER_DATASET"}
        setup(Path(t), freeze=freeze, policy=frozen_policy())  # policy still says DS
        try:
            sp.load_threshold_verified()
            raise AssertionError("must reject a policy computed on a different dataset")
        except ValueError as e:
            assert "SELECTION_POLICY_HASH_MISMATCH" in str(e)


def test_threshold_is_top_decile_of_survivors() -> None:
    # Unit-check the definition itself: p90 over the veto survivors = top-decile cutoff.
    rng = np.random.default_rng(0)
    s_eqm = rng.normal(0, 0.05, 10000)
    tail = rng.uniform(0, 0.2, 10000)
    survivors = tail < 0.1017
    thr = float(np.quantile(s_eqm[survivors], 0.90))
    frac_above = (s_eqm[survivors] >= thr).mean()
    assert abs(frac_above - 0.10) < 0.01  # ~top 10% of survivors clear it


if __name__ == "__main__":
    test_load_threshold_verified_ok()
    test_missing_policy_is_hard_stop()
    test_hash_mismatch_hard_stop()
    test_candidate_mismatch_hard_stop()
    test_dataset_mismatch_hard_stop()
    test_threshold_is_top_decile_of_survivors()
    print("test_gen2_selection_policy: OK")
