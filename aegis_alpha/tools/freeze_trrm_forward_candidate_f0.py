#!/usr/bin/env python3
"""FASE-F0 freeze TRRM champion for passive forward research."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.trrm_forward_common_f0 import (
    CHAMPION_BUDGET,
    CHAMPION_ENGINE,
    CHAMPION_POLICY,
    CHAMPION_WINDOW_DAYS,
    DEFAULT_SIGNAL_GLOB,
    DIAGNOSTIC_HORIZONS,
    EXPECTED_FEATURE_HASH,
    PRIMARY_HORIZON,
    RESEARCH_OUTPUT_ROOT,
    SCHEMA_VERSION,
    atomic_write_text,
    compact_utc_stamp,
    combined_model_hash,
    guard_no_enforcement_imports,
    inspect_turbo_signal_source,
    load_json,
    load_pipeline_checked,
    make_history_seed,
    model_file_hashes,
    replay_e21_engine,
    safe_research_path,
    schema_document,
    sha256_file,
    sha256_json,
    utc_now,
    validate_champion_artifacts,
    write_json,
)
from aegis_alpha.tools.train_trrm_honest_e2 import MedianImputer, StandardScalerLite

DEFAULT_E2_JSON = Path("/home/jasan/Develop/aegis_phase_e2_trrm_honest_20260710T173714Z.json")
DEFAULT_E21_JSON = Path("/home/jasan/Develop/aegis_phase_e21_trrm_calibration_20260710T183052Z.json")
DEFAULT_FABLE_JSON = Path("/home/jasan/Develop/aegis_fable_trrm_policy_metrics_a_20260710T215425Z.json")
DEFAULT_MODEL_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_POLICY_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e21/20260710T183052Z")
DEFAULT_INTERNAL = Path("/home/jasan/Develop/aegis_phase_e21_internal_predictions_20260710T183052Z.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze FASE-F0 TRRM research candidate")
    p.add_argument("--e2-report-json", default=str(DEFAULT_E2_JSON))
    p.add_argument("--e21-report-json", default=str(DEFAULT_E21_JSON))
    p.add_argument("--fable-audit-json", default=str(DEFAULT_FABLE_JSON))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--policy-dir", default=str(DEFAULT_POLICY_DIR))
    p.add_argument("--internal-predictions", default=str(DEFAULT_INTERNAL))
    p.add_argument("--output-root", default=str(RESEARCH_OUTPUT_ROOT))
    p.add_argument("--candidate-id", default="")
    p.add_argument("--freeze-time", default="")
    p.add_argument("--target", default="target.tail_risk_roe_030")
    p.add_argument("--feature-hash", default=EXPECTED_FEATURE_HASH)
    p.add_argument("--budget", type=float, default=CHAMPION_BUDGET)
    p.add_argument("--rolling-window-days", type=int, default=CHAMPION_WINDOW_DAYS)
    p.add_argument("--primary-horizon", type=int, default=PRIMARY_HORIZON)
    p.add_argument("--diagnostic-horizons", nargs="+", type=int, default=DIAGNOSTIC_HORIZONS)
    p.add_argument("--write-artifacts", default="true")
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def git_identity() -> dict[str, str]:
    return {
        "local_user_name": git_value(["config", "--local", "user.name"]),
        "local_user_email": git_value(["config", "--local", "user.email"]),
        "author_ident": git_value(["var", "GIT_AUTHOR_IDENT"]),
        "committer_ident": git_value(["var", "GIT_COMMITTER_IDENT"]),
    }


def write_readme(path: Path, candidate_id: str) -> None:
    text = f"""# TRRM F0 Forward Research Candidate

Candidate: `{candidate_id}`

This directory is research-only. It records passive scores and hypothetical
decisions only. It does not block entries, change sizing, change SL/TP, change
guards, send orders, or influence execution.

Primary decision horizon: H12. H6 and H24 are diagnostic-only.

`NO_DECISION` means the research collector could not produce a valid
hypothetical decision. It is not RETAIN, not REJECT, and not an execution
instruction.

Forward labels and outcomes are not read during F0:
`FORWARD_OUTCOMES_NOT_EVALUATED`.

Manual collection command:

```bash
/home/jasan/.venv_rocm62/bin/python aegis_alpha/tools/collect_trrm_forward_scores_f0.py --candidate-dir {path}
```
"""
    atomic_write_text(path / "README.md", text)


def manifest_conflict(candidate_dir: Path, manifest: dict[str, Any]) -> tuple[bool, str]:
    path = candidate_dir / "candidate_manifest.json"
    if not path.exists():
        return False, ""
    existing = load_json(path)
    def stable(obj: dict[str, Any]) -> dict[str, Any]:
        out = dict(obj)
        out.pop("history_seed", None)
        out.pop("git_identity", None)
        return out
    if sha256_json(stable(existing)) == sha256_json(stable(manifest)):
        return False, "existing_identical"
    return True, "FREEZE_MANIFEST_CONFLICT"


def run_freeze(args: argparse.Namespace) -> dict[str, Any]:
    guard_no_enforcement_imports(Path(__file__))
    freeze_time = args.freeze_time or utc_now()
    candidate_id = args.candidate_id or f"trrm_e21_f0_{compact_utc_stamp()}"
    candidate_dir = Path(args.output_root) / candidate_id
    safe_research_path(candidate_dir)
    e2 = load_json(Path(args.e2_report_json))
    e21 = load_json(Path(args.e21_report_json))
    fable = load_json(Path(args.fable_audit_json))
    pipeline = load_pipeline_checked(Path(args.model_dir))
    integrity = validate_champion_artifacts(e2, e21, fable, pipeline, args)
    if integrity["status"] != "OK":
        raise ValueError("ARTIFACT_INTEGRITY_ERROR")
    dense_csv = Path(e21.get("paths", {}).get("dense_csv", "/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv"))
    replay = replay_e21_engine(dense_csv, Path(args.internal_predictions), pipeline, args.budget, args.rolling_window_days)
    if replay["status"] != "OK":
        raise ValueError("ENGINE_REPLAY_MISMATCH")
    source = inspect_turbo_signal_source(DEFAULT_SIGNAL_GLOB)
    model_hashes = model_file_hashes(Path(args.model_dir))
    model_hash = combined_model_hash(Path(args.model_dir))
    engine_file = REPO / "aegis_alpha/tools/calibrate_trrm_operating_point_e21.py"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "frozen_at_utc": freeze_time,
        "branch": git_value(["branch", "--show-current"]),
        "repository_commit": git_value(["rev-parse", "HEAD"]),
        "commit_c610340_present": subprocess.call(["git", "merge-base", "--is-ancestor", "c610340", "HEAD"], cwd=REPO) == 0,
        "model_path": str(Path(args.model_dir)),
        "model_file_hashes": model_hashes,
        "model_hash": model_hash,
        "model_class": "RandomForestClassifier",
        "feature_hash": args.feature_hash,
        "feature_list": integrity["feature_list"],
        "target": args.target,
        "policy_method": CHAMPION_POLICY,
        "budget": args.budget,
        "rolling_window_days": args.rolling_window_days,
        "engine_name": CHAMPION_ENGINE,
        "engine_source_file": str(engine_file),
        "engine_source_sha256": sha256_file(engine_file),
        "primary_horizon": args.primary_horizon,
        "diagnostic_horizons": args.diagnostic_horizons,
        "opportunity_semantics": source,
        "source_type": source["source_kind"],
        "fallback_no_decision_behavior": {
            "missing_h12": "NO_DECISION",
            "insufficient_history": "NO_DECISION",
            "h6_h24_never_control_primary_decision": True,
            "enforcement_action": "NONE",
        },
        "expected_schema": "schema.json",
        "opened_lockbox_status": {
            "already_opened": True,
            "used_for_selection": False,
            "diagnostic_only": True,
        },
        "research_only": True,
        "enforcement_enabled": False,
        "labels_enabled": False,
        "maturity_criteria_future_only": {
            "minimum_tail_positive_events": 50,
            "preferred_tail_positive_events": 100,
            "minimum_calendar_days": 60,
            "preferred_calendar_days": 90,
            "minimum_regime_count": 2,
        },
        "git_identity": git_identity(),
        "artifact_integrity": integrity,
        "engine_replay": replay,
    }
    conflict, reason = manifest_conflict(candidate_dir, manifest)
    if conflict:
        raise ValueError(reason)
    if parse_bool(args.write_artifacts):
        candidate_dir.mkdir(parents=True, exist_ok=True)
        schema = schema_document()
        write_json(candidate_dir / "schema.json", schema)
        if not (candidate_dir / "history_seed.jsonl").exists():
            seed = make_history_seed(Path(args.internal_predictions), freeze_time, candidate_dir / "history_seed.jsonl")
        else:
            seed = {
                "path": str(candidate_dir / "history_seed.jsonl"),
                "sha256": sha256_file(candidate_dir / "history_seed.jsonl"),
                "labels_present": False,
            }
        manifest["history_seed"] = seed
        write_json(candidate_dir / "candidate_manifest.json", manifest)
        atomic_write_text(candidate_dir / "candidate_manifest.sha256", sha256_file(candidate_dir / "candidate_manifest.json") + "\n")
        atomic_write_text(candidate_dir / "history_seed.sha256", sha256_file(candidate_dir / "history_seed.jsonl") + "\n")
        for name in ("model_monitor_scores.jsonl", "opportunity_scores.jsonl"):
            p = candidate_dir / name
            if not p.exists():
                atomic_write_text(p, "")
        write_json(candidate_dir / "collection_state.json", {"candidate_id": candidate_id, "last_run": None, "updated_at_utc": freeze_time})
        write_readme(candidate_dir, candidate_id)
    payload = {
        "decision": "FROZEN",
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "frozen_at_utc": freeze_time,
        "artifact_integrity": integrity,
        "engine_replay": replay,
        "opportunity_source": source,
        "manifest_sha256": sha256_file(candidate_dir / "candidate_manifest.json") if (candidate_dir / "candidate_manifest.json").exists() else None,
        "history_seed": manifest.get("history_seed"),
        "git_identity": git_identity(),
    }
    print(json.dumps(payload, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_freeze(parse_args(argv))
        return 0
    except ValueError as exc:
        msg = str(exc)
        if "ARTIFACT_INTEGRITY_ERROR" in msg:
            print(json.dumps({"decision": "ARTIFACT_INTEGRITY_ERROR", "reason": msg}))
            return 2
        if "ENGINE_REPLAY_MISMATCH" in msg:
            print(json.dumps({"decision": "ENGINE_REPLAY_MISMATCH", "reason": msg}))
            return 3
        if "FREEZE_MANIFEST_CONFLICT" in msg:
            print(json.dumps({"decision": "FREEZE_MANIFEST_CONFLICT", "reason": msg}))
            return 4
        if "ENFORCEMENT_PATH_DETECTED" in msg:
            print(json.dumps({"decision": "ENFORCEMENT_PATH_DETECTED", "reason": msg}))
            return 5
        print(json.dumps({"decision": "RESEARCH_NOT_READY", "reason": msg}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
