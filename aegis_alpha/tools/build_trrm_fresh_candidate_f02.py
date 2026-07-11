#!/usr/bin/env python3
"""FASE-F0.2 build a fresh TRRM forward candidate from refreshed features."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import feature_hash, score_frame
from aegis_alpha.tools.prepare_trrm_fresh_seed_f01 import git_identity, labels_in_seed_rows
from aegis_alpha.tools.train_trrm_honest_e2 import MedianImputer, StandardScalerLite, TARGET
from aegis_alpha.tools.trrm_forward_common_f0 import (
    CHAMPION_BUDGET,
    CHAMPION_ENGINE,
    CHAMPION_POLICY,
    CHAMPION_WINDOW_DAYS,
    DEFAULT_SIGNAL_GLOB,
    DIAGNOSTIC_HORIZONS,
    EXPECTED_FEATURE_HASH,
    PRIMARY_HORIZON,
    SCHEMA_VERSION,
    atomic_write_text,
    combined_model_hash,
    compact_utc_stamp,
    guard_no_enforcement_imports,
    inspect_turbo_signal_source,
    load_json,
    load_pipeline_checked,
    model_file_hashes,
    parse_dt,
    replay_e21_engine,
    safe_research_path,
    schema_document,
    sha256_file,
    utc_now,
    write_json,
)

DEFAULT_PREVIOUS = Path("/home/jasan/Develop/aegis_forward_research/trrm_f0/trrm_e21_f0_20260710T232340Z")
DEFAULT_D2 = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_E21 = Path("/home/jasan/Develop/aegis_phase_e21_trrm_calibration_20260710T183052Z.json")
DEFAULT_INTERNAL = Path("/home/jasan/Develop/aegis_phase_e21_internal_predictions_20260710T183052Z.csv")
DEFAULT_MODEL = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_OUTPUT = Path("/home/jasan/Develop/aegis_forward_research/trrm_f02")
EXPECTED_MODEL_HASH = "edf7490c18db043a52dbe2b8fa03fe42748be60240423cb8cccea21b3e630992"


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def load_seed_features(paths: list[Path], freeze_ts: pd.Timestamp, days: int) -> pd.DataFrame:
    start = freeze_ts - pd.Timedelta(days=days)
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "id.timestamp" not in df.columns:
            continue
        ts = pd.to_datetime(df["id.timestamp"], utc=True)
        part = df[(ts >= start) & (ts < freeze_ts)].copy()
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_ts"] = pd.to_datetime(out["id.timestamp"], utc=True)
    out = out.drop_duplicates(subset=["id.symbol", "id.timestamp", "id.timeframe", "id.horizon"], keep="last")
    out = out.sort_values(["_ts", "id.symbol", "id.horizon"]).drop(columns=["_ts"]).reset_index(drop=True)
    return out


def build_seed(df: pd.DataFrame, pipeline: dict[str, Any], freeze_ts: pd.Timestamp, seed_path: Path) -> dict[str, Any]:
    scores = score_frame(df, pipeline)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        ts = parse_dt(str(row["id.timestamp"]))
        if ts is None or ts >= freeze_ts:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "stream": "HISTORY_SEED",
                "market_timestamp": str(ts),
                "symbol": row["id.symbol"],
                "timeframe": row["id.timeframe"],
                "horizon": int(row["id.horizon"]),
                "score": float(scores[i]),
                "source_reference": "fresh_causal_feature_seed_f02",
            }
        )
    if labels_in_seed_rows(rows):
        raise ValueError("FORWARD_LABEL_COLUMN_DETECTED")
    atomic_write_text(seed_path, "".join(json.dumps(row, separators=(",", ":"), default=json_default) + "\n" for row in rows))
    times = [parse_dt(row["market_timestamp"]) for row in rows]
    times = [t for t in times if t is not None]
    first = min(times) if times else None
    last = max(times) if times else None
    return {
        "path": str(seed_path),
        "sha256": sha256_file(seed_path),
        "rows": len(rows),
        "history_start": str(first) if first is not None else None,
        "history_end": str(last) if last is not None else None,
        "coverage_days": float((last - first).total_seconds() / 86400.0) if first is not None and last is not None else 0.0,
        "last_age_hours": float((freeze_ts - last).total_seconds() / 3600.0) if last is not None else None,
        "symbols": sorted({str(row["symbol"]) for row in rows}),
        "horizons": sorted({int(row["horizon"]) for row in rows}),
        "labels_present": False,
    }


def write_readme(candidate_dir: Path, candidate_id: str) -> None:
    atomic_write_text(
        candidate_dir / "README.md",
        "\n".join(
            [
                "# TRRM F0.2 Fresh Forward Research Candidate",
                "",
                f"Candidate: `{candidate_id}`",
                "",
                "Research-only passive score collection. Enforcement is disabled.",
                "H12 is primary; H6/H24 are diagnostic-only.",
                "Feature store was rebuilt from fresh 5m market data and contains no labels.",
                "FORWARD_OUTCOMES_NOT_EVALUATED.",
                "",
            ]
        ),
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    guard_no_enforcement_imports(Path(__file__))
    freeze_time = args.freeze_time or utc_now()
    freeze_ts = parse_dt(freeze_time)
    if freeze_ts is None:
        raise ValueError("invalid freeze time")
    model_dir = Path(args.model_dir)
    pipeline = load_pipeline_checked(model_dir)
    features = list(pipeline["features"])
    model_hash = combined_model_hash(model_dir)
    integrity = {
        "model_hash": model_hash,
        "model_hash_exact": model_hash == args.model_hash,
        "feature_hash": feature_hash(features),
        "feature_hash_exact": feature_hash(features) == args.feature_hash == EXPECTED_FEATURE_HASH,
        "target_exact": pipeline.get("target") == TARGET == args.target,
        "policy_exact": args.policy == CHAMPION_POLICY,
        "budget_exact": abs(float(args.budget) - CHAMPION_BUDGET) < 1e-12,
        "window_exact": int(args.rolling_window_days) == CHAMPION_WINDOW_DAYS,
        "engine_exact": args.engine == CHAMPION_ENGINE,
    }
    if not all(v for k, v in integrity.items() if k.endswith("_exact")):
        raise ValueError("ARTIFACT_INTEGRITY_ERROR")
    e21 = load_json(Path(args.e21_report_json))
    dense_csv = Path(e21.get("paths", {}).get("dense_csv", args.d2_csv))
    replay = replay_e21_engine(dense_csv, Path(args.internal_predictions), pipeline, args.budget, args.rolling_window_days)
    if replay.get("status") != "OK":
        raise ValueError("ENGINE_REPLAY_MISMATCH")
    seed_df = load_seed_features([Path(args.d2_csv), Path(args.incremental_features)], freeze_ts, int(args.rolling_window_days))
    output_root = Path(args.output_root)
    candidate_id = args.candidate_id or f"trrm_e21_f02_{compact_utc_stamp()}"
    candidate_dir = output_root / candidate_id
    safe_research_path(candidate_dir)
    if candidate_dir.exists() and (candidate_dir / "candidate_manifest.json").exists():
        raise ValueError("FREEZE_MANIFEST_CONFLICT")
    temp_seed = Path(tempfile.mkdtemp(prefix="trrm_f02_seed_")) / "history_seed.jsonl"
    seed = build_seed(seed_df, pipeline, freeze_ts, temp_seed)
    base_payload = {
        "phase": "F0.2",
        "decision": "F02_FRESH_SEED_READY",
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "freeze_time": freeze_time,
        "artifact_integrity": integrity,
        "engine_replay": replay,
        "fresh_seed": seed,
        "labels_read": False,
        "performance_evaluated": False,
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    if seed["last_age_hours"] is None or seed["last_age_hours"] > float(args.maximum_seed_age_hours):
        base_payload["decision"] = "RECENT_SEED_STALE"
        return base_payload
    if seed["coverage_days"] < float(args.minimum_recent_seed_days):
        base_payload["decision"] = "RECENT_SEED_INSUFFICIENT"
        return base_payload
    source = inspect_turbo_signal_source(source_glob=DEFAULT_SIGNAL_GLOB)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_seed = candidate_dir / "history_seed.jsonl"
    atomic_write_text(final_seed, temp_seed.read_text(encoding="utf-8"))
    seed["path"] = str(final_seed)
    seed["sha256"] = sha256_file(final_seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "F0.2",
        "candidate_id": candidate_id,
        "supersedes_candidate_id": Path(args.previous_candidate_dir).name,
        "supersession_reason": "FRESH_MARKET_REFRESHED_30D_HISTORY_SEED_REQUIRED",
        "previous_candidate_modified": False,
        "frozen_at_utc": freeze_time,
        "branch": git_value(["branch", "--show-current"]),
        "repository_commit": git_value(["rev-parse", "HEAD"]),
        "model_path": str(model_dir),
        "model_file_hashes": model_file_hashes(model_dir),
        "model_hash": model_hash,
        "model_class": "RandomForestClassifier",
        "feature_hash": args.feature_hash,
        "feature_list": features,
        "target": args.target,
        "policy_method": CHAMPION_POLICY,
        "budget": args.budget,
        "rolling_window_days": args.rolling_window_days,
        "engine_name": CHAMPION_ENGINE,
        "primary_horizon": PRIMARY_HORIZON,
        "diagnostic_horizons": DIAGNOSTIC_HORIZONS,
        "feature_store_csv": str(Path(args.incremental_features)),
        "feature_alignment": {"max_gap_minutes": 10, "rule": "feature_timestamp <= opportunity_timestamp"},
        "fresh_seed": True,
        "fresh_seed_days": seed["coverage_days"],
        "fresh_seed_start": seed["history_start"],
        "fresh_seed_end": seed["history_end"],
        "fresh_seed_last_age_hours": seed["last_age_hours"],
        "fresh_seed_sha256": seed["sha256"],
        "history_seed": seed,
        "source_rotation_mode": "glob",
        "source_glob": DEFAULT_SIGNAL_GLOB,
        "opportunity_semantics": source,
        "source_type": source["source_kind"],
        "fallback_no_decision_behavior": {
            "missing_h12": "NO_DECISION",
            "insufficient_history": "NO_DECISION",
            "source_alignment_error": "NO_DECISION",
            "h6_h24_never_control_primary_decision": True,
            "enforcement_action": "NONE",
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
        "artifact_integrity": integrity,
        "engine_replay": replay,
        "git_identity": git_identity(),
    }
    write_json(candidate_dir / "candidate_manifest.json", manifest)
    atomic_write_text(candidate_dir / "candidate_manifest.sha256", sha256_file(candidate_dir / "candidate_manifest.json") + "\n")
    atomic_write_text(candidate_dir / "history_seed.sha256", seed["sha256"] + "\n")
    write_json(candidate_dir / "schema.json", schema_document())
    atomic_write_text(candidate_dir / "model_monitor_scores.jsonl", "")
    atomic_write_text(candidate_dir / "opportunity_scores.jsonl", "")
    write_json(candidate_dir / "collection_state.json", {"candidate_id": candidate_id, "last_run": None, "updated_at_utc": freeze_time, "source_files": []})
    (candidate_dir / "collection_runs").mkdir(exist_ok=True)
    write_readme(candidate_dir, candidate_id)
    report = {**base_payload, "manifest_sha256": sha256_file(candidate_dir / "candidate_manifest.json")}
    if parse_bool(args.write_report):
        report_json = output_root / f"aegis_phase_f02_candidate_{compact_utc_stamp()}.json"
        report_md = output_root / f"aegis_phase_f02_candidate_{compact_utc_stamp()}.md"
        write_json(report_json, report)
        atomic_write_text(report_md, f"# FASE-F0.2 Candidate\n\n- decision: {report['decision']}\n- candidate_id: {candidate_id}\n- FORWARD_OUTCOMES_NOT_EVALUATED\n")
        report["report_json"] = str(report_json)
        report["report_md"] = str(report_md)
    print(json.dumps(report, indent=2, default=json_default))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build F0.2 fresh TRRM candidate")
    p.add_argument("--previous-candidate-dir", default=str(DEFAULT_PREVIOUS))
    p.add_argument("--d2-csv", default=str(DEFAULT_D2))
    p.add_argument("--incremental-features", required=True)
    p.add_argument("--e21-report-json", default=str(DEFAULT_E21))
    p.add_argument("--internal-predictions", default=str(DEFAULT_INTERNAL))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--candidate-id", default="")
    p.add_argument("--freeze-time", default="")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--feature-hash", default=EXPECTED_FEATURE_HASH)
    p.add_argument("--model-hash", default=EXPECTED_MODEL_HASH)
    p.add_argument("--policy", default=CHAMPION_POLICY)
    p.add_argument("--budget", type=float, default=CHAMPION_BUDGET)
    p.add_argument("--rolling-window-days", type=int, default=CHAMPION_WINDOW_DAYS)
    p.add_argument("--engine", default=CHAMPION_ENGINE)
    p.add_argument("--minimum-recent-seed-days", type=float, default=25.0)
    p.add_argument("--maximum-seed-age-hours", type=float, default=24.0)
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_build(parse_args(argv))
        return 0 if payload["decision"] == "F02_FRESH_SEED_READY" else 2
    except Exception as exc:
        decision = "RESEARCH_NOT_READY"
        for item in ("ARTIFACT_INTEGRITY_ERROR", "ENGINE_REPLAY_MISMATCH", "RECENT_SEED_STALE", "RECENT_SEED_INSUFFICIENT"):
            if item in str(exc):
                decision = item
                break
        print(json.dumps({"decision": decision, "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
