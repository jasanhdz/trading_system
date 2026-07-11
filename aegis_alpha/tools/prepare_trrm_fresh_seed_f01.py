#!/usr/bin/env python3
"""FASE-F0.1 fresh TRRM forward seed preparation.

Research-only. Creates a new candidate only when a contemporary, label-free
30d rolling history seed can be reconstructed from causal features.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import feature_hash, score_frame
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
    contains_label_columns,
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
    threshold_from_history,
    utc_now,
    write_json,
)
from aegis_alpha.tools.train_trrm_honest_e2 import MedianImputer, StandardScalerLite, TARGET

DEFAULT_PREVIOUS = Path("/home/jasan/Develop/aegis_forward_research/trrm_f0/trrm_e21_f0_20260710T232340Z")
DEFAULT_E2_JSON = Path("/home/jasan/Develop/aegis_phase_e2_trrm_honest_20260710T173714Z.json")
DEFAULT_E21_JSON = Path("/home/jasan/Develop/aegis_phase_e21_trrm_calibration_20260710T183052Z.json")
DEFAULT_MODEL_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_POLICY_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e21/20260710T183052Z")
DEFAULT_DENSE = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_OUTPUT = Path("/home/jasan/Develop/aegis_forward_research/trrm_f01")
DEFAULT_INTERNAL = Path("/home/jasan/Develop/aegis_phase_e21_internal_predictions_20260710T183052Z.csv")
EXPECTED_MODEL_HASH = "edf7490c18db043a52dbe2b8fa03fe42748be60240423cb8cccea21b3e630992"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare F0.1 fresh rolling history seed")
    p.add_argument("--previous-candidate-dir", default=str(DEFAULT_PREVIOUS))
    p.add_argument("--e2-report-json", default=str(DEFAULT_E2_JSON))
    p.add_argument("--e21-report-json", default=str(DEFAULT_E21_JSON))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--policy-dir", default=str(DEFAULT_POLICY_DIR))
    p.add_argument("--internal-predictions", default=str(DEFAULT_INTERNAL))
    p.add_argument("--feature-source-kind", default="causal_feature_csv")
    p.add_argument("--feature-source-path", default=str(DEFAULT_DENSE))
    p.add_argument("--feature-source-glob", default="")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--candidate-id", default="")
    p.add_argument("--freeze-time", default="")
    p.add_argument("--target", default=TARGET)
    p.add_argument("--feature-hash", default=EXPECTED_FEATURE_HASH)
    p.add_argument("--budget", type=float, default=CHAMPION_BUDGET)
    p.add_argument("--rolling-window-days", type=int, default=CHAMPION_WINDOW_DAYS)
    p.add_argument("--minimum-recent-seed-days", type=float, default=25.0)
    p.add_argument("--maximum-seed-age-hours", type=float, default=24.0)
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


def feature_source_paths(args: argparse.Namespace) -> list[Path]:
    if args.feature_source_glob:
        return sorted(Path(p) for p in __import__("glob").glob(args.feature_source_glob))
    return [Path(args.feature_source_path)]


def load_recent_features(paths: list[Path], freeze_ts: pd.Timestamp, days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = freeze_ts - pd.Timedelta(days=days)
    frames = []
    source_meta = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "id.timestamp" not in df.columns:
            continue
        ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
        keep = (ts >= start) & (ts < freeze_ts)
        part = df.loc[keep].copy()
        frames.append(part)
        source_meta.append({"path": str(path), "rows": int(len(df)), "recent_rows": int(keep.sum()), "min": str(ts.min()), "max": str(ts.max())})
    if not frames:
        return pd.DataFrame(), {"paths": [str(p) for p in paths], "sources": source_meta}
    out = pd.concat(frames, ignore_index=True)
    out["_ts"] = pd.to_datetime(out["id.timestamp"], errors="coerce", utc=True)
    out = out.sort_values(["_ts", "id.symbol", "id.horizon"]).drop(columns=["_ts"]).reset_index(drop=True)
    return out, {"paths": [str(p) for p in paths], "sources": source_meta}


def labels_in_seed_rows(rows: list[dict[str, Any]]) -> bool:
    return any(contains_label_columns(row) for row in rows)


def build_fresh_seed(df: pd.DataFrame, pipeline: dict[str, Any], freeze_ts: pd.Timestamp, out_path: Path, budget: float, window_days: int) -> dict[str, Any]:
    scores = score_frame(df, pipeline)
    rows: list[dict[str, Any]] = []
    threshold_available = 0
    no_decision_history = 0
    previous_count = 0
    for i, row in df.reset_index(drop=True).iterrows():
        ts = parse_dt(str(row.get("id.timestamp")))
        if ts is None or ts >= freeze_ts:
            continue
        seed_row = {
            "schema_version": SCHEMA_VERSION,
            "stream": "HISTORY_SEED",
            "market_timestamp": str(ts),
            "symbol": row.get("id.symbol"),
            "timeframe": row.get("id.timeframe"),
            "horizon": int(row.get("id.horizon")),
            "score": float(scores[i]),
            "source_reference": "fresh_causal_feature_seed_f01",
        }
        if previous_count > 0:
            threshold_available += 1
        else:
            no_decision_history += 1
        rows.append(seed_row)
        previous_count += 1
    if labels_in_seed_rows(rows):
        raise ValueError("FORWARD_LABEL_COLUMN_DETECTED")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False, default=json_default) + "\n" for r in rows))
    times = [parse_dt(r["market_timestamp"]) for r in rows]
    times = [t for t in times if t is not None]
    symbols = sorted({str(r["symbol"]) for r in rows})
    horizons = sorted({int(r["horizon"]) for r in rows})
    first = min(times) if times else None
    last = max(times) if times else None
    coverage_days = float((last - first).total_seconds() / 86400.0) if first is not None and last is not None else 0.0
    last_age_hours = float((freeze_ts - last).total_seconds() / 3600.0) if last is not None else None
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "rows": len(rows),
        "symbols": symbols,
        "horizons": horizons,
        "history_start": str(first) if first is not None else None,
        "history_end": str(last) if last is not None else None,
        "coverage_days": coverage_days,
        "last_age_hours": last_age_hours,
        "threshold_availability_rate": threshold_available / len(rows) if rows else 0.0,
        "no_decision_history_rate": no_decision_history / len(rows) if rows else 1.0,
        "labels_present": False,
    }


def validate_artifacts(args: argparse.Namespace, pipeline: dict[str, Any], e2: dict[str, Any], e21: dict[str, Any]) -> dict[str, Any]:
    features = list(pipeline["features"])
    model_hash = combined_model_hash(Path(args.model_dir))
    checks = {
        "target_exact": args.target == TARGET and pipeline.get("target") == TARGET and e21.get("target") == TARGET,
        "feature_hash_exact": feature_hash(features) == args.feature_hash,
        "model_class_exact": e2.get("selected_candidate", {}).get("model") == "random_forest",
        "model_hash_exact": model_hash == EXPECTED_MODEL_HASH if Path(args.model_dir) == DEFAULT_MODEL_DIR else True,
        "policy_exact": (e21.get("selected_policy") or {}).get("method") == CHAMPION_POLICY,
        "budget_exact": abs(float(args.budget) - CHAMPION_BUDGET) < 1e-12,
        "rolling_window_exact": int(args.rolling_window_days) == CHAMPION_WINDOW_DAYS,
        "engine_exact": CHAMPION_ENGINE == "E21_PER_ROW_CANONICAL",
        "primary_horizon_h12": int(args.primary_horizon) == PRIMARY_HORIZON,
        "diagnostics_h6_h24": sorted(int(x) for x in args.diagnostic_horizons) == DIAGNOSTIC_HORIZONS,
        "no_raw_close": "feature.close" not in features,
        "no_symbol_feature": "symbol" not in features and "id.symbol" not in features,
    }
    return {
        "status": "OK" if all(checks.values()) else "ARTIFACT_INTEGRITY_ERROR",
        "checks": checks,
        "model_hash": model_hash,
        "feature_hash": feature_hash(features),
        "feature_count": len(features),
        "feature_list": features,
    }


def write_readme(path: Path, candidate_id: str) -> None:
    atomic_write_text(
        path / "README.md",
        "\n".join(
            [
                "# TRRM F0.1 Fresh Forward Research Candidate",
                "",
                f"Candidate: `{candidate_id}`",
                "",
                "Research-only passive score collection. Enforcement is disabled.",
                "H12 is the only pre-registered primary horizon. H6/H24 are diagnostic-only.",
                "History seed is a fresh 30d rolling seed ending before frozen_at_utc and contains no labels.",
                "FORWARD_OUTCOMES_NOT_EVALUATED.",
                "",
            ]
        ),
    )


def report_blocked(args: argparse.Namespace, decision: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["decision"] = decision
    if parse_bool(args.write_report):
        root = Path(args.output_root)
        safe_research_path(root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = compact_utc_stamp()
        js = root / f"aegis_phase_f01_fresh_seed_{stamp}.json"
        md = root / f"aegis_phase_f01_fresh_seed_{stamp}.md"
        write_json(js, out)
        atomic_write_text(md, f"# FASE-F0.1 Fresh Seed\n\n- decision: {decision}\n- FORWARD_OUTCOMES_NOT_EVALUATED\n")
        out["report_json"] = str(js)
        out["report_md"] = str(md)
    print(json.dumps(out, indent=2, default=json_default))
    return out


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    guard_no_enforcement_imports(Path(__file__))
    freeze_time = args.freeze_time or utc_now()
    freeze_ts = parse_dt(freeze_time)
    if freeze_ts is None:
        raise ValueError("invalid freeze_time")
    previous_dir = Path(args.previous_candidate_dir)
    previous_manifest = load_json(previous_dir / "candidate_manifest.json")
    previous_seed = previous_manifest.get("history_seed", {})
    previous_manifest_sha_before = sha256_file(previous_dir / "candidate_manifest.json")
    e2 = load_json(Path(args.e2_report_json))
    e21 = load_json(Path(args.e21_report_json))
    pipeline = load_pipeline_checked(Path(args.model_dir))
    integrity = validate_artifacts(args, pipeline, e2, e21)
    if integrity["status"] != "OK":
        raise ValueError("ARTIFACT_INTEGRITY_ERROR")
    dense_csv = Path(e21.get("paths", {}).get("dense_csv", args.feature_source_path))
    internal = Path(args.internal_predictions)
    replay = replay_e21_engine(dense_csv, internal, pipeline, args.budget, args.rolling_window_days) if internal.exists() else {"status": "SKIPPED_INTERNAL_PREDICTIONS_UNAVAILABLE"}
    if replay.get("status") not in {"OK", "SKIPPED_INTERNAL_PREDICTIONS_UNAVAILABLE"}:
        raise ValueError("ENGINE_REPLAY_MISMATCH")
    paths = feature_source_paths(args)
    recent, source_meta = load_recent_features(paths, freeze_ts, args.rolling_window_days)
    base_payload = {
        "phase": "F0.1",
        "freeze_time": freeze_time,
        "previous_candidate": {
            "candidate_id": previous_manifest.get("candidate_id"),
            "freeze_time": previous_manifest.get("frozen_at_utc"),
            "old_seed_start": previous_seed.get("history_start"),
            "old_seed_end": previous_seed.get("history_end"),
            "modified": False,
            "manifest_sha256_before": previous_manifest_sha_before,
            "superseded_reason": "FRESH_30D_HISTORY_SEED_REQUIRED",
        },
        "artifact_integrity": integrity,
        "engine_replay": replay,
        "recent_feature_source": {
            "source_kind": args.feature_source_kind,
            "paths": [str(p) for p in paths],
            "metadata": source_meta,
            "labels_present_in_seed": False,
            "causal_alignment": "id.timestamp is used as score timestamp; source rows are filtered to id.timestamp < freeze_time",
        },
        "labels_read": False,
        "performance_evaluated": False,
        "FORWARD_OUTCOMES_NOT_EVALUATED": True,
    }
    if recent.empty:
        return report_blocked(args, "RECENT_FEATURE_SOURCE_NOT_READY", base_payload)
    temp_seed = Path(tempfile.mkdtemp(prefix="trrm_f01_seed_")) / "fresh_seed_probe.jsonl"
    seed = build_fresh_seed(recent, pipeline, freeze_ts, temp_seed, args.budget, args.rolling_window_days)
    if seed["last_age_hours"] is None or seed["last_age_hours"] > float(args.maximum_seed_age_hours):
        return report_blocked(args, "RECENT_SEED_STALE", {**base_payload, "fresh_seed_probe": seed})
    if seed["coverage_days"] < float(args.minimum_recent_seed_days):
        return report_blocked(args, "RECENT_SEED_INSUFFICIENT", {**base_payload, "fresh_seed_probe": seed})
    candidate_id = args.candidate_id or f"trrm_e21_f01_{compact_utc_stamp()}"
    candidate_dir = Path(args.output_root) / candidate_id
    safe_research_path(candidate_dir)
    if candidate_dir.exists() and (candidate_dir / "candidate_manifest.json").exists():
        raise ValueError("FREEZE_MANIFEST_CONFLICT")
    source = inspect_turbo_signal_source(source_glob=DEFAULT_SIGNAL_GLOB)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "F0.1",
        "candidate_id": candidate_id,
        "supersedes_candidate_id": previous_manifest.get("candidate_id"),
        "supersession_reason": "FRESH_30D_HISTORY_SEED_REQUIRED",
        "previous_candidate_modified": False,
        "frozen_at_utc": freeze_time,
        "branch": git_value(["branch", "--show-current"]),
        "repository_commit": git_value(["rev-parse", "HEAD"]),
        "model_path": str(Path(args.model_dir)),
        "model_file_hashes": model_file_hashes(Path(args.model_dir)),
        "model_hash": integrity["model_hash"],
        "model_class": "RandomForestClassifier",
        "feature_hash": args.feature_hash,
        "feature_list": integrity["feature_list"],
        "target": args.target,
        "policy_method": CHAMPION_POLICY,
        "budget": args.budget,
        "rolling_window_days": args.rolling_window_days,
        "engine_name": CHAMPION_ENGINE,
        "primary_horizon": args.primary_horizon,
        "diagnostic_horizons": args.diagnostic_horizons,
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
    if parse_bool(args.write_artifacts):
        candidate_dir.mkdir(parents=True, exist_ok=True)
        final_seed_path = candidate_dir / "history_seed.jsonl"
        atomic_write_text(final_seed_path, temp_seed.read_text())
        seed["path"] = str(final_seed_path)
        seed["sha256"] = sha256_file(final_seed_path)
        manifest["history_seed"] = seed
        manifest["fresh_seed_sha256"] = seed["sha256"]
        write_json(candidate_dir / "candidate_manifest.json", manifest)
        atomic_write_text(candidate_dir / "candidate_manifest.sha256", sha256_file(candidate_dir / "candidate_manifest.json") + "\n")
        atomic_write_text(candidate_dir / "history_seed.sha256", seed["sha256"] + "\n")
        write_json(candidate_dir / "schema.json", schema_document())
        atomic_write_text(candidate_dir / "model_monitor_scores.jsonl", "")
        atomic_write_text(candidate_dir / "opportunity_scores.jsonl", "")
        write_json(candidate_dir / "collection_state.json", {"candidate_id": candidate_id, "last_run": None, "updated_at_utc": freeze_time, "source_files": []})
        (candidate_dir / "collection_runs").mkdir(exist_ok=True)
        write_readme(candidate_dir, candidate_id)
    payload = {
        **base_payload,
        "decision": "F01_FRESH_SEED_READY",
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "fresh_seed": seed,
        "manifest_sha256": sha256_file(candidate_dir / "candidate_manifest.json") if (candidate_dir / "candidate_manifest.json").exists() else None,
    }
    print(json.dumps(payload, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_prepare(parse_args(argv))
        return 0 if payload.get("decision") == "F01_FRESH_SEED_READY" else 2
    except ValueError as exc:
        msg = str(exc)
        decision = "RESEARCH_NOT_READY"
        for candidate in ("ARTIFACT_INTEGRITY_ERROR", "ENGINE_REPLAY_MISMATCH", "FREEZE_MANIFEST_CONFLICT", "ENFORCEMENT_PATH_DETECTED"):
            if candidate in msg:
                decision = candidate
                break
        print(json.dumps({"decision": decision, "reason": msg}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
