#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle, write_json  # noqa: E402


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_h12_tail_risk_candidate_v052.json")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/candidate_audit_v053.json")
REQUIRED_TOP_LEVEL = (
    "status",
    "model_paths",
    "sklearn_version",
    "entry_rule",
    "sizing_config",
    "risk_guard",
    "oos_metrics",
    "reason_not_live",
)
REQUIRED_METRICS = (
    "median_balance",
    "p25_balance",
    "worst_balance",
    "p25_pf",
    "profitable_window_pct",
    "median_trades",
    "worst_max_dd",
)


def _artifact_version(path: str) -> str | None:
    match = re.search(r"_(v\d+)\.joblib$", path)
    return match.group(1) if match else None


def _model_snapshot(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "artifact_version": _artifact_version(str(path)),
    }
    if not path.exists():
        return out
    try:
        bundle = load_model_bundle(path)
        metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
        out.update(
            {
                "signal_name": bundle.get("signal_name") if isinstance(bundle, dict) else None,
                "target_key": bundle.get("target_key") if isinstance(bundle, dict) else None,
                "model_kind": bundle.get("model_kind") if isinstance(bundle, dict) else None,
                "bundle_sklearn_version": metadata.get("sklearn_version"),
                "bundle_created_at": metadata.get("created_at"),
            }
        )
    except Exception as exc:  # pragma: no cover - report-only safety
        out["load_error"] = repr(exc)
    return out


def audit_strategy_candidate(candidate_path: Path, report_path: Path) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    missing_fields: list[str] = []
    missing_files: list[str] = []

    for field in REQUIRED_TOP_LEVEL:
        if field not in candidate:
            missing_fields.append(field)

    if candidate.get("status") != "OFFLINE_CANDIDATE_NOT_LIVE":
        missing_fields.append("status_OFFLINE_CANDIDATE_NOT_LIVE")
    if candidate.get("live_enabled") is True:
        missing_fields.append("live_enabled_must_not_be_true")

    model_paths = candidate.get("model_paths", {})
    for field in ("long_edge_h12", "long_tail_risk_h12"):
        path_value = model_paths.get(field)
        if not path_value:
            missing_fields.append(f"model_paths.{field}")
            continue
        if not Path(path_value).exists():
            missing_files.append(path_value)

    entry_rule = candidate.get("entry_rule") or {}
    for field in ("model", "mode", "value", "side"):
        if field not in entry_rule:
            missing_fields.append(f"entry_rule.{field}")

    sizing = candidate.get("sizing_config") or {}
    if "mode" not in sizing:
        missing_fields.append("sizing_config.mode")
    if not sizing.get("bands"):
        missing_fields.append("sizing_config.bands")

    risk_guard = candidate.get("risk_guard") or {}
    for field in ("max_window_loss_pct", "pause_after_loss_steps", "pause_after_2_losses_steps", "max_trades_per_day"):
        if field not in risk_guard:
            missing_fields.append(f"risk_guard.{field}")

    metrics = candidate.get("oos_metrics") or {}
    for field in REQUIRED_METRICS:
        if field not in metrics:
            missing_fields.append(f"oos_metrics.{field}")

    if not candidate.get("reason_not_live"):
        missing_fields.append("reason_not_live")

    model_versions = {
        name: _model_snapshot(Path(path))
        for name, path in model_paths.items()
        if isinstance(path, str)
    }
    passed = not missing_fields and not missing_files
    report = {
        "schema_version": "aegis_candidate_audit_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "candidate_path": str(candidate_path),
        "passed": bool(passed),
        "missing_fields": missing_fields,
        "missing_files": missing_files,
        "candidate_status": candidate.get("status"),
        "live_enabled": candidate.get("live_enabled", False),
        "sklearn_version": {
            "candidate": candidate.get("sklearn_version"),
            "runtime": sklearn.__version__,
        },
        "model_versions": model_versions,
        "metrics_snapshot": {field: metrics.get(field) for field in REQUIRED_METRICS},
        "reason_not_live": candidate.get("reason_not_live", []),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    audit_strategy_candidate(Path(args.candidate), Path(args.report))


if __name__ == "__main__":
    main()
