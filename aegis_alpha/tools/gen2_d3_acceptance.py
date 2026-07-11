#!/usr/bin/env python3
"""GEN2-D3 acceptance audit (spec v1.1 §9) — emits the single phase decision.

Aggregates gate evidence (G1-G8), runs the Gen1-vs-canonical label impact
diagnostic (H3), pre-registers the Gen2 lockbox manifest, and decides:
D3_CANONICAL_READY / D3_CANONICAL_PARTIAL / D3_CANONICAL_REJECTED.
The decision is emitted by this auditor, never by a human.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_build import FROZEN_TARGET, load_series_symbol  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import (  # noqa: E402
    GEN2_ROOT,
    LOCKBOX_MANIFEST_PATH,
    REPORTS_ROOT,
    SERIES_ROOT,
    TIMEFRAME,
    read_csv_exact,
    utc_now,
    utc_stamp,
    validate_gen2_path,
    verify_manifest_checksums,
)
from aegis_alpha.turbo.short_quality_v4_labels import ShortV4Config, compute_short_path_metrics_v4  # noqa: E402

GEN1_DENSE = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
H3_RELATIVE_LIMIT = 0.20
TAIL_ROE = 0.30


# ------------------------------------------------------------- label impact / H3
def label_impact_vs_gen1(series_dir: Path, gen1_dense_path: Path = GEN1_DENSE) -> dict[str, Any]:
    """Recompute tail labels at Gen1's exact sampled rows using canonical candles."""
    gen1 = read_csv_exact(gen1_dense_path, usecols=["id.symbol", "id.timestamp", "id.horizon", "target.tail_risk_roe_030", "future_eval.future_mae_roe_proxy"])
    gen1["_ts"] = pd.to_datetime(gen1["id.timestamp"], errors="coerce")
    rows_total = 0
    rows_matched = 0
    flips_01 = 0
    flips_10 = 0
    per_symbol: dict[str, Any] = {}
    per_horizon: dict[int, dict[str, int]] = {}
    per_month: dict[str, dict[str, int]] = {}
    mae_deltas: list[float] = []
    old_pos = 0
    new_pos = 0
    for symbol, g in gen1.groupby("id.symbol"):
        path = series_dir / f"{symbol}_{TIMEFRAME}.csv"
        if not path.exists():
            continue
        candles = load_series_symbol(series_dir, symbol)
        high = candles["high"].to_numpy(float)
        low = candles["low"].to_numpy(float)
        close = candles["close"].to_numpy(float)
        ts_index = {t: i for i, t in enumerate(pd.to_datetime(candles["timestamp"]))}
        sym = {"rows": 0, "flips_01": 0, "flips_10": 0, "old_prevalence_rows": 0}
        for _, row in g.iterrows():
            rows_total += 1
            idx = ts_index.get(row["_ts"])
            horizon = int(row["id.horizon"])
            if idx is None or idx + horizon >= len(close):
                continue
            metrics = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=idx, horizon=horizon, config=ShortV4Config(horizon=horizon))
            new_mae = float(metrics.get("mae_roe_proxy") or 0.0)
            old_mae = float(row["future_eval.future_mae_roe_proxy"] or 0.0)
            new_label = int(new_mae >= TAIL_ROE)
            old_label = int(row["target.tail_risk_roe_030"])
            rows_matched += 1
            sym["rows"] += 1
            old_pos += old_label
            new_pos += new_label
            mae_deltas.append(new_mae - old_mae)
            month = str(row["_ts"])[:7]
            per_month.setdefault(month, {"rows": 0, "flips": 0})
            per_month[month]["rows"] += 1
            per_horizon.setdefault(horizon, {"rows": 0, "flips_01": 0, "flips_10": 0})
            per_horizon[horizon]["rows"] += 1
            if old_label != new_label:
                per_month[month]["flips"] += 1
                if new_label:
                    flips_01 += 1
                    sym["flips_01"] += 1
                    per_horizon[horizon]["flips_01"] += 1
                else:
                    flips_10 += 1
                    sym["flips_10"] += 1
                    per_horizon[horizon]["flips_10"] += 1
        per_symbol[symbol] = sym
    old_prev = old_pos / rows_matched if rows_matched else 0.0
    new_prev = new_pos / rows_matched if rows_matched else 0.0
    if old_prev:
        rel_change = abs(new_prev - old_prev) / old_prev
    else:
        rel_change = float("inf") if new_prev > 0 else 0.0
    return {
        "gen1_rows": rows_total,
        "rows_matched_on_canonical": rows_matched,
        "match_rate": rows_matched / rows_total if rows_total else 0.0,
        "old_prevalence": old_prev,
        "new_prevalence": new_prev,
        "relative_prevalence_change": rel_change,
        "direction": "up" if new_prev >= old_prev else "down",
        "flips_0_to_1_real_tails_missed_by_gen1": flips_01,
        "flips_1_to_0_fake_tails_in_gen1": flips_10,
        "mae_delta_mean": float(np.mean(mae_deltas)) if mae_deltas else 0.0,
        "mae_delta_p95": float(np.quantile(mae_deltas, 0.95)) if mae_deltas else 0.0,
        "per_symbol": per_symbol,
        "per_horizon": per_horizon,
        "per_month": dict(sorted(per_month.items())),
        "h3_violated": bool(rel_change > H3_RELATIVE_LIMIT),
        "h3_limit": H3_RELATIVE_LIMIT,
    }


# ----------------------------------------------------------------- lockbox (A2)
def write_lockbox_manifest(dataset_manifest: dict[str, Any], spec_approval_utc: str) -> dict[str, Any]:
    manifest = {
        "schema": "gen2_lockbox_manifest_v1",
        "created_at_utc": utc_now().isoformat(),
        "dataset_reference": dataset_manifest["artifact_id"],
        "historical_development_period": {
            "start": dataset_manifest["timestamp_min"],
            "end": "2026-04-26 23:59:59",
            "usage": "train + validation folds with 120-minute embargo (expanding, pre-lockbox)",
        },
        "semi_blind_historical_confirmation": {
            "start": "2026-04-27 00:00:00",
            "end": dataset_manifest["timestamp_max"],
            "status": "BURNED_BY_GEN1 - downgraded to regression check only, never primary evidence",
        },
        "untouched_forward_start": spec_approval_utc,
        "embargo_minutes": 120,
        "allowed_query_count_per_candidate": 1,
        "current_query_count": 0,
        "query_log": [],
        "unlock_conditions": "candidate fully frozen by hash (model + features + target + policy + thresholds)",
        "forbidden_uses": [
            "algorithm selection", "threshold selection", "calibration fitting",
            "feature engineering", "sampling decisions", "ECON1 development",
            "any inspection of forward labels before candidate freeze",
        ],
        "contains_no_forward_metrics": True,
    }
    validate_gen2_path(LOCKBOX_MANIFEST_PATH)
    if LOCKBOX_MANIFEST_PATH.exists():
        raise FileExistsError(f"lockbox manifest already exists (immutability): {LOCKBOX_MANIFEST_PATH}")
    LOCKBOX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# -------------------------------------------------------------------- decision
def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    dataset_dir = Path(args.dataset_dir)
    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    series_dir = SERIES_ROOT / dataset_manifest["series_version"]
    series_manifest = json.loads((series_dir / "series_manifest.json").read_text())
    snapshot_dir = Path(series_manifest["source_snapshot_dir"])
    g3 = json.loads(Path(args.g3_report).read_text())
    double = json.loads(Path(args.double_report).read_text()) if args.double_report else {}
    replay = json.loads(Path(args.replay_report).read_text()) if args.replay_report else {}
    g6 = json.loads(Path(args.g6_report).read_text()) if args.g6_report else {}

    snap_ok, snap_errors = verify_manifest_checksums(snapshot_dir, "snapshot_manifest")
    series_ok, series_errors = verify_manifest_checksums(series_dir, "series_manifest")
    dataset_ok, dataset_errors = verify_manifest_checksums(dataset_dir, "dataset_manifest")

    impact = label_impact_vs_gen1(series_dir)

    gates = {
        "G1_G2_schema_duplicates": {"passed": not series_manifest["gates"]["g1_g2_errors"], "errors": series_manifest["gates"]["g1_g2_errors"]},
        "G3_finality_refetch": {"passed": bool(g3.get("passed")), "mismatch_rate": g3.get("mismatch_rate"), "bars_checked": g3.get("bars_checked")},
        "G4_gaps": {"passed": all(g["passes"] for g in series_manifest["gates"]["g4_gaps"]), "detail": series_manifest["gates"]["g4_gaps"]},
        "G5_coverage": {"passed": all(g["passes"] for g in series_manifest["gates"]["g5_coverage"]), "detail": series_manifest["gates"]["g5_coverage"]},
        "G6_sensor_diagnostic_nonblocking": {"passed": True, "informational": {k: g6.get(k) for k in ("decision", "aggregate")} if g6 else "not run"},
        "G7_independent_replay": {"passed": bool(replay.get("passed")), "parity": replay.get("parity"), "rows": replay.get("rows_expected")},
        "G8_label_sanity_h3": {"passed": not impact["h3_violated"], "relative_change": impact["relative_prevalence_change"], "direction": impact["direction"]},
        "FEATURE_CONTRACT": {"passed": bool(dataset_manifest.get("feature_contract_ok")), "feature_hash": dataset_manifest.get("feature_hash")},
        "DOUBLE_BUILD": {"passed": bool(double.get("passed")), "checks": double.get("checks")},
        "ARTIFACT_INTEGRITY": {"passed": snap_ok and series_ok and dataset_ok, "errors": snap_errors + series_errors + dataset_errors},
    }
    excluded_symbols = series_manifest.get("excluded_symbols", [])
    blocking = [k for k, v in gates.items() if not v["passed"] and k != "G6_sensor_diagnostic_nonblocking"]
    if blocking:
        decision = "D3_CANONICAL_REJECTED"
        reason = f"blocking gates failed: {blocking}"
    elif excluded_symbols and len(excluded_symbols) <= 2:
        decision = "D3_CANONICAL_PARTIAL"
        reason = f"gates passed with {len(excluded_symbols)} documented symbol exclusions"
    else:
        decision = "D3_CANONICAL_READY"
        reason = "all blocking gates passed; dataset is canonical and reproducible"

    lockbox: dict[str, Any] | str
    if decision != "D3_CANONICAL_REJECTED" and not LOCKBOX_MANIFEST_PATH.exists():
        lockbox = write_lockbox_manifest(dataset_manifest, args.spec_approval_utc)
    elif LOCKBOX_MANIFEST_PATH.exists():
        lockbox = f"already exists: {LOCKBOX_MANIFEST_PATH}"
    else:
        lockbox = "not created (rejected)"

    payload = {
        "schema": "gen2_d3_acceptance_v1",
        "generated_at": stamp,
        "spec": "GEN2_D3_SPEC.md d3-spec-v1.1 (FROZEN)",
        "decision": decision,
        "reason": reason,
        "frozen_target": FROZEN_TARGET,
        "dataset": {k: dataset_manifest.get(k) for k in ("artifact_id", "series_version", "rows_dense", "rows_strided", "feature_hash", "timestamp_min", "timestamp_max", "target_rates")},
        "gates": gates,
        "excluded_symbols": excluded_symbols,
        "label_impact_vs_gen1": impact,
        "lockbox_manifest": lockbox,
        "next_phase": "GEN2-RV2 — TRRM V2 + QMAE V2 specification and training" if decision != "D3_CANONICAL_REJECTED" else "STOP — investigate failed gates",
        "safety": {
            "no_live_touched": True, "no_orders": True, "no_model_training": True,
            "sqlite_used_only_as_readonly_diagnostic": True, "research_only_artifacts": True,
        },
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_ROOT / f"aegis_gen2_d3_acceptance_{stamp}.json"
    md_path = REPORTS_ROOT / f"aegis_gen2_d3_acceptance_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    md_lines = [
        "# GEN2-D3 Acceptance Audit",
        "",
        f"- decision: **{decision}**",
        f"- reason: {reason}",
        f"- dataset: {dataset_manifest['artifact_id']} rows={dataset_manifest['rows_dense']}/{dataset_manifest['rows_strided']} range={dataset_manifest['timestamp_min']} -> {dataset_manifest['timestamp_max']}",
        f"- feature_hash: {dataset_manifest.get('feature_hash')}",
        "",
        "## Gates",
        *[f"- {k}: {'PASS' if v['passed'] else 'FAIL'}" for k, v in gates.items()],
        "",
        "## Label impact vs Gen1 (diagnostic)",
        f"- prevalence: {impact['old_prevalence']:.4f} -> {impact['new_prevalence']:.4f} ({impact['direction']}, rel change {impact['relative_prevalence_change']:.3%})",
        f"- real tails missed by Gen1: {impact['flips_0_to_1_real_tails_missed_by_gen1']}",
        f"- fake tails in Gen1: {impact['flips_1_to_0_fake_tails_in_gen1']}",
        f"- H3 violated: {impact['h3_violated']}",
        "",
        f"## Next phase\n- {payload['next_phase']}",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    print(json.dumps({"decision": decision, "reason": reason, "gates": {k: v["passed"] for k, v in gates.items()}, "md": str(md_path)}, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-D3 acceptance audit")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--g3-report", required=True)
    p.add_argument("--double-report", default="")
    p.add_argument("--replay-report", default="")
    p.add_argument("--g6-report", default="")
    p.add_argument("--spec-approval-utc", default="2026-07-11 09:00:00")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_acceptance(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
