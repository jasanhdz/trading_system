#!/usr/bin/env python3
"""Fable-A audit: normalize and cross-check TRRM policy metrics for E2/E2.1/E2.2.

Research-only, read-only over research artifacts. Every metric in the normalized
table is recomputed from prediction CSVs with ONE shared function and labeled with
an explicit scope (which rows), population (dense/strided), and engine (which
threshold implementation), so that numbers from different reports can only be
compared when those labels match.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402

TARGET = "target.tail_risk_roe_030"
DEFAULT_OUT = Path("/home/jasan/Develop")
DEFAULTS = {
    "e2_json": "/home/jasan/Develop/aegis_phase_e2_trrm_honest_20260710T173714Z.json",
    "e2_lockbox_strided": "/home/jasan/Develop/aegis_phase_e2_lockbox_predictions_strided_20260710T173714Z.csv",
    "e21_json": "/home/jasan/Develop/aegis_phase_e21_trrm_calibration_20260710T183052Z.json",
    "e21_lockbox_csv": "/home/jasan/Develop/aegis_phase_e21_opened_lockbox_diagnostic_20260710T183052Z.csv",
    "e22_json": "/home/jasan/Develop/aegis_phase_e22_trrm_horizon_policy_20260710T203641Z.json",
    "e22_internal_csv": "/home/jasan/Develop/aegis_phase_e22_internal_predictions_20260710T203641Z.csv",
    "e22_lockbox_csv": "/home/jasan/Develop/aegis_phase_e22_opened_lockbox_diagnostic_20260710T203641Z.csv",
    "static_threshold": 0.39101951472531293,
}
STATUSES = (
    "METRICS_CONSISTENT",
    "METRICS_SCOPE_AMBIGUOUS",
    "METRICS_AGGREGATION_BUG",
    "ARTIFACT_MISMATCH",
    "DATASET_POPULATION_MISMATCH",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def shared_policy_metrics(y: np.ndarray, reject: np.ndarray, budget: float | None = None) -> dict[str, Any]:
    """Single metric definition used for every row of the normalized table."""
    y = np.asarray(y, dtype=int)
    reject = np.asarray(reject, dtype=bool)
    n = len(y)
    positives = int(y.sum())
    rejected = int(reject.sum())
    captured = int(((y == 1) & reject).sum())
    retained = y[~reject]
    prevalence = positives / n if n else 0.0
    rejection = rejected / n if n else 0.0
    precision = captured / rejected if rejected else 0.0
    residual = float(retained.mean()) if len(retained) else 0.0
    out = {
        "rows": n,
        "positives": positives,
        "prevalence": prevalence,
        "realized_rejection_rate": rejection,
        "tail_capture_rate": captured / positives if positives else 0.0,
        "residual_tail_rate": residual,
        "lift": (precision / prevalence) if prevalence else 0.0,
        "false_negatives": positives - captured,
    }
    if budget is not None:
        out["requested_budget"] = budget
        out["absolute_budget_error"] = abs(rejection - budget)
    return out


def table_row(phase: str, policy: str, scope: str, population: str, engine: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"phase": phase, "policy": policy, "scope": scope, "population": population, "engine": engine, **metrics}


def close(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def audit_population(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    stats = {}
    for name, df in frames.items():
        stats[name] = {
            "rows": int(len(df)),
            "positives": int(df[TARGET].astype(int).sum()),
            "ts_min": str(df["id.timestamp"].min()),
            "ts_max": str(df["id.timestamp"].max()),
        }
    values = list(stats.values())
    identical = all(v == values[0] for v in values[1:]) if len(values) > 1 else True
    return {"per_artifact": stats, "identical": bool(identical)}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    findings: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    table: list[dict[str, Any]] = []

    e2 = json.loads(Path(args.e2_json).read_text())
    e21 = json.loads(Path(args.e21_json).read_text())
    e22 = json.loads(Path(args.e22_json).read_text())
    e2_lock = pd.read_csv(args.e2_lockbox_strided)
    e21_lock = pd.read_csv(args.e21_lockbox_csv)
    e22_lock = pd.read_csv(args.e22_lockbox_csv)
    e22_internal = pd.read_csv(args.e22_internal_csv)

    # --- 1. Lockbox population identity across phases -------------------------------
    population = audit_population({"e2_strided_lockbox": e2_lock, "e21_lockbox": e21_lock, "e22_lockbox": e22_lock})
    checks["lockbox_population_identical"] = population["identical"]
    if not population["identical"]:
        findings.append({"severity": "HIGH", "kind": "DATASET_POPULATION_MISMATCH", "detail": population["per_artifact"]})

    # --- 2. E2 static threshold reproduction ----------------------------------------
    y = e2_lock[TARGET].astype(int).to_numpy()
    reject = e2_lock["risk_probability"].astype(float).to_numpy() >= float(args.static_threshold)
    static = shared_policy_metrics(y, reject, budget=None)
    table.append(table_row("E2", "STATIC_ABSOLUTE_THRESHOLD", "opened_lockbox", "strided", "frozen_threshold", static))
    flag_col = e2_lock.get("rejected_at_frozen_threshold")
    checks["e2_static_flag_column_matches_threshold"] = bool(flag_col is not None and (flag_col.astype(int).to_numpy() == reject.astype(int)).all())

    # --- 3. E2.1: aggregate must equal the mean of its own fold metrics -------------
    e21_sel = e21.get("selected_policy") or {}
    e21_folds = [f for f in e21_sel.get("folds", []) if not f.get("skipped")]
    e21_agg = e21_sel.get("aggregate", {})
    fold_rejs = [f["metrics"]["realized_rejection_rate"] for f in e21_folds]
    fold_caps = [f["metrics"]["tail_capture_rate"] for f in e21_folds]
    checks["e21_aggregate_rejection_is_fold_mean"] = close(np.mean(fold_rejs) if fold_rejs else None, e21_agg.get("mean_realized_rejection"))
    checks["e21_aggregate_capture_is_fold_mean"] = close(np.mean(fold_caps) if fold_caps else None, e21_agg.get("mean_tail_capture"))
    for f in e21_folds:
        table.append(table_row("E2.1", str(e21_sel.get("method")), f["fold"], "dense_eval", "e21_per_row_rolling",
                               {k: f["metrics"].get(k) for k in ("rows", "positives", "prevalence", "realized_rejection_rate", "tail_capture_rate", "residual_tail_rate", "lift")}))
    table.append(table_row("E2.1", str(e21_sel.get("method")), "folds_mean(3)", "dense_eval", "e21_per_row_rolling",
                           {"realized_rejection_rate": e21_agg.get("mean_realized_rejection"), "tail_capture_rate": e21_agg.get("mean_tail_capture"),
                            "residual_tail_rate": e21_agg.get("mean_residual_tail_rate"), "lift": e21_agg.get("mean_lift")}))
    # E2.1 horizon_balance in the report is the LAST fold only — scope finding.
    checks["e21_horizon_balance_labeled_last_fold_scope"] = bool(
        isinstance(e21.get("horizon_balance_scope"), str) and "last" in str(e21.get("horizon_balance_scope")).lower()
    )
    if not checks["e21_horizon_balance_labeled_last_fold_scope"]:
        findings.append({
            "severity": "MEDIUM",
            "kind": "METRICS_SCOPE_AMBIGUOUS",
            "detail": "E2.1 report key 'horizon_balance' is computed on the final pre-lockbox fold only, while 'aggregate' averages all folds; the report does not label the scope.",
        })

    # --- 4. E2.2: recompute everything from the internal predictions CSV ------------
    e22_sel = e22.get("selected_policy") or {}
    e22_agg = e22_sel.get("aggregate", {})
    grp = e22_internal.groupby("fold")
    fold_names = sorted(grp.groups)
    recomputed_fold_rej = {}
    per_horizon_by_fold: dict[str, dict[int, float]] = {}
    for fold_name in fold_names:
        g = grp.get_group(fold_name)
        m = shared_policy_metrics(g[TARGET].astype(int).to_numpy(), g["reject"].astype(bool).to_numpy(), budget=float(e22_sel.get("budget", 0.30)))
        recomputed_fold_rej[fold_name] = m["realized_rejection_rate"]
        table.append(table_row("E2.2", str(e22_sel.get("method")), fold_name, "dense_eval", "e22_day_batched", m))
        per_horizon_by_fold[fold_name] = {}
        for horizon, gh in g.groupby(g["id.horizon"].astype(int)):
            hm = shared_policy_metrics(gh[TARGET].astype(int).to_numpy(), gh["reject"].astype(bool).to_numpy())
            per_horizon_by_fold[fold_name][int(horizon)] = hm["realized_rejection_rate"]
            table.append(table_row("E2.2", str(e22_sel.get("method")), f"{fold_name}/H{horizon}", "dense_eval", "e22_day_batched", hm))
    json_fold_rej = {f["fold"]: f["metrics"]["realized_rejection_rate"] for f in e22_sel.get("folds", []) if not f.get("skipped")}
    checks["e22_fold_metrics_reproduce_from_csv"] = all(close(recomputed_fold_rej.get(k), v, 1e-9) for k, v in json_fold_rej.items())
    checks["e22_aggregate_rejection_is_fold_mean"] = close(np.mean(list(json_fold_rej.values())) if json_fold_rej else None, e22_agg.get("mean_realized_rejection"))
    # horizon spread in aggregate = spread of per-horizon FOLD MEANS (macro), not of any single fold
    horizons = sorted({h for d in per_horizon_by_fold.values() for h in d})
    horizon_fold_means = {h: float(np.mean([per_horizon_by_fold[f][h] for f in fold_names if h in per_horizon_by_fold[f]])) for h in horizons}
    spread = max(horizon_fold_means.values()) - min(horizon_fold_means.values()) if horizon_fold_means else None
    checks["e22_horizon_spread_is_macro_over_folds"] = close(spread, e22_agg.get("horizon_rejection_spread"), 1e-6)
    for h, v in horizon_fold_means.items():
        table.append(table_row("E2.2", str(e22_sel.get("method")), f"folds_mean/H{h}", "dense_eval", "e22_day_batched", {"realized_rejection_rate": v}))
    # selected_horizon_balance in the report is the LAST fold only — scope finding.
    last = fold_names[-1] if fold_names else None
    reported_hb = {int(r["horizon"]): r["realized_rejection_rate"] for r in e22.get("selected_horizon_balance", [])}
    matches_last_fold = bool(last) and all(close(per_horizon_by_fold.get(last, {}).get(h), v, 1e-9) for h, v in reported_hb.items())
    checks["e22_selected_horizon_balance_equals_last_fold"] = matches_last_fold
    checks["e22_horizon_balance_labeled_last_fold_scope"] = bool(
        isinstance(e22.get("selected_horizon_balance_scope"), str) and "last" in str(e22.get("selected_horizon_balance_scope")).lower()
    )
    if matches_last_fold and not checks["e22_horizon_balance_labeled_last_fold_scope"]:
        findings.append({
            "severity": "MEDIUM",
            "kind": "METRICS_SCOPE_AMBIGUOUS",
            "detail": "E2.2 'selected_horizon_balance' (H6 0.387 / H12 0.379 / H24 0.400) is the FINAL fold only; 'aggregate.mean_realized_rejection' (0.292) averages all folds. Both are correct; the report mixes scopes without labels.",
        })

    # --- 5. Opened-lockbox engine comparability (E2.1 report vs E2.2 reference) -----
    e21_lock_json = (e21.get("opened_lockbox_diagnostic") or {}).get("selected_policy", {})
    y21 = e21_lock[TARGET].astype(int).to_numpy()
    m21 = shared_policy_metrics(y21, e21_lock["reject"].astype(bool).to_numpy())
    table.append(table_row("E2.1", "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", "opened_lockbox", "strided", "e21_per_row_rolling", m21))
    checks["e21_lockbox_csv_matches_json"] = close(m21["realized_rejection_rate"], e21_lock_json.get("realized_rejection_rate")) and close(
        m21["tail_capture_rate"], e21_lock_json.get("tail_capture_rate")
    )
    e22_lock_json = ((e22.get("opened_lockbox_diagnostic") or {}).get("selected_e22_policy") or {}).get("metrics", {})
    y22 = e22_lock[TARGET].astype(int).to_numpy()
    m22 = shared_policy_metrics(y22, e22_lock["reject"].astype(bool).to_numpy())
    table.append(table_row("E2.2", str(e22_sel.get("method")), "opened_lockbox", "strided", "e22_day_batched", m22))
    checks["e22_lockbox_csv_matches_json"] = close(m22["realized_rejection_rate"], e22_lock_json.get("realized_rejection_rate")) and close(
        m22["tail_capture_rate"], e22_lock_json.get("tail_capture_rate")
    )
    g_ref = ((e22.get("opened_lockbox_diagnostic") or {}).get("global_e21_reference") or {}).get("metrics", {})
    if g_ref:
        table.append(table_row("E2.2", "GLOBAL_ROLLING_REFERENCE (re-implementation of E2.1)", "opened_lockbox", "strided", "e22_day_batched",
                               {k: g_ref.get(k) for k in ("rows", "positives", "prevalence", "realized_rejection_rate", "tail_capture_rate", "residual_tail_rate", "lift")}))
    engines_differ = bool(g_ref) and not (
        close(g_ref.get("realized_rejection_rate"), e21_lock_json.get("realized_rejection_rate"), 1e-6)
        and close(g_ref.get("tail_capture_rate"), e21_lock_json.get("tail_capture_rate"), 1e-6)
    )
    checks["e22_global_reference_uses_different_engine_than_e21"] = engines_differ
    if engines_differ:
        findings.append({
            "severity": "MEDIUM",
            "kind": "ARTIFACT_MISMATCH",
            "detail": (
                "E2.2's 'global_e21_reference' re-implements the E2.1 policy with the E2.2 day-batched engine "
                "(minimum-history NO_DECISION semantics) instead of the E2.1 per-row engine (fallback to full "
                "calibration below 50 samples). Same population, different mechanics: rejection "
                f"{g_ref.get('realized_rejection_rate'):.5f} vs {e21_lock_json.get('realized_rejection_rate'):.5f}, "
                f"capture {g_ref.get('tail_capture_rate'):.4f} vs {e21_lock_json.get('tail_capture_rate'):.4f}. "
                "Any F0 freeze must name exactly one engine."
            ),
        })

    # --- Status ----------------------------------------------------------------------
    aggregation_ok = all(
        checks[k]
        for k in (
            "e21_aggregate_rejection_is_fold_mean",
            "e21_aggregate_capture_is_fold_mean",
            "e22_fold_metrics_reproduce_from_csv",
            "e22_aggregate_rejection_is_fold_mean",
            "e22_horizon_spread_is_macro_over_folds",
            "e21_lockbox_csv_matches_json",
            "e22_lockbox_csv_matches_json",
        )
    )
    if not checks["lockbox_population_identical"]:
        status = "DATASET_POPULATION_MISMATCH"
    elif not aggregation_ok:
        status = "METRICS_AGGREGATION_BUG"
    elif findings:
        status = "METRICS_SCOPE_AMBIGUOUS"
    else:
        status = "METRICS_CONSISTENT"

    payload = {
        "schema_version": "fable_trrm_policy_metrics_audit_a_v1",
        "generated_at": stamp,
        "mode": "research-only read-only audit",
        "status": status,
        "target": TARGET,
        "checks": checks,
        "findings": findings,
        "lockbox_population": population,
        "normalized_table": table,
        "scope_glossary": {
            "fold_i": "metrics on that pre-lockbox fold's evaluation rows (dense sampling)",
            "folds_mean(3)": "unweighted mean of the three fold-level metrics (macro over folds)",
            "fold_i/Hx": "subset of that fold's evaluation rows with id.horizon == x",
            "folds_mean/Hx": "unweighted mean over folds of the per-horizon metric (macro)",
            "opened_lockbox": "strided rows with timestamp >= lockbox start; diagnostic-only, previously observed",
        },
        "engine_glossary": {
            "e21_per_row_rolling": "threshold recomputed per evaluation row from trailing window (cal+prior eval); fallback to full calibration when <50 samples",
            "e22_day_batched": "threshold fixed per UTC day from history strictly before that day; NO_DECISION (threshold=inf, fail-open) when history under minimums",
            "frozen_threshold": "single absolute threshold frozen in FASE-E2 validation",
        },
    }
    json_path = out_dir / f"aegis_fable_trrm_policy_metrics_a_{stamp}.json"
    md_path = out_dir / f"aegis_fable_trrm_policy_metrics_a_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    md_lines = [
        "# Fable-A TRRM Policy Metrics Audit",
        "",
        f"- status: {status}",
        f"- generated_at: {stamp}",
        "",
        "## Checks",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
        "## Findings",
        *([f"- [{f['severity']}] {f['kind']}: {f['detail']}" for f in findings] or ["- none"]),
        "",
        "## Normalized table (single shared metric function)",
        "",
        "| phase | policy | scope | population | engine | rows | rejection | capture | residual | lift |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        *[
            "| {phase} | {policy} | {scope} | {population} | {engine} | {rows} | {rej} | {cap} | {res} | {lift} |".format(
                phase=r["phase"], policy=r["policy"], scope=r["scope"], population=r["population"], engine=r["engine"],
                rows=r.get("rows", ""),
                rej=("%.4f" % r["realized_rejection_rate"]) if r.get("realized_rejection_rate") is not None else "",
                cap=("%.4f" % r["tail_capture_rate"]) if r.get("tail_capture_rate") is not None else "",
                res=("%.5f" % r["residual_tail_rate"]) if r.get("residual_tail_rate") is not None else "",
                lift=("%.4f" % r["lift"]) if r.get("lift") is not None else "",
            )
            for r in table
        ],
        "",
        "## Scope glossary",
        *[f"- {k}: {v}" for k, v in payload["scope_glossary"].items()],
        "",
        "## Engine glossary",
        *[f"- {k}: {v}" for k, v in payload["engine_glossary"].items()],
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    print(json.dumps({"status": status, "checks_failed": [k for k, v in checks.items() if not v], "findings": len(findings), "md": str(md_path)}, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fable-A TRRM policy metrics consistency audit (research-only).")
    for key, value in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", default=value)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return p


def main() -> int:
    run_audit(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
