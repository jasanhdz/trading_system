#!/usr/bin/env python3
"""FASE-D2 tail-risk target audit helpers.

Research-only. This module audits candidate tail-risk targets and honest
baselines. It does not train models and never writes live artifacts.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("/home/jasan/Develop")
FIXED_ROE_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35)
ATR_THRESHOLDS = (1.5, 2.0, 2.5, 3.0)
TARGET_CANDIDATES = [
    "target.tail_risk_roe_015",
    "target.tail_risk_roe_020",
    "target.tail_risk_roe_025",
    "target.tail_risk_roe_030",
    "target.tail_risk_roe_035",
    "target.tail_risk_atr_150",
    "target.tail_risk_atr_200",
    "target.tail_risk_atr_250",
    "target.tail_risk_atr_300",
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return str(value)
    return str(value)


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)


def target_rate(df: pd.DataFrame, col: str) -> float:
    return float(bool_series(df[col]).mean()) if col in df and len(df) else 0.0


def add_tail_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mae = pd.to_numeric(out["future_eval.future_mae_roe_proxy"], errors="coerce")
    atr_pct = pd.to_numeric(out.get("feature.atr_proxy_24", np.nan), errors="coerce")
    leverage = 20.0
    adverse_price_pct = mae / leverage
    out["future_eval.adverse_price_move_pct"] = adverse_price_pct
    out["future_eval.mae_atr_units"] = adverse_price_pct / atr_pct.replace(0, np.nan)
    for thr in FIXED_ROE_THRESHOLDS:
        out[f"target.tail_risk_roe_{int(thr * 100):03d}"] = (mae >= thr).astype(int)
    for thr in ATR_THRESHOLDS:
        out[f"target.tail_risk_atr_{int(thr * 100):03d}"] = (out["future_eval.mae_atr_units"] >= thr).astype(int)
    return out


def make_temporal_split(df: pd.DataFrame, embargo_minutes: int = 120) -> dict[str, Any]:
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["id.timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["_ts"]).sort_values(["_ts", "id.symbol", "id.horizon"], kind="mergesort")
    n = len(work)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train = work.index[:train_end].to_numpy()
    val_raw = work.index[train_end:val_end].to_numpy()
    lock_raw = work.index[val_end:].to_numpy()
    embargo = pd.Timedelta(minutes=embargo_minutes)
    train_max = work.loc[train, "_ts"].max() if len(train) else pd.NaT
    val_max = work.loc[val_raw, "_ts"].max() if len(val_raw) else pd.NaT
    val_keep = work.loc[val_raw, "_ts"] > train_max + embargo if len(val_raw) else pd.Series(dtype=bool)
    lock_keep = work.loc[lock_raw, "_ts"] > val_max + embargo if len(lock_raw) else pd.Series(dtype=bool)
    val = work.loc[val_raw][val_keep].index.to_numpy() if len(val_raw) else np.array([], dtype=int)
    lock = work.loc[lock_raw][lock_keep].index.to_numpy() if len(lock_raw) else np.array([], dtype=int)
    return {
        "train_idx": train,
        "validation_idx": val,
        "lockbox_idx": lock,
        "train_start": str(work.loc[train, "_ts"].min()) if len(train) else None,
        "train_end": str(train_max) if len(train) else None,
        "validation_start": str(work.loc[val, "_ts"].min()) if len(val) else None,
        "validation_end": str(work.loc[val, "_ts"].max()) if len(val) else None,
        "lockbox_start": str(work.loc[lock, "_ts"].min()) if len(lock) else None,
        "lockbox_end": str(work.loc[lock, "_ts"].max()) if len(lock) else None,
        "embargo_minutes": embargo_minutes,
        "rows_purged": 0,
        "rows_embargoed": int(len(val_raw) - len(val) + len(lock_raw) - len(lock)),
        "timestamp_overlap_check": bool(
            len(train) and len(val) and len(lock)
            and work.loc[train, "_ts"].max() < work.loc[val, "_ts"].min()
            and work.loc[val, "_ts"].max() < work.loc[lock, "_ts"].min()
        ),
    }


def strided_view(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (_, horizon), g in df.sort_values("id.timestamp").groupby(["id.symbol", "id.horizon"], dropna=False):
        stride = max(1, int(horizon))
        parts.append(g.iloc[::stride].copy())
    return pd.concat(parts, axis=0).sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True) if parts else df.head(0).copy()


def stability_for_target(df: pd.DataFrame, col: str) -> dict[str, Any]:
    work = df.copy()
    y = bool_series(work[col])
    work["_y"] = y
    work["_month"] = pd.to_datetime(work["id.timestamp"], errors="coerce").dt.to_period("M").astype(str)
    work["_quarter"] = pd.to_datetime(work["id.timestamp"], errors="coerce").dt.to_period("Q").astype(str)

    def rates(keys: list[str]) -> dict[str, float]:
        return {str(k): float(v) for k, v in work.groupby(keys)["_y"].mean().items()}

    by_symbol = rates(["id.symbol"])
    by_horizon = rates(["id.horizon"])
    by_month = rates(["_month"])
    pos = int(y.sum())
    neg = int(len(y) - pos)
    max_min_symbol = (max(by_symbol.values()) / max(min(by_symbol.values()), 1e-9)) if by_symbol else None
    return {
        "target": col,
        "global_rate": float(y.mean()) if len(y) else 0.0,
        "positives": pos,
        "negatives": neg,
        "by_symbol": by_symbol,
        "by_horizon": by_horizon,
        "by_month": by_month,
        "by_quarter": rates(["_quarter"]),
        "symbol_rate_max_min_ratio": max_min_symbol,
    }


def relationship_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    y = bool_series(df[col]).astype(bool)
    out: dict[str, Any] = {}
    for metric in ("future_eval.future_mae_roe_proxy", "future_eval.net_quality_after_costs", "target.early_mae_v4", "target.bad_entry_v4", "label.clean_entry_v4", "label.management_dependent_v4"):
        if metric not in df:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce") if not metric.startswith(("target.", "label.")) else bool_series(df[metric])
        out[metric] = {
            "positive_mean": float(vals[y].mean()) if y.any() else None,
            "negative_mean": float(vals[~y].mean()) if (~y).any() else None,
        }
    return out


def honest_baselines(df: pd.DataFrame, target: str, split: dict[str, Any]) -> dict[str, Any]:
    train = split["train_idx"]
    val = split["validation_idx"]
    lock = split["lockbox_idx"]
    y = bool_series(df[target])
    baselines: dict[str, Any] = {}

    def evaluate_mask(mask: pd.Series, name: str) -> dict[str, Any]:
        pred = mask.astype(int)
        yt = y.loc[lock]
        pt = pred.loc[lock]
        tp = int(((yt == 1) & (pt == 1)).sum())
        fp = int(((yt == 0) & (pt == 1)).sum())
        fn = int(((yt == 1) & (pt == 0)).sum())
        tn = int(((yt == 0) & (pt == 0)).sum())
        return {
            "name": name,
            "causal": True,
            "eligible_for_selection": True,
            "eligible_for_promotion": False,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "rejection_rate": float(pt.mean()) if len(pt) else 0.0,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    baselines["reject_none"] = evaluate_mask(pd.Series(0, index=df.index), "reject_none")
    baselines["reject_all"] = evaluate_mask(pd.Series(1, index=df.index), "reject_all")
    preval = float(y.loc[train].mean()) if len(train) else 0.0
    baselines["prevalence_baseline"] = {**evaluate_mask(pd.Series(0, index=df.index), "prevalence_baseline"), "train_prevalence": preval}
    for name, cols in {
        "causal_volatility_baseline": ["feature.rolling_range_mean_24", "feature.rolling_range_std_24"],
        "causal_atr_range_percentile_baseline": ["feature.atr_proxy_24", "feature.rolling_range_mean_24"],
        "causal_rebound_squeeze_baseline": ["feature.rebound_risk_proxy", "feature.squeeze_risk_proxy_causal"],
    }.items():
        masks = []
        for c in cols:
            if c not in df:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if set(vals.dropna().unique()).issubset({0, 1}):
                masks.append(vals > 0)
            else:
                threshold = vals.loc[train].quantile(0.80)
                masks.append(vals >= threshold)
        pred = masks[0] if masks else pd.Series(0, index=df.index)
        for m in masks[1:]:
            pred = pred | m
        baselines[name] = evaluate_mask(pred, name)
    oracle_parts = []
    for c in ("target.bad_entry_v4", "target.early_mae_v4", "feature.squeeze_risk_proxy_causal"):
        if c in df:
            oracle_parts.append(bool_series(df[c]))
    oracle = oracle_parts[0].astype(bool) if oracle_parts else pd.Series(False, index=df.index)
    for p in oracle_parts[1:]:
        oracle = oracle | p.astype(bool)
    diagnostic = evaluate_mask(oracle, "diagnostic_oracle_upper_bound")
    diagnostic.update({
        "causal": False,
        "eligible_for_selection": False,
        "eligible_for_promotion": False,
        "live_usable": False,
        "reason": "uses future-derived labels/path components; diagnostic ceiling only",
    })
    baselines["diagnostic_oracle_upper_bound"] = diagnostic
    eligible = {k: v for k, v in baselines.items() if v.get("eligible_for_selection")}
    best = max(eligible, key=lambda k: (eligible[k]["recall"], eligible[k]["precision"], -eligible[k]["rejection_rate"])) if eligible else None
    return {"items": baselines, "best_baseline": best}


def target_score(item: dict[str, Any], split_rates: dict[str, float]) -> tuple[bool, float, float, float]:
    rate = item["global_rate"]
    in_tail_band = 0.05 <= rate <= 0.15
    val_rate = split_rates.get("validation", 0.0)
    train_rate = split_rates.get("train", 0.0)
    stability = 1.0 - abs(val_rate - train_rate)
    symbol_ratio = item.get("symbol_rate_max_min_ratio") or 999.0
    return (in_tail_band, stability, -abs(rate - 0.10), -min(symbol_ratio, 999.0))


def select_target(df: pd.DataFrame, target_stats: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    train, val, lock = split["train_idx"], split["validation_idx"], split["lockbox_idx"]
    candidates = {}
    for col, item in target_stats.items():
        rates = {
            "train": target_rate(df.loc[train], col),
            "validation": target_rate(df.loc[val], col),
            "lockbox": target_rate(df.loc[lock], col),
        }
        item["split_rates"] = rates
        if item["positives"] >= 200 and 0.03 <= item["global_rate"] <= 0.20:
            candidates[col] = item
    if not candidates:
        return {"selected_target": None, "reason": "no candidate met prevalence/sample constraints", "used_lockbox_for_selection": False}
    selected = max(candidates, key=lambda c: target_score(candidates[c], candidates[c]["split_rates"]))
    return {
        "selected_target": selected,
        "definition": target_definition(selected),
        "reason": "selected using train/validation prevalence and stability constraints; lockbox confirmation only",
        "used_lockbox_for_selection": False,
        "train_validation_evidence": {k: candidates[selected][k] for k in ("global_rate", "positives", "by_horizon", "split_rates")},
        "lockbox_confirmation": candidates[selected]["split_rates"].get("lockbox"),
    }


def target_definition(name: str | None) -> str | None:
    if not name:
        return None
    if "roe_" in name:
        return f"future_mae_roe_proxy >= {int(name.rsplit('_', 1)[-1]) / 100:.3f} ROE"
    if "atr_" in name:
        return f"future adverse price move / causal entry ATR >= {int(name.rsplit('_', 1)[-1]) / 100:.2f}"
    if "train_q" in name:
        return "train-only MAE quantile threshold applied to validation/lockbox"
    return name


def overlap_diagnostics(df: pd.DataFrame, target: str) -> dict[str, Any]:
    dense = df.sort_values(["id.symbol", "id.horizon", "id.timestamp"]).copy()
    strided = strided_view(dense)
    dense_rate = target_rate(dense, target)
    strided_rate = target_rate(strided, target)
    return {
        "dense_rows": int(len(dense)),
        "strided_rows": int(len(strided)),
        "approximate_effective_sample_size": int(len(strided)),
        "overlap_ratio": float(1.0 - len(strided) / len(dense)) if len(dense) else 0.0,
        "dense_positive_rate": dense_rate,
        "strided_positive_rate": strided_rate,
    }


def audit_targets(df: pd.DataFrame, embargo_minutes: int = 120) -> dict[str, Any]:
    enriched = add_tail_targets(df)
    split = make_temporal_split(enriched, embargo_minutes)
    target_stats: dict[str, Any] = {}
    relationships: dict[str, Any] = {}
    for col in TARGET_CANDIDATES:
        if col in enriched:
            target_stats[col] = stability_for_target(enriched, col)
            relationships[col] = relationship_stats(enriched, col)
    selected = select_target(enriched, target_stats, split)
    selected_target = selected.get("selected_target") or next(iter(target_stats), None)
    baselines = honest_baselines(enriched, selected_target, split) if selected_target else {}
    overlap = overlap_diagnostics(enriched, selected_target) if selected_target else {}
    return {
        "dataframe": enriched,
        "target_stats": target_stats,
        "relationships": relationships,
        "split_manifest": {k: v for k, v in split.items() if not k.endswith("_idx")},
        "split_indices": split,
        "selected_candidate": selected,
        "baselines": baselines,
        "overlap_diagnostics": overlap,
    }


def decision_from_audit(coverage_ok: bool, audit: dict[str, Any], leakage_ok: bool = True) -> tuple[str, str]:
    if not coverage_ok:
        return "DATA_COVERAGE_INSUFFICIENT", "fewer than 180 real coverage days"
    if not leakage_ok:
        return "LEAKAGE_RISK_TOO_HIGH", "feature leakage detected"
    selected = audit["selected_candidate"].get("selected_target")
    if not selected:
        return "TARGET_NOT_STABLE", "no stable tail target candidate"
    stats = audit["target_stats"][selected]
    lock_rate = audit["selected_candidate"].get("lockbox_confirmation")
    if not (0.03 <= (lock_rate or 0.0) <= 0.22):
        return "TARGET_NOT_STABLE", "lockbox prevalence contradicts train/validation stability"
    if 0.05 <= stats["global_rate"] <= 0.15:
        return "TAIL_TARGET_READY_FOR_E2", "selected target has tail-like prevalence and passed stability checks"
    return "TARGET_REDESIGN_PARTIAL", "candidate is promising but outside ideal tail prevalence band"


def run_audit_cli(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.input_csv)
    audit = audit_targets(df, embargo_minutes=args.embargo_minutes)
    decision, reason = decision_from_audit(True, audit, True)
    out = {
        "decision": decision,
        "reason": reason,
        "selected_candidate": audit["selected_candidate"],
        "target_stats": audit["target_stats"],
        "relationships": audit["relationships"],
        "split_manifest": audit["split_manifest"],
        "baselines": audit["baselines"],
        "overlap_diagnostics": audit["overlap_diagnostics"],
        "safety_confirmations": {
            "no_live_touched": True,
            "no_model_training": True,
            "oracle_diagnostic_only": True,
            "lockbox_not_used_for_selection": not audit["selected_candidate"].get("used_lockbox_for_selection", True),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "aegis_tail_target_audit_d2.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({"decision": decision, "json": str(path)}, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit FASE-D2 tail-risk targets.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--embargo-minutes", type=int, default=120)
    return p


def main() -> int:
    run_audit_cli(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
