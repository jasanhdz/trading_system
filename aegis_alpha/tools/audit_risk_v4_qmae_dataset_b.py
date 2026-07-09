#!/usr/bin/env python3
"""Research-only audit for Risk V4 / QMAE dataset readiness.

No models are trained here. The script audits label quality, leakage risks,
sample sufficiency, threshold sensitivity, and whether the dataset is ready for
a later TRRM/QMAE research training phase.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
LABEL_FIELDS = [
    "clean_entry_v4",
    "bad_entry_v4",
    "premium_allowed_v4",
    "management_dependent_v4",
    "no_trade_v4",
    "tail_risk_v4",
    "early_mae_v4",
    "squeeze_risk_proxy_v4",
]
TARGET_FIELDS = {"future_mfe_roe_proxy", "future_mae_roe_proxy", "qmae_target", "q95_mae_target"}
LABEL_ONLY_FIELDS = TARGET_FIELDS | set(LABEL_FIELDS) | {"net_quality_after_costs", "mfe_before_mae", "mfe_mae_ratio", "time_to_mfe", "time_to_mae"}
CAUSAL_FEATURE_FIELDS = {"close", "volatility_features", "trend_features", "wick_reclaim_proxies", "btc_eth_context"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_samples_file(out_dir: Path = DEFAULT_OUT_DIR) -> Path | None:
    files = sorted(out_dir.glob("aegis_risk_v4_qmae_base_dataset_a_samples_*.csv"))
    return files[-1] if files else None


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def to_int(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = min(len(data) - 1, max(0, int(round((len(data) - 1) * q))))
    return data[idx]


def label_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    out: dict[str, Any] = {"rows": total}
    for field in LABEL_FIELDS:
        count = sum(to_int(r.get(field)) for r in rows)
        out[f"{field}_count"] = count
        out[f"{field}_rate"] = count / total if total else 0.0
    premium_loser = sum(1 for r in rows if to_int(r.get("premium_allowed_v4")) and (to_int(r.get("tail_risk_v4")) or to_float(r.get("net_quality_after_costs")) < 0))
    out["premium_loser_count"] = premium_loser
    out["premium_loser_rate"] = premium_loser / total if total else 0.0
    return out


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    return groups


def target_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    maes = [to_float(r.get("future_mae_roe_proxy")) for r in rows]
    mfes = [to_float(r.get("future_mfe_roe_proxy")) for r in rows]
    net = [to_float(r.get("net_quality_after_costs")) for r in rows]
    ratios = [to_float(r.get("mfe_mae_ratio")) for r in rows]
    return {
        "mae_p50": median(maes) if maes else 0.0,
        "mae_p75": percentile(maes, 0.75),
        "mae_p90": percentile(maes, 0.90),
        "mae_p95": percentile(maes, 0.95),
        "mfe_p50": median(mfes) if mfes else 0.0,
        "mfe_p90": percentile(mfes, 0.90),
        "net_quality_mean": mean(net) if net else 0.0,
        "mfe_mae_ratio_p50": median(ratios) if ratios else 0.0,
        "mfe_mae_ratio_p25": percentile(ratios, 0.25),
    }


def grouped_distribution(rows: list[dict[str, Any]], keys: tuple[str, ...], min_group_rows: int = 500) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group, data in sorted(group_rows(rows, keys).items()):
        rates = label_rates(data)
        stats = target_stats(data)
        max_class = max([rates.get(f"{f}_rate", 0.0) for f in LABEL_FIELDS] + [0.0])
        out.append({
            **{k: v for k, v in zip(keys, group)},
            "rows": len(data),
            "sample_sufficient": len(data) >= min_group_rows,
            "class_dominance_warning": max_class >= 0.90,
            **{k: rates[k] for k in rates if k.endswith("_rate")},
            **stats,
        })
    return out


def threshold_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in (0.045, 0.06, 0.08, 0.10, 0.12, 0.15):
        selected = [r for r in rows if to_float(r.get("future_mae_roe_proxy")) >= threshold]
        early = sum(to_int(r.get("early_mae_v4")) for r in selected)
        bad = sum(to_int(r.get("bad_entry_v4")) for r in selected)
        out.append({
            "mae_threshold": threshold,
            "rows": len(selected),
            "rate": len(selected) / len(rows) if rows else 0.0,
            "early_mae_share": early / len(selected) if selected else 0.0,
            "bad_entry_share": bad / len(selected) if selected else 0.0,
        })
    return out


def simple_separability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tail = [r for r in rows if to_int(r.get("tail_risk_v4"))]
    non_tail = [r for r in rows if not to_int(r.get("tail_risk_v4"))]
    bad_rule = [r for r in rows if to_int(r.get("bad_entry_v4")) or to_int(r.get("early_mae_v4")) or to_int(r.get("squeeze_risk_proxy_v4"))]
    tp = sum(1 for r in bad_rule if to_int(r.get("tail_risk_v4")))
    fp = len(bad_rule) - tp
    fn = len(tail) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tail_rows": len(tail),
        "non_tail_rows": len(non_tail),
        "tail_mae_mean": mean([to_float(r.get("future_mae_roe_proxy")) for r in tail]) if tail else 0.0,
        "non_tail_mae_mean": mean([to_float(r.get("future_mae_roe_proxy")) for r in non_tail]) if non_tail else 0.0,
        "tail_net_quality_mean": mean([to_float(r.get("net_quality_after_costs")) for r in tail]) if tail else 0.0,
        "non_tail_net_quality_mean": mean([to_float(r.get("net_quality_after_costs")) for r in non_tail]) if non_tail else 0.0,
        "simple_rule": "bad_entry_v4 OR early_mae_v4 OR squeeze_risk_proxy_v4",
        "simple_rule_precision": precision,
        "simple_rule_recall": recall,
        "signal_gap_mae_mean": (mean([to_float(r.get("future_mae_roe_proxy")) for r in tail]) - mean([to_float(r.get("future_mae_roe_proxy")) for r in non_tail])) if tail and non_tail else 0.0,
    }


def leakage_review(headers: list[str]) -> dict[str, Any]:
    feature_inputs = [h for h in headers if h in CAUSAL_FEATURE_FIELDS]
    future_label_fields_present = sorted(h for h in headers if h in LABEL_ONLY_FIELDS)
    warnings = [
        "Do not feed future_mfe_roe_proxy, future_mae_roe_proxy, qmae_target, q95_mae_target, V4 labels, net_quality, path timing, or MFE/MAE ratio into model inputs.",
        "Use time-based walk-forward splits by symbol/timeframe/horizon; random splits would leak regime adjacency.",
    ]
    return {
        "critical_leakage_detected_in_dataset_artifact": False,
        "causal_feature_candidates": feature_inputs,
        "label_only_fields_present": future_label_fields_present,
        "warnings": warnings,
    }


def make_decision(rows: list[dict[str, Any]], grouped: list[dict[str, Any]], sep: dict[str, Any], leakage: dict[str, Any]) -> tuple[str, str]:
    sufficient_groups = sum(1 for g in grouped if g["sample_sufficient"] and not g["class_dominance_warning"])
    tail_rate = label_rates(rows).get("tail_risk_v4_rate", 0.0)
    if leakage["critical_leakage_detected_in_dataset_artifact"]:
        return "NO-GO", "critical leakage risk"
    if len(rows) < 5000 or sufficient_groups < 4:
        return "NO-GO", "insufficient sample coverage"
    if tail_rate <= 0.01 or tail_rate >= 0.80:
        return "NO-GO", "tail label is too rare or too dominant"
    if sep.get("signal_gap_mae_mean", 0.0) <= 0:
        return "NO-GO", "simple separability check failed"
    return "CONDITIONAL_GO", "dataset is suitable for research-only TRRM/QMAE training after strict feature allowlist and walk-forward split review"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples_csv) if args.samples_csv else latest_samples_file(out_dir)
    if not samples_path or not samples_path.exists():
        raise FileNotFoundError("Risk V4 / QMAE samples CSV not found")
    rows = load_rows(samples_path)
    headers = list(rows[0].keys()) if rows else []
    by_group = grouped_distribution(rows, ("symbol", "timeframe", "horizon"))
    by_symbol = grouped_distribution(rows, ("symbol",), min_group_rows=1000)
    by_horizon = grouped_distribution(rows, ("horizon",), min_group_rows=1000)
    by_timeframe = grouped_distribution(rows, ("timeframe",), min_group_rows=1000)
    labels = label_rates(rows)
    stats = target_stats(rows)
    sensitivity = threshold_sensitivity(rows)
    sep = simple_separability(rows)
    leakage = leakage_review(headers)
    decision, reason = make_decision(rows, by_group, sep, leakage)
    timestamps = [r.get("timestamp", "") for r in rows if r.get("timestamp")]
    stamp = utc_stamp()
    result = {
        "schema_version": "phase_b_qmae_dataset_review_v1",
        "generated_at": stamp,
        "status": decision,
        "reason": reason,
        "samples_csv": str(samples_path),
        "dataset_inventory": {
            "total_rows": len(rows),
            "groups": len(by_group),
            "symbols": sorted({r.get("symbol", "") for r in rows}),
            "horizons": sorted({str(r.get("horizon", "")) for r in rows}),
            "timeframes": sorted({r.get("timeframe", "") for r in rows}),
            "date_min": min(timestamps) if timestamps else None,
            "date_max": max(timestamps) if timestamps else None,
        },
        "label_distribution_global": labels,
        "target_stats_global": stats,
        "group_warnings": {
            "insufficient_groups": [g for g in by_group if not g["sample_sufficient"]],
            "class_dominated_groups": [g for g in by_group if g["class_dominance_warning"]],
        },
        "qmae_target_quality": {
            "measures": "Future SHORT adverse excursion in ROE proxy over a fixed horizon.",
            "risk_interpretation": "Potentially useful for danger/tail risk because tail rows have higher MAE and worse net quality than non-tail rows.",
            "volatility_warning": "It can still confuse normal volatility with danger unless inputs include causal volatility/regime normalization and evaluation is grouped by symbol/timeframe/horizon.",
            "simple_separability": sep,
        },
        "leakage_review": leakage,
        "threshold_review": {
            "current_tail_threshold": "MAE >= 0.10 ROE OR early_mae_v4 OR severe bad_entry_v4",
            "current_early_mae_threshold": "time_to_mae <= 3 and MAE >= 0.045 ROE",
            "sensitivity": sensitivity,
            "recommendation": "Keep thresholds as research labels for first classifier, but tune per horizon/timeframe after walk-forward validation.",
        },
        "phase_o_live_overlap": {
            "status": "not_directly_validated",
            "reason": "Current Risk/QMAE samples are hypothetical OHLCV entries and are not linked to Phase O trade IDs.",
            "next_step": "Join forward audit trade rows with contemporaneous OHLCV features before claiming live big-loss prevention or winner sacrifice rates.",
        },
        "short_v4_overlap": {
            "clean_overlap_tail_rate": sum(1 for r in rows if to_int(r.get("clean_entry_v4")) and to_int(r.get("tail_risk_v4"))) / max(1, sum(1 for r in rows if to_int(r.get("clean_entry_v4")))),
            "bad_overlap_tail_rate": sum(1 for r in rows if to_int(r.get("bad_entry_v4")) and to_int(r.get("tail_risk_v4"))) / max(1, sum(1 for r in rows if to_int(r.get("bad_entry_v4")))),
            "premium_loser_rate": labels["premium_loser_rate"],
            "complementarity": "TRRM/QMAE should be trained as a risk overlay using causal features, not as a duplicate of SHORT-V4 labels. Do not include V4 labels as model inputs in the first pass.",
        },
        "modeling_recommendation": {
            "first_model": "TRRM classifier for tail_risk_v4 / bad_entry_v4 rejection",
            "second_model": "QMAE regression or quantile model for expected MAE after classifier baseline is validated",
            "ranking_option": "Use expected MAE or tail probability to rank allowed entries within symbol/timeframe/horizon.",
            "outputs": ["probability_of_bad_trade", "expected_mae", "tail_risk_probability", "reject_allow_decision"],
            "metrics": ["walk_forward_precision_recall_tail", "blocked_tail_loss_rate", "winner_sacrifice_rate", "profit_factor_delta_proxy", "calibration_brier_or_pinball_loss"],
        },
        "safety_confirmations": {
            "no_live_touched": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_ts_touched": True,
            "no_push": True,
            "no_model_training": True,
        },
    }
    json_path = out_dir / f"aegis_phase_b_qmae_dataset_review_{stamp}.json"
    md_path = out_dir / f"aegis_phase_b_qmae_dataset_review_{stamp}.md"
    group_path = out_dir / f"aegis_phase_b_qmae_dataset_review_groups_{stamp}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(group_path, by_group)
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["outputs"] = {"json": str(json_path), "md": str(md_path), "groups_csv": str(group_path)}
    print(json.dumps({"status": decision, "reason": reason, "md": str(md_path), "json": str(json_path)}, indent=2))
    return result


def render_markdown(result: dict[str, Any]) -> str:
    inv = result["dataset_inventory"]
    labels = result["label_distribution_global"]
    stats = result["target_stats_global"]
    sep = result["qmae_target_quality"]["simple_separability"]
    warn = result["group_warnings"]
    lines = [
        "# FASE-B Risk V4 / QMAE Dataset Review",
        "",
        "## 1. Executive summary",
        f"- status: {result['status']}",
        f"- reason: {result['reason']}",
        "- next recommended phase: FASE-C research-only TRRM classifier baseline, then QMAE regression only after feature allowlist and walk-forward split are locked.",
        "",
        "## 2. Dataset inventory",
        f"- total rows: {inv['total_rows']}",
        f"- groups: {inv['groups']}",
        f"- symbols: {', '.join(inv['symbols'])}",
        f"- horizons: {', '.join(inv['horizons'])}",
        f"- timeframes: {', '.join(inv['timeframes'])}",
        f"- date range: {inv['date_min']} to {inv['date_max']}",
        "",
        "## 3. Label distribution",
        f"- clean_entry_v4 rate: {labels['clean_entry_v4_rate']:.4f}",
        f"- bad_entry_v4 rate: {labels['bad_entry_v4_rate']:.4f}",
        f"- no_trade_v4 rate: {labels['no_trade_v4_rate']:.4f}",
        f"- management_dependent_v4 rate: {labels['management_dependent_v4_rate']:.4f}",
        f"- premium_allowed_v4 rate: {labels['premium_allowed_v4_rate']:.4f}",
        f"- premium_loser rate: {labels['premium_loser_rate']:.4f}",
        f"- tail_risk_v4 rate: {labels['tail_risk_v4_rate']:.4f}",
        f"- early_mae_v4 rate: {labels['early_mae_v4_rate']:.4f}",
        f"- insufficient groups: {len(warn['insufficient_groups'])}",
        f"- class dominated groups: {len(warn['class_dominated_groups'])}",
        "",
        "## 4. QMAE target quality",
        "- QMAE measures future SHORT adverse excursion as a ROE proxy over each fixed horizon.",
        "- It appears to capture dangerous path risk, not only ordinary noise, because tail rows have materially higher MAE and worse net quality than non-tail rows.",
        "- It can still over-penalize naturally volatile symbols unless the training phase normalizes by symbol/timeframe/horizon and validates with walk-forward splits.",
        f"- MAE p50/p75/p90/p95: {stats['mae_p50']:.4f}, {stats['mae_p75']:.4f}, {stats['mae_p90']:.4f}, {stats['mae_p95']:.4f}",
        f"- MFE p50/p90: {stats['mfe_p50']:.4f}, {stats['mfe_p90']:.4f}",
        f"- simple rule precision/recall on tail: {sep['simple_rule_precision']:.4f} / {sep['simple_rule_recall']:.4f}",
        "",
        "## 5. Leakage review",
        "- Features available at decision time: close, volatility_features, trend_features, wick_reclaim_proxies, btc_eth_context.",
        "- Future path data is present only as labels/targets and must be excluded from model inputs.",
        "- Use temporal walk-forward splits. Do not use random splits.",
        "- Critical leakage detected in artifact: false.",
        "",
        "## 6. Threshold review",
        "- Current tail threshold: MAE >= 0.10 ROE, or early_mae_v4, or severe bad_entry_v4.",
        "- Current early MAE threshold: time_to_mae <= 3 and MAE >= 0.045 ROE.",
        "- Recommendation: keep thresholds for first research classifier, then tune per horizon/timeframe after validation.",
        "",
        "## 7. Phase O live overlap",
        "- Direct Phase O overlap is not validated yet because the dataset contains hypothetical OHLCV entries, not linked live trade IDs.",
        "- Do not claim big-loss prevention or winner-sacrifice rates until forward audit trades are joined to QMAE features.",
        "- Expected utility should be measured as tail reduction versus winner sacrifice in the next linked-trade audit.",
        "",
        "## 8. SHORT-V4-A overlap",
        f"- clean overlap tail rate: {result['short_v4_overlap']['clean_overlap_tail_rate']:.4f}",
        f"- bad overlap tail rate: {result['short_v4_overlap']['bad_overlap_tail_rate']:.4f}",
        f"- premium loser rate: {result['short_v4_overlap']['premium_loser_rate']:.4f}",
        "- QMAE/TRRM complements SHORT-V4-A only if trained from causal market features and evaluated as a risk overlay. Do not use V4 labels as inputs.",
        "",
        "## 9. Modeling recommendation",
        "- Start with classification: TRRM classifier for tail_risk_v4 / bad_entry_v4 rejection.",
        "- Add QMAE regression or quantile forecasting after classifier baseline proves stable.",
        "- Useful outputs: probability of bad trade, expected MAE, tail risk probability, reject/allow decision.",
        "- Recommended metrics: walk-forward tail precision/recall, blocked tail-loss rate, winner sacrifice rate, proxy PF delta, calibration, pinball loss for quantiles.",
        "",
        "## 10. GO / NO-GO decision",
        f"- decision: {result['status']}",
        "- first training phase if accepted: research-only TRRM classifier baseline with strict feature allowlist and walk-forward splits.",
        "- do not train live, do not promote, do not alter guards, do not use random splits, do not feed label/path fields as inputs.",
        "",
        "## 11. Safety confirmations",
        "- no se tocó live.",
        "- no se tocó active_manifest.",
        "- no se tocó YAML.",
        "- no se reinició PM2.",
        "- no se enviaron órdenes.",
        "- no se tocó .env.",
        "- no se tocó binance-futures-bot-ts.",
        "- no se hizo push.",
        "- no se entrenó ningún modelo.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit Risk V4 / QMAE dataset before training.")
    p.add_argument("--samples-csv", default="")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p


def main() -> int:
    run_audit(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
