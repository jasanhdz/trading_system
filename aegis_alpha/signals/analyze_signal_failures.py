#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle, safe_float  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.signals.combination_utils import (  # noqa: E402
    ComboSpec,
    RuleCondition,
    load_models,
    predict_scores,
    simulate_combo_window,
    threshold_for_rule,
)


DEFAULT_DATASET = Path("aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
DEFAULT_MODELS = Path("aegis_alpha/models/signals")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/signal_combination_report_v050.json")
DEFAULT_OUTPUT = Path("aegis_alpha/logs/signals/signal_failure_analysis_v051.json")
DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
WINDOW_STEPS = 4032
TARGET_COMBOS = ("A_long_edge_h12_top3", "long_edge_h48_top3", "H_ensemble_h12h24_risk50")
REPORT_COMBO_ALIASES = {"long_edge_h48_top3": "C_long_edge_h48_top3"}


def _combo_registry() -> dict[str, ComboSpec]:
    return {
        "A_long_edge_h12_top3": ComboSpec("A_long_edge_h12_top3", (RuleCondition("long_edge_h12", "top_pct", 0.03),)),
        "long_edge_h48_top3": ComboSpec("long_edge_h48_top3", (RuleCondition("long_edge_h48", "top_pct", 0.03),)),
        "H_ensemble_h12h24_risk50": ComboSpec(
            "H_ensemble_h12h24_risk50",
            (RuleCondition("long_edge_h12", "top_pct", 0.05), RuleCondition("long_edge_h24", "top_pct", 0.05)),
            (RuleCondition("long_failure_risk_h24", "bottom_pct", 0.50),),
        ),
    }


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _window_id(window: dict[str, Any]) -> str:
    return f"{window['source']}|{window['start_step']}|{window['end_step']}"


def _extract_worst_windows(report: dict[str, Any], combo_name: str) -> dict[str, list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    report_combo = REPORT_COMBO_ALIASES.get(combo_name, combo_name)
    for row in report.get("reports", []):
        if row.get("combo") == report_combo:
            windows = list(row.get("windows", []))
            break
    if not windows:
        return {"balance": [], "max_dd": [], "profit_factor": []}

    def take(metric: str, reverse: bool = False) -> list[dict[str, Any]]:
        ordered = sorted(windows, key=lambda w: float(w.get(metric, 0.0)), reverse=reverse)
        return ordered[:10]

    return {
        "balance": take("balance", reverse=False),
        "max_dd": take("max_dd", reverse=True),
        "profit_factor": take("profit_factor", reverse=False),
    }


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([float(t["return"]) for t in trades], dtype=np.float32)
    losses = [t for t in trades if float(t["return"]) < 0.0]
    def bucket(score: float) -> str:
        if score < 0.55:
            return "<0.55"
        if score < 0.60:
            return "[0.55,0.60)"
        if score < 0.65:
            return "[0.60,0.65)"
        if score < 0.70:
            return "[0.65,0.70)"
        return ">=0.70"

    return {
        "trade_count": int(len(trades)),
        "loss_count": int(len(losses)),
        "profit_factor": safe_float(np.sum(returns[returns > 0.0]) / max(-np.sum(returns[returns < 0.0]), 1e-10)) if len(returns) else 0.0,
        "avg_return": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
        "losses_by_regime": dict(Counter(t["entry_regime"] for t in losses)),
        "losses_by_exit_reason": dict(Counter(t["reason"] for t in losses)),
        "losses_by_vol_bucket": dict(Counter(t["vol_bucket"] for t in losses)),
        "losses_by_agreement": dict(Counter(str(bool(t["agreement_h12_h48"])) for t in losses)),
        "losses_by_h12_score_bucket": dict(Counter(bucket(float(t["h12_score"])) for t in losses)),
        "losses_by_h48_score_bucket": dict(Counter(bucket(float(t["h48_score"])) for t in losses)),
        "losses_by_edge_gap_bucket": dict(
            Counter(
                "<-0.001" if float(t["edge_gap"]) < -0.001 else
                "[-0.001,0)" if float(t["edge_gap"]) < 0.0 else
                "[0,0.001)" if float(t["edge_gap"]) < 0.001 else
                ">=0.001"
                for t in losses
            )
        ),
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    best = summary.get("best_combo", {})
    if float(best.get("worst_balance", 0.0)) < 19.0:
        recs.append("La cola sigue dañada: mantener bloqueo adicional o reducir exposición en ventanas donde h12/h48 discrepan.")
    if float(best.get("worst_max_dd", 1.0)) >= 0.10:
        recs.append("El drawdown sigue viniendo de reversals profundos; tail-risk todavía no filtra lo suficiente.")
    if float(best.get("p25_pf", 0.0)) < 1.0:
        recs.append("La calidad media es mejor que la cola, pero el nuevo target de tail-risk todavía no estabiliza el percentil 25.")
    if not recs:
        recs.append("No hay un bucket único dominante; la cola está repartida entre regime shift y edge deterioration.")
    return recs


def analyze_signal_failures(
    config_path: str,
    dataset_path: Path,
    model_dir: Path,
    combo_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    dataset = _load_npz(dataset_path)
    market = load_signal_market(config_path)
    combo_report = json.loads(combo_report_path.read_text(encoding="utf-8"))

    edge_models = load_models(model_dir, ("long_edge_h12", "long_edge_h24", "long_edge_h48", "long_failure_risk_h24"))
    tail_models: dict[str, Any] = {}
    for name in ("long_tail_risk_h12", "long_tail_risk_h24", "long_tail_risk_h48"):
        path = model_dir / f"aegis_{name}_v051.joblib"
        if path.exists():
            tail_models[name] = load_model_bundle(path)["estimator"]

    preds = predict_scores(market, edge_models)
    tail_preds = predict_scores(market, tail_models) if tail_models else {}

    combo_specs = _combo_registry()
    selected_combos = {name: combo_specs[name] for name in TARGET_COMBOS}

    analysis: dict[str, Any] = {
        "schema_version": "aegis_signal_failure_analysis_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_path": str(dataset_path),
        "combo_report_path": str(combo_report_path),
        "model_dir": str(model_dir),
        "selected_combos": list(selected_combos),
        "worst_windows": {},
        "trade_details": {},
        "loss_summary": {},
        "recommendations": [],
    }

    all_trades: list[dict[str, Any]] = []
    for combo_name, combo in selected_combos.items():
        windows = _extract_worst_windows(combo_report, combo_name)
        combo_report_entries: dict[str, list[dict[str, Any]]] = {}
        combo_trades: list[dict[str, Any]] = []

        all_preds = {**preds, **tail_preds}
        edge_rule_thresholds = {
            rule.model_name: threshold_for_rule(all_preds[rule.model_name], rule)
            for rule in combo.edge_rules
        }
        risk_rule_thresholds = {
            rule.model_name: threshold_for_rule(all_preds[rule.model_name], rule)
            for rule in combo.risk_rules
            if rule.model_name in all_preds
        }
        agreement_thresholds = {
            "long_edge_h12": threshold_for_rule(preds["long_edge_h12"], RuleCondition("long_edge_h12", "top_pct", 0.03)),
            "long_edge_h48": threshold_for_rule(preds["long_edge_h48"], RuleCondition("long_edge_h48", "top_pct", 0.03)),
        }
        thresholds = {**edge_rule_thresholds, **risk_rule_thresholds, **agreement_thresholds}

        for bucket_name, bucket_windows in windows.items():
            detailed: list[dict[str, Any]] = []
            for window in bucket_windows:
                start_step = int(window["start_step"])
                source = str(window["source"])
                sim = simulate_combo_window(
                    market=market,
                    preds=all_preds,
                    combo=combo,
                    thresholds=thresholds,
                    start_step=start_step,
                    window_steps=WINDOW_STEPS,
                    fee_multiplier=1.0,
                    source=source,
                )
                detailed.append(
                    {
                        "window": _window_id(window),
                        "source": source,
                        "balance": safe_float(window.get("balance", 0.0)),
                        "max_dd": safe_float(window.get("max_dd", 0.0)),
                        "profit_factor": safe_float(window.get("profit_factor", 0.0)),
                        "trade_count": int(window.get("trades", 0)),
                        "simulated_trades": sim["trades_detail"],
                    }
                )
                combo_trades.extend(sim["trades_detail"])
                all_trades.extend(sim["trades_detail"])
            combo_report_entries[bucket_name] = detailed
        analysis["worst_windows"][combo_name] = combo_report_entries
        analysis["trade_details"][combo_name] = combo_trades

    flat_trades = [t for trades in analysis["trade_details"].values() for t in trades]
    losses = [t for t in flat_trades if float(t["return"]) < 0.0]
    analysis["loss_summary"] = {
        "total_trades": int(len(flat_trades)),
        "loss_count": int(len(losses)),
        "losses_by_combo": {
            combo: int(sum(1 for trade in trades if float(trade["return"]) < 0.0))
            for combo, trades in analysis["trade_details"].items()
        },
        "losses_by_regime": dict(Counter(t["entry_regime"] for t in losses)),
        "losses_by_vol_bucket": dict(Counter(t["vol_bucket"] for t in losses)),
        "losses_by_exit_reason": dict(Counter(t["reason"] for t in losses)),
        "losses_by_agreement": dict(Counter(str(bool(t["agreement_h12_h48"])) for t in losses)),
        "losses_by_h12_score_bucket": dict(Counter(
            "<0.55" if float(t["h12_score"]) < 0.55 else
            "[0.55,0.60)" if float(t["h12_score"]) < 0.60 else
            "[0.60,0.65)" if float(t["h12_score"]) < 0.65 else
            "[0.65,0.70)" if float(t["h12_score"]) < 0.70 else
            ">=0.70"
            for t in losses
        )),
        "losses_by_h48_score_bucket": dict(Counter(
            "<0.55" if float(t["h48_score"]) < 0.55 else
            "[0.55,0.60)" if float(t["h48_score"]) < 0.60 else
            "[0.60,0.65)" if float(t["h48_score"]) < 0.65 else
            "[0.65,0.70)" if float(t["h48_score"]) < 0.70 else
            ">=0.70"
            for t in losses
        )),
        "losses_by_edge_gap_bucket": dict(Counter(
            "<-0.001" if float(t["edge_gap"]) < -0.001 else
            "[-0.001,0)" if float(t["edge_gap"]) < 0.0 else
            "[0,0.001)" if float(t["edge_gap"]) < 0.001 else
            ">=0.001"
            for t in losses
        )),
        "losses_by_horizon_pair": {
            "h12_gt_h48": int(sum(1 for t in losses if float(t["h12_score"]) >= float(t["h48_score"]))),
            "h48_gt_h12": int(sum(1 for t in losses if float(t["h48_score"]) > float(t["h12_score"]))),
        },
    }
    best_summary = {}
    for combo_name, bucket_windows in analysis["worst_windows"].items():
        if bucket_windows.get("balance"):
            best_summary = {
                "combo": combo_name,
                "worst_balance": min(float(w["balance"]) for w in bucket_windows["balance"]),
                "worst_max_dd": max(float(w["max_dd"]) for w in bucket_windows["max_dd"]),
                "worst_pf": min(float(w["profit_factor"]) for w in bucket_windows["profit_factor"]),
            }
            break
    analysis["best_worst_summary"] = best_summary
    analysis["recommendations"] = _recommendations({"best_combo": best_summary})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Signal failure analysis -> {output_path}")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODELS))
    parser.add_argument("--combo-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    analyze_signal_failures(
        config_path=args.config,
        dataset_path=Path(args.dataset),
        model_dir=Path(args.model_dir),
        combo_report_path=Path(args.combo_report),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
