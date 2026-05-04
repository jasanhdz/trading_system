#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import load_model_bundle  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.signals.combination_utils import (  # noqa: E402
    ComboSpec,
    RuleCondition,
    load_models,
    predict_scores,
    simulate_combo_window,
    threshold_for_rule,
)
from aegis_alpha.tools.build_long_edge_candidate_dataset import BASE_GUARD  # noqa: E402
from aegis_alpha.tools.evaluate_long_edge_robustness import select_robust_windows  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_MODEL_DIR = Path("aegis_alpha/models/signals")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/horizon_agreement_report_v051.json")
DEFAULT_SEEDS = (9101, 9203)
WINDOW_STEPS = 4032
FEE_MULTIPLIERS = (1.0, 1.25)


COMBOS: tuple[ComboSpec, ...] = (
    ComboSpec("A_h12_top3", (RuleCondition("long_edge_h12", "top_pct", 0.03),)),
    ComboSpec("B_h12_top3_h48_gt0", (RuleCondition("long_edge_h12", "top_pct", 0.03), RuleCondition("long_edge_h48", "gt", 0.0))),
    ComboSpec("C_h12_top3_h48_top30", (RuleCondition("long_edge_h12", "top_pct", 0.03), RuleCondition("long_edge_h48", "top_pct", 0.30))),
    ComboSpec("D_h12_top3_h48_top20", (RuleCondition("long_edge_h12", "top_pct", 0.03), RuleCondition("long_edge_h48", "top_pct", 0.20))),
    ComboSpec("E_h12_top3_tail50", (RuleCondition("long_edge_h12", "top_pct", 0.03),), (RuleCondition("long_tail_risk_h12", "bottom_pct", 0.50),)),
    ComboSpec("F_h12_top3_tail30", (RuleCondition("long_edge_h12", "top_pct", 0.03),), (RuleCondition("long_tail_risk_h12", "bottom_pct", 0.30),)),
    ComboSpec("G_h12_top3_h48_gt0_tail50", (RuleCondition("long_edge_h12", "top_pct", 0.03), RuleCondition("long_edge_h48", "gt", 0.0)), (RuleCondition("long_tail_risk_h12", "bottom_pct", 0.50),)),
    ComboSpec("H_h48_top3_h12_gt0", (RuleCondition("long_edge_h48", "top_pct", 0.03), RuleCondition("long_edge_h12", "gt", 0.0))),
)


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    return {
        "median_balance": float(np.median(balances)),
        "p25_balance": float(np.quantile(balances, 0.25)),
        "worst_balance": float(np.min(balances)),
        "median_pf": float(np.median(pfs)),
        "p25_pf": float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": float(np.mean(balances > initial_balance)),
        "median_trades": float(np.median(trades)),
        "trades_per_month": float(np.median([w["trades_per_month"] for w in windows])),
        "worst_max_dd": float(np.max(max_dd)),
        "avg_return_per_trade": float(np.median(avg_returns)),
        "exposure": float(np.median(exposure)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_regime": int(sum(w["skipped_by_regime"] for w in windows)),
        "skipped_by_signal": int(sum(w["skipped_by_signal"] for w in windows)),
    }


def _select_windows(market: Any, seeds: tuple[int, ...]) -> list[tuple[int, str]]:
    window_map: dict[int, set[str]] = {}
    for seed in seeds:
        selected = select_robust_windows(
            market,
            window_steps=WINDOW_STEPS,
            seed=seed,
            target_max=144,
            recent_windows=24,
            random_windows=24,
            regime_windows_per_regime=6,
            non_overlap_windows=24,
        )
        for start_step, source in selected:
            window_map.setdefault(int(start_step), set()).add(str(source))
    windows = sorted((start, "+".join(sorted(sources))) for start, sources in window_map.items())
    if len(windows) > 144:
        keep = np.linspace(0, len(windows) - 1, 144).round().astype(int)
        windows = [windows[idx] for idx in keep]
    if len(windows) < 100:
        raise RuntimeError(f"Window selection produced only {len(windows)} windows; need >=100")
    return windows


def evaluate_horizon_agreement(
    config_path: str,
    model_dir: Path,
    report_path: Path,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    market = load_signal_market(config_path)
    windows = _select_windows(market, seeds)
    models = load_models(model_dir, ("long_edge_h12", "long_edge_h24", "long_edge_h48"))
    tail_path = model_dir / "aegis_long_tail_risk_h12_v051.joblib"
    if not tail_path.exists():
        raise FileNotFoundError(f"Missing tail-risk model: {tail_path}")
    models["long_tail_risk_h12"] = load_model_bundle(tail_path)["estimator"]
    preds = predict_scores(market, models)
    agreement_thresholds = {
        "long_edge_h12": threshold_for_rule(preds["long_edge_h12"], RuleCondition("long_edge_h12", "top_pct", 0.03)),
        "long_edge_h48": threshold_for_rule(preds["long_edge_h48"], RuleCondition("long_edge_h48", "top_pct", 0.03)),
        "long_tail_risk_h12": threshold_for_rule(preds["long_tail_risk_h12"], RuleCondition("long_tail_risk_h12", "bottom_pct", 0.50)),
    }

    reports: list[dict[str, Any]] = []
    for fee_multiplier in FEE_MULTIPLIERS:
        for combo in COMBOS:
            combo_thresholds = {
                rule.model_name: threshold_for_rule(preds[rule.model_name], rule)
                for rule in (*combo.edge_rules, *combo.risk_rules)
            }
            thresholds = {**agreement_thresholds, **combo_thresholds}
            combo_windows = [
                simulate_combo_window(
                    market=market,
                    preds=preds,
                    combo=combo,
                    thresholds=thresholds,
                    start_step=int(start_step),
                    window_steps=WINDOW_STEPS,
                    fee_multiplier=fee_multiplier,
                    source=source,
                )
                for start_step, source in windows
            ]
            summary = _summary(combo_windows, market.cfg.risk.initial_balance)
            reports.append(
                {
                    "combo": combo.name,
                    "fee_multiplier": fee_multiplier,
                    "edge_rules": [rule.__dict__ for rule in combo.edge_rules],
                    "risk_rules": [rule.__dict__ for rule in combo.risk_rules],
                    "summary": summary,
                    "windows": combo_windows,
                }
            )
            print(
                f"{combo.name} fee={fee_multiplier:.2f} p25pf={summary['p25_pf']:.2f} "
                f"worst={summary['worst_balance']:.2f} dd={summary['worst_max_dd']:.1%} "
                f"prof={summary['profitable_window_pct']:.1%} trades={summary['median_trades']:.1f}"
            )

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        s = row["summary"]
        return (
            float(s["p25_pf"]),
            float(s["worst_balance"]),
            -float(s["worst_max_dd"]),
            float(s["profitable_window_pct"]),
            float(s["median_trades"]),
        )

    ranking = sorted(reports, key=rank_key, reverse=True)
    report = {
        "schema_version": "aegis_horizon_agreement_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config_path": config_path,
        "model_dir": str(model_dir),
        "window_count": len(windows),
        "seeds": list(seeds),
        "allowed_regimes": ["mixed", "chop", "high_vol"],
        "ranking": [
            {
                "combo": row["combo"],
                "fee_multiplier": row["fee_multiplier"],
                **row["summary"],
            }
            for row in ranking
        ],
        "best": {
            "combo": ranking[0]["combo"],
            "fee_multiplier": ranking[0]["fee_multiplier"],
            **ranking[0]["summary"],
        } if ranking else None,
        "reports": reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Horizon agreement report -> {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--seeds", default="9101,9203")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    evaluate_horizon_agreement(
        config_path=args.config,
        model_dir=Path(args.model_dir),
        report_path=Path(args.report),
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
