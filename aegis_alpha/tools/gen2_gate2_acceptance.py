#!/usr/bin/env python3
"""GEN2 gate-2 acceptance audits: EQM1 predictive decision + ECON1 economic decision.

Reproducible decisions per frozen specs; no thresholds altered after results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import utc_stamp  # noqa: E402


def decide_eqm(training: dict[str, Any]) -> tuple[str, str, dict[str, bool]]:
    reg = training["results"]["reg"].get(training["reg_winner"], {}).get("aggregate", {})
    h1 = bool(reg.get("h1_top_decile_positive_all_folds"))
    h2 = bool(reg.get("h2_stability"))
    h3 = bool(training.get("h3_incrementality_pass"))
    hyps = {"H1_signal": h1, "H2_stability": h2, "H3_incrementality": h3}
    if not h1:
        return "GEN2_EQM_REJECTED", "top-decile net quality not positive on all folds (H0 not rejected)", hyps
    if h1 and h2 and h3:
        return "GEN2_EQM_READY", "H1+H2+H3 pass; predictive edge consistent, stable and incremental", hyps
    failed = [k for k, v in hyps.items() if not v]
    if len(failed) == 1:
        return "GEN2_EQM_PROMISING", f"H1 holds; fails {failed[0]}", hyps
    return "GEN2_EQM_PROMISING", f"H1 holds; fails {failed}", hyps


def decide_econ(econ: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    s = econ["strategies"]
    main = s["eqm_plus_trrm"]
    base = main["B_base"]
    pess = main["C_pesimista"]
    folds = main["per_fold_expectancy_base"]
    rules = s["rule_momentum_plus_trrm"]["B_base"]
    checks = {
        "1_expectancy_positive": base.get("expectancy", 0) > 0,
        "2_folds_majority_positive": sum(1 for v in folds if v > 0) >= 3,
        "3_ci_lower_ok": base.get("expectancy_ci_lo", -9) >= -0.02,
        "4_pf_min_1_3": (base.get("profit_factor") or 0) >= 1.3,
        "4b_pf_target_1_5": (base.get("profit_factor") or 0) >= 1.5,
        "5_no_catastrophic_fold": min(folds) > -abs(base.get("avg_win", 0)) * 3 if folds else False,
        "6_drawdown_le_150": base.get("max_drawdown", 1e9) <= 150.0,
        "7_beats_random": base.get("expectancy", 0) > s["random_plus_trrm"]["B_base"].get("expectancy", 0),
        "8_beats_two_rule_baselines": sum(1 for b in ("rule_momentum_plus_trrm", "rule_volatility_plus_trrm") if base.get("expectancy", 0) > s[b]["B_base"].get("expectancy", -9)) >= 2,
        "9_symbol_share_le_40": (base.get("max_symbol_share") or 1) <= 0.40,
        "10_month_share_le_40": (base.get("max_month_share") or 1) <= 0.40,
        "11_trade_share_le_30": (base.get("max_trade_share") or 1) <= 0.30,
        "12_pessimist_pf_ge_1": (pess.get("profit_factor") or 0) >= 1.0,
        "13_min_trades_300": base.get("trades", 0) >= 300,
        "14_trrm_incremental": base.get("expectancy", 0) > s["eqm_only"]["B_base"].get("expectancy", -9),
        "15_eqm_incremental": base.get("expectancy", 0) > rules.get("expectancy", -9),
    }
    hard = {k: v for k, v in checks.items() if k != "4b_pf_target_1_5"}
    rules_checks_pass = (
        rules.get("expectancy", 0) > 0 and (rules.get("profit_factor") or 0) >= 1.3
        and rules.get("expectancy", 0) > s["random_plus_trrm"]["B_base"].get("expectancy", 0)
    )
    if all(hard.values()):
        return "GEN2_ECONOMIC_EDGE_READY", "all 15 pre-registered criteria pass in base scenario", checks
    failed = [k for k, v in hard.items() if not v]
    if not checks["15_eqm_incremental"] and rules_checks_pass:
        return "RULES_PLUS_TRRM_CHAMPION", "rules+TRRM passes economics; EQM not incremental — EQM returns to research", checks
    if checks["1_expectancy_positive"] and checks["7_beats_random"] and len(failed) <= 3:
        return "GEN2_ECONOMIC_EDGE_PROMISING", f"positive and better than random but fails: {failed}", checks
    return "GEN2_ECONOMIC_EDGE_REJECTED", f"failed criteria: {failed}", checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eqm-training-json", required=True)
    p.add_argument("--econ-report-json", default="")
    args = p.parse_args(argv)
    stamp = utc_stamp()
    training = json.loads(Path(args.eqm_training_json).read_text())
    eqm_dec, eqm_reason, hyps = decide_eqm(training)
    out: dict[str, Any] = {"schema": "gen2_gate2_acceptance_v1", "generated_at": stamp,
                           "eqm_decision": eqm_dec, "eqm_reason": eqm_reason, "eqm_hypotheses": hyps}
    if args.econ_report_json:
        econ = json.loads(Path(args.econ_report_json).read_text())
        econ_dec, econ_reason, checks = decide_econ(econ)
        out.update({"econ_decision": econ_dec, "econ_reason": econ_reason, "econ_checks": checks})
    report = Path(args.eqm_training_json).parent / f"gate2_acceptance_{stamp}.json"
    report.write_text(json.dumps(out, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(out, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
