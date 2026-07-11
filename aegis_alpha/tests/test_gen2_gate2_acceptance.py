#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.gen2_gate2_acceptance import decide_econ, decide_eqm


def base(expect: float, pf: float, trades: int = 400) -> dict:
    return {"expectancy": expect, "profit_factor": pf, "trades": trades, "expectancy_ci_lo": expect - 0.05,
            "max_drawdown": 50.0, "avg_win": 1.0, "max_symbol_share": 0.2, "max_month_share": 0.2, "max_trade_share": 0.05}


def econ_fixture(main_expect: float) -> dict:
    s = {
        "eqm_plus_trrm": {"B_base": base(main_expect, 1.6), "C_pesimista": base(main_expect * 0.5, 1.1), "per_fold_expectancy_base": [main_expect] * 4},
        "eqm_only": {"B_base": base(-0.05, 0.9)},
        "random_plus_trrm": {"B_base": base(-0.1, 0.6)},
        "rule_momentum_plus_trrm": {"B_base": base(-0.12, 0.6)},
        "rule_volatility_plus_trrm": {"B_base": base(-0.06, 0.8)},
        "trrm_only": {"B_base": base(-0.13, 0.4)},
        "rule_momentum": {"B_base": base(-0.18, 0.5)},
    }
    return {"strategies": s}


def test_econ_ready_and_rejected() -> None:
    dec, _, checks = decide_econ(econ_fixture(0.14))
    assert dec == "GEN2_ECONOMIC_EDGE_READY", (dec, [k for k, v in checks.items() if not v])
    dec2, _, _ = decide_econ(econ_fixture(-0.30))
    assert dec2 == "GEN2_ECONOMIC_EDGE_REJECTED"


def test_eqm_decisions() -> None:
    def training(h1: bool, h2: bool, h3: bool) -> dict:
        return {"reg_winner": "m", "results": {"reg": {"m": {"aggregate": {"h1_top_decile_positive_all_folds": h1, "h2_stability": h2}}}}, "h3_incrementality_pass": h3}

    assert decide_eqm(training(True, True, True))[0] == "GEN2_EQM_READY"
    assert decide_eqm(training(True, False, True))[0] == "GEN2_EQM_PROMISING"
    assert decide_eqm(training(False, True, True))[0] == "GEN2_EQM_REJECTED"


def test_freeze_requires_econ_approval() -> None:
    import argparse
    import json
    import tempfile

    import aegis_alpha.tools.gen2_system_freeze as fz
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        fz.FREEZE_PATH = tmp / "GEN2_SYSTEM_FREEZE.json"
        gate = tmp / "gate2.json"
        gate.write_text(json.dumps({"econ_decision": "GEN2_ECONOMIC_EDGE_REJECTED"}))
        try:
            fz.freeze_system(argparse.Namespace(trrm_dir=str(tmp), eqm_dir=str(tmp), econ_dir=str(tmp), gate2_json=str(gate)))
            raise AssertionError("freeze must fail without economic approval")
        except (ValueError, FileNotFoundError):
            pass


if __name__ == "__main__":
    test_econ_ready_and_rejected()
    test_eqm_decisions()
    test_freeze_requires_econ_approval()
    print("test_gen2_gate2_acceptance: OK")
