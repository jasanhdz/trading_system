#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.decision_brain.decision_brain_shadow import evaluate_decision_brain_shadow
from aegis_alpha.entry_quality.entry_quality_shadow import evaluate_entry_quality_shadow
from aegis_alpha.event_risk.event_risk_detector import evaluate_event_risk_auto
from aegis_alpha.turbo.turbo_signal import evaluate_turbo_shadow


def _prob_sum(row: dict[str, Any]) -> float:
    return float(row.get("enter_now_prob", 0.0) or 0.0) + float(row.get("wait_confirmation_prob", 0.0) or 0.0) + float(row.get("manual_only_prob", 0.0) or 0.0) + float(row.get("do_not_enter_prob", 0.0) or 0.0)


def smoke_symbol(symbol: str) -> dict[str, Any]:
    turbo = evaluate_turbo_shadow(symbol)
    turbo["execute"] = False
    turbo["production_allowed"] = False
    entry_quality = evaluate_entry_quality_shadow(symbol, {"turbo": turbo})
    turbo["entry_quality_model"] = entry_quality
    event_risk_auto = evaluate_event_risk_auto({
        "symbol": symbol,
        "btc": None,
        "eth": None,
        "current": {
            "symbol": symbol,
            "turbo_action": (turbo.get("raw") or {}).get("action") or turbo.get("action"),
            "turbo_score": (turbo.get("raw") or {}).get("turbo_score") or turbo.get("turbo_score"),
        },
        "market": {},
        "api_warnings": [],
    })
    side = str((turbo.get("raw") or {}).get("action") or turbo.get("action") or "UNKNOWN")
    decision = evaluate_decision_brain_shadow(
        symbol=symbol,
        side=side,
        turbo_context=turbo,
        entry_quality_model=entry_quality,
        event_risk_auto=event_risk_auto,
        news_sentiment=None,
    )
    prob_sum = _prob_sum(decision)
    if decision.get("mode") != "SHADOW":
        raise RuntimeError(f"{symbol}: mode_not_shadow")
    if decision.get("execute") is not False:
        raise RuntimeError(f"{symbol}: execute_not_false")
    if decision.get("production_allowed") is not False:
        raise RuntimeError(f"{symbol}: production_allowed_not_false")
    if decision.get("decision") not in {"ENTER_NOW", "WAIT_CONFIRMATION", "MANUAL_ONLY", "DO_NOT_ENTER", "UNKNOWN"}:
        raise RuntimeError(f"{symbol}: invalid_decision")
    if abs(prob_sum - 1.0) > 0.02 and decision.get("recommendation") != "INSUFFICIENT_DATA":
        raise RuntimeError(f"{symbol}: probability_sum_invalid:{prob_sum}")
    return {
        "symbol": symbol,
        "decision": decision.get("decision"),
        "enter_now_prob": decision.get("enter_now_prob"),
        "wait_confirmation_prob": decision.get("wait_confirmation_prob"),
        "manual_only_prob": decision.get("manual_only_prob"),
        "do_not_enter_prob": decision.get("do_not_enter_prob"),
        "recommendation": decision.get("recommendation"),
        "feature_status": decision.get("feature_status"),
        "feature_parity_pct": decision.get("feature_parity_pct"),
        "latency_ms": decision.get("latency_ms"),
        "execute": decision.get("execute"),
        "production_allowed": decision.get("production_allowed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SUIUSDT")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    rows = [smoke_symbol(symbol) for symbol in symbols]
    print(json.dumps({"ok": True, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
