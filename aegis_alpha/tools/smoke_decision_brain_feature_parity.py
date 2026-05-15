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


def _event_context(symbol: str, turbo: dict[str, Any], entry_quality: dict[str, Any]) -> dict[str, Any]:
    raw = turbo.get("raw") if isinstance(turbo.get("raw"), dict) else {}
    gated = turbo.get("gated") if isinstance(turbo.get("gated"), dict) else {}
    return {
        "symbol": symbol,
        "turbo_action": raw.get("action") or turbo.get("action"),
        "turbo_score": raw.get("turbo_score", turbo.get("turbo_score")),
        "gated_action": gated.get("action"),
        "entry_quality_score": entry_quality.get("entry_quality_score"),
        "tail_risk_score": entry_quality.get("tail_risk_score"),
        "recommendation": entry_quality.get("recommendation"),
        "freshness": turbo.get("freshness") or raw.get("freshness") or {},
    }


def smoke_symbol(symbol: str) -> dict[str, Any]:
    turbo = evaluate_turbo_shadow(symbol)
    turbo["execute"] = False
    turbo["production_allowed"] = False
    entry_quality = evaluate_entry_quality_shadow(symbol, {"turbo": turbo})
    turbo["entry_quality_model"] = entry_quality
    event_risk_auto = evaluate_event_risk_auto({
        "symbol": symbol,
        "btc": _event_context("BTCUSDT", turbo, entry_quality) if symbol == "BTCUSDT" else None,
        "eth": _event_context("ETHUSDT", turbo, entry_quality) if symbol == "ETHUSDT" else None,
        "current": _event_context(symbol, turbo, entry_quality),
        "market": {},
        "api_warnings": [],
    })
    decision = evaluate_decision_brain_shadow(
        symbol=symbol,
        side=str((turbo.get("raw") or {}).get("action") or turbo.get("action") or "UNKNOWN"),
        turbo_context=turbo,
        entry_quality_model=entry_quality,
        event_risk_auto=event_risk_auto,
        news_sentiment=None,
    )
    if decision.get("mode") != "SHADOW":
        raise RuntimeError(f"{symbol}: mode_not_shadow")
    if decision.get("execute") is not False:
        raise RuntimeError(f"{symbol}: execute_not_false")
    if decision.get("production_allowed") is not False:
        raise RuntimeError(f"{symbol}: production_allowed_not_false")
    if decision.get("decision") is None:
        raise RuntimeError(f"{symbol}: decision_brain_missing")
    return {
        "symbol": symbol,
        "decision": decision.get("decision"),
        "enter_now_prob": decision.get("enter_now_prob"),
        "wait_confirmation_prob": decision.get("wait_confirmation_prob"),
        "manual_only_prob": decision.get("manual_only_prob"),
        "do_not_enter_prob": decision.get("do_not_enter_prob"),
        "feature_status": decision.get("feature_status"),
        "feature_parity_pct": decision.get("feature_parity_pct"),
        "missing_features_count": decision.get("missing_features_count"),
        "critical_missing_groups": decision.get("critical_missing_groups"),
        "available_feature_groups": decision.get("available_feature_groups"),
        "approximated_features_count": len(decision.get("approximated_features") or []),
        "latency_ms": decision.get("latency_ms"),
        "execute": decision.get("execute"),
        "production_allowed": decision.get("production_allowed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SUIUSDT,LINKUSDT")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    rows = [smoke_symbol(symbol) for symbol in symbols]
    if rows and all(float(row.get("feature_parity_pct") or 0.0) < 75.0 for row in rows):
        raise SystemExit(json.dumps({"ok": False, "reason": "feature_parity_below_75_for_all_symbols", "rows": rows}, indent=2))
    print(json.dumps({"ok": True, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
