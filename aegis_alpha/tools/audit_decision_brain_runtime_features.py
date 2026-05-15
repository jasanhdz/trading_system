#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.decision_brain.feature_builder import build_decision_brain_features
from aegis_alpha.decision_brain.model_loader import load_decision_brain_artifacts
from aegis_alpha.entry_quality.entry_quality_shadow import evaluate_entry_quality_shadow
from aegis_alpha.event_risk.event_risk_detector import evaluate_event_risk_auto
from aegis_alpha.turbo.turbo_signal import evaluate_turbo_shadow


LOG_DIR = REPO_ROOT / "aegis_alpha/logs/decision_brain"
DATASET_META_PATH = REPO_ROOT / "aegis_alpha/data/processed/decision_brain/decision_brain_dataset_v010_meta.json"
MODEL_MANIFEST_PATH = REPO_ROOT / "aegis_alpha/models/decision_brain/v010/model_manifest.json"


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


def _symbol_meta(symbol: str, feature_columns: list[str]) -> dict[str, Any]:
    turbo = evaluate_turbo_shadow(symbol)
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
    _vector, meta = build_decision_brain_features(
        symbol=symbol,
        side=str((turbo.get("raw") or {}).get("action") or turbo.get("action") or "UNKNOWN"),
        turbo_context=turbo,
        entry_quality_model=entry_quality,
        event_risk_auto=event_risk_auto,
        news_sentiment=None,
        feature_columns=feature_columns,
    )
    return meta


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}


def _recommendations(missing_by_group: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in missing_by_group:
        if group == "market_mtf":
            out[group] = "Use entry_quality.runtime_feature_cache market/MTF fields."
        elif group == "portfolio":
            out[group] = "Expose a safe TS state snapshot later; runtime now uses neutral unavailable values."
        elif group == "news_sentiment":
            out[group] = "Ensure collector has written latest_event_sentiment_risk.json."
        elif group == "event_risk":
            out[group] = "Use event_risk_auto block and read overlay YAML read-only."
        elif group == "btc_eth_context":
            out[group] = "Use event_risk_auto BTC/ETH context; enrich peer returns in a later phase."
        else:
            out[group] = "Inspect feature_columns mapping and add a safe runtime source."
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SUIUSDT,LINKUSDT")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    artifacts = load_decision_brain_artifacts()
    if artifacts is None:
        raise SystemExit("decision_brain_artifacts_missing")
    rows = {symbol: _symbol_meta(symbol, artifacts.feature_columns) for symbol in symbols}
    combined_missing: dict[str, list[str]] = {}
    for row in rows.values():
        for group, features in (row.get("missing_features_by_group") or {}).items():
            combined_missing.setdefault(group, [])
            combined_missing[group].extend([item for item in features if item not in combined_missing[group]])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_expected_features": len(artifacts.feature_columns),
        "feature_columns_path": str(REPO_ROOT / "aegis_alpha/models/decision_brain/v010/feature_columns.json"),
        "model_manifest_present": MODEL_MANIFEST_PATH.exists(),
        "dataset_meta_present": DATASET_META_PATH.exists(),
        "dataset_meta_features_count": _load_json(DATASET_META_PATH).get("features_count"),
        "model_manifest_status": _load_json(MODEL_MANIFEST_PATH).get("status"),
        "symbols": rows,
        "combined_missing_by_group": {key: sorted(value) for key, value in combined_missing.items()},
        "recommended_fix_by_group": _recommendations(combined_missing),
        "leakage_notes": [
            "No outcome/future labels are used at runtime.",
            "Portfolio context is neutral/unavailable, not read from Binance.",
            "Market/MTF features come from local SQLite recent candles through EntryQuality runtime cache.",
        ],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOG_DIR / f"decision_brain_runtime_feature_audit_{stamp}.json"
    md_path = LOG_DIR / f"decision_brain_runtime_feature_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Decision Brain Runtime Feature Audit",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Total expected features: `{report['total_expected_features']}`",
        f"- Model manifest present: `{report['model_manifest_present']}`",
        f"- Dataset meta present: `{report['dataset_meta_present']}`",
        "",
        "## Symbols",
    ]
    for symbol, row in rows.items():
        lines.extend([
            f"### {symbol}",
            f"- Feature status: `{row.get('feature_status')}`",
            f"- Feature parity pct: `{row.get('feature_parity_pct')}`",
            f"- Missing features count: `{row.get('missing_features_count')}`",
            f"- Critical missing groups: `{', '.join(row.get('critical_missing_groups') or []) or 'none'}`",
            f"- Available groups: `{', '.join(row.get('available_feature_groups') or [])}`",
            "",
        ])
    lines.extend(["## Recommended Fix By Group"])
    for group, recommendation in report["recommended_fix_by_group"].items():
        lines.append(f"- `{group}`: {recommendation}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "md": str(md_path), "symbols": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
