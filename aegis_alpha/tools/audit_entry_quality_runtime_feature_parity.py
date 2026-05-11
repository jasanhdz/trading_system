#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegis_alpha.entry_quality.feature_builder import FEATURE_GROUPS, build_entry_quality_features
from aegis_alpha.entry_quality.model_loader import FEATURE_COLUMNS_PATH, load_entry_quality_models


LOG_DIR = REPO_ROOT / "aegis_alpha/logs/entry_quality"
META_PATH = REPO_ROOT / "aegis_alpha/data/processed/entry_quality/entry_quality_dataset_v020_meta.json"
DEFAULT_SYMBOLS = ("ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT")


def _turbo_context(symbol: str) -> dict[str, Any]:
    del symbol
    return {
        "turbo": {
            "raw": {
                "action": "LONG",
                "turbo_score": 0.72,
                "votes": {"long": 2, "short": 1, "neutral": 0},
                "recent_scores": {
                    "long_7d": 0.001,
                    "long_14d": 0.002,
                    "long_30d": 0.0015,
                    "short_7d": -0.001,
                    "short_14d": 0.0,
                    "short_30d": -0.002,
                },
            }
        }
    }


def _group_missing(feature_columns: list[str], missing: list[str]) -> dict[str, list[str]]:
    missing_set = set(missing)
    present = set(feature_columns)
    grouped: dict[str, list[str]] = {}
    for group, columns in FEATURE_GROUPS.items():
        expected = [col for col in columns if col in present]
        grouped[group] = [col for col in expected if col in missing_set]
    return grouped


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Quality Runtime Feature Parity Audit",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Expected feature columns: `{report['feature_columns_count']}`",
        f"- Average parity: `{report['average_feature_parity_pct']:.2f}%`",
        f"- Symbols audited: `{len(report['symbols'])}`",
        "",
        "## Symbols",
        "",
        "| Symbol | Status | Parity | Missing | Approximated | Critical Groups |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in report["symbols"]:
        lines.append(
            f"| {item['symbol']} | {item['feature_status']} | {item['feature_parity_pct']:.2f}% | "
            f"{item['missing_features_count']} | {', '.join(item['approximated_features']) or '-'} | "
            f"{', '.join(item['critical_missing_groups']) or '-'} |"
        )
    lines.extend(["", "## Missing Features Top", ""])
    if report["missing_features_top"]:
        for item in report["missing_features_top"]:
            lines.append(f"- `{item['feature']}`: {item['count']}")
    else:
        lines.append("- No missing features in audited runtime vectors.")
    lines.extend(["", "## Feature Groups", ""])
    for group, columns in report["feature_groups"].items():
        lines.append(f"- `{group}`: {len(columns)} columns")
    lines.append("")
    lines.append("Note: `quote_volume` may be approximated as `close * volume` when the runtime candle store does not persist quote volume directly.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    symbols = tuple(argv or DEFAULT_SYMBOLS)
    cache = load_entry_quality_models()
    feature_columns = list(cache.feature_columns)
    missing_counter: dict[str, int] = defaultdict(int)
    items: list[dict[str, Any]] = []
    for symbol in symbols:
        vector = build_entry_quality_features(symbol, _turbo_context(symbol))
        for feature in vector.missing_features:
            missing_counter[feature] += 1
        items.append(
            {
                "symbol": symbol,
                "feature_status": vector.feature_status,
                "feature_parity_pct": vector.feature_parity_pct,
                "missing_features_count": len(vector.missing_features),
                "missing_features": vector.missing_features,
                "missing_features_by_group": _group_missing(feature_columns, vector.missing_features),
                "approximated_features": vector.approximated_features or [],
                "critical_missing_groups": vector.critical_missing_groups or [],
                "feature_build_latency_ms": round(float(vector.feature_build_latency_ms), 3),
            }
        )
    avg = sum(float(item["feature_parity_pct"]) for item in items) / max(len(items), 1)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "dataset_meta_path": str(META_PATH),
        "dataset_meta_exists": META_PATH.exists(),
        "feature_columns_count": len(feature_columns),
        "feature_groups": {group: [col for col in columns if col in feature_columns] for group, columns in FEATURE_GROUPS.items()},
        "average_feature_parity_pct": avg,
        "symbols": items,
        "missing_features_top": [
            {"feature": feature, "count": count}
            for feature, count in sorted(missing_counter.items(), key=lambda item: item[1], reverse=True)
        ],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = LOG_DIR / f"runtime_feature_parity_audit_{stamp}.json"
    md_path = LOG_DIR / f"runtime_feature_parity_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(md_path, report)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
