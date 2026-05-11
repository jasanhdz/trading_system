#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegis_alpha.entry_quality.entry_quality_shadow import evaluate_entry_quality_shadow


DEFAULT_SYMBOLS = ("ETHUSDT", "BTCUSDT")


def _turbo_context() -> dict:
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


def main(argv: list[str] | None = None) -> int:
    symbols = argv or list(DEFAULT_SYMBOLS)
    failed = False
    for symbol in symbols:
        result = evaluate_entry_quality_shadow(symbol, _turbo_context())
        print(
            f"{symbol} status={result.get('feature_status')} "
            f"parity={result.get('feature_parity_pct')} "
            f"missing={result.get('missing_features_count')} "
            f"approximated={','.join(result.get('approximated_features') or []) or '-'} "
            f"feature_ms={result.get('feature_build_latency_ms')} "
            f"total_ms={result.get('total_latency_ms')} "
            f"rec={result.get('recommendation')}"
        )
        if result.get("feature_status") == "insufficient" or result.get("execute") is not False or result.get("production_allowed") is not False:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
