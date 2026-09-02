#!/usr/bin/env python3
"""Deterministic, offline audit of Aegis probability and context-path parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import aegis
from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import (
    Candle,
    FeedQuality,
    MarketSnapshot,
    PortfolioContext,
    SymbolSeries,
)
from aegis.live_decision import (
    CurrentBrainDecisionService,
    CurrentBrainEngine,
    CurrentBrainPaths,
    compatibility_response,
)

VARIANT_COUNT = 20
BAR_COUNT = 96
FIXED_CLOSE = datetime(2026, 8, 2, 20, 0, tzinfo=timezone.utc)
ROW_KEY = ("variant", "symbol")
AUDITED_FIELDS = (
    "feature_vector_hash",
    "feature_values_hash",
    "feature_count",
    "long_prob",
    "short_prob",
    "neutral_prob",
    "candidate_raw_score",
    "candidate_calibrated_score",
    "candidate_side",
    "selected",
    "decision",
    "verdict",
)
NUMERIC_FIELDS = (
    "long_prob",
    "short_prob",
    "neutral_prob",
    "candidate_raw_score",
    "candidate_calibrated_score",
)


class StaticProvider:
    def __init__(self, snapshot: MarketSnapshot) -> None:
        self.snapshot_value = snapshot
        self.calls = 0

    def snapshot(self) -> MarketSnapshot:
        self.calls += 1
        return self.snapshot_value


def synthetic_snapshot(variant: int) -> MarketSnapshot:
    series = []
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        candles = []
        base = 10.0 + symbol_index * 3.0
        for index in range(BAR_COUNT):
            close_time = FIXED_CLOSE - timedelta(minutes=5 * (BAR_COUNT - 1 - index))
            open_time = close_time - timedelta(minutes=5)
            drift = (symbol_index - 5) * 0.00008 + variant * 0.00001
            open_price = base * (1.0 + drift * index)
            close = open_price * (1.0 + drift + (((index + variant) % 5) - 2) * 0.00003)
            high = max(open_price, close) * 1.001
            low = min(open_price, close) * 0.999
            candles.append(
                Candle(
                    open_time,
                    close_time,
                    open_price,
                    high,
                    low,
                    close,
                    1000.0 + symbol_index * 10 + index + variant,
                    True,
                    "AUDIT_FROZEN_FIXTURE",
                    f"{variant}-{index}",
                )
            )
        series.append(
            SymbolSeries(
                symbol,
                tuple(candles),
                FIXED_CLOSE,
                FeedQuality(source_lag_ms=5_000),
            )
        )
    return MarketSnapshot(
        FIXED_CLOSE,
        "5m",
        CANONICAL_SYMBOL_SET_HASH,
        tuple(series),
        PortfolioContext(available_slots=1, operational_time=FIXED_CLOSE),
    )


def market_context(snapshot: MarketSnapshot, symbol: str) -> dict[str, Any]:
    captured_at_ms = int((snapshot.closed_at + timedelta(seconds=5)).timestamp() * 1000)
    observed_at_ms = int(snapshot.closed_at.timestamp() * 1000)
    universe: dict[str, Any] = {}
    for item in snapshot.series:
        universe[item.symbol] = {
            "source": "WEBSOCKET",
            "status": "FRESH",
            "observedAtMs": observed_at_ms,
            "websocketObservedAtMs": observed_at_ms,
            "restFallbackCount": 0,
            "candles": [
                {
                    "openTime": int(candle.open_time.timestamp() * 1000),
                    "closeTime": int(candle.close_time.timestamp() * 1000) - 1,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in item.candles
            ],
        }
    return {
        "version": "AEGIS_MARKET_CONTEXT_V1",
        "source": "SHARED_MARKET_DATA_RUNTIME",
        "status": "FRESH",
        "symbol": symbol,
        "capturedAtMs": captured_at_ms,
        "universeCandles5m": universe,
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarized_result(batch: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    result = batch["results"][symbol]
    response = compatibility_response(batch, symbol, "offline-audit")
    return {
        "feature_vector_hash": result["feature_vector_hash"],
        "feature_values_hash": stable_hash(result["research_features"]),
        "feature_count": result["feature_count"],
        "long_prob": response["long_prob"],
        "short_prob": response["short_prob"],
        "neutral_prob": response["neutral_prob"],
        "candidate_raw_score": result["candidate"]["raw_score"],
        "candidate_calibrated_score": result["candidate"]["calibrated_score"],
        "candidate_side": result["candidate"]["side"],
        "selected": result["selected"],
        "decision": response["aegis"]["decision_brain"]["decision"],
        "verdict": response["meta_verdict"],
    }


def replay_engine(artifact_root: Path) -> CurrentBrainEngine:
    engine = CurrentBrainEngine(
        CurrentBrainPaths(
            artifact_root,
            artifact_root / "config",
            artifact_root / "config/bundles/aegis-prospective-shadow-candidate-v1.json",
        )
    )
    engine.initialize()
    engine.evaluate = engine.evaluate_replay  # type: ignore[method-assign]
    return engine


def context_path_parity(
    snapshot: MarketSnapshot, artifact_root: Path
) -> list[dict[str, Any]]:
    try:
        from aegis.context_inference import predict_from_snapshot
        from aegis.market_context import market_snapshot_from_context
    except ImportError:
        return [
            {
                "symbol": symbol,
                "supported": False,
                "provider_calls": None,
                "exact_match": None,
                "provider_hash": None,
                "direct_hash": None,
            }
            for symbol in CANONICAL_SYMBOLS
        ]

    parsed = market_snapshot_from_context(
        market_context(snapshot, CANONICAL_SYMBOLS[0]),
        expected_symbol=CANONICAL_SYMBOLS[0],
    )
    provider = StaticProvider(snapshot)
    provider_service = CurrentBrainDecisionService(
        replay_engine(artifact_root), provider, cache_seconds=60
    )
    direct_provider = StaticProvider(snapshot)
    direct_service = CurrentBrainDecisionService(
        replay_engine(artifact_root), direct_provider, cache_seconds=-1
    )
    rows = []
    for symbol in CANONICAL_SYMBOLS:
        provider_response = dict(provider_service.predict(symbol, "provider-audit"))
        direct_response = dict(
            predict_from_snapshot(
                direct_service, parsed, symbol, "direct-context-audit"
            )
        )
        normalized_provider = normalized_response(provider_response)
        normalized_direct = normalized_response(direct_response)
        rows.append(
            {
                "symbol": symbol,
                "supported": True,
                "provider_calls": provider.calls,
                "exact_match": normalized_provider == normalized_direct,
                "provider_hash": stable_hash(normalized_provider),
                "direct_hash": stable_hash(normalized_direct),
            }
        )
    if direct_provider.calls != 0:
        raise AssertionError("direct context path unexpectedly called REST provider")
    return rows


def normalized_response(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": response["symbol"],
        "long_prob": response["long_prob"],
        "short_prob": response["short_prob"],
        "neutral_prob": response["neutral_prob"],
        "meta_verdict": response["meta_verdict"],
        "features": response["features"],
        "decision_brain": response["aegis"]["decision_brain"],
        "turbo": response["aegis"]["turbo"],
        "canonical_raw_score": response["metadata"]["canonical_raw_score"],
        "canonical_calibrated_score": response["metadata"][
            "canonical_calibrated_score"
        ],
    }


def keyed_rows(payload: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("audit result has no rows")
    keyed = {tuple(row[name] for name in ROW_KEY): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("audit result contains duplicate row keys")
    return keyed


def diversity(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(keyed_rows(payload).values())
    calibrated = [float(row["candidate_calibrated_score"]) for row in rows]
    return {
        "row_count": len(rows),
        "unique_by_field": {
            field: len({json.dumps(row[field], sort_keys=True) for row in rows})
            for field in AUDITED_FIELDS
        },
        "calibrated_score_min": min(calibrated),
        "calibrated_score_max": max(calibrated),
        "calibrated_score_range": max(calibrated) - min(calibrated),
    }


def compare_results(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two audit results")
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    baseline_path, baseline = loaded[0]
    baseline_rows = keyed_rows(baseline)
    comparisons = []
    for candidate_path, candidate in loaded[1:]:
        candidate_rows = keyed_rows(candidate)
        if candidate_rows.keys() != baseline_rows.keys():
            raise ValueError(f"row keys differ: {baseline_path} vs {candidate_path}")
        mismatch_counts = {field: 0 for field in AUDITED_FIELDS}
        maximum_absolute_delta = {field: 0.0 for field in NUMERIC_FIELDS}
        for key, baseline_row in baseline_rows.items():
            candidate_row = candidate_rows[key]
            for field in AUDITED_FIELDS:
                if baseline_row[field] != candidate_row[field]:
                    mismatch_counts[field] += 1
            for field in NUMERIC_FIELDS:
                maximum_absolute_delta[field] = max(
                    maximum_absolute_delta[field],
                    abs(float(baseline_row[field]) - float(candidate_row[field])),
                )
        parity = candidate.get("context_path_parity", [])
        comparisons.append(
            {
                "candidate": str(candidate_path),
                "candidate_runtime": candidate.get("runtime", {}),
                "mismatch_counts": mismatch_counts,
                "total_mismatches": sum(mismatch_counts.values()),
                "maximum_absolute_delta": maximum_absolute_delta,
                "context_supported": sum(bool(row.get("supported")) for row in parity),
                "context_exact_matches": sum(
                    row.get("exact_match") is True for row in parity
                ),
            }
        )
    return {
        "schema": "aegis-probability-regression-comparison-v1",
        "baseline": str(baseline_path),
        "baseline_runtime": baseline.get("runtime", {}),
        "diversity": {str(path): diversity(payload) for path, payload in loaded},
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--compare", nargs="+", type=Path)
    args = parser.parse_args()

    if args.compare:
        output = compare_results(args.compare)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0
    if args.artifact_root is None:
        parser.error("--artifact-root is required unless --compare is used")

    root = Path(aegis.__file__).resolve().parents[2]
    artifact_root = args.artifact_root.resolve()
    engine = replay_engine(artifact_root)
    rows = []
    context_parity = []
    for variant in range(VARIANT_COUNT):
        snapshot = synthetic_snapshot(variant)
        batch = engine.evaluate_replay(snapshot)
        for symbol in CANONICAL_SYMBOLS:
            rows.append(
                {
                    "variant": variant,
                    "symbol": symbol,
                    **summarized_result(batch, symbol),
                }
            )

        if variant in (0, VARIANT_COUNT - 1):
            for parity in context_path_parity(snapshot, artifact_root):
                context_parity.append(
                    {
                        "variant": variant,
                        **parity,
                    }
                )

    output = {
        "schema": "aegis-probability-regression-audit-v1",
        "runtime": {
            "git_sha": git_sha(root),
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "joblib": package_version("joblib"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "fixture": {
            "variants": VARIANT_COUNT,
            "symbols": list(CANONICAL_SYMBOLS),
            "bars_per_symbol": BAR_COUNT,
            "closed_at": FIXED_CLOSE.isoformat(),
        },
        "rows": rows,
        "context_path_parity": context_parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
