#!/usr/bin/env python3
"""Run the frozen rules-only retrospective replay without opening holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


SANDBOX = Path(__file__).resolve().parents[1]
REPOSITORY = SANDBOX.parents[1]
for path in (SANDBOX / "src", REPOSITORY / "src"):
    sys.path.insert(0, str(path))

from aegis_strategy_router.research.retrospective_falsification import run_symbol  # noqa: E402


DEFAULT_CONFIG = SANDBOX / "config" / "retrospective_falsification_v1.json"
DEFAULT_CANDLES = REPOSITORY / "data" / "aegis_strategy_router_retrospective_v1" / "candles_1m"
DEFAULT_OUTPUT = SANDBOX / "artifacts" / "retrospective_falsification_v1" / "raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_freeze(config: dict[str, object]) -> None:
    rules = config["rules"]
    checks = {
        SANDBOX / str(rules["document"]): str(rules["document_sha256"]),
        SANDBOX / "src/aegis_strategy_router/candidates/base.py": str(rules["base_sha256"]),
        SANDBOX / "src/aegis_strategy_router/candidates/frozen_rules.py": str(rules["frozen_rules_sha256"]),
        SANDBOX / "src/aegis_strategy_router/candidates/generators.py": str(rules["generators_sha256"]),
    }
    mismatches = [
        f"{path}: expected {expected}, got {_sha256(path)}"
        for path, expected in checks.items()
        if _sha256(path) != expected
    ]
    if mismatches:
        raise RuntimeError("FROZEN_RULE_HASH_MISMATCH\n" + "\n".join(mismatches))


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candle-root", type=Path, default=DEFAULT_CANDLES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _verify_freeze(config)
    symbols = tuple(args.symbols or config["symbols"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = _parse(config["candidate_start_inclusive"])
    last = _parse(config["candidate_last_inclusive"])
    pending = []
    for symbol in symbols:
        audit = args.output_root / symbol / "audit.json"
        if audit.exists() and not args.overwrite:
            print(f"SKIP_COMPLETE {symbol}", flush=True)
        else:
            pending.append(symbol)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = {
            pool.submit(
                run_symbol,
                symbol=symbol,
                candle_root=args.candle_root,
                start_at=start,
                last_at=last,
                output_root=args.output_root,
            ): symbol
            for symbol in pending
        }
        for future in as_completed(jobs):
            symbol = jobs[future]
            result = future.result()
            results.append(result)
            print(
                f"COMPLETE {symbol} snapshots={result.snapshots} "
                f"population={result.population_candidates} "
                f"independent={result.independent_episodes}",
                flush=True,
            )
    manifest = {
        "schema": "aegis-strategy-router-retrospective-replay-manifest-v1",
        "classification": "RETROSPECTIVE_DISCOVERY_ONLY",
        "config_sha256": _sha256(args.config),
        "frozen_rules_verified": True,
        "symbols_requested": list(symbols),
        "symbols_completed_this_run": sorted(result.symbol for result in results),
        "sealed_holdouts_loaded": False,
        "rules_changed_during_backtest": False,
    }
    (args.output_root / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
