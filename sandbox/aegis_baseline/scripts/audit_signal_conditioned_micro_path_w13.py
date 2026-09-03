#!/usr/bin/env python3
"""Audit W13 signal/L2 overlap and stop before modeling when it is insufficient."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from aegis.research.signal_conditioned_micro_path_w13 import (
    W13SampleRequirements,
    assess_sample_gate,
    stable_signal_episode_id,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_and_verify(root: Path, config: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = resolve(root, config["sources"][path_key])
    if sha256(path) != config["sources"][hash_key]:
        raise ValueError(f"AEGIS_W13_SOURCE_HASH_MISMATCH:{path_key}")
    return path


def read_signals(path: Path) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not (
                row.get("independent") is True
                and row.get("side") == "SHORT"
                and row.get("entry_brain_action") == "SHORT"
            ):
                continue
            timestamp = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            timestamp_us = int(timestamp.timestamp() * 1_000_000)
            signals.append({
                "aegis_signal_episode_id": stable_signal_episode_id(row["symbol"], row["side"], timestamp_us),
                "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
                "date": timestamp.date().isoformat(),
                "symbol": str(row["symbol"]),
                "side": str(row["side"]),
            })
    return signals


def report_text(verdict: dict[str, Any]) -> str:
    audit = verdict["data_audit"]
    gate = audit["sample_gate"]
    return f"""# Aegis W13 Signal-Conditioned Micro-Path Confirmation - Result

## Verdict

`{verdict['status']}`

W13 stopped at the preregistered historical sample gate. No feature discovery,
target inspection, model fit, threshold selection, validation, bootstrap or
economic test was run.

## Historical Intersection

- Frozen Aegis action rows in the source dataset: {audit['source_frozen_signals']}.
- Frozen actions on a date with validated Tardis L2: {audit['date_overlap_signals']}.
- Counterfactual actions with both date and symbol L2 coverage: **{audit['counterfactual_signal_l2_episodes']}**.
- Strict no-lookahead signal episodes: **{audit['strict_no_lookahead_signal_l2_episodes']}**.
- Counterfactual TRAIN/VALIDATION: {audit['counterfactual_partition_counts']['W13_TRAIN']} / {audit['counterfactual_partition_counts']['W13_VALIDATION']}.
- Strict W13_TRAIN: {gate['observed']['W13_TRAIN']['episodes']} vs minimum {audit['requirements']['minimum_train_episodes']}.
- Strict W13_VALIDATION: {gate['observed']['W13_VALIDATION']['episodes']} vs minimum {audit['requirements']['minimum_validation_episodes']}.
- Covered directions: SHORT only. Transfer to LONG is untested.

The qualified brain artifact was created on 2026-07-21, after every available
Tardis day. The stored features are pre-entry, but applying the later artifact
is a counterfactual replay, not a strict no-lookahead reconstruction and not
evidence that this exact model was running in production on those dates.

## Why Modeling Stopped

The five counterfactual episodes cannot support TRAIN/VALIDATION, ablations,
latency stress, symbol stability or 10,000-iteration episode bootstrap. Resampling five
episodes would not create independent evidence. The minimum was not lowered.

This is **not** evidence that signal-conditioned micro-path confirmation lacks
edge. It means the requested historical test is not measurable with the local
intersection.

## Passive Collector Design

An inert collector primitive was added for a future, separately authorized
prospective observation phase. It accepts externally supplied Aegis signals and
BOOK/QUOTE/TRADE events, retains -30s/+180s around each immutable signal, and
has no socket, credential, decision, order or execution interface. It is
disabled and was not deployed.

## Safety

- FINAL_HOLDOUT_W13: `SEALED_NOT_OPENED`.
- Prior holdouts opened: 0.
- Production/TypeScript/Brain/guards/leverage/PM2 changes: 0.
- Authenticated requests, production WebSockets, Shadow and orders: 0.
- `W13_READY_FOR_LIVE = FALSE`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/experiments/aegis_signal_conditioned_micro_path_w13.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/governance/aegis_prospective_validation/live/signal_conditioned_micro_path_w13"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(root, str(args.config))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "aegis-signal-conditioned-micro-path-w13-v1":
        raise ValueError("AEGIS_W13_CONFIG_INVALID")

    signal_path = load_and_verify(root, config, "signal_dataset", "signal_dataset_sha256")
    load_and_verify(root, config, "signal_manifest", "signal_manifest_sha256")
    load_and_verify(root, config, "frozen_brain_bundle", "frozen_brain_bundle_sha256")
    load_and_verify(root, config, "frozen_brain_training_config", "frozen_brain_training_config_sha256")
    load_and_verify(root, config, "frozen_brain_training_state", "frozen_brain_training_state_sha256")
    l2_manifest_path = load_and_verify(root, config, "l2_manifest", "l2_manifest_sha256")
    l2_manifest = json.loads(l2_manifest_path.read_text(encoding="utf-8"))
    if not l2_manifest.get("all_partitions_pass"):
        raise ValueError("AEGIS_W13_L2_MANIFEST_INVALID")

    available = {(part["date"], part["symbol"]) for part in l2_manifest["parts"] if part.get("passes")}
    available_dates = {date for date, _ in available}
    signals = read_signals(signal_path)
    date_overlap = [row for row in signals if row["date"] in available_dates]
    eligible = [row for row in date_overlap if (row["date"], row["symbol"]) in available]
    train_dates = set(config["partitions"]["train_dates"])
    validation_dates = set(config["partitions"]["validation_dates"])
    counterfactual_partitions = {
        "W13_TRAIN": [row for row in eligible if row["date"] in train_dates],
        "W13_VALIDATION": [row for row in eligible if row["date"] in validation_dates],
    }
    if config["sources"].get("strict_no_lookahead_replay_available") is not False:
        raise ValueError("AEGIS_W13_LOOKAHEAD_AUTHORITY_AMBIGUOUS")
    partitions = {"W13_TRAIN": [], "W13_VALIDATION": []}
    gate_config = config["sample_gate"]
    requirements = W13SampleRequirements(
        minimum_train_episodes=int(gate_config["minimum_train_episodes"]),
        minimum_validation_episodes=int(gate_config["minimum_validation_episodes"]),
        minimum_symbols_per_partition=int(gate_config["minimum_symbols_per_partition"]),
        minimum_temporal_days_per_partition=int(gate_config["minimum_temporal_days_per_partition"]),
        minimum_directions_per_partition=int(gate_config["minimum_directions_per_partition"]),
    )
    gate = assess_sample_gate(partitions, requirements)
    if gate["passes"]:
        raise RuntimeError("AEGIS_W13_SAMPLE_GATE_UNEXPECTEDLY_PASSED_REVIEW_REQUIRED")

    by_date_symbol = Counter((row["date"], row["symbol"]) for row in eligible)
    missing_symbol_coverage = Counter(row["symbol"] for row in date_overlap if (row["date"], row["symbol"]) not in available)
    audit = {
        "source_frozen_signals": len(signals),
        "date_overlap_signals": len(date_overlap),
        "counterfactual_signal_l2_episodes": len(eligible),
        "strict_no_lookahead_signal_l2_episodes": 0,
        "counterfactual_partition_counts": {
            partition: len(rows) for partition, rows in counterfactual_partitions.items()
        },
        "eligible_by_date_symbol": {f"{date}|{symbol}": count for (date, symbol), count in sorted(by_date_symbol.items())},
        "date_overlap_missing_symbol_l2": dict(sorted(missing_symbol_coverage.items())),
        "available_l2_dates": sorted(available_dates),
        "available_l2_symbols": sorted({symbol for _, symbol in available}),
        "requirements": requirements.__dict__,
        "sample_gate": gate,
        "replay_semantics": config["sources"]["replay_semantics"],
        "contemporary_production_signal_claim": False,
        "strict_no_lookahead_replay_available": False,
        "qualified_brain_at_utc": config["sources"]["frozen_brain_qualified_at_utc"],
        "l2_reconstruction_valid": True,
    }
    false_flags = {
        "W13_MICRO_PATH_INFORMATION_FOUND": False,
        "W13_TAKER_CONFIRMATION_VALUE_FOUND": False,
        "W13_FLOW_RESPONSE_VALUE_FOUND": False,
        "W13_ORDERBOOK_INCREMENTAL_VALUE_FOUND": False,
        "W13_ABSORPTION_VALUE_FOUND": False,
        "W13_REMAINING_EDGE_FOUND": False,
        "W13_EXECUTED_TRADE_EDGE_FOUND": False,
        "W13_PER_SIGNAL_EDGE_FOUND": False,
        "W13_LATENCY_GATE_PASSED": False,
        "W13_COST_GATE_PASSED": False,
        "W13_MODELING_JUSTIFIED": False,
        "W13_READY_FOR_PROSPECTIVE_OBSERVATION": False,
        "W13_READY_FOR_SHADOW": False,
        "W13_READY_FOR_LIVE": False,
    }
    verdict = {
        "schema_version": "aegis-signal-conditioned-micro-path-w13-verdict-v1",
        "experiment_id": config["experiment_id"],
        "status": "AEGIS_W13_BLOCKED_INSUFFICIENT_HISTORICAL_SAMPLE",
        "W13_HISTORICAL_SAMPLE_SUFFICIENT": False,
        "W13_HISTORICAL_SAMPLE_INSUFFICIENT": True,
        "W13_BLOCKED_INSUFFICIENT_HISTORICAL_SAMPLE": True,
        "W13_ORDERBOOK_RECONSTRUCTION_VALID": True,
        "W13_SHORT_ONLY_EVIDENCE": True,
        "W13_PASSIVE_COLLECTOR_DESIGN_READY": True,
        **false_flags,
        "final_holdout": "SEALED_NOT_OPENED",
        "modeling_executed": False,
        "economic_hypothesis_tested": False,
        "data_audit": audit,
        "safety": {
            "prior_holdouts_opened": 0,
            "production_changes": 0,
            "typescript_changes": 0,
            "authenticated_requests": 0,
            "exchange_mutations": 0,
            "orders": 0,
            "shadow": False,
            "collector_deployed": False,
        },
    }
    output = resolve(root, str(args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    (output / "aegis_signal_conditioned_micro_path_w13_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = report_text(verdict)
    (output / "aegis_signal_conditioned_micro_path_w13_result.md").write_text(result, encoding="utf-8")
    (output / "aegis_signal_conditioned_micro_path_w13_data_audit.md").write_text(result.replace(" - Result", " - Data Audit", 1), encoding="utf-8")
    print(json.dumps({
        "status": verdict["status"],
        "counterfactual_overlap": len(eligible),
        "strict_no_lookahead_overlap": 0,
        "blockers": gate["blockers"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
