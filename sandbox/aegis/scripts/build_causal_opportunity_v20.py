#!/usr/bin/env python3
"""Build and audit the preregistered V20 causal opportunity population."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.causal_opportunity_v20 import (
    classify_opportunities,
    economic_summary,
    group_rows,
    matched_random_control,
    monthly_bootstrap_mean_interval,
    opportunity_record,
    viability,
)
from aegis.utils import Sha256HashProvider, sha256_file


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V20 {name} must be a mapping")
    return value


def _outcome(source: Mapping[str, Any]) -> dict[str, Any]:
    protection = source["protection_profiles"]["CURRENT_TS"]
    contract = source["v10_contract_outcomes"]["ROE_10_H12"]
    return {
        "timestamp": str(source["timestamp"]),
        "symbol": str(source["symbol"]),
        "side": str(source["side"]),
        "protected_net_return": float(protection["worst_net_return"]),
        "contract_utility": float(contract["realized_utility"]),
        "mae_fraction": float(source["mae_fraction"]),
        "mfe_fraction": float(source["mfe_fraction"]),
        "time_underwater_bars": float(source["time_underwater_bars"]),
        "break_even_armed": bool(protection["break_even_armed"]),
        "trailing_armed": bool(protection["trailing_armed"]),
    }


@contextmanager
def _deterministic_gzip_text(path: Path):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def build(root: Path, config: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    bindings = {
        root / str(authority["source_dataset"]): str(authority["source_dataset_sha256"]),
        root / str(authority["source_manifest"]): str(authority["source_manifest_sha256"]),
        root / str(authority["source_validation"]): str(authority["source_validation_sha256"]),
        root / str(authority["source_config"]): str(authority["source_config_sha256"]),
    }
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise ValueError(f"V20 authority mismatch: {path}")

    source_path = root / str(authority["source_dataset"])
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "opportunities.jsonl.gz"
    temporary = dataset_path.with_suffix(".jsonl.gz.tmp")
    candidates: list[Mapping[str, Any]] = []
    population: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_rows = 0
    overlaps: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    with gzip.open(source_path, "rt", encoding="utf-8") as source, _deterministic_gzip_text(
        temporary
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = _mapping(json.loads(line), f"source:{line_number}")
            side = str(row["side"])
            population[side].append(_outcome(row))
            families = classify_opportunities(row, config)
            overlaps[len(families)] += 1
            for family in families:
                candidate = opportunity_record(row, family)
                target.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")))
                target.write("\n")
                candidates.append(candidate)
                family_counts[f"{side}::{family.value}"] += 1
            source_rows += 1
            first_timestamp = first_timestamp or str(row["timestamp"])
            last_timestamp = str(row["timestamp"])
    os.replace(temporary, dataset_path)
    os.chmod(dataset_path, 0o600)

    groups = group_rows(candidates, "side", "family")
    gate = _mapping(config["viability_gate"], "viability_gate")
    family_reports: dict[str, Any] = {}
    eligible: list[str] = []
    family_order = {
        name: index
        for index, name in enumerate(str(item) for item in config["opportunity_families"])
    }
    for (side, family), rows in sorted(groups.items()):
        assessment = viability(rows, gate)
        identity = f"{side}::{family}"
        if assessment["passed"]:
            eligible.append(identity)
        random_rows = matched_random_control(
            population[side],
            len(rows),
            seed=200020 + (0 if side == "LONG" else 100) + family_order[family],
        )
        regimes = group_rows(rows, "regime")
        family_reports[identity] = {
            **assessment,
            "monthly_block_bootstrap_mean": monthly_bootstrap_mean_interval(
                rows,
                seed=200020 + (0 if side == "LONG" else 100) + family_order[family],
            ),
            "controls": {
                "no_trade": {"mean_protected_net": 0.0},
                "unconditional_side": economic_summary(population[side]),
                "random_matched_count": economic_summary(random_rows),
            },
            "regime_diagnostics_selection_effect_none": {
                regime[0]: economic_summary(regime_rows)
                for regime, regime_rows in sorted(regimes.items())
            },
            "exit_reason_counts": dict(
                sorted(Counter(str(row["protected_exit_reason"]) for row in rows).items())
            ),
        }

    report = {
        "schema_id": "aegis-causal-opportunity-v20-feasibility-v1",
        "experiment_id": str(config["experiment_id"]),
        "config_content_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset_sha256": sha256_file(source_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "source_rows": source_rows,
        "opportunity_rows": len(candidates),
        "evidence_start": first_timestamp,
        "evidence_end": last_timestamp,
        "family_counts": dict(sorted(family_counts.items())),
        "opportunity_family_overlap_counts": {
            str(count): rows for count, rows in sorted(overlaps.items())
        },
        "family_reports": family_reports,
        "eligible_side_families": eligible,
        "V20_READY_FOR_MODELING": bool(eligible),
        "V20_READY_FOR_SHADOW": False,
        "V20_READY_FOR_LIVE": False,
        "model_training_executed": False,
        "model_exported": False,
        "model_stage_disposition": (
            "ELIGIBLE_FAMILIES_REQUIRE_SEPARATE_FUTURE_VALIDATION"
            if eligible
            else "NO_SIMPLE_FAMILY_EDGE_MODEL_TRAINING_PROHIBITED"
        ),
        "known_data_gaps": dict(config["known_data_gaps"]),
        "selection_effect": "NONE",
        "live_changed": False,
        "shadow_changed": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    report_path = output_root / "feasibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)
    manifest = {
        "schema_id": "aegis-causal-opportunity-v20-manifest-v1",
        "dataset": str(dataset_path),
        "dataset_sha256": report["dataset_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "source_rows": source_rows,
        "opportunity_rows": len(candidates),
        "feature_count": 186,
        "selection_effect": "NONE",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_opportunity_dataset_v20.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/opportunity_dataset_v20")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    report = build(root, config, output_root)
    print(
        json.dumps(
            {
                "opportunity_rows": report["opportunity_rows"],
                "eligible_side_families": report["eligible_side_families"],
                "V20_READY_FOR_MODELING": report["V20_READY_FOR_MODELING"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
