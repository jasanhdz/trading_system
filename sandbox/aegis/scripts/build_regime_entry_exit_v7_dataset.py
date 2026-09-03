#!/usr/bin/env python3
"""Enrich the immutable V6 dataset with V7 attribution and protection replays."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.data import CanonicalBar
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.regime_entry_exit_v7 import (
    TrajectoryAuditContract,
    replay_protection_profiles,
    trajectory_attribution,
    v7_feature_vector,
)
from aegis.training.hybrid_directional import DirectionalSide
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _mapping, _source_series


def _audit_contract(config: Mapping[str, Any]) -> TrajectoryAuditContract:
    raw = _mapping(config["trajectory_audit"], "trajectory_audit")
    cost = float(
        _mapping(
            _mapping(config["protection_profiles"], "protection_profiles")["shared"],
            "protection_profiles.shared",
        )["round_trip_cost_fraction"]
    )
    return TrajectoryAuditContract(
        round_trip_cost_fraction=cost,
        maximum_clean_mae_fraction=float(raw["maximum_clean_mae_fraction"]),
        maximum_clean_positive_bar=int(raw["maximum_clean_positive_bar"]),
        late_entry_extension_atr=float(raw["late_entry_extension_atr"]),
        late_entry_positive_bar=int(raw["late_entry_positive_bar"]),
        minimum_available_net_fraction=float(raw["minimum_available_net_fraction"]),
    )


def _profiles(config: Mapping[str, Any]) -> Mapping[str, TsProtectionConfig]:
    raw = _mapping(config["protection_profiles"], "protection_profiles")
    shared = _mapping(raw["shared"], "protection_profiles.shared")
    result = {}
    for name, values in raw.items():
        if name in {"shared", "profile_choice_source", "production_effect"}:
            continue
        profile = _mapping(values, f"protection_profiles.{name}")
        result[str(name)] = TsProtectionConfig(
            leverage=float(shared["leverage"]),
            hard_stop_roe=float(shared["hard_stop_roe"]),
            take_profit_roe=float(shared["take_profit_roe"]),
            break_even_trigger_roe=float(profile["break_even_trigger_roe"]),
            break_even_offset_fraction=float(profile["break_even_offset_fraction"]),
            trailing_activation_roe=float(profile["trailing_activation_roe"]),
            trailing_callback_roe=float(profile["trailing_callback_roe"]),
            use_atr_trailing=bool(shared["use_atr_trailing"]),
            atr_period=int(shared["atr_period"]),
            atr_multiplier=float(shared["atr_multiplier"]),
            round_trip_cost_fraction=float(shared["round_trip_cost_fraction"]),
        )
    if set(result) != {
        "CURRENT_TS",
        "LOCK_AT_5_ROE",
        "LOCK_AT_10_ROE",
        "LOCK_AT_20_ROE",
    }:
        raise ValueError("V7 protection profile identities are invalid")
    return result


def _canonical(values: list[Any]) -> tuple[CanonicalBar, ...]:
    return tuple(
        CanonicalBar(
            value.open_time,
            value.open,
            value.high,
            value.low,
            value.close,
            value.volume,
        )
        for value in values
    )


def build_dataset(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    maximum_rows: int | None = None,
) -> Mapping[str, Any]:
    authority = _mapping(config["authority"], "authority")
    source_dataset = root / str(authority["source_dataset"])
    source_manifest_path = root / str(authority["source_manifest"])
    if sha256_file(source_dataset) != str(authority["source_dataset_sha256"]):
        raise ValueError("V7 source dataset hash mismatch")
    if sha256_file(source_manifest_path) != str(authority["source_manifest_sha256"]):
        raise ValueError("V7 source manifest hash mismatch")
    source_manifest = _mapping(
        json.loads(source_manifest_path.read_text()), "source_manifest"
    )
    if int(source_manifest["model_feature_count"]) != int(
        authority["source_feature_count"]
    ):
        raise ValueError("V7 source feature schema mismatch")
    history_bars = int(authority["history_bars"])
    horizon_bars = int(authority["horizon_bars"])
    candles, common, candle_inventory = _source_series(
        root / str(authority["candle_database"]),
        root / str(authority["public_candle_delta"]),
        lookback_days=int(authority["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=horizon_bars,
    )
    index_by_open = {timestamp: index for index, timestamp in enumerate(common)}
    profiles = _profiles(config)
    audit_contract = _audit_contract(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    current_replay_mismatches = 0
    rows = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    with gzip.open(source_dataset, "rt", encoding="utf-8") as source, gzip.open(
        temporary, "wt", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if maximum_rows is not None and rows >= maximum_rows:
                break
            if not line.strip():
                continue
            row = dict(_mapping(json.loads(line), f"source:{line_number}"))
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
            try:
                future_index = index_by_open[timestamp]
            except KeyError as exc:
                raise ValueError(
                    "V7 candle source does not cover a source row"
                ) from exc
            symbol = str(row["symbol"])
            history_values = candles[symbol][future_index - history_bars : future_index]
            future_values = candles[symbol][future_index : future_index + horizon_bars]
            if (
                len(history_values) != history_bars
                or len(future_values) != horizon_bars
                or abs(float(row["entry_price"]) - future_values[0].open) > 1e-10
            ):
                raise ValueError("V7 candle alignment mismatch")
            features, archetype, context = v7_feature_vector(row)
            attribution = trajectory_attribution(row, context, audit_contract)
            protection = replay_protection_profiles(
                side=DirectionalSide(str(row["side"])),
                history=_canonical(history_values),
                future=_canonical(future_values),
                profiles=profiles,
            )
            current_difference = abs(
                float(protection["CURRENT_TS"]["worst_net_return"])
                - float(row["price_protection_worst_net_return"])
            )
            if current_difference > 1e-10:
                current_replay_mismatches += 1
            best_hindsight = max(
                protection,
                key=lambda name: float(protection[name]["worst_net_return"]),
            )
            enriched = {
                **row,
                "v7_features": features,
                "v7_archetype": archetype.value,
                "v7_context": context,
                "trajectory_attribution": attribution,
                "protection_profiles": protection,
                "hindsight_best_protection_profile": best_hindsight,
                "hindsight_best_protected_net": float(
                    protection[best_hindsight]["worst_net_return"]
                ),
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            target.write(json.dumps(enriched, sort_keys=True, separators=(",", ":")))
            target.write("\n")
            rows += 1
            first_timestamp = first_timestamp or str(row["timestamp"])
            last_timestamp = str(row["timestamp"])
            counts[str(attribution["responsibility"])] += 1
            profile_counts[best_hindsight] += 1
            if rows % 10000 == 0:
                print(json.dumps({"rows": rows}, sort_keys=True), flush=True)
    if rows == 0:
        raise ValueError("V7 produced no rows")
    if current_replay_mismatches:
        raise ValueError(
            f"V7 current protection replay mismatches: {current_replay_mismatches}"
        )
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-regime-entry-exit-v7-dataset-manifest-v1",
        "config_sha256": Sha256HashProvider().digest_value(config),
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_sha256": sha256_file(source_dataset),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "dataset": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "rows": rows,
        "evidence_start": first_timestamp,
        "evidence_end": last_timestamp,
        "symbols": source_manifest["symbols"],
        "sides": source_manifest["sides"],
        "source_feature_count": source_manifest["model_feature_count"],
        "v7_feature_count": len(features),
        "trajectory_responsibility_counts": dict(sorted(counts.items())),
        "hindsight_best_profile_counts": dict(sorted(profile_counts.items())),
        "current_protection_replay_mismatches": current_replay_mismatches,
        "candle_source": candle_inventory,
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    os.chmod(manifest_path, 0o600)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_regime_entry_exit_v7_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regime_entry_exit_v7/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/regime_entry_exit_v7/dataset_manifest.json"),
    )
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    result = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(config_path.read_text()), "config"),
        output=output,
        manifest_path=manifest,
        maximum_rows=args.maximum_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
