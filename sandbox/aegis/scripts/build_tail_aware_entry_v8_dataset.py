#!/usr/bin/env python3
"""Build V8 soft-routing, multihorizon and tail-risk research evidence."""

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
from aegis.research.regime_entry_exit_v7 import replay_protection_profiles
from aegis.research.tail_aware_entry_v8 import (
    TailLabelContract,
    classify_forward_regime,
    tail_labels,
    v8_feature_vector,
)
from aegis.training.hybrid_directional import DirectionalSide
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_v21_shadow import _mapping, _source_series


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


def _profile_grid(config: Mapping[str, Any]) -> Mapping[str, TsProtectionConfig]:
    raw = _mapping(config["protection_grid"], "protection_grid")
    base_cost = float(
        _mapping(config["cost_sensitivity"], "cost_sensitivity")[
            "replay_base_round_trip_fraction"
        ]
    )
    common = {
        "leverage": float(raw["leverage"]),
        "take_profit_roe": float(raw["take_profit_roe"]),
        "trailing_callback_roe": float(raw["trailing_callback_roe"]),
        "use_atr_trailing": bool(raw["use_atr_trailing"]),
        "atr_period": int(raw["atr_period"]),
        "atr_multiplier": float(raw["atr_multiplier"]),
        "round_trip_cost_fraction": base_cost,
    }
    current = _mapping(raw["current_ts_control"], "current_ts_control")
    profiles = {
        "CURRENT_TS": TsProtectionConfig(
            hard_stop_roe=float(current["hard_stop_roe"]),
            break_even_trigger_roe=float(current["break_even_trigger_roe"]),
            break_even_offset_fraction=float(current["break_even_offset_fraction"]),
            trailing_activation_roe=float(current["trailing_activation_roe"]),
            trailing_callback_roe=float(current["trailing_callback_roe"]),
            **{
                key: value
                for key, value in common.items()
                if key != "trailing_callback_roe"
            },
        )
    }
    retained = float(raw["retained_trigger_fraction"])
    leverage = float(raw["leverage"])
    for stop in raw["hard_stop_roe"]:
        for trigger in raw["lock_trigger_roe"]:
            stop_value = float(stop)
            trigger_value = float(trigger)
            name = f"STOP_{int(abs(stop_value) * 100):02d}_LOCK_{int(trigger_value * 100):02d}"
            profiles[name] = TsProtectionConfig(
                hard_stop_roe=stop_value,
                break_even_trigger_roe=trigger_value,
                break_even_offset_fraction=trigger_value * retained / leverage,
                trailing_activation_roe=trigger_value,
                **common,
            )
    if len(profiles) != 10:
        raise ValueError("V8 protection grid must contain ten profiles")
    return profiles


def _tail_contract(config: Mapping[str, Any]) -> TailLabelContract:
    raw = _mapping(config["trajectory"], "trajectory")
    return TailLabelContract(
        clean_mae_fraction=float(raw["clean_mae_fraction"]),
        clean_positive_bar=int(raw["clean_positive_bar"]),
        late_positive_bar=int(raw["late_positive_bar"]),
        catastrophic_net_fraction=float(raw["catastrophic_net_fraction"]),
    )


def _cost_adjusted(
    base_net: float, *, base_cost: float, costs: Mapping[str, float]
) -> Mapping[str, float]:
    gross = float(base_net) + base_cost
    result = {name: gross - cost for name, cost in costs.items()}
    if not all(value == value and abs(value) < 10.0 for value in result.values()):
        raise ValueError("V8 cost adjustment is invalid")
    return result


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
        raise ValueError("V8 source dataset hash mismatch")
    if sha256_file(source_manifest_path) != str(authority["source_manifest_sha256"]):
        raise ValueError("V8 source manifest hash mismatch")
    source_manifest = _mapping(
        json.loads(source_manifest_path.read_text()), "source_manifest"
    )
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
    profiles = _profile_grid(config)
    tail_contract = _tail_contract(config)
    regime_config = _mapping(config["forward_regime"], "forward_regime")
    horizons = tuple(int(value) for value in regime_config["horizons_bars"])
    cost_config = _mapping(config["cost_sensitivity"], "cost_sensitivity")
    base_cost = float(cost_config["replay_base_round_trip_fraction"])
    costs = {
        "expected": float(cost_config["expected_round_trip_fraction"]),
        "stress": float(cost_config["stress_round_trip_fraction"]),
        "severe": float(cost_config["severe_round_trip_fraction"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = 0
    mismatch_count = 0
    label_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    best_profile_counts: Counter[str] = Counter()
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    regime_cache: dict[datetime, Mapping[str, Any]] = {}
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
                raise ValueError("V8 candle source does not cover source row") from exc
            symbol = str(row["symbol"])
            history_values = candles[symbol][future_index - history_bars : future_index]
            future_values = candles[symbol][future_index : future_index + horizon_bars]
            if (
                len(history_values) != history_bars
                or len(future_values) != horizon_bars
                or abs(float(row["entry_price"]) - future_values[0].open) > 1e-10
            ):
                raise ValueError("V8 candle alignment mismatch")
            forward_regime = regime_cache.get(timestamp)
            if forward_regime is None:
                returns = {
                    horizon: {
                        candidate: (
                            candles[candidate][future_index + horizon - 1].close
                            / candles[candidate][future_index].open
                            - 1.0
                        )
                        for candidate in source_manifest["symbols"]
                    }
                    for horizon in horizons
                }
                forward_regime = classify_forward_regime(
                    returns,
                    btc_threshold_at_24_fraction=float(
                        regime_config["btc_threshold_at_24_fraction"]
                    ),
                    breadth_threshold=float(regime_config["breadth_threshold"]),
                    range_breadth_band=tuple(
                        float(value) for value in regime_config["range_breadth_band"]
                    ),
                    consensus_horizons=int(regime_config["consensus_horizons"]),
                )
                regime_cache[timestamp] = forward_regime
            features, memberships = v8_feature_vector(row)
            replay = replay_protection_profiles(
                side=DirectionalSide(str(row["side"])),
                history=_canonical(history_values),
                future=_canonical(future_values),
                profiles=profiles,
            )
            if (
                abs(
                    float(replay["CURRENT_TS"]["worst_net_return"])
                    - float(
                        row["protection_profiles"]["CURRENT_TS"]["worst_net_return"]
                    )
                )
                > 1e-10
            ):
                mismatch_count += 1
            profile_costs = {
                name: _cost_adjusted(
                    float(value["worst_net_return"]),
                    base_cost=base_cost,
                    costs=costs,
                )
                for name, value in replay.items()
            }
            stress = {name: value["stress"] for name, value in profile_costs.items()}
            labels = tail_labels(row, stress, tail_contract)
            enriched = {
                **row,
                "v8_features": features,
                "soft_archetype_memberships": memberships,
                "forward_regime_multihorizon": forward_regime,
                "v8_protection_profiles": replay,
                "v8_profile_cost_returns": profile_costs,
                "v8_tail_labels": labels,
                "selection_effect": "NONE",
                "exchange_authority": False,
                "exchange_mutations": 0,
            }
            target.write(json.dumps(enriched, sort_keys=True, separators=(",", ":")))
            target.write("\n")
            rows += 1
            first_timestamp = first_timestamp or str(row["timestamp"])
            last_timestamp = str(row["timestamp"])
            regime_counts[str(forward_regime["label"])] += 1
            best_profile_counts[str(labels["hindsight_best_profile"])] += 1
            for name in (
                "clean_entry",
                "late_entry",
                "positive_stress_net",
                "catastrophic_stress_loss",
            ):
                label_counts[f"{row['side']}::{name}"] += int(bool(labels[name]))
            if rows % 10000 == 0:
                print(json.dumps({"rows": rows}, sort_keys=True), flush=True)
    if rows == 0:
        raise ValueError("V8 produced no rows")
    if mismatch_count:
        raise ValueError(f"V8 current profile replay mismatches: {mismatch_count}")
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-tail-aware-entry-v8-dataset-manifest-v1",
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
        "v8_feature_count": len(features),
        "profile_count": len(profiles),
        "label_counts": dict(sorted(label_counts.items())),
        "forward_regime_counts": dict(sorted(regime_counts.items())),
        "hindsight_best_profile_counts": dict(sorted(best_profile_counts.items())),
        "current_profile_replay_mismatches": mismatch_count,
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
        default=Path("config/experiments/aegis_tail_aware_entry_v8_research.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tail_aware_entry_v8/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/tail_aware_entry_v8/dataset_manifest.json"),
    )
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    result = build_dataset(
        root=root,
        config=_mapping(yaml.safe_load(resolve(args.config).read_text()), "config"),
        output=resolve(args.output),
        manifest_path=resolve(args.manifest),
        maximum_rows=args.maximum_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
