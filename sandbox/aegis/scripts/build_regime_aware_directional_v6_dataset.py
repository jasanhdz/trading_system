#!/usr/bin/env python3
"""Build the causal LONG/SHORT v6 dataset with current-brain lifecycle replay."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import DeterministicFeaturePipeline
from aegis.live_decision import (
    CONFIGURATION_SHA256,
    MODEL_ARTIFACT_SHA256,
    MODEL_BUNDLE_SHA256,
    CurrentBrainEngine,
)
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.long_entry_v21_shadow import multitimeframe_long_features
from aegis.research.regime_aware_directional_v6 import (
    CommitteeObservation,
    DirectionalPathContract,
    ExitEyeReplayConfig,
    REGIME_ROUTER_FEATURE_NAMES,
    classify_regime_axes,
    directional_path_outcome,
    directional_role,
    realized_global_regime,
    regime_aware_feature_vector,
    regime_router_feature_vector,
    replay_full_lifecycle,
)
from aegis.training.hybrid_directional import DirectionalSide
from aegis.utils import Sha256HashProvider, sha256_file
from train_long_entry_archetypes_v2_shadow import _snapshot
from train_long_entry_v21_shadow import _mapping, _source_series


def _path_contract(config: Mapping[str, Any]) -> DirectionalPathContract:
    raw = _mapping(config["labels"], "labels")
    return DirectionalPathContract(
        leverage=float(raw["leverage_for_roe_translation"]),
        roe_checkpoints=tuple(
            float(value) for value in raw["protectable_roe_checkpoints"]
        ),
        primary_protectable_roe=float(raw["primary_protectable_roe"]),
        favorable_atr_multiple=float(raw["favorable_atr_multiple"]),
        adverse_atr_multiple=float(raw["adverse_atr_multiple"]),
        favorable_floor_fraction=float(raw["favorable_floor_fraction"]),
        adverse_floor_fraction=float(raw["adverse_floor_fraction"]),
        favorable_ceiling_fraction=float(raw["favorable_ceiling_fraction"]),
        adverse_ceiling_fraction=float(raw["adverse_ceiling_fraction"]),
        fast_success_bars=int(raw["fast_success_bars"]),
        early_reversal_bars=int(raw["early_reversal_bars"]),
        round_trip_cost_fraction=float(raw["round_trip_cost_fraction"]),
    )


def _protection(config: Mapping[str, Any]) -> TsProtectionConfig:
    raw = _mapping(config["typescript_price_protection"], "typescript_price_protection")
    return TsProtectionConfig(
        leverage=float(raw["leverage"]),
        hard_stop_roe=float(raw["hard_stop_roe"]),
        take_profit_roe=float(raw["take_profit_roe"]),
        break_even_trigger_roe=float(raw["break_even_trigger_roe"]),
        break_even_offset_fraction=float(raw["break_even_offset_fraction"]),
        trailing_activation_roe=float(raw["trailing_activation_roe"]),
        trailing_callback_roe=float(raw["trailing_callback_roe"]),
        use_atr_trailing=bool(raw["use_atr_trailing"]),
        atr_period=int(raw["atr_period"]),
        atr_multiplier=float(raw["atr_multiplier"]),
        round_trip_cost_fraction=float(raw["round_trip_cost_fraction"]),
    )


def _exit_eye(config: Mapping[str, Any]) -> ExitEyeReplayConfig:
    raw = _mapping(config["exit_eye_replay"], "exit_eye_replay")
    return ExitEyeReplayConfig(
        enabled=bool(raw["enabled"]),
        min_roe_to_protect=float(raw["min_roe_to_protect"]),
        min_peak_roe_to_protect=float(raw["min_peak_roe_to_protect"]),
        min_giveback_from_peak_roe=float(raw["min_giveback_from_peak_roe"]),
        neutral_votes_to_protect=int(raw["neutral_votes_to_protect"]),
        opposite_votes_to_close=int(raw["opposite_votes_to_close"]),
        min_roe_to_close_on_opposite=float(raw["min_roe_to_close_on_opposite"]),
        min_peak_roe_to_close_on_opposite=float(
            raw["min_peak_roe_to_close_on_opposite"]
        ),
        close_on_neutral_decay=bool(raw["close_on_neutral_decay"]),
        neutral_close_votes=int(raw["neutral_close_votes"]),
        min_roe_to_close_on_neutral=float(raw["min_roe_to_close_on_neutral"]),
        min_peak_roe_to_close_on_neutral=float(raw["min_peak_roe_to_close_on_neutral"]),
        min_giveback_to_close_on_neutral=float(raw["min_giveback_to_close_on_neutral"]),
        require_consecutive_neutral_close=int(raw["require_consecutive_neutral_close"]),
        require_consecutive_neutral=int(raw["require_consecutive_neutral"]),
        require_consecutive_opposite=int(raw["require_consecutive_opposite"]),
        min_minutes_in_trade=float(raw["min_minutes_in_trade"]),
    )


def committee_observation(
    batch: Mapping[str, Any], symbol: str
) -> CommitteeObservation:
    result = _mapping(_mapping(batch["results"], "results")[symbol], f"result:{symbol}")
    predictions = result.get("predictions")
    if not isinstance(predictions, Sequence) or not predictions:
        return CommitteeObservation("HOLD", 0, 0, 0, available=False)
    sides = []
    for item in predictions:
        side_value = _mapping(item, "prediction").get("side")
        sides.append(str(getattr(side_value, "value", side_value)))
    candidate = _mapping(result["candidate"], "candidate")
    candidate_value = candidate.get("side")
    candidate_side = str(getattr(candidate_value, "value", candidate_value))
    action = (
        candidate_side
        if bool(result.get("selected")) and candidate_side in {"LONG", "SHORT"}
        else "HOLD"
    )
    return CommitteeObservation(
        action=action,
        long_votes=sum(value == "LONG" for value in sides),
        short_votes=sum(value == "SHORT" for value in sides),
        neutral_votes=sum(value == "NO_TRADE" for value in sides),
        available=True,
    )


class BrainBatchCache:
    def __init__(
        self, engine: CurrentBrainEngine, candles: Mapping[str, Sequence[Any]]
    ) -> None:
        self.engine = engine
        self.candles = candles
        self.values: OrderedDict[int, Mapping[str, Any]] = OrderedDict()
        self.evaluations = 0

    def get(self, index: int) -> Mapping[str, Any]:
        cached = self.values.get(index)
        if cached is not None:
            self.values.move_to_end(index)
            return cached
        value = self.engine.evaluate_replay(_snapshot(self.candles, index, 96))
        self.values[index] = value
        self.evaluations += 1
        return value

    def discard_before(self, index: int) -> None:
        for key in tuple(self.values):
            if key < index:
                del self.values[key]


def build_dataset(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    manifest_path: Path,
    lookback_days: int | None = None,
    maximum_snapshots: int | None = None,
) -> Mapping[str, Any]:
    source = _mapping(config["source"], "source")
    sampling = _mapping(config["sampling"], "sampling")
    router_contract = _mapping(
        _mapping(config["regime"], "regime")["probabilistic_router"],
        "probabilistic_router",
    )
    history_bars = int(sampling["history_bars"])
    horizon_bars = int(sampling["horizon_bars"])
    stride = int(sampling["entry_stride_bars"])
    independent_stride = int(sampling["independent_test_stride_bars"])
    if (
        stride <= 0
        or independent_stride < stride
        or int(router_contract["label_horizon_bars"]) != horizon_bars
    ):
        raise ValueError("v6 sampling contract is invalid")
    candles, common, source_inventory = _source_series(
        root / str(source["base_database"]),
        root / str(source["public_delta"]),
        lookback_days=lookback_days or int(source["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=horizon_bars,
    )
    path_contract = _path_contract(config)
    protection = _protection(config)
    exit_eye = _exit_eye(config)
    pipeline = DeterministicFeaturePipeline()
    engine = CurrentBrainEngine()
    engine.initialize()
    cache = BrainBatchCache(engine, candles)
    independent_every = max(1, math.ceil(independent_stride / stride))
    first = history_bars - 1
    last = len(common) - horizon_bars
    indices = list(range(first, last, stride))
    if maximum_snapshots is not None:
        indices = indices[:maximum_snapshots]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    realized_regime_counts: Counter[str] = Counter()
    hasher = Sha256HashProvider()
    row_count = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for evaluation_number, index in enumerate(indices):
            current = cache.get(index)
            future_batches = [
                cache.get(position)
                for position in range(index + 1, index + 1 + horizon_bars)
            ]
            timestamp = candles[CANONICAL_SYMBOLS[0]][index].close_time
            current_results = _mapping(current["results"], "current_results")
            features_by_symbol = {
                symbol: _mapping(
                    _mapping(current_results[symbol], f"current:{symbol}")[
                        "research_features"
                    ],
                    f"research_features:{symbol}",
                )
                for symbol in CANONICAL_SYMBOLS
            }
            router_features = regime_router_feature_vector(features_by_symbol)
            future_returns = {
                symbol: (
                    candles[symbol][index + horizon_bars].close
                    / candles[symbol][index + 1].open
                    - 1.0
                )
                for symbol in CANONICAL_SYMBOLS
            }
            realized_regime = realized_global_regime(
                future_returns,
                btc_direction_threshold_fraction=float(
                    router_contract["btc_direction_threshold_fraction"]
                ),
                cross_section_breadth_threshold=float(
                    router_contract["cross_section_breadth_threshold"]
                ),
            )
            realized_regime_counts[realized_regime] += 1
            independent = evaluation_number % independent_every == 0
            for symbol in CANONICAL_SYMBOLS:
                result = _mapping(current_results[symbol], f"current:{symbol}")
                base = features_by_symbol[symbol]
                history = candles[symbol][index - history_bars + 1 : index + 1]
                future = candles[symbol][index + 1 : index + 1 + horizon_bars]
                multitimeframe, context = multitimeframe_long_features(
                    base, history, pipeline=pipeline
                )
                regime = classify_regime_axes(base, context)
                observations = [
                    committee_observation(batch, symbol) for batch in future_batches
                ]
                entry_observation = committee_observation(current, symbol)
                for side in DirectionalSide:
                    role = directional_role(side, regime["direction"])
                    outcome = directional_path_outcome(
                        signal=candles[symbol][index],
                        future=future,
                        atr_fraction=float(base["atr_12"]),
                        side=side,
                        contract=path_contract,
                    )
                    lifecycle = replay_full_lifecycle(
                        side=side,
                        history=history,
                        future=future,
                        observations=observations,
                        protection=protection,
                        exit_eye=exit_eye,
                    )
                    features = regime_aware_feature_vector(
                        multitimeframe_features=multitimeframe,
                        base_features=base,
                        side=side,
                        regime=regime,
                    )
                    row = {
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "side": side.value,
                        "independent": independent,
                        "regime": regime,
                        "directional_role": role.value,
                        "features": features,
                        "regime_router_features": router_features,
                        "realized_global_regime": realized_regime,
                        "entry_brain_action": entry_observation.action,
                        "entry_brain_votes": {
                            "long": entry_observation.long_votes,
                            "short": entry_observation.short_votes,
                            "neutral": entry_observation.neutral_votes,
                        },
                        **outcome,
                        **lifecycle,
                    }
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")
                    row_count += 1
                    counts[f"{side.value}_target"] += int(
                        bool(outcome["target_before_stop"])
                    )
                    counts[f"{side.value}_protectable"] += int(
                        bool(outcome["protectable_advantage"])
                    )
                    counts[f"{side.value}_early_reversal"] += int(
                        bool(outcome["early_reversal"])
                    )
                    regime_counts[f"{side.value}::{regime['identity']}"] += 1
                    role_counts[f"{side.value}::{role.value}"] += 1
            cache.discard_before(index + 1)
            if (evaluation_number + 1) % 250 == 0:
                print(
                    json.dumps(
                        {
                            "snapshots": evaluation_number + 1,
                            "total_snapshots": len(indices),
                            "rows": row_count,
                            "brain_evaluations": cache.evaluations,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    os.replace(temporary, output)
    os.chmod(output, 0o600)
    manifest = {
        "schema_id": "aegis-regime-aware-directional-v6-dataset-manifest-v1",
        "config_sha256": hasher.digest_value(config),
        "dataset_path": str(output.resolve()),
        "dataset_sha256": sha256_file(output),
        "source": source_inventory,
        "evidence_start": candles[CANONICAL_SYMBOLS[0]][
            indices[0]
        ].close_time.isoformat(),
        "evidence_end": candles[CANONICAL_SYMBOLS[0]][
            indices[-1]
        ].close_time.isoformat(),
        "entry_snapshots": len(indices),
        "brain_evaluations": cache.evaluations,
        "rows": row_count,
        "symbols": list(CANONICAL_SYMBOLS),
        "sides": [side.value for side in DirectionalSide],
        "feature_schema": pipeline.schema_version,
        "model_feature_count": len(features),
        "label_counts": dict(sorted(counts.items())),
        "regime_counts": dict(sorted(regime_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "realized_regime_counts": dict(sorted(realized_regime_counts.items())),
        "regime_router_feature_count": len(REGIME_ROUTER_FEATURE_NAMES),
        "regime_router_feature_names": list(REGIME_ROUTER_FEATURE_NAMES),
        "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
        "model_bundle_sha256": MODEL_BUNDLE_SHA256,
        "configuration_sha256": CONFIGURATION_SHA256,
        "entry_rule": "NEXT_BAR_OPEN",
        "committee_source": "CURRENT_BRAIN_CANONICAL_REPLAY",
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temp_manifest, manifest_path)
    os.chmod(manifest_path, 0o600)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/experiments/aegis_regime_aware_directional_v6_shadow.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regime_aware_directional_v6/canonical_dataset.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/regime_aware_directional_v6/dataset_manifest.json"),
    )
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--maximum-snapshots", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    config = _mapping(yaml.safe_load(config_path.read_text()), "v6_config")
    result = build_dataset(
        root=root,
        config=config,
        output=output,
        manifest_path=manifest,
        lookback_days=args.lookback_days,
        maximum_snapshots=args.maximum_snapshots,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
