"""Replay the hybrid committee with price-dependent TypeScript protection."""

from __future__ import annotations

import argparse
import gc
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from aegis.data import CanonicalBar, CanonicalSeriesSource, DataPurpose
from aegis.research.hybrid_ts_protection_replay import (
    IntrabarPath,
    ProtectionReplayResult,
    TsProtectionConfig,
    replay_ts_price_protection,
)
from aegis.training.dataset import (
    build_e2_hourly_long_dataset,
    build_e2_hourly_short_dataset,
)
from aegis.training.hybrid_directional import (
    DirectionalSide,
    HybridDirectionalRow,
    HybridDirectionalSelection,
    fit_hybrid_directional_committee,
    paired_directional_rows,
)
from aegis.training.run_state import atomic_write_json
from aegis.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("replay timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _block(
    rows: Sequence[HybridDirectionalRow], start: str, end: str
) -> tuple[HybridDirectionalRow, ...]:
    start_ns = int(_utc(start).timestamp() * 1_000_000_000)
    end_ns = int(_utc(end).timestamp() * 1_000_000_000)
    return tuple(row for row in rows if start_ns <= row.timestamp_ns <= end_ns)


def _load_material(
    hybrid_config: Mapping[str, Any],
) -> tuple[
    tuple[HybridDirectionalRow, ...],
    Mapping[str, Any],
]:
    source_config = _mapping(hybrid_config["source"], "source")
    source = CanonicalSeriesSource(
        Path(str(source_config["path"])),
        DataPurpose.REPLAY,
        expected_manifest_sha256=str(source_config["manifest_sha256"]),
    )
    audit = source.audit(verify_content=True)
    if not audit.finality_verified:
        raise ValueError("hybrid replay source finality is not verified")
    sampling = _mapping(hybrid_config["sampling"], "sampling")
    first = _utc(str(sampling["expected_rows"]["first_anchor_utc"]))
    last = _utc(str(sampling["expected_rows"]["last_dev_anchor_utc"]))
    history_bars = int(sampling["history_bars"])
    horizon_bars = int(sampling["horizon_bars"])
    start = first - timedelta(minutes=history_bars * 5)
    end = last + timedelta(minutes=horizon_bars * 5)
    if end > _utc(str(hybrid_config["lockbox"]["start"])):
        raise ValueError("hybrid replay lockbox access prohibited")
    series = source.load(start=start, end=end)
    long_build = build_e2_hourly_long_dataset(
        series,
        sampling,
        dataset_id=f"{hybrid_config['experiment_id']}-long-replay",
        source_finality_verified=True,
    )
    short_build = build_e2_hourly_short_dataset(
        series,
        sampling,
        dataset_id=f"{hybrid_config['experiment_id']}-short-replay",
        source_finality_verified=True,
    )
    rows = paired_directional_rows(
        long_build.dataset,
        short_build.dataset,
        round_trip_cost_fraction=float(
            hybrid_config["labels"]["round_trip_cost_fraction"]
        ),
    )
    return rows, {
        "manifest_sha256": audit.manifest_sha256,
        "finality_verified": audit.finality_verified,
        "long_dataset_sha256": long_build.dataset.artifact_hash,
        "short_dataset_sha256": short_build.dataset.artifact_hash,
        "directional_rows": len(rows),
    }


def _load_replay_series(
    hybrid_config: Mapping[str, Any], protection: TsProtectionConfig
) -> Mapping[str, tuple[CanonicalBar, ...]]:
    source_config = _mapping(hybrid_config["source"], "source")
    source = CanonicalSeriesSource(
        Path(str(source_config["path"])),
        DataPurpose.REPLAY,
        expected_manifest_sha256=str(source_config["manifest_sha256"]),
    )
    folds = tuple(
        _mapping(item, "fold") for item in hybrid_config["fold_protocol"]["folds"]
    )
    start = min(_utc(str(item["scoring_start"])) for item in folds) - timedelta(
        minutes=(protection.atr_period + 1) * 5
    )
    end = max(_utc(str(item["scoring_end"])) for item in folds) + timedelta(
        minutes=(int(hybrid_config["sampling"]["horizon_bars"]) + 1) * 5
    )
    return source.load(start=start, end=end)


def _fit_fold(
    rows: Sequence[HybridDirectionalRow],
    fold: Mapping[str, Any],
    hybrid_config: Mapping[str, Any],
) -> tuple[HybridDirectionalSelection, ...]:
    training = _mapping(hybrid_config["training"], "training")
    trace: list[HybridDirectionalSelection] = []
    fit_hybrid_directional_committee(
        _block(rows, str(fold["train_start"]), str(fold["train_end"])),
        _block(
            rows,
            str(fold["calibration_start"]),
            str(fold["calibration_end"]),
        ),
        _block(rows, str(fold["scoring_start"]), str(fold["scoring_end"])),
        seed=int(training["seed"]) + int(fold["id"]),
        embargo_minutes=int(hybrid_config["fold_protocol"]["embargo_minutes"]),
        round_trip_cost_fraction=float(
            hybrid_config["labels"]["round_trip_cost_fraction"]
        ),
        classifier_parameters=_mapping(training["classifier"], "classifier"),
        regressor_parameters=_mapping(training["regressor"], "regressor"),
        selection_trace=trace,
    )
    return tuple(trace)


def _protection_config(raw: Mapping[str, Any]) -> TsProtectionConfig:
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


def _replay_selection(
    selection: HybridDirectionalSelection,
    series: Mapping[str, tuple[CanonicalBar, ...]],
    indexes: Mapping[str, Mapping[datetime, int]],
    paths: Sequence[IntrabarPath],
    config: TsProtectionConfig,
    horizon_bars: int,
) -> Mapping[str, ProtectionReplayResult]:
    anchor = datetime.fromtimestamp(
        selection.timestamp_ns / 1_000_000_000, timezone.utc
    )
    position = indexes[selection.symbol].get(anchor)
    if position is None or position < config.atr_period + 1:
        raise ValueError("selected candidate has no causal price path")
    symbol_series = series[selection.symbol]
    history = symbol_series[:position]
    future = symbol_series[position : position + horizon_bars]
    if len(future) != horizon_bars:
        raise ValueError("selected candidate future horizon is incomplete")
    expected_terminal = (
        (future[-1].close - future[0].open) / future[0].open
    ) * selection.side.sign - config.round_trip_cost_fraction
    if abs(expected_terminal - selection.observed_net_return_after_costs) > 1e-12:
        raise ValueError("selected candidate terminal return does not match label")
    return {
        path.value: replay_ts_price_protection(
            side=selection.side,
            history=history,
            future=future,
            path=path,
            config=config,
        )
        for path in paths
    }


def _metrics(
    selections: Sequence[HybridDirectionalSelection],
    results: Sequence[Mapping[str, ProtectionReplayResult]],
    paths: Sequence[IntrabarPath],
) -> Mapping[str, Any]:
    baseline = [item.observed_net_return_after_costs for item in selections]
    path_metrics: dict[str, Any] = {}
    for path in paths:
        values = [item[path.value].net_return_after_costs for item in results]
        reasons = Counter(item[path.value].exit_reason.value for item in results)
        path_metrics[path.value] = {
            "mean_net_expectancy": mean(values),
            "win_rate": sum(value > 0.0 for value in values) / len(values),
            "mean_delta_vs_terminal": mean(
                value - original for value, original in zip(values, baseline)
            ),
            "break_even_armed_rate": sum(
                item[path.value].break_even_armed for item in results
            )
            / len(results),
            "trailing_armed_rate": sum(
                item[path.value].trailing_armed for item in results
            )
            / len(results),
            "exit_reasons": dict(sorted(reasons.items())),
        }
    worst = [
        min(item[path.value].net_return_after_costs for path in paths)
        for item in results
    ]
    best = [
        max(item[path.value].net_return_after_costs for path in paths)
        for item in results
    ]
    return {
        "signals": len(selections),
        "model_only_terminal": {
            "mean_net_expectancy": mean(baseline),
            "win_rate": sum(value > 0.0 for value in baseline) / len(baseline),
        },
        "protection_by_intrabar_path": path_metrics,
        "intrabar_bound": {
            "worst_case_mean_net_expectancy": mean(worst),
            "best_case_mean_net_expectancy": mean(best),
            "worst_case_win_rate": sum(value > 0.0 for value in worst) / len(worst),
            "best_case_win_rate": sum(value > 0.0 for value in best) / len(best),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_hybrid_ts_protection_replay_v1.yaml"),
    )
    args = parser.parse_args()
    replay_path = (ROOT / args.config).resolve()
    replay_config = _mapping(
        yaml.safe_load(replay_path.read_text(encoding="utf-8")), "replay config"
    )
    if replay_config.get("schema_version") != "aegis-hybrid-ts-protection-replay-v1":
        raise SystemExit("AEGIS_HYBRID_TS_PROTECTION_REPLAY_CONFIG_INVALID")
    hybrid_path = ROOT / str(replay_config["hybrid_experiment"]["path"])
    if sha256_file(hybrid_path) != str(replay_config["hybrid_experiment"]["sha256"]):
        raise SystemExit("AEGIS_HYBRID_TS_PROTECTION_HYBRID_CONFIG_DRIFT")
    for authority in (
        "trading_service",
        "profit_guardian",
        "micro_live_gate",
        "technical_indicators",
        "runtime_configuration",
    ):
        item = replay_config["typescript_authority"][authority]
        if sha256_file(ROOT / str(item["path"])) != str(item["sha256"]):
            raise SystemExit("AEGIS_HYBRID_TS_PROTECTION_TYPESCRIPT_DRIFT")

    hybrid_config = _mapping(
        yaml.safe_load(hybrid_path.read_text(encoding="utf-8")), "hybrid config"
    )
    rows, dataset = _load_material(hybrid_config)
    protection = _protection_config(_mapping(replay_config["protection"], "protection"))
    paths = tuple(IntrabarPath(value) for value in replay_config["intrabar_paths"])
    fold_selections = []
    for fold in hybrid_config["fold_protocol"]["folds"]:
        fold_selections.append(
            (
                _mapping(fold, "fold"),
                _fit_fold(rows, _mapping(fold, "fold"), hybrid_config),
            )
        )
        gc.collect()
    del rows
    gc.collect()

    series = _load_replay_series(hybrid_config, protection)
    indexes = {
        symbol: {bar.timestamp: index for index, bar in enumerate(values)}
        for symbol, values in series.items()
    }
    horizon = int(hybrid_config["sampling"]["horizon_bars"])

    folds = []
    all_positive = True
    for fold, selections in fold_selections:
        fold_sides: dict[str, Any] = {}
        for side in DirectionalSide:
            side_selections = tuple(item for item in selections if item.side is side)
            side_results = tuple(
                _replay_selection(item, series, indexes, paths, protection, horizon)
                for item in side_selections
            )
            side_metrics = _metrics(side_selections, side_results, paths)
            robust_positive = side_metrics["intrabar_bound"][
                "worst_case_mean_net_expectancy"
            ] > 0.0 and all(
                side_metrics["protection_by_intrabar_path"][path.value][
                    "mean_net_expectancy"
                ]
                > 0.0
                for path in paths
            )
            fold_sides[side.value] = {
                **side_metrics,
                "robust_positive": robust_positive,
            }
            all_positive = all_positive and robust_positive
        folds.append({"fold_id": int(fold["id"]), "sides": fold_sides})

    conclusion = (
        "PRICE_PROTECTION_REVERSES_EXPECTANCY_ROBUSTLY"
        if all_positive
        else "PRICE_PROTECTION_DOES_NOT_ESTABLISH_POSITIVE_EXPECTANCY"
    )
    report = {
        "schema_id": "aegis-hybrid-ts-protection-replay-report-v1",
        "experiment_id": replay_config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "replay_config_path": str(replay_path.relative_to(ROOT)),
        "replay_config_sha256": sha256_file(replay_path),
        "hybrid_config_path": str(hybrid_path.relative_to(ROOT)),
        "hybrid_config_sha256": sha256_file(hybrid_path),
        "dataset": dataset,
        "typescript_authority": replay_config["typescript_authority"],
        "protection": asdict(protection),
        "scope": replay_config["scope"],
        "folds": folds,
        "robust_positive_every_fold_and_direction": all_positive,
        "conclusion": conclusion,
        "live_eligible": False,
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    output = ROOT / str(replay_config["output"]["report"])
    atomic_write_json(output, report)
    print(
        yaml.safe_dump(
            {
                "report": str(output.relative_to(ROOT)),
                "conclusion": conclusion,
                "robust_positive_every_fold_and_direction": all_positive,
                "exchange_mutations": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if all_positive else 2


if __name__ == "__main__":
    raise SystemExit(main())
