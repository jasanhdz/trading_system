#!/usr/bin/env python3
"""Replay the sequential entry methodology over immutable closed-bar journals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from aegis.research.entry_methodology_v2_shadow import (
    EntryMethodologyV2Policy,
    assess_entry_methodology_v2_shadow,
    label_clean_entry_path,
)
from aegis.utils import canonical_json, sha256_file


def rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield value


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def label_metrics(values: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not values:
        return {"rows": 0}
    count = len(values)
    return {
        "rows": count,
        "clean_path_success_rate": sum(
            bool(row["clean_path_success"]) for row in values
        )
        / count,
        "fast_edge_success_rate": sum(bool(row["fast_edge_success"]) for row in values)
        / count,
        "positive_net_rate": sum(
            float(row["net_return_after_costs"]) > 0.0 for row in values
        )
        / count,
        "mean_net_return_after_costs": sum(
            float(row["net_return_after_costs"]) for row in values
        )
        / count,
        "mean_mae_fraction": sum(float(row["mae_fraction"]) for row in values) / count,
        "mean_mfe_fraction": sum(float(row["mfe_fraction"]) for row in values) / count,
        "mean_underwater_bars": sum(int(row["underwater_bars"]) for row in values)
        / count,
        "classifications": dict(
            sorted(Counter(str(row["classification"]) for row in values).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/hybrid_directional_live_experiment/decisions.jsonl"),
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("data/hybrid_directional_shadow/signals.jsonl"),
    )
    parser.add_argument(
        "--intelligence",
        type=Path,
        default=Path("data/entry_intelligence_shadow/signals.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/entry_methodology_v2_shadow/replay.json"),
    )
    args = parser.parse_args()
    policy = EntryMethodologyV2Policy()

    signal_by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows(args.signals):
        signal_by_symbol[str(row["symbol"])].append(row)
    signal_position: dict[tuple[str, str], int] = {}
    for symbol, values in signal_by_symbol.items():
        values.sort(key=lambda row: str(row["market_timestamp"]))
        for index, row in enumerate(values):
            signal_position[(str(row["market_timestamp"]), symbol)] = index

    intelligence = {
        (str(row["market_timestamp"]), str(row["symbol"])): row
        for row in rows(args.intelligence)
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows(args.decisions):
        grouped[str(row["market_timestamp"])].append(row)

    states: dict[tuple[str, str], Mapping[str, Any]] = {}
    tiers: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    labels_by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    labels_by_side_tier: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    selected_labels: list[Mapping[str, Any]] = []
    grade_a_rows: list[Mapping[str, Any]] = []
    grade_a_cycle_times: list[datetime] = []
    missing_intelligence = 0
    missing_horizon = 0
    evaluated = 0

    for timestamp, candidates in sorted(grouped.items()):
        confirmed = [
            row
            for row in candidates
            if row.get("confirmation", {}).get("state") == "CONFIRMED"
        ]
        side_counts = Counter(str(row["side"]) for row in confirmed)
        cycle_has_grade_a = False
        for candidate in sorted(
            candidates, key=lambda row: (str(row["symbol"]), str(row["side"]))
        ):
            symbol = str(candidate["symbol"])
            side = str(candidate["side"])
            context = intelligence.get((timestamp, symbol), {})
            if not context:
                missing_intelligence += 1
            assessment = assess_entry_methodology_v2_shadow(
                market_timestamp=timestamp,
                side=side,
                prediction=candidate,
                confirmation=candidate.get("confirmation", {}),
                confirmation_features=candidate.get("confirmation_features", {}),
                current_layer={},
                entry_intelligence=context,
                confirmed_same_side=side_counts[side],
                confirmed_total=len(confirmed),
                previous=states.get((symbol, side)),
                policy=policy,
            )
            states[(symbol, side)] = assessment
            tier = str(assessment["tier"])
            tiers[tier] += 1
            actions[str(assessment["counterfactual_action"])] += 1
            evaluated += 1
            cycle_has_grade_a |= tier == "A"

            position = signal_position.get((timestamp, symbol))
            symbol_signals = signal_by_symbol.get(symbol, [])
            if position is None or position + policy.horizon_bars >= len(
                symbol_signals
            ):
                missing_horizon += 1
                continue
            future = symbol_signals[position + 1 : position + 1 + policy.horizon_bars]
            entry_price = float(future[0]["market_bar"]["open"])
            label = dict(
                label_clean_entry_path(
                    side=side,
                    entry_price=entry_price,
                    future_bars=[row["market_bar"] for row in future],
                    policy=policy,
                )
            )
            label.update(
                {
                    "market_timestamp": timestamp,
                    "symbol": symbol,
                    "candidate_side": side,
                    "tier": tier,
                }
            )
            labels_by_tier[tier].append(label)
            labels_by_side_tier[(side, tier)].append(label)
            if bool(candidate.get("selected")):
                selected_labels.append(label)
            if tier == "A":
                grade_a_rows.append(label)
        if cycle_has_grade_a:
            grade_a_cycle_times.append(parse_time(timestamp))

    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(grade_a_cycle_times, grade_a_cycle_times[1:])
    ]
    non_overlapping: list[Mapping[str, Any]] = []
    last_kept: dict[tuple[str, str], datetime] = {}
    for label in sorted(grade_a_rows, key=lambda row: str(row["market_timestamp"])):
        timestamp = parse_time(str(label["market_timestamp"]))
        key = (str(label["symbol"]), str(label["candidate_side"]))
        if key in last_kept and timestamp - last_kept[key] < timedelta(minutes=60):
            continue
        non_overlapping.append(label)
        last_kept[key] = timestamp
    midpoint = len(non_overlapping) // 2

    report = {
        "schema_id": "aegis-entry-methodology-v2-shadow-replay-v1",
        "source": {
            "decisions": str(args.decisions),
            "decisions_sha256": sha256_file(args.decisions),
            "signals": str(args.signals),
            "signals_sha256": sha256_file(args.signals),
            "intelligence": str(args.intelligence),
            "intelligence_sha256": sha256_file(args.intelligence),
        },
        "policy": {
            "horizon_bars": policy.horizon_bars,
            "fast_edge_bars": policy.fast_edge_bars,
            "favorable_barrier_fraction": policy.favorable_barrier_fraction,
            "adverse_barrier_fraction": policy.adverse_barrier_fraction,
            "round_trip_cost_fraction": policy.round_trip_cost_fraction,
            "maximum_wait_bars": policy.maximum_wait_bars,
            "threshold_provenance": "EXISTING_H12_TAIL_AND_COST_CONTRACT",
        },
        "evaluated_candidate_rows": evaluated,
        "evaluated_cycles": len(grouped),
        "tier_counts": dict(sorted(tiers.items())),
        "action_counts": dict(sorted(actions.items())),
        "grade_a_cycles": len(grade_a_cycle_times),
        "grade_a_cycle_fraction": (
            len(grade_a_cycle_times) / len(grouped) if grouped else 0.0
        ),
        "maximum_grade_a_gap_hours": max(gaps) if gaps else None,
        "missing_intelligence_rows": missing_intelligence,
        "missing_mature_horizon_rows": missing_horizon,
        "outcomes_by_tier": {
            tier: label_metrics(labels_by_tier[tier]) for tier in ("A", "B", "C")
        },
        "outcomes_by_side_and_tier": {
            f"{side}|{tier}": label_metrics(labels_by_side_tier[(side, tier)])
            for side in ("LONG", "SHORT")
            for tier in ("A", "B", "C")
        },
        "current_selected_control": label_metrics(selected_labels),
        "non_overlapping_grade_a": {
            "all": label_metrics(non_overlapping),
            "first_half": label_metrics(non_overlapping[:midpoint]),
            "second_half": label_metrics(non_overlapping[midpoint:]),
            "embargo_minutes_per_symbol_side": 60,
        },
        "promotion_state": "SHADOW_EVIDENCE_REQUIRED",
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
