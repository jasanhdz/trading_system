#!/usr/bin/env python3
"""Replay the seven-point counterfactual over immutable runtime journals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from aegis.research.seven_point_entry_shadow import assess_seven_point_entry_shadow
from aegis.utils import canonical_json, sha256_file


def rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"non-object row in {path}")
                yield value


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def metrics(values: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    count = len(values)
    if count == 0:
        return {"rows": 0}
    return {
        "rows": count,
        "mean_net_return_after_costs": sum(float(item["net"]) for item in values)
        / count,
        "mean_mae_fraction": sum(float(item["mae"]) for item in values) / count,
        "mean_mfe_fraction": sum(float(item["mfe"]) for item in values) / count,
        "win_rate_after_costs": sum(float(item["net"]) > 0.0 for item in values)
        / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/hybrid_directional_live_experiment/decisions.jsonl"),
    )
    parser.add_argument(
        "--intelligence",
        type=Path,
        default=Path("data/entry_intelligence_shadow/signals.jsonl"),
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path("data/hybrid_directional_shadow/outcomes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/entry_quality_v4_shadow/replay.json"),
    )
    args = parser.parse_args()

    intelligence = {
        (str(row["market_timestamp"]), str(row["symbol"])): row
        for row in rows(args.intelligence)
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows(args.decisions):
        grouped[str(row["market_timestamp"])].append(row)
    outcomes = {
        (str(row["market_timestamp"]), str(row["symbol"])): row
        for row in rows(args.outcomes)
    }

    dispositions: Counter[str] = Counter()
    eligible_times: list[datetime] = []
    missing_intelligence = 0
    matured: dict[tuple[str, str], list[Mapping[str, float]]] = defaultdict(list)
    candidate_outcomes: list[Mapping[str, Any]] = []
    evaluated = 0
    for timestamp, candidates in sorted(grouped.items()):
        confirmed = [
            item
            for item in candidates
            if item.get("confirmation", {}).get("state") == "CONFIRMED"
        ]
        side_counts = Counter(str(item["side"]) for item in confirmed)
        cycle_has_candidate = False
        for candidate in candidates:
            context = intelligence.get((timestamp, str(candidate["symbol"])), {})
            if not context:
                missing_intelligence += 1
            assessment = assess_seven_point_entry_shadow(
                side=str(candidate["side"]),
                prediction=candidate,
                confirmation=candidate.get("confirmation", {}),
                confirmation_features=candidate.get("confirmation_features", {}),
                current_layer={},
                entry_intelligence=context,
                confirmed_same_side=side_counts[str(candidate["side"])],
                confirmed_total=len(confirmed),
            )
            disposition = str(assessment["disposition"])
            dispositions[disposition] += 1
            evaluated += 1
            outcome = outcomes.get((timestamp, str(candidate["symbol"])))
            if outcome is not None:
                directional = outcome.get("directional_outcomes", {})
                side_outcome = (
                    directional.get(str(candidate["side"]), {})
                    if isinstance(directional, Mapping)
                    else {}
                )
                if isinstance(side_outcome, Mapping):
                    matured[(disposition, str(candidate["side"]))].append(
                        {
                            "net": float(side_outcome["net_return_after_costs"]),
                            "mae": float(side_outcome["mae_fraction"]),
                            "mfe": float(side_outcome["mfe_fraction"]),
                        }
                    )
                    if disposition == "COUNTERFACTUAL_QUALITY_CANDIDATE":
                        candidate_outcomes.append(
                            {
                                "timestamp": timestamp,
                                "symbol": str(candidate["symbol"]),
                                "side": str(candidate["side"]),
                                "net": float(
                                    side_outcome["net_return_after_costs"]
                                ),
                                "mae": float(side_outcome["mae_fraction"]),
                                "mfe": float(side_outcome["mfe_fraction"]),
                            }
                        )
            cycle_has_candidate |= disposition == "COUNTERFACTUAL_QUALITY_CANDIDATE"
        if cycle_has_candidate:
            eligible_times.append(parse_time(timestamp))

    gaps = [
        (right - left).total_seconds() / 3600.0
        for left, right in zip(eligible_times, eligible_times[1:])
    ]
    outcome_metrics = {}
    for (disposition, side), values in sorted(matured.items()):
        outcome_metrics[f"{disposition}|{side}"] = metrics(values)

    non_overlapping = []
    last_kept: dict[tuple[str, str], datetime] = {}
    for value in sorted(candidate_outcomes, key=lambda item: str(item["timestamp"])):
        timestamp = parse_time(str(value["timestamp"]))
        key = (str(value["symbol"]), str(value["side"]))
        if key in last_kept and timestamp - last_kept[key] < timedelta(minutes=60):
            continue
        non_overlapping.append(value)
        last_kept[key] = timestamp
    midpoint = len(non_overlapping) // 2
    temporal = {
        "all": metrics(non_overlapping),
        "first_half": metrics(non_overlapping[:midpoint]),
        "second_half": metrics(non_overlapping[midpoint:]),
        "by_side": {
            side: metrics(
                [value for value in non_overlapping if value["side"] == side]
            )
            for side in ("LONG", "SHORT")
        },
        "embargo_minutes_per_symbol_side": 60,
    }
    report = {
        "schema_id": "aegis-seven-point-entry-shadow-replay-v1",
        "source": {
            "decisions": str(args.decisions),
            "decisions_sha256": sha256_file(args.decisions),
            "intelligence": str(args.intelligence),
            "intelligence_sha256": sha256_file(args.intelligence),
            "outcomes": str(args.outcomes),
            "outcomes_sha256": sha256_file(args.outcomes),
        },
        "evaluated_candidate_rows": evaluated,
        "evaluated_cycles": len(grouped),
        "dispositions": dict(sorted(dispositions.items())),
        "cycles_with_quality_candidate": len(eligible_times),
        "candidate_cycle_fraction": (
            len(eligible_times) / len(grouped) if grouped else 0.0
        ),
        "maximum_gap_hours": max(gaps) if gaps else None,
        "missing_intelligence_rows": missing_intelligence,
        "matured_outcome_metrics": outcome_metrics,
        "non_overlapping_quality_candidate_metrics": temporal,
        "historical_tail_risk_limitation": (
            "NOT_RECORDED_IN_HYBRID_DECISION_JOURNAL; dangerous confluence "
            "is evaluated prospectively after deployment"
        ),
        "selection_effect": "NONE",
        "exchange_mutations": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
