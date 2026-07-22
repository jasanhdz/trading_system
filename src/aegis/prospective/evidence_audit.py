"""Read-only economic audit for prospective signal and outcome journals.

This module is deliberately isolated from the decision, Shadow, and Live
runtime packages. It observes immutable evidence; it cannot alter routing or
exchange state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


AUDIT_SCHEMA = "aegis-current-brain-prospective-evidence-audit-v1"
DEFAULT_SEED = 20260722


class EvidenceAuditError(RuntimeError):
    """Raised when append-only evidence cannot be interpreted safely."""


def _stable_bytes(path: Path, attempts: int = 3) -> bytes:
    for attempt in range(attempts):
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
            if payload and not payload.endswith(b"\n"):
                raise EvidenceAuditError("EVIDENCE_JOURNAL_PARTIAL_RECORD")
            return payload
        if attempt + 1 < attempts:
            time.sleep(0.05)
    raise EvidenceAuditError("EVIDENCE_JOURNAL_CHANGED_DURING_READ")


def _parse_jsonl(payload: bytes, identity: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvidenceAuditError(f"EVIDENCE_JSON_INVALID:{line_number}") from exc
        if not isinstance(row, dict) or identity not in row:
            raise EvidenceAuditError(f"EVIDENCE_SCHEMA_INVALID:{line_number}")
        row_identity = str(row[identity])
        if row_identity in identities:
            raise EvidenceAuditError(f"EVIDENCE_DUPLICATE_IDENTITY:{row_identity}")
        identities.add(row_identity)
        rows.append(row)
    return tuple(rows)


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceAuditError(f"EVIDENCE_NUMBER_INVALID:{field}") from exc
    if not math.isfinite(number):
        raise EvidenceAuditError(f"EVIDENCE_NUMBER_NONFINITE:{field}")
    return number


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _economic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    returns = [_finite(row["net_return_fraction"], "net_return_fraction") for row in rows]
    gains = math.fsum(value for value in returns if value > 0.0)
    losses = -math.fsum(value for value in returns if value < 0.0)
    return {
        "count": len(returns),
        "mean_net_return": statistics.fmean(returns) if returns else None,
        "median_net_return": statistics.median(returns) if returns else None,
        "profit_factor": gains / losses if losses > 0.0 else (None if gains == 0.0 else "INFINITE"),
        "win_rate": sum(value > 0.0 for value in returns) / len(returns) if returns else None,
        "tail_event_rate": (
            statistics.fmean(_finite(row["tail_event"], "tail_event") for row in rows)
            if rows
            else None
        ),
        "mean_mfe": (
            statistics.fmean(_finite(row["mfe_fraction"], "mfe_fraction") for row in rows)
            if rows
            else None
        ),
        "mean_mae": (
            statistics.fmean(_finite(row["mae_fraction"], "mae_fraction") for row in rows)
            if rows
            else None
        ),
    }


def _hour_block_bootstrap(
    rows: Sequence[Mapping[str, Any]], repetitions: int, seed: int
) -> dict[str, Any]:
    blocks: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        timestamp = datetime.fromisoformat(str(row["signal_timestamp_utc"]).replace("Z", "+00:00"))
        block = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        blocks[block].append(_finite(row["net_return_fraction"], "net_return_fraction"))
    ordered = [blocks[key] for key in sorted(blocks)]
    if not ordered or repetitions <= 0:
        return {"block_count": len(ordered), "repetitions": repetitions, "ci95": None}
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(repetitions):
        sample = [ordered[generator.randrange(len(ordered))] for _ in ordered]
        flattened = [value for block in sample for value in block]
        means.append(statistics.fmean(flattened))
    return {
        "block_unit": "UTC_HOUR",
        "block_count": len(ordered),
        "repetitions": repetitions,
        "seed": seed,
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
    }


def _stage_funnel(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = {
        "d3_evidence_present": 0,
        "rv2_evidence_present": 0,
        "trrm_passed": 0,
        "qmae_valid": 0,
        "eqm_eligible": 0,
        "econ1_eligible": 0,
        "selected": 0,
    }
    d3_final_matches = 0
    for signal in signals:
        components = signal.get("component_evidence", {})
        d3 = components.get("d3", {})
        if d3.get("status") == "PASS":
            stages["d3_evidence_present"] += 1
        if components.get("rv2", {}).get("status") == "PASS":
            stages["rv2_evidence_present"] += 1
        if components.get("trrm", {}).get("output", {}).get("passed") is True:
            stages["trrm_passed"] += 1
        if components.get("qmae", {}).get("output", {}).get("valid") is True:
            stages["qmae_valid"] += 1
        if components.get("eqm", {}).get("output", {}).get("eligible") is True:
            stages["eqm_eligible"] += 1
        if components.get("econ1", {}).get("output", {}).get("eligible") is True:
            stages["econ1_eligible"] += 1
        final_action = signal.get("final_decision", {}).get("action")
        if final_action == "ENTER_NOW":
            stages["selected"] += 1
        if d3.get("output", {}).get("decision") == final_action:
            d3_final_matches += 1
    return {
        "evaluations": len(signals),
        "counts": stages,
        "d3_decision_equals_final_decision_count": d3_final_matches,
        "d3_decision_field_independent_gate_evidence": "NOT_ESTABLISHED",
    }


def _stage_economics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gates = (
        "trrm_passed",
        "qmae_valid",
        "eqm_eligible",
        "econ1_eligible",
        "selected",
    )
    result: dict[str, Any] = {}
    for gate in gates:
        passed = [row for row in rows if row[gate] is True]
        failed = [row for row in rows if row[gate] is not True]
        passed_metrics = _economic_metrics(passed)
        failed_metrics = _economic_metrics(failed)
        result[gate] = {
            "passed": passed_metrics,
            "failed": failed_metrics,
            "pass_minus_fail_mean_net_return": (
                passed_metrics["mean_net_return"] - failed_metrics["mean_net_return"]
                if passed and failed
                else None
            ),
        }
    regimes: dict[str, Any] = {}
    for regime in sorted({str(row["d3_regime"]) for row in rows}):
        regime_rows = [row for row in rows if row["d3_regime"] == regime]
        regimes[regime] = _economic_metrics(regime_rows)
    result["d3_regime"] = regimes
    return result


def _score_deciles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"deciles": [], "top_minus_bottom_mean_net_return": None}
    ordered = sorted(
        rows,
        key=lambda row: (
            _finite(row["calibrated_score"], "calibrated_score"),
            str(row["symbol"]),
            str(row["evaluation_id"]),
        ),
    )
    buckets: list[list[Mapping[str, Any]]] = [[] for _ in range(min(10, len(ordered)))]
    for index, row in enumerate(ordered):
        bucket = min(len(buckets) - 1, index * len(buckets) // len(ordered))
        buckets[bucket].append(row)
    deciles = []
    for index, bucket in enumerate(buckets, start=1):
        scores = [_finite(row["calibrated_score"], "calibrated_score") for row in bucket]
        deciles.append({
            "rank": index,
            "score_min": min(scores),
            "score_max": max(scores),
            "economics": _economic_metrics(bucket),
        })
    bottom = deciles[0]["economics"]["mean_net_return"]
    top = deciles[-1]["economics"]["mean_net_return"]
    return {
        "deciles": deciles,
        "top_minus_bottom_mean_net_return": top - bottom,
        "higher_rank_is_better": top > bottom,
    }


def _unique_count(
    rows: Iterable[Mapping[str, Any]], extractor: Callable[[Mapping[str, Any]], Any]
) -> int:
    return len({format(_finite(extractor(row), "variability"), ".17g") for row in rows})


def audit_evidence(
    signal_path: Path,
    outcome_path: Path,
    *,
    bootstrap_repetitions: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    signal_payload = _stable_bytes(signal_path)
    outcome_payload = _stable_bytes(outcome_path)
    signals = _parse_jsonl(signal_payload, "prospective_signal_id")
    outcomes = _parse_jsonl(outcome_payload, "prospective_signal_id")
    signal_ids = {str(row["prospective_signal_id"]) for row in signals}
    outcome_by_id = {str(row["prospective_signal_id"]): row for row in outcomes}

    joined: list[dict[str, Any]] = []
    for signal in signals:
        outcome = outcome_by_id.get(str(signal["prospective_signal_id"]))
        if outcome is None:
            continue
        components = signal.get("component_evidence", {})
        upstream = signal.get("upstream_model", {})
        joined.append({
            **outcome,
            "signal_timestamp_utc": signal["signal_timestamp_utc"],
            "evaluation_id": signal["evaluation_id"],
            "symbol": signal["symbol"],
            "selected": signal.get("final_decision", {}).get("action") == "ENTER_NOW",
            "short_probability": upstream.get("short_probability"),
            "expected_return": components.get("econ1", {}).get("output", {}).get("expected_return"),
            "calibrated_score": components.get("econ1", {}).get("output", {}).get("calibrated_score"),
            "quality_probability": upstream.get("quality_probability"),
            "tail_risk_probability": upstream.get("tail_risk_probability"),
            "d3_regime": components.get("d3", {}).get("output", {}).get("regime", "NOT_PRESENT"),
            "trrm_passed": components.get("trrm", {}).get("output", {}).get("passed") is True,
            "qmae_valid": components.get("qmae", {}).get("output", {}).get("valid") is True,
            "eqm_eligible": components.get("eqm", {}).get("output", {}).get("eligible") is True,
            "econ1_eligible": components.get("econ1", {}).get("output", {}).get("eligible") is True,
        })

    selected = [row for row in joined if row["selected"]]
    rejected = [row for row in joined if not row["selected"]]
    by_symbol: dict[str, Any] = {}
    for symbol in sorted({str(row["symbol"]) for row in joined}):
        symbol_rows = [row for row in joined if row["symbol"] == symbol]
        by_symbol[symbol] = {
            "all": _economic_metrics(symbol_rows),
            "selected": _economic_metrics([row for row in symbol_rows if row["selected"]]),
            "rejected": _economic_metrics([row for row in symbol_rows if not row["selected"]]),
        }

    cycles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        cycles[str(row["evaluation_id"]).rsplit(":", 1)[0]].append(row)
    selected_cycle_deltas: list[float] = []
    for cycle_rows in cycles.values():
        cycle_selected = [row for row in cycle_rows if row["selected"]]
        if not cycle_selected:
            continue
        cycle_mean = statistics.fmean(_finite(row["net_return_fraction"], "net_return_fraction") for row in cycle_rows)
        selected_cycle_deltas.extend(
            _finite(row["net_return_fraction"], "net_return_fraction") - cycle_mean
            for row in cycle_selected
        )

    scores = [_finite(row["calibrated_score"], "calibrated_score") for row in selected]
    expected = [_finite(row["expected_return"], "expected_return") for row in selected]
    selected_returns = [_finite(row["net_return_fraction"], "net_return_fraction") for row in selected]
    bootstrap = _hour_block_bootstrap(selected, bootstrap_repetitions, seed)
    ci = bootstrap["ci95"]
    warnings: list[str] = []
    if joined and _unique_count(joined, lambda row: row["short_probability"]) == 1:
        warnings.append("BASE_DIRECTIONAL_SHORT_PROBABILITY_CONSTANT")
    score_correlation = _correlation(scores, selected_returns)
    if score_correlation is not None and score_correlation < 0.0:
        warnings.append("SELECTED_SCORE_OUTCOME_CORRELATION_NEGATIVE")
    if ci is not None and ci[0] <= 0.0 <= ci[1]:
        warnings.append("SELECTED_EXPECTANCY_CI_INCLUDES_ZERO")
    symbol_means = [
        metrics["selected"]["mean_net_return"]
        for metrics in by_symbol.values()
        if metrics["selected"]["mean_net_return"] is not None
    ]
    if any(value > 0.0 for value in symbol_means) and any(value < 0.0 for value in symbol_means):
        warnings.append("SELECTED_SYMBOL_PERFORMANCE_HETEROGENEOUS")
    stage_funnel = _stage_funnel(signals)
    score_deciles = _score_deciles(joined)
    if stage_funnel["d3_decision_equals_final_decision_count"] == len(signals) and signals:
        warnings.append("D3_DECISION_FIELD_NOT_INDEPENDENT_OF_FINAL_SELECTION")
    if score_deciles["top_minus_bottom_mean_net_return"] is not None and not score_deciles["higher_rank_is_better"]:
        warnings.append("CALIBRATED_SCORE_TOP_DECILE_NOT_BETTER_THAN_BOTTOM_DECILE")

    return {
        "schema_id": AUDIT_SCHEMA,
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_effect": "NONE_OBSERVATIONAL_ONLY",
        "source": {
            "signal_path": str(signal_path.resolve()),
            "signal_sha256": hashlib.sha256(signal_payload).hexdigest(),
            "signal_count": len(signals),
            "outcome_path": str(outcome_path.resolve()),
            "outcome_sha256": hashlib.sha256(outcome_payload).hexdigest(),
            "outcome_count": len(outcomes),
            "matured_join_count": len(joined),
            "unmatured_signal_count": len(signals) - len(joined),
            "orphan_outcome_count": sum(
                str(row["prospective_signal_id"]) not in signal_ids for row in outcomes
            ),
        },
        "decision_counts": {"selected": len(selected), "rejected": len(rejected)},
        "economics": {
            "selected": _economic_metrics(selected),
            "rejected": _economic_metrics(rejected),
            "selected_hour_block_bootstrap": bootstrap,
            "selected_score_outcome_correlation": score_correlation,
            "selected_expected_return_outcome_correlation": _correlation(expected, selected_returns),
            "selected_same_cycle_delta_mean": (
                statistics.fmean(selected_cycle_deltas) if selected_cycle_deltas else None
            ),
            "selected_same_cycle_delta_positive_rate": (
                sum(value > 0.0 for value in selected_cycle_deltas) / len(selected_cycle_deltas)
                if selected_cycle_deltas
                else None
            ),
        },
        "variability": {
            "unique_short_probabilities": _unique_count(joined, lambda row: row["short_probability"]) if joined else 0,
            "unique_expected_returns": _unique_count(joined, lambda row: row["expected_return"]) if joined else 0,
            "unique_calibrated_scores": (
                _unique_count(joined, lambda row: row["calibrated_score"]) if joined else 0
            ),
            "unique_quality_probabilities": (
                _unique_count(joined, lambda row: row["quality_probability"]) if joined else 0
            ),
            "unique_tail_risk_probabilities": (
                _unique_count(joined, lambda row: row["tail_risk_probability"])
                if joined
                else 0
            ),
        },
        "stage_funnel": stage_funnel,
        "stage_economics": _stage_economics(joined),
        "calibrated_score_ranking": score_deciles,
        "by_symbol": by_symbol,
        "warnings": warnings,
        "evidence_verdict": (
            "INCONCLUSIVE_CONTINUE_PROSPECTIVE_OBSERVATION"
            if warnings
            else "NO_WARNING_FROM_CURRENT_SAMPLE"
        ),
        "execution_recommendation": "NO_AUTOMATIC_RUNTIME_CHANGE",
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        os.chmod(output_path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit prospective evidence without touching runtime state")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    if arguments.bootstrap_repetitions <= 0:
        raise EvidenceAuditError("BOOTSTRAP_REPETITIONS_INVALID")
    report = audit_evidence(
        arguments.signals,
        arguments.outcomes,
        bootstrap_repetitions=arguments.bootstrap_repetitions,
        seed=arguments.seed,
    )
    write_report(report, arguments.output)
    print(
        json.dumps(
            {
                "schema_id": report["schema_id"],
                "evidence_verdict": report["evidence_verdict"],
                "output": str(arguments.output),
                "runtime_effect": report["runtime_effect"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
