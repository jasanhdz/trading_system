"""Read-only, causally aligned benchmark against frozen Gen2 forward evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..freeze import BundleLifecycleState
from ..utils import Sha256HashProvider, canonical_json, sha256_file


class BenchmarkIntegrityError(RuntimeError):
    pass


GEN2_ROOT = Path("/home/jasan/Develop/aegis_gen2").resolve()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _assert_outside_gen2(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == GEN2_ROOT or GEN2_ROOT in resolved.parents:
        raise BenchmarkIntegrityError("benchmark output cannot be written inside aegis_gen2")
    return resolved


def _read_jsonl_read_only(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file() or GEN2_ROOT not in path.resolve().parents:
        raise BenchmarkIntegrityError("historical input must be a file below the read-only Gen2 root")
    before = sha256_file(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise BenchmarkIntegrityError(f"invalid JSONL object at line {line_number}")
            rows.append(payload)
    if sha256_file(path) != before:
        raise BenchmarkIntegrityError("historical Gen2 input mutated during read")
    return tuple(rows)


@dataclass(frozen=True)
class PairedDecision:
    timestamp: datetime
    symbol: str
    new_score: float
    gen2_score: float
    new_tail_score: float
    gen2_tail_score: float
    new_qmae_q90: float
    gen2_qmae_q90: float
    new_action: str
    gen2_action: str


@dataclass(frozen=True)
class PairedBenchmarkReport:
    schema_version: str
    bundle_id: str
    bundle_state: BundleLifecycleState
    matched_rows: int
    unmatched_new_rows: int
    unmatched_gen2_rows: int
    mean_new_score: float
    mean_gen2_score: float
    new_no_trade_rate: float
    gen2_no_trade_rate: float
    pairs: tuple[PairedDecision, ...]
    source_hash_before: str
    source_hash_after: str
    report_hash: str


def build_paired_benchmark(
    *, current_rows: Sequence[Mapping[str, Any]], gen2_decisions_path: Path,
    bundle_id: str, bundle_state: BundleLifecycleState,
) -> PairedBenchmarkReport:
    if bundle_state is not BundleLifecycleState.CANDIDATE:
        raise BenchmarkIntegrityError("paired benchmark requires a real CANDIDATE bundle")
    source_hash = sha256_file(gen2_decisions_path)
    historical = _read_jsonl_read_only(gen2_decisions_path)
    historic_index = {(_utc(str(row["ts"])), str(row["symbol"])): row for row in historical}
    if len(historic_index) != len(historical):
        raise BenchmarkIntegrityError("duplicate timestamp/symbol in historical decisions")
    current_index: dict[tuple[datetime, str], Mapping[str, Any]] = {}
    for row in current_rows:
        timestamp = _utc(str(row["timestamp"]))
        feature_timestamp = _utc(str(row["feature_timestamp"]))
        if feature_timestamp > timestamp:
            raise BenchmarkIntegrityError("current decision uses a future feature timestamp")
        if str(row.get("side")) != "SHORT":
            raise BenchmarkIntegrityError("paired benchmark is SHORT-only")
        key = (timestamp, str(row["symbol"]))
        if key in current_index:
            raise BenchmarkIntegrityError("duplicate timestamp/symbol in current decisions")
        current_index[key] = row
    keys = sorted(set(current_index) & set(historic_index))
    pairs = tuple(PairedDecision(
        timestamp=timestamp, symbol=symbol,
        new_score=float(current_index[(timestamp, symbol)]["eqm_score"]),
        gen2_score=float(historic_index[(timestamp, symbol)]["eqm_score"]),
        new_tail_score=float(current_index[(timestamp, symbol)]["tail_score"]),
        gen2_tail_score=float(historic_index[(timestamp, symbol)]["tail_score"]),
        new_qmae_q90=float(current_index[(timestamp, symbol)]["qmae_q90"]),
        gen2_qmae_q90=float(historic_index[(timestamp, symbol)]["qmae_q90"]),
        new_action=str(current_index[(timestamp, symbol)]["hypothetical_action"]),
        gen2_action=str(historic_index[(timestamp, symbol)]["hypothetical_action"]),
    ) for timestamp, symbol in keys)
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    def no_trade(actions: Sequence[str]) -> float:
        return mean([float(value in {"NO_TRADE", "NO_DECISION"}) for value in actions])
    unsigned = {
        "schema_version": "aegis-paired-gen2-benchmark-v1", "bundle_id": bundle_id,
        "bundle_state": bundle_state, "matched_rows": len(pairs),
        "unmatched_new_rows": len(current_index) - len(pairs),
        "unmatched_gen2_rows": len(historic_index) - len(pairs),
        "mean_new_score": mean([row.new_score for row in pairs]),
        "mean_gen2_score": mean([row.gen2_score for row in pairs]),
        "new_no_trade_rate": no_trade([row.new_action for row in pairs]),
        "gen2_no_trade_rate": no_trade([row.gen2_action for row in pairs]),
        "pairs": pairs, "source_hash_before": source_hash, "source_hash_after": sha256_file(gen2_decisions_path),
    }
    return PairedBenchmarkReport(**unsigned, report_hash=Sha256HashProvider().digest_value(unsigned))


def write_benchmark_report(report: PairedBenchmarkReport, output_path: Path) -> Path:
    destination = _assert_outside_gen2(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return destination

