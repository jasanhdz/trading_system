"""Deterministic audit for Entry Quality V2 Shadow and paper journals."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import CANONICAL_SYMBOLS


class EntryQualityV2EvidenceError(RuntimeError):
    pass


def _rows(path: Path, identity: str) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    parsed: list[dict[str, Any]] = []
    identities: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                raise EntryQualityV2EvidenceError(
                    "AEGIS_ENTRY_QUALITY_V2_EVIDENCE_INVALID"
                )
            row = json.loads(line)
            key = str(row[identity])
            if key in identities:
                raise EntryQualityV2EvidenceError(
                    "AEGIS_ENTRY_QUALITY_V2_EVIDENCE_DUPLICATE"
                )
            identities.add(key)
            parsed.append(row)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EntryQualityV2EvidenceError(
            "AEGIS_ENTRY_QUALITY_V2_EVIDENCE_INVALID"
        ) from exc
    return tuple(parsed)


def _finite(value: Any, identity: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EntryQualityV2EvidenceError(f"{identity} is non-finite")
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _performance(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    net = [_finite(row["net_return_fraction"], "net_return_fraction") for row in rows]
    mae = [_finite(row["mae_fraction"], "mae_fraction") for row in rows]
    mfe = [_finite(row["mfe_fraction"], "mfe_fraction") for row in rows]
    return {
        "matured_episodes": len(rows),
        "mean_net_return_fraction": statistics.fmean(net) if net else None,
        "win_rate": sum(value > 0.0 for value in net) / len(net) if net else None,
        "mean_mae_fraction": statistics.fmean(mae) if mae else None,
        "mean_mfe_fraction": statistics.fmean(mfe) if mfe else None,
        "maximum_mae_fraction": max(mae) if mae else None,
    }


def audit_entry_quality_v2_evidence(
    signal_path: Path,
    outcome_path: Path,
    *,
    minimum_matured_episodes: int = 300,
) -> Mapping[str, Any]:
    if minimum_matured_episodes <= 0:
        raise ValueError("minimum matured episodes must be positive")
    signals = _rows(signal_path, "event_id")
    outcomes = _rows(outcome_path, "event_id")
    signal_by_id = {str(row["event_id"]): row for row in signals}
    joined: list[dict[str, Any]] = []
    for outcome in outcomes:
        event_id = str(outcome["event_id"])
        signal = signal_by_id.get(event_id)
        if signal is None:
            raise EntryQualityV2EvidenceError(
                "AEGIS_ENTRY_QUALITY_V2_OUTCOME_WITHOUT_SIGNAL"
            )
        joined.append(
            {
                **outcome,
                "score": _finite(signal["v2"]["score"], "v2.score"),
                "selected": bool(signal["v2"]["selected"]),
                "signal_timestamp": str(signal["market_timestamp"]),
            }
        )
    selected = [row for row in signals if bool(row["v2"]["selected"])]
    by_symbol = {
        symbol: _performance(
            [row for row in joined if str(row["symbol"]) == symbol]
        )
        for symbol in CANONICAL_SYMBOLS
    }
    scores = [_finite(row["score"], "score") for row in joined]
    returns = [
        _finite(row["net_return_fraction"], "net_return_fraction")
        for row in joined
    ]
    non_overlapping: list[dict[str, Any]] = []
    for symbol in CANONICAL_SYMBOLS:
        last_timestamp = None
        for row in sorted(
            (item for item in joined if str(item["symbol"]) == symbol),
            key=lambda item: str(item["signal_timestamp"]),
        ):
            timestamp = datetime.fromisoformat(
                str(row["signal_timestamp"]).replace("Z", "+00:00")
            )
            if last_timestamp is None or timestamp >= last_timestamp + timedelta(
                minutes=5 * int(row["horizon_bars"])
            ):
                non_overlapping.append(row)
                last_timestamp = timestamp
    selected_joined = [row for row in joined if bool(row["selected"])]
    return {
        "schema_id": "aegis-entry-quality-v2-shadow-evidence-audit-v1",
        "signal_records": len(signals),
        "decision_cycles": len(
            {str(row["decision_cycle_id"]) for row in signals}
        ),
        "symbols_observed": sorted({str(row["symbol"]) for row in signals}),
        "control_selected_count": sum(
            bool(row["control"]["selected"]) for row in signals
        ),
        "v2_selected_count": len(selected),
        "matured_episode_count": len(joined),
        "matured_selected_episode_count": len(selected_joined),
        "non_overlapping_episode_count": len(non_overlapping),
        "minimum_matured_episodes": minimum_matured_episodes,
        "global_performance": _performance(joined),
        "selected_performance": _performance(selected_joined),
        "non_overlapping_performance": _performance(non_overlapping),
        "per_symbol": by_symbol,
        "score_net_return_correlation": _correlation(scores, returns),
        "exchange_mutations": 0,
        "automatic_promotion": False,
        "evidence_state": (
            "EVIDENCE_READY_FOR_OWNER_REVIEW"
            if len(non_overlapping) >= minimum_matured_episodes
            else "COLLECTING_SHADOW_EVIDENCE"
        ),
    }
