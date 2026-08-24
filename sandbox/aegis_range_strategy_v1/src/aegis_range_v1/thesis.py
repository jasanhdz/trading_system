from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from .candidates import RangeCandidate
from .models import PendingEntry, Side
from .numeric import canonical_decimal_12dp, iso_utc_millis

SCHEMA_VERSION = "aegis-range-thesis-v1"
TYPESCRIPT_GIT_HEAD = "bb034431e0ce05c8e0f978453c46dcff6efb981c"
SOURCE_MANIFEST_SHA256 = "39cd5b8371ef4d193fcce22e6d6392ceaff8b802b52b4589d0c27cfa583a7704"
SPLIT_MANIFEST_SHA256 = "a67766d8ab446c657260550d37c55589d94cc11afba064ae3f043db803868c03"

THESIS_KEYS = frozenset(
    {
        "ATR_entry",
        "cluster_tolerance_atr",
        "cost_scenario",
        "decision_at",
        "entry_available_at",
        "entry_fill",
        "max_adx",
        "midpoint_at_entry",
        "min_chop_risk",
        "min_range_amplitude_pct",
        "min_safety_volume_ratio",
        "range_confidence_at_entry",
        "range_confirmed_at",
        "range_episode_id",
        "range_id",
        "regime_at_entry",
        "rejection_min_wick_body_ratio",
        "resistance_at_entry",
        "schema_version",
        "side",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "stop_at_entry",
        "stop_buffer_atr",
        "support_at_entry",
        "symbol",
        "tail_risk_score_at_entry",
        "target_at_entry",
        "target_buffer_atr",
        "typescript_git_head",
    }
)


@dataclass(frozen=True, slots=True)
class ThesisArtifact:
    payload: dict[str, str | None]
    serialized: str
    sha256: str


def build_thesis(
    pending: PendingEntry,
    candidate: RangeCandidate,
    entry_fill: float,
    stop: float,
    target: float,
    *,
    cost_scenario: str = "BASELINE",
) -> ThesisArtifact:
    payload: dict[str, str | None] = {
        "ATR_entry": canonical_decimal_12dp(pending.atr_entry),
        "cluster_tolerance_atr": canonical_decimal_12dp(candidate.cluster_tolerance_atr),
        "cost_scenario": cost_scenario,
        "decision_at": iso_utc_millis(pending.decision_at),
        "entry_available_at": iso_utc_millis(pending.entry_available_at),
        "entry_fill": canonical_decimal_12dp(entry_fill),
        "max_adx": canonical_decimal_12dp(candidate.max_adx),
        "midpoint_at_entry": canonical_decimal_12dp(pending.midpoint),
        "min_chop_risk": canonical_decimal_12dp(candidate.min_chop_risk),
        "min_range_amplitude_pct": canonical_decimal_12dp(candidate.min_range_amplitude_pct),
        "min_safety_volume_ratio": canonical_decimal_12dp(candidate.min_safety_volume_ratio),
        "range_confidence_at_entry": canonical_decimal_12dp(pending.range_confidence_at_entry),
        "range_confirmed_at": iso_utc_millis(pending.range_confirmed_at),
        "range_episode_id": pending.range_episode_id,
        "range_id": pending.range_id,
        "regime_at_entry": pending.regime_at_entry,
        "rejection_min_wick_body_ratio": canonical_decimal_12dp(candidate.rejection_min_wick_body_ratio),
        "resistance_at_entry": canonical_decimal_12dp(pending.resistance),
        "schema_version": SCHEMA_VERSION,
        "side": pending.side,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "stop_at_entry": canonical_decimal_12dp(stop),
        "stop_buffer_atr": canonical_decimal_12dp(candidate.stop_buffer_atr),
        "support_at_entry": canonical_decimal_12dp(pending.support),
        "symbol": pending.symbol,
        "tail_risk_score_at_entry": None if pending.tail_risk_score_at_entry is None else canonical_decimal_12dp(pending.tail_risk_score_at_entry),
        "target_at_entry": canonical_decimal_12dp(target),
        "target_buffer_atr": canonical_decimal_12dp(candidate.target_buffer_atr),
        "typescript_git_head": TYPESCRIPT_GIT_HEAD,
    }
    if frozenset(payload) != THESIS_KEYS:
        raise RuntimeError("thesis schema mismatch")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ThesisArtifact(payload, serialized, hashlib.sha256(serialized.encode("utf-8")).hexdigest())
