from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from aegis_range_v1.candidates import ALLOWED_VALUES, RangeCandidate, candidate_grid
from aegis_range_v1.models import PendingEntry
from aegis_range_v1.numeric import canonical_decimal_12dp, range_episode_id, range_id
from aegis_range_v1.thesis import THESIS_KEYS, build_thesis


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("1.25"), "1.250000000000"),
        (Decimal("-1.25"), "-1.250000000000"),
        (Decimal("1.0000000000005"), "1.000000000000"),
        (Decimal("1.0000000000015"), "1.000000000002"),
        (Decimal("9.9999999999995"), "10.000000000000"),
        (Decimal("0"), "0.000000000000"),
        (Decimal("-0"), "0.000000000000"),
    ],
)
def test_canonical_decimal_half_even_vectors(value, expected):
    assert canonical_decimal_12dp(value) == expected


def test_candidate_grid_is_exactly_384_and_rejects_outside_values(candidate):
    grid = candidate_grid()
    assert len(grid) == 384
    assert len(set(grid)) == 384
    for name, expected in ALLOWED_VALUES.items():
        assert tuple(sorted({getattr(item, name) for item in grid})) == tuple(sorted(expected))
    with pytest.raises(ValueError):
        RangeCandidate(0.25, 0.0125, 1.0, 0.35, 0.0, 25, 0.62, 0.5)


def make_pending(origin, tail=None):
    return PendingEntry(
        "BTCUSDT",
        "LONG",
        origin,
        origin + timedelta(minutes=5),
        "episode-id",
        "range-id",
        origin - timedelta(hours=1),
        90.0,
        110.0,
        100.0,
        2.0,
        "ACCUMULATION_RANGE",
        73.25,
        tail,
    )


def test_ids_are_deterministic_and_snapshot_sensitive(origin):
    episode_a = range_episode_id("BTCUSDT", origin, "support", "resistance")
    episode_b = range_episode_id("BTCUSDT", origin, "support", "resistance")
    assert episode_a == episode_b
    first = range_id(episode_a, origin, 90.0, 110.0, 100.0)
    assert first == range_id(episode_a, origin, 90.0, 110.0, 100.0)
    assert first != range_id(episode_a, origin + timedelta(minutes=5), 90.0, 110.0, 100.0)


def test_thesis_exact_schema_sorted_utf8_and_e4_only_changes_hash(origin, candidate):
    without_e4 = build_thesis(make_pending(origin), candidate, 94.0, 89.3, 100.0)
    with_e4 = build_thesis(make_pending(origin, 0.25), candidate, 94.0, 89.3, 100.0)
    decoded = json.loads(without_e4.serialized)
    assert frozenset(decoded) == THESIS_KEYS
    assert list(decoded) == sorted(decoded)
    assert " " not in without_e4.serialized
    assert decoded["tail_risk_score_at_entry"] is None
    assert without_e4.sha256 != with_e4.sha256
    assert decoded["side"] == json.loads(with_e4.serialized)["side"]
    assert decoded["range_id"] == json.loads(with_e4.serialized)["range_id"]


def test_thesis_golden_hash(origin, candidate):
    thesis = build_thesis(make_pending(origin), candidate, 94.0, 89.3, 100.0)
    fixture_path = Path(__file__).parents[1] / "fixtures" / "thesis_golden_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["synthetic_only"] is True
    assert thesis.sha256 == fixture["expected_sha256"]
