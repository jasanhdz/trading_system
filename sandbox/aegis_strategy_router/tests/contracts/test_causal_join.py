from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from aegis_strategy_router.adapters.causal_join import causal_asof


@dataclass(frozen=True)
class Row:
    available_at: datetime
    value: int


def test_backward_asof_never_selects_future() -> None:
    start = datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc)
    rows = [Row(start + timedelta(minutes=index), index) for index in range(4)]
    selected = causal_asof(
        start + timedelta(minutes=1, seconds=30), rows,
        available_at=lambda row: row.available_at,
    )
    assert selected == rows[1]


def test_exact_match_policy_is_explicit() -> None:
    start = datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc)
    rows = [Row(start, 0), Row(start + timedelta(minutes=1), 1)]
    assert causal_asof(start + timedelta(minutes=1), rows, available_at=lambda row: row.available_at) == rows[1]
    assert causal_asof(
        start + timedelta(minutes=1), rows,
        available_at=lambda row: row.available_at,
        allow_exact_matches=False,
    ) == rows[0]

