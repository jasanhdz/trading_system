from __future__ import annotations

from dataclasses import dataclass
from itertools import product

ALLOWED_VALUES = {
    "cluster_tolerance_atr": (0.20, 0.30),
    "min_range_amplitude_pct": (0.0125, 0.0200, 0.0300),
    "rejection_min_wick_body_ratio": (1.0, 1.5),
    "stop_buffer_atr": (0.35, 0.50),
    "target_buffer_atr": (0.00, 0.10),
    "max_adx": (20.0, 25.0),
    "min_chop_risk": (0.62, 0.70),
    "min_safety_volume_ratio": (0.50, 0.75),
}


@dataclass(frozen=True, slots=True)
class RangeCandidate:
    cluster_tolerance_atr: float
    min_range_amplitude_pct: float
    rejection_min_wick_body_ratio: float
    stop_buffer_atr: float
    target_buffer_atr: float
    max_adx: float
    min_chop_risk: float
    min_safety_volume_ratio: float

    def __post_init__(self) -> None:
        for name, allowed in ALLOWED_VALUES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} is outside the preregistered grid")

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in ALLOWED_VALUES}


def candidate_grid() -> tuple[RangeCandidate, ...]:
    names = tuple(ALLOWED_VALUES)
    return tuple(
        RangeCandidate(**dict(zip(names, values, strict=True)))
        for values in product(*(ALLOWED_VALUES[name] for name in names))
    )
