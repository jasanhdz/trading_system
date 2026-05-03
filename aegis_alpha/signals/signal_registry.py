from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalSpec:
    name: str
    side: str
    horizon: int
    target_type: str
    model_type: str
    status: str = "research"


SIGNAL_REGISTRY: tuple[SignalSpec, ...] = (
    SignalSpec("long_edge_h6", "LONG", 6, "long_net_return", "regressor"),
    SignalSpec("long_edge_h12", "LONG", 12, "long_net_return", "regressor"),
    SignalSpec("long_edge_h24", "LONG", 24, "long_net_return", "regressor"),
    SignalSpec("long_edge_h48", "LONG", 48, "long_net_return", "regressor"),
    SignalSpec("short_edge_h6", "SHORT", 6, "short_net_return", "regressor"),
    SignalSpec("short_edge_h12", "SHORT", 12, "short_net_return", "regressor"),
    SignalSpec("short_edge_h24", "SHORT", 24, "short_net_return", "regressor"),
    SignalSpec("short_edge_h48", "SHORT", 48, "short_net_return", "regressor"),
    SignalSpec("long_failure_risk_h12", "LONG", 12, "failure_bad_long", "classifier"),
    SignalSpec("long_failure_risk_h24", "LONG", 24, "failure_bad_long", "classifier"),
    SignalSpec("long_failure_risk_h48", "LONG", 48, "failure_bad_long", "classifier"),
)


def get_signal_spec(name: str) -> SignalSpec:
    for spec in SIGNAL_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(name)

