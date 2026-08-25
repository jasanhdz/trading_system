"""Frozen, offline modeling primitives for the W11 ephemeral-regime study."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260825
OPPORTUNITY_THRESHOLD = 0.55
DIRECTION_THRESHOLD = 0.55
MINIMUM_TRAIN_ROWS = 200
MINIMUM_CLASS_ROWS = 10
SIMILARITY_QUANTILE = 0.10
SIMILARITY_METHODS = (
    "standardized_euclidean",
    "cosine",
    "diagonal_covariance",
)
EXPIRATION_PRIORITY = ("EDGE_DECAY", "REGIME_DRIFT", "TTL")
TTL_BY_WINDOW_HOURS = {6: 6, 12: 6, 24: 12, 48: 24, 72: 24}


def _utc(value: datetime | str | pd.Timestamp) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


@dataclass(frozen=True)
class Prediction:
    decision: str
    opportunity_probability: float
    long_probability: float
    expected_edge_bps: float


@dataclass
class CandidateModel:
    """Auditable two-stage logistic candidate which fails closed."""

    opportunity_model: Pipeline | None
    direction_model: Pipeline | None
    active: bool
    failure_reason: str | None
    expected_edge_bps: float
    feature_names: tuple[str, ...]

    def predict(self, features: pd.DataFrame | np.ndarray) -> list[Prediction]:
        rows = _as_frame(features, self.feature_names)
        if not self.active or self.opportunity_model is None or self.direction_model is None:
            return [Prediction("SKIP", 0.0, 0.5, self.expected_edge_bps) for _ in range(len(rows))]

        opportunity = self.opportunity_model.predict_proba(rows)[:, 1]
        long_probability = self.direction_model.predict_proba(rows)[:, 1]
        predictions: list[Prediction] = []
        for p_opp, p_long in zip(opportunity, long_probability, strict=True):
            if p_opp < OPPORTUNITY_THRESHOLD:
                decision = "SKIP"
            elif p_long >= DIRECTION_THRESHOLD:
                decision = "LONG"
            elif p_long <= 1.0 - DIRECTION_THRESHOLD:
                decision = "SHORT"
            else:
                decision = "SKIP"
            predictions.append(
                Prediction(decision, float(p_opp), float(p_long), self.expected_edge_bps)
            )
        return predictions


def _as_frame(
    features: pd.DataFrame | np.ndarray, feature_names: Sequence[str] | None = None
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        if feature_names:
            missing = set(feature_names).difference(features.columns)
            if missing:
                raise ValueError(f"missing model features: {sorted(missing)}")
            return features.loc[:, list(feature_names)]
        return features.copy()
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    names = tuple(feature_names or (f"x{i}" for i in range(array.shape[1])))
    if len(names) != array.shape[1]:
        raise ValueError("feature count does not match fitted model")
    return pd.DataFrame(array, columns=names)


def _binary_labels(values: Sequence[object], positive_strings: set[str]) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value.upper() in positive_strings)
        else:
            result.append(bool(value))
    return np.asarray(result, dtype=int)


def train_candidate(
    features: pd.DataFrame | np.ndarray,
    opportunity_labels: Sequence[object],
    direction_labels: Sequence[object],
    *,
    expected_edge_bps: float = 0.0,
    seed: int = SEED,
) -> CandidateModel:
    """Fit the preregistered candidate, returning an inactive model on any class failure."""

    frame = _as_frame(features)
    opportunity = _binary_labels(opportunity_labels, {"1", "TRUE", "OPPORTUNITY"})
    direction = _binary_labels(direction_labels, {"1", "TRUE", "LONG"})
    if len(frame) != len(opportunity) or len(frame) != len(direction):
        raise ValueError("features and labels must have equal lengths")

    names = tuple(str(column) for column in frame.columns)
    failure: str | None = None
    if len(frame) < MINIMUM_TRAIN_ROWS:
        failure = "INSUFFICIENT_TRAIN_ROWS"
    elif len(np.unique(opportunity)) != 2 or np.bincount(opportunity, minlength=2).min() < MINIMUM_CLASS_ROWS:
        failure = "OPPORTUNITY_CLASS_COUNT"
    else:
        opportunity_directions = direction[opportunity == 1]
        if (
            len(np.unique(opportunity_directions)) != 2
            or np.bincount(opportunity_directions, minlength=2).min() < MINIMUM_CLASS_ROWS
        ):
            failure = "DIRECTION_CLASS_COUNT"

    if failure is not None:
        return CandidateModel(None, None, False, failure, float(expected_edge_bps), names)

    def pipeline() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.25,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    opportunity_model = pipeline().fit(frame, opportunity)
    mask = opportunity == 1
    direction_model = pipeline().fit(frame.loc[mask], direction[mask])
    return CandidateModel(
        opportunity_model,
        direction_model,
        True,
        None,
        float(expected_edge_bps),
        names,
    )


@dataclass(frozen=True)
class SimilarityModel:
    method: str
    threshold: float
    median: np.ndarray = field(repr=False)
    scale: np.ndarray = field(repr=False)
    reference: np.ndarray = field(repr=False)
    reference_variance: np.ndarray = field(repr=False)

    def score(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        values = np.where(np.isnan(values), self.median, values)
        z_values = (values - self.median) / self.scale
        z_reference = (self.reference - self.median) / self.scale
        if self.method == "standardized_euclidean":
            return -np.sqrt(np.sum((z_values - z_reference) ** 2, axis=1))
        if self.method == "cosine":
            denominator = np.linalg.norm(z_values, axis=1) * np.linalg.norm(z_reference)
            numerator = z_values @ z_reference
            return np.divide(numerator, denominator, out=np.zeros(len(values)), where=denominator > 0)
        if self.method == "diagonal_covariance":
            return -np.sqrt(np.sum((values - self.reference) ** 2 / self.reference_variance, axis=1))
        raise ValueError(f"unknown similarity method: {self.method}")


def _similarity_template(training: pd.DataFrame | np.ndarray, method: str) -> SimilarityModel:
    values = np.asarray(training, dtype=float)
    if values.ndim != 2 or len(values) < 4:
        raise ValueError("similarity training requires at least four ordered rows")
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isnan(values), median, values)
    scale = np.std(filled, axis=0)
    scale[scale == 0] = 1.0
    final_quarter = filled[-max(1, len(filled) // 4) :]
    reference = np.mean(final_quarter, axis=0)
    variance = np.var(final_quarter, axis=0)
    variance[variance == 0] = scale[variance == 0] ** 2
    variance[variance == 0] = 1.0
    return SimilarityModel(method, float("nan"), median, scale, reference, variance)


def standardized_euclidean_similarity(
    training: pd.DataFrame | np.ndarray, observations: pd.DataFrame | np.ndarray
) -> np.ndarray:
    return _similarity_template(training, "standardized_euclidean").score(observations)


def cosine_similarity(
    training: pd.DataFrame | np.ndarray, observations: pd.DataFrame | np.ndarray
) -> np.ndarray:
    return _similarity_template(training, "cosine").score(observations)


def diagonal_covariance_similarity(
    training: pd.DataFrame | np.ndarray, observations: pd.DataFrame | np.ndarray
) -> np.ndarray:
    return _similarity_template(training, "diagonal_covariance").score(observations)


def fit_regime_similarity(
    training: pd.DataFrame | np.ndarray,
    validation: pd.DataFrame | np.ndarray,
    validation_net_edge_bps: Sequence[float],
) -> SimilarityModel:
    """Select one diagnostic using validation rank correlation only, then freeze it."""

    edges = np.asarray(validation_net_edge_bps, dtype=float)
    if len(validation) != len(edges):
        raise ValueError("validation features and edge must have equal lengths")
    candidates = [_similarity_template(training, method) for method in SIMILARITY_METHODS]
    relationships: list[float] = []
    for candidate in candidates:
        score = candidate.score(validation)
        if len(np.unique(score)) < 2 or len(np.unique(edges)) < 2:
            relationship = float("nan")
        else:
            relationship = pd.Series(score).corr(pd.Series(edges), method="spearman")
        relationships.append(float(relationship) if np.isfinite(relationship) else float("-inf"))
    # np.argmax preserves the frozen method order, giving standardized Euclidean tie priority.
    selected = candidates[int(np.argmax(relationships))]
    threshold = float(np.quantile(selected.score(training), SIMILARITY_QUANTILE))
    return SimilarityModel(
        selected.method,
        threshold,
        selected.median.copy(),
        selected.scale.copy(),
        selected.reference.copy(),
        selected.reference_variance.copy(),
    )


@dataclass(frozen=True)
class ValidationCandidate:
    window_hours: int
    horizon_minutes: int
    stress_net_bps: float
    baseline_net_bps: float
    trade_count: int
    symbol_counts: Mapping[str, int]
    bootstrap_probability_positive: float
    payload: object = None

    @property
    def eligible(self) -> bool:
        total = sum(self.symbol_counts.values())
        largest_fraction = max(self.symbol_counts.values(), default=0) / total if total else 1.0
        return (
            self.trade_count >= 12
            and total == self.trade_count
            and len(self.symbol_counts) >= 3
            and largest_fraction <= 0.5
            and self.baseline_net_bps > 0.0
            and self.stress_net_bps > 0.0
            and self.bootstrap_probability_positive >= 0.8
        )


def select_validation_candidate(
    candidates: Iterable[ValidationCandidate],
) -> ValidationCandidate | None:
    """Causal per-creation selection: stress edge, shorter window, then horizon."""

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda candidate: (
            -candidate.stress_net_bps,
            candidate.window_hours,
            candidate.horizon_minutes,
        ),
    )[0]


def make_instance_id(
    created_at: datetime | str | pd.Timestamp,
    window_hours: int,
    horizon_minutes: int,
    sequence: int,
) -> str:
    stamp = _utc(created_at).strftime("%Y%m%dT%H%M%SZ")
    return f"ERE_{stamp}_{window_hours}H_{horizon_minutes}M_{sequence}"


@dataclass(frozen=True)
class FrozenExpert:
    instance_id: str
    model_version: str
    window_hours: int
    horizon_minutes: int
    sequence: int
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    created_at: datetime
    expires_at: datetime
    similarity_method: str
    similarity_threshold: float

    def __post_init__(self) -> None:
        fields = (
            "training_start",
            "training_end",
            "validation_start",
            "validation_end",
            "created_at",
            "expires_at",
        )
        for name in fields:
            object.__setattr__(self, name, _utc(getattr(self, name)))
        expected_id = make_instance_id(
            self.created_at, self.window_hours, self.horizon_minutes, self.sequence
        )
        if self.instance_id != expected_id:
            raise ValueError(f"instance_id must equal {expected_id}")
        if self.window_hours not in TTL_BY_WINDOW_HOURS:
            raise ValueError("window_hours is not preregistered")
        expected_expiry = self.created_at + timedelta(hours=TTL_BY_WINDOW_HOURS[self.window_hours])
        if self.expires_at != expected_expiry:
            raise ValueError("expires_at does not match the primary frozen TTL")
        if self.similarity_method not in SIMILARITY_METHODS:
            raise ValueError("similarity_method is not preregistered")

    def attribution(
        self,
        *,
        decision_id: str,
        decision_at: datetime | str | pd.Timestamp,
        similarity: float,
        expected_edge_bps: float,
        decision: str,
        reason: str,
        symbol: str = "",
        confidence: float | None = None,
        expiration_reason: str | None = None,
    ) -> dict[str, object]:
        direction = decision if decision in {"LONG", "SHORT"} else "SKIP"
        return {
            "decision_id": decision_id,
            "decision_at": _utc(decision_at).isoformat(),
            "model_family": "ERE",
            "model_version": self.model_version,
            "model_instance_id": self.instance_id,
            "training_window": {
                "start": self.training_start.isoformat(),
                "end": self.training_end.isoformat(),
            },
            "validation_window": {
                "start": self.validation_start.isoformat(),
                "end": self.validation_end.isoformat(),
            },
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "regime_similarity_at_decision": float(similarity),
            "expected_edge_bps": float(expected_edge_bps),
            "expected_direction": direction,
            "horizon_minutes": self.horizon_minutes,
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "symbol": symbol,
            "expiration_reason": expiration_reason,
        }


@dataclass(frozen=True)
class ResolvedOutcome:
    opened_at: datetime
    resolved_at: datetime
    baseline_net_bps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "opened_at", _utc(self.opened_at))
        object.__setattr__(self, "resolved_at", _utc(self.resolved_at))


@dataclass(frozen=True)
class Expiration:
    instance_id: str
    reason: str
    expired_at: datetime


class ExpirationRegistry:
    """Append-only identity and expiration registry."""

    def __init__(self) -> None:
        self._metadata: dict[str, FrozenExpert] = {}
        self._expired: dict[str, Expiration] = {}

    def register(self, expert: FrozenExpert) -> None:
        if expert.instance_id in self._metadata:
            raise ValueError(f"duplicate instance_id: {expert.instance_id}")
        self._metadata[expert.instance_id] = expert

    def expire(self, expert: FrozenExpert, reason: str, at: datetime) -> Expiration:
        if expert.instance_id not in self._metadata:
            self.register(expert)
        existing = self._expired.get(expert.instance_id)
        if existing is not None:
            return existing
        expiration = Expiration(expert.instance_id, reason, _utc(at))
        self._expired[expert.instance_id] = expiration
        return expiration

    def get(self, instance_id: str) -> Expiration | None:
        return self._expired.get(instance_id)

    @property
    def expired(self) -> Mapping[str, Expiration]:
        return MappingProxyType(self._expired.copy())


@dataclass
class _GuardianState:
    consecutive_drift: int = 0
    last_snapshot_at: datetime | None = None


class ExpirationGuardian:
    def __init__(self, registry: ExpirationRegistry | None = None) -> None:
        self.registry = registry or ExpirationRegistry()
        self._state: dict[str, _GuardianState] = {}

    def add(self, expert: FrozenExpert) -> None:
        self.registry.register(expert)
        self._state[expert.instance_id] = _GuardianState()

    def evaluate(
        self,
        expert: FrozenExpert,
        at: datetime | str | pd.Timestamp,
        *,
        similarity: float,
        resolved_outcomes: Iterable[ResolvedOutcome] = (),
    ) -> Expiration | None:
        now = _utc(at)
        existing = self.registry.get(expert.instance_id)
        if existing is not None:
            return existing
        if expert.instance_id not in self._state:
            self.add(expert)
        state = self._state[expert.instance_id]

        valid = sorted(
            (
                outcome
                for outcome in resolved_outcomes
                if outcome.opened_at >= expert.created_at
                and expert.created_at < outcome.resolved_at <= now
            ),
            key=lambda outcome: (outcome.resolved_at, outcome.opened_at),
        )
        edge_decay = (
            len(valid) >= 12
            and float(np.mean([outcome.baseline_net_bps for outcome in valid[-12:]])) < -2.0
        )

        if state.last_snapshot_at is None or now > state.last_snapshot_at:
            state.consecutive_drift = (
                state.consecutive_drift + 1
                if similarity < expert.similarity_threshold
                else 0
            )
            state.last_snapshot_at = now
        regime_drift = state.consecutive_drift >= 3
        ttl = now >= expert.expires_at
        conditions = {"EDGE_DECAY": edge_decay, "REGIME_DRIFT": regime_drift, "TTL": ttl}
        for reason in EXPIRATION_PRIORITY:
            if conditions[reason]:
                return self.registry.expire(expert, reason, now)
        return None

    def is_active(self, expert: FrozenExpert, at: datetime | str | pd.Timestamp) -> bool:
        return self.registry.get(expert.instance_id) is None and _utc(at) < expert.expires_at


@dataclass(frozen=True)
class BootstrapResult:
    observed_mean: float
    probability_positive: float
    ci_lower: float
    ci_upper: float
    draws: np.ndarray = field(repr=False)


def temporal_block_bootstrap(
    timestamps: Sequence[datetime | str | pd.Timestamp],
    values: Sequence[float],
    *,
    block_hours: int = 1,
    draws: int = 500,
    seed: int = SEED,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Resample complete synchronized UTC temporal blocks with stable ordering."""

    if len(timestamps) != len(values) or len(values) == 0:
        raise ValueError("timestamps and non-empty values must have equal lengths")
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamps, utc=True), "value": np.asarray(values, dtype=float)}
    ).sort_values("timestamp", kind="stable")
    frequency = f"{block_hours}h"
    frame["block"] = frame["timestamp"].dt.floor(frequency)
    blocks = [group["value"].to_numpy() for _, group in frame.groupby("block", sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[draw] = float(np.mean(np.concatenate([blocks[index] for index in selected])))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        observed_mean=float(frame["value"].mean()),
        probability_positive=float(np.mean(samples > 0.0)),
        ci_lower=float(np.quantile(samples, alpha)),
        ci_upper=float(np.quantile(samples, 1.0 - alpha)),
        draws=samples,
    )
