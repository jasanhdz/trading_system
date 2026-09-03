"""Mature frozen prospective signals into immutable row-level SHORT targets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import Candle
from ..training.labels import ShortLabelConfig, build_short_path_label
from ..utils import Sha256HashProvider, canonical_json, sha256_file


PROTOCOL_VERSION = "aegis-prospective-validation-v1"
SIGNAL_SCHEMA = "aegis-prospective-signal-evidence-v1"
OUTCOME_SCHEMA = "aegis-prospective-outcome-v1"
LABEL_CODE_SHA256 = "e5b7dbdfe1bb1b6156d34aba76e1d00b625ecb24fca3ac59421e96912e1d2453"


class ProspectiveOutcomeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _utc(value: str, code: str) -> datetime:
    if not value.endswith("Z") and not (len(value) >= 6 and value[-6] in "+-" and value[-3] == ":"):
        raise ProspectiveOutcomeError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveOutcomeError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveOutcomeError(code)
    return parsed.astimezone(timezone.utc)


def _hash(value: Any) -> str:
    return Sha256HashProvider().digest_value(value)


@dataclass(frozen=True)
class ActivationContract:
    active: bool
    activated_at_utc: datetime
    cohort_id: str
    model_artifact_hash: str
    configuration_hash: str
    python_commit: str
    typescript_commit: str

    def __post_init__(self) -> None:
        if self.activated_at_utc.tzinfo is None or self.activated_at_utc.utcoffset() is None:
            raise ProspectiveOutcomeError("PROSPECTIVE_ACTIVATION_TIME_INVALID")
        for value, code, size in (
            (self.model_artifact_hash, "PROSPECTIVE_MODEL_HASH_INVALID", 64),
            (self.configuration_hash, "PROSPECTIVE_CONFIG_HASH_INVALID", 64),
            (self.python_commit, "PROSPECTIVE_CODE_HASH_INVALID", 40),
            (self.typescript_commit, "PROSPECTIVE_CODE_HASH_INVALID", 40),
        ):
            if len(value) != size or any(character not in "0123456789abcdef" for character in value):
                raise ProspectiveOutcomeError(code)


@dataclass(frozen=True)
class ProspectiveSignalEvidence:
    prospective_signal_id: str
    cohort_id: str
    symbol: str
    side: str
    signal_timestamp_utc: datetime
    information_cutoff_utc: datetime
    model_artifact_hash: str
    configuration_hash: str
    source_python_commit: str
    source_typescript_commit: str

    @classmethod
    def parse(cls, value: Mapping[str, Any], activation: ActivationContract) -> "ProspectiveSignalEvidence":
        required = {
            "schema_id", "prospective_signal_id", "cohort_id", "protocol_version", "symbol", "side",
            "signal_timestamp_utc", "information_cutoff_utc", "model_artifact_hash", "configuration_hash",
            "source_python_commit", "source_typescript_commit",
        }
        if not required.issubset(value):
            raise ProspectiveOutcomeError("PROSPECTIVE_SIGNAL_SCHEMA_INVALID")
        if value["schema_id"] != SIGNAL_SCHEMA or value["protocol_version"] != PROTOCOL_VERSION:
            raise ProspectiveOutcomeError("PROSPECTIVE_PROTOCOL_MISMATCH")
        signal_time = _utc(str(value["signal_timestamp_utc"]), "PROSPECTIVE_SIGNAL_TIME_INVALID")
        cutoff = _utc(str(value["information_cutoff_utc"]), "PROSPECTIVE_INFORMATION_CUTOFF_INVALID")
        if cutoff > signal_time:
            raise ProspectiveOutcomeError("PROSPECTIVE_INFORMATION_CUTOFF_INVALID")
        if not activation.active or signal_time < activation.activated_at_utc.astimezone(timezone.utc):
            raise ProspectiveOutcomeError("PROSPECTIVE_PREACTIVATION_SIGNAL")
        checks = (
            ("cohort_id", activation.cohort_id, "PROSPECTIVE_COHORT_MISMATCH"),
            ("model_artifact_hash", activation.model_artifact_hash, "PROSPECTIVE_MODEL_HASH_MISMATCH"),
            ("configuration_hash", activation.configuration_hash, "PROSPECTIVE_CONFIG_HASH_MISMATCH"),
            ("source_python_commit", activation.python_commit, "PROSPECTIVE_CODE_HASH_MISMATCH"),
            ("source_typescript_commit", activation.typescript_commit, "PROSPECTIVE_CODE_HASH_MISMATCH"),
        )
        for field, expected, code in checks:
            if value[field] != expected:
                raise ProspectiveOutcomeError(code)
        signal_id = str(value["prospective_signal_id"])
        if len(signal_id) != 64 or any(character not in "0123456789abcdef" for character in signal_id):
            raise ProspectiveOutcomeError("PROSPECTIVE_SIGNAL_ID_INVALID")
        symbol = str(value["symbol"])
        if symbol != symbol.upper() or not symbol.isalnum():
            raise ProspectiveOutcomeError("PROSPECTIVE_SIGNAL_SYMBOL_INVALID")
        if value["side"] not in {"SHORT", "NO_TRADE"}:
            raise ProspectiveOutcomeError("PROSPECTIVE_SIGNAL_SIDE_INVALID")
        return cls(
            signal_id, activation.cohort_id, symbol, str(value["side"]), signal_time, cutoff,
            activation.model_artifact_hash, activation.configuration_hash,
            activation.python_commit, activation.typescript_commit,
        )


class ProspectiveOutcomeJournal:
    """Append-only canonical journal with duplicate and conflict detection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._payloads: dict[str, str] = {}
        if path.exists():
            self._recover()

    def _recover(self) -> None:
        try:
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line:
                    raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_JOURNAL_INVALID")
                value = json.loads(line)
                signal_id = str(value["prospective_signal_id"])
                payload = canonical_json(value)
                if signal_id in self._payloads:
                    raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_DUPLICATE")
                self._payloads[signal_id] = payload
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_JOURNAL_INVALID") from exc

    def append(self, outcome: Mapping[str, Any]) -> None:
        signal_id = str(outcome["prospective_signal_id"])
        payload = canonical_json(outcome)
        existing = self._payloads.get(signal_id)
        if existing == payload:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_DUPLICATE")
        if existing is not None:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_CONFLICT")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_PERSISTENCE_FAILED") from exc
        self._payloads[signal_id] = payload

    @property
    def sha256(self) -> str:
        return sha256_file(self.path) if self.path.exists() else _hash("")


class ProspectiveOutcomeMaturator:
    def __init__(self, activation: ActivationContract, journal: ProspectiveOutcomeJournal) -> None:
        self.activation = activation
        self.journal = journal
        self.config = ShortLabelConfig()
        self.label_config_sha256 = _hash(asdict(self.config))

    def mature_and_persist(
        self,
        envelope: Mapping[str, Any],
        signal_candle: Candle,
        future_candles: Sequence[Candle],
        *,
        as_of_utc: datetime,
    ) -> Mapping[str, Any]:
        signal = ProspectiveSignalEvidence.parse(envelope, self.activation)
        if signal.side != "SHORT":
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_NO_TRADE_PROHIBITED")
        if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_TIME_INVALID")
        if signal_candle.close_time != signal.signal_timestamp_utc:
            raise ProspectiveOutcomeError("PROSPECTIVE_SIGNAL_CANDLE_MISMATCH")
        if len(future_candles) != self.config.horizon_bars:
            raise ProspectiveOutcomeError("PROSPECTIVE_MARKET_DATA_INCOMPLETE")
        termination = future_candles[-1].close_time
        if as_of_utc.astimezone(timezone.utc) < termination:
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_PREMATURE")
        label = build_short_path_label(signal_candle, future_candles, self.config)
        if not label.valid:
            raise ProspectiveOutcomeError("PROSPECTIVE_MARKET_DATA_INCOMPLETE")
        required = (
            label.terminal_short_return,
            label.mfe_fraction,
            label.mae_fraction,
            label.net_quality_after_costs,
            label.round_trip_cost_fraction,
        )
        if any(value is None or not math.isfinite(value) for value in required):
            raise ProspectiveOutcomeError("PROSPECTIVE_OUTCOME_INVALID")
        gross = float(label.terminal_short_return)
        fees = 2.0 * self.config.fee_bps_per_side / 10_000.0
        slippage = 2.0 * self.config.slippage_bps_per_side / 10_000.0
        funding = self.config.funding_bps_per_hour / 10_000.0
        outcome = {
            "schema_id": OUTCOME_SCHEMA,
            "prospective_signal_id": signal.prospective_signal_id,
            "horizon_bars": self.config.horizon_bars,
            "termination_timestamp_utc": termination.isoformat().replace("+00:00", "Z"),
            "gross_return_fraction": gross,
            "net_return_fraction": gross - fees - slippage - funding,
            "mfe_fraction": float(label.mfe_fraction),
            "mae_fraction": float(label.mae_fraction),
            "tail_event": int(label.tail_event),
            "qmae": float(label.mae_fraction),
            "clean_quality": int(label.clean_entry),
            "net_quality_after_costs": float(label.net_quality_after_costs),
            "label_valid": True,
            "fees_fraction": fees,
            "slippage_fraction": slippage,
            "funding_fraction": funding,
            "termination_reason": "HORIZON_COMPLETE",
            "missingness": [],
            "validation_state": "VALID",
            "label_code_sha256": LABEL_CODE_SHA256,
            "label_config_sha256": self.label_config_sha256,
        }
        self.journal.append(outcome)
        return outcome

