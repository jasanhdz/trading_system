"""Qualified Python scientific bridge for the public-only Shadow service."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aegis.config import CANONICAL_SYMBOLS, load_brain_config
from aegis.domain import Candle, PortfolioContext, decision_request_from_dict
from aegis.features import DeterministicFeaturePipeline, MarketSnapshotValidator
from aegis.prospective.activation import validate_activation_record
from aegis.prospective.model_qualification import (
    CANDIDATE_IDENTITY,
    load_qualified_candidate,
)
from aegis.prospective.outcomes import (
    ActivationContract,
    ProspectiveOutcomeJournal,
    ProspectiveOutcomeMaturator,
)
from aegis.training.experiment import evaluate_authoritative_feature_batch
from aegis.utils import Sha256HashProvider, canonical_json, to_primitive


CONFIGURATION_SHA256 = "f944b0210b31928a519dc63459be3f1d53de811517dc1bbe9753596314579ec1"
MODEL_ARTIFACT_SHA256 = "386742c20d74a3b67d47cd95629c646195472e05e9e8d136587d40989a82e3d1"


class ShadowBridgeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _hash(value: Any) -> str:
    return Sha256HashProvider().digest_value(value)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ShadowBridgeError("SHADOW_TIMESTAMP_INVALID")
    return parsed.astimezone(timezone.utc)


def _component(status: str, source: Mapping[str, Any], output: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": status,
        "input_hash": _hash(source),
        "output_hash": _hash(output),
        "output": output,
    }


def evaluate_snapshot(
    payload: Mapping[str, Any], *, config_dir: Path, candidate_path: Path,
    activation_path: Path,
) -> Mapping[str, Any]:
    activation = validate_activation_record(activation_path)
    if activation["configuration_sha256"] != CONFIGURATION_SHA256:
        raise ShadowBridgeError("PROSPECTIVE_CONFIG_HASH_MISMATCH")
    if activation["trained_artifact_sha256"] != MODEL_ARTIFACT_SHA256:
        raise ShadowBridgeError("PROSPECTIVE_MODEL_HASH_MISMATCH")
    candidate = load_qualified_candidate(candidate_path)
    if candidate.model_identity != CANDIDATE_IDENTITY or candidate.model_artifact_hash != MODEL_ARTIFACT_SHA256:
        raise ShadowBridgeError("PROSPECTIVE_MODEL_IDENTITY_MISMATCH")
    config = load_brain_config(config_dir)
    if config.config_hash != CONFIGURATION_SHA256:
        raise ShadowBridgeError("PROSPECTIVE_CONFIG_HASH_MISMATCH")

    request = decision_request_from_dict(payload)
    if request.snapshot.closed_at < _utc(str(activation["activation_timestamp_utc"])):
        raise ShadowBridgeError("PROSPECTIVE_EVENT_BEFORE_ACTIVATION")
    MarketSnapshotValidator(config.universe).validate(request.snapshot, datetime.now(timezone.utc))
    features = DeterministicFeaturePipeline(
        candidate.source.normalizer,
        schema_version=candidate.source.feature_schema_version,
    ).transform(request.snapshot)
    pipeline = evaluate_authoritative_feature_batch(
        candidate.source,
        features,
        timestamp=request.snapshot.closed_at,
        config={"protocol": {"friction_fraction": 0.001}},
        request_id=request.request_id,
        decision_cycle_id=request.decision_cycle_id,
        portfolio=request.snapshot.portfolio,
    )
    selected_hashes = {item.candidate_hash for item in pipeline.selection.selected}
    predictions = {item.symbol: item for item in pipeline.predictions.predictions}
    layers = {item.symbol: item for item in pipeline.layers.results}
    candidates = []
    for candidate_row in pipeline.candidates.candidates:
        prediction = predictions[candidate_row.symbol]
        layer = layers[candidate_row.symbol]
        upstream = to_primitive(prediction)
        d3_output = {
            "decision": "ENTER_NOW" if candidate_row.candidate_hash in selected_hashes else "DO_NOT_ENTER",
            "regime": layer.regime.value,
            "regime_confidence": layer.regime_confidence,
            "side": layer.side.value,
        }
        rv2_output = {"tail_risk_probability": layer.rv2_tail_risk}
        trrm_output = {
            "compatibility": layer.trrm_compatibility,
            "passed": "TRRM_TAIL_RISK_VETO" not in {reason.value for reason in layer.reason_codes},
        }
        qmae_output = {
            "q90": layer.qmae_q90,
            "quality": layer.qmae_quality,
            "valid": prediction.qmae_valid,
        }
        eqm_output = {"score": layer.eqm_score, "eligible": layer.eligible}
        econ1_output = {
            "calibrated_score": candidate_row.calibrated_score,
            "expected_return": candidate_row.expected_return,
            "eligible": candidate_row.eligible,
        }
        source = {"symbol": candidate_row.symbol, "feature_hash": features.feature_hash}
        components = {
            "d3": _component("PASS", source, d3_output),
            "rv2": _component("PASS", upstream, rv2_output),
            "trrm": _component("PASS" if trrm_output["passed"] else "REJECT", rv2_output, trrm_output),
            "qmae": _component("PASS" if prediction.qmae_valid else "REJECT", upstream, qmae_output),
            "eqm": _component("PASS" if layer.eqm_score >= 0 else "REJECT", qmae_output, eqm_output),
            "econ1": _component("PASS" if candidate_row.eligible else "REJECT", eqm_output, econ1_output),
        }
        candidates.append({
            "symbol": candidate_row.symbol,
            "side": "SHORT" if candidate_row.side.value == "SHORT" else "NO_TRADE",
            "candidate": to_primitive(candidate_row),
            "upstream_model": upstream,
            "component_evidence": components,
            "selected": candidate_row.candidate_hash in selected_hashes,
        })
    if {item["symbol"] for item in candidates} != set(CANONICAL_SYMBOLS):
        raise ShadowBridgeError("PROSPECTIVE_CANDIDATE_POPULATION_INVALID")
    return {
        "schema_id": "aegis-prospective-shadow-bridge-result-v1",
        "cohort_id": activation["cohort_id"],
        "model_identity": CANDIDATE_IDENTITY,
        "model_artifact_hash": MODEL_ARTIFACT_SHA256,
        "configuration_hash": CONFIGURATION_SHA256,
        "decision_cycle_id": request.decision_cycle_id,
        "signal_timestamp_utc": request.snapshot.closed_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "selection": to_primitive(pipeline.selection),
        "candidates": candidates,
    }


def _candle(value: Mapping[str, Any]) -> Candle:
    return Candle(
        _utc(str(value["open_time"])), _utc(str(value["close_time"])),
        float(value["open"]), float(value["high"]), float(value["low"]),
        float(value["close"]), float(value["volume"]), bool(value["is_closed"]),
        str(value["source"]), str(value["sequence"]) if value.get("sequence") is not None else None,
    )


def mature(payload: Mapping[str, Any], *, activation_path: Path, journal_path: Path) -> Mapping[str, Any]:
    record = validate_activation_record(activation_path)
    activation = ActivationContract(
        True, _utc(str(record["activation_timestamp_utc"])), str(record["cohort_id"]),
        str(record["trained_artifact_sha256"]), str(record["configuration_sha256"]),
        str(record["python_commit"]), str(record["typescript_commit"]),
    )
    future = tuple(_candle(item) for item in payload["future_candles"])
    return ProspectiveOutcomeMaturator(
        activation, ProspectiveOutcomeJournal(journal_path),
    ).mature_and_persist(
        payload["envelope"], _candle(payload["signal_candle"]), future,
        as_of_utc=_utc(str(payload["as_of_utc"])),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualified public-only prospective Shadow bridge")
    parser.add_argument("command", choices=("evaluate", "mature"))
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--outcome-journal", type=Path)
    args = parser.parse_args(argv)
    try:
        incoming = json.load(sys.stdin)
        if args.command == "evaluate":
            result = evaluate_snapshot(
                incoming, config_dir=args.config_dir, candidate_path=args.candidate,
                activation_path=args.activation,
            )
        else:
            if args.outcome_journal is None:
                raise ShadowBridgeError("PROSPECTIVE_OUTCOME_JOURNAL_REQUIRED")
            result = mature(incoming, activation_path=args.activation, journal_path=args.outcome_journal)
        sys.stdout.write(canonical_json(result) + "\n")
        return 0
    except Exception as exc:
        code = exc.code if hasattr(exc, "code") else "SHADOW_BRAIN_EVALUATION_FAILED_CLOSED"
        sys.stderr.write(str(code) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
