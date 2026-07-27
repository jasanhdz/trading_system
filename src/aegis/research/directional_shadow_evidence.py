"""Evaluate directional Shadow journals without granting exchange authority."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..config import CANONICAL_SYMBOLS
from ..utils import canonical_json, sha256_file
from .directional_challenger import (
    DirectionalEvidenceRow,
    DirectionalSelectionMetrics,
    selection_metrics,
)


class DirectionalShadowEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class DirectionalShadowEvidenceConfig:
    signal_journal: Path
    outcome_journal: Path
    report_path: Path
    side: str
    offline_validation_state: str
    minimum_selected_outcomes: int
    minimum_independent_blocks: int
    maximum_symbol_concentration: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_block_minutes: int

    def __post_init__(self) -> None:
        if self.side != "LONG":
            raise DirectionalShadowEvidenceError(
                "AEGIS_DIRECTIONAL_SHADOW_SIDE_INVALID"
            )
        if min(
            self.minimum_selected_outcomes,
            self.minimum_independent_blocks,
            self.bootstrap_resamples,
            self.bootstrap_block_minutes,
        ) <= 0:
            raise DirectionalShadowEvidenceError(
                "AEGIS_DIRECTIONAL_SHADOW_EVIDENCE_INVALID"
            )
        if not 0.0 < self.maximum_symbol_concentration <= 1.0:
            raise DirectionalShadowEvidenceError(
                "AEGIS_DIRECTIONAL_SHADOW_CONCENTRATION_INVALID"
            )


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectionalShadowEvidenceError(f"{identity} must be a mapping")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DirectionalShadowEvidenceError(
            "AEGIS_DIRECTIONAL_SHADOW_TIMESTAMP_INVALID"
        )
    return parsed.astimezone(timezone.utc)


def _rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(
                    _mapping(json.loads(line), f"{path}:{line_number}")
                )
            except json.JSONDecodeError as exc:
                raise DirectionalShadowEvidenceError(
                    "AEGIS_DIRECTIONAL_SHADOW_JSONL_INVALID"
                ) from exc
    return tuple(rows)


def load_directional_shadow_evidence_config(
    path: Path,
    *,
    repo_root: Path,
) -> DirectionalShadowEvidenceConfig:
    payload = _mapping(yaml.safe_load(path.read_text()), "dual shadow")
    if (
        payload.get("schema_version")
        != "aegis-entry-quality-v3-dual-shadow-runtime-v1"
        or payload.get("mode") != "SHADOW"
    ):
        raise DirectionalShadowEvidenceError(
            "AEGIS_DIRECTIONAL_SHADOW_CONFIG_INVALID"
        )
    evidence = _mapping(payload["evidence"], "evidence")
    evaluation = _mapping(payload["evaluation"], "evaluation")
    artifact = _mapping(payload["artifact"], "artifact")
    journal_root = _resolve(path.parent, evidence["journal_root"])
    return DirectionalShadowEvidenceConfig(
        signal_journal=journal_root / str(evidence["signal_journal"]),
        outcome_journal=journal_root / str(evidence["outcome_journal"]),
        report_path=_resolve(repo_root, evaluation["report_path"]),
        side=str(payload["side"]),
        offline_validation_state=str(artifact["offline_validation_state"]),
        minimum_selected_outcomes=int(
            evaluation["minimum_selected_outcomes"]
        ),
        minimum_independent_blocks=int(
            evaluation["minimum_independent_blocks"]
        ),
        maximum_symbol_concentration=float(
            evaluation["maximum_symbol_concentration"]
        ),
        bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        bootstrap_block_minutes=int(
            evaluation["bootstrap_block_minutes"]
        ),
    )


def _variant_metrics(
    rows: Sequence[DirectionalEvidenceRow],
    selected: Sequence[int],
    config: DirectionalShadowEvidenceConfig,
) -> Mapping[str, Any]:
    metrics: DirectionalSelectionMetrics = selection_metrics(
        rows,
        selected,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_block_minutes=config.bootstrap_block_minutes,
    )
    evidence_passed = (
        metrics.signals >= config.minimum_selected_outcomes
        and metrics.independent_blocks >= config.minimum_independent_blocks
        and metrics.block_mean_net_expectancy > 0.0
        and metrics.expectancy_ci95_low is not None
        and metrics.expectancy_ci95_low > 0.0
        and metrics.symbol_concentration
        <= config.maximum_symbol_concentration
    )
    return {
        **asdict(metrics),
        "evidence_passed": evidence_passed,
        "requirements": {
            "minimum_selected_outcomes": config.minimum_selected_outcomes,
            "minimum_independent_blocks": (
                config.minimum_independent_blocks
            ),
            "maximum_symbol_concentration": (
                config.maximum_symbol_concentration
            ),
            "expectancy_ci95_low_must_exceed_zero": True,
        },
    }


def _per_symbol(
    rows: Sequence[DirectionalEvidenceRow],
    selected: Sequence[int],
) -> Mapping[str, Any]:
    chosen = set(selected)
    result = {}
    for symbol in CANONICAL_SYMBOLS:
        population = [row for row in rows if row.symbol == symbol]
        selected_rows = [
            row
            for index, row in enumerate(rows)
            if index in chosen and row.symbol == symbol
        ]
        result[symbol] = {
            "outcomes": len(population),
            "selected_outcomes": len(selected_rows),
            "mean_population_net_return": (
                sum(row.net_return for row in population) / len(population)
                if population
                else None
            ),
            "mean_selected_net_return": (
                sum(row.net_return for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else None
            ),
            "mean_selected_mae": (
                sum(row.mae for row in selected_rows) / len(selected_rows)
                if selected_rows
                else None
            ),
        }
    return result


def build_directional_shadow_evidence(
    config: DirectionalShadowEvidenceConfig,
) -> Mapping[str, Any]:
    if not config.signal_journal.is_file() or not config.outcome_journal.is_file():
        raise DirectionalShadowEvidenceError(
            "AEGIS_DIRECTIONAL_SHADOW_INPUT_MISSING"
        )
    signals = {
        str(row["event_id"]): row for row in _rows(config.signal_journal)
    }
    outcomes = _rows(config.outcome_journal)
    evidence_rows = []
    model_selected = []
    regime_selected = []
    seen_symbols = set()
    for outcome in outcomes:
        event_id = str(outcome["event_id"])
        signal = signals.get(event_id)
        if signal is None:
            raise DirectionalShadowEvidenceError(
                "AEGIS_DIRECTIONAL_SHADOW_SIGNAL_MISSING"
            )
        symbol = str(outcome["symbol"])
        seen_symbols.add(symbol)
        shadow = _mapping(signal["long_shadow"], "long_shadow")
        regime = _mapping(shadow["regime"], "regime")
        score = float(shadow["score"])
        net = float(outcome["net_return_fraction"])
        mae = float(outcome["mae_fraction"])
        if not all(math.isfinite(value) for value in (score, net, mae)):
            raise DirectionalShadowEvidenceError(
                "AEGIS_DIRECTIONAL_SHADOW_NONFINITE"
            )
        evidence_rows.append(
            DirectionalEvidenceRow(
                timestamp=_timestamp(outcome["signal_timestamp"]),
                symbol=symbol,
                score=score,
                net_return=net,
                mae=mae,
                bad_entry=net <= 0.0,
                regime=(
                    f"{regime['direction']}|{regime['volatility']}|"
                    f"{regime['structure']}"
                ),
            )
        )
        index = len(evidence_rows) - 1
        if bool(outcome["model_only_selected"]):
            model_selected.append(index)
        if bool(outcome["regime_confirmed_selected"]):
            regime_selected.append(index)
    if seen_symbols != set(CANONICAL_SYMBOLS):
        raise DirectionalShadowEvidenceError(
            "AEGIS_DIRECTIONAL_SHADOW_SYMBOL_POPULATION_INVALID"
        )

    model_metrics = _variant_metrics(
        evidence_rows, model_selected, config
    )
    regime_metrics = _variant_metrics(
        evidence_rows, regime_selected, config
    )
    offline_ready = config.offline_validation_state == "PASSED"
    shadow_ready = bool(regime_metrics["evidence_passed"])
    report = {
        "schema_id": "aegis-directional-shadow-evidence-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "side": config.side,
        "runtime_authority": "SHADOW_ONLY",
        "outcomes": len(evidence_rows),
        "symbols": sorted(seen_symbols),
        "loss_label": "NET_RETURN_AT_HORIZON_NOT_POSITIVE",
        "model_only": {
            **model_metrics,
            "per_symbol": _per_symbol(evidence_rows, model_selected),
        },
        "regime_confirmed": {
            **regime_metrics,
            "per_symbol": _per_symbol(evidence_rows, regime_selected),
        },
        "regime_uplift": {
            "block_mean_net_expectancy_delta": (
                float(regime_metrics["block_mean_net_expectancy"])
                - float(model_metrics["block_mean_net_expectancy"])
            ),
            "mean_mae_delta": (
                float(regime_metrics["mean_mae"])
                - float(model_metrics["mean_mae"])
            ),
        },
        "readiness": {
            "state": (
                "READY_FOR_OWNER_REVIEW"
                if offline_ready and shadow_ready
                else (
                    "OFFLINE_VALIDATION_FAILED"
                    if not offline_ready
                    else "COLLECTING_INDEPENDENT_SHADOW_EVIDENCE"
                )
            ),
            "offline_validation_passed": offline_ready,
            "shadow_evidence_passed": shadow_ready,
            "automatic_training": False,
            "automatic_promotion": False,
            "owner_promotion_required": True,
        },
        "exchange_mutations": 0,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=config.report_path.parent,
        delete=False,
    ) as handle:
        handle.write(canonical_json(report) + "\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    os.replace(temporary, config.report_path)
    report["report_sha256"] = sha256_file(config.report_path)
    return report

