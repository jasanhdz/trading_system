"""Atomically validate and switch the Entry Quality V2 runtime mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from aegis.research.shadow_runtime import load_entry_quality_v2_config
from aegis.research.shadow_evidence import audit_entry_quality_v2_evidence
from aegis.utils import canonical_json


PYTHON_API_PM2_NAME = "02-Aegis-API"
LIVE_OPPORTUNITY_SOURCE = "PROMOTED_SHORT_OPPORTUNITY_MODEL"


def _mapping(value: Any, identity: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return dict(value)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".mode-switch.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def candidate_config(path: Path, mode: str) -> bytes:
    payload = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "entry_quality_v2",
    )
    payload["mode"] = mode
    if mode == "LIVE":
        opportunity = _mapping(payload.get("opportunity"), "opportunity")
        opportunity["source"] = LIVE_OPPORTUNITY_SOURCE
        payload["opportunity"] = opportunity
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolved(root: Path, value: Any) -> Path:
    return (root / str(value)).resolve()


def _live_authority(
    path: Path,
    root: Path,
    *,
    promotion_path_override: Path | None = None,
) -> tuple[bytes, Path, bytes]:
    payload = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "entry_quality_v2",
    )
    opportunity = _mapping(payload.get("opportunity"), "opportunity")
    promotion = _mapping(payload.get("live_promotion"), "live_promotion")
    readiness_path = _resolved(
        root,
        promotion.get("technical_readiness_record_path"),
    )
    readiness_content = readiness_path.read_bytes()
    if _sha256(readiness_content) != str(
        promotion.get("technical_readiness_record_sha256")
    ):
        raise ValueError("AEGIS_ENTRY_QUALITY_V2_READINESS_AUTHORITY_MISMATCH")
    readiness = _mapping(
        json.loads(readiness_content),
        "technical_readiness_record",
    )
    if (
        readiness.get("state") != "LIVE_READY_NOT_ACTIVE"
        or readiness.get("artifact_sha256")
        != opportunity.get("artifact_sha256")
        or readiness.get("automatic_live_activation") is not False
    ):
        raise ValueError("AEGIS_ENTRY_QUALITY_V2_TECHNICAL_READINESS_INCOMPLETE")

    evidence = _mapping(payload.get("evidence"), "evidence")
    journal_root = (path.parent / str(evidence["journal_root"])).resolve()
    minimum = int(promotion["minimum_non_overlapping_episodes"])
    audit = audit_entry_quality_v2_evidence(
        journal_root / str(evidence["signal_journal"]),
        journal_root / str(evidence["outcome_journal"]),
        minimum_matured_episodes=minimum,
    )
    if audit.get("evidence_state") != "EVIDENCE_READY_FOR_OWNER_REVIEW":
        raise ValueError("AEGIS_ENTRY_QUALITY_V2_SHADOW_EVIDENCE_INCOMPLETE")

    required_source = str(promotion["required_opportunity_source"])
    record = {
        "schema_id": "aegis-entry-quality-v2-live-promotion-v1",
        "state": "OWNER_APPROVED_FOR_LIVE_SWITCH",
        "authorization": "EXPLICIT_MODE_SWITCH_COMMAND",
        "authorized_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "artifact_sha256": opportunity["artifact_sha256"],
        "opportunity_source": required_source,
        "technical_readiness_record_path": str(
            readiness_path.relative_to(root)
        ),
        "technical_readiness_record_sha256": _sha256(readiness_content),
        "shadow_evidence": audit,
        "automatic_activation": False,
        "exchange_mutations": 0,
    }
    record_content = (canonical_json(record) + "\n").encode("utf-8")
    record_path = promotion_path_override or _resolved(
        root,
        promotion["promotion_record_path"],
    )
    promotion["promotion_record_path"] = str(record_path)
    promotion["promotion_record_sha256"] = _sha256(record_content)
    payload["live_promotion"] = promotion
    opportunity["source"] = required_source
    payload["opportunity"] = opportunity
    payload["mode"] = "LIVE"
    candidate = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    return candidate, record_path, record_content


def validate_candidate(path: Path, content: bytes, repo_root: Path) -> None:
    temporary = path.with_name(path.name + ".mode-switch-validation.tmp")
    try:
        temporary.write_bytes(content)
        load_entry_quality_v2_config(temporary, repo_root=repo_root)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("SHADOW", "LIVE"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/entry_quality_v2.yaml"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restart-python-api", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = (root / args.config).resolve()
    before = path.read_bytes()
    promotion_path: Path | None = None
    promotion_before: bytes | None = None
    promotion_created = False
    if args.mode == "LIVE":
        check_path = path.with_name(".entry-quality-v2-live-promotion-check.json")
        candidate, promotion_path, promotion_content = _live_authority(
            path,
            root,
            promotion_path_override=check_path if args.check else None,
        )
        if promotion_path.exists():
            promotion_before = promotion_path.read_bytes()
            if not args.check and promotion_before != promotion_content:
                raise SystemExit(
                    "AEGIS_ENTRY_QUALITY_V2_PROMOTION_RECORD_CONFLICT"
                )
        else:
            _atomic_write(promotion_path, promotion_content)
            promotion_created = True
        try:
            validate_candidate(path, candidate, root)
        finally:
            if args.check and promotion_created:
                promotion_path.unlink(missing_ok=True)
    else:
        candidate = candidate_config(path, args.mode)
        validate_candidate(path, candidate, root)
    if args.check:
        print(f"ENTRY_QUALITY_V2_{args.mode}_SWITCH_VALID")
        return 0
    _atomic_write(path, candidate)
    if not args.restart_python_api:
        print(
            f"ENTRY_QUALITY_V2_MODE_SET_{args.mode}; "
            f"restart {PYTHON_API_PM2_NAME} to apply"
        )
        return 0
    result = subprocess.run(
        ("pm2", "restart", PYTHON_API_PM2_NAME),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"ENTRY_QUALITY_V2_MODE_ACTIVE_{args.mode}")
        return 0
    _atomic_write(path, before)
    if promotion_path is not None:
        if promotion_before is not None:
            _atomic_write(promotion_path, promotion_before)
        elif promotion_created:
            promotion_path.unlink(missing_ok=True)
    subprocess.run(
        ("pm2", "restart", PYTHON_API_PM2_NAME),
        check=False,
        capture_output=True,
        text=True,
    )
    raise SystemExit("AEGIS_ENTRY_QUALITY_V2_MODE_SWITCH_ROLLED_BACK")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
