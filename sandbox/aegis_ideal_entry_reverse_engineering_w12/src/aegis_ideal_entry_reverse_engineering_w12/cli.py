"""W12 reproducible audit and run commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import SANDBOX, load_config, verify_source
from .experiment import run_experiment
from .performance import write_compute_reports
from .reporting import write_results


def repository_root() -> Path:
    for parent in SANDBOX.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("W12 repository root not found")


def audit() -> dict:
    config = load_config()
    repository = repository_root()
    authority = verify_source(config, repository)
    environment, benchmark = write_compute_reports()
    payload = {
        "status": "PASS", "source_manifest_sha256": authority["manifest_sha256"],
        "symbols": authority["symbols"], "compute_backend": benchmark["selected"],
        "workers": environment["parallelism"]["workers"],
        "external_holdouts_accessed": False, "model_results_produced": False,
    }
    path = SANDBOX / "artifacts" / "data_audit_runtime.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aegis-ideal-entry-reverse-engineering-w12")
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be between 1 and the frozen maximum 4")
    if args.command == "audit":
        audit()
        return 0
    config = load_config()
    result = run_experiment(config, repository_root(), workers=args.workers)
    write_results(result, config, repository_root())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
