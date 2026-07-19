#!/usr/bin/env python3
"""CLI for the isolated EXIT_EXCURSION_D1A diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnostics.exit_excursion_d1a.experiment import (
    BaselineReproductionError,
    DataLimitationError,
    EntryDriftError,
    NondeterministicError,
    execute_attempt,
    finalize_attempts,
    write_run_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, default=Path("config/diagnostics/exit_excursion_d1a.json"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/experiments/exit_excursion_d1a"))
    parser.add_argument("--attempt", type=int, choices=(1, 2))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        repository = args.repository.resolve()
        preregistration = (repository / args.preregistration).resolve() if not args.preregistration.is_absolute() else args.preregistration
        output = (repository / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
        if args.finalize == (args.attempt is not None):
            parser.error("choose exactly one of --attempt or --finalize")
        if args.finalize:
            result = finalize_attempts(repository, preregistration, output)
        else:
            attempt_root = output / f"attempt_{args.attempt}"
            result = execute_attempt(repository, preregistration, attempt_root)
            write_run_manifest(
                output / f"run_manifest_attempt_{args.attempt}.json", attempt=int(args.attempt),
                repository=repository, preregistration_path=preregistration, result=result,
            )
    except EntryDriftError as error:
        print(json.dumps({"verdict": "EXIT_D1A_BLOCKED_BY_ENTRY_DRIFT", "error": str(error)}))
        return 3
    except BaselineReproductionError as error:
        print(json.dumps({"verdict": "EXIT_D1A_BLOCKED_BY_BASELINE_REPRODUCTION_FAILURE", "error": str(error)}))
        return 4
    except DataLimitationError as error:
        print(json.dumps({"verdict": "EXIT_D1A_BLOCKED_BY_DATA_LIMITATION", "error": str(error)}))
        return 5
    except NondeterministicError as error:
        print(json.dumps({"verdict": "EXIT_D1A_NON_DETERMINISTIC", "error": str(error)}))
        return 6
    print(json.dumps({"verdict": "EXIT_D1A_COMPLETE", "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
