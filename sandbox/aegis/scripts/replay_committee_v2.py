"""Run the frozen Committee V2 paired replay without network or exchange use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.research.committee_v2_replay import (
    assess_committee_v2_replay,
    load_committee_v2_replay_config,
    load_forward_episodes,
    run_historical_replay,
    write_replay_report,
)


def _progress(current: int, total: int) -> None:
    print(
        json.dumps(
            {
                "historical_cycles_completed": current,
                "historical_cycles_total": total,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_committee_v2_replay_v1.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "historical", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config if args.config.is_absolute() else (root / args.config).resolve()
    )
    config = load_committee_v2_replay_config(
        config_path,
        repo_root=root,
    )
    reports = []
    episode_sets = []
    if args.mode in {"forward", "all"}:
        episodes, provenance = load_forward_episodes(config)
        report = assess_committee_v2_replay(
            episodes,
            config,
            source="FORWARD_JOURNAL_REPLAY",
            provenance=provenance,
        )
        write_replay_report(config.fast_report, report)
        reports.append(report)
        episode_sets.append(episodes)
    if args.mode in {"historical", "all"}:
        episodes, provenance = run_historical_replay(
            config,
            progress=_progress,
        )
        report = assess_committee_v2_replay(
            episodes,
            config,
            source="POST_CUTOFF_CAUSAL_REPLAY",
            provenance=provenance,
        )
        write_replay_report(config.historical_report, report)
        reports.append(report)
        episode_sets.append(episodes)
    if args.mode == "all":
        combined = tuple(row for population in episode_sets for row in population)
        report = assess_committee_v2_replay(
            combined,
            config,
            source="FORWARD_PLUS_POST_CUTOFF_CAUSAL_REPLAY",
            provenance={
                "component_reports": [
                    {
                        "source": item["source"],
                        "verdict": item["verdict"],
                    }
                    for item in reports
                ]
            },
        )
        write_replay_report(config.combined_report, report)
        reports.append(report)
    print(
        json.dumps(
            [
                {
                    "source": report["source"],
                    "global_episodes": report["population"]["global_purged_episodes"],
                    "verdict": report["verdict"],
                }
                for report in reports
            ],
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
