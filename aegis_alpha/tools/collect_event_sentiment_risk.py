#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.event_risk.news_sentiment import classify_event_sentiment_risk
from aegis_alpha.event_risk.news_sources import collect_news_sources


DEFAULT_MANUAL_CONFIG = "aegis_alpha/config/manual_event_risk.json"
DEFAULT_LATEST_OUTPUT = "aegis_alpha/data/processed/event_risk/latest_event_sentiment_risk.json"
DEFAULT_JSONL_DIR = "aegis_alpha/logs/event_risk"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def collect_event_sentiment_risk(
    *,
    manual_config_path: str = DEFAULT_MANUAL_CONFIG,
    latest_output_path: str = DEFAULT_LATEST_OUTPUT,
    jsonl_dir: str = DEFAULT_JSONL_DIR,
    include_http: bool = True,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    sources = collect_news_sources(
        manual_config_path=manual_config_path,
        include_http=include_http,
        timeout_seconds=timeout_seconds,
    )
    result = classify_event_sentiment_risk(sources)
    result["collector"] = {
        "name": "aegis_news_sentiment_risk_collector",
        "version": "0.1",
        "manual_config_path": manual_config_path,
        "latest_output_path": latest_output_path,
        "jsonl_dir": jsonl_dir,
        "include_http": include_http,
        "timeout_seconds": timeout_seconds,
    }

    _write_json(Path(latest_output_path), result)
    _append_jsonl(Path(jsonl_dir) / f"event_sentiment_risk_{_today()}.jsonl", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect external news/sentiment risk in SHADOW mode.")
    parser.add_argument("--manual-config", default=DEFAULT_MANUAL_CONFIG)
    parser.add_argument("--latest-output", default=DEFAULT_LATEST_OUTPUT)
    parser.add_argument("--jsonl-dir", default=DEFAULT_JSONL_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--no-http", action="store_true", help="Use only manual fallback; do not fetch HTTP/RSS.")
    args = parser.parse_args()

    result = collect_event_sentiment_risk(
        manual_config_path=args.manual_config,
        latest_output_path=args.latest_output,
        jsonl_dir=args.jsonl_dir,
        include_http=not args.no_http,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
