from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from .readiness import SYMBOLS
from .sweep_reclaim_discovery import load_regime_cache, replay_structure, structural_candidates
from .train_backtest import load_train_candles, load_train_funding


def replay_symbol_counts(
    repo_root: Path,
    run_a: Path,
    symbol: str,
    *,
    progress: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    candles = load_train_candles(repo_root, symbol)
    funding = load_train_funding(repo_root, symbol)
    snapshots = load_regime_cache(run_a / "regime_cache" / f"{symbol}.csv.gz", len(candles))
    totals = {"opportunities": 0, "entries": 0, "paths": 0, "passages": 0}

    for index, candidate in enumerate(structural_candidates()):
        structure_started = time.monotonic()
        opportunities, entries, paths, passages = replay_structure(symbol, candidate, candles, snapshots, funding)
        counts = (len(opportunities), len(entries), len(paths), len(passages))
        for key, count in zip(totals, counts):
            totals[key] += count
        if progress:
            print(
                f"pid={os.getpid()} {symbol} struct{index}: "
                f"{counts[0]} opps {counts[1]} ents {counts[2]} paths {counts[3]} passages "
                f"in {time.monotonic() - structure_started:.1f}s",
                flush=True,
            )
        del opportunities, entries, paths, passages

    return {
        "symbol": symbol,
        **totals,
        "elapsed_seconds": time.monotonic() - started,
    }


def _replay_symbol_task(task: tuple[str, str, str, bool]) -> dict[str, Any]:
    repo_root, run_a, symbol, progress = task
    return replay_symbol_counts(Path(repo_root), Path(run_a), symbol, progress=progress)


def parallel_replay_counts(
    repo_root: Path,
    run_a: Path,
    *,
    workers: int,
    symbols: Sequence[str] = SYMBOLS,
    progress: bool = False,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    unknown = set(symbols) - set(SYMBOLS)
    if unknown:
        raise ValueError(f"symbols outside frozen universe: {sorted(unknown)}")

    tasks = [(str(repo_root), str(run_a), symbol, progress) for symbol in symbols]
    if workers == 1:
        return [_replay_symbol_task(task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        futures = {executor.submit(_replay_symbol_task, task): task[2] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress:
                print(
                    f"completed {result['symbol']} in {result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    order = {symbol: index for index, symbol in enumerate(symbols)}
    return sorted(results, key=lambda item: order[item["symbol"]])
