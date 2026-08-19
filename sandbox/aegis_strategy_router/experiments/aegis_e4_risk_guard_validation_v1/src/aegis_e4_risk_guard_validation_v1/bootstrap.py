"""Block bootstrap confidence intervals for E4 Risk Guard Validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def block_bootstrap_policy_delta(
    df: pd.DataFrame,
    allowed_mask: pd.Series,
    cost_bps: float = 14.0,
    n_samples: int = 10000,
    seed: int = 20260819,
) -> dict[str, float]:
    working = df.copy()
    working["net_bps"] = working["gross_bps"] - cost_bps
    working["allowed"] = allowed_mask.to_numpy(bool)
    working["net_signal_guard"] = np.where(working["allowed"], working["net_bps"], 0.0)
    working["net_signal_baseline"] = working["net_bps"]
    working["delta_per_signal"] = working["net_signal_guard"] - working["net_signal_baseline"]

    # Group by temporal block (e.g. date)
    working["date_block"] = pd.to_datetime(working["signal_timestamp"], utc=True).dt.date.astype(str)
    blocks = working.groupby("date_block", sort=False).agg({
        "net_signal_guard": "mean",
        "net_signal_baseline": "mean",
        "delta_per_signal": "mean",
    })

    unique_blocks = blocks.to_numpy()
    n_blocks = len(unique_blocks)
    rng = np.random.default_rng(seed)

    boot_deltas = np.empty(n_samples)
    boot_guard_nets = np.empty(n_samples)
    boot_base_nets = np.empty(n_samples)

    batch_size = 500
    for start in range(0, n_samples, batch_size):
        cnt = min(batch_size, n_samples - start)
        idx = rng.integers(0, n_blocks, size=(cnt, n_blocks))
        sampled_means = unique_blocks[idx].mean(axis=1)
        boot_guard_nets[start:start + cnt] = sampled_means[:, 0]
        boot_base_nets[start:start + cnt] = sampled_means[:, 1]
        boot_deltas[start:start + cnt] = sampled_means[:, 2]

    return {
        "n_blocks": n_blocks,
        "n_samples": n_samples,
        "mean_delta_bps": float(np.mean(boot_deltas)),
        "delta_ci_low_95": float(np.percentile(boot_deltas, 2.5)),
        "delta_ci_high_95": float(np.percentile(boot_deltas, 97.5)),
        "guard_net_mean_bps": float(np.mean(boot_guard_nets)),
        "guard_net_ci_low_95": float(np.percentile(boot_guard_nets, 2.5)),
        "guard_net_ci_high_95": float(np.percentile(boot_guard_nets, 97.5)),
        "baseline_net_mean_bps": float(np.mean(boot_base_nets)),
        "baseline_net_ci_low_95": float(np.percentile(boot_base_nets, 2.5)),
        "baseline_net_ci_high_95": float(np.percentile(boot_base_nets, 97.5)),
    }
