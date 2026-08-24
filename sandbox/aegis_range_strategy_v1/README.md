# Aegis Range Strategy V1 - R1

Pure Python implementation of the approved R0 contract. This package contains no
market-data loader, backtester, optimizer, runtime integration or E4 model call.

## Modules

- `data_adapter.py`: strict synthetic/input 1m validation and causal 5m bars.
- `regime.py` and `atr.py`: 160-bar read-only regime boundary and raw Wilder ATR.
- `levels.py` and `detector.py`: causal levels, ranked pairs and range episodes.
- `safety.py` and `signal.py`: hard blockers, descriptive score and rejection.
- `breakout.py`: episode and frozen-trade close counters.
- `lifecycle.py`: pending entry, immutable thesis, quotas, exits and cooldown.
- `engine.py`: ordered per-bar state machine and deterministic audit outputs.
- `numeric.py` and `thesis.py`: IDs and canonical HALF_EVEN thesis hashing.

Run only the synthetic R1 tests:

```bash
python -m pytest
```

R2 and every economic-data phase remain blocked.
