# AEGIS W11 Ephemeral Regime Alpha

Offline, disposable research sandbox for testing whether a model trained only on the
most recent market state can produce short-lived net economic edge.

## Safety Boundary

- Reads local historical public market data only.
- Does not import or modify production, TypeScript, E4, PM2, exchange, live config,
  canonical datasets, existing shadow services, or sealed holdouts.
- Writes only below this sandbox.
- An expired model instance is immutable and can never reactivate.

## Frozen Study

- Source: 11-symbol Binance futures 1-minute candles from May-December 2023.
- Causal bars: completed 5-minute bars; decisions every 15 minutes.
- Recent training windows: 6h, 12h, 24h, 48h, 72h.
- Forward targets: 5m, 15m, 30m, 60m.
- Costs: 14 bps baseline, 20 bps stress, 30 bps severe stress.
- Final prospective interval: `[2023-11-01, 2024-01-01)`.

See `w11_data_audit.md` and `w11_preregistration.md` before interpreting results.

## Reproduction

From the repository root:

```bash
PYTHONPATH=sandbox/aegis_ephemeral_regime_w11/src \
  .venv/bin/python -m aegis_ephemeral_regime_w11.cli audit

PYTHONPATH=sandbox/aegis_ephemeral_regime_w11/src \
  .venv/bin/python -m aegis_ephemeral_regime_w11.cli run

PYTHONPATH=sandbox/aegis_ephemeral_regime_w11/src \
  .venv/bin/pytest sandbox/aegis_ephemeral_regime_w11/tests
```

Generated machine-readable outputs live in `artifacts/`. The final human report is
`w11_ephemeral_regime_result.md`; the verdict is
`w11_ephemeral_regime_verdict.json`.
