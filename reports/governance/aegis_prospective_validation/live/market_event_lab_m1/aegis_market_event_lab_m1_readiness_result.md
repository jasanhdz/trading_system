# Market Event Laboratory M1 Readiness Result

## Verdict

`M1_READY_FOR_EXPERIMENTS = FALSE`

This is a data-maturity result, not an economic result. No event hypothesis was
backtested, no model was trained, and no threshold was selected.

## Evidence found

The existing public database contains all 11 symbols and approximately 540
days of futures candle microstructure and funding/mark observations. It does
not yet contain the event evidence required by M1:

| Source | Observed coverage | Result |
|---|---:|---|
| Futures candle microstructure | 539.997 days | Available |
| Funding and recorded mark price | 539.667 days | Available |
| Open interest | 1.743 days | Immature |
| Depth | 0.008 days, 11 snapshots | Immature |
| Aggregate trade buckets | 0 | Absent |
| Liquidation events | 0 | Absent |
| Spot reference | 0 | Absent |
| Derived basis | 0 | Absent |

Consequently, every event family is blocked:

- `LIQUIDATION_CASCADE_REVERSAL`: aggregate trades and liquidations are absent;
  depth is immature.
- `OI_CONFIRMED_BREAKOUT`: aggregate trades are absent; OI is immature.
- `SPOT_FUTURES_DISLOCATION`: spot reference and basis are absent.
- `DEPTH_ABSORPTION_REVERSAL`: aggregate trades are absent; depth is immature.

## Infrastructure completed

- Frozen causal feature contracts reject missing, extra, reordered, non-finite,
  future-dated, wrong-schema, wrong-dtype, and wrong-hash observations.
- Event rules receive thresholds fitted elsewhere on TRAIN and contain no
  permissive defaults.
- Prospective SQLite schemas exist for aggregate trades, liquidations, spot,
  basis, and book ticker evidence.
- Economic paths use next-bar open, LONG/SHORT-correct MAE and MFE, fees,
  slippage, funding, and conservative same-bar target/stop resolution.
- Market-wide correlated events are collapsed causally using the first event,
  not the best later outcome.
- Portfolio replay accounts for capital, overlapping symbols, capacity,
  drawdown, and duplicate identities.
- The private trial ledger is append-only, locked, fsynced, permission `0600`,
  and SHA-256 hash chained.

## Integrity

- Preregistration SHA-256:
  `a1e86e6d1703cc445e2afa6a13c37b7f75909821ed27350f6d16a6a83c13acb8`
- Evidence database SHA-256:
  `fd7f5a4af37d09ad6a3b76000ec819843efe94f718922e993ae3d607e904b2f1`
- Readiness report hash:
  `cf5000b4dac7964961b016a9b60a75039f10e3f4f87e73fd86290b31293f4b82`
- Trial ledger record hash:
  `3cc2b8fcf1297c264c486648cd9b2fb40673028f492c65a57d1fc60e663d37bf`

The audit made zero network calls, zero exchange calls, zero exchange
mutations, and no runtime changes.

## Next evidence step

Populate the prospective source tables continuously and immutably. Re-run the
same readiness audit after all required sources reach the frozen 60-day gate.
Do not inspect economic outcomes or adjust thresholds before that gate.
