# Aegis Market-Event Fast Track M1A - Full Retrospective Result

## Verdict

`M1A_RETROSPECTIVE_EDGE_NOT_DEMONSTRATED`

- `M1A_READY_FOR_FORWARD_SHADOW=false`
- `M1A_READY_FOR_LIVE=false`
- Validation gates passed: `0`
- Runtime, Shadow and Live changes: `NONE`
- Exchange requests and mutations: `0`

The frozen first-pass micro-pattern rules do not demonstrate positive net
economic expectancy. They must not be promoted or used to alter the running
trading system.

## Reproducible Scope

- Code commit: `9a0611fc0bed02724de618aa67bb7982715791bc`
- Universe: the 11 canonical symbols
- Source: checksum-verified Binance public Spot and USD-M Futures 1-minute
  klines
- Period: 2024-01 through 2026-07
- Source archives: 682
- Source minutes per symbol: 1,357,920
- Entry: next 1-minute bar open after causal event confirmation
- Horizons: 15, 30, 60, 120 and 240 minutes
- Costs: frozen fees, slippage and horizon funding from the M1A protocol
- Raw candidates: 949,270
- Independent candidates: 169,949
- Evaluated paths across horizons: 849,702
- Regime-matched random paths: 849,661

The evaluated families were trend pullback continuation, compression breakout,
liquidity sweep rejection, flow-price absorption reversal, exhaustion
reversal, Spot/Futures divergence convergence and multi-timeframe reclaim.
`SESSION_FUNDING_DISLOCATION` was not evaluated because the verified funding
history was not part of the completed archive set. It has no inferred result.

## Economic Result

Every pattern/direction/horizon group in retrospective validation had negative
net expectancy. No group met the complete preregistered gate.

The closest validation result was
`SPOT_FUTURES_DIVERGENCE_CONVERGENCE:LONG` at 240 minutes:

- events: 999
- net expectancy: -0.06593%
- profit factor: 0.9021
- day-block bootstrap 95% lower expectancy: -0.24383%
- matched-random expectancy: -0.11417%

It outperformed its matched random control but still lost after costs, had a
profit factor below one and had an uncertainty interval that included material
loss. That is not an operable edge.

At aggregate validation level, LONG gross returns ranged from approximately
0.0042% to 0.0266% across horizons, while net returns ranged from -0.1383% to
-0.1534%. SHORT aggregate gross returns were negative at every validation
horizon. The rules therefore did not generate enough gross information to pay
the frozen execution costs.

The pseudo-holdout also had zero passing groups. It cannot authorize promotion
in any case because prior experiments indirectly observed that period.

## Pilot Reconciliation

The earlier ADA-only July-2026 pilot found one positive
`TREND_PULLBACK_CONTINUATION:LONG` slice. The complete eleven-symbol,
multi-period run did not reproduce it. The pilot was a small-sample observation,
not a robust edge.

Only one discovery group was positive:
`SPOT_FUTURES_DIVERGENCE_CONVERGENCE:LONG` at 240 minutes, with net expectancy
0.03049%. Its bootstrap lower bound was -0.13588%, and validation expectancy
was -0.06593%. This is consistent with discovery noise or temporal instability,
not promotion evidence.

## Interpretation

The experiment answered the short-term question without waiting 60 days:
simple causal candle, flow, basis and regime patterns can be evaluated quickly
from verified public history, but this frozen M1A formulation does not provide
a tradable advantage. Adding a classifier on top of these labels now would
risk learning noise rather than repairing a demonstrated base edge.

No threshold, pattern definition, symbol-specific exception or horizon was
changed after inspecting validation or pseudo-holdout results.

## Next Research Boundary

Further work must be a separately preregistered experiment, not a tuning pass
over M1A. A defensible next track should address the evidence gaps directly:

1. complete causal funding, mark-price and liquidation/context archives;
2. strengthen regime representation with market breadth, relative BTC state,
   volatility and liquidity interactions;
3. test event families with an economic mechanism distinct from these failed
   first-pass rules;
4. retain untouched temporal validation and a genuinely fresh forward gate;
5. compare against no-trade and matched controls after identical costs.

News or whale data may be considered only after timestamp provenance, revision
policy and entity attribution are demonstrably causal. They must not be added
as retrospective explanations of losses.

## Validation

- Focused M1A tests: 16 passed
- Python unit regression: 757 passed, 5 failed
- The five failures are pre-existing branch-authority assertions that require
  the literal branch `feature/aegis-ts-clean-rebuild`; this experiment runs on
  `work/entry-quality-evidence-20260726`. They do not exercise M1A code.
- Python compilation: passed
- Git whitespace validation: passed
- `black`: unavailable for validation because the installed process hung even
  with one worker; no broad formatting rewrite was performed
- TypeScript source and runtime: unchanged

## Private Evidence

- Report: `data/market_event_fast_track_m1a/full_run_01/full_report.json`
- Report SHA-256:
  `374838985a6c0abfd37b8c05bfac763d2735eda31374ce14d7dbae68d251c7a8`
- Raw candidates SHA-256:
  `ef35ffc3f1f10abbded34d0c27d9a69e7278b0b6be937c13a76b420c5fd7db84`
- Independent candidates SHA-256:
  `99790b16f5322207e4f9ba29024779d68d92def9dfb03f6ab657dc6e46219c2e`
- Evaluated events SHA-256:
  `c53fe630021185a88c850c9aae263e3d537095905e4e737405cb35beef6a2a85`
- Matched controls SHA-256:
  `ee22f92a1e9b62eab951e5f07c98ef75efbc5f0f48984e17f0f8a2d0baca9680`

Private evidence permissions were validated as `0600`.
