# E4 Execution Realism Report

The causal reference is the next available 1-minute open, not the decision
bar's already-known close. Implementation shortfall from decision close to that
open is recorded per row. Exact subsecond p50/p90 fill prices cannot be
reconstructed from this historical candle panel, so no fabricated millisecond
price is reported.

At the primary top-10% coverage:

- gross expectancy: +0.05 bps;
- net at 14 bps: -13.95 bps;
- realistic next-open net at 14 bps: -13.96 bps;
- net at 20 bps: -19.95 bps;
- block-bootstrap 95% CI for 14-bps net: [-14.35, -12.89] bps.

Execution modeling is therefore not what causes failure. The gross directional
advantage is already approximately zero before costs.
