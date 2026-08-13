# Aegis Economic Alpha Backcast A2 - Result

## Verdict

`A2_FROZEN_A1_CLUES_REFUTED`

A2 applied the unchanged A1 contract to June through December 2023. It used
291,200 causal states and retained 3,335 primary and control outcomes. Neither
of the two frozen hypotheses passed the preregistered economic gate.

## Results

| Hypothesis | Events | Gross mean | Net mean at 14 bps | Profit factor | Mean MAE | A1 net mean | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trend acceptance LONG 60m | 140 | -0.0450% | -0.1850% | 0.783 | 1.562% | +0.4575% | FAIL |
| Carry convergence SHORT 1440m | 326 | +0.0098% | -0.1302% | 0.910 | 3.214% | +0.4076% | FAIL |

Trend LONG was negative before costs. Only four symbols had positive average
net results. Its simple-momentum control returned +0.1666% net while the A1
top rank returned -0.1850%.

Carry SHORT had approximately zero gross expectancy and became negative after
costs. The frozen top rank returned -0.1302% net, while the daily-spaced set of
all eligible candidates returned +0.0440%. Thus the rank score removed value.
Its 95% bootstrap expectancy interval crossed zero widely.

## Meaning

The positive A1 point estimates did not transfer to a different historical
regime. This is evidence against training another ranker on these hypotheses.
More model capacity would be attempting to recover a relationship that is not
stable at the transparent mechanism level.

The result also clarifies the architectural problem: Aegis has often treated
every timestamp as a prediction problem. The evidence instead supports an
abstention-first design in which the system first asks whether an independent
market event has enough gross movement to pay costs. Direction and symbol
ranking should only be modeled after that condition is demonstrated.

## Recommended Successor

Do not tune A1 or A2. A separately preregistered Opportunity Atlas B1 should:

1. cluster correlated symbols and timestamps into independent market events;
2. identify gross opportunity before assigning LONG or SHORT;
3. separate market-wide movement from symbol-specific residual movement;
4. model cost exceedance, MAE, MFE and time-to-positive as separate targets;
5. use abstention as the default action;
6. compare every candidate with no-trade and timestamp-matched random controls;
7. reserve a new chronological forward period for confirmation.

Machine learning becomes appropriate only if B1 finds a stable conditional
opportunity population. It should then rank candidates within that population,
not predict every candle.

## Reproducibility And Safety

- Archive manifest SHA-256: `c59f1d202d32a9e23f35780f59868181fe2e7d6f5f050028ec15e3a7908e71f8`
- Causal panel SHA-256: `2c19ceb09f4ffe8968c733ec1b80d4fa35e9009f4dc8b09e7ca9279873d272f1`
- Candidate table SHA-256: `441ed978fd055e59b0f97259689de3ca7afe94c5274dc053732023d9ffbe4bbc`
- Outcome table SHA-256: `02a8243b100fb7c0ec48aaf13f1e4d397c816886a5242263c4369945f1c8b094`
- Private result SHA-256: `736adf7e86eafc2cc9fa5df903bf3c9cabf919e0452396dd76fc9802a2edca70`
- Repeated execution reproduced all hashes and counts.
- Public archive downloads: 396 checksum-verified files, 414,120,129 bytes.
- Authenticated exchange calls: `0`.
- Exchange mutations: `0`.
- PM2, Live, Shadow and TypeScript changes: `NONE`.
- Promotion authority: `NONE`.
