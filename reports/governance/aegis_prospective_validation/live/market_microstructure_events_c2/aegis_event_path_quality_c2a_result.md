# Aegis Event Path Quality C2A - Result

## Verdict

`C2A_NO_INCREMENTAL_NET_EDGE_VALIDATION`

C2A replaced the proposed 60-day waiting period with checksum-verified public
history, but the two preregistered taker-flow detectors did not demonstrate
positive net economic utility. The final holdout remains sealed. No detector,
threshold or horizon was changed after observing outcomes.

## Data And Method

- Venue: Binance USD-M perpetual futures.
- Universe: the canonical 11 symbols, LONG and SHORT independently.
- Source period: August 2025 through July 2026.
- Sources: 132 monthly aggregate-trade archives and 132 monthly 1-minute kline
  archives, each identified by SHA-256.
- Causal rows: 11,485,500. Entry is the next complete 1-minute open.
- Partitions: TRAIN 2025-08-01 through 2026-01-31; VALIDATION 2026-02-01
  through 2026-04-30; FINAL HOLDOUT 2026-05-01 through 2026-07-31.
- Holdout state: `SEALED`.
- Registered detectors: flow impulse continuation and flow absorption reversal.
- Horizons: 15, 60 and 240 minutes.
- Symmetric barriers: 20, 42 and 80 bps, adverse-first on same-bar ambiguity.
- Primary round-trip economic cost: 14 bps.
- Event cooldown: 15 minutes.

The dataset manifest SHA-256 is
`10eabecfa7aaac3eac0d663b73967e41b04d3661c56c4dd6d546bf16f19a5bd3`.
The private TRAIN/VALIDATION evaluation SHA-256 is
`68f1c8d81e159e93b0a33dffb17f05fc35225657ac708991eaa0e5c7be47a3ef`.

## Best Result Per Detector And Side

Values are the best net expectancy found among the nine preregistered
horizon/barrier contracts. They are reported in basis points per event.

| Partition | Detector | Side | Contract | Events | Net expectancy | 95% CI | Profit factor | Matched control |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | Absorption reversal | LONG | H60/F80/A80 | 359 | -9.78 | [-18.62, -1.20] | 0.74 | -15.87 |
| TRAIN | Absorption reversal | SHORT | H240/F80/A80 | 346 | -9.37 | [-19.54, 0.31] | 0.78 | -6.48 |
| TRAIN | Impulse continuation | LONG | H240/F80/A80 | 10,288 | -13.68 | [-16.62, -10.72] | 0.68 | -16.33 |
| TRAIN | Impulse continuation | SHORT | H240/F80/A80 | 13,792 | -13.80 | [-16.51, -10.74] | 0.68 | -12.95 |
| VALIDATION | Absorption reversal | LONG | H15/F80/A80 | 130 | -7.74 | [-18.41, 2.10] | 0.70 | -18.46 |
| VALIDATION | Absorption reversal | SHORT | H60/F80/A80 | 143 | -10.78 | [-23.83, 0.33] | 0.71 | -16.57 |
| VALIDATION | Impulse continuation | LONG | H15/F42/A42 | 5,375 | -13.77 | [-15.14, -12.43] | 0.39 | -14.67 |
| VALIDATION | Impulse continuation | SHORT | H240/F80/A80 | 6,096 | -13.21 | [-17.86, -8.84] | 0.69 | -12.10 |

No VALIDATION contract had positive net expectancy. The best gross diagnostic
was absorption-reversal LONG at approximately +6.26 bps before the registered
14 bps cost, but it became -7.74 bps net, had only 130 events and its confidence
interval crossed zero. This is insufficient evidence, not a promotable edge.

## Interpretation

Aggressive taker flow by itself is not a profitable entry rule in this sample.
Frequent impulse continuation is especially weak after costs. Absorption may
contain a small gross signal, but its magnitude is below realistic execution
cost and its sample is too uncertain. Searching the sealed holdout or tuning
the registered thresholds now would convert evaluation data into training data
and overstate the evidence.

The next justified experiment is a new preregistered C2B hypothesis that uses
absorption only as one input, not as an entry rule. It should test causally
available regime, volatility, spread/depth and liquidation/open-interest
context, train only on TRAIN, select once on VALIDATION and preserve a fresh
holdout. It must compare against price-only, C1 and matched controls and remain
research-only unless net utility transfers.

## Safety

- TypeScript, Live, Shadow and PM2 changes: `NONE`.
- Credentials loaded: `NO`.
- Authenticated exchange requests: `0`.
- Exchange mutations: `0`.
- C2A ready for modeling: `FALSE`.
- C2A ready for Shadow: `FALSE`.
- C2A ready for Live: `FALSE`.

