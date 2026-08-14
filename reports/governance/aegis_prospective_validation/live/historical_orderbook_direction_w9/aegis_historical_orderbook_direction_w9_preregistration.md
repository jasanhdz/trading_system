# Aegis W9 Historical Order Book Direction - Preregistration

- Mode: `OFFLINE_HISTORICAL_RESEARCH_ONLY`.
- Config SHA-256: `df8562129100dabf2a8097e1b27a536110ccfd503a6131b56f02693c3ad0af3c`.
- W1-W8 holdouts: sealed and prohibited.
- Frozen W7 Opportunity: `H60:LOGISTIC_L2:P_GTE_0.7`; refit prohibited.
- New information family: sequential L2, quotes, aggressive trades and price response.
- Unit: independent `opportunity_episode_id`, never individual L2 updates.
- TRAIN: first-day samples from September through December 2025.
- VALIDATION: first-day samples from January through April 2026.
- FINAL_HOLDOUT_W9: future independent evidence, `SEALED_NOT_OPENED`.
- Targets: preregistered first-barrier families at 25/50/75 bps and 1 ATR.
- Observation latency: 0/100/250/500/1000 ms.
- Costs: 14 bps baseline, 20 bps stress.
- Minimum economic effect: +3 bps net expectancy after baseline costs.
- Data gate: at least 1,000 TRAIN and 500 VALIDATION episodes, eight symbols
  and three months per partition.
- Gate failure action: stop before modeling; do not interpret missing evidence
  as evidence of no edge.
- Production, TypeScript, Aegis Brain, guards, leverage, PM2, Shadow, Live,
  authenticated requests and orders: prohibited.

The exact machine-readable preregistration is
`config/experiments/aegis_historical_orderbook_direction_w9.yaml`.
