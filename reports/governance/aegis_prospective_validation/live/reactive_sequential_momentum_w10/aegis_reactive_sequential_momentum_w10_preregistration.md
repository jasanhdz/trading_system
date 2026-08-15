# Aegis W10 Reactive Sequential Momentum Navigation - Preregistration

## Hypothesis

W10 does not predict the initial break. It tests whether movement already
visible in causal price, book and trade-flow state can be joined late enough to
confirm persistence but early enough to retain positive value after all costs.

## Population

- Source: the 30 validated Tardis symbol-days used by W9.1.
- Symbols: ADA, BNB, BTC, ETH, SOL and XRP perpetual futures.
- Unit: fixed non-overlapping 120-second `momentum_episode_id` every five
  minutes, with a decision every five seconds.
- TRAIN: 2024-09-01, 2025-03-01 and 2025-09-01.
- VALIDATION: 2025-12-01 and 2026-03-01.
- FINAL_HOLDOUT_W10: future independent evidence, `SEALED_NOT_OPENED`.

## Frozen Policy Family

- Direction model target: first +/-20 bps barrier over the next 60 seconds.
- Models: regularized multinomial logistic, depth-3 tree and constrained
  histogram gradient boosting.
- ENTER requires 2-3 consecutive high-confidence observations.
- HOLD uses a lower threshold than ENTER (hysteresis).
- EXIT requires 2-3 consecutive deterioration observations, except immediate
  strong adverse invalidation.
- Every exit enters a 20-40 second cooldown.
- At most two entries per episode; direct LONG/SHORT flips are structurally
  prohibited.
- All model and policy selection uses TRAIN only.

## Economics And Gates

- Base round-trip cost: 14 bps; stress: 20 bps.
- Latency: 0/100/250/500/1000 ms; 250 ms must remain positive.
- Required validation: at least +2 bps/trade and +0.5 bps/episode, positive
  bootstrap lower bound, profit factor >=1.10, four positive symbols, both
  validation dates positive and every simple baseline beaten.
- Anti-churn: <=3 trades/hour, <=0.15 reentries/episode, zero direct flips,
  median hold >=15 seconds and costs <=65% of positive gross PnL.
- Bootstrap: 10,000 symbol-day blocks.

W7 Opportunity remains frozen and is only a diagnostic if existing overlap is
sufficient. No W9.1 retuning, production, TypeScript, brain, guard, leverage,
PM2, Shadow, Live, authenticated request or exchange mutation is allowed.
