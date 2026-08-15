# Aegis W11 Entry Safety Gate - Preregistration

## Frozen question

Given an immutable historical Aegis `LONG` or `SHORT` signal, can an isolated
entry-safety policy choose `ENTER_NOW`, `WAIT_1M`, `WAIT_2M`, `WAIT_3M`, or
`SKIP` and improve net expectancy without changing direction?

## Splits

- TRAIN: signal timestamp before 2026-07-01 UTC.
- VALIDATION: 2026-07-01 through 2026-07-31 UTC.
- FINAL_HOLDOUT: 2026-08-01 onward, initially `SEALED_NOT_OPENED`.

No W1-W10 holdout is opened or reused. W11 holdout remains sealed unless every
TRAIN/VALIDATION gate passes.

## Frozen counterfactual

- Delays: 0, 1, 2, and 3 complete minutes.
- Primary economic horizon: 60 minutes from each candidate entry.
- Primary utility: side-oriented fixed-horizon underlying return minus 14 bps.
- Stress cost: 20 bps.
- Diagnostic first barriers: favorable 30 bps versus adverse 20 bps.
- Same-bar ambiguity is resolved adverse-first.
- Recorded Live fill is used for `ENTER_NOW`; delayed entries use the first
  1m open at or after the decision boundary.
- Future MFE, MAE, barriers, and returns are labels only, never features.

## Frozen model families

Four regularized logistic risk detectors use only their named causal feature
families: exhaustion, opposition, space, and volatility. A regularized linear
timing model estimates the 60m net utility of each candidate delay. The combined
policy may wait only when its frozen predicted utility supports doing so and may
skip using deterministic reason priority.

The ablation family is limited to ENTER_NOW, each specialized detector alone,
confirmation timing alone, and the combined gate. Thresholds are selected on
TRAIN only and frozen before VALIDATION.

## Primary gate

W11 must have positive net bps per original signal, improve ENTER_NOW by at
least 2 bps per signal, retain at least 50% of historically clean entries, avoid
at least 20% of historically bad entries, execute at least 35 validation signals
across at least four symbols, remain positive at 20 bps cost, and have a 10,000
episode-bootstrap 95% lower bound above zero for improvement.

Classification counts are diagnostic. The promotion criterion is economic and
includes skipped opportunities as zero, so stopping nearly all trading cannot
manufacture a pass.

## Restrictions

Python offline research only. No production, TypeScript, Aegis Brain, guards,
leverage, PM2, Shadow, authenticated API, orders, or financial state changes.
`W11_READY_FOR_LIVE` remains `FALSE` regardless of outcome.
