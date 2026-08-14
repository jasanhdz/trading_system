# Aegis W3 Intrabar Wave State and Optimal Timing Preregistration

## Purpose

W3 tests whether closed one-minute state contains economically useful timing
information that was hidden by W1/W2 aggregation. It does not retune either
prior experiment. W3A tests entry timing; W3B independently tests exit timing.
Neither study may borrow a passing result from the other.

## Prior Evidence Boundary

W1 and W2 remain immutable negative evidence. Their final holdouts remain
sealed. W3 selection uses January through September 2024, which belonged only
to W2's broad development population. W2 VALIDATION begins in October 2024 and
is therefore inside the sealed W3 holdout, not W3 selection.

The repository has complete one-minute USD-M klines for all 11 symbols from
2024-01-01 through 2026-07-31. They contain OHLC, quote/base volume, trade
count, and Binance taker-buy volume. Checksum-verified tick-level aggTrades
exist from August 2025 through July 2026, but overlap W1 selection/holdout and
will not be used to select W3. Historical order-book depth does not exist and
will not be reconstructed.

## Frozen Splits

- TRAIN_W3: 2024-01-08 through 2024-06-30.
- VALIDATION_W3: 2024-07-01 through 2024-09-30.
- FINAL_HOLDOUT_W3: 2024-10-01 through 2026-07-31, `SEALED`.
- Purge: 180 minutes at partition boundaries.

## Wave Identity

`wave_episode_id` is a SHA-256 identity over experiment version, symbol, side,
and closed 5m impulse timestamp. A closed 5m bar opens an observation window
when volume ratio 20 is at least 1.25 and absolute body is at least 0.10 ATR.
Direction is the body sign. A 30-minute symbol/side cooldown prevents treating
correlated minute decisions as independent waves. The anchor is not an order.

## W3A Entry Timing

At fixed offsets 0, 1, 2, 3, 5, 8, and 10 minutes after the anchor, W3A records
causal state and evaluates entry at the next complete 1m open. Its primary
contract is +0.50 ATR before -0.25 ATR within 10 minutes. Secondary contracts
are fixed in YAML. Baselines are immediate entry, wait one minute, impulse
extreme break, a 25%-50% pullback, and no trade.

Regularized logistic regression and shallow histogram gradient boosting are
the only predictive models. Family and probability threshold are chosen on
TRAIN only. VALIDATION tests one frozen policy per side.

## W3B Exit Timing

W3B uses immediate simulated positions from the same independent anchors. It
activates only after peak MFE reaches 0.50 ATR and asks whether 0.25 ATR of
giveback occurs before a new 0.25 ATR favorable extreme within three minutes.
Baselines are bounded hold, two fixed ATR trailings, two fixed giveback rules,
and time exits after 3, 5, or 10 minutes. Reentry and partial reduction are
prohibited.

W3B compares regularized logistic regression, shallow gradient boosting, and
discrete-time hazard logistic regression. Selection is TRAIN-only.

## Economics And Gates

Base round-trip cost is 14 bps, stressed at 20 and 30 bps. Same-minute barrier
ambiguity resolves adverse-first. Execution occurs one minute after a closed
decision. No leverage enters the objective.

W3A must have at least +2 bps net expectancy, beat the best frozen baseline by
at least 2 bps, have positive bootstrap lower bounds, PF at least 1.10, survive
20 bps, remain within 0.40 ATR mean MAE, and be stable across at least 7 symbols
and 3/4 temporal folds.

W3B must improve net expectancy by at least 2 bps and median Profit Capture
Ratio by at least 5 percentage points, with positive bootstrap lower bounds,
20 bps survival, and the same symbol/fold stability.

Ten thousand day-clustered episode bootstraps, Bayesian superiority estimates,
and Benjamini-Hochberg FDR are frozen. Expanding walk-forward is run only if a
candidate first passes the basic gates. The holdout opens only after all
TRAIN/VALIDATION gates pass.

## Safety

W3 is historical research. It does not modify TypeScript, production, PM2,
WebSockets, credentials, risk, trailing, or exchange state. Shadow is not
created unless a complete W3 gate and holdout later justify it. Live remains
false in all outcomes of this task.
