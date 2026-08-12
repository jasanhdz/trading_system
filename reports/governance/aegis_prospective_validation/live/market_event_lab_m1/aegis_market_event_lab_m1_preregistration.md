# Aegis Market Event Laboratory M1

## Status

`PREREGISTERED_RESEARCH_ONLY`

M1 tests a new source of information. It does not modify V15, V17, Live,
Shadow, TypeScript execution, capital, guards, or exchange state.

## Why this experiment exists

The prior candle-based experiments did not demonstrate stable net economic
utility. Repeating feature, threshold, seed, or model searches over the same
evidence would increase selection bias without adding information. M1 instead
asks whether explicit market events contain incremental causal information.

The experiment studies four hypotheses independently:

1. liquidation cascade followed by absorption and reversal;
2. breakout confirmed by aggressive flow and expanding open interest;
3. spot-futures basis or funding dislocation followed by convergence;
4. aggressive flow absorbed by the book before a reversal.

Failure of one family is not evidence for another family. No family may run
without its required physical source.

## Causal contract

At an event timestamp, a feature may use only observations whose exchange
timestamp is less than or equal to that event timestamp. Entries are priced at
the next closed-bar open. Labels and path outcomes are never inputs. Missing,
extra, stale, non-finite, reordered, or schema-incompatible data fail closed.

The canonical sources are futures candles, aggregate trade buckets, open
interest, funding, mark price, spot reference, derived basis, depth snapshots,
and liquidation events. Synthetic replacement values and silent zero filling
are prohibited.

## Evaluation

Data are divided chronologically into design/train, calibration, and a final
holdout. The holdout is opened once. Purge and embargo separate adjacent
partitions. Thresholds and quantiles are learned only from training data.

The primary objective is net expectancy after fees, slippage, and funding, not
classification accuracy. MAE, MFE, target-before-stop, drawdown, CVaR,
frequency, temporal stability, symbol concentration, and uncertainty intervals
are mandatory. LONG and SHORT are reported separately.

Controls include no-trade, time-matched random direction, simple momentum,
simple mean reversion, and frozen V15/V17 results where comparison is valid.
Events within correlated time clusters are not counted as independent wins.

## Model policy

The first pass uses only regularized logistic regression and shallow histogram
gradient boosting. A more complex model is prohibited unless a simple model
first demonstrates positive validation edge. Seed mining, feature mining, and
threshold mining on validation or holdout are prohibited.

## Trial accountability

Every trial receives a unique identity and is appended to a SHA-256 hash chain.
Failed and negative trials remain in the ledger. Reusing a trial identity or
rewriting an existing record invalidates the ledger.

## Gates

An event family is not ready to test until every required source has at least
60 days of causal coverage for all 11 symbols and at least 300 independent
events per direction. Shadow eligibility additionally requires a sealed final
holdout, positive lower uncertainty bounds for expectancy and profit factor,
controlled MAE and concentration, temporal stability, superiority to the
time-matched random control, and no material leakage finding.

M1 cannot activate Shadow or Live. Promotion requires a later, separately
authorized decision based on new evidence.
