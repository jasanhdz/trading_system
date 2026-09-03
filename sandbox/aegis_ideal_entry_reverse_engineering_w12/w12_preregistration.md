# W12 Ideal Entry Reverse Engineering Preregistration

Status: **FROZEN BEFORE LABEL AND MODEL RESULTS**

Seed: `20260826`

Scope: isolated offline historical research only.

## Falsifiable Claim

Retrospective ideal paths are trivial to find. W12 succeeds only if a frozen causal
procedure concentrates later ideal zones and positive net economic returns in the
prospective top 2% without future information. The null is that ideal-entry DNA is
not prospectively distinguishable beyond controls and costs.

## Partitions and Sampling

The fixed partitions are those in `config/w12_frozen.json`. Decisions are synchronized
15-minute snapshots. Entry is next-minute open. Horizons are 15/30/60 minutes. A
60-minute purge is applied at boundaries. No random split has evidentiary authority.

## Teachers

For each symbol, timestamp, side and horizon, a vectorized 1-minute future path creates:

- **A MFE/MAE:** GOOD when MFE >=30 bps, MAE <=20 bps and MFE/(MAE+5) >=2.
- **B barrier order:** GOOD when +30 bps is reached before -20 bps; a same-minute
  collision is adverse-first.
- **C risk/reward:** GOOD when MFE/(MAE+5) >=2 and MFE-14 >=10 bps.
- **D path quality:** GOOD when close-path efficiency >=0.35, time to MFE is within
  the first half of the horizon, pre-MFE adverse excursion <=20 bps and directional
  persistence >=0.25.
- **E economic:** GOOD when MFE >=30 bps and favorable barrier is first, leaving
  positive room after 14/20 bps costs.

Primary binary label is majority vote (at least 3/5). Strict 5/5 and fixed weighted
consensus >=0.70 are diagnostics. Teacher consistency is assessed by pairwise
agreement and month/symbol prevalence, not by choosing the teacher with the best
prospective backtest.

Continuous quality is 0-100 from fixed clipped components: economic 30%, risk/reward
25%, barrier 20%, path 25%. Components normalize against the preregistered barriers
and caps; no prospective quantiles enter the score.

## Zones

Consecutive majority-positive snapshots for the same symbol/side/horizon separated
by no more than 15 minutes form one zone. The highest fixed quality score is the
`best_entry_timestamp`. The primary positive statistical unit is that best entry.
Negatives within 30 minutes of any positive zone are excluded from training. This
prevents adjacent timestamps from inflating effective N.

## Features

Features contain only closed information at T0: returns 1/3/5/10/15/30/60m;
acceleration, persistence and efficiency; ATR/realized volatility/range and
compression; causal recent high/low/range location and breakout; EMA distances and
slopes; relative volume/acceleration/percentile; taker imbalance and persistence;
BTC/ETH context, basket return, breadth, dispersion, beta/correlation, relative
strength/rank; and temporal deltas describing T-60/T-30/T-15/T-5/T-1 to T0.

No label, path, future, teacher, MFE, MAE, barrier, quality or economic outcome column
may enter a model matrix. An automated provenance audit fails the experiment if any
source availability exceeds decision time or future mutation changes past features.

## Discovery Analysis

Discovery-only analysis reports prevalence, standardized median differences, robust
rank-biserial effect, univariate mutual information, monotonic decile rates and simple
predeclared interactions (momentum x flow, compression x breakout, relative strength
x breadth). It is descriptive and cannot add features after validation.

## Models and Selection

Frozen candidate models are regularized logistic regression, moderate random forest,
small histogram gradient boosting, histogram gradient boosting quality regression,
and two-stage opportunity then side. Class weights are allowed; SMOTE is forbidden.

All preprocessing is fitted on discovery. Candidate ranking uses validation top-2%
net14 expectancy, then validation top-2% precision, simpler model, shorter horizon.
One frozen model/formulation/horizon advances once to prospective. Top 1/2/5/10%
are preregistered diagnostics; top 2% is primary.

Classification reports ROC AUC, PR AUC, calibration/Brier, precision, recall, lift and
precision at all four top fractions. Regression reports MAE, Spearman and the same
ranked economic cuts. Primary economics orient next-open to horizon-close return by
predicted side and subtract 14/20/30 bps.

## Controls and Baselines

Discovery/validation select nothing from controls. Prospective controls are:

- deterministic discovery-label shuffle;
- causal features shifted backward by 24h relative to labels;
- eight seeded random features;
- 100 synchronized random-entry selections of equal size;
- always skip, always long, always short, 15m momentum and 15m mean reversion.

The primary model must beat these controls economically. Shuffle/noise must lose
predictive lift; otherwise the hypothesis is considered unsupported.

## Inference, Stability and Success

Confidence intervals resample complete synchronized UTC days, 10,000 draws, seed
20260826. Results are reported by symbol, side, month, horizon and volatility tercile.
Grade A requires positive prospective net14 and net20, positive bootstrap lower bound,
at least 100 trades/four symbols, no symbol above 50%, top-2 precision lift >=1.5,
multiple positive periods, baseline superiority, expected negative-control failure and
a passed leakage audit. B requires real predictive concentration but insufficient
economics. C means no predictive edge over controls. D is insufficient data or leakage.

## Performance Contract

Correctness and determinism dominate speed. Base features are computed once per
symbol, labels in independent symbol x horizon tasks, and frozen model candidates in
independent jobs. Worker count is bounded by physical cores and memory, BLAS threads
are one per worker, caches require input/config/schema hashes, and stage checkpoints
permit resume. GPU is used only if a same-method representative benchmark shows an
auditable benefit; capacity does not authorize additional hyperparameters.

No result authorizes E4, TypeScript, Shadow, deployment or production changes.
