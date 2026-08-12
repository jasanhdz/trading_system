# Aegis SHORT Reversal Exit Compatibility X1A Result

## Verdict

`X1A_READY_FOR_SHADOW = FALSE`

`X1A_READY_FOR_LIVE = FALSE`

The corrected split-public flow contract produced 33 independent SHORT
extreme-reversal events on evidence strictly after the V21 holdout. The
candidate `CURRENT_TS` exit improved on the V21 `LOCK_AT_5_ROE` exit but failed
every primary economic and stability gate except sample count and same-event
exit outperformance.

## Data Integrity

- Future source rows: 5,940.
- Future SHORT rows with complete causal flow: 2,959.
- Complete eleven-symbol timestamps: 269 of 270.
- Invalid physical bars: 0.
- Missing or synthetic values inserted: 0.
- Candidate events after 60-minute symbol spacing: 33.
- Entry-rule modifications: none.

The flow used 24 contiguous bars ending five minutes before entry. Base volume
and taker-buy volume were joined exactly by symbol and open timestamp from the
two hash-bound public sources.

## Economics

| Metric | CURRENT_TS candidate | V21 LOCK_AT_5 | Random matched SHORT |
|---|---:|---:|---:|
| Events | 33 | 33 | 33 |
| Mean protected net | -0.0838% | -0.1495% | +0.0655% |
| Win rate | 48.5% | 54.5% | 57.6% |
| Profit factor | 0.66 | 0.34 | 1.45 |
| Mean MAE | 0.618% | 0.618% | 0.372% |
| Mean underwater bars | 13.85 | 13.85 | 10.70 |

An additional five basis points of round-trip cost reduced candidate mean net
to -0.1338%. The daily-block bootstrap 95% interval was approximately -0.252%
to +0.079% per event, with a negative median. All three temporal thirds were
negative. The candidate did not beat no-trade or the matched random control.

The candidate exited 18 events at horizon, 12 through break-even, and 3 through
trailing. Diagnostic exits were also negative: `LOCK_AT_10_ROE` averaged
-0.0541% and `LOCK_AT_20_ROE` averaged -0.1158%. They were not eligible to
replace the preregistered candidate.

## Interpretation

The positive same-event `CURRENT_TS` result observed inside V21 did not
transfer to genuinely later evidence. The later sample shows both entry and
path deterioration: lower win rate, negative expectancy, greater MAE than the
random control, and no positive temporal block. Changing the exit alone does
not rescue the SHORT reversal hypothesis.

This result should not be used to tune the reversal thresholds, select symbols,
or choose a different exit on these same events. That would reuse the holdout
as training data. X1A is frozen as negative evidence.

## Recommendation

1. Do not model, Shadow-activate, or Live-promote the V21 SHORT reversal rule.
2. Stop iterating conventional candle/reversal thresholds on the same history.
3. Continue prospective collection of order-flow, open-interest change, depth,
   liquidations, spread, and basis with immutable timestamps and outcomes.
4. Preregister a new economic hypothesis only after those sources have enough
   history. The new hypothesis should first beat no-trade and random without a
   model.
5. If a base rule becomes positive, use ML first as an abstention and path-risk
   estimator. Do not ask a model to manufacture edge from a negative rule.

## Reproducibility And Safety

- X1A preregistration commit: `eaa73f6`
- X1A implementation commit: `93d2701`
- Candidate dataset SHA-256:
  `9f8540d31b66a1dfcc406994d834074165093b69984bceab34a3fb072a3a5e6b`
- Result SHA-256:
  `886573ad592bfef737c7fc2f93df9adb0aeb07827aa65146e6d6bce82ce51032`
- Manifest SHA-256:
  `34a8e3283a8c83f6ed5a49420717ae7bdd2f045d98abeffaa1ee32f66c48054c`
- Deterministic rerun: identical hashes
- Focused tests: 18 passed
- Full Python regression: 750 passed, 5 historical branch-authority failures
- Model trained/exported: no/no
- Live/Shadow changes: none/none
- Exchange calls/mutations: 0/0
