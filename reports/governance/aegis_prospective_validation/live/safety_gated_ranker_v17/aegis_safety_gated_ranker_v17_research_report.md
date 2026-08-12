# Aegis Safety-Gated Economic Ranker V17 Research Report

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V17 tested the preregistered two-stage hypothesis on 96,382 canonical rows. It
used the V15 direction-specific features, put V15 clean/danger/q90-MAE
predictions in front of the V16 pairwise ranker, and kept gate and ranking
calibration chronologically separate.

The candidate achieved zero successful folds for LONG and zero for SHORT. Mean
utility remained negative in every evaluated fold. No model was exported and
neither Shadow nor Live changed.

## Calibration Integrity

- Safety heads and ranker were fitted only on each training partition.
- The first chronological half of calibration selected the safety gate.
- The second chronological half selected the rank threshold.
- Test outcomes did not select models, gates, or thresholds.
- LONG fold 4 was recorded as `CALIBRATION_INFEASIBLE`; its 108 ranking-
  calibration survivors could not produce the preregistered minimum of 20
  timestamp-level selections. The minimum was not relaxed after inspection.

## LONG Results

| Fold | Status | Selected | Mean utility | Positive rate | Mean MAE | Adverse-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | EVALUATED | 104 | -0.22697% | 16.35% | 0.37518% | 3.85% |
| 2 | EVALUATED | 28 | -0.23217% | 14.29% | 0.43958% | 10.71% |
| 3 | EVALUATED | 50 | -0.19630% | 28.00% | 0.60501% | 16.00% |
| 4 | CALIBRATION_INFEASIBLE | 0 | N/A | N/A | N/A | N/A |

Against V15, V17 improved neither mean utility nor mean MAE nor adverse-first
rate in any fold. Against V16, mean utility improved in three folds and MAE in
two, but coverage passed in only one fold and utility never became positive.

The LONG gate is not stable enough across periods. Its safety thresholds
removed opportunity without reliably reducing realized path risk relative to
V15.

## SHORT Results

| Fold | Status | Selected | Mean utility | Positive rate | Mean MAE | Adverse-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | EVALUATED | 27 | -0.12452% | 25.93% | 0.25465% | 0.00% |
| 2 | EVALUATED | 31 | -0.12296% | 38.71% | 0.35091% | 3.23% |
| 3 | EVALUATED | 109 | -0.18697% | 22.02% | 0.33211% | 3.67% |
| 4 | EVALUATED | 99 | -0.23650% | 15.15% | 0.34325% | 5.05% |

The SHORT gate corrected V16's main pathology: it improved realized MAE and
adverse-first rate in all four folds, CVaR in three, and mean utility in three.
However, it never produced positive mean utility, never preserved V16's
positive rate or clean rate, and met both opportunity-frequency constraints in
only one fold.

Against V15, SHORT mean utility improved in three folds and positive rate in
all four, but MAE and adverse-first rate improved in only two. This is useful
diagnostic evidence, not a stable policy.

## Interpretation

V17 confirms the architectural distinction discovered by V16:

- the ranker can identify relatively more favorable outcomes;
- the safety gate can remove many dangerous V16 choices, especially SHORT;
- neither component currently identifies selections with positive net utility;
- stronger abstention can make path statistics look safer while making the
  opportunity stream too sparse.

The bottleneck is no longer the order in which these existing heads are
combined. The learned safety boundary changes materially across periods, and
the surviving candidates still have negative economics after the frozen cost
and trajectory contract.

## Recommendation

Do not create V18 by adding another score or tuning these thresholds on the
same data. Freeze V17 and accumulate a genuinely post-V16 temporal holdout.
Before another selector is proposed:

1. measure calibration drift of danger and q90-MAE by side, regime, symbol,
   and month;
2. decompose negative utility into direction error, timing, excursion, and
   frozen costs;
3. require out-of-fold safety predictions for any future gate training;
4. define a minimum viable opportunity rate before optimization;
5. test whether any stable subgroup has positive utility before fitting a
   global ranker;
6. retain V15 and V16 as controls on the new untouched period;
7. require positive utility and safety improvements before requesting a
   separate Shadow authorization.

This result is not evidence that V17 should be activated. It is evidence that
the two-stage decomposition is technically coherent but the present learned
signals are not economically sufficient.

## Safety and Reproducibility

- Preregistration commit: `73127a0`
- Implementation commit: `deb7a02`
- Configuration SHA-256:
  `b1410a68f5a9157b84f19ff139bd5497e41428e05b5afcffad2531aa9314201a`
- Validation SHA-256:
  `06f6309fbd898117877d9954b4fb9981243ba17b953e52c2ad35f564fe6c99f4`
- Focused V15/V16/V17 tests: 13 passed
- Full regression: 708 passed, 6 inherited failures
- Inherited failures: five branch-authority assertions and one pandas
  `ArrowStringArray` allowlist incompatibility
- Python compilation: passed
- Git whitespace validation: passed
- Black: all three V17 files reported unchanged; compiled Black 26.5.1 did
  not terminate after reporting and was bounded by timeout
- Deterministic rerun: identical validation SHA-256
- Model exported: no
- Shadow changed: no
- Live changed: no
- Exchange calls: 0
- Exchange mutations: 0
