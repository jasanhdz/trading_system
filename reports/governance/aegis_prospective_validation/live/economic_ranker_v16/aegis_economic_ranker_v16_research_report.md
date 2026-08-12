# Aegis Economic Ranker V16 Research Report

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

The pairwise ranker learned weak but measurable cross-sectional ordering. It did
not produce positive expected utility and materially worsened MAE, adverse-first
rate, and loss tails. No model was exported and neither Shadow nor Live changed.

## Design

- Evidence: 96,382 directional rows from the frozen V11 dataset.
- LONG contract: unchanged V15 129-position contract.
- SHORT contract: unchanged V15 168-position contract.
- Control: V15 independent clean, danger, and q90-MAE heads.
- Candidate: a standardized pairwise logistic ranker.
- Pairs: candidates from the same timestamp only, with both orientations.
- Target: canonical trajectory tier, then utility after frozen costs, lower MAE,
  and less time underwater.
- Thresholds: selected on calibration only.
- Validation: four purged expanding train/calibration/test folds.

All evidence was already available when V15 completed. V16 is retrospective and
could not authorize promotion even if the hypothesis passed.

## Ranking Skill

| Side | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | ---: | ---: | ---: | ---: |
| LONG | 54.67% | 55.48% | 53.12% | 53.43% |
| SHORT | 52.49% | 50.63% | 50.30% | 50.27% |

The ranker exceeded random pair ordering in all folds. LONG showed a modest,
repeatable ranking signal. SHORT was only marginally above random in three of
four folds.

## Economic Comparison

| Side | Pair skill | Positive utility | Better utility | Better CVaR | Better positive rate | Lower adverse-first | Lower MAE | Better clean rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LONG | 4/4 | 0/4 | 2/4 | 1/4 | 4/4 | 0/4 | 0/4 | 3/4 |
| SHORT | 4/4 | 0/4 | 3/4 | 0/4 | 4/4 | 0/4 | 0/4 | 4/4 |

The ranker increased positive and clean selections, but paid for that with
larger adverse excursions and worse tails. It never achieved positive mean
utility.

In fold 4, LONG positive rate improved from 16.59% to 23.61%, while mean MAE
worsened from 0.2370% to 0.3021% and adverse-first rate rose from 0.98% to
3.70%. SHORT positive rate improved from 13.00% to 21.59%, while mean MAE rose
from 0.2847% to 0.4677% and adverse-first rate from 2.69% to 9.66%.

## Interpretation

V16 demonstrates that relative opportunity ordering and absolute entry safety
are different tasks. The pairwise objective can recognize more winners, but its
weak ranking margin is not sufficient to reject dangerous paths. A single
ranking score should not be entrusted with both responsibilities.

The result also explains why V15's independent-head composite was conservative:
its explicit danger and MAE penalties reduced adverse paths, even though it
ranked profitable opportunities poorly. Replacing those safety estimates with a
ranker would worsen the system.

## Recommended Next Experiment

V17 should test a two-stage architecture, still offline:

1. a calibrated feasibility gate rejects high danger and excessive q90 MAE;
2. the pairwise ranker orders only candidates that pass that gate;
3. gate thresholds are learned on calibration data and frozen for test;
4. LONG and SHORT remain separate;
5. utility must be positive after costs, CVaR and MAE must improve, and
   opportunity frequency must remain viable;
6. a genuinely new post-V16 holdout is mandatory before any Shadow proposal.

This is not authorization to implement V16 or V17 in runtime.
