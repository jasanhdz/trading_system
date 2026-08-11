# Competing Barrier V10 Validation Summary

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V10 identifies reproducible direction and barrier-path structure, but it does
not convert that statistical skill into positive out-of-sample entry utility.
No model was exported and neither Shadow nor Live was changed.

## Frozen evidence

- Evidence interval: 2025-08-09 through 2026-08-09.
- Source pairs scanned: 96,371.
- Independent, side-neutral episodes: 48,191.
- Rows: 96,382 (one LONG and one SHORT view per episode).
- Symbols: all 11 canonical symbols.
- Contracts: 5%, 10%, and 20% ROE at 30, 60, and 120 minutes.
- Reporting leverage used only to translate ROE to price movement: 15x.
- Severe round-trip cost: 20 basis points.
- Labels: future OHLC only.
- Exchange calls and mutations: zero.

## Component skill

The side-neutral direction classifier passed in all four chronological folds.
Its accuracies were 50.14%, 63.51%, 73.56%, and 59.06%, versus corresponding
training-majority accuracies of 44.10%, 62.68%, 72.54%, and 52.90%.

The competing-risk gate also passed in all four folds for each side. LONG had
6, 6, 8, and 8 skilled contracts out of nine. SHORT had 6, 6, 9, and 8.
This is evidence that the inputs contain information about direction and which
barrier state is likely. It is not evidence of positive trading expectancy.

## Economic validation

| Side | Fold | Calibration result | Test selections | Mean test utility | Control mean |
|---|---:|---|---:|---:|---:|
| LONG | 1 | fewer than 30 eligible | 0 | N/A | -0.1963% |
| LONG | 2 | fewer than 30 eligible | 0 | N/A | -0.1913% |
| LONG | 3 | fewer than 30 eligible | 0 | N/A | -0.2098% |
| LONG | 4 | fewer than 30 eligible | 0 | N/A | -0.1999% |
| SHORT | 1 | policy available | 26 | -0.3702% | -0.2039% |
| SHORT | 2 | fewer than 30 eligible | 0 | N/A | -0.2087% |
| SHORT | 3 | fewer than 30 eligible | 0 | N/A | -0.1902% |
| SHORT | 4 | fewer than 30 eligible | 0 | N/A | -0.2001% |

The only calibration-eligible SHORT policy selected 26 test episodes. Its mean
utility, lower-tail result, payoff ratio, and opportunity-gap gates all failed.
It performed worse than the unfiltered primary-contract control.

## Interpretation

V10 resolves two conceptual defects from V9:

1. Labels no longer contain the causal condition being predicted.
2. Exact future return regression has been replaced with explicit competing
   events and an honest uncertain state.

The failure is now economically informative. At severe costs, predicted edge
is too small or too rare. The models often know which broad path is more likely,
but their probability separation is insufficient to pay for adverse outcomes,
unknown states, and costs. Classification accuracy alone must not promote this
system.

## Prohibited post-result changes

This experiment must not be made to pass by lowering the 30-selection minimum,
reducing the frozen cost assumption, selecting only one favorable fold, or
tuning thresholds on test outcomes. Any such hypothesis requires a separately
preregistered V11 experiment.

## Recommended next experiment

A V11 design should keep the outcome-only labels and non-overlapping episodes,
then investigate calibration and contract specialization without changing V10.
Recommended ablations are: one model per horizon family versus one shared
ordinal model; explicit probability calibration diagnostics by symbol/regime;
and utility attribution that shows whether failure comes from cost, adverse
probability, or unresolved probability. V11 should remain offline first and
must pass the same incremental economic gates before any separately authorized
Shadow deployment.

## Safety state

- Runtime selection effect: `NONE`.
- Model exported: `false`.
- Shadow activated: `false`.
- Live activated: `false`.
- Exchange calls: `0`.
- Exchange mutations: `0`.

## Artifact hashes

- Configuration: `45f04ebd8028b15519e7df1058c60dcaac3c191ab36bd9f034d3626fd9e1d20f`.
- Dataset: `4d5557be5fdd5fa3aec2828b8734d468107d2f004808dd3ff6274fb2bcc223b5`.
- Dataset manifest: `dd61d2966266d9f4b2b5577c6b450fa939077d37e442c04b03c7ce996a77b369`.
- Validation: `5f495a04a5f2432775e1153aba9515eb9f58c1b06c0a47dc48a757966d1329fb`.
