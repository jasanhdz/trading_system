# Aegis Economic Alpha Discovery A1 - Result

## Verdict

`A1_NO_PREREGISTERED_MECHANISM_PASSED`

A1 evaluated 995,808 causal 15-minute states across all 11 symbols. It retained
59,488 fixed-horizon outcomes including controls. None of the 30 frozen
mechanism-side-horizon identities passed every preregistered economic gate.
Consequently A1 does not authorize modeling, Shadow or Live.

## Best Result Per Identity

The figures below use the primary 14-bps round-trip cost and daily symbol
spacing. Selecting these rows after observing validation is descriptive only.

| Mechanism | Side | Best horizon | Events | Gross mean | Net mean | Profit factor | Mean MAE | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trend acceptance | LONG | 60m | 83 | 0.5975% | 0.4575% | 2.333 | 0.8345% | FAIL |
| Trend acceptance | SHORT | 480m | 191 | -0.0099% | -0.1499% | 0.865 | 2.1550% | FAIL |
| Extreme reversal | LONG | 240m | 118 | 0.0214% | -0.1186% | 0.861 | 2.5609% | FAIL |
| Extreme reversal | SHORT | 1440m | 164 | 0.3192% | 0.1792% | 1.119 | 3.5051% | FAIL |
| Carry convergence | LONG | 720m | 75 | 0.5540% | 0.4140% | 1.424 | 2.8721% | FAIL |
| Carry convergence | SHORT | 1440m | 300 | 0.5476% | 0.4076% | 1.355 | 2.8378% | FAIL |

## Interpretation

The strongest clue is trend-acceptance LONG at 60 minutes. Its net mean and
bootstrap lower bound were positive, and seven symbols were positive. It still
failed for two decisive reasons: only 83 daily-spaced events were available,
and the frozen top-rank score did not beat random eligible selection. Random
eligible produced 0.5254% net mean, regime-matched random produced 0.4625%, and
the frozen top rank produced 0.4575%. The final temporal third was also
slightly negative at -0.0102%.

This means A1 found evidence that the eligibility condition may identify an
interesting state family, but no evidence that its score ranks entry quality.
It would be invalid to call this a tradable edge or train a model on it now.

Carry SHORT at 24 hours was broad and positive in all three temporal thirds,
but its 95% bootstrap lower bound remained negative and all eligible events
outperformed the top-rank selection. Carry LONG at 12 hours had only 75 events
and failed uncertainty and control comparisons. Extreme reversal was unstable
and generally weak after costs.

## Reproducibility And Safety

- Causal panel SHA-256: `a07fd4f2cafc796239e7a9c13c13037c18ed81fed8285d3ce83ddfb6f77f9797`
- Candidate table SHA-256: `6abf3f49f7a6f0aedda21e822731b14f89ebed8b0ed2ef7a1105f264d00bd5c2`
- Outcome table SHA-256: `155fc24ce079f8f82bf68f94034f99909c093eec414f42849d4516701d3bd0a0`
- Private result SHA-256: `fed5675dc88afe2c1bf40c8c61eaa937aa9460bca3bb26e1c43ac3b773840fa1`
- A second run reproduced the same hashes and counts.
- Focused tests: `5 passed`.
- Unit suite: `769 passed, 5 failed`.
- The five failures are branch-authority checks that require the historical
  branch name `feature/aegis-ts-clean-rebuild`; the active research branch is
  `work/entry-quality-evidence-20260726`.
- Exchange calls: `0`.
- Exchange mutations: `0`.
- Runtime, PM2, Live and Shadow changes: `NONE`.

## Evidence-Based Next Experiment

Do not tune A1 on the opened validation window. A separate A2 preregistration
should test whether trend-acceptance eligibility, without the failed rank
score, generalizes to an untouched historical archive and then to new forward
events. It should also test whether a ranking objective trained only on TRAIN
can beat eligible-random selection. A2 must retain the same realistic costs,
cluster uncertainty by day and timestamp, and keep LONG and SHORT separate.

No machine-learning experiment is justified until ranking adds measurable
incremental value over the transparent eligibility rule and controls.
