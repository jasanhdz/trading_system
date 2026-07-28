# Aegis Specialized Committee V2 Shadow Contract

## Status

- Contract version: `aegis-specialized-committee-v2-shadow-contract-v1`
- Runtime mode: `SHADOW`
- Runtime authority: `OBSERVATIONAL_ONLY`
- Exchange authority: `NONE`
- Automatic training: `PROHIBITED`
- Automatic Live promotion: `PROHIBITED`

## Objective

Committee V2 measures whether specialized, causally available evidence can
improve the current entry decision without changing the production decision.
It records what the current control selected, what each specialist observed,
what a counterfactual meta-selector would have done, and the later outcome.

This phase addresses only the first conceptual problem: the existing system
has one eligible directional estimator rather than several independent
directional models. Committee V2 must represent that fact explicitly while
collecting the evidence required to justify future specialization.

## Current Directional Authority

The only eligible directional member is:

- `short_opportunity`: current SHORT champion plus Entry Quality V2 evidence.

Its direction probability means side authority, not calibrated profitability
confidence. A single estimator cannot supply two independent directional
votes. The runtime therefore reports:

- `directional_consensus=NOT_APPLICABLE_SINGLE_ELIGIBLE_DIRECTIONAL_MEMBER`;
- `candidate_confidence=NOT_APPLICABLE_SINGLE_ESTIMATOR`;
- `fabricated_votes=0`.

The experimental LONG member remains non-voting because its current offline
validation result is `FAILED`.

## Specialist Members

The observer records these non-voting specialists:

- `short_reversal_risk`: causal reversal and failed-breakdown proxies;
- `entry_timing`: causal timing and confirmation proxies;
- `qmae`: current adverse-excursion estimate;
- `tail_risk`: current RV2/TRRM tail-risk evidence;
- `regime`: current factorized regime observer;
- `long_opportunity`: experimental LONG challenger, observation only.

These members do not become independent directional votes merely because they
produce related risk or context values.

## Counterfactual Selector

The selector can emit only:

- `ENTER_NOW`;
- `WAIT_CONFIRMATION`;
- `DO_NOT_ENTER`.

It is an unvalidated counterfactual observer. It cannot change the canonical
selection, TypeScript guards, capital, leverage, sizing, orders, positions, or
exchange state. At most one paper entry is recorded per cycle, matching the
current canonical concurrency semantics.

## Evidence

Signals and matured outcomes are written to append-only private journals under
`data/committee_v2_shadow/` with restrictive permissions. Outcome labels use
the configured 12-bar horizon and round-trip cost assumption. Duplicate
decision cycles and timestamps are idempotent.

The evidence must permit direct paired comparison between:

- current canonical control;
- Committee V2 counterfactual decision;
- each specialist contribution;
- realized net return, MAE, MFE, and overlap identity.

## Promotion Gate

No component may receive Live authority until a separately authorized review
shows incremental value over the current control using:

- at least 300 non-overlapping prospective episodes;
- at least 50 episodes for any symbol-specific conclusion;
- purged walk-forward analysis;
- execution-aware net returns;
- MAE and tail-loss comparison;
- coverage and abstention analysis;
- confidence intervals;
- zero unexplained decision or journal-integrity discrepancies.

Promotion requires a new owner authorization tied to exact code, configuration,
model, and evidence hashes. Editing YAML from `SHADOW` to `LIVE` is not a valid
promotion mechanism.

## Prohibitions

Committee V2 must not:

- create, cancel, modify, or close an exchange order;
- change leverage, margin, capital, sizing, or TypeScript guards;
- fabricate votes or use majority voting over dependent evidence;
- train or replace models automatically;
- claim profitability from a directional probability;
- make the failed LONG challenger operational;
- alter Shadow or the current Live decision.
