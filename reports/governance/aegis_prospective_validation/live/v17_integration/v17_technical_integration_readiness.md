# V17 Final Technical Readiness

Date: 2026-08-12 UTC

## Gate

`V17_READY_FOR_LIVE = FALSE`

V17 is now reproducible and its Python/TypeScript numerical boundary is proven,
but it is not economically promotable. LONG passed 0/4 folds and SHORT passed
0/4 folds. Every evaluated holdout has negative mean utility. LONG fold 4
cannot produce a valid preregistered calibration policy; reducing the minimum
selection count would still select negative-utility candidates.

No V17 execution authority, activation record, runtime selector, exchange
request, or service restart was created by this work.

## Root Causes

1. `model_exported: false` was deliberate in the preregistered research code,
   which had no final-fit/export prescription. A deterministic, inspectable
   research artifact now exists, but it remains `RESEARCH_ONLY`.
2. V17 lacks edge rather than merely failing a strict threshold. LONG fold
   utilities are -0.22697%, -0.23217%, -0.19630%, and infeasible. SHORT fold
   utilities are -0.12452%, -0.12296%, -0.18697%, and -0.23650%.
3. LONG fold 4 had only 47 distinct ranking-calibration timestamps. Quantiles
   0.80, 0.90, and 0.95 selected 13, 5, and 3 events, with mean utility
   -0.19162%, -0.24640%, and -0.27104%. The preregistered minimum is 20.
4. The 129 LONG and 168 SHORT inputs are exact direction-specific selections
   from the V9 176-feature vector, not padding. They require 576 closed 5m bars
   to construct 15m/1h and rolling causal context. Current Live's 83 features
   are the base of that chain, not a directly interchangeable schema.
5. The existing TypeScript entry path does not durably represent order intent,
   ambiguous acknowledgement, or partial-fill state. It therefore cannot
   prove read-before-retry, restart-before-bracket, complete local-state loss,
   or duplicate-event recovery without an execution lifecycle redesign.
6. The API restart loop came from loading the large current-brain model twice:
   once as `__main__` and again when Uvicorn imported `aegis.live_api:app`.
   Memory exceeded PM2's 10 GiB limit. Uvicorn now uses one lazy application
   factory. The API has remained online for more than two hours at about 5.6
   GiB with zero unstable restarts.

## Feature And Artifact Contract

- Schema: `aegis-v17-v9-directional-features-v1`
- LONG: 129 ordered float64 features; schema hash
  `373ad9bb654833ff872b3ee50d475cbfecdcaec43b85f3369c14119f31b7c03e`
- SHORT: 168 ordered float64 features; schema hash
  `3631db9e26c1aacf2ef0b9fa8383296d62ad09a143f9eadc2715c2ae0cbe9532`
- Required history: 576 closed 5m bars
- Normalizers: frozen StandardScaler for clean, danger, and ranker; native
  histogram-gradient-boosting input for MAE q90
- Failures are explicit for missing/extra/reordered/non-float/non-finite
  features, insufficient/open history, version/hash drift, normalizer drift,
  and nested model hash drift.

Research artifact SHA-256:
`befd2555bc7600f1f27f4f876fc82a165248f93ee46fbd0a73e3148f310d938d`.
An independent rerun produced a byte-identical file with the same hash.

## Python/TypeScript Parity

The golden dataset contains 22 closed historical events: one LONG and one
SHORT event for every canonical symbol. Python reload and TypeScript frozen
evaluation both pass. Maximum Python serialization differences over 256 rows
per side were:

| Side | Clean | Danger | MAE q90 | Rank |
| --- | ---: | ---: | ---: | ---: |
| LONG | 2.22e-16 | 1.67e-16 | 1.04e-17 | 1.56e-7 |
| SHORT | 1.67e-16 | 2.22e-16 | 2.78e-17 | 1.71e-7 |

Probability/MAE tolerance is 1e-12. Rank tolerance is 1e-6 because the
research SGD ranker is fitted/served from float32 while the inspectable JSON
evaluators accumulate in float64. No decision, selected flag, policy status,
feature order, or schema mismatch occurred.

## Lifecycle Result

The existing fake pipeline passes sizing, exchange filters, market intent,
position confirmation retries, SL/TP placement and read-back, emergency close
on unconfirmed position or bracket failure, bracket reconstruction, break-even,
trailing, reconciliation, exit, and accounting telemetry. Focused TypeScript
coverage passed 115/115.

The final lifecycle gate remains failed because the current exchange port
collapses a market response to `avgPrice` and `orderId`. It has no durable
pre-request intent/ack state for partial fill, timeout ambiguity,
read-before-retry, restart before brackets, total state loss, or duplicate
event identity. A parallel simulator would not prove the production pipeline,
so these cases were not falsely marked complete.

## Tests And Failure Classification

- Python focused V17/API/loader: 61 passed.
- Python full: 724 passed, 5 failed.
- The five failures are category E (environment): historical Phase E/E5
  preflights require branch `feature/aegis-ts-clean-rebuild`, while the
  authorized work is on `work/entry-quality-evidence-20260726`.
- The prior pandas failure was category D (fixture): pandas 3 inferred an
  Arrow-backed index. The fixture now explicitly requests `dtype=object`; the
  frozen pickle allowlist was not widened.
- TypeScript full: 762 passed.
- Three prior TypeScript failures were category B: stale source-digest
  expectations for already-authorized files. The digest inventory was updated.
- TypeScript build/strict compilation: passed.
- Prettier on modified TypeScript: passed.
- No separate TypeScript `typecheck` or `lint` scripts exist.
- Read-only audit static import/endpoint validation: passed.
- Python compilation and Git whitespace checks: passed.

## Blocking Conditions

- V17 LONG policy calibration is infeasible.
- V17 LONG and SHORT have no positive holdout utility in any fold.
- No executable/promotion artifact exists; only a non-authoritative research
  artifact exists.
- Required ambiguous-order, partial-fill, durable idempotency, and complete
  recovery scenarios are not represented by the current execution contract.
- The historical branch-bound Python preflights do not pass on this work branch.
- V17 has not earned Shadow promotion evidence.

## V18 Evidence-Based Direction

Do not tune V17 thresholds on the same holdouts. Preregister V18 with a new
untouched temporal period, out-of-fold safety predictions, labels aligned to
net lifecycle utility, side/regime calibration-drift analysis, a minimum
opportunity-rate constraint, and stable subgroup tests before fitting a global
ranker. Keep V15/V16/V17 as fixed controls.

There is no responsible manual `ready -> live` procedure for V17 at this time.
