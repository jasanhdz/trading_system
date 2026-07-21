# Aegis Prospective Model Qualification Protocol v1

## Status and authority

Status: `FROZEN_BEFORE_CANDIDATE_EVALUATION`

Qualification identity: `aegis-prospective-model-qualification-v1`

Prospective protocol: `aegis-prospective-validation-v1`

This protocol qualifies one already-trained model for prospective Shadow observation. It does not reopen historical E5, establish profitability, open the prospective cohort, authorize persistent Shadow, or authorize Live.

## Candidate source and selection

The sole authorized source candidate is the immutable Phase E E3 pre-lockbox validation bundle:

`reports/experiments/e3_validation_official/attempt_1/aegis-short-candidate-e3/runs/d742d9bc0ae867bb/experimental_bundle.json`

Its physical SHA-256 is `386742c20d74a3b67d47cd95629c646195472e05e9e8d136587d40989a82e3d1`. The attempt-2 mirror has the same bytes and hash. Smoke bundles are excluded because their persisted purpose is `PHASE_E_SMOKE_MECHANICS`. The offline reference is excluded because it declares `trained=false` and `purpose=OFFLINE_REFERENCE_ONLY`.

Selection is based only on persisted lineage and intended purpose. No performance value, economic result, E5 row, semi-blind value, or lockbox value may participate. No alternate candidate, checkpoint, seed, or architecture may be substituted after qualification begins.

The source bundle remains immutable. Successful qualification creates a distinct bundle named `aegis-prospective-shadow-candidate-v1` with approval scope `PROSPECTIVE_SHADOW_ONLY` and `approved_for_live=false`.

## Architecture and contracts

- Bundle schema: `aegis-model-bundle-v2`.
- Architecture: deterministic JSON-serialized scikit-learn tree ensembles consumed by `aegis.models.DeterministicModelRuntime`.
- Feature schema: `aegis-features-v2`.
- Ordered feature count: 83.
- Feature hash: `2dc278b4353585fe22503233187e12832cabfd67e2a2e58f4cd683ee6f3b9454`.
- Historical training label schema: `aegis-labels-short-v4`.
- Prospective outcome label contract SHA-256: `d1cbd83874d9823be2db9931052818d36a32ebbfde2625e83b0cf7403ab1e66d`.
- Training implementation: `src/aegis/training/phase_e.py` at commit `aea3437e0a969aa72ba6adb2331ca6e87020c7ad`.
- Inference implementation: `src/aegis/models.py` at the qualification implementation commit recorded in the resulting manifest.
- Frozen seed: `20260718`.
- Source training selection: the final deterministic refit produced by the preregistered Phase E pipeline; no post hoc checkpoint selection is permitted.

## Authorized data boundary

The trained source artifact and its compact lineage manifests are authorized inputs:

- dataset manifest SHA-256 `6c2e97c8ac7bb28a167c0a0783dab9b27ebff69e1ecf34e7052869e9944c5a1c`;
- dataset identity `1ffd0eaf07515d3a1a5fd6363f09c2d8ffe1e1f3925989486dee398e25b8c294`;
- fold manifest SHA-256 `c2bf619fd3372583119b0d7ad7808609e72e8c926705a29349e33f8b018398a7`;
- run manifest SHA-256 `61035fadcfbe65dd135275cc0761548734f909ea9f77aa7ff85e84811aebca73`;
- preflight report SHA-256 `842ffa7d37b3b544d1ddcca3c022e62bd94f19d5ccb6ebe3880481024c6c926c`;
- competition configuration SHA-256 `70c889223b1466ed3e0817a63e7cfafb5b0966bf7c4d3eb3d06544c70c097f79`.

These artifacts were frozen before E5 opened. Qualification may verify their hashes and metadata but must not reopen their row-level training or validation payloads. Model-health validation uses deterministic synthetic feature fixtures marked `PREACTIVATION_NON_COHORT`. It must not calculate economic performance.

Prohibited inputs are historical E5 Fold 3 or Fold 4 rows, combined historical E5 row sources, semi-blind data, lockbox data, prospective cohort events, private exchange data, account data, and any reconstructed historical target.

## Leakage and temporal gates

Qualification must verify from frozen lineage that chronological splits, embargoes, feature cutoff rules, target construction, duplicate handling, normalizer fitting, and calibration fitting were controlled by the preregistered Phase E implementation. It must reject a lineage mismatch, manifest mismatch, feature after its information cutoff, train-validation duplicate, validation-fitted normalizer not frozen by the source pipeline, or any prohibited-data dependency.

No row-level historical data is read during qualification. A passing report therefore means the immutable lineage contract validates; it does not rerun historical science.

## Attempt and determinism rules

Training-attempt limit: zero new training attempts. The already-trained immutable artifact is reused. Qualification-attempt limit: one canonical attempt.

Artifact-selection rule: the exact source path and SHA-256 frozen above. A technical qualification failure is preserved and blocks approval; it does not authorize retraining, seed changes, hyperparameter search, checkpoint shopping, or protocol amendment based on observed outputs.

Determinism classification requires byte-identical source/bundle sealing and exact repeated inference outputs for identical canonical inputs. Batch and single-row predictions must agree exactly under the deterministic JSON runtime. Any drift is `NONDETERMINISTIC` and blocking.

## Qualification gates

Lineage gates require valid source, dataset, feature, label, code, configuration, and environment identities; `trained=true`; no prohibited source access; and no leakage finding.

Inference gates require successful load, exact feature order/count/hash, finite inputs and outputs, declared output domains, deterministic repeated inference, exact batch/single agreement, non-constant outputs on the frozen synthetic fixture, and fail-closed behavior for missing features, unsupported symbols, unsupported intervals, corrupt artifacts, and mismatched hashes.

Integration gates require the TypeScript candidate mode to accept only the qualified identity and hash, record that identity in prospective evidence, preserve D3/RV2/TRRM/QMAE/EQM/ECON1 and final decision contracts, preserve recorder observational equivalence, and emit simulated intents only.

Operational gates require startup mismatch denial, public-only endpoint policy, credential denial, order-operation denial, bounded preactivation smoke, deterministic replay, checkpoint/resume, restart recovery, stale-model detection, latency recording, and memory recording.

## Approval and failure semantics

Approval means only that the trained candidate is technically suitable for separately authorized prospective Shadow observation. The approved bundle must declare:

- `trained=true`;
- `approved=true`;
- `approval_scope=PROSPECTIVE_SHADOW_ONLY`;
- `approved_for_live=false`;
- lifecycle `SHADOW_APPROVED`.

Approval is prohibited if any required gate fails. Qualification does not use profitability, return, win rate, ranking, p-values, or prospective outcomes. Qualification events remain `PREACTIVATION_NON_COHORT`. The activation boundary remains `NOT_OPENED`.

Deterministic terminal failures include candidate ambiguity, unauthorized training lineage, leakage, source/hash/config/version mismatch, nondeterministic inference, model load failure, non-finite output, degenerate output caused by load or preprocessing, integration failure, observational-equivalence failure, endpoint-policy failure, credential access, order operation, and attempt-limit violation.

## Unchanged safety state

Historical E5 remains non-executable. Historical Discovery and Confirmation remain `NOT_STARTED_NOT_EXECUTABLE`. The lockbox remains `NOT_CONSUMED`, `consumed_queries=[]`, and `budget_remaining=1`. The prospective activation boundary, persistent Shadow, USD 16 stage, USD 100 stage, and Live remain inactive.

