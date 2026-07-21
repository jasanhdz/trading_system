# E5 Owner Authorized Amendment 07

## Historical E5 Closure for Missing Contemporaneous Row Targets

**Status:** FINAL - OWNER AUTHORIZED  
**Classification:** `HISTORICAL_E5_NON_EXECUTABLE_MISSING_CONTEMPORANEOUS_ROW_TARGETS`  
**Authority type:** Historical execution closure; no new scientific method  
**Effective date:** 2026-07-21

## 1. Owner Authorization

The Owner authorizes this amendment to close the historical E5 execution lane
because mandatory contemporaneous row-level targets cannot be established from
the frozen evidence. This amendment neither supplies those targets nor permits
their reconstruction.

The closure is an inconclusive evidence-availability determination. It is not a
negative scientific result, does not establish absence of edge or profitability,
and does not reject the current Aegis brain.

## 2. Triggering Evidence Boundary

Static schema and source-contract inspection, without opening scientific rows,
established all of the following:

- `EconomicTrade` persists `gross_return_fraction`, `net_return_fraction`,
  `mfe_fraction`, and `mae_fraction`;
- `EconomicTrade` does not persist `tail_event`, `qmae`, `clean_quality`,
  `net_quality_after_costs`, or `label_valid`;
- those five values existed only in the in-memory `TrainingTarget`;
- no immutable row-level target artifact was persisted;
- `dataset_manifest.json` contains summary metadata and hashes, not row-level
  target values; and
- no exact source-field mapping exists from which a blind custodial projection
  could copy the required values without recomputation or inference.

The terminal status preceding this amendment was
`E5_HISTORICAL_LABEL_SOURCE_MAPPING_UNRESOLVABLE`.

## 3. Authority Basis

This conclusion applies the existing chain rather than reopening it:

- the original preregistration freezes historical E3 entries and forbids
  result-driven reselection;
- Protocol Patch 02 reserves Folds 1-2 for Discovery and Folds 3-4 for one-shot
  Confirmation;
- Amendments 01-04 leave the governed populations, outcomes, controls,
  inference, and data boundaries unchanged;
- Execution Specification `E5-R012` forbids null mandatory label fields;
- Execution Specification `E5-R020` requires valid mandatory labels for each
  horizon population;
- Execution Specification input contract identifies `tail_event`, `qmae`,
  `clean_quality`, `net_quality_after_costs`, and `label_valid` as target fields
  under `aegis-labels-short-v4`;
- Amendment 05 prohibits retrospective reconstruction or imputation of
  historical scientific evidence and keeps essential fields fail-closed;
- Amendment 06 creates a narrow derived-identity authority for `trade_id` only
  and explicitly leaves all other essential fields under Amendment 05; and
- Phase 0 and Phase 1A prove engineering primitives and clean-room entry
  projection, not availability of the missing scientific targets.

## 4. Frozen Closure Rules

Rule `E5-A07-R001`: Historical E5 Discovery and Confirmation are permanently
`NOT_EXECUTABLE` with the currently available evidence.

Rule `E5-A07-R002`: No implementation may fabricate, infer, default, impute,
recompute, regenerate, or substitute any of the five missing mandatory targets.
Aggregate scores, `EconomicTrade` returns or excursions, current label code,
current-brain output, market history, and later outcomes are not substitutes.

Rule `E5-A07-R003`: The 702-row sealed Fold 1-2 entry manifest remains immutable
and preserved for audit. It does not become a complete E5 scientific dataset and
must not be used to bypass the mandatory-label contract.

Rule `E5-A07-R004`: Phase 0 remains a valid synthetic engineering validation.
Phase 1A remains a valid clean-room custodial projection. Neither result implies
that historical E5 is scientifically executable.

Rule `E5-A07-R005`: Historical E5 is inconclusive. No claim about edge,
profitability, loss, model quality, component quality, or current-brain fitness
may be inferred from this closure.

Rule `E5-A07-R006`: No future task may silently reopen historical E5. Reopening
requires discovery of an authentic contemporaneous immutable row-level target
artifact, independent verification of its identity, provenance, hash, complete
one-to-one linkage, and a separate explicit Owner authorization issued before
any Discovery or Confirmation access.

Rule `E5-A07-R007`: A prospective current-brain protocol is scientifically
separate. Prospective observations cannot repair, complete, replace, or be
merged into the historical E5 population.

## 5. Failure and Enforcement

| Code | Trigger | Required behavior |
|---|---|---|
| `E5_HISTORICAL_EXECUTION_CLOSED` | request to run historical Discovery or Confirmation | fail before scientific input access |
| `E5_HISTORICAL_TARGET_ARTIFACT_UNAVAILABLE` | required contemporaneous row-target artifact remains absent | retain non-executable closure |
| `E5_HISTORICAL_TARGET_RECONSTRUCTION_PROHIBITED` | attempted computation, inference, default, or imputation | fail before target creation |
| `E5_HISTORICAL_REOPEN_AUTHORITY_REQUIRED` | attempted reopening without verified artifact and later Owner authority | fail closed |
| `E5_HISTORICAL_ARTIFACT_MUTATION_PROHIBITED` | attempted mutation of sealed historical artifacts | fail before write |

## 6. Unchanged Science and Preserved State

This amendment changes none of D1-D12, historical entries, folds, populations,
symbols, cycles, sides, scores, labels, horizons, costs, funding, barriers, C1,
C2, matching, bootstrap, temporal permutation, Holm, gates, partition meaning,
one-shot semantics, or lockbox budget. It closes execution because a mandatory
input cannot be proven to exist.

The frozen state is:

```text
scientific_rows_manually_inspected = 0
fold_3_4_values_exposed = 0
semi_blind = NOT_ACCESSED
lockbox = NOT_CONSUMED
consumed_queries = []
budget_remaining = 1
historical_discovery = NOT_STARTED
historical_confirmation = NOT_STARTED
shadow = NOT_STARTED
live = NOT_STARTED
```

The remaining lockbox query is intact and is not allocated to this closure or
to any prospective protocol.

## 7. Effective Authority Order

For historical E5, the effective chain is Original Preregistration -> Protocol
Patch 02 -> Owner Amendments 01-06 -> Execution Specification engineering
contract -> this Amendment 07 closure. This amendment controls only historical
executability and reopening conditions. Earlier scientific rules remain intact.

## 8. Authorization Record

Owner authorization was supplied in the task titled "AEGIS - Historical E5
Closure, Prospective Current-Brain Validation, and Shadow Readiness
Implementation" on 2026-07-21. The committed file hash and commit identity are
the immutable authorization record.

