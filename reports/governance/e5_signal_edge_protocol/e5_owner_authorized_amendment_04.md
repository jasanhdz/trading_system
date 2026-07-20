# E5 Owner-Authorized Amendment 04: Control Self-Matching Prohibition

## 1. Title and Status

| Field | Frozen value |
|---|---|
| Document | `E5_OWNER_AUTHORIZED_AMENDMENT_04` |
| Status | `OWNER_AUTHORIZED_PROSPECTIVE_SCIENTIFIC_AMENDMENT` |
| Experiment | `E5_SIGNAL_EDGE_CONTROL_TEST` |
| Accepted alias | `E5_ENTRY_EDGE_INVESTIGATION` |
| Scope | C1 and C2 control self-matching prohibition only |
| Created UTC | `2026-07-20T19:34:51Z` |
| Repository branch | `feature/aegis-ts-clean-rebuild` |
| Repository HEAD before amendment | `5003630ae42a806f79466ec10a4c052ce2a6f28a` |

## 2. Owner Authorization

The experiment owner prospectively authorizes the S1-S8 rules in this
amendment. They resolve whether an experimental observation may be assigned to
itself or its own canonical experimental unit as a C1 or C2 control.

These rules were selected without observing matching coverage, paired deltas,
p-values, gates, or scientific outcomes. Their sole scientific purpose is to
require each control assignment to be distinct from the experimental unit under
the identity applicable to that control procedure.

## 3. Prospective Timing Declaration

This amendment was created before E5 implementation, input-dataset
materialization, discovery, confirmation, and scientific-result inspection. No
E5 scientific row was inspected. Semi-blind and lockbox data were not accessed.
Only static governance, source, configuration, schema, documentation, type, and
repository-structure inspection was permitted.

## 4. Governance Hierarchy

Authority is ordered as follows:

1. Original preregistration.
2. Protocol Patch 02.
3. Owner-Authorized Amendment 01.
4. Owner-Authorized Amendment 02.
5. Owner-Authorized Amendment 03.
6. This Owner-Authorized Amendment 04.

A later document controls only where it explicitly clarifies, resolves, or
prospectively amends an earlier unresolved rule. This amendment controls only
the C1 and C2 self-matching rule. Every other governance clause remains frozen.

## 5. Verified Hashes and Commits

Every predecessor file was verified byte-for-byte and every predecessor commit
was verified reachable before this amendment was written.

| Authority | Path | Commit | SHA-256 |
|---|---|---|---|
| Original preregistration | `reports/governance/e5_signal_edge_protocol/e5_protocol_preregistration.md` | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` |
| Protocol Patch 02 | `reports/governance/e5_signal_edge_protocol/e5_protocol_patch_02.md` | `92191db1a7c4135252377f64f51b174f180dcd53` | `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e` |
| Owner Amendment 01 | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_01.md` | `943b98a5091c4d9238f754a1e42e63540a4579a6` | `c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4` |
| Owner Amendment 02 | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_02.md` | `521289606117a478debfca00d2e1fbaa5c2a4301` | `b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f` |
| Owner Amendment 03 | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_03.md` | `5003630ae42a806f79466ec10a4c052ce2a6f28a` | `871be087550eb9d632795ded2c8f2633f1e481838198f0ee3ce53b9c8e9a350e` |

The physical SHA-256 of this amendment is recorded with the containing commit.
The containing commit records the post-commit repository identity. Neither
self-referential value is embedded literally in this file.

## 6. Safety State

At creation:

- Python tracked working tree: `CLEAN`.
- TypeScript working tree: `CLEAN` at
  `53105ee34c6e29f960c37c3516a58fffd2aa5906`.
- E5 execution specification: absent.
- E5 implementation and scientific datasets: absent.
- E5 discovery and confirmation: not executed.
- E5 scientific results: absent and uninspected.
- Semi-blind data: untouched.
- Lockbox data: untouched.
- Lockbox: `NOT_CONSUMED`.
- `consumed_queries=[]`.
- `budget_remaining=1`.

## 7. Description of the Identified Gap

The original C1 rule selects uniformly from structurally eligible symbols in
the experimental cycle but does not exclude the experimental symbol
(`e5_protocol_preregistration.md`, lines 171-180). The original C2 rule assigns
the experimental symbol multiset to eligible cycles but does not exclude the
original experimental symbol-cycle pair
(`e5_protocol_preregistration.md`, lines 182-192). Amendment 02 specifies C2's
randomized augmenting-path procedure without resolving that identity question
(`e5_owner_authorized_amendment_02.md`, lines 101-126).

This omission changes the control-assignment distribution and can affect paired
deltas, p-values, matching coverage, and gates. S1-S8 resolve only that omission.

## 8. S1: General Self-Matching Principle

1. An experimental observation must never be assigned to itself as a control.
2. A control must be a distinct canonical control unit from the experimental
   unit under the identity applicable to that control procedure.
3. Apply self-match prohibition before random selection, candidate shuffling,
   graph construction, augmenting-path search, replicate assignment, and
   outcome computation.
4. A prohibited self-edge never enters a randomized candidate set or matching
   graph.
5. Self-match filtering is outcome-blind and cannot depend on realized return,
   barrier event, funding, score, label, p-value, matching coverage, or final
   verdict.

## 9. S2: C1 Self-Matching Rule

For every C1 experimental observation, a candidate with the same canonical
symbol is ineligible:

`candidate_symbol != experimental_symbol`

1. The exclusion applies regardless of fold, month, horizon, timestamp,
   candidate ordering, seed, or reuse status.
2. C1 selects only from structurally eligible distinct symbols.
3. The experimental symbol cannot remain as a fallback.
4. If no distinct structurally eligible symbol remains, C1 is infeasible for
   that observation and emits `C1_NO_DISTINCT_SYMBOL_CONTROL`.
5. Infeasibility propagates under existing C1, coverage, and `NOT_COMPUTABLE`
   governance.
6. No caliper, structural rule, or other eligibility restriction may be
   relaxed, and the experimental symbol cannot be substituted to restore a
   match.

Prospective rationale: C1 is a cross-symbol structural control. The same symbol
does not provide the independently assigned symbol-level counterfactual that C1
is intended to measure.

## 10. S3: C2 Self-Matching Rule

The exact original experimental symbol-cycle pair is prohibited. A C2 candidate
is ineligible exactly when:

`candidate_symbol == experimental_symbol AND candidate_cycle_id == experimental_cycle_id`

The prohibited natural self-match identity is `(symbol, cycle_id)`.

1. The exact pair cannot enter the candidate list, bipartite graph, randomized
   ordering, augmenting-path traversal, or final matching artifact.
2. A different eligible cycle from the same symbol remains eligible only when
   every existing C2 rule passes.
3. This amendment does not impose
   `candidate_symbol != experimental_symbol` on all C2 candidates. It imposes
   only `candidate_pair != experimental_pair`.
4. Same-symbol, different-cycle candidates remain subject to existing temporal,
   cycle, fold, month-stratum, horizon, reuse, graph-eligibility, randomization,
   and augmenting-path rules.
5. Same-symbol eligibility never bypasses a structural or temporal restriction.
6. If removal of the original pair makes C2 infeasible, matching fails closed
   under the existing C2 infeasibility rules and emits
   `C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL`.

Prospective rationale: the exact symbol-cycle pair is C2's canonical
experimental unit and cannot control itself. Excluding every other cycle of the
same symbol would create a stronger cross-symbol restriction absent from the
original C2 design.

## 11. S4: Filtering Order

The C1 processing order is:

1. Build the authoritative structurally eligible candidate pool.
2. Remove every candidate whose symbol equals the experimental symbol.
3. Apply every remaining governed filter.
4. Apply deterministic canonical ordering.
5. Apply governed randomization.
6. Select the control.
7. Record the assignment and exclusion counts.

The C2 processing order is:

1. Build the authoritative structurally eligible candidate-pair pool.
2. Remove every pair whose symbol and cycle ID both equal the experimental
   symbol and cycle ID.
3. Apply every remaining governed edge-eligibility filter.
4. Construct the bipartite graph without prohibited self-edges.
5. Apply deterministic canonical ordering.
6. Apply D2-A randomized adjacency ordering with the frozen seed architecture.
7. Run the authorized augmenting-path matcher.
8. Record assignments, unmatched units, and exclusion counts.

In both procedures, self-match filtering occurs before scientific
randomization.

## 12. S5: Identity Requirements

The future execution specification must define deterministic canonical
identities sufficient to enforce S1-S4.

C1 records at minimum:

- experimental observation ID;
- experimental symbol; and
- candidate symbol.

C2 records at minimum:

- experimental observation ID;
- experimental symbol;
- experimental cycle ID;
- candidate symbol; and
- candidate cycle ID.

The C2 predicate operates on canonical normalized identities, never
inconsistently formatted raw strings. Identity normalization is deterministic,
versioned, and outcome-independent. No identity may include return, barrier,
funding, score, label, p-value, coverage, or verdict.

## 13. S6: Coverage Consequences

1. Self-match prohibition may reduce matching coverage. This is an accepted
   prospective consequence of requiring a genuine control.
2. Coverage cannot be restored by allowing self-matches, relaxing calipers,
   changing folds, month strata, horizons, or seeds, replacing the matcher,
   selecting on outcomes, creating synthetic controls, or reclassifying an
   ineligible candidate.
3. Coverage and unmatched counts are reported transparently by the scopes
   already required in governance.
4. Coverage effects cannot support a post-hoc amendment after E5 scientific
   outcomes are observed.

## 14. S7: Audit Artifacts

Each C1 assignment record contains:

- experimental observation ID;
- experimental symbol;
- pre-self-filter candidate count;
- self-match exclusion count;
- post-self-filter candidate count;
- selected control symbol, or null only when infeasible; and
- infeasibility code where applicable.

Each C2 assignment record contains:

- experimental observation ID;
- experimental symbol;
- experimental cycle ID;
- pre-self-filter candidate-edge count;
- self-edge exclusion count;
- post-self-filter candidate-edge count;
- selected control symbol and cycle ID, or null only when infeasible;
- matching replicate ID; and
- infeasibility code where applicable.

No valid artifact may contain:

- C1: `control_symbol == experimental_symbol`; or
- C2: `control_symbol == experimental_symbol AND control_cycle_id == experimental_cycle_id`.

Candidate and edge counts must reconcile exactly before and after filtering.

## 15. S8: Validation

Phase 0 and implementation tests use synthetic, non-scientific fixtures for the
following seven cases:

1. C1 pool contains only the experimental symbol. Expected:
   `C1_NO_DISTINCT_SYMBOL_CONTROL`.
2. C1 pool contains the experimental symbol and one distinct eligible symbol.
   Expected: exclude the experimental symbol and select the distinct symbol.
3. C2 pool contains only the exact experimental pair. Expected:
   `C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL`.
4. C2 pool contains the exact pair and the same symbol at a different eligible
   cycle. Expected: exclude the exact pair and retain the different cycle.
5. C2 pool contains the exact pair and a different symbol at an eligible cycle.
   Expected: exclude the exact pair and retain the different symbol.
6. Randomized C2 matching receives shuffled source order. Expected:
   byte-identical assignments under the same frozen seed.
7. A graph input contains a prohibited self-edge. Expected: Phase 0 rejection
   or graph-construction failure before augmenting-path execution.

Validation also rejects any assignment artifact violating an S7 invariant or
failing count reconciliation.

## 16. Relationship to D2-A

1. D2-A remains unchanged.
2. Randomized augmenting-path matching and candidate randomization remain
   authoritative.
3. Reuse constraints and fold, horizon, and month isolation remain unchanged.
4. Amendment 04 removes only prohibited exact self-edges before graph
   randomization and matching.
5. The exclusion authorizes neither a greedy substitute nor outcome-based
   rematching.

## 17. Relationship to C1

1. The original C1 structural eligibility rules remain unchanged.
2. C1 receives exactly one added authoritative eligibility restriction:
   `candidate_symbol != experimental_symbol`.
3. Every other C1 selection, randomization, replacement, caliper, coverage, and
   failure-propagation rule remains unchanged.

## 18. Explicit Non-Changes

This amendment does not modify D1-B, D2-A, D3-B, D4-A, D5-A, D6-C, D7-C,
D8-A, D9-D, D10-A, D11-A, or D12-B. It does not modify the historical funding
contract or provider; horizons; folds; month strata; calipers; reuse rules;
replicate counts; seeds; MREM; alpha; Holm family; bootstrap; permutations;
concentration; label economics; IC; discovery or confirmation authority; final
verdict; semi-blind or lockbox prohibitions; lockbox state; or lockbox budget.

## 19. Prohibited Post-Hoc Changes

After this amendment is committed, no observed E5 result may motivate changing:

- whether C1 permits the experimental symbol;
- whether C2 permits the original symbol-cycle pair;
- the self-match identity;
- filtering order;
- same-symbol, different-cycle eligibility;
- failure or fallback behavior; or
- restoration of matching coverage.

## 20. Completeness Declaration

The control self-matching scientific gap is fully resolved. C1 excludes the
experimental symbol. C2 excludes the exact experimental symbol-cycle pair but
retains otherwise valid same-symbol, different-cycle candidates. Both filters
run before randomization and fail closed when no valid control remains. The
required identities, audit records, failure codes, and seven synthetic
validation cases are frozen prospectively.

No other E5 scientific rule changed. No implementation, test code, dataset,
discovery execution, confirmation execution, or scientific result was created
or inspected. Semi-blind and lockbox resources remain untouched.
