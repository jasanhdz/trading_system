# E5 Deterministic Execution Specification

## 1. Title, Status, and Authority

| Field | Value |
|---|---|
| Specification ID | `E5_EXECUTION_SPECIFICATION` |
| Specification version | `e5-execution-spec-v1` |
| Experiment ID | `E5_SIGNAL_EDGE_CONTROL_TEST` |
| Accepted alias | `E5_ENTRY_EDGE_INVESTIGATION` |
| Status | `PROSPECTIVE_ENGINEERING_CONTRACT_UNIMPLEMENTED` |
| Created UTC | `2026-07-20T19:52:36Z` |
| Python HEAD before specification | `a76553d15a239735bbb909f96ff3f06426148f50` |
| TypeScript HEAD | `53105ee34c6e29f960c37c3516a58fffd2aa5906` |

Rule `E5-R001`: this document translates binding governance into deterministic
engineering behavior. It has no independent scientific authority. Governance
wins on conflict; implementation must stop with `UNAUTHORIZED_SCIENTIFIC_CHOICE`
rather than reinterpret it. Authority: all six governance documents. Owner:
governance verifier. Validation: clause traceability. Artifact:
`governance_manifest.json`.

## 2. Governance Hierarchy

Authority order is original preregistration, Protocol Patch 02, Owner Amendment
01, Owner Amendment 02, Owner Amendment 03, then Owner Amendment 04. A later
document controls only its explicit prospective amendment. Failed proposals,
the Phase 0 revalidation artifact, and this specification are not scientific
authority.

Rule `E5-R002`: load all six files as immutable bytes, verify the manifest in
Section 3, and verify commit ancestry in the stated order before any E5 input is
opened. Owner: governance verifier. Validator: Phase 0. Failure:
`GOVERNANCE_HASH_MISMATCH`, `GOVERNANCE_COMMIT_MISMATCH`, or
`GOVERNANCE_CONFLICT`.

## 3. Verified Governance Manifest

| Order | Document | Commit | SHA-256 |
|---:|---|---|---|
| 1 | `e5_protocol_preregistration.md` | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` |
| 2 | `e5_protocol_patch_02.md` | `92191db1a7c4135252377f64f51b174f180dcd53` | `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e` |
| 3 | `e5_owner_authorized_amendment_01.md` | `943b98a5091c4d9238f754a1e42e63540a4579a6` | `c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4` |
| 4 | `e5_owner_authorized_amendment_02.md` | `521289606117a478debfca00d2e1fbaa5c2a4301` | `b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f` |
| 5 | `e5_owner_authorized_amendment_03.md` | `5003630ae42a806f79466ec10a4c052ce2a6f28a` | `871be087550eb9d632795ded2c8f2633f1e481838198f0ee3ce53b9c8e9a350e` |
| 6 | `e5_owner_authorized_amendment_04.md` | `a76553d15a239735bbb909f96ff3f06426148f50` | `a177980633c3280d6eaf6a4a798a6eb623f3692878639894869d2a39f8643774` |

Rule `E5-R003`: the physical specification hash is stored in its containing
commit and future execution manifest, not in this self-referential file.

## 4. Scope

This contract covers governance verification, authorized input materialization,
discovery Folds 1-2, immutable discovery freeze, one-shot development
confirmation Folds 3-4, deterministic artifacts, and final verdict evaluation.
It creates no implementation, data, result, Phase 0 execution, or operational
authority.

## 5. Non-Authority Declaration

Rule `E5-R004`: no implementation option in this document changes a population,
estimand, outcome, matching distribution, hypothesis, alpha, MREM, multiplicity
family, gate, or verdict. Engineering choices are versioned serialization,
identity, module, and deterministic algorithm mechanics required to realize
already-authorized behavior. Failure: `UNAUTHORIZED_SCIENTIFIC_CHOICE`.

## 6. Repository Boundaries

The Python repository is the sole canonical scientific implementation
authority. It owns every population, outcome, matching, resampling, statistic,
gate, classification, and verdict. The TypeScript repository owns no E5
scientific behavior and remains operationally unchanged.

TypeScript may later provide a byte-preserving Binance transport adapter or
schema-only transport validation only when Python independently validates raw
bytes, source identity, hashes, normalization, and completeness. TypeScript
`FundingSnapshot` is forbidden as historical input. No scientific rule is
implemented independently in both languages.

Rule `E5-R005`: production trading integration is out of scope and forbidden.
Artifact: `source_state_manifest.json`. Failure: `UNAUTHORIZED_SCIENTIFIC_CHOICE`.

## 7. Data-Authority Boundaries

| Class | Identity | Permitted stage | Permitted operations | Forbidden operations | Failure |
|---|---|---|---|---|---|
| A: development discovery | E3 SCORING Folds 1-2 in hashed boundary manifest | Discovery only | load authorized values, compute frozen procedures | confirmation access, training, mutation | `DEVELOPMENT_DATA_BOUNDARY_MISMATCH` |
| B: development confirmation | E3 SCORING Folds 3-4 in sealed confirmation-input manifest | One-shot confirmation after freeze | load and evaluate frozen procedures | discovery inspection, refit, second run | `CONFIRMATION_DEPENDENCY_MISMATCH` |
| C: semi-blind | IDs and paths in prohibited-data policy only | None | identity/policy non-access checks | open, hash scientific content, materialize, query | `SEMIBLIND_ACCESS_ATTEMPT` |
| D: lockbox | state metadata only | None | read-only state assertion | open scientific content, query, write state | `LOCKBOX_ACCESS_ATTEMPT` or `LOCKBOX_MUTATION_ATTEMPT` |

Rule `E5-R006`: loaders accept only allowlisted absolute roots resolved from the
hashed `data_boundary_manifest.json`; symlinks and path traversal outside those
roots fail closed. Validator: prohibited-data guard. Artifact:
`prohibited_data_guard_report.json`.

## 8. Development Discovery Partition Contract

Discovery contains only Fold 1 scoring timestamps
`[2025-04-16T02:55:00Z, 2025-05-31T23:59:59Z]` and Fold 2 scoring timestamps
`[2025-07-17T02:55:00Z, 2025-08-31T23:59:59Z]`, from
`config/experiments/aegis_short_candidate_e3.yaml:70-83`. Interval endpoints
are inclusive exactly as the frozen fold resolver applies them.

Rule `E5-R007`: discovery cannot open Fold 3-4 scientific values. Schema-only
metadata is accepted only from a separately hashed schema artifact containing
no rows. Failure: `PROHIBITED_DATA_REFERENCE`.

## 9. Development Confirmation Partition Contract

Confirmation contains only Fold 3 scoring timestamps
`[2025-10-16T14:55:00Z, 2025-11-30T23:59:59Z]` and Fold 4 scoring timestamps
`[2026-01-15T02:55:00Z, 2026-02-28T23:59:59Z]`. It is opened only after
`DISCOVERY_FROZEN` and ledger transition to `STARTED`.

Rule `E5-R008`: the sealed confirmation input hash must match before every read
and resume. Failure: `CONFIRMATION_DEPENDENCY_MISMATCH`.

## 10. Semi-Blind Prohibition

Rule `E5-R009`: no E5 process reads, materializes, hashes as scientific input,
queries, or references data at or after the semi-blind boundary
`2026-04-27T00:00:00Z`. The latest permissible development candle remains
`2026-04-26T23:59:59Z`. Lockbox-state metadata is not scientific data. Failure:
`SEMIBLIND_ACCESS_ATTEMPT`.

## 11. Lockbox Prohibition

Expected state before, during, and after every E5 stage is:

```text
lockbox = NOT_CONSUMED
consumed_queries = []
budget_remaining = 1
```

Rule `E5-R010`: E5 has no lockbox binding or query. It creates no lockbox
transaction, query, consumption, or budget artifact and has no write permission
to global lockbox state. Confirmation means development Fold 3-4 evaluation.
Any access or mutation attempt fails closed; lockbox consequence is `NO_CHANGE`.

## 12. Module Ownership

Every module is prospective Python ownership unless marked transport-only.

| Module | Responsibility | Inputs -> outputs | Forbidden responsibility | Deterministic/test boundary | Primary failures |
|---|---|---|---|---|---|
| governance verifier | hashes, commits, hierarchy | governance -> governance manifest | scientific rows | six hash/ancestry vectors | governance failures |
| source-state verifier | clean commits and versions | Git metadata -> source-state manifest | source edits | clean/dirty fixtures | `DIRTY_WORKTREE` |
| data-boundary guard | classes A-D | policies -> boundary report | row analysis | allow/deny fixtures | boundary failures |
| prohibited-data guard | block C/D | paths/policies/state -> guard report | prohibited row reads | path and permission fixtures | access failures |
| input manifest loader | schema/hash validation | manifests -> typed references | scientific defaults | schema fixtures | input failures |
| identity builder | Section 14 IDs | identity fields -> IDs | outcomes in IDs | golden vectors | `DUPLICATE_IDENTITY` |
| fold resolver | literal fold windows | timestamps -> fold IDs | inferred folds | boundary vectors | `TIME_ALIGNMENT_FAILURE` |
| time utility | UTC/calendar rules | timestamps -> canonical time IDs | local time | boundary vectors | time failure |
| decimal utility | funding decimals | text -> exact decimals | float-first parsing | Amendment 03 vectors | funding rate failure |
| seed utility | KDF and PCG64 | tuples -> streams | mutable seeds | Section 17 vectors | determinism failure |
| population builder | D1 eligibility | entries/bars -> horizon manifests | cross-horizon deletion | missing-path fixtures | horizon failure |
| outcome engine | returns and events | entries/bars -> outcomes | imputation | price/path vectors | outcome failure |
| fixed-cost engine | E3 costs | cost config -> fractions | funding scalar | parity vectors | input mismatch |
| funding transport | official raw retrieval | source config -> raw bytes | normalization/science | request fixtures | funding source failures |
| funding normalizer | Amendment 03 JSONL | raw bytes -> canonical JSONL | clipping/deduplication | decimal/ordering fixtures | funding validation failures |
| funding completeness | coverage proof | raw manifest -> coverage | zero imputation | pagination fixtures | source incomplete |
| ATR engine | Wilder ATR(14) | bars -> ATR values | SMA/fill | golden vectors | ATR failure |
| quintile freezer | Type 7 boundaries | discovery ATR -> boundary artifact | confirmation recompute | tie vectors | artifact mismatch |
| C1 matcher | same-cycle random symbols | candidates/seeds -> assignments | same-symbol controls | S8 vectors | C1 failures |
| C2 matcher | randomized augmenting paths | graph/seeds -> assignments | greedy/cost matching | S8 graph vectors | C2 failures |
| expectancy calculator | return metrics | outcomes -> metrics | alternate weights | metric vectors | outcome failure |
| power simulator | D6-C | discovery residuals -> power | alternate estimand | nested seed vectors | power failure |
| permutation engine | D7-C | complete weeks -> nulls | row shuffles | shift vectors | permutation failure |
| bootstrap engine | D12-B | week blocks -> CI | IID rows | percentile vectors | bootstrap failure |
| Holm adjuster | 12-test family | p-values -> decisions | family mutation | tie vectors | Holm incomplete |
| concentration evaluator | D9-D | returns -> pooled/fold report | fold gate creation | zero-PnL vectors | concentration failure |
| label classifier | D10-A | labels/H12 returns -> classes | new labels/gates | truth table | label failures |
| IC calculator | D11-A | scores/returns -> diagnostics | verdict input | rank vectors | IC not computable |
| discovery orchestrator | Section 41 machine | modules -> freeze | confirmation reads | state transitions | discovery failure |
| freeze writer | immutable seal | discovery artifacts -> freeze | overwrite | hash DAG vectors | freeze mismatch |
| confirmation ledger | one-shot custody | hashes/state -> ledger | lockbox writes | start/resume vectors | confirmation failures |
| confirmation orchestrator | Section 44 machine | frozen inputs -> results | second run | state fixtures | confirmation failures |
| verdict evaluator | conjunctive gates | gate registry -> verdict | diagnostics as gates | exhaustive truth table | verdict failure |
| artifact serializer | canonical bytes | typed artifacts -> files | timestamps in identities | byte fixtures | artifact mismatch |
| hash verifier | SHA-256 DAG | files -> hash manifest | mutable dependencies | tamper fixtures | artifact mismatch |
| Phase 0 validator | pre-execution conformance | static/synthetic inputs -> status | scientific execution | Section 51 | phase failure |
| CLI orchestrator | explicit stage commands | manifests -> state calls | scientific logic | command-state tests | unauthorized choice |

Rule `E5-R011`: each module has unit tests for its deterministic boundary and
integration tests for every listed dependency, but no test is created by this
document.

## 13. Input Contracts

| Input | Authority/schema and required fields | Validation/hash | Stages/failure |
|---|---|---|---|
| governance documents | Section 3 bytes | exact SHA and ancestry | all; governance failures |
| software manifest | Python, OS, architecture, packages, BLAS, tzdata, commits | canonical JSON/SHA | all; `SOFTWARE_HASH_MISMATCH` |
| source-state manifest | both HEADs and clean status | Git object verification | all; `DIRTY_WORKTREE` |
| frozen E3 entry manifest | `trade_id,symbol,fold,signal_timestamp,entry_timestamp,entry_price,side,score,strategy_id` | physical/canonical hashes; unique identity | discovery/confirmation partitioned; input failures |
| development bars | `timestamp,open,high,low,close,volume`; UTC, float64, five-minute | `gen2_d3_series_v1` manifest and file hashes | authorized partition only; input/time failures |
| cycle catalog | cycle ID, signal timestamp, fold, eligible symbol set | unique cycle and canonical order | control construction; duplicate/time failure |
| fold definitions | literal E3 windows and embargo | config hash | all; time failure |
| symbol registry | eleven symbols in `config/universe.yaml:5-17` | ordered hash `f6448e...ccd348` | all; symbol failure |
| scores | final frozen E3 scalar, observation ID | finite float64, no recalibration | score modules; input failure |
| target labels | `tail_event,qmae,clean_quality,net_quality_after_costs,label_valid` | `aegis-labels-short-v4`; target hash | label module; label failure |
| fixed costs | scenario ID, fee/slippage bps per side | E3 source parity; config hash | outcomes; input mismatch |
| trade notional | equal fixed notional; `100.0` currency units for absolute diagnostics | original protocol lines 107-111,143; static constant `scripts/diagnostics/compat_replay/historical_adapter.py:16-18` at commit `d373b38954870f44824cbc71bb30758d0befb5fc` | metrics; input mismatch |
| funding raw artifacts | immutable official Binance USDⓈ-M bytes and request metadata | raw SHA-256 | materialization/outcomes; funding failures |
| funding normalized | seven Amendment 03 fields | `e5-funding-history-v1`, canonical JSONL hash | outcomes; funding failures |
| funding manifest | provider, requests, pages, mapping, coverage, hashes, reconciliation | Amendment 03 manifest validation | all scientific stages; funding manifest failure |
| seed configuration | base `20260718`, namespaces, KDF version | seed-manifest hash and vectors | randomized modules; determinism failure |
| data-boundary manifest | classes A-D, roots, IDs, time limits | canonical JSON/hash | all; boundary failure |
| prohibited-data policy | denied roots/IDs and read/write policy | canonical JSON/hash | all; prohibited access |
| lockbox-state record | status, consumed queries, budget only | read-only exact expected values | all; lockbox failures |
| discovery-freeze manifest | Section 42 dependency DAG | sealed hash | confirmation; freeze mismatch |
| confirmation ledger | Section 43 schema | atomic state/hash validation | confirmation; ledger failures |

Field types are closed as follows. IDs, symbols, strategies, schema versions,
hashes, and classifications are nonempty UTF-8 strings; SHA-256 values are 64
lowercase hexadecimal characters. Fold is integer 1-4 at input and canonical
`F1`-`F4` after validation. Timestamps parse timezone-bearing ISO-8601 to signed
int64 UTC epoch milliseconds. Prices, OHLCV, scores, and governed statistical
inputs are finite float64 in quote currency or dimensionless return units as
declared. Bars are unique by `(symbol,open_ms)` and ordered symbol/open time;
cycles by `(fold,signal_ms)`; entries by `trade_id` and experimental identity;
scores by observation ID; labels by observation ID and target field. Costs are
finite decimal bps per side. Notional is positive exact decimal `100.0` currency
units. Funding fields retain Amendment 03 exact types. Any required null, key
collision, noncanonical order, unit mismatch, or timestamp mismatch fails before
scientific computation. Every input is byte-hashed before its permitted stage;
confirmation inputs cannot be hashed or opened by discovery.

Rule `E5-R012`: nullability is forbidden for identity, authority, score,
required label, cost, and required outcome fields. Diagnostic undefined values
serialize as `NOT_COMPUTABLE`; mandatory undefined values fail their gate.

## 14. Canonical Identities

All non-funding IDs use SHA-256 over UTF-8 compact JSON arrays with no
whitespace, in the field order shown. Timestamps are canonical UTC epoch
milliseconds, symbols uppercase, side `SHORT`, fold `F1`-`F4`, and horizons
`H12|H48|H96`. Collision between unequal source tuples is
`DUPLICATE_IDENTITY`; no suffix is added.

| ID | Versioned source tuple |
|---|---|
| `observation_id` | `["obs-v1",fold,signal_ms,symbol,"SHORT",trade_id]` |
| `entry_id` | `["entry-v1",observation_id,entry_ms,canonical_entry_price]` |
| `cycle_id` | `["cycle-v1",fold,signal_ms]` |
| `symbol_id` | `["symbol-v1",symbol]` |
| `fold_id` | literal `F1` through `F4` |
| `month_stratum_id` | `["month-v1",fold,"YYYY-MM"]` from cycle UTC month |
| `horizon_id` | literal `H12`, `H48`, `H96` |
| `week_block_id` | `["week-v1",fold,iso_year,iso_week]` |
| `match_replicate_id` | `["match-v1",control,horizon,fold,month_or_null,index]` |
| `bootstrap_id` | `["bootstrap-v1",statistic,horizon,scope,index]` |
| `permutation_id` | `["permutation-v1",test,horizon,index]` |
| `power_replicate_id` | `["power-v1",index]` |
| `artifact_id` | `["artifact-v1",relative_posix_path,dependency_hash]` |
| `discovery_run_id` | `["discovery-v1",governance_hashes,spec_hash,input_hash]` |
| `confirmation_run_id` | `["confirmation-v1",discovery_freeze_hash,confirmation_input_hash]` |
| `funding_record_id` | Amendment 03 F7 exact string SHA-256 |
| `raw_funding_artifact_id` | `["funding-raw-v1",provider,request_tuple,raw_sha256]` |
| `manifest_id` | `["manifest-v1",schema_version,content_sha256]` |

Rule `E5-R013`: identities contain no outcome, p-value, match success,
profitability, classification, or verdict. Validator: identity golden vectors.

## 15. Time Semantics

Rule `E5-R014`: all time is UTC; local time and daylight saving are forbidden.
A canonical bar with open timestamp `t` occupies `[t,t+5m)` and is completed at
`t+5m`. Signal timestamp is the frozen hourly anchor and entry is the next
five-minute bar open. Horizon H terminates at the close of exactly H consecutive
completed bars after entry: H12=60m, H48=240m, H96=480m.

Barrier scanning includes bars 1 through H96 in chronological order. If both
barriers occur in one bar, adverse is first. Funding uses UTC epoch milliseconds
and `(entry_ms,termination_ms]`. Fold windows use their literal inclusive
endpoints. Month strata use the cycle's UTC `YYYY-MM`. ISO weeks are
`[Monday 00:00:00Z,next Monday 00:00:00Z)`; a complete week lies wholly inside
its fold. Boundary-partial weeks are excluded from permutation observed and
null support.

## 16. Numerical and Exact-Decimal Semantics

Rule `E5-R015`: market prices, returns, ATR, ranks, statistics, and resampling
statistics use IEEE-754 binary64. Parse validated market decimals directly to
float64. Reject NaN and infinity. Use canonical row order and left-to-right
float64 reduction; parallel reduction is forbidden. No intermediate rounding.
Gate comparisons use full machine values; display rounding never enters a gate.

Funding rate parsing, canonicalization, and event summation use arbitrary-
precision base-10 decimal with the exact Amendment 03 text rules. Convert total
funding once to float64 only when combining with gross return; record both exact
decimal text and resulting float64. Rank ties use average ranks, except decile
membership uses the deterministic total score order from the original protocol.
Type 7 is used for ATR boundaries and bootstrap percentiles.

## 17. Deterministic Seed Architecture

Base seed is `20260718`. C1 preserves the original replicate formula exactly:
the canonical encoding of `protocol_id || control_id || replicate_index` is
UTF-8 text with literal separator `||`, protocol ID
`E5_PROTOCOL_PREREGISTRATION`, control ID `C1`, and an unsigned base-10 index;
the PCG64 seed is the SHA-256 digest interpreted as one unsigned big-endian
integer modulo `2**64` (the digest's final eight bytes). All later-authorized
substreams use compact UTF-8 JSON
`[experiment_id,spec_version,base_seed,namespace,...context]`; SHA-256's first
eight bytes as unsigned big-endian seed the isolated NumPy PCG64 stream. No
stream is shared across namespaces.

Rule `E5-R016`: contexts are:

- C1: one original-formula stream per replicate and canonical
  horizon/fold/observation iteration within that stream;
- C2: `C2,horizon,fold,month,replicate_index`;
- bootstrap: `BOOTSTRAP,statistic,horizon,scope,replicate_index`;
- power outer: `POWER,replicate_index`; inner bootstrap additionally includes
  outer index;
- permutation: `permutation_test_name,horizon,fold,repetition_index`;
- diagnostic bootstrap: `DIAGNOSTIC_IC,horizon,scope,replicate_index`.

| Vector | Canonical input | SHA-256 / seed prefix |
|---|---|---|
| C1 | `E5_PROTOCOL_PREREGISTRATION||C1||0` | `958ca29e6af77cf297f8d3ae772927588892ad11eae3c6a1e48971b81a0b46c0`; `0xe48971b81a0b46c0` |
| C2 | `["E5_SIGNAL_EDGE_CONTROL_TEST","e5-execution-spec-v1",20260718,"C2","H12","F1","2025-04",0]` | `b5879bbb4007d7f0287934c0dd7bff7d5512be5792a6c10c2f49d0c3b427f316`; `0xb5879bbb4007d7f0` |
| bootstrap | `["E5_SIGNAL_EDGE_CONTROL_TEST","e5-execution-spec-v1",20260718,"BOOTSTRAP","H12","F1",0]` | `5711691ffdda359e4f597a10e6fff3b85f40472892ebc10bf6f5cbe21778a94f`; `0x5711691ffdda359e` |
| permutation | `["E5_SIGNAL_EDGE_CONTROL_TEST","e5-execution-spec-v1",20260718,"C_MONO_H12","H12","F3",0]` | `e4c898fb35722cddd5ca8a00ada11ed900b4dfada431855da3d1aece277a67f1`; `0xe4c898fb35722cdd` |

## 18. Deterministic Serialization

Rule `E5-R017`: structured metadata uses UTF-8 canonical JSON: keys sorted by
UTF-8 bytes, no insignificant whitespace, LF final newline, integers as decimal,
finite float64 as the shortest round-tripping decimal, null as JSON `null`.
Row manifests use canonical JSONL in explicit schema-field order and canonical
row order. Funding uses Amendment 03's stricter key order. Original required
Parquet outputs are deterministic projections from canonical JSONL using
PyArrow 25.0.0, Parquet 2.6, no compression, no dictionary encoding, no
statistics, data-page version 1.0, one row group, and canonical column/row
order; both projection bytes and canonical source are hashed. Audit timestamps
live in separate run envelopes and do not enter scientific identities.

## 19. Eligibility Pipeline

Rule `E5-R018`: apply these stages in order and emit a manifest after each:

1. governance-authorized dataset membership;
2. discovery/confirmation partition authority;
3. frozen E3 entry or candidate membership;
4. symbol authorization;
5. canonical identity validation;
6. duplicate identity rejection;
7. fold assignment;
8. timestamp validation;
9. completed-bar availability;
10. horizon completeness;
11. barrier computability;
12. fixed-cost computability;
13. funding-source coverage;
14. funding-event computability;
15. score availability;
16. mandatory-target availability;
17. ATR warm-up;
18. ATR bucket assignment for ATR modules;
19. control structural eligibility and Amendment 04 self filtering;
20. quarantine and integrity checks.

No realized outcome is used before outcome construction or as an eligibility
filter, except horizon completeness checks existence rather than value.

## 20. Exclusion Taxonomy and Precedence

Primary exclusion is the first failed stage in Section 19. All later detectable
secondary reasons are recorded without changing the primary code. Stable codes
are `DATASET_UNAUTHORIZED`, `PARTITION_UNAUTHORIZED`, `ENTRY_NOT_FROZEN`,
`SYMBOL_UNAUTHORIZED`, `IDENTITY_INVALID`, `DUPLICATE_IDENTITY`,
`FOLD_INVALID`, `TIMESTAMP_INVALID`, `BAR_UNAVAILABLE`,
`HORIZON_INCOMPLETE`, `BARRIER_NOT_COMPUTABLE`, `COST_NOT_COMPUTABLE`,
`FUNDING_NOT_COMPUTABLE`, `SCORE_MISSING`, `TARGET_MISSING`,
`ATR_NOT_COMPUTABLE`, `CONTROL_INELIGIBLE`, and `INTEGRITY_QUARANTINE`.

Rule `E5-R019`: every excluded, unmatched, invalid, or incomplete unit is
reported by fold, symbol, horizon, module, and reason. Silent exclusion is
`UNAUTHORIZED_SCIENTIFIC_CHOICE`.

## 21. Horizon-Specific Population Construction

Rule `E5-R020` implements D1-B. H12, H48, and H96 have separate complete-case
populations. A unit is included at H only when its frozen entry, all H bars,
costs, funding coverage, score, mandatory labels, and structural eligibility are
valid. Missing H48/H96 never removes H12; missing H96 never removes H48. No
imputation, shortened horizon, synthetic terminal price, or horizon substitution
exists. C1 and C2 are regenerated per horizon.

Each horizon/fold manifest lists included and excluded observation IDs, reasons,
counts by symbol/fold/month, and attrition relative to frozen entries. Estimates
are horizon-specific, not a common-cohort curve. All twelve Holm tests remain
one family. No new sample minimum is added.

## 22. Gross and Net Outcome Construction

Entry is the frozen next canonical five-minute bar open. H exit is bar H close. For
SHORT:

`gross_return_H = (entry_price - exit_price_H) / entry_price`.

`fixed_cost_return_H = gross_return_H - fixed_round_trip_cost`.

`B_BASE_net_return_H = gross_return_H - B_BASE_fixed_cost + total_funding_return_H`.

Rule `E5-R021`: termination for a fixed-horizon return is its H close;
termination for the barrier diagnostic is the first barrier event or H96 close.
Outcome fields are observation ID, horizon, prices/times, gross return, fixed
cost, exact funding text, funding float64, B_BASE net return, barrier class,
termination reason, and computability. Missing mandatory input propagates
`OUTCOME_NOT_COMPUTABLE` and then gate failure when mandatory.

Before any label is read for analysis, independently recompute the frozen legacy
E3 H12 gross and B_BASE label under the original E3 label formula solely as an
input-integrity check. Absolute difference must be <=`1e-12`, and every frozen
H12 experimental entry must reconcile. That legacy value never becomes the E5
economic outcome: E5 B_BASE replaces legacy scalar funding with Amendment 03
events under Sections 22-24. A path already excluded prospectively under D1
remains in the exclusion manifest; a new loss, changed path, or label drift is a
technical input failure and does not produce a scientific verdict.

## 23. Cost Accounting

Static E3 authority `src/aegis/training/econ.py:17-35` freezes per-side costs:
A_OPTIMISTIC fee 4 bps/slippage 1 bp; B_BASE fee 5 bps/slippage 2 bps;
C_PESSIMISTIC fee 5 bps/slippage 5 bps. Fixed round-trip fractions are `0.0010`,
`0.0014`, and `0.0020`. Historical funding from Amendment 03 replaces the
continuous scalar component and is reported separately. Entry and exit each
apply one fee and one slippage component.

Rule `E5-R022`: all gates use B_BASE unless the original gate explicitly names
pessimistic sensitivity. Leverage, compounding, and sizing are absent.

## 24. Historical Funding Accounting

Rule `E5-R023` implements Amendment 03 exactly. Provider is official Binance
USDⓈ-M Futures. Materialization uses official REST historical funding endpoint
identity `/fapi/v1/fundingRate` or an official first-party archival export
declared before retrieval. One materialization uses one mechanism; substitution
after start is forbidden. REST requests freeze `symbol,startTime,endTime,limit`
with `limit=1000`; next page starts at prior maximum `fundingTime+1` and repeats
until an empty page or all returned times exceed the requested end. Retries use
identical parameters and retain every response byte.

The symbol-to-contract map is the identity map for `ETHUSDT`, `BTCUSDT`,
`SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`,
`LINKUSDT`, `SUIUSDT`, and `LTCUSDT`, with registry hash
`f6448e67daf1d017e16cc6b331f6494e97e178824474994fff08864303ccd348`.
Process symbols in UTF-8 lexical order and pages in ascending request time.
Transport is single-flight. For transport errors, make at most five retries of
the identical request after 1, 2, 4, 8, and 16 seconds; an official
`Retry-After` value replaces that retry's delay but never its parameters. Five
failed retries make the materialization incomplete. Name each raw artifact by
zero-padded symbol page index plus response SHA-256; preserve exact body bytes
and separate request/response metadata.

Raw artifacts are immutable and SHA-256 hashed before normalization. Normalize
exactly seven fields under `e5-funding-history-v1`; use exact decimals, natural
key `(symbol,funding_time_utc_ms)`, Amendment 03 record ID, canonical ordering,
and deterministic JSONL. Duplicates and conflicts fail closed. Retrieval spans
earliest possible entry through latest H96 termination for all eleven mapped
contracts and proves pagination/export completeness.

Include events only in `(entry_ms,termination_ms]`. For E5 SHORT, each event
return equals its signed decimal rate; exact sum is positive receipt or negative
payment. No event under proven complete coverage means `0.0`. Incomplete
coverage is `FUNDING_NOT_COMPUTABLE`. No interpolation, proration, continuous
accrual, clipping, third party, `funding_bps_per_hour`, `FundingSnapshot`, mark-
price notional, or leverage multiplier is allowed.

Scientific statistics use return space. Frozen notional is 100.0 currency units
only for absolute diagnostics: `funding_pnl = 100.0 * funding_return`.

## 25. Barrier Geometry

B_BASE fixed cost is `0.0014`; favorable and adverse gross SHORT barriers are
`+0.0028` and `-0.0028`. At each H96 path bar, favorable is reached when
`(entry-low)/entry >= 0.0028`; adverse when `(entry-high)/entry <= -0.0028`.
Test both on each bar; both means adverse-first. Stop at first classified bar or
H96 close. Funding never changes barriers, order, or termination.

Rule `E5-R024`: report gross barrier class and final net economics separately.
This diagnostic cannot override H12 gates.

## 26. Wilder ATR(14)

For completed bar i:
`TR_i=max(high_i-low_i,abs(high_i-close_{i-1}),abs(low_i-close_{i-1}))`.
First ATR is arithmetic mean of 14 consecutive valid TR values. Thereafter
`ATR_i=(13*ATR_{i-1}+TR_i)/14`. Entry uses latest fully computed ATR from the
last completed bar strictly before entry. A gap or missing OHLC resets warm-up;
14 new consecutive TR values are required. No fill, interpolation, current bar,
future bar, or session reset.

Rule `E5-R025`: float64, canonical chronological order, no intermediate
rounding; unavailable is `ATR_NOT_COMPUTABLE`.

## 27. Type 7 ATR Quintiles

Rule `E5-R026`: use valid pre-entry ATR values from all frozen Fold 1-2 entry
observations before outcome-dependent filtering. Sort ascending. For n values
and probability p, `h=(n-1)p`, `j=floor(h)`, `g=h-j`, and
`q=(1-g)x[j]+g*x[j+1]` with zero-based indices and endpoint clamping. Compute
p=.20,.40,.60,.80 once.

Assign Q1 `x<=q20`; Q2 `q20<x<=q40`; Q3 `q40<x<=q60`; Q4
`q60<x<=q80`; Q5 `x>q80`. Equality enters lower bucket. Duplicate boundaries
and empty buckets remain; equal ATR values are never split. Seal boundaries and
reference-population hash before confirmation; F3-F4 cannot recompute them.

## 28. C1 Control Construction

For each horizon, fold, replicate, and experimental entry, build candidates from
the same frozen signal cycle. Candidate symbols must be in the eleven-symbol
universe, structurally valid, SHORT-available, and outcome-complete at that
horizon. Scores, vetoes, thresholds, ranks, and realized outcome values are not
eligibility inputs. Select exactly one symbol uniformly with PCG64 from the canonically
sorted post-filter list. One entry per cycle makes within-cycle without-
replacement exact. Repeat 10,000 times and preserve timestamps, fold, count,
entry rule, side, horizon, and costs.

Rule `E5-R027`: for an observation to count as controlled it must have exactly
one C1 assignment in every one of the 10,000 requested replicates. An
uncontrolled observation remains in the coverage denominator and is never
silently discarded or reclassified. Paired C1 inference uses the declared
controlled subset and the horizon/fold coverage must be at least 0.95; lower
coverage is `E5_EXECUTION_BLOCKED_BY_CONTROL_COVERAGE`. Artifact:
`c1_matching_manifest.jsonl`.

## 29. C1 Same-Symbol Prohibition

Rule `E5-R028` implements Amendment 04 S2/S4. After structural pool creation and
before ordering/randomization, remove every
`candidate_symbol == experimental_symbol`. Never restore it. Record observation
ID, experimental symbol, pre-count, removed count, post-count, selected symbol,
and infeasibility. Empty post-pool emits `C1_NO_DISTINCT_SYMBOL_CONTROL`.
Any assignment with equal symbols emits `C1_SELF_MATCH_DETECTED`. Calipers and
structural eligibility remain unchanged.

## 30. C2 Graph Construction

For each horizon, fold, UTC month, and replicate, left nodes are experimental
entries in observation-ID order and carry the exact experimental symbol
multiset. Right nodes are structurally eligible cycles ordered by symbol,
cycle timestamp, and cycle ID. An edge exists when the left symbol is eligible
for SHORT entry at the right cycle, lies in the same fold/month, has complete
horizon outcome and costs/funding, and passes every original structural rule.
No score, rank, label, or outcome value enters edge creation.

Rule `E5-R029`: each right cycle has capacity one. The graph must support every
left node to produce a valid replicate.

## 31. C2 Exact Self-Edge Prohibition

Rule `E5-R030` implements Amendment 04 S3/S4. Before canonical edge ordering and
randomization, remove exactly edges satisfying
`candidate_symbol == experimental_symbol AND candidate_cycle_id == experimental_cycle_id`.
Same-symbol, different-cycle edges remain when every other rule passes. Record
pre-edge count, removed self-edge count, post-edge count, assignments, and
unmatched reasons. No prohibited edge enters traversal. Empty/distinctness-
caused infeasibility emits `C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL`; a surviving
self-edge emits `C2_SELF_EDGE_DETECTED`.

## 32. Randomized Augmenting-Path Matching

Rule `E5-R031` implements D2-A. For each stratum/replicate, initialize its C2
PCG64 stream, independently permute each left node's canonically sorted
adjacency list, then run this fixed Kuhn augmenting-path algorithm:

1. iterate left nodes in canonical order;
2. reset visited-right set for that left node;
3. traverse its seeded adjacency order;
4. claim an unmatched right node, or recursively rematch its current left node;
5. use deterministic recursion order and stop at the first augmenting path;
6. continue until all left nodes are processed.

The objective is maximum cardinality only. Perfect cardinality is mandatory.
Right nodes are not reused within a replicate; candidates may recur across
replicates and horizons. Invalid replicates invoke D12; fewer than 9,500 valid
replicates is `C2_MATCHING_INFEASIBLE`. Greedy, cost, nearest-time, volatility,
outcome, partial, and post-hoc rematching substitutes are forbidden.

## 33. Primary Expectancy Procedure

The experimental unit is one frozen fixed-notional SHORT entry. Primary H12
expectancy is the equal-trade arithmetic mean of B_BASE net return. Pooled
metrics concatenate folds and weight each trade equally. Fold metrics use only
that fold. Control-ensemble expectancy is calculated per replicate; its mean is
the control reference and paired delta is experimental mean minus control mean.

Rule `E5-R032`: primary absolute success requires H12 point estimate at least
`0.0005` and week-block CI90 lower bound strictly above zero. Economic utility
also requires PF at least 1.10 with CI lower bound above 1.00, both control
deltas at least 0.0005 with lower bounds above zero, control p-values <=0.05,
pessimistic expectancy >=0, and original CVaR/drawdown/outlier gates. Undefined
mandatory metrics fail.

PF is sum positive returns divided by absolute sum negative returns; zero loss
denominator is invalid under D12. CVaR q is the arithmetic mean of observations
at or below the Type 7 q quantile. Drawdown is maximum peak-to-trough decline of
the chronological cumulative sum of fixed-notional return fractions.

For each control ensemble and horizon, the Monte Carlo p-value is
`(1 + count(control_replicate_expectancy >= experimental_expectancy)) /
(1 + N_valid)`. C1 requires all 10,000 requested replicate assignments for each
matched observation; C2 validity follows D12. The paired delta population is
the transparent horizon-specific matched population, and matching coverage over
all eligible experimental observations must remain at least 0.95.

All original complete metrics use the same canonical rows and these closed
definitions. Return quantiles are Type 7. Median is q0.50; sample standard
deviation uses denominator `n-1`; the week-cluster standard error of the
equal-trade mean is
`sqrt(W/(W-1) * sum_w(sum_i_in_w(r_i-mean_r))^2) / n` for `W>=2` and is
`NOT_COMPUTABLE` otherwise. Win rate is count(return>0)/n; zero is not a win.
Average win/loss condition on strict sign; payoff is average win divided by
absolute average loss; PF, CVaR, and drawdown follow the formulas above.
Turnover is `2 * fixed_notional * trade_count`, and total cost is the exact sum
of fixed execution cost plus signed realized funding.

Kendall tau-b uses concordant/discordant pair counts with standard tie
corrections. ROC AUC is the average-rank Mann-Whitney statistic; average
precision groups equal scores, then sums each group recall increment times
post-group precision. Directional class is gross return>0 and economic class is
B_BASE return>0; equality is negative. OLS slope uses an intercept and ordinary
unweighted least squares over the ten bin means. Undefined denominators are
reported, never replaced.

For outlier gate 22, sort experimental H12 B_BASE returns descending, then
observation ID, remove exactly `ceil(0.01*n)` experimental rows, and remove their
paired C1/C2 records without rematching. For gate 23, compute Type 7 P1 and P99
on experimental H12 B_BASE returns and clip experimental and already-paired
control returns to those same two boundaries without rematching. Leave-one-
symbol-out removes one canonical symbol and its paired rows, then recomputes all
pooled quantities without regenerating controls. Best-trade, best-1%, best-5%,
and best-symbol concentration divide the named positive-PnL contribution by
total positive PnL, using the same deterministic order and
`NO_POSITIVE_PNL` behavior as Section 41.

## 34. Fold-Centered Residual Power Simulation

Rule `E5-R033` implements D6-C. Use discovery entries only. In each discovery
fold compute trade-weighted mean `m_f` and residual `e_i=r_i-m_f`. Reconstruct
`r_i*=m_f+e_i+0.0005`. Resample complete ISO-week blocks with replacement within
their original fold, drawing the same number of blocks as observed, preserving
all trades. Concatenate folds with equal trade weight.

Each of 10,000 outer simulations reruns the complete primary expectancy point
and CI test. Its inner CI requests 10,000 bootstrap replicates with a seed
substream containing the outer index. Success requires estimate >=0.0005 and CI
lower >0. Estimated power is successes/valid outer simulations and must be
>=0.80. D12 requires >=9,500 valid outer simulations and >=9,500 valid inner
replicates per valid outer simulation. No global centering or control-delta
substitution.

## 35. Complete-Week Temporal Permutations

Rule `E5-R034` implements D7-C. Use only complete ISO weeks wholly contained in
each fold; exclude boundary-partial weeks from observed and null support. Each
fold needs at least four blocks. Order blocks chronologically. For each of
10,000 repetitions and fold, draw uniformly one shift in `[1,W-1]`, shared by
all symbols, and circularly shift outcome-week assignments while scores,
entries, and identities remain fixed. Repeated shifts are allowed. Unequal W by
fold is allowed. Never split a week or shuffle rows.

The original pooled-expectancy permutation and each D8 statistic use this same
operator. Fewer than 9,500 valid null values is
`PERMUTATION_VALIDITY_FAILURE`.

The original permutation statistic is pooled experimental B_BASE net
expectancy. Its one-sided p-value is
`(1 + count(null_expectancy >= observed_expectancy)) / (1 + N_valid)`, with
equality in the tail and D12 validity.

## 36. Spread Inference

Sort eligible candidates by score descending, then symbol ascending, then
timestamp ascending, exactly as frozen. For n rows, assign ordered index i
(zero based) to decile `10-floor(10*i/n)`. This creates ten counts differing by
at most one and follows the deterministic total order. Bottom is bin 1; top is
bin 10.
Spread is equal-trade mean B_BASE net return in bin 10 minus bin 1.

Rule `E5-R035`: confirmation spread tests use pooled F3-F4 complete-week support
at each horizon, D7 null recomputation, one-sided favorable direction, and
`p=(1+count(null>=observed))/(1+N_valid)`. Equality enters tail. Economic MREM is
0.0005. Artifact: `temporal_permutation_manifest.json`.

## 37. Monotonicity Inference

Using the same ten bins, compute bin mean B_BASE net return. Primary trend is
Spearman rho between bin index and bin mean with average ranks; report Kendall
tau-b, OLS slope, adjacent downward count/magnitude, and bin10-bin1 spread.

Rule `E5-R036`: confirmation monotonicity uses pooled F3-F4 complete-week
support, D7 null, one-sided rho>0, plus-one p-value, and D12 validity. Global
gate separately requires pooled all-fold rho>0, pooled spread >=0.0005, pooled
spread CI lower>0, and positive fold spread in at least three folds. Strict
monotonicity is not required.

## 38. Bootstrap Procedure

Rule `E5-R037` implements D12-B and the original dependence correction. Request
10,000 PCG64 replicates. Independently within each fold, sample UTC ISO-week
blocks with replacement, drawing the observed number of blocks. A fold-boundary
partial week remains a bootstrap block when it contains eligible observations;
D7 complete-week exclusion applies only to temporal permutation and the D6 power
simulation. Keep all cycles, symbols, entries, and paired controls in a selected
week together. Concatenate folds, weight trades equally, and preserve within-week
order for drawdown.

Replicate validity requires all inputs/intermediates, valid denominators, and a
finite statistic. Exclude invalid values and report reasons; do not redraw,
retry, reseed, or extend. At least 9,500 valid replicates and four source weeks
per mandatory fold statistic are required. CI90 is Type 7 q0.05/q0.95 over
finite values. Mandatory failure is gate FAIL; diagnostic failure is
`NOT_COMPUTABLE` only.

## 39. Multiplicity and Holm

The ordered confirmation registry is:

| Position | Test ID | Statistic/horizon | Raw p-value source |
|---:|---|---|---|
| 1 | `A_C1_H12` | real-C1 net delta H12 | C1 random-control ensemble |
| 2 | `A_C1_H48` | real-C1 net delta H48 | C1 random-control ensemble |
| 3 | `A_C1_H96` | real-C1 net delta H96 | C1 random-control ensemble |
| 4 | `A_C2_H12` | real-C2 net delta H12 | C2 random-control ensemble |
| 5 | `A_C2_H48` | real-C2 net delta H48 | C2 random-control ensemble |
| 6 | `A_C2_H96` | real-C2 net delta H96 | C2 random-control ensemble |
| 7 | `B_SPREAD_H12` | top-bottom net spread H12 | D7 permutation |
| 8 | `B_SPREAD_H48` | top-bottom net spread H48 | D7 permutation |
| 9 | `B_SPREAD_H96` | top-bottom net spread H96 | D7 permutation |
| 10 | `C_MONO_H12` | decile rho H12 | D7 permutation |
| 11 | `C_MONO_H48` | decile rho H48 | D7 permutation |
| 12 | `C_MONO_H96` | decile rho H96 | D7 permutation |

Rule `E5-R038`: all tests use pooled F3-F4 only and one-sided alpha 0.05. Sort
raw p ascending, tie by registry position. At ordered rank i=1..12, reject while
`p_i <= .05/(13-i)`; after first non-rejection, reject none later. Adjusted
p-values are cumulative maxima of `(13-i)*p_i`, capped at 1, restored to registry
order. Missing/nonfinite p remains in family and makes the family incomplete and
failed. Discovery and four-fold pooled p-values never enter.

A horizon is full statistical PASS only when its four tests reject. All three
horizons require favorable economic direction and governed MREM; at least two
horizons require full statistical PASS; H96 cannot be sole full pass. H12
original gates cannot be rescued.

## 40. Fold Predicates

Rule `E5-R039`: `FOLD_ECONOMIC_PASS(k)` uses H12, the original primary fold
scope, and requires all: >=100 eligible trades; C1 and C2 fold delta each >0 and
>=0.0005; real B_BASE expectancy >0; top-bottom spread >0 and >=0.0005; control
coverage >=0.95; and no data-integrity defect. Missing/undefined is FAIL.
Ordinary fold concentration, IC, pooled monotonicity, and Holm are excluded.

`FOLD_GATE_PASS = FOLD_ECONOMIC_PASS(3) AND FOLD_ECONOMIC_PASS(4) AND
COUNT_PASS(F1..F4)>=3`. Original stability additionally requires no fold
expectancy below -0.0005 and positive C1/C2 deltas in at least three folds.

## 41. Concentration

For each symbol, positive-PnL contribution is the sum of `max(B_BASE_H12,0)`;
denominator is that sum across all experimental trades. Pooled concentration is
maximum symbol contribution/denominator and must be <=0.30. Zero denominator
sets ratio 0 and flag `NO_POSITIVE_PNL`; other economic gates still apply.

Rule `E5-R040`: fold calculations use the same formula but are diagnostic.
Only duplicate contribution, corrupt symbol, inconsistent aggregation, or
impossible totals can fail a fold as integrity defects. Artifact:
`concentration_report.json`.

## 42. Label-to-Economics Classification

| Family | Field | Type | Favorable |
|---|---|---|---|
| TRRM | `tail_event` | binary | lower / 0 |
| EQM-clean | `clean_quality` | binary | higher / 1 |
| EQM-net | `net_quality_after_costs` | continuous | higher |
| QMAE | `qmae` | continuous | lower |

Rule `E5-R041`: aliases are not extra targets; auxiliary metadata never blocks.
H12 is primary; H48/H96 are diagnostics. Binary association is favorable-class
minus unfavorable-class mean B_BASE H12; ordered/continuous association is
Spearman with average ranks, oriented favorable. Connection requires favorable
pooled direction, correct ordering, and favorable direction in >=3 folds.
Materiality uses H12 economic difference >=0.0005.

Classify `LABEL_ECONOMICS_DISCONNECTED` when pooled direction is non-favorable,
ordering fails, or <3 folds favor; `LABEL_ECONOMICS_CONNECTED_EFFECT_TOO_SMALL`
when connected but difference <0.0005; otherwise
`LABEL_ECONOMICS_CONNECTED_MATERIAL`. Only disconnected mandatory target blocks
E6. Report prevalence, gross/net returns, barrier probabilities, fold/symbol
stability, and score association.

## 43. Diagnostic IC

Rule `E5-R042`: compute average-rank Spearman between frozen score and gross and
B_BASE net return at H12/H48/H96 for each symbol-fold, fold pooled, discovery
pooled, confirmation pooled, all-fold pooled, and each cycle. Require >=2 pairs
and nonconstant ranks; otherwise `IC_NOT_COMPUTABLE`. Cycle summaries are
unweighted count, mean, median, and fraction positive. D12 week-block bootstrap
provides CI where a temporal scope exists. IC has no MREM, significance count,
Holm membership, gate, blocking, or rescue authority.

## 44. Discovery State Machine

States are `DISCOVERY_NOT_STARTED`, `DISCOVERY_VALIDATING`,
`DISCOVERY_RUNNING`, `DISCOVERY_ARTIFACTS_PENDING`,
`DISCOVERY_VALIDATING_OUTPUTS`, `DISCOVERY_FROZEN`, `DISCOVERY_FAILED`.

Rule `E5-R043`: sequence is governance/spec/source verification; boundary and
guard verification; discovery input and funding validation; horizon population
and outcome construction; F1-F2 ATR freeze; controls/statistics/diagnostics;
power; artifact write; completeness validation; freeze seal. Failure retains
logs and can restart before confirmation only from identical inputs and hashes.
No confirmation rows, semi-blind, or lockbox are read.

## 45. Discovery Freeze

Rule `E5-R044`: seal governance/spec/source/software/input/boundary/guard
manifests; symbol registry; funding raw/normalized/manifest hashes; horizon and
exclusion manifests; ATR boundaries; seeds; algorithms; discovery results;
hypothesis and 12-test registries; schemas; confirmation plan/input identity;
lockbox-state record; and complete artifact hash DAG. Once confirmation starts,
no sealed byte changes. Failure: `DISCOVERY_FREEZE_MISMATCH`.

## 46. One-Shot Confirmation Ledger

Ledger schema contains experiment `E5`, confirmation run ID, status
`NOT_STARTED|STARTED|COMPLETED|FAILED`, six governance hashes, spec hash,
discovery-freeze hash, software/seed/confirmation-input/funding hashes, audit
start/completion timestamps, checkpoint hashes, final artifact hash, and failure
code.

Rule `E5-R045`: atomic compare-and-set permits one `NOT_STARTED -> STARTED`.
Continuation requires same run ID and all hashes. Second start, changed input,
seed, funding, code, or dependency is rejected. Ledger is not lockbox state and
never changes it.

## 47. Confirmation State Machine

States are `CONFIRMATION_NOT_STARTED`, `CONFIRMATION_VALIDATING_FREEZE`,
`CONFIRMATION_STARTED`, `CONFIRMATION_RUNNING`,
`CONFIRMATION_ARTIFACTS_PENDING`, `CONFIRMATION_VALIDATING_OUTPUTS`,
`CONFIRMATION_COMPLETED`, `CONFIRMATION_FAILED`.

Rule `E5-R046`: verify governance/spec/freeze/guard/lockbox/input/funding; create
run ID; mark STARTED; load only F3-F4; run all mandatory procedures; produce 12
p-values; Holm; fold, concentration, labels, diagnostics; verdict; seal; mark
COMPLETED or FAILED; verify lockbox unchanged. After STARTED, only idempotent
same-run resume from sealed checkpoints is allowed. No second or selective run.

Within that single ledger run, execute two predeclared deterministic lanes from
clean independent report roots with identical sealed dependencies and no metric
exposure between lanes. They share the same `confirmation_run_id` and are not
separate confirmation starts. Compare every canonical scientific byte before
verdict evaluation; write `run_manifest_attempt_1.json` and
`run_manifest_attempt_2.json`. Any difference is `DETERMINISM_FAILURE`, emits no
scientific verdict, and cannot authorize another run.

## 48. Final Verdict

Rule `E5-R047`: serialize this complete original gate registry with expected
value, actual value, pass/fail, and evidence path:

| Gate | Mandatory predicate |
|---:|---|
| 1 | input hashes and frozen E3 entry identities match |
| 2 | frozen labels reproduce within `1e-12` and no unauthorized H12 entry loss occurs |
| 3 | the two predeclared in-run deterministic lanes are byte-identical |
| 4 | power >=0.80 at MREM |
| 5 | H12 B_BASE expectancy >=0.0005 |
| 6 | H12 B_BASE expectancy CI90 lower >0 |
| 7 | B_BASE PF >=1.10 and PF CI90 lower >1.00 |
| 8 | H12 experimental-minus-C1 mean delta >=0.0005 |
| 9 | H12 experimental-minus-C2 mean delta >=0.0005 |
| 10 | both H12 control-delta CI90 lower bounds >0 |
| 11 | both H12 random-control Monte Carlo p-values <=0.05 |
| 12 | original pooled-expectancy week-block permutation p-value <=0.05 |
| 13 | pooled score monotonicity and top-bottom spread gates pass |
| 14 | top-bottom spread is positive in >=3 folds |
| 15 | experimental expectancy is positive in >=3 folds |
| 16 | C1 and C2 deltas are each positive in >=3 folds |
| 17 | no fold experimental expectancy < -0.0005 |
| 18 | all symbol-stability and leave-one-symbol-out gates pass |
| 19 | pooled positive-PnL symbol concentration <=0.30 |
| 20 | pessimistic-scenario expectancy >=0 |
| 21 | CVaR and drawdown gates pass against both controls |
| 22 | excluding best 1% retains positive expectancy and both positive control deltas |
| 23 | P1/P99 winsorized results retain positive expectancy and both positive control deltas |
| 24 | every fold has >=100 completed eligible experimental trades |
| 25 | no prohibited access, fitting, tuning, or protocol mutation occurred |

The later conjunctive requirements also pass: the complete 12-test Holm horizon
rule; `FOLD_ECONOMIC_PASS(3)`; `FOLD_ECONOMIC_PASS(4)`; at least three of four
folds pass; no disconnected mandatory target; and every amendment-preserved
mandatory rule. These requirements refine the listed gates and do not replace
one. Diagnostics including IC and ordinary fold concentration are excluded.

Any mandatory scientific failure after technically valid execution emits
`CLOSE_THIS_SIGNAL_FAMILY`. Mandatory `NOT_COMPUTABLE` is failure. Technical
inability before a valid execution emits no scientific verdict and records
`FINAL_VERDICT_NOT_COMPUTABLE`. No other verdict exists. E6_JUSTIFIED authorizes
only separate E6 preregistration.

## 49. Artifact Contracts

Canonical root is `reports/science/e5_signal_edge_control_test/`; the alias never
creates another root. Each JSON artifact uses Section 18 serialization and each
JSONL uses its declared field order. All are immutable after their sealing
stage.

| Artifact | Producer / consumer | Canonical relative path | Core schema/dependencies |
|---|---|---|---|
| `governance_manifest.json` | verifier/all | `governance/` | six paths, commits, hashes |
| `execution_spec_manifest.json` | verifier/all | `governance/` | spec path/hash/version |
| `source_state_manifest.json` | source verifier/all | `governance/` | both commits/clean states |
| `software_manifest.json` | Phase 0/all | `governance/` | runtime/package/BLAS/tzdata |
| `data_boundary_manifest.json` | boundary guard/loaders | `governance/` | classes A-D roots/IDs |
| `prohibited_data_guard_report.json` | guard/orchestrators | `governance/` | deny checks and lockbox state |
| `input_manifest.json` | loader/all | `inputs/` | all input hashes |
| `seed_manifest.json` | seed utility/random modules | `governance/` | KDF, vectors, streams |
| `symbol_registry_manifest.json` | loader/all | `inputs/` | eleven symbols/mapping/hash |
| `funding_raw_manifest.json` | funding transport/normalizer | `inputs/funding/raw/` | requests/raw hashes |
| `funding_rejection_report.jsonl` | funding normalizer/audit | `inputs/funding/` | every rejected raw record and failure code |
| `e5_funding_history_v1.jsonl` | normalizer/outcomes | `inputs/funding/` | Amendment 03 records |
| `e5_funding_history_v1_manifest.json` | normalizer/all | `inputs/funding/` | coverage/reconciliation/hash |
| `horizon_population_manifest_h12.json` | population/all | `populations/` | H12 IDs/exclusions/counts |
| `horizon_population_manifest_h48.json` | population/all | `populations/` | H48 IDs/exclusions/counts |
| `horizon_population_manifest_h96.json` | population/all | `populations/` | H96 IDs/exclusions/counts |
| `exclusion_manifest.jsonl` | pipeline/audit | `populations/` | primary/secondary reasons |
| `atr_boundaries.json` | quintile freezer/confirmation | `discovery/` | Type 7 boundaries/input hash |
| `c1_matching_manifest.jsonl` | C1/statistics | `controls/` | assignments/self-filter counts |
| `c2_matching_manifest.jsonl` | C2/statistics | `controls/` | graph/matches/self-edge counts |
| `power_manifest.json` | power/verdict | `discovery/` | residuals/seeds/validity/power |
| `bootstrap_manifest.json` | bootstrap/statistics | `statistics/` | blocks/seeds/validity/CI |
| `temporal_permutation_manifest.json` | permutation/Holm | `statistics/` | weeks/shifts/nulls/p-values |
| `discovery_results.json` | discovery/freeze | `discovery/` | authorized metrics only |
| `discovery_freeze_manifest.json` | freeze/confirmation | `discovery/` | sealed dependency DAG |
| `e5_confirmation_ledger.json` | ledger/confirmation | `confirmation/` | Section 46 state |
| `confirmation_results.json` | confirmation/verdict | `confirmation/` | all mandatory metrics |
| `holm_registry.json` | Holm/verdict | `confirmation/` | twelve tests/raw/adjusted/pass |
| `label_economics_registry.json` | labels/verdict | `diagnostics/` | four targets/classes |
| `ic_report.json` | IC/audit | `diagnostics/` | all scopes/NOT_COMPUTABLE |
| `concentration_report.json` | concentration/verdict | `diagnostics/` | pooled gate/fold diagnostics |
| `final_verdict.json` | verdict/audit | `verdict/` | every gate and one verdict |
| `artifact_hash_manifest.json` | hash verifier/audit | root | full lexical dependency DAG |

The original Section 15 filenames remain mandatory deterministic compatibility
artifacts. They are immutable projections of the canonical artifacts above and
cannot contain a different scientific value:

| Governed artifact | Producer / consumer | Canonical relative path | Canonical dependency |
|---|---|---|---|
| `preregistration.md` | verifier/audit | `governance/original/` | exact original preregistration bytes |
| `run_manifest_attempt_1.json` | confirmation/audit | `confirmation/lane_1/` | first in-run deterministic lane |
| `run_manifest_attempt_2.json` | confirmation/audit | `confirmation/lane_2/` | second in-run deterministic lane |
| `environment_manifest.json` | Phase 0/audit | `governance/` | byte-identical content projection of software manifest |
| `input_manifest.json` | loader/all | `inputs/` | canonical input manifest above |
| `entry_set_manifest.json` | population/audit | `populations/` | frozen entry identity projection |
| `eligible_population_manifest.json` | population/audit | `populations/` | three horizon manifests and exclusions |
| `score_manifest.json` | loader/score modules | `inputs/` | score schema/hash/identity projection |
| `label_manifest.json` | loader/label modules | `inputs/` | target and recomputation projection |
| `control_definition.json` | matchers/audit | `controls/` | C1/C2 rules, seeds, self filters |
| `control_replicates.parquet` | matchers/statistics | `controls/` | deterministic projection of C1/C2 JSONL |
| `permutation_manifest.json` | permutation/audit | `statistics/` | temporal permutation manifest projection |
| `permutation_results.parquet` | permutation/Holm | `statistics/` | deterministic null-statistic projection |
| `experimental_trades.parquet` | outcomes/statistics | `outcomes/` | deterministic outcome-row projection |
| `score_metrics.json` | score evaluator/audit | `metrics/` | correlations/AUC/AP/lift |
| `decile_metrics.json` | score evaluator/audit | `metrics/` | bin counts/returns/spreads/monotonicity |
| `economic_metrics.json` | expectancy/audit | `metrics/` | pooled/fold/symbol return economics |
| `fold_metrics.json` | expectancy/verdict | `metrics/` | four fold metric projections |
| `symbol_metrics.json` | expectancy/verdict | `metrics/` | eleven symbol metric projections |
| `stability_metrics.json` | stability/verdict | `metrics/` | fold/symbol/leave-one-out results |
| `outlier_sensitivity.json` | sensitivity/verdict | `metrics/` | governed exclusion/winsorized results |
| `power_report.json` | power/verdict | `discovery/` | power manifest projection |
| `decision_gates.json` | verdict/audit | `verdict/` | all original and amended predicates |
| `scientific_summary.json` | verdict/audit | `verdict/` | machine-readable result projection |
| `scientific_summary.md` | verdict/audit | `verdict/` | deterministic human-readable projection |
| `scientific_aggregate.json` | hash verifier/audit | root | aggregate scientific hash and dependencies |
| `determinism_report.json` | Phase 0/confirmation | `validation/` | synthetic and two-lane comparisons |

Rule `E5-R048`: every artifact declares schema version, producer, consumer,
stage, relative path, dependencies, SHA-256, mutability, and validation result.
C1/C2 invariants are mandatory. No lockbox transaction/query/budget artifact is
permitted.

## 50. Hashing, Checkpointing, and Resume

Rule `E5-R049`: SHA-256 hashes exact bytes. Tree manifests use POSIX relative
paths in UTF-8 lexical order. A manifest omits its own hash; a sidecar or parent
stores it. Dependency edges are acyclic and include source commits,
configuration, schema, software, and input hashes.

Before confirmation STARTED, deterministic rebuild from identical authorized
inputs is allowed. After STARTED, checkpoints belong to the confirmation run ID
and contain stage, dependency hashes, code/spec/seed/input/funding hashes,
artifact hashes, and next state. Resume verifies every byte and continues only
the same run. Changed or incomplete checkpoint is
`CONFIRMATION_RESUME_INVALID`; no recomputation with changed conditions.

## 51. Failure Taxonomy

All rows have lockbox consequence `NO_CHANGE`; a mutation attempt additionally
records the unauthorized attempt. `TECH` emits no verdict until execution is
valid; `SCI` fails the named mandatory gate after valid execution; `DIAG` only
records `NOT_COMPUTABLE`. Retry is `PRE` only before confirmation STARTED with
identical dependencies, `SAME` only same-run idempotent resume, or `NO`.
Severity is blocking technical for `TECH`, mandatory scientific-gate failure for
`SCI`, and nonblocking diagnostic for `DIAG`. Checkpoint consequence is:
`TECH` invalidates the triggering stage checkpoint; `SCI` seals the failed
mandatory result; `DIAG` seals the diagnostic reason. After ledger STARTED,
every `TECH` row marks that same ledger run FAILED unless its row says
unchanged; no failure creates a second run. Thus each row below fully determines
trigger, stage, severity, retry, checkpoint, ledger, verdict, artifact, and the
common lockbox consequence.

| Code | Stage/trigger | Class/retry | Ledger/verdict | Required artifact |
|---|---|---|---|---|
| `GOVERNANCE_HASH_MISMATCH` | governance bytes differ | TECH/NO | unchanged/none | governance manifest |
| `GOVERNANCE_COMMIT_MISMATCH` | commit absent/order wrong | TECH/NO | unchanged/none | governance manifest |
| `GOVERNANCE_CONFLICT` | unresolved binding conflict | TECH/NO | unchanged/none | conflict report |
| `DIRTY_WORKTREE` | either tree dirty | TECH/PRE | unchanged/none | source-state manifest |
| `SPECIFICATION_ALREADY_EXISTS` | duplicate spec path | TECH/NO | unchanged/none | source-state manifest |
| `INPUT_SCHEMA_MISMATCH` | required schema/field invalid | TECH/PRE | unchanged/none | input manifest |
| `INPUT_HASH_MISMATCH` | input hash differs | TECH/NO | fail if started/none | input manifest |
| `SOFTWARE_HASH_MISMATCH` | environment differs | TECH/NO | fail if started/none | software manifest |
| `DEVELOPMENT_DATA_BOUNDARY_MISMATCH` | class A/B mismatch | TECH/NO | fail if started/none | boundary report |
| `DATASET_UNAUTHORIZED` | dataset outside authorized manifest | TECH/NO | fail if started/none | exclusion manifest |
| `PARTITION_UNAUTHORIZED` | row belongs to wrong A/B stage | TECH/NO | fail if started/none | exclusion manifest |
| `ENTRY_NOT_FROZEN` | entry absent from frozen E3 set | TECH/NO | fail if started/none | exclusion manifest |
| `SYMBOL_UNAUTHORIZED` | symbol outside frozen registry | TECH/NO | fail if started/none | exclusion manifest |
| `IDENTITY_INVALID` | identity source/normalization invalid | TECH/NO | fail if started/none | exclusion manifest |
| `PROHIBITED_DATA_REFERENCE` | manifest references C/D | TECH/NO | fail/none | guard report |
| `SEMIBLIND_ACCESS_ATTEMPT` | C open attempted | TECH/NO | fail/none | guard report |
| `LOCKBOX_ACCESS_ATTEMPT` | D scientific read attempted | TECH/NO | fail/none | guard report |
| `LOCKBOX_MUTATION_ATTEMPT` | lockbox write attempted | TECH/NO | fail/none | guard report |
| `DUPLICATE_IDENTITY` | unequal rows share ID | TECH/NO | fail if started/none | exclusion manifest |
| `FOLD_INVALID` | fold absent or timestamp outside literal fold | TECH/NO | fail if started/none | exclusion manifest |
| `TIMESTAMP_INVALID` | timestamp lacks canonical UTC alignment | TECH/NO | fail if started/none | exclusion manifest |
| `TIME_ALIGNMENT_FAILURE` | UTC/bar/fold mismatch | TECH/PRE | fail if started/none | exclusion manifest |
| `BAR_UNAVAILABLE` | required canonical bar absent | SCI/NO | unchanged/close when mandatory | exclusion manifest |
| `HORIZON_INCOMPLETE` | complete H path absent | SCI/NO | unchanged/close when mandatory | exclusion manifest |
| `HORIZON_NOT_COMPUTABLE` | H complete case absent | SCI/NO | unchanged/close when mandatory | population manifest |
| `BARRIER_NOT_COMPUTABLE` | mandatory H96 path event undefined | DIAG/NO | unchanged/no independent effect | exclusion manifest |
| `COST_NOT_COMPUTABLE` | frozen cost cannot be evaluated | SCI/NO | unchanged/close when mandatory | exclusion manifest |
| `OUTCOME_NOT_COMPUTABLE` | outcome input missing | SCI/NO | unchanged/close when mandatory | confirmation results |
| `ATR_NOT_COMPUTABLE` | warm-up/gap failure | DIAG/NO | unchanged/no independent effect | exclusion manifest |
| `FUNDING_SOURCE_UNAUTHORIZED` | provider not official | TECH/NO | fail/none | funding manifest |
| `FUNDING_SOURCE_INCOMPLETE` | coverage unproved | TECH/NO | fail/none | funding manifest |
| `FUNDING_SOURCE_HASH_MISMATCH` | raw hash mismatch | TECH/NO | fail/none | raw manifest |
| `FUNDING_PROVIDER_MISMATCH` | provider literal differs | TECH/NO | fail/none | funding manifest |
| `FUNDING_SYMBOL_UNAUTHORIZED` | symbol outside eleven | TECH/NO | fail/none | funding manifest |
| `FUNDING_SYMBOL_MAPPING_MISMATCH` | contract map differs | TECH/NO | fail/none | symbol registry |
| `FUNDING_TIMESTAMP_INVALID` | event time invalid | TECH/NO | fail/none | rejection report |
| `FUNDING_RATE_INVALID` | decimal invalid | TECH/NO | fail/none | rejection report |
| `FUNDING_DUPLICATE_NATURAL_KEY` | duplicate symbol/time | TECH/NO | fail/none | rejection report |
| `FUNDING_DUPLICATE_RECORD_ID` | duplicate ID | TECH/NO | fail/none | rejection report |
| `FUNDING_CONFLICTING_RATE` | same key/different rate | TECH/NO | fail/none | rejection report |
| `FUNDING_RECORD_ID_MISMATCH` | recomputation differs | TECH/NO | fail/none | rejection report |
| `FUNDING_ORDER_INVALID` | JSONL order wrong | TECH/NO | fail/none | funding manifest |
| `FUNDING_RAW_NORMALIZED_RECONCILIATION_FAILURE` | counts/records differ | TECH/NO | fail/none | funding manifest |
| `FUNDING_MANIFEST_INVALID` | manifest incomplete | TECH/NO | fail/none | funding manifest |
| `FUNDING_SCHEMA_VERSION_MISMATCH` | version differs | TECH/NO | fail/none | funding manifest |
| `FUNDING_NOT_COMPUTABLE` | required interval uncovered | SCI/NO | unchanged/close when mandatory | outcome result |
| `C1_NO_DISTINCT_SYMBOL_CONTROL` | C1 post-filter empty | SCI/NO | unchanged/close through coverage | C1 manifest |
| `C1_SELF_MATCH_DETECTED` | equal-symbol assignment | TECH/NO | fail/none | C1 manifest |
| `C1_MATCHING_INFEASIBLE` | <9500 complete replicates | SCI/NO | unchanged/close | C1 manifest |
| `C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL` | self-edge removal causes infeasibility | SCI/NO | unchanged/close through validity | C2 manifest |
| `C2_SELF_EDGE_DETECTED` | exact pair survives | TECH/NO | fail/none | C2 manifest |
| `C2_MATCHING_INFEASIBLE` | <9500 perfect replicates | SCI/NO | unchanged/close | C2 manifest |
| `CONTROL_INELIGIBLE` | candidate fails structural control rules | SCI/NO | unchanged/coverage denominator | matching manifest |
| `E5_EXECUTION_BLOCKED_BY_CONTROL_COVERAGE` | matched coverage <0.95 | SCI/NO | unchanged/close | matching manifests |
| `INSUFFICIENT_COMPLETE_WEEKS` | mandatory fold has <4 weeks | SCI/NO | unchanged/close | permutation manifest |
| `BOOTSTRAP_VALIDITY_FAILURE` | <9500 finite replicates | SCI/NO | unchanged/close when mandatory | bootstrap manifest |
| `PERMUTATION_VALIDITY_FAILURE` | <9500 finite nulls | SCI/NO | unchanged/close | permutation manifest |
| `POWER_NOT_COMPUTABLE` | valid outer/inner threshold fails | SCI/NO | unchanged/close | power manifest |
| `SCORE_MISSING` | frozen score absent/nonfinite | TECH/NO | fail if started/none | exclusion manifest |
| `TARGET_MISSING` | mandatory frozen target absent | TECH/NO | fail if started/none | exclusion manifest |
| `LABEL_SCHEMA_AMBIGUITY` | target registry mismatch | TECH/NO | fail/none | label registry |
| `LABEL_ECONOMICS_DISCONNECTED` | mandatory target disconnected | SCI/NO | unchanged/close | label registry |
| `IC_NOT_COMPUTABLE` | rank statistic undefined | DIAG/NO | unchanged/no effect | IC report |
| `CONCENTRATION_NOT_COMPUTABLE` | pooled accounting invalid | SCI/NO | unchanged/close | concentration report |
| `INTEGRITY_QUARANTINE` | duplicate/corrupt/impossible accounting | TECH/NO | fail if started/none | exclusion manifest |
| `HOLM_FAMILY_INCOMPLETE` | any of 12 missing/nonfinite | SCI/NO | unchanged/close | Holm registry |
| `DISCOVERY_FREEZE_MISMATCH` | sealed byte differs | TECH/NO | unchanged/none | freeze manifest |
| `CONFIRMATION_ALREADY_STARTED` | second start | TECH/NO | unchanged/none | ledger |
| `CONFIRMATION_DEPENDENCY_MISMATCH` | started dependency differs | TECH/NO | fail/none | ledger |
| `CONFIRMATION_RESUME_INVALID` | checkpoint invalid | TECH/NO | fail/none | ledger |
| `ARTIFACT_HASH_MISMATCH` | artifact byte differs | TECH/NO | fail if started/none | hash manifest |
| `DETERMINISM_FAILURE` | repeated bytes differ | TECH/NO | fail/none | determinism report |
| `UNAUTHORIZED_SCIENTIFIC_CHOICE` | untraced choice/default | TECH/NO | fail/none | traceability report |
| `FINAL_VERDICT_NOT_COMPUTABLE` | technical execution invalid | TECH/NO | fail/no scientific verdict | final verdict artifact |

## 52. Determinism Validation

Rule `E5-R050`: synthetic tests only. Required vectors are:

1. six governance hashes and ancestry;
2. every canonical identity and collision rejection;
3. shuffled input order invariance;
4. byte-identical full reruns;
5. UTC/bar/horizon/fold/month/week boundaries;
6. D1 independent populations;
7. SHORT return and fixed costs;
8. same-bar adverse-first barrier;
9. Wilder initialization, recurrence, gap reset;
10. Type 7 boundaries, ties, duplicate boundaries;
11. four seed digests and PCG64 replay;
12. C1 only-self failure;
13. C1 self plus distinct selection;
14. C2 only exact-pair failure;
15. C2 exact plus same-symbol/different-cycle eligibility;
16. C2 exact plus different-symbol eligibility;
17. C2 shuffled-source byte identity;
18. prohibited edge rejected before traversal;
19. augmenting-path perfect/infeasible graphs;
20. fold-centered residual reconstruction;
21. nested power stream isolation;
22. complete-week shared circular shifts;
23. spread decile remainder/tie assignment;
24. monotonicity average-rank rho;
25. bootstrap Type 7 and 9,500 threshold;
26. Holm ties/missing member;
27. pooled concentration and zero-positive-PnL;
28. D10 complete truth table;
29. IC constant/defined cases;
30. funding decimal canonicalization and record ID;
31. funding duplicate/order/reconciliation rejection;
32. funding `(entry,termination]` boundaries;
33. zero events under complete coverage;
34. incomplete funding coverage;
35. one-shot start and second-start rejection;
36. same-run resume and changed-dependency rejection;
37. prohibited-data path/reference/open attempts; and
38. unchanged lockbox state before/after every failure.

## 53. Phase 0 Contract

Phase 0 emits exactly `E5_PHASE_0_PASS` or `E5_PHASE_0_BLOCKED` and consumes no
scientific confirmation row, funding row, semi-blind row, lockbox row, or query.

Rule `E5-R051`: verify six governance hashes/commits/order; spec hash; both clean
repositories; source/software versions; schemas; A-D boundaries; symbol map;
fixed notional; funding source/schema/identity/decimal/manifest; all identities;
seed vectors; ATR/Type7/C1/C2/permutation/bootstrap/Holm/label synthetic vectors;
artifact schemas; self-filter ordering; same-symbol different-cycle C2 behavior;
one-shot ledger; prohibited-data guard; unchanged lockbox; no confirmation start
or result; full traceability; and no unauthorized default. Any failure blocks.

## 54. Implementation Acceptance Criteria

Implementation is complete only when every `E5-R*` rule has one Python owner,
one validator, passing synthetic tests, implemented artifact schema, and mapped
failure; all modules and inputs are hash-bound; deterministic runs are
byte-identical; no scientific configuration is mutable; no placeholder/TODO or
fallback changes behavior; Phase 0 passes; and lockbox remains untouched.

Rule `E5-R052`: acceptance authorizes only separately instructed E5 execution,
not automatic discovery, confirmation, lockbox, or operations.

## 55. Prohibited Implementation Behavior

Rule `E5-R053` prohibits: confirmation reads in discovery; semi-blind/lockbox
reads; lockbox writes/queries/budget changes; lockbox-shaped E5 artifacts;
`FundingSnapshot`, `funding_bps_per_hour`, third-party funding, continuous
funding, interpolation, proration, incomplete-coverage zero, dynamic funding
notional, leverage multiplier, silent funding deduplication; C1 same-symbol;
C2 exact self-edge; self filters after randomization; greedy/outcome rematching;
seed changes; selective invalid-replicate replacement; preferred-result retry;
second confirmation; changed inputs after STARTED; Holm member changes; alpha,
MREM, folds, horizons, populations, D10 registry, favorable directions, or 9,500
threshold changes; return imputation; shortened horizons; future/incomplete ATR;
gate rounding; diagnostics promoted to gates; and artifact cherry-picking.

## 56. Traceability Matrix, Scientific-Gap Audit, and Completeness

| Trace | Governance clause | Decision | Spec rules/modules | Status |
|---|---|---|---|---|
| T-001 | P Governance/data/lockbox | - | R001-R010/verifier+guards | FULLY_SPECIFIED |
| T-002 | P §1 question | - | R004,R047/verdict | FULLY_SPECIFIED |
| T-003 | P §2 H0/H1 | - | R032,R047/verdict | FULLY_SPECIFIED |
| T-004 | P §3 edge/alpha/beta | - | R032-R038/power+Holm | FULLY_SPECIFIED |
| T-005 | P §3 controls/seeds | - | R016,R027-R031/matchers | FULLY_SPECIFIED |
| T-006 | P §3 utility/MREM | - | R021-R023,R032/verdict | FULLY_SPECIFIED |
| T-007 | P §4 data/evaluation | - | R006-R009,R020-R023 | FULLY_SPECIFIED |
| T-008 | P §5 units/dependence | - | R013,R033-R037 | FULLY_SPECIFIED |
| T-009 | P §6 experimental arm | - | R020-R023 | FULLY_SPECIFIED |
| T-010 | P §6 C1 | - | R027-R028 | FULLY_SPECIFIED |
| T-011 | P §6 C2 | D2-A | R029-R031 | FULLY_SPECIFIED |
| T-012 | P §6 control comparison | - | R032,R038 | FULLY_SPECIFIED |
| T-013 | P §7 permutation | D7-C | R034-R036 | FULLY_SPECIFIED |
| T-014 | P §8 bootstrap | D12-B | R037 | FULLY_SPECIFIED |
| T-015 | P §9 score evaluation | - | R035-R036,R042 | FULLY_SPECIFIED |
| T-016 | P §10 label integrity | - | R018-R021,R041 | FULLY_SPECIFIED |
| T-017 | P §11 monotonicity | - | R035-R036 | FULLY_SPECIFIED |
| T-018 | P §12 fold stability | - | R039 | FULLY_SPECIFIED |
| T-019 | P §12 symbol stability | - | R040,R047 | FULLY_SPECIFIED |
| T-020 | P §13 economic utility | - | R021-R023,R032,R047 | FULLY_SPECIFIED |
| T-021 | P §14 complete metrics | - | R032,R035-R043 | FULLY_SPECIFIED |
| T-022 | P §15 artifacts | - | R048-R049 | ENGINEERING_DETAIL_FROZEN |
| T-023 | P §16 tests/determinism | - | R050-R051 | ENGINEERING_DETAIL_FROZEN |
| T-024 | P §17 threats/bias/leakage | - | R006-R010,R053 | FULLY_SPECIFIED |
| T-025 | P §18 gates | - | R032,R036-R041,R047 | FULLY_SPECIFIED |
| T-026 | P §19 outcomes | - | R047 | FULLY_SPECIFIED |
| T-027 | P §20 no reinterpretation | - | R004,R047,R053 | FULLY_SPECIFIED |
| T-028 | P §21 safety | - | R005-R010,R053 | FULLY_SPECIFIED |
| T-029 | Patch §1 identity | - | R001,R013 | FULLY_SPECIFIED |
| T-030 | Patch §2 discovery/confirmation | - | R007-R008,R043-R046 | FULLY_SPECIFIED |
| T-031 | Patch §3 fold conjunction | - | R039,R047 | FULLY_SPECIFIED |
| T-032 | Patch §4 final logic | - | R047 | FULLY_SPECIFIED |
| T-033 | Patch §5 horizons | D1-B | R014,R020-R021 | FULLY_SPECIFIED |
| T-034 | Patch §6 ATR buckets | D4-A,D5-A | R025-R026 | FULLY_SPECIFIED |
| T-035 | Patch §7 barrier | D3-B | R021,R024 | FULLY_SPECIFIED |
| T-036 | Patch §8 coverage | - | R027-R032,R039 | FULLY_SPECIFIED |
| T-037 | Patch §9 custody | - | R044-R046 | FULLY_SPECIFIED |
| T-038 | Patch §10 governance | - | R001-R004 | NOT_APPLICABLE |
| T-039 | A1 Decision 1 IC MREM | D11-A | R042 | FULLY_SPECIFIED |
| T-040 | A1 Decision 2 IC diagnostic | D11-A | R042 | FULLY_SPECIFIED |
| T-041 | A1 Decision 3 sample size | - | R039,R042 | FULLY_SPECIFIED |
| T-042 | A1 Decision 4 monotonicity | - | R036 | FULLY_SPECIFIED |
| T-043 | A1 Decision 5 fold predicate | - | R039 | FULLY_SPECIFIED |
| T-044 | A1 Decision 6 test family | - | R038 | FULLY_SPECIFIED |
| T-045 | A1 Decision 7 success | - | R038-R047 | FULLY_SPECIFIED |
| T-046 | A1 Decision 8 horizons | - | R020,R038-R039 | FULLY_SPECIFIED |
| T-047 | A1 Decision 9 label module | D10-A | R041 | FULLY_SPECIFIED |
| T-048 | A1 Decision 10 diagnostics | - | R040-R042,R047 | FULLY_SPECIFIED |
| T-049 | A2 D1 | D1-B | R020 | FULLY_SPECIFIED |
| T-050 | A2 D2 | D2-A | R029-R031 | FULLY_SPECIFIED |
| T-051 | A2 D3 | D3-B | R021-R024 | FULLY_SPECIFIED |
| T-052 | A2 D4 | D4-A | R025 | FULLY_SPECIFIED |
| T-053 | A2 D5 | D5-A | R026 | FULLY_SPECIFIED |
| T-054 | A2 D6 | D6-C | R033 | FULLY_SPECIFIED |
| T-055 | A2 D7 | D7-C | R034 | FULLY_SPECIFIED |
| T-056 | A2 D8 | D8-A | R035-R036 | FULLY_SPECIFIED |
| T-057 | A2 D9 | D9-D | R040 | FULLY_SPECIFIED |
| T-058 | A2 D10 | D10-A | R041 | FULLY_SPECIFIED |
| T-059 | A2 D11 | D11-A | R042 | FULLY_SPECIFIED |
| T-060 | A2 D12 | D12-B | R037 | FULLY_SPECIFIED |
| T-061 | A2 custody/non-changes | - | R043-R047,R053 | FULLY_SPECIFIED |
| T-062 | A3 F1 source | - | R023 | FULLY_SPECIFIED |
| T-063 | A3 F2 event semantics | - | R014,R023 | FULLY_SPECIFIED |
| T-064 | A3 F3-F4 schema/units | - | R015,R017,R023 | FULLY_SPECIFIED |
| T-065 | A3 F5-F6 sign/notional | - | R021,R023 | FULLY_SPECIFIED |
| T-066 | A3 F7-F8 identity/order | - | R013,R017,R023 | FULLY_SPECIFIED |
| T-067 | A3 F9 completeness | - | R023 | FULLY_SPECIFIED |
| T-068 | A3 F10-F12 artifacts/decimal | - | R015,R017,R023,R048 | FULLY_SPECIFIED |
| T-069 | A3 F13-F14 validation/failures | - | R023,R051 | FULLY_SPECIFIED |
| T-070 | A3 D3/Phase0 relationships | D3-B | R023-R024,R051 | FULLY_SPECIFIED |
| T-071 | A4 S1 general prohibition | - | R028,R030 | FULLY_SPECIFIED |
| T-072 | A4 S2 C1 | - | R027-R028 | FULLY_SPECIFIED |
| T-073 | A4 S3 C2 | - | R029-R031 | FULLY_SPECIFIED |
| T-074 | A4 S4 filtering order | - | R028,R030 | FULLY_SPECIFIED |
| T-075 | A4 S5 identities | - | R013,R028,R030 | FULLY_SPECIFIED |
| T-076 | A4 S6 coverage | - | R027-R032,R039 | FULLY_SPECIFIED |
| T-077 | A4 S7 audit | - | R028,R030,R048 | FULLY_SPECIFIED |
| T-078 | A4 S8 validation | - | R050-R051 | FULLY_SPECIFIED |

### Authoritative Source Locators

Trace labels resolve to these exact immutable file lines. `P` is
`e5_protocol_preregistration.md`: Governance 3-20; §1 22-28; §2 29-56; §3
57-125; §4 126-151; §5 152-163; §6 164-209; §7 210-230; §8 231-248; §9
249-268; §10 269-285; §11 286-304; §12 305-330; §13 331-349; §14 350-381;
§15 382-415; §16 416-459; §17 460-508; §18 509-541; §19 542-564; §20
565-590; §21 591-end.

`Patch` is `e5_protocol_patch_02.md`: authority 3-16; §1 17-24; §2 25-44;
§3 45-74; §4 75-94; §5 95-108; §6 109-125; §7 126-142; §8 143-160; §9
161-183; §10 184-191; §11 192-210; §12 211-end.

`A1` is `e5_owner_authorized_amendment_01.md`: scope 3-30; D1 31-45; D2
46-62; D3 63-75; D4 76-87; D5 88-112; D6 113-136; D7 137-154; D8
155-169; D9 170-196; D10 197-204; final space 205-226; validation 227-243;
record 244-end.

`A2` is `e5_owner_authorized_amendment_02.md`: status/hierarchy/safety 3-55;
selection registry 56-72; D1 73-100; D2 101-130; D3 131-153; D4 154-175;
D5 176-191; D6 192-209; D7 210-233; D8 234-253; D9 254-275; D10
276-317; D11 318-335; D12 336-360; consistency/non-changes/determinism
361-end.

`A3` is `e5_owner_authorized_amendment_03.md`: authority/safety/scope 3-101;
F1 104-127; F2 128-143; F3 144-161; F4 162-177; F5 178-191; F6 192-209;
F7 210-227; F8 228-240; F9 241-258; F10 259-274; F11 275-290; F12
291-306; F13 307-333; F14 334-359; schema/relationships/validation/non-changes
360-end.

`A4` is `e5_owner_authorized_amendment_04.md`: authority/safety/gap 3-97; S1
98-111; S2 112-134; S3 135-163; S4 164-190; S5 191-214; S6 215-227; S7
228-258; S8 259-281; relationships/non-changes/completeness 282-end.

### Rule Metadata Index

The authority column points to trace rows above, which resolve through the
exact source locators. Validation owner is the named module's synthetic suite
plus Phase 0 unless stated otherwise.

| Rule | Authority | Implementation / validation owner | Required artifact | Failure |
|---|---|---|---|---|
| R001 | T-001,T-027,T-038 | governance verifier / Phase 0 | governance manifest | `UNAUTHORIZED_SCIENTIFIC_CHOICE` |
| R002 | T-001,T-029,T-038 | governance verifier / Phase 0 | governance manifest | governance hash/commit/conflict |
| R003 | T-022,T-037 | hash verifier / Phase 0 | execution-spec manifest | `ARTIFACT_HASH_MISMATCH` |
| R004 | T-027,T-061 | governance verifier / trace validator | traceability report | `UNAUTHORIZED_SCIENTIFIC_CHOICE` |
| R005 | T-028,T-061 | CLI orchestrator / boundary tests | source-state manifest | `UNAUTHORIZED_SCIENTIFIC_CHOICE` |
| R006 | T-001,T-007,T-024 | data-boundary guard / guard tests | data-boundary manifest | boundary/prohibited-data failures |
| R007 | T-007,T-030 | discovery orchestrator / state tests | prohibited-data guard report | `SEMIBLIND_ACCESS_ATTEMPT` |
| R008 | T-007,T-030 | confirmation orchestrator / state tests | confirmation input manifest | confirmation dependency failure |
| R009 | T-001,T-024,T-028 | prohibited-data guard / guard tests | prohibited-data guard report | semi-blind/lockbox access failures |
| R010 | T-001,T-028,T-061 | prohibited-data guard / lockbox-state tests | prohibited-data guard report | lockbox access/mutation failures |
| R011 | T-023 | every module / Phase 0 | determinism report | `DETERMINISM_FAILURE` |
| R012 | T-007,T-021 | input loader / schema tests | input manifest | `INPUT_SCHEMA_MISMATCH` |
| R013 | T-008,T-029,T-066,T-075 | identity builder / identity vectors | input manifest | `DUPLICATE_IDENTITY` |
| R014 | T-008,T-033,T-063 | time utility / boundary vectors | input manifest | `TIME_ALIGNMENT_FAILURE` |
| R015 | T-004,T-014,T-034,T-064 | decimal/statistics utilities / numeric vectors | software manifest | numeric or funding validation failure |
| R016 | T-005,T-050,T-054,T-055 | seed utility / seed vectors | seed manifest | `DETERMINISM_FAILURE` |
| R017 | T-022,T-064,T-066,T-068 | serializer / byte vectors | artifact hash manifest | `ARTIFACT_HASH_MISMATCH` |
| R018 | T-007,T-016 | population builder / pipeline tests | exclusion manifest | input/horizon failures |
| R019 | T-007,T-036 | population builder / reconciliation tests | exclusion manifest | `INPUT_SCHEMA_MISMATCH` |
| R020 | T-007,T-009,T-033,T-049 | population builder / D1 vectors | horizon population manifests | `HORIZON_NOT_COMPUTABLE` |
| R021 | T-006,T-009,T-020,T-051,T-065 | outcome engine / outcome vectors | confirmation results | `OUTCOME_NOT_COMPUTABLE` |
| R022 | T-006,T-020 | fixed-cost engine / parity vectors | confirmation results | `INPUT_SCHEMA_MISMATCH` |
| R023 | T-051,T-062-T-070 | funding modules / funding vectors | funding manifests and JSONL | funding failure taxonomy |
| R024 | T-035,T-051,T-070 | outcome engine / barrier vectors | confirmation results | `OUTCOME_NOT_COMPUTABLE` |
| R025 | T-034,T-052 | ATR engine / Wilder vectors | exclusion manifest | `ATR_NOT_COMPUTABLE` |
| R026 | T-034,T-053 | quintile freezer / Type 7 vectors | ATR boundaries | `ARTIFACT_HASH_MISMATCH` |
| R027 | T-010,T-036,T-072,T-076 | C1 matcher / coverage vectors | C1 matching manifest | C1 infeasibility failures |
| R028 | T-071,T-072,T-074,T-075,T-077 | C1 matcher / A4 S8 vectors | C1 matching manifest | C1 distinct-symbol/self-match failures |
| R029 | T-011,T-036,T-050 | C2 matcher / graph vectors | C2 matching manifest | `C2_MATCHING_INFEASIBLE` |
| R030 | T-071,T-073-T-077 | C2 matcher / A4 S8 vectors | C2 matching manifest | C2 distinct-pair/self-edge failures |
| R031 | T-005,T-011,T-050 | C2 matcher / replay vectors | C2 matching manifest | `C2_MATCHING_INFEASIBLE` |
| R032 | T-003,T-006,T-012,T-020,T-021,T-025 | expectancy calculator / metric vectors | discovery/confirmation results | `OUTCOME_NOT_COMPUTABLE` |
| R033 | T-008,T-054 | power simulator / residual/nested-seed vectors | power manifest | `POWER_NOT_COMPUTABLE` |
| R034 | T-008,T-013,T-055 | permutation engine / shift vectors | temporal permutation manifest | permutation/week failures |
| R035 | T-013,T-015,T-056 | permutation engine / spread vectors | temporal permutation manifest | `PERMUTATION_VALIDITY_FAILURE` |
| R036 | T-013,T-015,T-017,T-025,T-042,T-056 | permutation engine / monotonicity vectors | temporal permutation manifest | `PERMUTATION_VALIDITY_FAILURE` |
| R037 | T-008,T-014,T-060 | bootstrap engine / percentile vectors | bootstrap manifest | `BOOTSTRAP_VALIDITY_FAILURE` |
| R038 | T-003,T-004,T-012,T-025,T-044-T-046 | Holm adjuster / family vectors | Holm registry | `HOLM_FAMILY_INCOMPLETE` |
| R039 | T-018,T-025,T-031,T-036,T-041,T-043,T-046 | verdict evaluator / fold truth table | confirmation results | mandatory fold gate failure |
| R040 | T-019,T-025,T-057 | concentration evaluator / concentration vectors | concentration report | `CONCENTRATION_NOT_COMPUTABLE` |
| R041 | T-016,T-025,T-047,T-058 | label classifier / truth table | label economics registry | label schema/disconnection failures |
| R042 | T-015,T-021,T-039,T-040,T-048,T-059 | IC calculator / rank vectors | IC report | `IC_NOT_COMPUTABLE` |
| R043 | T-030,T-061 | discovery orchestrator / state tests | discovery results | discovery technical failure |
| R044 | T-022,T-030,T-037,T-061 | freeze writer / tamper tests | discovery-freeze manifest | `DISCOVERY_FREEZE_MISMATCH` |
| R045 | T-030,T-037,T-061 | confirmation ledger / atomic-state tests | E5 confirmation ledger | confirmation custody failures |
| R046 | T-030,T-037,T-061 | confirmation orchestrator / state tests | confirmation results | confirmation dependency/resume failures |
| R047 | T-002,T-003,T-006,T-019-T-021,T-025-T-027,T-031-T-032,T-045,T-061 | verdict evaluator / exhaustive truth table | final verdict | `FINAL_VERDICT_NOT_COMPUTABLE` |
| R048 | T-022,T-068,T-077 | serializer / schema tests | artifact hash manifest | `ARTIFACT_HASH_MISMATCH` |
| R049 | T-022,T-037 | hash verifier / DAG tests | artifact hash manifest | artifact/confirmation-resume failures |
| R050 | T-023,T-078 | Phase 0 validator / 38 synthetic vectors | determinism report | `DETERMINISM_FAILURE` |
| R051 | T-023,T-070,T-078 | Phase 0 validator / Phase 0 integration | Phase 0 report | Phase 0 blocked status |
| R052 | T-027,T-028,T-061 | Phase 0 validator / acceptance audit | Phase 0 report | `UNAUTHORIZED_SCIENTIFIC_CHOICE` |
| R053 | T-024,T-027,T-028,T-061 | CLI and guards / negative tests | prohibited-data guard report | applicable stable failure code |

Scientific-gap audit: populations `CLOSED`; outcomes `CLOSED`; costs `CLOSED`;
trade notional `CLOSED`; funding `CLOSED`; ATR `CLOSED`; quintiles `CLOSED`; C1
and C1 self-match `CLOSED`; C2, self-edge, and randomization `CLOSED`;
expectancy `CLOSED`; power `CLOSED`; permutation `CLOSED`; spread `CLOSED`;
monotonicity `CLOSED`; bootstrap `CLOSED`; Holm `CLOSED`; folds `CLOSED`;
concentration `CLOSED`; labels `CLOSED`; IC `CLOSED`; discovery `CLOSED`;
confirmation `CLOSED`; final verdict `CLOSED`.

All D1-D12 decisions, Amendment 03 funding rules, and Amendment 04 self-matching
rules are represented without a scientific change. No clause retains a
prohibited unresolved status. Remaining work is
implementation and synthetic validation under this contract. No implementation,
test, dataset, download, Phase 0 run, discovery, confirmation, scientific row
inspection, semi-blind access, or lockbox access occurred while creating it.
