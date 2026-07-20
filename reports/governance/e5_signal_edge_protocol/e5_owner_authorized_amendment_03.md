# E5 Owner-Authorized Amendment 03: Canonical Historical Funding Contract

## 1. Title and Status

| Field | Frozen value |
|---|---|
| Document | `E5_OWNER_AUTHORIZED_AMENDMENT_03` |
| Status | `OWNER_AUTHORIZED_PROSPECTIVE_SCIENTIFIC_AMENDMENT` |
| Experiment | `E5_SIGNAL_EDGE_CONTROL_TEST` |
| Accepted alias | `E5_ENTRY_EDGE_INVESTIGATION` |
| Scope | Canonical historical funding input contract for D3-B only |
| Schema version | `e5-funding-history-v1` |
| Created UTC | `2026-07-20T18:42:18Z` |
| Repository branch | `feature/aegis-ts-clean-rebuild` |
| Repository HEAD before amendment | `521289606117a478debfca00d2e1fbaa5c2a4301` |

## 2. Owner Authorization

The experiment owner prospectively authorizes the complete funding contract in
this amendment. The authority comes from the owner's direct instruction. No E5
scientific row, funding-history row, outcome, discovery result, or confirmation
result was inspected or used to select these rules.

The prospective rationale is to replace an undefined continuous scalar
approximation with discrete historically observed funding events; use
first-party exchange records; prevent funding look-ahead, interpolation, and
unsupported proration; make source and normalized artifacts auditable; avoid
rate clipping; eliminate duplicate and missing-data discretion; preserve D3-B;
and express funding directly in return space.

## 3. Prospective Timing Declaration

This amendment was created before any E5 implementation, E5 dataset
construction, E5 funding-history materialization, discovery execution,
confirmation execution, or E5 scientific-result inspection. No historical
funding API was called and no funding record was downloaded or inspected while
creating it. Static governance, source, schema, configuration, documentation,
type, and repository-structure inspection were the only permitted inspections.

## 4. Governance Hierarchy

Authority is ordered as follows:

1. Original preregistration.
2. Procedural Patch 02.
3. Owner-Authorized Amendment 01.
4. Owner-Authorized Amendment 02.
5. This Owner-Authorized Amendment 03.

A later document controls only where it explicitly and prospectively resolves
an omitted definition. This amendment controls only the previously undefined
historical funding input contract required by D3-B. Every other clause remains
frozen. In particular, the original protocol's development-only and lockbox
prohibitions remain binding (`e5_protocol_preregistration.md`, lines 5-20 and
126-150), and Amendment 02's D3-B remains binding
(`e5_owner_authorized_amendment_02.md`, lines 131-152).

## 5. Verified Hashes and Commits

The predecessor files were verified byte-for-byte and their commits were
verified reachable before this amendment was written.

| Authority | Path | Commit | SHA-256 |
|---|---|---|---|
| Original preregistration | `reports/governance/e5_signal_edge_protocol/e5_protocol_preregistration.md` | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` |
| Patch 02 | `reports/governance/e5_signal_edge_protocol/e5_protocol_patch_02.md` | `92191db1a7c4135252377f64f51b174f180dcd53` | `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e` |
| Owner Amendment 01 | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_01.md` | `943b98a5091c4d9238f754a1e42e63540a4579a6` | `c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4` |
| Owner Amendment 02 | `reports/governance/e5_signal_edge_protocol/e5_owner_authorized_amendment_02.md` | `521289606117a478debfca00d2e1fbaa5c2a4301` | `b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f` |

The physical SHA-256 of this amendment is recorded with the containing commit.
The commit itself records the post-commit repository identity. Neither value is
embedded as a literal in the content it identifies.

## 6. Safety State

At creation:

- Python tracked working tree: `CLEAN`.
- TypeScript working tree: `CLEAN` at
  `53105ee34c6e29f960c37c3516a58fffd2aa5906`.
- E5 implementation: absent.
- E5 scientific and funding datasets: absent.
- E5 discovery and confirmation: not executed.
- E5 scientific results: absent and uninspected.
- Semi-blind data: not accessed for E5.
- Lockbox data: not accessed for E5.
- Lockbox: `NOT_CONSUMED`.
- `consumed_queries=[]`.
- `budget_remaining=1`.

## 7. Scope of the Identified Gap

The original protocol requires duration-dependent funding in B_BASE economics
(`e5_protocol_preregistration.md`, lines 105-111) but did not define a canonical
historical funding dataset. Amendment 02 froze fixed barrier geometry and
realized funding, including the interval strictly after entry through
termination and `NOT_COMPUTABLE` handling for missing funding
(`e5_owner_authorized_amendment_02.md`, lines 131-149). This amendment supplies
only the missing historical input contract. It does not alter D3-B or any
scientific procedure, estimand, population, gate, or verdict.

## 8. Binding Decisions F1-F14

### F1: Source Authority

1. The sole authoritative scientific source is the official Binance USDⓈ-M
   Futures historical funding-rate service or an official first-party Binance
   archival export representing the same funding events.
2. The source covers only the eleven canonical uppercase symbols already
   frozen by E5 configuration.
3. Each symbol maps to its exact Binance USDⓈ-M perpetual-contract
   identifier, and the mapping is frozen before retrieval.
4. Third-party providers, aggregators, charting platforms, and reconstructed
   funding histories are prohibited.
5. The operational TypeScript `FundingSnapshot` and Python scalar
   `funding_bps_per_hour` model are not E5 historical sources.
6. Retrieval occurs only during the future authorized E5 input-materialization
   stage. Every raw response or official export is retained immutably and
   hashed before normalization.
7. The input manifest records `provider=BINANCE_USDM_FUTURES`, retrieval
   mechanism, official endpoint or export identity, request parameters,
   requested UTC start and end, pagination sequence, response count, raw path,
   raw SHA-256, retrieval-software version, and retrieval timestamp.
8. Retrieval timestamp is audit metadata only and enters no outcome, artifact
   identity, or gate. Source substitution after materialization starts is
   prohibited.

### F2: Funding Event Semantics

1. Historical funding is a sequence of discrete settlement events. One record
   represents one funding rate effective at one official settlement timestamp.
2. Continuous hourly accrual, linear interpolation, proration before or after
   settlement, and synthetic events are prohibited.
3. No event in a holding interval means zero funding only after source
   completeness for that interval has been validated.
4. An event contributes exactly when
   `entry_timestamp < funding_time <= termination_timestamp`.
5. An event at entry is excluded; an event at termination is included.
6. Comparisons use UTC Unix epoch milliseconds. A rate becomes effective only
   at its own `funding_time` and is never applied earlier.
7. A provider funding interval does not imply continuous accrual. Only the
   discrete effective settlement event is used.

### F3: Canonical Schema

Schema version is `e5-funding-history-v1`. Each normalized scientific record
contains exactly these fields in this order:

| Field | Type | Frozen semantics |
|---|---|---|
| `schema_version` | string | Exact value `e5-funding-history-v1` |
| `provider` | string | Exact value `BINANCE_USDM_FUTURES` |
| `symbol` | string | Canonical uppercase E5 symbol and exact mapped Binance perpetual-contract identifier |
| `funding_time_utc_ms` | signed 64-bit integer | Official effective settlement time in UTC Unix epoch milliseconds |
| `funding_rate_decimal` | string | Canonical signed base-10 decimal fraction of notional |
| `source_artifact_sha256` | string | Lowercase 64-character hexadecimal SHA-256 of the immutable raw source artifact |
| `funding_record_id` | string | Lowercase 64-character hexadecimal deterministic identity defined by F7 |

No outcome, position PnL, horizon, label, model score, or E5 result may appear
in a canonical funding record.

### F4: Rate Units

1. `funding_rate_decimal` is a signed decimal fraction of position notional;
   `0.0001` means `0.01%`.
2. It is not basis points, basis points per hour, percent units, an annualized
   rate, an hourly rate, or a continuously compounded rate.
3. The normalized value is a base-10 decimal string. Scientific notation in
   the canonical artifact is prohibited.
4. Parsing and normalization use exact decimal arithmetic. Binary
   floating-point parsing cannot precede validation of the canonical decimal.
5. Clipping, winsorization, truncation, and arbitrary magnitude bounds are
   prohibited.
6. A rate is invalid only if absent, syntactically invalid, non-finite, not
   representable as a signed base-10 decimal, or inconsistent with the
   immutable raw source.

### F5: Position Sign and Return

1. A positive funding rate means LONG pays and SHORT receives. A negative rate
   means LONG receives and SHORT pays.
2. Per event, `funding_return_event = +funding_rate_decimal` for SHORT and
   `funding_return_event = -funding_rate_decimal` for LONG.
3. E5 is SHORT, so its event return is the signed `funding_rate_decimal`.
4. Total E5 funding return is the exact-decimal sum of included event returns
   in `(entry_timestamp, termination_timestamp]`.
5. Positive total funding return increases B_BASE net return; negative total
   funding return decreases it.
6. Funding does not alter either barrier, event ordering, or termination time.
   Barrier classification remains separate from final net profitability.

### F6: Notional Convention

1. Funding is computed in return space as a fraction of the same frozen trade
   notional used by the E5 economic model.
2. No additional mark-price series is needed for return-based E5 statistics;
   every included event contributes its signed rate directly as a fractional
   return on frozen notional.
3. Diagnostic currency amounts use
   `funding_pnl_event = frozen_trade_notional * funding_return_event` and total
   `funding_pnl = frozen_trade_notional * total_funding_return`.
4. Notional does not vary with mark price, unrealized PnL, leverage, price
   movement, or funding time. Leverage does not multiply funding return.
5. Dynamic mark-price notional is prohibited.
6. The execution specification must identify and cite the already frozen
   trade-notional source. If that source cannot be located in governance or
   static configuration, it must report a separate governance gap and cannot
   invent a value.

### F7: Record Identity and Uniqueness

The natural uniqueness key is `(symbol, funding_time_utc_ms)`. Exactly one
normalized record is permitted per natural key.

`funding_record_id = SHA-256(UTF-8("e5-funding-history-v1" + "|" + symbol + "|" + canonical_base10(funding_time_utc_ms)))`

The rate is deliberately excluded from identity so conflicting rates for one
natural event cannot become separate valid records.

1. Recompute and verify every `funding_record_id`.
2. Duplicate natural keys and duplicate record IDs are prohibited.
3. A shared natural key with different rates is a source conflict.
4. Byte-identical records sharing a natural key are also duplicate-input
   failures.
5. Silent deduplication, last-write-wins, first-write-wins, and rate averaging
   are prohibited. Any duplicate fails normalized-artifact validation.

### F8: Ordering

Canonical order is:

1. `symbol`, ascending by exact UTF-8 bytewise lexical order;
2. `funding_time_utc_ms`, ascending numerically;
3. `funding_record_id`, ascending lexically as an integrity tie-breaker.

The third field cannot be needed in valid data because natural-key uniqueness
prohibits a legitimate same-symbol timestamp tie. Source ordering has no
scientific authority. Normalize into canonical order before serialization and
hashing.

### F9: Completeness and Missing Data

1. For every symbol, retrieval starts no later than the earliest possible E5
   entry and ends no earlier than the latest possible H96 termination.
2. Retrieval is complete over that full UTC interval. Pagination continues
   until the official source proves exhaustion of the requested interval.
3. Record every request and page boundary. API or export errors, truncation,
   unverified pagination, or inconsistent page overlap make the source
   incomplete.
4. A complete retrieval with no event in a holding interval means zero events.
   Incomplete retrieval never implies zero.
5. If completeness cannot be established for a symbol and required interval,
   every affected observation is `FUNDING_NOT_COMPUTABLE`. Missing mandatory
   funding makes its B_BASE net outcome `NOT_COMPUTABLE` under D3-B.
6. Forward fill, backward fill, interpolation, neighboring-symbol use,
   `funding_bps_per_hour`, `FundingSnapshot`, and zero imputation under
   unverified completeness are prohibited.

### F10: Raw and Normalized Artifacts

There are exactly two artifact layers:

1. Immutable raw source artifacts preserve exact provider response or export
   bytes without editing or reformatting. Each records its SHA-256, relative
   path, request or page association, byte size, retrieval metadata, and
   provider identity.
2. The canonical normalized artifact defaults to
   `artifacts/e5/inputs/funding/e5_funding_history_v1.jsonl`; its manifest
   defaults to
   `artifacts/e5/inputs/funding/e5_funding_history_v1_manifest.json`.

The deterministic repository root may change only when repository architecture
requires it. The filenames and scientific identity remain unambiguous.

### F11: Canonical Serialization

The normalized artifact is UTF-8 JSON Lines:

1. UTF-8 without BOM, LF line endings, exactly one object per physical line,
   no blank lines, and one final LF.
2. Keys appear exactly in the F3 order.
3. Strings use standard JSON escaping.
4. `funding_time_utc_ms` is a JSON integer;
   `funding_rate_decimal` is a JSON string.
5. There is no insignificant whitespace outside string values.
6. Records use F8 ordering.
7. Hash the complete serialized file with SHA-256.
8. Re-normalization from the same valid raw bytes with the same normalization
   software version must be byte-identical.

### F12: Decimal Canonicalization

Canonical rate text has an optional leading minus, at least one integer digit,
one decimal point, and at least one fractional digit. It has no leading plus or
exponent notation. Remove unnecessary leading integer zeros while retaining one
zero before the decimal point. Remove trailing fractional zeros while retaining
at least one fractional digit. Normalize negative zero to `0.0`. Preserve the
exact decimal value.

| Raw or parsed value | Canonical value |
|---|---|
| `"0"` | `"0.0"` |
| `"00.0100"` | `"0.01"` |
| `"-0.0000"` | `"0.0"` |
| `"1E-4"` | Parse exactly from the raw provider value, then normalize to `"0.0001"`; exponent notation is invalid in the canonical artifact |

### F13: Validation Rules

The normalized artifact is valid only when all checks pass:

1. exact schema-version match;
2. exact provider match;
3. symbol membership in the frozen eleven-symbol registry;
4. exact frozen symbol-to-contract mapping;
5. integer UTC epoch-millisecond timestamp;
6. timestamp within source request or export coverage;
7. canonical decimal rate;
8. source hash present in the raw manifest;
9. source hash matching immutable bytes;
10. exact recomputation of `funding_record_id`;
11. natural-key uniqueness;
12. record-ID uniqueness;
13. canonical ordering;
14. raw-to-normalized reconciliation;
15. pagination or export completeness;
16. normalized SHA-256 matching the manifest;
17. normalized count reconciling to accepted source records;
18. every rejected source record recorded with its exact failure reason;
19. no silently dropped rejected record; and
20. no scientific outcome or model score used in normalization.

Validation never depends on whether funding improves or worsens an E5 result.

### F14: Failure Codes

The stable funding failure taxonomy contains:

1. `FUNDING_SOURCE_UNAUTHORIZED`;
2. `FUNDING_SOURCE_INCOMPLETE`;
3. `FUNDING_SOURCE_HASH_MISMATCH`;
4. `FUNDING_PROVIDER_MISMATCH`;
5. `FUNDING_SYMBOL_UNAUTHORIZED`;
6. `FUNDING_SYMBOL_MAPPING_MISMATCH`;
7. `FUNDING_TIMESTAMP_INVALID`;
8. `FUNDING_RATE_INVALID`;
9. `FUNDING_DUPLICATE_NATURAL_KEY`;
10. `FUNDING_DUPLICATE_RECORD_ID`;
11. `FUNDING_CONFLICTING_RATE`;
12. `FUNDING_RECORD_ID_MISMATCH`;
13. `FUNDING_ORDER_INVALID`;
14. `FUNDING_RAW_NORMALIZED_RECONCILIATION_FAILURE`;
15. `FUNDING_MANIFEST_INVALID`;
16. `FUNDING_SCHEMA_VERSION_MISMATCH`; and
17. `FUNDING_NOT_COMPUTABLE`.

Every code is deterministic and auditable. No failure authorizes source
substitution, interpolation, zero imputation, a continuous scalar fallback, an
operational snapshot fallback, or a retry with changed scientific parameters.

## 9. Canonical Funding Schema

F3 is the complete normative schema. `e5-funding-history-v1` is the only valid
version, and the seven named fields are the complete record. F11 fixes their
physical order and representation. Adding, removing, renaming, or semantically
overloading a field requires new prospective owner authority before any data
inspection.

## 10. Source-Authority Contract

F1 is binding. Official Binance USDⓈ-M Futures records are the sole authority.
Raw bytes are retained and hashed before normalization. The frozen symbol map,
requests, pagination, source identity, and hashes form the source-provenance
chain. Operational snapshots, continuous scalar models, and third-party
histories cannot enter that chain.

## 11. Event and Interval Semantics

F2 is binding. Funding consists only of official discrete settlement events.
For each observation, include exactly those events in the half-open/closed UTC
interval `(entry_timestamp, termination_timestamp]`. The entry boundary is
exclusive and termination boundary inclusive. There is no continuous accrual,
proration, interpolation, or application before the effective timestamp.

## 12. Rate and Sign Conventions

F4 and F5 are binding. Rates are exact signed decimal fractions of notional.
For E5 SHORT positions, each included event contributes the signed provider
rate directly to funding return. Positive means receipt and increases B_BASE
net return; negative means payment and decreases it.

## 13. Notional Convention

F6 is binding. Funding uses fixed-notional return space, consistent with the
original equal-fixed-notional E5 contract (`e5_protocol_preregistration.md`,
lines 107-111 and 132-145). Dynamic mark-price notional and leverage scaling
are prohibited. A future execution specification must cite the physical frozen
notional source or report a distinct governance gap without inventing it.

## 14. Identity and Uniqueness

F7 is binding. `(symbol, funding_time_utc_ms)` is the natural key and its
deterministic SHA-256 is the record identity. Rates are excluded from identity.
Every duplicate or conflict fails closed; there is no deduplication policy that
can retain one competing record.

## 15. Ordering

F8 is binding. UTF-8 bytewise symbol order, numeric event-time order, then
lexical record-ID order is the only canonical sequence. Input order cannot
affect serialization, hashes, or scientific calculations.

## 16. Completeness and Missing-Data Behavior

F9 is binding. Completeness is established per canonical symbol over the full
potential E5 entry-through-H96 interval. A proven-complete interval with no
event contributes zero; an unproven or incomplete interval makes every affected
net outcome `FUNDING_NOT_COMPUTABLE`. No imputation or alternative source may
convert incompleteness to zero.

## 17. Serialization and Decimal Canonicalization

F11 and F12 are binding. Canonical JSONL bytes and canonical exact-decimal text
are part of scientific input identity. Binary floating point, display rounding,
source ordering, locale, platform newline defaults, and JSON key-order defaults
cannot alter the artifact.

## 18. Raw and Normalized Artifact Contracts

F10 is binding. Raw source bytes and normalized JSONL are distinct immutable
layers. Raw files preserve provider evidence. Normalized records preserve only
the F3 scientific schema and reference their raw artifact by SHA-256. The
normalized manifest links both layers and proves reconciliation.

## 19. Canonical Funding Manifest

The manifest contains at minimum:

- `schema_version` and `provider`;
- canonical-symbol-registry hash and symbol-to-contract mapping;
- required UTC coverage start and end;
- retrieval mechanism and official source identity;
- request and pagination records;
- raw artifact paths, SHA-256 values, and byte sizes;
- accepted and rejected record counts;
- rejection-report path;
- normalized artifact path, SHA-256, byte size, and record count;
- earliest and latest funding timestamp per symbol;
- coverage-completeness status per symbol;
- normalization-software commit and configuration hash;
- canonical-order declaration;
- duplicate-check and reconciliation results; and
- creation timestamp as non-scientific audit metadata.

Creation time enters no scientific calculation or artifact identity.

## 20. Validation Rules

F13 is the complete scientific validation contract. Validation is fail-closed,
reconciles every accepted and rejected raw record, and is independent of model
scores and E5 outcomes. A valid normalized artifact must pass all twenty checks;
partial validity cannot authorize use.

## 21. Failure Taxonomy

F14 is the minimum stable failure taxonomy. Implementations may add narrower
engineering subcodes only when they map one-to-one to an F14 parent and do not
alter failure consequences. No code permits a scientific fallback. An affected
mandatory net outcome is `NOT_COMPUTABLE` whenever required funding cannot be
established under this contract.

## 22. Relationship to D3-B

1. D3-B remains unchanged.
2. Fixed execution costs remain inside fixed-cost economic return.
3. Funding remains outside barrier geometry.
4. Funding enters only final realized B_BASE net return.
5. Funding events are discrete official settlement events.
6. The inclusion interval is `(entry_timestamp, termination_timestamp]`.
7. E5 SHORT funding return is the exact signed sum of official
   `funding_rate_decimal` values in that interval.
8. A favorable barrier event does not guarantee positive B_BASE net return.
9. Missing or incomplete mandatory coverage makes the affected net outcome
   `NOT_COMPUTABLE`.

## 23. Relationship to the Execution Specification

A future execution specification must consume this contract without
modification. It must define the exact materialization module and supported
official retrieval mechanism; deterministic requests and pagination; raw and
normalized hashing; exact-decimal arithmetic; all F14 codes; source
completeness proof; and synthetic conformance tests that use no E5 scientific
rows. It may freeze engineering details only when they cannot change this
scientific contract.

## 24. Relationship to Phase 0

Phase 0 must validate statically and synthetically:

- schema version and field registry;
- source-authority configuration and frozen symbol mapping;
- record-identity and decimal-canonicalization vectors;
- duplicate rejection and canonical ordering;
- raw-to-normalized reconciliation and manifest schema;
- sign convention and interval boundaries;
- zero-event behavior under proven completeness; and
- incomplete-source `NOT_COMPUTABLE` behavior.

Phase 0 must not download or inspect E5 funding-history rows.

## 25. Explicit Non-Changes

This amendment does not modify D1-B, D2-A, D3-B, D4-A, D5-A, D6-C, D7-C,
D8-A, D9-D, D10-A, D11-A, or D12-B. It changes no E5 hypothesis, frozen entry,
H12/H48/H96 horizon, fold, MREM, alpha, twelve-test Holm family, matching,
bootstrap, permutation, label registry, concentration rule, final verdict,
discovery authority, development-confirmation authority, semi-blind
prohibition, lockbox prohibition, lockbox state, or lockbox budget.

## 26. Prohibited Post-Hoc Changes

After this amendment is committed, no E5 result may motivate a change to the
source provider, discrete treatment, timestamp interpretation, inclusion
interval, units, LONG or SHORT sign, frozen-notional convention, decimal
normalization, natural key, duplicate policy, completeness policy, missing-data
policy, serialization, schema version, validation rules, or fallback behavior.

## 27. Owner-Authorization Record

The owner's direct instruction creating this document is the authorization
record. No cryptographic owner signature is asserted or fabricated. The
physical SHA-256 is recorded in the containing commit message, and the Git
commit records this amendment's immutable repository identity.

## 28. Completeness Declaration

The canonical historical funding input gap required by D3-B is fully resolved.
All source, event, schema, timestamp, unit, identity, uniqueness, order,
duplicate, completeness, missing-data, sign, notional, serialization, hashing,
validation, and version properties are frozen prospectively. No scientific
choice within this contract is delegated to implementation. No E5 scientific
or funding row was inspected; no E5 implementation, dataset, or execution was
created; no API was called; semi-blind and lockbox resources remain untouched.
