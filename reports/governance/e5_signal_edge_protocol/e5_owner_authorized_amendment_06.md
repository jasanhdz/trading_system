# E5 Owner-Authorized Amendment 06

## Canonical Derived Historical Trade Identity

**Status:** prospective owner-authorized scientific amendment

**Protocol:** `E5_SIGNAL_EDGE_PROTOCOL`

**Amendment:** `E5_OWNER_AUTHORIZED_AMENDMENT_06`

**Authorized at:** `2026-07-20T23:55:02Z`

**Effective:** when this document is committed

**Scientific rows inspected to select these rules:** none

## 1. Owner Authorization

The experiment owner prospectively authorizes the canonical derived historical
trade identity defined below. This authorization occurs before Phase 1A export,
Discovery, Confirmation, Shadow, or Live execution and without opening a
scientific row or Fold 3-4 payload.

The authorization resolves only the technical construction of `trade_id` when
the frozen historical record did not persist one. It does not reconstruct
historical scientific evidence and does not change the frozen E3 experiment.

## 2. Triggering Blocker

The E5 Phase 1A pre-implementation review stopped with:

```text
E5_HISTORICAL_REQUIRED_FIELD_UNAVAILABLE
field = trade_id
```

No separate Phase 1A blocking report was persisted. The terminal review
established statically that:

- E5-R012 requires non-null `trade_id`;
- `EconomicTrade` does not persist `trade_id`;
- no authoritative pre-E5 row-level manifest supplies it;
- an existing diagnostic derives a later 24-character identifier; and
- Amendment 05 correctly prohibited treating that later identifier as
  contemporaneously persisted scientific evidence.

No scientific rows were opened, no Fold 3-4 values were exposed, and no Phase
1A implementation file was created.

## 3. Verified Authority Chain

The following exact bytes and reachable Git commits were verified before this
amendment was written.

| Order | Authority | SHA-256 | Commit |
|---:|---|---|---|
| 1 | `e5_protocol_preregistration.md` | `c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770` | `b8b86d012c40c4d10f10efb68e5eb9d86d4ac476` |
| 2 | `e5_protocol_patch_02.md` | `c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e` | `92191db1a7c4135252377f64f51b174f180dcd53` |
| 3 | `e5_owner_authorized_amendment_01.md` | `c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4` | `943b98a5091c4d9238f754a1e42e63540a4579a6` |
| 4 | `e5_owner_authorized_amendment_02.md` | `b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f` | `521289606117a478debfca00d2e1fbaa5c2a4301` |
| 5 | `e5_owner_authorized_amendment_03.md` | `871be087550eb9d632795ded2c8f2633f1e481838198f0ee3ce53b9c8e9a350e` | `5003630ae42a806f79466ec10a4c052ce2a6f28a` |
| 6 | `e5_owner_authorized_amendment_04.md` | `a177980633c3280d6eaf6a4a798a6eb623f3692878639894869d2a39f8643774` | `a76553d15a239735bbb909f96ff3f06426148f50` |
| 7 | `e5_execution_specification.md` | `751b4014f1072e6fd0a49fb3a8820ba60b1b3c556eb94f9fbb7911d70516ae09` | `34441e412f79bc7d12d253040019e857ab5cf2c8` |
| 8 | `phase0/e5_phase0_report.json` | `a11502e7334d2a288c1b9aae0ef9761540cb2d6ae103d4212b106d6509314d5d` | `33fc2c24fba442491585fe719d16a2256752550c` |
| 9 | `e5_owner_authorized_amendment_05.md` | `5a3a71f64c105df417fdf0067c222f1a71879756feddae6fbfa389ae4e5475de` | `a80fa55c23ad37362d1de26a2ae3469374466a1a` |

The combined historical source identity was verified by hashing only, without
opening row values:

```text
path = reports/experiments/e3_validation_official/attempt_1/aegis-short-candidate-e3/runs/d742d9bc0ae867bb/econ_report.json
sha256 = bff472758eacc211dff1b3e2209cbd96e8a845a68f45b9e31526ca2968e6e085
```

Relevant controlling clauses are:

- original preregistration lines 93-96: entry identity is symbol, fold, signal
  timestamp, entry timestamp, entry price, and side, with no rescoring or
  reselection;
- original preregistration lines 147-169: entry hashes bind before labels and
  frozen timestamps, symbols, scores, gates, and selection decisions do not
  change;
- Patch 02 lines 25-43: Fold 1-2 Discovery, Fold 3-4 one-shot Confirmation, and
  no result-driven scientific change;
- Amendment 01 lines 205-224: the final decision space and pre-execution,
  unconsumed-lockbox state remain fixed;
- Amendment 02 lines 361-390: D1-D12 remain mutually consistent and frozen
  entries, folds, scores, Holm, Confirmation, and lockbox authority do not
  change;
- Amendment 03 lines 500-517: funding validation and all unrelated E5
  authority remain unchanged;
- Amendment 04 lines 282-308: control matching, self-match restrictions, folds,
  horizons, and unrelated authority remain unchanged;
- Execution Specification lines 197 and 216-240: `trade_id` is mandatory,
  non-funding IDs use SHA-256 over compact UTF-8 JSON arrays, timestamps use UTC
  epoch milliseconds, and required identity fields are non-null;
- Execution Specification lines 243-265: `trade_id` feeds `observation_id` and
  identities cannot contain outcomes or results;
- Phase 0 report: 38 of 38 synthetic categories passed, scientific rows
  inspected were zero, semi-blind was `NOT_ACCESSED`, and lockbox was
  `NOT_CONSUMED`; and
- Amendment 05 lines 121-160, 220-262, and 516-555: frozen historical evidence
  cannot be reconstructed, `trade_id` remains essential, and all modern
  component and scientific prohibitions remain active.

No contradiction requires changing D1-D12 or reopening a frozen scientific
decision.

## 4. Scope and Definitions

This amendment controls only `trade_id` for frozen historical E3 rows.

- **Historical scientific evidence:** a field describing signal generation,
  model or component output, eligibility, score, label, direction, entry,
  horizon, economic result, control assignment, or scientific population.
- **Canonical technical identity:** a deterministic, outcome-independent name
  for one already-frozen entry, used only for identity, duplicate and conflict
  checks, authorized joins, provenance, ordering, checkpointing, resume, and
  audit.
- **Canonical preimage:** the exact versioned compact JSON array in Section 8.
- **Legacy diagnostic ID:** the 24-character identifier produced by the later
  D1A diagnostic helper reviewed in Section 12.
- **Authorized payload:** the Fold 1-2 fields and provenance permitted by
  Amendment 05, excluding prohibited Fold 3-4 payloads.

## 5. Evidence Versus Technical Identity

Rule `E5-A06-R001`: historical scientific evidence and canonical technical
identity are distinct. A canonical `trade_id` identifies an existing frozen row
but supplies no evidence about whether that row should exist, how it was scored,
what a component decided, or what outcome occurred.

Rule `E5-A06-R002`: deriving `trade_id` cannot alter membership, eligibility,
ordering outside the frozen canonical ordering contract, score, label,
direction, entry, horizon, cost, funding, outcome, control assignment,
statistic, gate, or verdict.

Rule `E5-A06-R003`: `trade_id` may be used only for row identity, duplicate and
conflict detection, authorized joins, provenance, deterministic ordering,
checkpoint and resume identity, and audit. It is invalid as model evidence,
component evidence, a scientific covariate, or a selection/exclusion basis
except for already-governed duplicate and identity-conflict handling.

## 6. Authority Classification

Rule `E5-A06-R004`: the exact authority classification is:

```text
OWNER_AUTHORIZED_CANONICAL_DERIVED_IDENTITY
```

This classification means that `trade_id` was not contemporaneously persisted,
is deterministically reproducible from authoritative frozen fields, and is
owner-authorized for the technical purposes in Section 5. The classification
must accompany the identity-scheme version in manifest provenance. It must
never be represented as contemporaneously persisted evidence.

## 7. Scheme and Exact Identity Tuple

Rule `E5-A06-R005`: the identity scheme and domain-separation string are both:

```text
e5-historical-trade-id-v1
```

Rule `E5-A06-R006`: the exact ordered source tuple is the original
preregistered entry identity, in this order:

| Position | Canonical element | Frozen source path | Canonical type | Nullability |
|---:|---|---|---|---|
| 1 | `symbol` | `report.trades[].signal.symbol` | uppercase ASCII string in the frozen E5 symbol registry | forbidden |
| 2 | `fold` | `report.trades[].signal.fold` | string `F1`, `F2`, `F3`, or `F4` | forbidden |
| 3 | `signal_timestamp_utc_ms` | `report.trades[].signal.timestamp` | signed 64-bit UTC epoch-millisecond integer | forbidden |
| 4 | `entry_timestamp_utc_ms` | `report.trades[].entry_timestamp` | signed 64-bit UTC epoch-millisecond integer | forbidden |
| 5 | `entry_price_decimal` | `report.trades[].entry_price` | canonical positive base-10 decimal string | forbidden |
| 6 | `side` | `report.trades[].signal.side` | canonical enum string `LONG` or `SHORT`; E5 requires `SHORT` | forbidden |

The domain-separation string precedes these six elements in the serialized
preimage. No other field is permitted.

Rule `E5-A06-R007`: the tuple is sufficient by governance, not by observed
data: original preregistration lines 93-96 define these exact six fields as
entry identity. Actual uniqueness remains a mandatory exporter validation and
cannot be presumed from this declaration.

## 8. Canonicalization and Serialization

Rule `E5-A06-R008`: canonicalization is exact:

1. `symbol` must be a string. Strip ASCII space, tab, carriage return, and line
   feed only; convert ASCII letters to uppercase; then require an exact member
   of the frozen eleven-symbol E5 registry. Unicode normalization, character
   substitution, and symbol aliasing are prohibited.
2. `fold` accepts only source integer `1`, `2`, `3`, or `4`, or exact ASCII
   string `F1`, `F2`, `F3`, or `F4` after ASCII whitespace stripping and ASCII
   uppercasing. It serializes as `F1` through `F4`.
3. Each timestamp must be an ISO-8601 string with an explicit UTC offset or an
   already-validated signed 64-bit epoch-millisecond integer. Parse the instant,
   convert to UTC, and serialize its exact signed 64-bit Unix epoch-millisecond
   value. A source instant with nonzero precision below one millisecond is
   invalid; truncation and rounding are prohibited. Leap-second text unsupported
   by the frozen parser is invalid rather than adjusted.
4. `entry_price` must be parsed from its source JSON numeric token with exact
   base-10 decimal arithmetic before any binary floating-point conversion. It
   must be finite and strictly positive. Scientific notation is accepted only
   as source syntax and is expanded without value loss. Remove unnecessary
   leading integer zeros and trailing fractional zeros, preserve one integer
   digit and at least one fractional digit, prohibit a leading plus sign in the
   canonical result, and serialize zero only as `0.0`; strict positivity makes
   zero invalid here. Thus `100`, `100.0`, `100.00`, and `1.00E2` canonicalize
   to `100.0`.
5. `side` must be a string. Strip ASCII whitespace and uppercase ASCII letters;
   require exact enum value `LONG` or `SHORT`. No alias is permitted. The
   technical identity distinguishes the enum values; the E5 exporter separately
   requires `SHORT` and rejects `LONG` before an E5 row is sealed.
6. Every element is required and non-null. Boolean values are invalid for
   numeric fields. Non-finite numbers, unrecognized enums, non-ASCII symbols,
   and values outside signed 64-bit timestamp range are invalid.

Canonicalization failure emits a Section 15 failure code and produces no ID.

Rule `E5-A06-R009`: the canonical preimage is exactly this JSON array:

```text
["e5-historical-trade-id-v1",symbol,fold,signal_timestamp_utc_ms,entry_timestamp_utc_ms,entry_price_decimal,side]
```

Serialization rules are:

- UTF-8 without BOM;
- one JSON array;
- no whitespace outside string values;
- no trailing newline;
- elements in the exact order above;
- JSON strings escaped under the standard JSON grammar;
- timestamp elements encoded as JSON integers;
- price encoded as a JSON string to preserve exact decimal text;
- ASCII-only canonical strings, making `ensure_ascii` behavior immaterial; and
- no object or dictionary serialization in the identity preimage.

Source dictionary key order, source row order, filesystem order, locale,
timezone environment, and runtime hash randomization have no effect.

## 9. Hashing and Output Format

Rule `E5-A06-R010`: compute:

```text
digest = SHA-256(canonical_preimage_utf8_bytes)
trade_id = lowercase_hex(digest)
```

The digest is 256 bits. Output is exactly 64 lowercase hexadecimal characters.
There is no output prefix, suffix, separator, truncation, UUID conversion, or
base encoding other than lowercase hexadecimal. Domain separation is provided
inside the preimage by `e5-historical-trade-id-v1`.

Rule `E5-A06-R011`: the canonical preimage tuple must be retained or be
recomputable within sealed provenance so equality of digests can be checked
against equality of preimages without exposing prohibited payloads. A digest
alone is insufficient for collision adjudication.

## 10. Fold, Cycle, Horizon, and Source-Run Roles

Rule `E5-A06-R012`: fold is intrinsic entry identity. Original preregistration
lines 93-96 explicitly include fold, and the experimental unit is fold-nested.
Canonical fold therefore appears directly in the preimage. Otherwise identical
entries assigned to different persisted folds receive different IDs.

Rule `E5-A06-R013`: cycle identity is represented transitively by canonical
`fold` plus `signal_timestamp_utc_ms`, the same source tuple used by
`cycle_id=["cycle-v1",fold,signal_ms]` in the Execution Specification. A separate
`cycle_id` digest is excluded to avoid redundant version coupling. Different
canonical cycle identities necessarily differ in fold, signal timestamp, or
both and therefore produce different `trade_id` values.

Rule `E5-A06-R014`: horizon is excluded. One frozen entry receives
horizon-specific H12, H48, and H96 populations and outcomes under D1-B, but its
historical entry identity does not change. Horizon-specific identities remain
separate governed IDs.

Rule `E5-A06-R015`: source-run identity, source path, source SHA-256, repository
path, checkout path, code commit, and artifact-generation timestamp are excluded
from the preimage. They belong in provenance. Moving or rewrapping the same
authoritative frozen row cannot change its technical identity. A source hash is
still mandatory before derivation under Amendment 05.

Rule `E5-A06-R016`: `strategy_id`, `scenario_id`, and score are excluded from
intrinsic identity. The blind exporter must separately validate
`strategy_id=full_stack`, `scenario_id=B_BASE`, `side=SHORT`, and the frozen
score under Amendment 05. These fields cannot be used to make a different
identity for a row that fails eligibility validation.

## 11. Prohibited Dependencies

Rule `E5-A06-R017`: the preimage and derivation cannot use:

- exit timestamp or price;
- gross, fixed-cost, funding, or net return;
- MFE, MAE, barrier event, realized PnL, or profitability;
- target labels or any post-entry label;
- control assignment, match result, or control outcome;
- D3, RV2, TRRM, QMAE, EQM, ECON1, or current Aegis output;
- model replay, current-brain execution, Discovery, Confirmation, or verdict;
- wall-clock time, process ID, hostname, random value, environment-specific
  path, filesystem metadata, or iteration order; or
- any field not enumerated in Section 7.

An attempted outcome dependency fails with
`E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED`. An attempted current-brain
dependency fails with `E5_CANONICAL_TRADE_ID_CURRENT_BRAIN_DEPENDENCY_PROHIBITED`.
Other unauthorized elements fail canonicalization before hashing.

## 12. Legacy Diagnostic Derivation Review

Static review of
`scripts/diagnostics/exit_excursion_d1a/experiment.py:141-174` and
`src/aegis/utils/hashing.py:17-51` establishes that the legacy diagnostic:

1. constructs a JSON object with `symbol`, `fold`, raw `signal_timestamp`, and
   raw `entry_timestamp`;
2. sorts object keys through `canonical_json`;
3. hashes UTF-8 JSON bytes with SHA-256; and
4. truncates the lowercase hexadecimal digest to 24 characters.

It is deterministic for identical raw values, independent of outcomes and
current-brain execution, and does not use Fold 3-4 values to derive a Fold 1-2
row ID. It is nevertheless only **partially compatible** and is not adopted as
the canonical E5 identity because it:

- omits preregistered entry-price and side identity fields;
- has no identity scheme or domain separator;
- hashes raw timestamp representations instead of canonical UTC epoch
  milliseconds; and
- truncates SHA-256 to 96 bits.

Rule `E5-A06-R018`: legacy diagnostic IDs and canonical v1 IDs are not expected
to match. A legacy ID has no E5 identity authority. It may be recorded only as
explicitly non-authoritative migration provenance if a future schema allows it;
it cannot replace `trade_id`, influence joins, or determine eligibility.

Rule `E5-A06-R019`: if any artifact claims a legacy ID is canonical v1, claims
the two schemes are expected to match, or supplies a purported compatibility
mapping that fails its declared relation, validation fails with
`E5_CANONICAL_TRADE_ID_LEGACY_MISMATCH`. The validator must not silently rewrite
either ID.

## 13. Duplicate, Conflict, and Collision Semantics

Rule `E5-A06-R020`: an exact duplicate is two source rows with the same canonical
preimage and identical canonical authorized payload. Amendment 05's existing
policy controls: fail with `E5_BLIND_EXPORT_DUPLICATE_IDENTITY`, emit no sealed
manifest, and perform no silent deduplication or double counting.

Rule `E5-A06-R021`: an identity conflict is two rows with the same canonical
preimage or same `trade_id` but different canonical authorized payloads. Fail
closed with `E5_CANONICAL_TRADE_ID_CONFLICT`. Do not select a winner, merge,
average, or infer which row is correct.

Rule `E5-A06-R022`: a cryptographic collision is unequal canonical preimages
with the same 64-character digest. Fail closed with
`E5_CANONICAL_TRADE_ID_HASH_COLLISION`. No suffix, rehash, alternate algorithm,
or row-order tie-breaker is authorized.

Rule `E5-A06-R023`: duplicate, conflict, and collision audits may report the
failure code, non-sensitive source ordinal, digest, and field names that differ
when permitted by the clean-room contract. They must not expose Fold 3-4
scientific payload values.

## 14. Versioning and Provenance

Rule `E5-A06-R024`: `e5-historical-trade-id-v1` identifies the complete tuple,
order, canonicalization, serialization, domain separation, SHA-256 algorithm,
lowercase hexadecimal encoding, and 64-character output contract.

Any change to one of those properties requires prospective owner authority and
a new scheme version. Existing IDs cannot be silently regenerated, migrated, or
reinterpreted under changed semantics.

Rule `E5-A06-R025`: identity provenance must bind:

- authority classification;
- scheme version;
- Amendment 06 file SHA-256;
- combined-source SHA-256;
- exporter code commit and configuration SHA-256;
- canonicalization implementation version;
- permitted input-field registry hash;
- output-manifest SHA-256; and
- duplicate, conflict, collision, and legacy-compatibility validation results.

Missing provenance fails with
`E5_CANONICAL_TRADE_ID_PROVENANCE_INCOMPLETE`.

## 15. Failure Codes

All failures are deterministic, fail closed, produce only clean-room-safe audit
metadata, create no substitute identity, and leave lockbox state unchanged.

| Code | Trigger | Consequence |
|---|---|---|
| `E5_CANONICAL_TRADE_IDENTITY_UNRESOLVABLE` | the authoritative six-field tuple cannot identify entries under governance | block Phase 1A; do not fabricate another tuple |
| `E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING` | one Section 7 field is absent or null | block affected export before hashing |
| `E5_CANONICAL_TRADE_ID_INPUT_INVALID` | required input has wrong type, range, registry membership, or enum | block affected export |
| `E5_CANONICAL_TRADE_ID_CANONICALIZATION_FAILED` | exact Section 8 canonicalization cannot complete losslessly | block affected export |
| `E5_CANONICAL_TRADE_ID_NONDETERMINISTIC` | identical authorized inputs yield different preimage bytes or IDs | quarantine output and block E5 |
| `E5_CANONICAL_TRADE_ID_CONFLICT` | same canonical identity has different authorized payload | block export; no winner |
| `E5_CANONICAL_TRADE_ID_HASH_COLLISION` | unequal preimages have the same SHA-256 digest | block export; no suffix or rehash |
| `E5_CANONICAL_TRADE_ID_VERSION_UNSUPPORTED` | scheme version is absent, unknown, or changed | reject artifact |
| `E5_CANONICAL_TRADE_ID_LEGACY_MISMATCH` | artifact falsely equates legacy and canonical schemes or violates declared compatibility | reject mapping and artifact |
| `E5_CANONICAL_TRADE_ID_PROVENANCE_INCOMPLETE` | Section 14 provenance binding is incomplete | reject artifact |
| `E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED` | preimage or derivation requests a downstream outcome, label, funding, PnL, or result | reject before hashing |
| `E5_CANONICAL_TRADE_ID_CURRENT_BRAIN_DEPENDENCY_PROHIBITED` | preimage or derivation requests current component or Aegis execution | reject before invocation |

Existing Amendment 05 failures remain active, including
`E5_BLIND_EXPORT_DUPLICATE_IDENTITY`,
`E5_BLIND_EXPORT_CONFLICTING_IDENTITY`, and
`E5_HISTORICAL_COMPONENT_REPLAY_PROHIBITED`.

## 16. Amendment 05 Compatibility

Rule `E5-A06-R026`: Amendment 05's historical scientific-evidence prohibition
remains fully active. This amendment creates one narrow exception to its
no-fabrication rule: `trade_id` may be constructed as the owner-authorized
canonical technical identity in Sections 7-9. The exception applies to no other
field.

Rule `E5-A06-R027`: `trade_id` remains essential under E5-R012 and
E5-A05-R012. The requirement is satisfied by a valid
`OWNER_AUTHORIZED_CANONICAL_DERIVED_IDENTITY` when no contemporaneously
persisted `trade_id` exists. This changes construction authority, not
nullability or importance.

Rule `E5-A06-R028`: Amendment 05 continues to prohibit reconstruction of D3,
RV2, TRRM, QMAE, EQM, ECON1, final Aegis decisions, scores, labels, outcomes,
eligibility, controls, and population membership. Its historical unavailable
states, two-lane model, clean-room boundary, and blind exporter restrictions are
unchanged.

## 17. Phase 1A Exporter Implications

Rule `E5-A06-R029`: the future blind exporter must derive `trade_id` inside the
custodial boundary only after combined-source SHA-256 and source-schema
validation. It must use the exact v1 algorithm and no generic or legacy helper
whose behavior differs.

For authorized Fold 1-2 rows, it must emit canonical `trade_id` and use it for
identity, duplicate and conflict detection, authorized joins, provenance,
checkpointing, resume, and ordering only where the existing canonical ordering
contract permits.

For Fold 3-4 rows, the clean-room rule remains controlling: no scientific
payload may be exposed. Identity derivation for a prohibited row is neither
required nor authorized unless a future implementation proves that derivation
is necessary for a non-sensitive integrity check already allowed by Amendment
05. Partition filtering remains prior to downstream payload materialization.

Rule `E5-A06-R030`: exporter validation must compare each emitted ID with its
retained canonical preimage, validate unique IDs and preimages, verify the
scheme and authority classification, and seal the identity provenance with the
Fold 1-2 manifest hash.

## 18. Required Future Synthetic Validation Vectors

No vector may use an E5 scientific row.

| ID | Synthetic case | Required result |
|---:|---|---|
| 1 | same frozen row derived twice | byte-identical preimage and `trade_id` |
| 2 | source row order reversed | each row retains the same `trade_id` |
| 3 | source dictionary keys reordered | unchanged ID because preimage is an array |
| 4 | equivalent explicit-offset UTC timestamps | identical epoch milliseconds and ID |
| 5 | different signal or entry timestamp | different ID |
| 6 | `100`, `100.0`, `100.00`, and `1.00E2` entry price | identical canonical `100.0` and ID |
| 7 | different canonical symbol | different ID |
| 8 | different canonical cycle identity through fold or signal time | different ID |
| 9 | otherwise identical `LONG` and `SHORT` generic identity fixtures | different IDs; E5 `LONG` fixture also fails eligibility |
| 10 | same entry evaluated at H12, H48, and H96 | same `trade_id`; horizon IDs differ separately |
| 11 | otherwise identical row in different persisted fold | different ID because fold is intrinsic |
| 12 | exact duplicate row and payload | `E5_BLIND_EXPORT_DUPLICATE_IDENTITY` |
| 13 | same identity with conflicting authorized payload | `E5_CANONICAL_TRADE_ID_CONFLICT` |
| 14 | missing required tuple field | `E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING` |
| 15 | null required tuple field | `E5_CANONICAL_TRADE_ID_REQUIRED_INPUT_MISSING` |
| 16 | invalid or sub-millisecond timestamp | `E5_CANONICAL_TRADE_ID_INPUT_INVALID` or canonicalization failure |
| 17 | invalid, non-finite, zero, or negative entry price | `E5_CANONICAL_TRADE_ID_INPUT_INVALID` |
| 18 | unsupported identity version | `E5_CANONICAL_TRADE_ID_VERSION_UNSUPPORTED` |
| 19 | attempted outcome inclusion | `E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED` |
| 20 | attempted realized-PnL inclusion | `E5_CANONICAL_TRADE_ID_OUTCOME_DEPENDENCY_PROHIBITED` |
| 21 | attempted current-brain component inclusion | `E5_CANONICAL_TRADE_ID_CURRENT_BRAIN_DEPENDENCY_PROHIBITED` |
| 22 | attempted wall-clock inclusion | canonicalization rejected before hashing |
| 23 | attempted filesystem-path inclusion | canonicalization rejected before hashing |
| 24 | legacy equality expectation | policy records `NOT_EXPECTED_TO_MATCH`; no legacy-equals-v1 case is asserted |
| 25 | legacy ID supplied as canonical or falsely mapped | `E5_CANONICAL_TRADE_ID_LEGACY_MISMATCH`, reported without row payload |
| 26 | repeated blind export | byte-identical IDs and manifest bytes |
| 27 | interrupted and resumed export | IDs and final manifest equal uninterrupted export |
| 28 | prohibited Fold 3-4 synthetic canary | no payload, preimage, ID, log, exception, checkpoint, or report exposure |

Vector 9 validates side's identity role in the generic derivation contract while
also confirming that actual E5 accepts only `SHORT`. Vector 24 explicitly
records that the reviewed legacy and canonical output formats have no expected
equality case.

## 19. Unchanged Scientific Decisions

This amendment does not modify:

- D1-B, D2-A, D3-B, D4-A, D5-A, D6-C, D7-C, D8-A, D9-D, D10-A, D11-A, or
  D12-B;
- historical E3 row membership, folds, symbols, cycles, sides, entries, or
  horizons;
- scores, labels, eligibility, costs, barriers, outcomes, or Amendment 03
  funding;
- C1, C2, Amendment 04 self-match rules, matching, or seeds;
- bootstrap, temporal permutation, the twelve-test Holm family, MREM, alpha,
  gates, fold predicates, diagnostics, or verdicts;
- Fold 1-2 Discovery or Fold 3-4 Confirmation partitions;
- one-shot Confirmation semantics;
- Amendment 05 evidence authority, unavailable-state meaning, or two-lane
  separation; or
- semi-blind and lockbox prohibitions, lockbox state, consumed queries, or
  budget.

The amendment resolves only canonical historical row identity.

## 20. Lockbox and Semi-Blind State

At authorization:

```text
scientific_rows_inspected = 0
fold_3_4_values_exposed = false
semi_blind = NOT_ACCESSED
lockbox = NOT_CONSUMED
consumed_queries = []
budget_remaining = 1
Discovery = NOT_STARTED
Confirmation = NOT_STARTED
Shadow = NOT_STARTED
Live = NOT_STARTED
```

No current-brain replay, blind export, network call, scientific execution, or
operational action occurred in creating this amendment.

## 21. Effective Authority Order

Effective scientific authority is:

1. Original E5 Preregistration
2. Protocol Patch 02
3. Owner-Authorized Amendment 01
4. Owner-Authorized Amendment 02
5. Owner-Authorized Amendment 03
6. Owner-Authorized Amendment 04
7. Owner-Authorized Amendment 05
8. Owner-Authorized Amendment 06

The E5 Execution Specification remains the subordinate engineering contract.
Where its mandatory `trade_id` input lacked construction authority, Amendment
06 now supplies that authority. All unrelated specification rules remain
controlling.

## 22. Prohibited Post-Hoc Changes

After this amendment is committed, no observed E5 value or result may motivate
changing:

- identity authority classification;
- scheme or domain-separation string;
- input tuple, order, or source fields;
- canonicalization or serialization;
- fold, cycle, horizon, or source-run identity roles;
- hash algorithm, encoding, length, or output format;
- duplicate, conflict, or collision behavior;
- legacy compatibility classification; or
- permitted and prohibited identity uses.

## 23. Owner Authorization Record

The experiment owner authorizes `E5_OWNER_AUTHORIZED_AMENDMENT_06` as a
prospective governance act at the timestamp stated in this document. The Git
commit containing this exact file and its SHA-256 constitute the immutable
authorization record. No handwritten or personal cryptographic signature is
asserted.

Python repository at authorization:

```text
branch = feature/aegis-ts-clean-rebuild
commit = a80fa55c23ad37362d1de26a2ae3469374466a1a
working_tree = CLEAN
```

TypeScript repository at authorization:

```text
branch = feature/aegis-ts-clean-rebuild
commit = 53105ee34c6e29f960c37c3516a58fffd2aa5906
working_tree = CLEAN
modified_by_this_amendment = false
```

## 24. Completeness Declaration

The canonical historical `trade_id` blocker is fully resolved prospectively.
The owner-authorized v1 identity uses the original preregistered six-field entry
identity, exact canonicalization, a versioned compact JSON-array preimage, full
SHA-256, and lowercase 64-character hexadecimal output. Fold is intrinsic;
horizon and source-run identity are excluded. The legacy diagnostic ID is
partially compatible but non-authoritative and is not expected to match.

`trade_id` remains essential. Scientific evidence reconstruction remains
prohibited. No identity implementation, blind exporter, scientific dataset,
Discovery, Confirmation, Shadow, Live action, semi-blind access, or lockbox
access is authorized or performed by this amendment.
