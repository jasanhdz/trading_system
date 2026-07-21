# Aegis Prospective Validation Protocol V1

**Protocol ID:** `aegis-prospective-validation-v1`  
**Status:** `FROZEN_INACTIVE`  
**Scientific lane:** Prospective current-brain validation; separate from historical E5  
**Side/timeframe:** SHORT, coordinated final 5-minute candles  
**Primary horizon:** 12 bars (60 minutes)

## 1. Separation and Activation

This protocol does not continue, complete, reopen, or replace historical E5.
Historical E5 remains closed under Owner Amendment 07. Historical entries,
targets, folds, semi-blind resources, and lockbox resources are prohibited.

No prospective observation is eligible until one activation manifest records:

1. this committed protocol and both committed schemas;
2. final Python and TypeScript source commits;
3. an approved, trained model artifact and its SHA-256;
4. the exact configuration hash and symbol-universe hash;
5. passing recorder, outcome, equivalence, safety, replay, and resume tests; and
6. `READY_FOR_SEPARATELY_AUTHORIZED_SHADOW_START` followed by separate Owner
   authorization to start collection.

Signals before that boundary are rejected as `PROSPECTIVE_PREACTIVATION_SIGNAL`.
No retrospective insertion is permitted. A changed code, model, configuration,
schema, label contract, universe, interval, cost, or outcome rule creates a new
protocol version and cohort.

The currently configured bundle `aegis-offline-reference-v1` is frozen for
engineering replay only. Its own artifact declares `approved=false`,
`trained=false`, and `purpose=OFFLINE_REFERENCE_ONLY`; therefore it cannot
activate prospective collection.

## 2. Question and Population

The prospective question is whether the complete frozen Aegis decision chain
produces operationally reliable and economically useful SHORT signals when
observed after preregistration without retrospective reconstruction.

Every candidate evaluated at a coordinated 5-minute decision cycle is recorded,
including rejected, wait, manual-only, no-trade, and enter-now candidates. The
eleven-symbol ordered universe and its hash are frozen in the protocol manifest.
No outcome may affect candidate recording, protocol eligibility, or future
configuration within this cohort.

## 3. Decision Chain and Evidence Mapping

The Python scientific brain owns the upstream proposal and ordered scientific
layers. TypeScript owns the operational entry-policy decision and Shadow
interlock. The recorder copies immutable outputs; it does not recalculate them.

| Envelope component | Authoritative prospective source |
|---|---|
| upstream model | Python `PredictionBatch` and frozen model bundle identity |
| D3 | Python ordered scientific layer output `REGIME` / D3 context |
| RV2 | Python ordered scientific layer output `RV2` |
| TRRM | Python ordered scientific layer output `TRRM` |
| QMAE | Python ordered scientific layer output `QMAE` |
| EQM | Python ordered scientific layer output `EQM` |
| ECON1 | Python ordered scientific layer output `ECON1` / net viability |
| final scientific selection | Python frozen `DecisionResponse` |
| final operational decision | TypeScript `AegisEntryDecisionResult` or strict Shadow gate result |
| routing | TypeScript Shadow-only routing result |

Missing component output fails envelope creation. An evidence hash reference is
not a substitute for the component record.

## 4. Prospective Signal Identity

Scheme: `aegis-prospective-signal-id-v1`.

The exact ordered tuple is:

1. protocol ID;
2. cohort ID;
3. model artifact SHA-256;
4. configuration SHA-256;
5. uppercase canonical symbol;
6. decision cycle ID;
7. canonical side (`SHORT` or `NO_TRADE`);
8. signal timestamp as UTC RFC3339 with millisecond precision;
9. information cutoff as UTC RFC3339 with millisecond precision; and
10. event sequence ID.

Serialize as a compact UTF-8 JSON array with no BOM, whitespace, or trailing
newline. The hash preimage is UTF-8 bytes of
`aegis-prospective-signal-id-v1`, one zero byte, then the serialized tuple.
Hash with full SHA-256 and encode as 64 lowercase hexadecimal characters.

No outcome, PnL, future price, funding result, future label, later operator
action, filesystem path, row order, or generation timestamp may enter the
identity. An identical identity and payload is a duplicate and fails
`PROSPECTIVE_DUPLICATE_SIGNAL`; an identical identity with a different payload
fails `PROSPECTIVE_SIGNAL_CONFLICT`.

## 5. Temporal Contract

All timestamps are UTC. `information_cutoff <= signal_timestamp`. Every market
input used by the decision must be final no later than the information cutoff.
The SHORT entry is the signal close under `SIGNAL_CLOSE`. Outcomes use exactly
the next 12 complete 5-minute bars. An outcome cannot be emitted before the last
required bar is final. Gaps, partial bars, duplicate bars, ambiguous timestamps,
or mismatched identities fail closed.

## 6. Outcome and Label Contract

Schema: `aegis-prospective-outcome-v1`. Source implementation:
`src/aegis/training/labels.py`, schema `aegis-labels-short-v4`, frozen by the
label-contract manifest.

- gross return: `(entry_price - terminal_close) / entry_price`;
- fees: 4 basis points per side;
- slippage: 1 basis point per side;
- round-trip cost: `0.001` return fraction;
- funding: `0` under the current label implementation;
- net return: gross return minus fees, slippage, and funding;
- MFE: maximum `(entry_price - future_low) / entry_price`, floored at zero;
- MAE/QMAE target: maximum `(future_high - entry_price) / entry_price`, floored at zero;
- `tail_event`: MAE at least `0.003`;
- `net_quality_after_costs`: MFE minus MAE minus `0.001`;
- `clean_quality`: exact `clean_entry` predicate in the frozen source;
- `label_valid`: true only after complete final contiguous horizon validation.

The exact clean predicate, ambiguous hit/stop rule, missingness behavior, and all
thresholds are enumerated in `prospective_label_contract.json`. Binary floating
point behavior is retained because it is the current frozen implementation;
changing numeric semantics requires a new protocol version.

Every target is written in a row-level immutable journal. No target may exist
only in memory. There is exactly one outcome per signal ID. Premature,
duplicate, conflicting, missing-signal, or incomplete-market outcomes fail
closed.

## 7. Shadow Contract

Shadow uses public market data or deterministic public-data replay only. It may
construct explicitly hypothetical intents and synthetic-balance quantities. It
cannot read credentials, sign requests, call private/account/order/position
endpoints, invoke a live adapter, or move money.

Startup requires mode exactly `SHADOW`, an active matching cohort, valid schemas,
matching code/model/config hashes, a synthetic balance source, and the endpoint
policy. There is no Shadow-to-Live CLI switch. Private operations always throw
their stable denial code.

Hypothetical fills are simulation records, never exchange fills. The V1 replay
fill rule is next final candle open, with the frozen 1 bp per-side slippage;
missing or stale data produces no fill. Live connectivity, private state, and
order behavior are outside this protocol.

## 8. Determinism, Checkpoints, and Audit

Canonical journals use sorted keys and newline-delimited UTF-8. Identity,
payload, prior-record, source, code, model, configuration, and schema hashes are
verified on append and resume. Checkpoints contain hashes, sequence counters,
and completed stage IDs, never credentials or future outcomes. Replay and
restart from identical inputs must produce byte-identical canonical evidence
and outcome journals.

## 9. Safety and Future Live Stages

Historical Discovery and Confirmation remain denied. Fold 3-4, semi-blind,
combined historical sources, and lockbox resources remain inaccessible.

The Owner's future USD 16 technical integration ceiling and USD 100 first formal
live-cohort ceiling are declarations only. They are inactive, require separate
authorizations, and are unreachable from this Shadow implementation.

## 10. Failure Codes

Required codes include `PROSPECTIVE_PREACTIVATION_SIGNAL`,
`PROSPECTIVE_DUPLICATE_SIGNAL`, `PROSPECTIVE_SIGNAL_CONFLICT`,
`PROSPECTIVE_MODEL_HASH_MISMATCH`, `PROSPECTIVE_CONFIG_HASH_MISMATCH`,
`PROSPECTIVE_CODE_HASH_MISMATCH`, `PROSPECTIVE_PROTOCOL_MISMATCH`,
`PROSPECTIVE_COMPONENT_EVIDENCE_INCOMPLETE`, `PROSPECTIVE_OUTCOME_PREMATURE`,
`PROSPECTIVE_OUTCOME_MISSING_SIGNAL`, `PROSPECTIVE_OUTCOME_DUPLICATE`,
`PROSPECTIVE_OUTCOME_CONFLICT`, `PROSPECTIVE_MARKET_DATA_INCOMPLETE`,
`SHADOW_PRIVATE_ENDPOINT_PROHIBITED`, `SHADOW_CREDENTIAL_ACCESS_PROHIBITED`,
`SHADOW_ORDER_OPERATION_PROHIBITED`, `SHADOW_LIVE_MODE_PROHIBITED`,
`SHADOW_UNAUTHORIZED_ENDPOINT`, `SHADOW_STALE_MARKET_DATA`,
`SHADOW_DUPLICATE_EVENT`, and `SHADOW_CHECKPOINT_CONFLICT`.

