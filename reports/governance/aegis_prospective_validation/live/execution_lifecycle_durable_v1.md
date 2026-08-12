# Aegis Durable Execution Lifecycle V1

## Scope

This lifecycle is an inactive execution candidate. It does not replace or
restart the current Live path. It adds a tested integration boundary for a
future challenger after separate compatibility validation.

## Durable Identity

Every entry is identified from the signal identity, symbol, side, quantity,
expected price, feature identity, policy identity, and exact protection policy.
The resulting Binance client entry and exit identifiers are deterministic.
Repeated processing therefore resolves to the same intent and request identity.

The intent, including stop ROE, take-profit ROE, leverage, and price precision,
is fsynced before an exchange mutation can be attempted.

## States

- `INTENT_CREATED`: immutable intent exists durably.
- `ORDER_SUBMITTING`: pre-request journal is durable.
- `ORDER_SUBMITTED`: transport acknowledged the request.
- `PARTIALLY_FILLED`: exchange confirms partial exposure.
- `FILLED`: exchange confirms complete intended exposure.
- `PROTECTION_PENDING`: financial exposure exists but exact protection is not confirmed.
- `PROTECTED`: both stop and take-profit cover the complete current position quantity.
- `EXIT_PENDING`: a deterministic reduce-only close is durably journaled.
- `CLOSED`: exchange conclusively reports no exposure.
- `RECONCILIATION_REQUIRED`: exchange truth is incomplete or acknowledgement is ambiguous.
- `FAILED_CLOSED`: definitive terminal rejection without exposure.

## Authority and Recovery

After timeout, retry is prohibited until order identity, fills, position,
protection, and close identity have been read. A conclusive `NOT_FOUND` is the
only condition that authorizes resubmission, and resubmission reuses the same
client order ID.

After restart, the append-only journal is replayed and nonterminal intents are
reconciled. If local state is completely absent, open exchange positions are
discovered and adopted. Exchange positions and fills are authoritative for
financial exposure; local state supplies intent and policy provenance.

Partial fills are protected for the actual exchange position quantity. A stop
and take-profit that exist but cover less than that quantity do not count as
protected. If protection cannot be created and verified, the lifecycle journals
an idempotent reduce-only emergency exit before requesting it.

## Validation

Twenty-one focused durable-lifecycle TypeScript tests pass, including seven
end-to-end acceptance scenarios. They cover deterministic identity,
duplicate intent conflict, partial fill, ambiguous timeout received by Binance,
ambiguous timeout not received, read-before-retry, crash between fill and
bracket, complete local-state loss, partial bracket coverage, bracket failure,
idempotent close, corrupted journal, ambiguous transport reads, and exact
protection reconciliation.

The complete fake lifecycle proceeds from durable intent through entry, fill,
exact bracket coverage, process restart, duplicate event, reduce-only exit, and
terminal accounting state. Adverse cases include definitive entry rejection,
partial fill, both timeout outcomes, stop rejection, missing take-profit,
restart before bracket, and complete journal loss. This rehearsal found and
fixed a candidate-only bug where requesting exit again after `CLOSED` could
degrade the terminal record to `RECONCILIATION_REQUIRED`.

`npm run build` passes. No exchange request was made by these tests; all
mutation-capable paths used fakes or mocked transport.

## Remaining Activation Work

The boundary remains disconnected from the current Live `TradingService`.
Connecting it requires a separate compatibility gate and startup rehearsal with
fake transport. This preserves the instruction not to change current Live
behavior before safety and compatibility are demonstrated.
