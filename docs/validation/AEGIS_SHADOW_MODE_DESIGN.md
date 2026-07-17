# Aegis Shadow Mode Design

## Safety boundary

Shadow is an explicit scientific integration mode, distinct from replay, paper and live. The prepared TypeScript configuration is:

```yaml
brain:
  mode: shadow
execution:
  enabledByConfig: false
```

It is not registered in `TradingService`, PM2 or an application composition root in this phase. The evaluated candidate was rejected, so activating a continuous data path would create evidence for an unapproved artifact.

## Coordinated cycle

`src/brain/shadow.ts` accepts an already acquired operational snapshot. Before calling Python it requires the exact eleven-symbol universe/hash, 5m timeframe, final aligned candles, no source-reported gaps or duplicates, and contiguous timestamps. IDs and input hashes are deterministic. Repeated cycle IDs are rejected.

For a valid cycle it performs the manifest handshake, sends the versioned request, verifies the returned cycle, and applies `StrictDecisionGate` with `mode: SHADOW`. The gate always includes `SHADOW_MODE_NON_EXECUTING`; `execution.enabledByConfig` remains false. The module imports no exchange, order, leverage, sizing, bracket or trading-service code.

The result records the cycle, input hash, bundle, decision/ranking, selected candidate hash, gate result, hypothetical action, reasons and stage latency. `JsonlShadowEvidenceRecorder` is append-only at the call level; `InMemoryShadowEvidenceRecorder` supports deterministic replay tests.

## Outcomes

Python `resolve_shadow_outcome` resolves only mature final candles after a frozen decision. Entry is the next H12 bar open, normal maturity exits at the twelfth bar close, and the frozen scientific stop distance can invalidate earlier in the hypothetical calculation. Friction is explicit. LONG and SHORT favorable/adverse excursions are recorded.

Shadow/replay outcomes always have `executed=false`, `accepted=false`, `NOT_EXECUTED`, `realized_pnl=null`, and `execution_mode` SHADOW or REPLAY. They never query fills or mutate the frozen decision.

## Failure behavior

Unavailable/not-ready Python, manifest mismatch, invalid snapshot, stale/invalid decision, duplicate cycle, unknown symbol, kill switch, no slot and authorization/config denial all produce a denied or failed-closed record. None invokes an operational bridge.

## Offline replay

Python tests demonstrate chronological deterministic replay, LONG, SHORT, NO_TRADE, maturity, partial candle rejection, temporal gaps, scientific invalidation, duplicate cycles and idempotent outcome evidence. TypeScript tests demonstrate coordinated-cycle validation, manifest failure, Python unavailable, NO_TRADE/gate denial and zero outcome/execution calls.

## Activation and rollback

No activation command is provided because the bundle is rejected. No restart is required or permitted. The prepared source can be removed by reverting the two TS shadow commits and the Python shadow commit; runtime state is unaffected because no process imports or schedules these modules.

Before future activation, an approved immutable shadow bundle must be configured, a read-only composition root must feed coordinated snapshots, retention/rotation must be specified for the JSONL evidence, and the operator must separately authorize any required process restart. Execution must remain disabled.
