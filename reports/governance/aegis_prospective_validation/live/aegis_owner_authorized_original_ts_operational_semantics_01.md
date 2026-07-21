# Aegis Owner-Authorized Original TypeScript Operational Semantics 01

## Authority and Purpose

The Owner authorizes restoration and verification of the TypeScript trading
bot's original operational semantics. This is a restoration authority, not an
authority to redesign capital management, risk, execution, or Live controls.

For this authority, **original behavior** means the behavior at the parent of
the first commit unique to `feature/aegis-ts-clean-rebuild`. Git parentage,
not timestamps or observed performance, determines that baseline.

## Authorized Operational Semantics

The pre-Phase-0 TypeScript implementation is authoritative for existing
trading guards, capital handling, position sizing, leverage, margin handling,
order eligibility, concurrent-position behavior, entries, exits, brackets,
stops, take profits, retries, reconciliation, recovery, risk, and Live-mode
behavior.

Existing original guards, their ordering, configuration sources, defaults,
and failure handling must be preserved. Existing original capital management,
sizing, and risk behavior must be preserved. Category-A operational changes
introduced after the baseline must be restored to the baseline semantics;
observability, governance, Shadow, test-only, and mutation-guarded read-only
audit facilities remain separate and may be preserved when they do not alter
the original operational path.

Prospective evidence will be collected before any new capital or risk
restriction is considered. A future restriction requires a separate,
explicit Owner authorization.

## Explicit Non-Authorization

The Owner does not authorize a newly invented USD 16 capital policy or any
additional operational maximum, reservation, concurrency limit, margin-mode
requirement, stage-exhaustion rule, or trading guard.

This authority defines no per-trade or lifecycle cap, concurrency cap, fee
reserve, funding reserve, slippage reserve, leverage value, isolated-only
margin rule, loss budget, exhaustion state, replenishment policy, or order
frequency limit.

The USD 16 technical stage and USD 100 formal stage remain declared future
stages only. Both remain inactive. This task does not approve the model for
Live, create a Live activation record, start a Live process, access private
exchange mutation paths, or authorize any real exchange order.

## Frozen Outcome

- Newly authorized limits: `NONE`
- Newly authorized guards: `NONE`
- USD 16 stage: `INACTIVE`
- USD 100 stage: `INACTIVE`
- Real-order authority: `NOT_AUTHORIZED`
- Automatic transition: `PROHIBITED`

