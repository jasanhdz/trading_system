# Aegis Unmanaged Live Stale-State Resolution 01

## Status

OWNER_AUTHORIZED; EFFECTIVE BEFORE CONTROLLED QUIESCE

This document authorizes only forensic reconciliation, mutation-guarded authenticated read-only Binance USD-M audits, a conditional stop-only quiesce of `01-Trading-Bot`, rollback by restarting that same PM2 service, and conditional retirement after every gate below passes. It authorizes no exchange mutation and no governed Live activation.

## Authority

- Historical E5 closure SHA-256: `e8825f9c70633b70ba29ce03bf59eeea0f1e7a53c211c1ddbac32814924eba09`.
- Prospective protocol SHA-256: `8e5a08107a373bb2aa8ece1d2b35ae3ea421605bae49ec17fd1fd9418e6b209a`.
- Model-qualification protocol SHA-256: `1e9346a857feb642c60780754243624c94487ac24f9c26be2e72d631d7bec8b6`.
- Shadow activation-record SHA-256: `46de14ea0b165575a5c0db1b11268b03587df43ab6ae29d5a94a32023131a7a4`.
- Qualified model-bundle SHA-256: `23b22403b70f7d6c385d1214e6543197f4ca4e57269af19b1013987891ed550a`.
- Prospective configuration SHA-256: `f944b0210b31928a519dc63459be3f1d53de811517dc1bbe9753596314579ec1`.
- Mutation-guarded read-only audit tooling commit: `672334381ece7c39437e678a99c652102f9fc7d8`.

## Triggering Inconsistency

The complete five-surface Binance audit reported one-way mode, no active position, no regular or algo order, and internally consistent account surfaces. The local symbol state reported `mode=IDLE` and `lastBracketStatus=PENDING`. This disagreement prevented retirement.

The uniquely resolved local file is:

- Absolute path: `/home/jasan/Develop/trading_system/binance-futures-bot-ts/data/state_PROD_AEGIS_STATE_JSON_DOGEUSDT.json`.
- SHA-256: `ae978ded240b90366200c52d573d3d1fa97799c942d2575cafbbbf46ae0a78e1`.
- Persisted type: `BotState` from `src/domain/types.ts`.
- Reader/writer: `FsStateStore` in `src/infra/logging/FsStateStore.ts`, instantiated by `src/main.ts` with key `aegis_state.json` and symbol-scoped by `TradingService.stateForSymbol`.
- PM2 owner: `01-Trading-Bot`, ID `0`, working directory `/home/jasan/Develop/trading_system/binance-futures-bot-ts`.

No file with the same basename exists elsewhere under the TypeScript repository.

## Frozen State Semantics

`BotState.mode` permits `IDLE`, `LONG_RIDE`, or `SHORT_RIDE`. `lastBracketStatus` permits `PENDING`, `OK`, or `FAILED_CLOSED`.

The complete static source map establishes:

1. `lastBracketStatus=PENDING` is written only by `TradingService.attachOpenExchangePositionsToSymbolState`, after `readActivePosition` returns an actual position, in the same state update that changes mode from `IDLE` to `LONG_RIDE` or `SHORT_RIDE`.
2. `lastBracketStatus=OK` is written after entry, position confirmation, bracket submission, and bracket validation, in the same update that establishes a ride mode.
3. `lastBracketStatus=FAILED_CLOSED` is written after bracket failure and emergency close, in the same update that sets mode to `IDLE`.
4. Production operational code has no read site for `lastBracketStatus`. The only non-test read is the dedicated read-only forensic audit classifier.
5. Startup attaches exchange positions only when local mode is `IDLE` and only after `readActivePosition` confirms exposure. It does not act from `lastBracketStatus`.
6. The symbol loop invokes position management only when mode is not `IDLE`. When mode is `IDLE`, it follows normal entry evaluation; stale bracket metadata is not an input to entry eligibility or order construction.
7. Bracket creation, recreation, cancellation, stop movement, and position close paths are reachable only through a confirmed entry flow or non-IDLE position-management flow. `lastBracketStatus` cannot independently invoke them.
8. Exit paths set mode to `IDLE` using merge-patch persistence and generally do not clear historical entry or bracket metadata. Therefore an earlier `PENDING` value can survive after the exchange position and orders disappear.
9. The observed DOGE state contains no `lastOrderId`, exchange-order ID, algo-order ID, pending-mutation field, `bracketsArmedAt`, or `bracketsAttached`. Its `probeModeActive` is false.
10. `FsStateStore` writes through a temporary file and atomic rename. A present `*.tmp` file is a mutation-in-flight indicator; none was observed at authorization time.

Accordingly, `lastBracketStatus` is historical lifecycle metadata, not an independently actionable pending-order instruction. `mode=IDLE` suppresses position management and bracket recreation.

## Operational Classification Contract

The local value may be classified `STALE_TERMINAL_BRACKET_RESIDUE` only if all of these remain true immediately before quiescence:

- the exact file and hash-bound forensic copy are unambiguous;
- mode remains `IDLE` and no local managed-position or persisted order identity exists;
- no temporary state write, child exchange worker, unmatched recent mutation marker, or unresolved retry is observed;
- a fresh five-endpoint audit is `COMPLETE`, one-way or hedge mode is known, all exact position amounts are zero, all regular/algo/unknown order counts are zero, account surfaces are consistent, and all mutation counters are zero;
- a complete running-process census finds no second Live-capable process;
- the PM2 recovery snapshot is complete and Shadow remains healthy.

Any failed or unverifiable condition produces `AMBIGUOUS_LOCAL_PENDING_STATE` or `ACTIVE_LOCAL_PENDING_OPERATION`; the service remains online.

## Forensic Preservation

Before stopping, create a byte-identical copy beneath `data/live_reconciliation/unmanaged_service_01/forensic/`, set restrictive permissions, and verify source and copy hashes. Preserve the original file, logs, journals, PM2 descriptor, and all audit reports. Do not edit, truncate, delete, or normalize the original or copy. Record any natural shutdown-time source change as a new hash without replacing the original evidence.

## Controlled Quiesce And Rollback

After all pre-quiesce gates pass, persist a quiesce-intent record and run only `pm2 stop 01-Trading-Bot`. Keep the PM2 entry and startup persistence intact. Verify bounded clean shutdown, PID and child exit, Shadow continuity, then run the complete five-endpoint audit immediately.

If exposure, an order, audit ambiguity, unexpected stop behavior, another Live process, or Shadow drift appears, do not delete the PM2 entry. Restart only the existing `01-Trading-Bot` entry from its captured descriptor, verify it is online, preserve evidence, and require owner review. No exchange repair is authorized.

When the immediate audit is flat, keep the service stopped for exactly 60 seconds, observe process/supervisor state, and repeat the complete audit. Any delayed exposure or order triggers the same rollback.

## Safe-Retirement Gate

Retirement requires confirmed `STALE_TERMINAL_BRACKET_RESIDUE`, verified forensic preservation, complete flat pre-stop and both post-stop audits, no mutation, no child or alternate Live process, and healthy unchanged Shadow. Then delete only `01-Trading-Bot`, run user-level `pm2 save`, and preserve application files, state, logs, journals, and credentials.

After deletion, observe for exactly 10 minutes. Periodically inspect PM2 and the process table. The old executable or a credential-enabled duplicate must not respawn. At the end, run the final complete five-endpoint audit. Any respawn or account-state change blocks readiness and requires owner review.

## Exchange-Mutation Prohibition

Only the five committed USER_DATA GET methods are authorized. POST, PUT, PATCH, DELETE, TRADE endpoints, order create/test/cancel/modify, position close/reduce, leverage or margin changes, position-mode changes, transfers, withdrawals, user-data trading streams, and account-setting changes remain prohibited. Every mutation counter must remain zero.

## Shadow Preservation

PM2 ID `20`, `aegis-prospective-shadow-cohort-1`, must remain online in exact `SHADOW` mode. Its activation, model, and configuration hashes must remain unchanged. Journals and checkpoint must remain valid; private-call, credential-read, and order-call counters must remain zero. Shadow must not be restarted or share Live state.

## Governed Live State

The governed Live activation record remains absent. `approved_for_live=false`; USD 16 and USD 100 stages remain `INACTIVE`; the governed order adapter remains disabled; automatic transition remains prohibited. Successful retirement permits only `READY_FOR_SEPARATELY_AUTHORIZED_USD16_LIVE_PREACTIVATION`.

## Owner Authorization

The owner authorized this bounded stale-state forensic resolution, stop-only quiesce, exact rollback, conditional PM2 retirement, and fixed observation windows in the task titled **AEGIS — Stale Local Live State Forensic Reconciliation, Controlled Quiesce, and Conditional Safe Retirement Retry**. This authorization does not activate governed Live and does not authorize an exchange mutation.
