# Aegis Prospective Shadow Activation 01

## Status and authority

Status: `OWNER_AUTHORIZED_PENDING_ATOMIC_ACTIVATION`

Activation authority: the owner instruction titled `AEGIS - PROSPECTIVE ACTIVATION BOUNDARY AND PERSISTENT SHADOW COHORT 1 START`.

Protocol: `aegis-prospective-validation-v1`, SHA-256 `8e5a08107a373bb2aa8ece1d2b35ae3ea421605bae49ec17fd1fd9418e6b209a`.

Cohort: `aegis-prospective-shadow-cohort-1`.

This authorization permits persistent public-market Shadow observation only. It does not authorize Live, private exchange access, credentials, signed requests, real balances, positions, or order operations.

## Frozen identities

- Qualified scientific Python commit: `fbb6c60f1c0ae50b184665f4afab4091618dd5b7`.
- Qualified operational TypeScript commit: `38a2ddf03432ff5bfadfd0edf86c0aa0b410b75e`.
- Model identity: `aegis-prospective-shadow-candidate-v1`.
- Model-bundle SHA-256: `23b22403b70f7d6c385d1214e6543197f4ca4e57269af19b1013987891ed550a`.
- Trained-artifact SHA-256: `386742c20d74a3b67d47cd95629c646195472e05e9e8d136587d40989a82e3d1`.
- Feature contract: `aegis-features-v2`, SHA-256 `2dc278b4353585fe22503233187e12832cabfd67e2a2e58f4cd683ee6f3b9454`.
- Configuration SHA-256: `f944b0210b31928a519dc63459be3f1d53de811517dc1bbe9753596314579ec1`.
- Label-contract SHA-256: `d1cbd83874d9823be2db9931052818d36a32ebbfde2625e83b0cf7403ab1e66d`.
- Signal-schema SHA-256: `636c3cdac150d455d2b3a68e3ae17de04d9d0768c3eb77d281b7b719f70aa59e`.
- Outcome-schema SHA-256: `c31496d5121d6aebce14a9641e219779ba89918c7b97ba914acc963ac411466a`.

Activation-only observational code may be committed after this document. The atomic activation record must bind the final Python and TypeScript repository commits. Those changes may not modify model weights, features, D3, RV2, TRRM, QMAE, EQM, ECON1, final Aegis decisions, risk semantics, or Live isolation.

## Universe and public data

The ordered symbol universe is `ETHUSDT`, `BTCUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `SUIUSDT`, and `LTCUSDT`. Its ordered hash is `f6448e67daf1d017e16cc6b331f6494e97e178824474994fff08864303ccd348`.

The interval is coordinated final `5m` candles. The public source is Binance USD-M Futures. The only permitted endpoints are HTTPS `GET https://fapi.binance.com/fapi/v1/klines` and public WebSocket `wss://fstream.binance.com/ws/<stream>`, as constrained by `aegis-shadow-endpoint-policy-v1`. Private, account, balance, position, listen-key, leverage, margin, and order paths are denied.

## Frozen Shadow economics

- Entry: signal close; simulated fill reporting follows the prospective protocol.
- Outcome horizon: 12 complete final 5-minute bars.
- Fee: 4 basis points per side.
- Slippage: 1 basis point per side.
- Round-trip cost: `0.001` return fraction.
- Funding: exact frozen V1 prospective label policy, zero for this cohort.
- Shadow balance: deterministic synthetic notional `aegis-shadow-synthetic-usd-10000-v1`, USD 10,000 virtual only.
- Sizing policy: `aegis-shadow-qualified-risk-fraction-v1`; model and operational risk outputs are observed unchanged and applied only to the synthetic notional.
- Every hypothetical intent must declare `execution_mode=SIMULATED_SHADOW` and cannot reach an exchange adapter.

## Inclusion boundary

The cohort begins at the activation timestamp durably stored in `shadow_cohort_1_activation.json`. No event may be accepted, buffered, or backfilled before that timestamp. Qualification and smoke events remain `PREACTIVATION_NON_COHORT`. A preactivation event fails `PROSPECTIVE_EVENT_BEFORE_ACTIVATION`.

Every coordinated final-candle candidate evaluation after activation is included regardless of action or later outcome. Outcome information, profitability, ranking, and operator preference cannot alter inclusion.

## Journals and checkpoints

Signal, intent, market-cache, outcome, checkpoint, logs, and mutable health files live under ignored `data/prospective_shadow/cohort_1/`. Canonical journals are append-only UTF-8 JSONL, duplicate-safe, conflict-safe, flushed and fsynced before downstream simulated intent handling. Checkpoints bind activation, code, model, configuration, journal-chain, sequence, and completed-cycle identities. A write or integrity failure stops or quarantines the cohort.

The outcome maturator consumes only sealed cohort evidence, waits for complete H12 data, persists every target row, and never creates or re-evaluates a signal.

## Cohort minimums

All conditions are required:

1. at least 14 consecutive calendar days of operation;
2. at least 100 fully persisted candidate evaluations;
3. at least 5 `ENTER_NOW` simulated intents;
4. every required `ENTER_NOW` outcome matured;
5. at least one validated controlled restart and checkpoint resume;
6. at least one validated public-feed reconnect;
7. complete evidence for 100 percent of evaluations;
8. complete immutable outcomes for 100 percent of matured candidates;
9. zero private exchange calls, credential reads, real orders, and money movement.

The maximum automatic observation window is 30 calendar days from activation. If minimums are unmet then, the cohort enters `SHADOW_COHORT_REVIEW_REQUIRED`; it does not extend automatically.

## Technical success and review

Technical success requires valid activation/code/model/config hashes, healthy public connectivity, deterministic checkpoint recovery, durable journals, complete component envelopes, bounded reconnect behavior, zero prohibited operations, and no unresolved identity conflict. These are operational gates, not profitability gates.

Routine public-feed loss may reconnect with bounded exponential backoff of 1, 2, 4, 8, and 16 seconds. Five consecutive startup/runtime failures quarantine the cohort. Integrity failures never auto-resume and require a separately owner-authorized recovery record.

## Stop conditions

Stop or quarantine on activation, protocol, code, model, feature, configuration, journal, checkpoint, clock, disk, or identity integrity failure; credential-variable detection; signed/private/user-stream request; private or Live adapter reachability; order operation; prohibited data access; or restart-loop exhaustion.

## Live prohibition

Completion can produce only `READY_FOR_SEPARATELY_AUTHORIZED_SHADOW_EVALUATION` or `SHADOW_COHORT_REVIEW_REQUIRED`. It cannot activate the USD 16 stage, USD 100 stage, credentials, private endpoints, Live approval, or an order adapter. Both budgets remain `INACTIVE`, `approved_for_live=false`, and automatic transition is prohibited.
