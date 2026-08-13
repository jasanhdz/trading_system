# Aegis Market Microstructure Events C2 - Readiness

## Verdict

`C2_COLLECTION_INFRASTRUCTURE_READY_DATA_PENDING`

C2 now has a causal, idempotent archive for public aggregate trades, forced
liquidation snapshots, top-20 depth snapshots and open-interest snapshots. It
also has strict event contracts for the six preregistered experimental
families. This is collection and experimental infrastructure, not evidence of
economic edge and not authority for modeling, Shadow or Live.

## Implemented Boundary

- Exact 33-stream allowlist for 11 symbols across aggregate trades, forced
  liquidation snapshots and top-20 depth snapshots.
- Public, unauthenticated open-interest polling with an exact endpoint.
- Canonical typed rows with finite-value, symbol, timestamp and causality
  validation.
- Natural-key deduplication and SHA-256-chained collection manifests.
- Checksum-identified import of local Binance aggregate-trade ZIP archives.
- Strict feature order and source-time checks for every C2 event vector.
- Fail-closed family readiness derived from actual source coverage.
- No credentials, private account payloads or exchange mutation capability.

Implementation commit: `f1c744a400d37c73bad343a957ae8514b2546ffb`.

## Authentic-Data Pilot

The only locally available aggregate-trade archive was imported twice:

| Item | Result |
| --- | ---: |
| Symbol/month | `ADAUSDT`, `2026-07` |
| Archive SHA-256 | `bc05d0f7c936a84420aa0733d32d3ced74852f21793788424001664a7b2c7a6f` |
| First import accepted | 2,574,816 |
| First import duplicates | 0 |
| Second import accepted | 0 |
| Second import duplicates | 2,574,816 |
| Canonical span | 30.99999 days |
| Authenticated requests | 0 |
| Exchange mutations | 0 |

The second pass demonstrates exact idempotency at realistic volume. The pilot
does not satisfy the preregistered evidence requirement because it covers only
one symbol, one source family and approximately 31 days.

## Coverage Gate

| Source | Rows | Symbols | Span | State |
| --- | ---: | ---: | ---: | --- |
| C2 aggregate trades | 2,574,816 | 1 | 31.00 days | Insufficient |
| C2 forced liquidations | 0 | 0 | 0 days | Missing |
| C2 depth snapshots | 0 | 0 | 0 days | Missing |
| C2 open interest | 0 | 0 | 0 days | Missing |
| Legacy futures klines | 1,710,710 | 11 | 540.00 days | Available |
| Legacy funding | 17,820 | 11 | 539.67 days | Available |
| Legacy mark price | 17,820 | 11 | 539.67 days | Available |
| Legacy open interest | 5,501 | 11 | 1.74 days | Insufficient |
| Legacy depth | 11 | 11 | 0.01 days | Insufficient |
| Legacy liquidations | 0 | 0 | 0 days | Missing |

Every experimental family remains `BLOCKED_COLLECTION_LT_60_DAYS`. No event
family was fitted or economically evaluated. Missing sources were not replaced
with zeros, candle proxies or retrospective data.

## Validation

- Focused C2 and M1 tests: `18 passed`.
- Full unit suite: `803 passed, 5 failed`. The five failures are unchanged
  branch-authority checks that require `feature/aegis-ts-clean-rebuild`; this
  isolated experiment runs on `work/entry-quality-evidence-20260726` and none
  of the failures reaches C2 behavior.
- Python bytecode compilation: `PASS`.
- Git whitespace validation: `PASS`.
- Real-volume archive import: `PASS`.
- Real-volume duplicate replay: `PASS`.
- `websocket-client 1.9.0` installed into the existing ROCm environment from
  the repository-declared `websocket-client>=1.6.0` requirement.
- `ruff`: unavailable in the intended environment.
- `black --check`: did not complete and was terminated; no formatter rewrites
  were made.

## Safety And Runtime State

- Model, features and trading decisions changed: `NO`.
- TypeScript changed: `NO`.
- PM2, Live or Shadow changed: `NO`.
- Public network collection started: `NO`.
- Credentials loaded: `NO`.
- Authenticated exchange calls: `0`.
- Exchange mutations: `0`.
- Automatic promotion: `DISABLED`.

## Next Evidence Step

Run the collector continuously as a separate research-only process and acquire
checksum-verified aggregate-trade archives for all 11 symbols. A seven-day
sample may validate operations only. Discovery remains blocked until required
sources cover at least 60 days and each family/side has 300 independent events.
Only then should each family be evaluated singly against random, price-only and
the frozen C1 baseline. Combining families remains prohibited until one family
transfers independently.
