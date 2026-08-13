# Aegis Market Microstructure Events C2 - Preregistration

## Objective

C2 acquires genuinely new point-in-time public market information and tests
explicit mechanisms instead of training another committee over OHLCV-derived
features. It extends Market Event Lab M1 and preserves C1 as the frozen
baseline.

## Current Evidence Boundary

The local database contains 540 days of futures klines and funding for all 11
symbols, but only 1.74 days of open interest, one depth observation per symbol
and no liquidation events. These sources cannot support an economic result.

Binance aggregate trades can be backfilled from public archives. Open interest
is treated as recent/prospective context. Liquidation and depth observations
are collected prospectively from public streams. The liquidation stream is a
snapshot of the latest liquidation per symbol per interval, not a complete
liquidation tape; C2 records this limitation in every manifest.

## Frozen Event Families

1. Open-interest-confirmed breakout.
2. Liquidation absorption reversal.
3. Liquidation continuation.
4. Depth absorption reversal.
5. Aggressor-flow impulse continuation.
6. BTC/ETH-to-altcoin lead-lag.
7. Spot/futures dislocation as a frozen control.

Each family must pass independently. Interactions and committees are forbidden
until one family demonstrates stable net utility against random, price-only
and C1 controls.

## Collection Integrity

- Public unauthenticated sources only.
- Exact endpoint and stream allowlists.
- Natural-key deduplication.
- Original exchange timestamps preserved.
- No synthetic zeros, future fill or retrospective timestamp correction.
- Validated canonical rows only; no credentials or private payloads.
- Restrictive file permissions and append-only chained manifests.
- Missing or stale required sources block the affected event family.

## Evidence Gates

Seven days permit only a technical pilot. Discovery requires at least 60 days
and 300 independent events per family and side. Positive net expectancy,
positive day-cluster confidence lower bound, profit factor above 1.10,
20-bps stress survival and superiority to all controls are required.

C2 cannot authorize Shadow or Live. Its full frozen source schemas, event
semantics, costs, horizons and gates are defined in
`config/experiments/aegis_market_microstructure_events_c2.yaml`.
