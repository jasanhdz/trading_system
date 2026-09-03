# Aegis Scientific Brain Implementation

## Scope

This implementation realizes the clean boundary defined by the architecture:
Python validates coordinated market snapshots, computes scientific features,
runs an immutable model bundle, applies the ordered scientific layers, compares
all eleven symbols, freezes a decision, and records evidence. TypeScript owns
all operational authority and only validates the scientific contract before
passing an allowed proposal to its existing gates.

This version is validated offline. The included `aegis-offline-reference-v1`
bundle is deterministic test/reference material and is not an authorization to
trade or a production promotion.

## Final Tree

```text
config/
  brain.yaml
  universe.yaml
  models.yaml
  bundles/aegis-offline-reference-v1.json
src/aegis/
  domain.py       immutable contracts and parsers
  config.py       frozen configuration and universe handshake
  features.py     snapshot validation and shared causal features
  models.py       checked bundle and deterministic inference
  layers.py       D3, RV2, TRRM, QMAE, EQM, ECON1
  decision.py     candidates, global selection, decision freeze
  evidence.py     append-only hash-chained evidence
  runtime.py      composition and A-Z orchestration
  api.py          five scientific HTTP endpoints
  training/       causal dataset, train, evaluate, registry
  utils/          canonical hashing and UTC clocks
binance-futures-bot-ts/src/brain/
  contract.ts
  client.ts
  manifest.ts
  decision-gate.ts
```

## Flow A-Z

1. TypeScript sends one coordinated 5m snapshot for the canonical eleven.
2. Python checks contract, universe hash, closed bars, ordering, gaps, OHLC,
   history depth, future timestamps, and freshness.
3. `DeterministicFeaturePipeline` computes individual and cross-sectional
   features in one pass for all symbols. Training imports this same pipeline.
4. A content-hashed bundle supplies the frozen normalizer and estimator heads.
5. D3 assigns deterministic market context; RV2 aggregates tail probability;
   TRRM computes temporal/risk compatibility; QMAE estimates adverse-excursion
   quality; EQM combines clean probability with directional edge; ECON1 removes
   frozen round-trip cost.
6. One candidate is built per symbol. The selection policy applies scientific
   and limited portfolio gates, ranks globally, resolves ties by symbol/hash,
   and selects at most one candidate or returns first-class `NO_TRADE`.
7. The freeze uses canonical SHA-256 inputs and snapshot time, so an identical
   request produces the same IDs and response.
8. Evidence is append-only and hash chained. Later TypeScript outcomes add new
   evidence and never mutate the frozen decision.
9. TypeScript validates manifest, response age, hashes, duplicate decisions,
   authorization, kill switch, slots, symbol, and side before yielding to the
   existing operational entry flow.

## Feature Formulas

The compact schema contains 39 justified causal features: price/log returns at
1/3/6/12/24 bars; candle range/body/wicks/close position; volume change,
z-score and ratio; range and true-range summaries; volatility ratio; EMA gaps
and slope; momentum acceleration; return z-score; directional persistence;
chop, trend strength and range expansion; plus relative returns, rank,
dispersion, breadth, market direction, concentration, and BTC/ETH divergence.

Every denominator is guarded, all outputs must be finite, windows are backward
looking, and no inference-time statistics replace the bundle normalizer.

## Scientific Semantics

- **D3:** the old name described the causal data contract. The clean architecture
  assigns D3 the explicit context/regime output while snapshot validation retains
  the old causal-data responsibility.
- **RV2:** mean calibrated tail probability across approved estimators.
- **TRRM:** `1 - tail_probability`; the configurable tail veto preserves the
  historical global p70/30-percent rejection operating concept.
- **QMAE:** q90 adverse excursion and quality relative to a frozen maximum.
- **EQM:** clean probability multiplied by positive directional expected edge.
- **ECON1:** directional expected edge minus frozen round-trip friction.
- **Composite score:** direction confidence, D3 confidence, TRRM compatibility,
  QMAE quality, clean quality, and ensemble agreement. Every factor is bounded.

## Models and Training

Runtime estimators are isolated from selection and expose standardized direction
probabilities, expected return, tail risk, q90 adverse excursion, clean quality,
and uncertainty. Bundle loading checks a canonical content hash, feature schema,
universe, timeframe, and bundle ID.

Offline training builds rows with the runtime feature pipeline, offers expanding
walk-forward folds with a 120-minute embargo, uses deterministic ridge linear
fits, evaluates explicit metrics, and publishes only explicitly accepted,
immutable artifacts. Evaluation never trains or auto-promotes a model.

## Contracts and Configuration

The contract version is `aegis-clean-rebuild-v1`. The canonical symbols are:
`ETHUSDT, BTCUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT,
AVAXUSDT, LINKUSDT, SUIUSDT, LTCUSDT`; the timeframe is `5m`; the ordered
symbol hash is `f6448e67daf1d017e16cc6b331f6494e97e178824474994fff08864303ccd348`.

`regimen.config.yaml` contains only integration expectations and keeps execution
disabled. Scientific thresholds, normalizers, and layer parameters remain in
Python configuration/bundles.

## Tests and Current Limits

Python tests cover immutable contracts, exact universe, tamper detection,
snapshot rejection, feature determinism, model compatibility, all layers,
global ranking, idempotent freeze, evidence, API routes, temporal training,
evaluation, and immutable publication. TypeScript tests cover strict parsing,
shared manifest serialization, handshake mismatches, timeout adapter behavior,
NO_TRADE, stale/duplicate decisions, kill switch, symbol allowlists, and disabled
execution. The full fixture is offline and does not contact Binance.

The configured Python environment does not include the `coverage` package, so
line coverage was not measured and no dependency was installed to manufacture a
number. Parse/import checks and the complete available Python/TypeScript suites
were executed instead.

Remaining work is operational validation: selecting and freezing a production
artifact after independent research acceptance, wiring the client into the
existing TypeScript orchestration under a separate authorization phase, load and
failure testing the HTTP service, and collecting forward evidence. None of those
steps are performed here.
