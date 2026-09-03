# E4 Feature Contract

The generated schema is hash-bound in
`artifacts/dataset_v1/feature_schema.json` with SHA-256
`dc87edf9f31fc9dae9201cb800e601b086925eda77b4026f1c515a326b6ed49c`.

## Families

- `BASE`: direction-normalized price, return, ATR, RSI, EMA, trend age,
  persistence, volatility, volume and causal recent-structure context at
  5m/15m/1h/4h.
- `FLOW`: real taker-buy/sell proportions, imbalance trajectory, persistence,
  price response, impact acceleration and impact decay.
- `CROSS_MARKET`: BTC/ETH impulses, agreement, breadth, dispersion, relative
  return and propagation lag.
- `REMAINING_MOVE`: consumed ATR move, impulse age, extension, RSI room,
  favorable/adverse structural space and momentum decay.
- `QUALITY`: warmup and source-availability flags.

Every timeframe carries an availability timestamp at or before `decision_at`.
Flat candles use an explicit zero-range flag and a documented neutral numeric
representation. Other non-finite required states fail closed; 190 rows were
excluded this way.

The allowlist rejects names containing future outcomes, MFE, MAE, PnL, labels,
barrier outcomes or exit information.
