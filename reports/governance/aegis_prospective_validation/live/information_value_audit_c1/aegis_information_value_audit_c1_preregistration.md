# Aegis Information Value Audit C1 - Preregistration

## Purpose

C1 determines whether any currently available causal information family adds
stable directional, economic or path-risk information beyond a transparent
price-state baseline. It is an information audit, not another committee.

Every source is evaluated separately before interactions are permitted. A
large model cannot compensate for an input family that carries no incremental
out-of-sample information.

## Available Families

- Price state: returns, volatility, moving-average distances, extension,
  breakout acceptance and wick rejection.
- Flow activity: public-kline taker imbalance and volume persistence.
- Derivatives carry: mark/spot basis, basis convergence and funding.
- Cross-market context: BTC state, causal beta, relative strength, breadth and
  common altcoin state.
- Calendar controls: hour and weekday cycles.

Open interest, liquidation events, historical order book and point-in-time news
are not available with the required aligned history. C1 prohibits filling these
sources with zeros, proxies or retrospectively collected values.

## Frozen Questions

For each available family, C1 asks whether adding it to PRICE_STATE improves:

1. timestamp-grouped ranking of residual utility;
2. favorable-before-adverse barrier probability;
3. MAE estimation;
4. top-decile net expectancy after 14 and 20 bps costs.

The same frozen model classes, partitions and metrics are used for every
candidate. No family interactions, hyperparameter search, threshold tuning,
seed selection, symbol removal or side removal are permitted.

Each model is fit separately by side, horizon and candidate. Residual value
uses standardized ridge regression with `alpha=10`; barrier quality uses
standardized class-balanced logistic regression with `C=1`; MAE uses histogram
gradient boosting with 100 iterations, 15 leaves, learning rate `0.05` and L2
regularization `1`. Top-decile economics uses the CALIBRATION q90 predicted
residual-value threshold and at most one selected symbol per timestamp and
side.

## Evidence Limits

The source history has already informed A1, A2, B1 and B2. C1 can reject data
families and justify acquisition or a new C2 experiment, but cannot authorize
Shadow or Live. All deployment flags remain false regardless of results.

The complete feature names, model parameters, gates, partitions, costs and
source hashes are frozen in
`config/experiments/aegis_information_value_audit_c1.yaml`.

## Safety

C1 is offline and read-only with respect to market archives. It performs no
network, exchange, PM2, Live, Shadow, credential, order, position or capital
operation.
