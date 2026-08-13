# Aegis Market-Event Economic Path M1B

M1B is a new research-only experiment. It does not repair M1A by changing its
thresholds after seeing results. It tests whether genuinely additional causal
information and explicit path-risk estimation can select a useful subset of
three populations disclosed in advance.

## Why This Experiment Exists

M1A found weak gross directional information but no net economic edge. V20 and
V21 independently showed that high win rate and attractive candle patterns did
not survive the loss distribution, costs and temporal changes. More committee
layers over the same candle information are therefore not justified.

M1B adds verified mark/spot basis and funding, keeps causal taker flow and
regime, and asks three separate questions: is a protected outcome likely to be
positive, how adverse can its path become, and what is its expected net utility?
The models may abstain. They may not invent a new direction or alter the
underlying candidate definitions.

## Populations

The populations are frozen before M1B data inspection:

1. M1A Spot/Futures dislocation LONG;
2. M1A compression breakout LONG;
3. V21 extreme reversal SHORT using current TypeScript protection as outcome.

Their selection used prior retrospective evidence and is explicitly
contaminated for promotion purposes. All history through July 2026 is
development evidence. Only observations beginning 2026-08-13 can eventually
provide fresh forward evidence, and even that requires a separate Shadow
review.

## Economic Contract

The primary outcome is the worst result across the two deterministic OHLC
intrabar paths under the existing TypeScript hard stop, take profit,
break-even, ATR trailing and fixed callback fallback. Fees, slippage and
funding are included. A positive fixed-horizon return alone cannot pass M1B.

The first permitted models are regularized logistic regression, quantile
gradient boosting for MAE and robust gradient boosting for protected net
utility. Deep learning, reinforcement learning, feature search, seed search
and validation threshold search are prohibited.

## Data Boundary

Required inputs are checksum-verified public futures klines, spot klines, mark
price klines and funding rates. Open interest, historical depth, liquidations
and tick aggTrades remain absent unless verified sources are available. Missing
sources are marked `NOT_PRESENT`; they are never filled with fabricated zeros.

## Promotion Boundary

Retrospective success cannot authorize Shadow or Live. M1B may only prepare a
forward collector when the feature contract, replay, model pipeline and
anti-leakage tests pass. No runtime model, process, configuration, PM2 service
or exchange state is changed by this experiment.
