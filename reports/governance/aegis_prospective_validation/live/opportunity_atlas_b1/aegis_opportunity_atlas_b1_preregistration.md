# Aegis Opportunity Atlas B1 - Preregistration

## Change Of Architecture

B1 does not ask one model to predict every candle. It treats each hourly
timestamp across all 11 symbols as one correlated market event and asks four
separate questions: whether sufficient gross opportunity exists, which side
contains it, which symbol ranks highest and how adverse the path is likely to
be.

Abstention is the default. A combined diagnostic may select at most one
symbol-side per independent event and only after the opportunity, direction,
ranking and path-risk components agree.

## Difference From V20

V20 began with side-specific V9/V14 opportunity families and none passed its
viability gate. B1 begins direction-agnostic, separates BTC-wide movement from
symbol residual movement, uses one cross-symbol event identity and evaluates
each predictive question independently before combination.

## Frozen Experiment

Features, partitions, model classes, seeds and policy thresholds are frozen in
the machine-readable protocol. Hyperparameter search and validation threshold
mining are prohibited. The evaluation includes realistic costs, funding,
MAE/MFE and timestamp/day clustered uncertainty.

All source history has been inspected by prior Aegis research. B1 is therefore
an architecture and separability diagnostic, not untouched evidence. It may
justify a separately preregistered forward experiment but cannot authorize
Shadow or Live.

## Safety

B1 is offline. It performs no authenticated request and cannot change PM2,
Live, Shadow, models, capital, credentials, orders or positions.
