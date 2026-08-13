# Aegis Event Path Quality C2A - Preregistration

## Question

C2A tests whether authentic taker-flow events improve prediction of trade path
quality, not merely terminal direction. The primary question is whether an
event reaches a favorable barrier before an adverse barrier with bounded MAE,
prompt positive excursion and positive net utility after costs.

## Frozen Data

The source is checksum-verified public Binance USD-M one-minute klines and
aggregate trades for all 11 canonical symbols. The intended acquisition range
is August 2025 through July 2026. June and July 2026 form the first bounded
technical batch; they cannot independently authorize model selection.

Open interest, forced liquidations and depth remain separate optional source
families. Missing observations cannot be represented by zero or reconstructed
from candles.

## Frozen Outcomes

- Favorable barrier before adverse barrier.
- Conservative adverse-first handling when both barriers occur in one minute.
- MAE and MFE over 15, 60 and 240 minutes.
- Whether MFE precedes MAE.
- Time to first favorable excursion and time to MFE.
- Terminal side return and net utility under 8, 14 and 20 bps costs.

The entry is the next complete one-minute open after the event minute. Future
bars may produce labels only and may never enter event features.

## Scientific Boundary

Train, validation and final holdout dates are frozen in
`config/experiments/aegis_event_path_quality_c2a.yaml`. Flow thresholds are
fit on TRAIN only. Event overlap is reduced with a fixed 15-minute cooldown.
Final holdout may be opened once and cannot be used for repeated threshold or
feature selection.

C2A must outperform matched random, price-only, unfiltered-event and the frozen
C1 baseline. Positive average return alone is insufficient: confidence bounds,
20-bps stress, symbol breadth, temporal stability and bounded concentration
must also pass.

## Authority

C2A is research-only. It cannot alter Runtime, PM2, Live, Shadow, TypeScript,
capital, guards or exchange state. No successful result automatically permits
modeling, Shadow or Live promotion.
