# Aegis SHORT Reversal Exit Compatibility X1A Preregistration

X1A preserves the X1 economic hypothesis and gate but corrects a physical data
coverage gap discovered before X1 opened any evaluable event. It is a separate
experiment and does not rewrite X1.

The V21 mathematical flow contract remains unchanged. For every closed
five-minute bar, X1A takes base volume from the Binance public candle delta and
taker-buy base volume from the Binance public microstructure database, joined
exactly by symbol and open timestamp. It computes the same local and
eleven-symbol market taker-flow formulas as V14. A timestamp is eligible only
when all eleven symbols have 24 contiguous closed bars ending five minutes
before entry. Missing, non-finite, misaligned, or physically invalid volume is
omitted fail-closed; no zero fill or synthetic value is allowed.

This is an explicit physical-source change from the historical V21 dataset,
not a claim of byte-identical source rows. The entry thresholds, ranking,
SHORT side, spacing, candidate `CURRENT_TS` exit, `LOCK_AT_5_ROE` control,
cost stress, uncertainty procedure, and economic gates remain frozen before
X1A evidence is opened.

X1A is research-only. It cannot alter Live, Shadow, TypeScript, PM2, exchange
state, guards, sizing, leverage, capital, or exits. Passing would justify a
separate prospective Shadow authorization only; it would not authorize Live.
