# W13-P Legacy Core-Path Pilot

Date: 2026-08-16

## Scope

This is descriptive instrumentation validation, not TRAIN, threshold discovery, or an
economic verdict. The sample contains ten post-fix, quality-v1 SHORT bundles: five
ADAUSDT and five LINKUSDT. All have complete logical windows, valid sequential L2 diff
events, BBO and raw trades, but no causal full-book base snapshot. They therefore do not
count toward quality-v2 W13 sample minimums.

## Early path

Directional values are oriented to SHORT and measured from the frozen T0 reference mid.

| Horizon | Median return | Favorable fraction | Median MFE | Median MAE |
| ---: | ---: | ---: | ---: | ---: |
| 30s | -2.12 bps | 20% | 0.00 bps | 3.18 bps |
| 60s | 0.00 bps | 40% | 4.43 bps | 3.19 bps |
| 180s | +5.64 bps | 60% | 5.66 bps | 5.48 bps |

Only one episode achieved more than 14 bps MFE in 180 seconds. The same single episode
ended above 14 bps; none reached +25 bps before -14 bps. On a symmetric 14-bps first
barrier, one was favorable-first and nine reached neither barrier.

The first observed BBO change occurred 10.5-39.1 seconds after T0. The first raw trade
occurred 10.9-39.9 seconds after T0. Thus the 100ms-5s decision states frequently contain
no new market event for these symbols/episodes; high timestamp resolution does not imply
immediate information arrival.

## Flow and BBO observations

- Favorable-first occurred in 5/10 episodes when comparing the first nonzero favorable
  and adverse BBO displacement.
- Spearman correlation between directional trade-flow imbalance and contemporaneous
  return was approximately 0.54 at 30s and 0.50 at 60s.
- T0 L1 BBO imbalance versus 180s directional endpoint had Spearman approximately 0.32.

These correlations are unstable descriptive observations at N=10, not evidence. They
only justify retaining flow x price-response and BBO state as preregistered feature
families once the full sample exists.

## Instrument finding

Absolute OBI L5/L10/L20, depth, depletion and replenishment cannot be reconstructed from
diffs alone without a causal base snapshot. Quality gate v2 now persists a full L2
checkpoint at or before T0-30s plus subsequent diffs. The ten legacy bundles remain
usable for path/trade-flow diagnostics but are excluded from full W13 learning.

## Conclusion

The pilot does not show economically useful confirmation. Typical 180-second favorable
excursion is below the 14-bps baseline cost, and early favorable/adverse ordering is
approximately balanced. The only defensible learning is methodological: information
arrives slowly in these episodes, flow-response remains worth measuring, and complete
L2 replay requires a causal base checkpoint. No trading rule or gate is justified.
