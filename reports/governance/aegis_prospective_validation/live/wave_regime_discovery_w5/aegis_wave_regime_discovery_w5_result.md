# Aegis W5 GOOD/BAD Wave Regime Result

## Status

`AEGIS_WAVE_REGIME_W5_NO_ROBUST_ECONOMIC_REGIME`

```text
W5_REGIME_DIFFERENTIATION_FOUND = TRUE
W5_ECONOMIC_REGIME_EDGE_FOUND = FALSE
W5_READY_FOR_FUTURE_CONFIRMATION = FALSE
W5_READY_FOR_SHADOW = FALSE
W5_READY_FOR_LIVE = FALSE
```

Observable context distinguishes some GOOD and BAD characteristics, but no
identified regime produced positive economic expectancy. Existing validation
and holdout periods were not opened.

## Population

| Partition | Episodes | Correlation clusters | GOOD | BAD | Net expectancy | Median MFE | Median MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | 59,596 | 8,214 | 14.51% | 64.57% | -13.65 bps | 0.286 ATR | 0.307 ATR |
| Validation | 62,773 | 8,361 | 12.99% | 66.38% | -14.05 bps | 0.273 ATR | 0.312 ATR |

All 11 symbols and both directions were retained. Every symbol-side validation
cell was negative; results were not rescued by excluding weak symbols.

## What Distinguished GOOD From BAD

The strongest descriptive differences known at entry were:

- GOOD occurred more often when current and 15m ATR fractions were larger.
- GOOD had slightly more 15m volume and more remaining RSI space.
- GOOD was less extended from MA25: median 0.60 ATR versus 0.81 ATR for BAD.
- GOOD had a smaller preceding directional move: 0.56 ATR versus 0.68 ATR.
- GOOD had slightly lower directional taker imbalance and lower short-window
  persistence. Stronger immediate aggression was not automatically safer.
- Weekends and calendar/session features showed association, but not economic
  edge and are not promoted as rules.

ATR dominated mutual information. This is partly mechanical: a +0.50 ATR
barrier is worth more bps in a high-volatility state and more easily clears a
fixed 14 bps cost. It does not mean ATR predicts the correct direction.

The logistic model reached ROC-AUC 0.825 and average precision 0.306 versus a
13.0% GOOD base rate. That classification result did not translate to profit.

## Frozen Candidate Validation

| Candidate | Validation N | GOOD | BAD | Net | 95% bootstrap interval | Stress 20 bps |
|---|---:|---:|---:|---:|---:|---:|
| Logistic top 20% | 12,600 | 31.56% | 67.33% | -14.29 bps | [-14.48, -12.35] | -20.29 bps |
| Logistic top 10% | 6,438 | 31.50% | 67.33% | -14.34 bps | [-15.45, -12.69] | -20.34 bps |

The model selected high-volatility episodes where outcomes became decisive:
GOOD approximately doubled, but BAD remained about two-thirds. It learned
“large outcome likely,” not “favorable outcome likely.” Both candidates had
zero positive symbols and zero positive temporal thirds.

No shallow-tree leaf or KMeans cluster had positive discovery expectancy, so
none was forwarded as a favorable regime.

## Cluster Findings

All six frozen clusters were negative in validation, ranging from -13.77 to
-14.35 bps. The least-bad cluster had moderate volume around 1.85x, weak or
opposed preceding direction, substantial RSI space and no breakout. It was
still decisively unprofitable.

The worst cluster had a mature aligned move: approximately 1.43 ATR preceding
directional displacement, 2.04 ATR MA25 extension, low RSI space, stronger
body/flow, trend alignment count 4 and a breakout proxy. Its validation result
was -14.35 bps. This supports the exhaustion concern descriptively, but does
not create a profitable inverse rule.

## Volume and Dangerous Contexts

- Moderate volume did not isolate edge; every fixed volume bin was negative.
- LONG volume >4x produced -14.53 bps in validation.
- SHORT volume 3-4x produced -14.70 bps; >4x produced -14.41 bps.
- Highest volatility quartile: -14.36 bps.
- Breakout proxy present: -14.33 bps.
- Most mature trend quartile: -14.20 bps versus -14.03 bps for the youngest.
- Extension >2 ATR had higher MAE and remained negative.
- BTC alignment did not improve expectancy: -14.08 bps aligned versus -14.02
  bps opposed/neutral.
- Europe session was worst (-14.26 bps); Asia was least bad (-13.76 bps), but
  no session was profitable.

These are diagnostic danger rankings, not validated production guards.

## Answers

1. Best waves tended to have more volatility, more 15m volume, more RSI space,
   less extension and a smaller preceding directional move.
2. Worst waves were more mature/extended, strongly aligned, breakout-like and
   often associated with extreme volume or high volatility.
3. Those differences were causal features available before entry.
4. Clusters were distinguishable descriptively, but their economics overlapped
   tightly and all remained negative.
5. No cluster, tree leaf, probability tier, volume bin or context had net edge.
6. The only frozen candidates reproduced as negative in validation.
7. There is not enough evidence for a specialized wave strategy or Shadow.

## Integrity

- W1 VALIDATION: unread.
- W1/W2/W3 holdouts: sealed and unread.
- 14 bps base and 20 bps stress costs retained.
- Correlated events grouped in UTC 15-minute clusters; bootstrap unit UTC day.
- No threshold was changed after validation.
- Production, TypeScript, WebSocket and PM2 changes: none.
- Authenticated requests and exchange mutations: zero.

Artifact identities:

- Config SHA-256: `d3f86c0ccf153b3051c248302f4a4452613392b0044339c04919cc316d44445b`.
- Private evaluation SHA-256: `58c692acbba3b546d1102d2b727443870d266c6d85e9bfc2963fd660eec2ca9b`.
- Episode inventory SHA-256: `e323d27e2b50d01b07d9b6cdbe733acca51acbeae4db8d2af7192bf305a05b80`.

The global negative result is not hiding a robust causal regime in the tested
feature space. This wave family should remain closed unless genuinely new,
untouched information and a new hypothesis are introduced.
