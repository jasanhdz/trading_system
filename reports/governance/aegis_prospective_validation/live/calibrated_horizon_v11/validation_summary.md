# Calibrated Horizon V11 Validation Summary

## Verdict

`RESEARCH_ONLY_NOT_PROMOTABLE`

V11 improves probability calibration and demonstrates useful horizon sharing,
but it does not produce a policy with sufficient positive, clean, directional,
and low-uncertainty candidates. No model was exported and neither Shadow nor
Live was changed.

## Evidence

- Evidence interval: 2025-08-09 through 2026-08-09.
- Independent episodes: 48,191.
- Rows: 96,382.
- Symbols: all 11 canonical symbols.
- V10 barrier outcomes reused unchanged: yes.
- Clean LONG labels: 5,500 (11.41%).
- Clean SHORT labels: 5,986 (12.42%).
- Nested windows per fold: train, probability calibration, policy, test.
- Exchange calls and mutations: zero.

## Component validation

| Component | LONG | SHORT | Required |
|---|---:|---:|---:|
| Direction skilled folds | 4/4 | 4/4 | 3/4 |
| Clean-entry skilled folds | 4/4 | 2/4 | 3/4 |
| Horizon-specialist skilled folds | 3/4 | 3/4 | 3/4 |
| Economic folds | 0/4 | 0/4 | 3/4 |

Direction remains reproducible. Hierarchical calibration produced test ECEs
below 0.10 in every direction fold. LONG clean-entry prediction is stable;
SHORT clean-entry prediction is not yet stable enough. Horizon sharing often
matches or improves the nine separately trained V10 controls, although one
fold per side has only one of three specialists passing the strict control.

## Policy result

No fold produced at least 30 candidates that simultaneously passed positive
conservative utility, direction probability, clean-entry probability, unknown
probability, and one-candidate-per-timestamp constraints.

| Side | Fold | Positive-utility candidates before joint gates | Mean predicted utility |
|---|---:|---:|---:|
| LONG | 1 | 46 | 0.0770% |
| LONG | 2 | 31 | 0.0626% |
| LONG | 3 | 23 | 0.1271% |
| LONG | 4 | 4 | 0.0988% |
| SHORT | 1 | 172 | 0.1606% |
| SHORT | 2 | 43 | 0.2683% |
| SHORT | 3 | 0 | N/A |
| SHORT | 4 | 11 | 0.1267% |

These counts are diagnostics, not validated trades. They come from the policy
window, and no threshold may be changed to force them through the joint gate.

## Utility attribution

Across complete policy populations, predicted total utility remained between
approximately -0.13% and -0.21%. Favorable and adverse predicted values mostly
cancel. The frozen severe cost contributes -0.20% to every candidate, while
unresolved-state penalties add a smaller negative amount. The clean-entry
bonus is intentionally too small to rescue a negative base utility.

The positive subsets are concentrated in 10%- and 20%-ROE contracts. They are
not jointly reliable: mean clean probabilities are generally 0.17 to 0.43,
and some subsets retain material unknown probability. SHORT fold 1 contains
172 positive-utility candidates, but mean clean probability is only 0.291;
therefore utility alone would admit trajectories that V11 cannot establish as
clean entries.

## What V11 establishes

1. Hierarchical calibration by supported symbol/regime groups is feasible and
   materially well calibrated outside sample.
2. Sharing barrier evidence inside 30-, 60-, and 120-minute specialists is
   usually as good as or better than separate per-contract models.
3. Clean-entry behavior is learnable for LONG.
4. SHORT clean-entry behavior remains temporally unstable.
5. Independently accurate heads do not automatically form a coherent trading
   decision when their probabilities are intersected after training.

## Recommended next experiment

V12 should preserve the V10/V11 labels, costs, episodes, and nested validation,
but replace independent post-hoc intersection with a preregistered joint state
model. It should directly predict coherent states such as directional-clean-
favorable, directional-adverse, and abstain. Barrier/horizon choice should be
made inside training or assigned by causal regime rather than maximizing nine
noisy utilities per candidate. SHORT clean-entry should receive a dedicated
ablation, while LONG may retain the current shared clean head.

V12 must compare this joint model against V11 without lowering the 30-candidate
minimum or the 20-basis-point severe cost. Protection behavior remains a
separate report-only replay until an entry policy passes economic gates.

## Safety

- Runtime effect: `NONE`.
- Model exported: `false`.
- Shadow activated: `false`.
- Live activated: `false`.
- Exchange calls: `0`.
- Exchange mutations: `0`.

## Artifact hashes

- Configuration: `4aa8dbaec916f21506badb58e93f101f4a2655bf02d8c310a33dd91f06eb6eac`.
- Dataset: `2c270a7b38b2f05c2b4ab78960b788c387e5d98bfae29da16a167f4f54548747`.
- Dataset manifest: `8648567317d7d7690c917afcb36049e1f4af01168c2f9f8b871718cbc41934f4`.
- Validation: `c257d4972cabc339243f9d81ba4561d028b703b31f8213fcf61cf30c6f15fe72`.
