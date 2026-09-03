# Specialists and Critics

## Shared specialist output

Every specialist must return the same conceptual contract:

| Field | Meaning |
|---|---|
| `specialist_id/version` | Reproducible model identity |
| `eligible` | Deterministic candidate rules passed |
| `side` | LONG or SHORT proposed by the setup |
| `routing_role` | `PROPOSER` or signal-opposed `CONFLICT_ONLY` |
| `horizon` | Intended evaluation/holding horizon |
| `p_favorable_first` | Calibrated favorable-barrier probability |
| `p_adverse_first` | Calibrated adverse-barrier probability |
| `p_neither` | Probability neither barrier occurs in time |
| `expected_mfe_unconditional` | `E[MFE from now]` |
| `expected_mae_unconditional` | `E[MAE from now]` |
| `expected_mfe_given_favorable` | `E[MFE \| favorable first]` |
| `expected_mae_given_adverse` | `E[MAE \| adverse first]` |
| `expected_terminal_return_given_neither` | `E[R_60m \| neither]` |
| `expected_time_to_event` | Timing estimate |
| `path_quality` | Expected MFE/MAE and efficiency geometry |
| `uncertainty` | Confidence interval/model uncertainty |
| `ood_score` | Distance from supported training distribution |
| `invalidation` | Causal structural invalidation description |
| `evidence` | Human-readable supporting and opposing facts |

The three barrier probabilities are one multinomial distribution and must sum
to one within `1e-6`. Independent binary probability heads are prohibited.
The initial primary outcome is directional/path safety, with the frozen
economic plausibility kill gate applied before a specialist may enter routing.

## Specialist 1: Trend Continuation

### Candidate rules

- 1h and 4h directional structure broadly aligned.
- 15m direction not structurally invalidated.
- Trend not classified as extreme shock.
- Proposed direction has measurable space to the next structural level.
- Setup is not merely one isolated same-color candle.

### Primary features

- directional EMA7/25/99 slopes and extensions on 15m/1h/4h;
- HH/HL or LH/LL structure state;
- trend age and cumulative directional move in ATR;
- directional RSI room on 15m/1h/4h;
- ATR percentile and realized volatility;
- 5m/15m path efficiency and velocity;
- directional taker imbalance and its persistence;
- distance to next support/resistance;
- BTC alignment and residual return versus BTC.

### Label family

- favorable barrier before adverse barrier;
- fixed-horizon directional return;
- MFE greater than MAE;
- no higher-timeframe structural invalidation before favorable excursion.

### Main failure modes

- trend is mature rather than continuing;
- higher-timeframe alignment is lagging;
- signal arrives directly at structural resistance/support;
- continuation model duplicates breakout evidence.

## Specialist 2: Pullback Continuation

### Candidate rules

- 1h/4h trend aligned with proposed direction.
- 1m/5m movement temporarily opposed.
- Pullback has not broken the higher-timeframe invalidation level.
- Candidate is evaluated as `WAIT`, never immediate reversal.

### Primary features

- pullback size as fraction of prior impulse and ATR;
- pullback duration and path efficiency;
- pullback volume versus impulse volume;
- opposing taker pressure and price response;
- favorable-flow recovery and velocity recovery;
- distance to last higher low/lower high;
- 15m/1h trend maturity and space;
- microstructure confirmation when prospectively available.

### Label family

- local realignment occurs before structural invalidation;
- after realignment, favorable barrier occurs before adverse barrier;
- remaining MFE after confirmation exceeds consumed move and MAE.

### Main failure modes

- a reversal is mislabeled as a pullback;
- confirmation arrives after the remaining move is exhausted;
- repeated WAIT evaluations are counted as independent samples.

## Specialist 3: Breakout and Retest

### Candidate rules

- level exists before the breakout timestamp;
- level has sufficient prior touches/age without future information;
- price closes beyond the level by a normalized penetration amount;
- next structural level leaves usable space;
- breakout and retest are represented as separate substates.

### Primary features

- number, recency, and dispersion of prior level touches;
- level age and strength;
- breakout penetration in ATR/bps;
- close location and body/range quality;
- volume and taker-flow change relative to pre-breakout context;
- price impact per aggressive flow;
- retest depth, duration, and rejection quality;
- depletion/replenishment when L2 exists;
- 1h/4h location and alignment;
- distance to next level.

### Label family

- breakout holds/retest holds;
- favorable barrier from actual post-confirmation entry occurs first;
- false-break return inside the range occurs before continuation;
- remaining MFE and adverse retest depth.

### Main failure modes

- hindsight-selected levels;
- treating a wick through a level as a breakout;
- entering after all useful displacement is consumed;
- very low candidate count from overly strict level definitions.

## Specialist 4: Range Mean Reversion

### Candidate rules

- low directional efficiency and flat higher-timeframe slope;
- stable, causally defined range boundaries;
- entry proposal points inward from a range edge;
- no confirmed breakout or volatility shock;
- sufficient distance to range midpoint/target.

### Primary features

- distance to range edge and midpoint;
- range width in ATR and age;
- repeated rejection/wick behavior;
- directional RSI extension and recovery;
- flow absorption/price refusal at the edge;
- declining breakout effectiveness;
- volatility contraction and spread/liquidity state;
- higher-timeframe location.

### Label family

- return to midpoint before range break;
- favorable barrier before adverse range invalidation;
- MAE relative to distance beyond the edge.

### Main failure modes

- the range is actually transitioning into a breakout;
- apparent overbought/oversold persists during a trend;
- range boundaries are unstable or too narrow.

## Specialist 5: Regime Transition and Reversal

### Candidate rules

- previous regime is explicitly identified;
- evidence of deterioration exists in that regime;
- new structure begins to form, not merely one opposing candle;
- transition remains a candidate until confirmation; it is not presumed.

### Primary features

- prior trend age, cumulative ATR move, and extension;
- slope decay across 15m/1h/4h;
- failure/recovery of higher highs/lows;
- break and reclaim of structural levels;
- volatility transition and volume expansion/contraction;
- directional flow becoming effective in the new direction;
- old-direction flow losing price impact;
- RSI reset/divergence candidates;
- BTC regime transition and relative-strength change.

### Label family

- new-direction favorable barrier before old-direction recovery;
- structural transition persists for the chosen horizon;
- MFE/MAE from confirmation timestamp;
- false transition returns to prior regime.

### Governance note

W14's 37 validation transition episodes with 59.5% gross directional success
are hypothesis-generation evidence only. They cannot choose rules, thresholds,
or promotion gates for this specialist.

## Critics

### Data Quality Critic

Checks missing bars, stale snapshots, timestamp order, exchange/local skew,
feature warmup, NaN substitution, L2 sequence integrity, and schema mismatch.
Invalid data is a hard abstention.

### Shock Critic

Uses volatility expansion, range shock, volume shock, spread/depth deterioration,
cross-market movement, and discontinuity. It begins `DIAGNOSTIC_ONLY`, including
CRITICAL observations. Any penalty or veto requires independent validation and
must be frozen before router validation.

### Exhaustion and Late-Entry Critic

Uses trend age, prior ATR displacement, EMA extension, directional RSI room,
consumed movement, distance from impulse origin, and declining price impact.
RSI alone is never a veto.

### Space and Structural-Level Critic

Uses causal distance to next support/resistance, invalidation distance,
available MFE geometry, range width, and structural congestion. A level touch
alone is not a veto.

### Conflict and Uncertainty Critic

Measures disagreement between specialists, correlated evidence, calibration
uncertainty, dominance margin, and unresolved timeframe contradictions.

### Out-of-Distribution Critic

Measures whether the current feature vector, symbol liquidity, volatility, or
regime has adequate support in the training/calibration data. High model
confidence cannot override critical OOD.

### Portfolio Correlation Critic (later phase)

Evaluates correlated exposure, BTC beta, simultaneous same-direction signals,
and concentration. It is excluded from initial entry-quality research so that
entry evidence and portfolio sizing are not confounded.
