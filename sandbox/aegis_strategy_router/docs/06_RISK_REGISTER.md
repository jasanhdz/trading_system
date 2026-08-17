# Risk Register and Failure Analysis

## Methodological risks

| Risk | Failure | Prevention | Detection |
|---|---|---|---|
| Unfalsifiable narratives | Every loss is explained by a new strategy | Freeze finite specialist catalog and allow NONE | Count post-hoc categories; reject unregistered hypotheses |
| Multiple comparisons | One specialist appears good by chance | Limit families, preregister, FDR, sealed holdout | Hypothesis registry and adjusted results |
| Small specialist samples | High variance appears as 60% accuracy | Minimum event/support gates | Wide CI, instability across folds/symbols |
| Outcome leakage | Future level/path contaminates features | Availability timestamps and leakage tests | Synthetic future-column injection |
| Overlap dependence | Repeated nearby candidates inflate N | Episode spacing, grouping, block bootstrap | Effective sample-size audit |
| Selection bias | Train on general events, test selected signals | Population-shift and OOD analysis | Feature/support comparison |
| Validation reuse | W14 findings become false confirmation | Treat W14 as discovery only; require new window | Split manifest and timestamp audit |
| Class baseline error | Compare 59.5% with incorrect 33.3% | Use empirical and binary baselines | Report class prevalence and naive models |

## Strategy-definition risks

| Risk | Failure | Prevention |
|---|---|---|
| Overlapping strategies | Same evidence counted as multiple confirmations | Compatibility and evidence-family matrices |
| Vague support/resistance | Hindsight levels make breakouts perfect | Causal level algorithm with versioned rules |
| Lagging higher timeframe | EMA alignment identifies mature move | Trend age, extension, remaining-space critics |
| RSI dogma | Oversold blocks profitable continuation | RSI is continuous context; never lone veto |
| Breakout sparsity | Strict rules yield a handful of trades | Event-rate audit before modeling; do not loosen after validation |
| Transition ambiguity | Pullback and reversal confused | Separate candidate states and require structural confirmation |
| Horizon mismatch | Correct theory evaluated at wrong duration | Specialist-owned horizon and compatibility matrix |

## Modeling risks

| Risk | Failure | Prevention |
|---|---|---|
| Uncalibrated confidence | 0.70 means different things per model | Per-specialist temporal calibration |
| Correlated experts | Router becomes falsely confident | Evidence de-duplication; no naive vote counting |
| Meta-overfit | Router learns in-sample specialist errors | Out-of-fold predictions only |
| OOD confidence | Model is confident in unseen volatility/symbol | OOD critic and support thresholds |
| Feature drift | Current market differs from training | Drift monitoring and prospective recalibration gates |
| Side asymmetry | SHORT evidence generalized to LONG | Separate reporting/calibration; no unsupported transfer |
| Symbol memorization | Model learns asset identity | pooled/leave-one-symbol-out and identity ablation |
| Black-box dependency | Full model works but no stable family contributes | Feature-family ablations and simple baselines |

## Router risks

| Risk | Failure | Prevention |
|---|---|---|
| Forced choice | Router always finds a winner | NONE and minimum dominance margin |
| Probability-only ranking | Small safe move beats larger-risk setup incorrectly | Compare path distribution and uncertainty |
| Excessive abstention | Apparent improvement from almost no trades | Per-signal and per-trade metrics; coverage gates |
| Strategy churn | Hypothesis changes every evaluation | Hysteresis, expiry, lifecycle states |
| Direct flips | LONG immediately becomes SHORT | Mandatory exit/flat/cooldown/new confirmation |
| Critic dictatorship | One noisy critic blocks everything | Hard veto only after independent validation |
| Duplicate evidence | EMA/price appears in many specialists | Root-cause evidence groups |
| Winner concentration | One model dominates all routing | Concentration report; compare directly to that specialist |

## Operational risks

| Risk | Failure | Prevention |
|---|---|---|
| Sandbox reaches production | Research changes financial behavior | No production imports; safety test and code-owner boundary |
| Data failure becomes entry | Missing value interpreted favorably | Fail to unknown/abstain |
| Model/config mismatch | Wrong calibration or features used | Version/hash contract |
| Silent exception | Router defaults to enter | Fail closed in research; production remains independent |
| Latency destroys setup | Confirmation price unavailable live | Execute-at-latency replay before shadow |
| Costs ignored indefinitely | Directional signal has no realizable value | Direction first, costs mandatory before deployment |
| Leverage masks weak edge | ROE looks large despite poor underlying path | No leverage in research targets |

## Most likely ways this program fails

1. Specialists describe market stories but none predicts path better than its
   empirical baseline.
2. Transition/reversal appears promising only because W14 validation inspired
   its definition.
3. Breakout/retest has too few independent events under honest causal levels.
4. Specialists share so much price/EMA evidence that routing adds no independent
   information.
5. Router improves per-signal results only by abstaining from nearly everything.
6. Direction improves modestly, but MAE/tail losses remain too large.
7. Performance depends on SHORT-only market periods and fails LONG or new
   regimes.

Each is an acceptable negative result. The sandbox exists so the whole program
can be removed cleanly if these risks materialize.

