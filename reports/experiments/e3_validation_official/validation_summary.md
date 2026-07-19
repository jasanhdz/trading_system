# Aegis E3 official validation run

Result: `VALIDATION_PRE_LOCKBOX_COMPLETE`.

Both independent executions used commit `aea3437`, run ID
`d742d9bc0ae867bb`, preregistration E3, competition v2, and separate report
roots. They completed in 1696.59 and 1691.82 seconds. All eight scientific
artifacts were byte-identical; aggregate hash:
`67a976ffcb0c90bad9440afb361293ab95ec4ae11baa1e33efb8b2fb475f8347`.

## Dataset

- 15,680 coordinated hourly cycles.
- 172,480 rows across 11 symbols.
- Zero skipped history cycles and zero quarantined labels.
- Dataset hash: `1ffd0eaf07515d3a1a5fd6363f09c2d8ffe1e1f3925989486dee398e25b8c294`.

## Models and validation

- TRRM: `trrm_logistic_baseline`.
- EQM clean: `eqm_random_forest_clean`.
- EQM net: `eqm_linear_net_baseline`.
- QMAE: `qmae_hist_gradient_boosting`.
- Model competition: `model_not_beaten=true`.
- Maximum ECE: `0.024506121208181495` (limit `0.08`).
- QMAE coverage by fold: `0.916192`, `0.899926`, `0.889306`, `0.887060`.

The bundle remains `EXPERIMENTAL`, `trained=true`, `approved=false`, with
content hash `941f6b462812d1779f53bec3aba35741719116ecf37392b380f62207fface0ba`.
The pre-lockbox threshold draft is `5.994241878766537e-05`.

## ECON

| Scenario | Trades | PF | Expectancy | Win rate | Max drawdown |
|---|---:|---:|---:|---:|---:|
| A optimistic | 1292 | 0.644794 | -0.00108418 | 0.435759 | 1.433252 |
| B base | 1292 | 0.537044 | -0.00153418 | 0.404799 | 2.007902 |
| C pessimistic | 1292 | 0.404056 | -0.00223418 | 0.353715 | 2.905428 |

B_BASE fold expectancies were `-0.00158949`, `-0.00208723`, `-0.00141203`,
and `-0.000890804`. The best directional baseline was `no_trade` at `0.0`;
full stack did not beat it.

The dev comparison fails the frozen positive-fold, PF, expectancy, worst-fold,
directional-baseline, model-beaten, and robust-ECON checks. It passes signal
count, concentration, ECE, calibration, QMAE, leakage, and SHORT separation.
This validation run does not issue a lockbox verdict or lifecycle promotion.

Lockbox remains `NOT_CONSUMED`, `consumed_queries=[]`. No Candidate, Selection
Policy, System Freeze, shadow action, or operational execution was created.
