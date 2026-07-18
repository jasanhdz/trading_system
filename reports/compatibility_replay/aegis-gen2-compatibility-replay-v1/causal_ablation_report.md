# Gen2 causal ablation report

## Scope

This report records the five closed-list, development-only ablations defined by
`aegis-gen2-compatibility-replay-v1`. Each stage was evaluated twice from clean
attempt directories. The Stage 0 historical control remains unchanged.

No stage accessed data after 2026-04-26, the semi-blind window, or the lockbox.
No stage produced a Candidate, Selection Policy, or Freeze.

## Results

| Stage | Changed axis | Trades | Net PF | Net expectancy | Win rate | Max drawdown | Stage 0 overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | None (historical control) | 688 | 1.740824 | 0.140125 | 0.579942 | 10.698516 | 1.000000 |
| 1 | Sampling | 2,404 | 0.646052 | -0.122125 | 0.405574 | 297.302596 | 0.085756 |
| 2 | Features | 688 | 1.845744 | 0.150601 | 0.585756 | 8.344470 | 0.472384 |
| 3 | Model capacity | 688 | 1.048032 | 0.011966 | 0.469477 | 35.196939 | 0.313953 |
| 4 | Runtime selection | 715 | 0.949592 | -0.021063 | 0.524476 | 55.462642 | 0.316860 |
| 5 | EQM fold population | 688 | 1.740824 | 0.140125 | 0.579942 | 10.698516 | 1.000000 |

The machine-readable metrics, fold and symbol breakdowns, trade sets, input
hashes, and deterministic attempt manifests are in `stage_1.json` through
`stage_5.json` and their corresponding stage directories.

## Mechanical attribution

- **Stage 1, sampling:** replacing the historical population with E2 hourly
  anchors is compatible with a large adverse impact: PF changed by -1.094772,
  expectancy by -0.262249, and the selected set expanded by 1,716 trades.
- **Stage 2, features:** the E2 83-feature schema did not explain the degradation
  in isolation under historical sampling and historical model capacity. PF and
  expectancy were slightly above Stage 0, while trade overlap was 47.24%.
- **Stage 3, capacity:** E2 smoke-scale capacity is compatible with a material
  adverse impact. PF changed by -0.692792 and expectancy by -0.128159; two folds
  had negative expectancy.
- **Stage 4, runtime selection:** the E2 absolute veto and top-one-per-cycle
  semantics are compatible with an adverse impact. PF changed by -0.791232 and
  expectancy by -0.161187.
- **Stage 5, EQM population:** the fold-level full-population diagnostic changed
  fold training evidence, but final trades remain identical to Stage 0 because
  final refit population and runtime were explicitly frozen. Its isolated final
  trading impact is therefore not identifiable from this preregistered stage.

These deltas are not additive and do not establish independence between axes.
They describe isolated compatibility-replay effects, not absolute causal proof
or a recommendation to change E3.

## Comparability notes

Stage 1 has 28 common trades with the E2 reference (2.19% relative to E2).
Stages 2-5 use historical sampling timestamps and therefore have no directly
matching E2 trade keys. This is recorded as zero overlap, not treated as a
performance comparison.

No additional regime taxonomy was introduced. A regime breakdown is absent
where no frozen historical regime definition was available to this harness.

## Determinism and safety

For every stage, attempts 1 and 2 produced identical canonical scientific
hashes, candidates, trade keys, metrics, and normalized manifests. Operational
timestamps and absolute paths are excluded from canonical scientific hashes.

The lockbox remained `NOT_CONSUMED` with `consumed_queries=[]`. The replay did
not write to `/home/jasan/Develop/aegis_gen2/`, did not import operational
Binance paths, and did not modify TypeScript or productive E3 code.
