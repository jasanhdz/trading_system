# Current-Brain Observational Evidence Plan V1

## Scope

This plan preserves the active Python decision service, TypeScript execution,
position management, protective orders, capital handling, and Shadow process.
It adds no gate, threshold, trading guard, symbol exclusion, or automatic
runtime action.

The evidence auditor is offline. It reads stable snapshots of the append-only
prospective signal and outcome journals and writes a private report. It is not
imported by the Live API, Shadow, or TypeScript.

## Questions

The continuing observation must distinguish four claims:

1. The pipeline is executable and deterministic.
2. The inputs and non-directional model outputs vary with the market.
3. Selection improves outcomes relative to rejected same-cycle candidates.
4. The calibrated score ranks future net outcomes monotonically and remains
   useful across symbols and time blocks.

Passing claims 1 and 2 does not establish claims 3 and 4.

## Measurements

Each audit freezes the source journal hashes and reports:

- selected and rejected net expectancy, profit factor, win rate, MFE, MAE,
  and tail-event rate;
- deterministic UTC-hour block-bootstrap uncertainty for selected expectancy;
- score/outcome and expected-return/outcome correlations;
- selected-versus-same-cycle outcome deltas;
- per-symbol selected and rejected economics;
- stage funnel and pass/fail economics for TRRM, QMAE, EQM, ECON1, and final
  selection;
- D3 regime-conditioned economics without treating the current D3 `decision`
  label as an independent gate;
- calibrated-score deciles and top-minus-bottom ranking spread;
- variability counts for directional probability, expected return, quality,
  tail risk, and calibrated score.

## Interpretation

The auditor emits warnings, not operational commands. Current warnings do not
pause production and must not be converted into symbol blocks or threshold
changes from the same sample. Any future strategy change requires a separately
defined out-of-sample protocol and owner authorization.

The current committee remains under observation while evidence accumulates.
No automatic runtime change is authorized by this plan.

## Invocation

```bash
PYTHONPATH=src python -m aegis.prospective.evidence_audit \
  --signals data/prospective_shadow/cohort_1/journals/signal_evidence_v1.jsonl \
  --outcomes data/prospective_shadow/cohort_1/journals/outcomes_v1.jsonl \
  --output data/current_brain_live_integration/current_brain_live_integration_01/prospective_evidence_audit.json \
  --bootstrap-repetitions 10000 \
  --seed 20260722
```

The report is written atomically with mode `0600` where supported. Credentials,
private Binance responses, orders, and PM2 state are not read or written.
