# AEGIS W12 Ideal Entry Reverse Engineering

Isolated offline experiment that labels historically ideal entry zones with future
paths, then tests whether strictly causal pre-entry features can identify those zones
in later chronological data.

## Safety

- Reads only local public historical candles selected in the frozen config.
- Does not read governed external holdouts.
- Does not import or modify production, TypeScript, E4, W11, Shadow, PM2, exchange,
  guards, live models, canonical datasets, or active processes.
- Writes only below this disposable sandbox.

## Scientific Boundary

Future OHLC may create teachers and targets. Every model feature must be available at
or before the decision timestamp. The primary evidence is prospective concentration
of ideal zones and net economic return, not retrospective existence of perfect paths.

Read `w12_data_audit.md` and `w12_preregistration.md` before interpreting results.

## Reproduction

From the repository root:

```bash
PYTHONPATH=sandbox/aegis_ideal_entry_reverse_engineering_w12/src \
  .venv/bin/python -m aegis_ideal_entry_reverse_engineering_w12.cli audit

PYTHONPATH=sandbox/aegis_ideal_entry_reverse_engineering_w12/src \
  .venv/bin/python -m aegis_ideal_entry_reverse_engineering_w12.cli run

PYTHONPATH=sandbox/aegis_ideal_entry_reverse_engineering_w12/src \
  .venv/bin/pytest sandbox/aegis_ideal_entry_reverse_engineering_w12/tests
```

The final report and verdict will be written to `w12_ideal_entry_result.md` and
`w12_ideal_entry_verdict.json`. Large deterministic tables and caches remain under
`artifacts/`.
