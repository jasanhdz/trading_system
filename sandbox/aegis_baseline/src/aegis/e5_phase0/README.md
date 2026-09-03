# E5 Phase 0

This package implements the offline, synthetic-only engineering validation
defined by `e5_execution_specification.md`. It does not load E5 scientific
rows, download funding, run discovery, run confirmation, or access semi-blind
or lockbox data.

From the repository root, run:

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python -m aegis.e5_phase0
```

The command verifies governance and clean source state, validates schemas and
prohibited-data guards, runs the 38 deterministic synthetic categories, and
writes non-scientific outputs under
`reports/governance/e5_signal_edge_protocol/phase0/`. A successful command emits
`E5_PHASE_0_PASS`. That status authorizes no scientific or operational stage.
