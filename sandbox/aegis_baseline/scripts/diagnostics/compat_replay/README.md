# Gen2 compatibility replay

This package is an isolated, development-only diagnostic harness. It reads frozen Gen2
artifacts, verifies their hashes, and reproduces the historical ECON1 control before any
ablation is allowed. It never imports Phase E or production inference, never accesses the
Clean Rebuild lockbox, and cannot publish a Candidate, Selection Policy, or System Freeze.

Run Stage 0 with:

```bash
PYTHONPATH=. /home/jasan/.venv_rocm62/bin/python \
  scripts/diagnostics/compat_replay/run_compatibility_replay.py
```

Run the closed Stage 1-5 ablations (two clean deterministic attempts each) with:

```bash
PYTHONPATH=src:. /home/jasan/.venv_rocm62/bin/python \
  scripts/diagnostics/compat_replay/run_compatibility_replay.py --mode ablations
```

The ablation runner pins the prior E2 dev ECON report by physical hash, never
reads beyond `2026-04-26T23:59:59Z`, and cannot acquire the shared lockbox.
