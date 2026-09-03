# E5 Phase 1A Blind Fold 1-2 Export

This package implements the custodial exporter authorized by E5 Owner
Amendments 05 and 06. It verifies the frozen combined-source hash, inspects
`signal.fold` before payload projection, skips Fold 3-4 records without
deserializing their scientific payload, and seals the authorized Fold 1-2 entry
manifest.

Run from the repository root:

```bash
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python -m aegis.e5_blind_fold12_export \
  --source reports/experiments/e3_validation_official/attempt_1/aegis-short-candidate-e3/runs/d742d9bc0ae867bb/econ_report.json \
  --expected-source-sha256 bff472758eacc211dff1b3e2209cbd96e8a845a68f45b9e31526ca2968e6e085 \
  --config config/e5_blind_fold12_export_v1.json \
  --output-root reports/governance/e5_signal_edge_protocol/phase1a \
  --deterministic-rerun
```

The command is offline and non-interactive. It cannot invoke Discovery,
Confirmation, Shadow, Live, network, credential, exchange, or order code. The
sealed row-level JSONL remains ignored under `phase1a/sealed/`; compact reports
are committed separately.
