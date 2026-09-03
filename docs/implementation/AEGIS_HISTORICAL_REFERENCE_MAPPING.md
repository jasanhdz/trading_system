# Aegis Historical Reference Mapping

The historical branch was inspected read-only. No merge, cherry-pick, directory
restore, or block copy was used.

| Historical concept | Historical location | Understood behavior | Clean implementation | Deliberate change | Test |
|---|---|---|---|---|---|
| D3 causal contract | `aegis_alpha/tools/build_trrm_causal_feature_dataset_d.py` and D3 reports | Causal timestamping, frozen feature order, dense/strided datasets | `features.py` validation and one shared pipeline; `layers.py` D3 context | Data-contract behavior stays in validation; D3 now emits regime/context as required by the new architecture | `test_features_models_layers.py` |
| Base market features | `build_trrm_causal_feature_dataset_d.py` | Returns, range/body/wicks, true range, volume, EMA, momentum, chop, relative BTC/ETH context | Compact 39-feature schema in `features.py` | Removed target builders, labels, duplicated variants, symbol-specific artifacts, and unused legacy dimensions | feature determinism tests |
| RV2/TRRM | `GEN2_RV2_SPEC.md`, `gen2_rv2_train.py` | Calibrated tail probability and global p70 threshold / 30% veto budget | RV2 aggregates tail probability; TRRM exposes `1-tail` and a frozen max-tail gate | No per-symbol model or operational enforcement in Python | layer tests |
| QMAE | `GEN2_RV2_SPEC.md`, `gen2_rv2_train.py` | q50/q90 quantile prediction, split-conformal q90 adjustment, temporal folds | Runtime consumes frozen q90; offline pipeline provides temporal folds and immutable artifacts | Conformal calibration must be published in a future approved bundle rather than recomputed online | layer/training tests |
| EQM1 | `GEN2_EQM1_SPEC.md`, `gen2_eqm1_train.py` | Expected net quality plus probability of a clean opportunity; composite product | EQM uses clean probability times positive directional edge | Ranking is global and cohesive; no separate daily policy scripts | layer/runtime tests |
| ECON1 | `GEN2_ECON1_SPEC.md` | Net edge after fees, slippage, and funding scenario | ECON1 subtracts frozen round-trip cost from directional expected edge | Cost is scientific viability only; final execution costs and sizing remain TypeScript | layer tests |
| Selection policy | `gen2_selection_policy.py`, `gen2_decision_loop.py` | TRRM survivor gate, absolute quality cutoff, deterministic best opportunity, H12 focus | `GlobalSelectionPolicy` compares all eleven and applies threshold, quality outputs, and portfolio compatibility | Eliminated daily files and operational state reads; ties resolve by score/symbol/hash | runtime tests |
| Decision freeze | Gen2 freeze tooling and manifests | Candidate/model/config hashes and no post-freeze mutation | `DecisionFreezer` creates deterministic immutable IDs and expiry | Snapshot time replaces random/time-of-call identity | deterministic runtime test |
| Forward evidence | Gen2 collectors/outcome resolver | Append-only decisions and later mature outcomes | Hash-chained `ScientificEvidenceEvent` and typed `DecisionOutcome` | Outcome cannot mutate policy or frozen decisions | evidence tests |
| Artifact registry | Gen2 system/model freeze tooling | Checksums, versioned candidates, explicit freeze | Content-hashed JSON bundle plus immutable offline registry | No active manifest, hot swap, or automatic promotion | model/registry tests |
| Walk-forward evaluation | `gen2_rv2_train.py` | Expanding temporal folds, 120-minute embargo, deterministic research | `walk_forward_splits` and deterministic trainer/evaluator | Generic reusable fold contract replaces phase-specific scripts | training tests |

## Rejected Historical Elements

- Python Binance adapters, order paths, PM2 tooling, Telegram, sizing, leverage,
  exchange filters, brackets, and reconciliation.
- Per-phase executable scripts and duplicated training/inference features.
- Symbol-specific policy optimization and mutable active manifests.
- Opened-lockbox reuse, implicit globals, daily policy reimplementation, and
  automatic promotion.
- Historical pickle/joblib artifacts as a clean-runtime contract.

These elements either belong to TypeScript operations or conflict with the clean
architecture's determinism, cohesion, and explicit artifact boundary.
