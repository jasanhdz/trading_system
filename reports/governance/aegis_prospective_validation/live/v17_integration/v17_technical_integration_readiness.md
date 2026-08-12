# V17 Technical Integration Readiness

Date: 2026-08-12 UTC

## Scope and safety state

This change prepares and tests the software boundary only. It does not select
V17, change the running services, alter credentials, or submit exchange
requests. V15/current production authority remains unchanged. V17 is a
challenger with `execution_authority: false`.

## Version selection

- The running Python API is assembled by `src/aegis/live_api.py`.
- Its operational selector is `config/hybrid_directional_live_experiment.yaml`.
- The TypeScript client consumes `POST /ml-v2/predict` and accepts a decision
  only when `CurrentBrainCanonicalDecision` validates the complete contract,
  authority, model, bundle, configuration, feature schema, symbol, and action.
- Per-symbol TypeScript execution is additionally selected by `mode: LIVE` and
  `AEGIS_LIVE_ENABLED=1`. Shadow symbols are evaluated but cannot execute.
- V17 is declared separately in `config/v17_execution_challenger.yaml`; it is
  visible in health and response telemetry, but cannot select an action.

## Signal and execution path

`closed market data -> Python features/model/layers/candidate -> HTTP response
-> AegisMLService -> canonical authority validation -> entry policy -> wallet
balance sizing -> exchange filters -> market intent -> position confirmation
-> SL/TP creation and read-back -> persisted state -> break-even/trailing ->
reconciliation -> exit`.

There is no new V17 order path. A future authorized V17 decision must enter the
same canonical boundary and therefore reuse all TypeScript execution,
protection, recovery, and telemetry behavior.

## V15 versus V17

| Property | V15 | V17 |
|---|---|---|
| Role | directional safety composite | safety gate followed by pairwise ranker |
| LONG features | 129 | same V15 contract, 129 |
| SHORT features | 168 | same V15 contract, 168 |
| Base feature family | decomposed V9 | decomposed V9 |
| Minimum causal 5m history | 576 bars due to 1h aggregation | 576 bars |
| Exported executable model | no | no |
| Historical verdict | RESEARCH_ONLY_NOT_PROMOTABLE | RESEARCH_ONLY_NOT_PROMOTABLE |
| Successful folds | did not pass aggregate contract | 0/4 LONG, 0/4 SHORT |
| Runtime authority | none | none |

V17 cannot be made a truthful runtime selector from the current files. Its
training script fits scikit-learn estimators inside each validation fold and
explicitly reports `model_exported: false`. The preregistration also prohibits
model export and Live promotion. There is no canonical final-fit prescription,
frozen calibration policy, serialized estimator, or artifact hash. Mapping the
83 current Live features to the 129/168 V17 contracts would fabricate inputs.

## Existing execution protections

- Exchange quantity and price rounding use `stepSize`, `tickSize`, precision,
  minimum quantity/notional, and adapter-side validation.
- Leverage and isolated margin are set before entry using the existing path.
- A market entry is followed by bounded position read-back.
- Failure to confirm the position invokes the existing emergency-close path.
- Required SL and TP are created and then read back from regular/algo order
  surfaces.
- With `close_if_bracket_fails: true`, creation or validation failure invokes
  emergency close and persists `FAILED_CLOSED`; an entry is not considered
  managed merely because the market request was acknowledged.
- On restart, persisted symbol state is reconciled with the exchange. Missing
  required brackets are recreated from persisted entry/risk values.
- Break-even arms from configured ROE and only improves the stop.
- Trailing activates from configured ROE/callback and only improves the stop.
- Existing daily trades, consecutive losses, daily loss, cooldown, duplicate
  symbol, portfolio/probe/momentum, Live-enable, and symbol-mode controls remain
  unchanged.

## Telemetry readiness

The isolated V17 compatibility module preserves expected price, market time,
feature hash, model identity/hash, policy identity, clean/danger probability,
MAE q90, and rank score without deriving missing trading values. It also
defines direction-aware slippage calculation. It is intentionally not imported
by `TradingService`, because changing that file would invalidate the byte-level
V15 control. Existing operational records already contain actual entry price,
quantity, leverage, position fraction, SL, TP, bracket confirmation, trailing
settings, fees, MAE, MFE, exit reason, PnL, and execution errors. Runtime V17
telemetry wiring belongs in the future hash-bound V17 activation change.

## Automated evidence

Focused tests prove:

- inactive V17 configuration has no exchange authority;
- malformed/non-finite V17 output fails closed;
- the V17 selected flag cannot disagree with its frozen gate and rank policy;
- TypeScript validates the canonical V17 evidence envelope losslessly and
  rejects malformed or policy-inconsistent decisions;
- existing fake lifecycle tests cover sizing, market intent, position
  confirmation, SL, TP, and telemetry through the control pipeline;
- unconfirmed positions and failed/missing brackets invoke fail-closed handling;
- restart management recreates missing brackets;
- canonical model/hash drift is rejected.

## Remaining blockers

1. Define a separately preregistered final-fit/export procedure for V17.
2. Fit and serialize the exact LONG and SHORT safety heads and rankers without
   test leakage.
3. Freeze gate/rank thresholds, model hashes, feature ordering, normalizers,
   and a 576-bar causal snapshot contract.
4. Validate direct-versus-runtime prediction parity and run V17 in Shadow.
5. Obtain forward evidence. Historical failure must not be reinterpreted as
   promotion evidence.
6. Add the resulting exact authority profile to TypeScript only after the
   artifact exists; the current hard-coded authority validation correctly
   rejects unknown V17 hashes.

Until these are complete, the state is
`TECHNICAL_EXECUTION_PIPELINE_COMPATIBLE_MODEL_ARTIFACT_REQUIRED`, not ready for
Live.

## Verification commands

```bash
cd /home/jasan/Develop/trading_system
PYTHONPATH=src /home/jasan/.venv_rocm62/bin/python -m pytest \
  tests/unit/test_v17_execution_challenger.py \
  tests/unit/test_live_decision_api.py -q

cd /home/jasan/Develop/trading_system/binance-futures-bot-ts
npm test -- --run \
  src/challengers/V17ExecutionCompatibility.test.ts \
  src/app/services/TradingService.aegis-live.test.ts \
  src/domain/services/CurrentBrainCanonicalDecision.test.ts \
  src/infra/adapters/BinanceAdapter.brackets.test.ts \
  src/infra/logging/AegisTurboHistoryLogger.test.ts
npm run build
git diff --check
```

## Preflight checklist

- [ ] V17 executable artifact exists and hashes match a frozen manifest.
- [ ] V17 129/168 feature vectors are generated causally from at least 576
      closed 5m bars for all configured symbols.
- [ ] Direct model and HTTP predictions have exact/tolerated parity.
- [ ] V17 Shadow output is current, finite, non-fallback, and reconciled.
- [ ] TypeScript recognizes exactly the approved V17 authority profile.
- [ ] Full fake lifecycle and restart tests pass.
- [ ] Read-only account audit is complete and all exposure is explained.
- [ ] Exactly one Python API and one TypeScript manager exist.
- [ ] Credentials and exchange mutations remain outside test processes.
- [ ] V17 activation remains absent until owner action.

## Final manual boundary

There is currently no responsible single `ready -> live` action because the
executable V17 artifact does not exist. After the blockers above are closed,
the only owner action should be activation of the exact hash-bound V17
authority record through the project's deployment procedure. No activation
command is supplied by this task, and no runtime service was changed.
