# Aegis Strategy Router Sandbox

Status: `PHASE_0_APPROVED_PHASE_1_CODE_NOT_STARTED`

This directory is the isolation boundary for the proposed hybrid
multi-strategy system. No implementation may be added until the design review
is explicitly approved. Phase 0 is approved; executable work remains limited
to the Phase 1 scope recorded below.

## Objective

Given a causal market snapshot and an optional frozen Aegis directional signal:

1. identify which predefined market hypotheses are eligible;
2. ask a dedicated specialist to estimate each hypothesis' path quality;
3. ask independent critics to identify risk, ambiguity, or invalid data;
4. route to `ENTER`, `WAIT`, or `SKIP` with an auditable explanation;
5. never invent a strategy retrospectively to explain a loss.

## Current boundary

- No production imports this sandbox.
- No TypeScript trading path is modified.
- No authenticated exchange access is permitted.
- No order, sizing, leverage, guard, exit, PM2, or account behavior is present.
- W1-W14 holdouts remain untouched.
- The W14 `REGIME_TRANSITION` observation is hypothesis generation only.

## Documentation index

- [System architecture](docs/01_SYSTEM_ARCHITECTURE.md)
- [Specialists and critics](docs/02_SPECIALISTS_AND_CRITICS.md)
- [Router specification](docs/03_ROUTER_SPECIFICATION.md)
- [Data, labels, and feature contracts](docs/04_DATA_AND_FEATURE_CONTRACTS.md)
- [Experimental and implementation plan](docs/05_EXPERIMENT_AND_IMPLEMENTATION_PLAN.md)
- [Risk register and failure analysis](docs/06_RISK_REGISTER.md)
- [Visual system map](docs/07_VISUAL_SYSTEM_MAP.md)
- [Isolation and deletion plan](docs/08_ISOLATION_AND_DELETION.md)
- [Design review gate](docs/09_DESIGN_REVIEW_GATE.md)
- [Phase 0 frozen decisions](docs/10_PHASE0_FROZEN_DECISIONS.md)

## Planned implementation tree

The following structure is a plan, not an implemented tree:

```text
sandbox/aegis_strategy_router/
├── README.md
├── docs/
├── config/                         # Experiment-only frozen configs
├── src/aegis_strategy_router/
│   ├── domain/                     # Contracts and immutable value objects
│   ├── features/                   # Adapters to existing causal features
│   ├── candidates/                 # Deterministic candidate generators
│   ├── specialists/                # One package per setup hypothesis
│   ├── critics/                    # Independent safety/risk evaluators
│   ├── calibration/                # Per-specialist probability calibration
│   ├── router/                     # Arbitration and abstention
│   ├── lifecycle/                  # Pending/entered/invalidation states
│   ├── datasets/                   # Episode builders, no raw data copies
│   └── evaluation/                 # Replay, metrics, bootstrap, reports
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── leakage/
│   ├── replay/
│   └── safety/
└── artifacts/                      # Ignored generated outputs only
```

Existing feature and replay code will be imported through adapters. It will not
be copied into this sandbox unless a dependency is proven unsuitable and the
replacement is explicitly approved.

## Removal criterion

If the hypothesis fails, deleting this directory and reverting one future
integration adapter must remove the entire experiment. Production must remain
behaviorally identical before and after that deletion.

Phase 0 is now methodologically frozen. Phase 1 is authorized in design but has
not started; no executable strategy-router code exists yet.
