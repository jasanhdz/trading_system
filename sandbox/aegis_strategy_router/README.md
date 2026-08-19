# Aegis Strategy Router Sandbox

Status: `PHASE_2_INDEPENDENT_MODE_COMPLETE_FRESH_SUPPORT_BLOCKED`

This directory is the isolation boundary for the proposed hybrid
multi-strategy system. No implementation may be added until the design review
is explicitly approved. Phase 0 is approved; executable work remains limited
to the Phase 1 scope recorded below.

## Objective

In the initial experiment, given a side-neutral causal market snapshot:

1. evaluate LONG and SHORT independently for each predefined hypothesis;
2. ask a dedicated specialist to estimate each hypothesis' path quality;
3. ask independent critics to identify risk, ambiguity, or invalid data;
4. route to `ENTER`, `WAIT`, or `SKIP` with an auditable explanation;
5. never invent a strategy retrospectively to explain a loss.

Aegis is excluded from initial discovery. A later Aegis transfer experiment is
permitted only after independently validated edge; see the scope amendment.

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
- [Phase 1 implementation report](docs/11_PHASE1_IMPLEMENTATION_REPORT.md)
- [Phase 2 governance amendment](docs/12_PHASE2_GOVERNANCE_AMENDMENT.md)
- [Phase 2 implementation report](docs/13_PHASE2_IMPLEMENTATION_REPORT.md)
- [Phase 2 pipeline and decision-gap review](docs/14_PHASE2_UNBLOCKING_REPORT.md)
- [Phase 2 deterministic rule freeze](docs/15_PHASE2_RULE_FREEZE.md)
- [Phase 2 rule execution report](docs/16_PHASE2_RULE_EXECUTION_REPORT.md)
- [Independent discovery amendment](docs/17_INDEPENDENT_STRATEGY_DISCOVERY_AMENDMENT.md)
- [Independent-mode transition report](docs/18_INDEPENDENT_MODE_TRANSITION_REPORT.md)
- [Shared market-data audit and reuse report](docs/19_SHARED_MARKET_DATA_AUDIT.md)

## Implementation tree

Phase 1 implements only the infrastructure shown below. Later-phase directories
remain conceptual and have not been created.

```text
sandbox/aegis_strategy_router/
├── README.md
├── docs/
├── src/aegis_strategy_router/
│   ├── adapters/                   # Causal joins and existing-feature adapter
│   ├── audit/                      # Frozen splits and label-free counts
│   ├── candidates/                 # Phase 2 contracts, substates, rules, gaps
│   ├── domain/                     # Frozen immutable value objects
│   ├── features/                   # Confirmed pivots and complete-linkage levels
│   ├── replay/                     # Deterministic snapshot construction
│   ├── safety/                     # Import/capability audit
│   └── schemas.py                  # Versioned feature schema and hash
├── tests/
│   ├── contracts/
│   ├── leakage/
│   ├── replay/
│   ├── safety/
│   └── unit/
└── pyproject.toml
```

Existing feature and replay code will be imported through adapters. It will not
be copied into this sandbox unless a dependency is proven unsuitable and the
replacement is explicitly approved.

## Removal criterion

If the hypothesis fails, deleting this directory and reverting one future
integration adapter must remove the entire experiment. Production must remain
behaviorally identical before and after that deletion.

Phase 0 remains frozen. Phase 1 technical infrastructure is accepted, including
deterministic structural clustering. Phase 2 deterministic implementation is
authorized separately from empirical validation. Fresh-data sufficiency,
specialists, routing, Shadow, and Live remain blocked.
The five candidate generators now execute the frozen definitions recorded in
`docs/15_PHASE2_RULE_FREEZE.md`. The fresh snapshot pipeline is operational,
but fresh population support remains insufficient and no edge was evaluated.
