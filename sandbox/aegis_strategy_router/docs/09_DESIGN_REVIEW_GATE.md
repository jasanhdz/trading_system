# Design Review Gate

## Decisions already frozen for the first implementation proposal

- Scope begins as signal-conditioned: Aegis side is immutable.
- Five strategy specialists only.
- Critics are separate from specialists.
- Initial router is deterministic and supports abstention.
- Entry routing is separate from position management.
- No direct side flip or reentry in the entry experiment.
- Direction/path safety is primary during specialist discovery.
- Costs, latency, and execution feasibility are mandatory before deployment.
- Leverage is excluded from all learning and promotion criteria.
- W14 transition evidence is discovery only.
- New code remains entirely inside the sandbox until a separately reviewed
  prospective event adapter is required.
- Existing W1-W14 holdouts remain sealed.

## Items that must be resolved before executable code

### Fresh experimental timeline

Define new TRAIN, calibration, VALIDATION, and FINAL_HOLDOUT periods that do not
reuse prior holdouts or validation-driven thresholds. If only prospective data
is fresh, implementation starts with collection/data audit rather than model
training.

### Causal structural-level algorithm

Choose and freeze one or a very small family of level definitions on TRAIN:

- fractal/pivot swings with known confirmation delay;
- rolling prior highs/lows;
- clustered repeated touches;
- optional volume-at-price only if historical construction is causal.

Document level availability time. A pivot requiring right-side bars becomes
available only after those bars close.

### Strategy horizon/barrier families

Perform a TRAIN-only movement-scale audit, then preregister a limited family per
specialist. Do not optimize every strategy on every horizon.

### Minimum sample gates

Use event-rate and effective-degrees-of-freedom audits to set final minima.
Insufficient sample means no model, not weaker evidence requirements.

### General-to-Aegis transfer

Define whether models train on general candidates and calibrate on Aegis
signals, or require an Aegis-only model. Quantify population shift before
choosing.

### LONG/SHORT policy

Decide whether one direction-normalized model is allowed initially. Any claim
of side symmetry requires independent side results. A side-specific specialist
must be preregistered, not selected because one side won validation.

### Router primary objective

Freeze the exact directional/path objective and uncertainty penalty. A possible
starting contract is favorable-first probability plus MFE/MAE geometry with a
tail-MAE constraint, but coefficients cannot be chosen after validation.

### Critic enforcement

Only data quality and critical OOD are hard abstentions by design. Every market
critic begins diagnostic unless independently validated as a veto.

## Approval checklist before Phase 1 code

- [ ] Specialist definitions approved.
- [ ] Critic definitions approved.
- [ ] Signal-conditioned scope approved.
- [ ] Fresh split/data acquisition approved.
- [ ] Structural-level algorithm selected.
- [ ] Candidate event-rate audit design approved.
- [ ] Snapshot and feature contracts approved.
- [ ] Router abstention and conflict behavior approved.
- [ ] Primary metrics and negative-result policy approved.
- [ ] Sandbox dependency and deletion boundaries approved.
- [ ] No production integration authorized.

Until this checklist is approved, the sandbox remains documentation-only.

