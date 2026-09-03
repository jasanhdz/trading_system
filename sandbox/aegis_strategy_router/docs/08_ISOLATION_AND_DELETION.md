# Isolation and Deletion Plan

## Isolation rules

All new executable work, if later authorized, lives under:

```text
sandbox/aegis_strategy_router/
```

Allowed external dependencies are read-only adapters to stable research APIs.
The sandbox may not import production execution implementations.

Production may not import the sandbox during offline research. Prospective
observation, if eventually approved, must use one narrow read-only adapter that
copies immutable signal/snapshot events without influencing timing or outcome.

## Artifact ownership

- Source, tests, frozen configs, and design documentation belong in sandbox.
- Large datasets remain in the repository's existing ignored data area and are
  referenced by immutable manifests/hashes.
- Generated models and reports are ignored artifacts with reproducibility
  manifests.
- No credentials or authenticated clients are stored in sandbox.

## Integration budget

Before prospective observation, permitted changes outside sandbox are limited
to one reviewed event-export adapter and its tests. It must be:

- asynchronous/non-blocking;
- read-only;
- bounded;
- fail-open for existing trading behavior;
- free of financial methods;
- removable without changing decision semantics.

Any requirement for broader production modification stops the program for
review.

## Deletion procedure

If the experiment fails before integration:

1. preserve final verdict/report externally if required by governance;
2. delete `sandbox/aegis_strategy_router/` in one commit;
3. remove ignored generated artifacts referenced only by its manifest;
4. run existing Python and TypeScript suites;
5. verify production Git tree and behavior are unchanged.

If a prospective adapter exists:

1. disable collection configuration;
2. remove the single adapter and tests in a second explicit commit;
3. verify bot operation without the sandbox process;
4. retain immutable collected data only if governance requires it.

## Success does not change the boundary automatically

Positive offline evidence authorizes only a proposal for prospective
observation. Positive prospective evidence authorizes only a proposal for
shadow. Neither moves code into production automatically.

