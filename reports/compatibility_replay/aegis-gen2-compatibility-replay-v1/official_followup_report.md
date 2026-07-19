# Official Stage 1b and Stage 4b execution

## Stage 1b

Result: `EDGE_ABSENT_ON_HOURLY`.

| Scope | Trades | PF | Net expectancy |
|---|---:|---:|---:|
| Pooled | 2404 | 0.636280075689 | -0.131134714627 |
| Fold 1 | 601 | 0.603594429797 | -0.159126774228 |
| Fold 2 | 601 | 0.535883999094 | -0.149735646566 |
| Fold 3 | 601 | 0.690472968936 | -0.115055991094 |
| Fold 4 | 601 | 0.709512974703 | -0.100620446621 |

Two independent executions produced identical scientific and trade-key hashes.
Every fold exceeded the preregistered minimum of 100 trades.

## Stage 4b

| Variant | Isolated lever | Trades | PF | Net expectancy |
|---|---|---:|---:|---:|
| Stage 0 | Historical control | 688 | 1.740823969157 | 0.140124590960 |
| Stage 4 | Compound E2 selection | 715 | 0.949592278003 | -0.021062504323 |
| 4b-A | Absolute veto | 895 | 0.864643252256 | -0.053218888783 |
| 4b-B | Top-1 per cycle | 848 | 1.196368247652 | 0.045778539296 |
| 4b-C | E2 threshold | 492 | 2.311016807864 | 0.228043124790 |

Each variant ran twice with identical canonical and trade-key hashes. These
effects are isolated diagnostics and are not additive; interactions among veto,
threshold, and top-1 remain part of the compound Stage 4 behavior.

## Descriptive answers

- Stage 1b: coherent hourly training did not recover edge under the frozen design.
- Stage 4b: the absolute veto was negative in isolation; top-1 and the E2
  threshold were positive in their isolated variants.
- Existing assumptions: Stage 1b closes the previously unanswered coherent
  hourly-training question for this exact design. Stage 4b separates the three
  frozen selection levers without changing the protocol.
- Fable audit: no inconsistency. The runs answer questions that audit left open.

Verdict: `READY_FOR_E3_VALIDATION`.

No E3 validation-run, Stage 5b, lockbox query, semi-blind access, or operational
execution occurred.
