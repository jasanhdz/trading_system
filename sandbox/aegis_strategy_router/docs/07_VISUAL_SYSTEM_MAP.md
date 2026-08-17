# Visual System Map

## Complete decision flow

```mermaid
flowchart TD
    A[Immutable causal snapshot<br/>1m 5m 15m 1h 4h 1d BTC] --> B{Data quality valid?}
    B -- No --> Z1[SKIP<br/>INVALID_DATA]
    B -- Yes --> C[Context and location map]

    C --> G1[Trend continuation<br/>candidate rules]
    C --> G2[Pullback continuation<br/>candidate rules]
    C --> G3[Breakout and retest<br/>candidate rules]
    C --> G4[Range mean reversion<br/>candidate rules]
    C --> G5[Regime transition reversal<br/>candidate rules]

    G1 --> S1[Continuation specialist]
    G2 --> S2[Pullback specialist]
    G3 --> S3[Breakout specialist]
    G4 --> S4[Mean reversion specialist]
    G5 --> S5[Transition specialist]

    C --> K1[Shock critic]
    C --> K2[Exhaustion and late-entry critic]
    C --> K3[Space and structural-level critic]
    C --> K4[Conflict and uncertainty critic]
    C --> K5[Out-of-distribution critic]

    S1 --> R[Deterministic router]
    S2 --> R
    S3 --> R
    S4 --> R
    S5 --> R
    K1 --> R
    K2 --> R
    K3 --> R
    K4 --> R
    K5 --> R

    R --> D{Decision}
    D -- Dominant safe hypothesis --> E[ENTER proposal]
    D -- Evidence incomplete --> W[PENDING / WAIT]
    D -- Conflict risk or no edge --> X[SKIP]

    W --> A2[New causal snapshot]
    A2 --> R
```

## Separation of responsibilities

```mermaid
flowchart LR
    Rules[Rules<br/>What setup might exist?]
    Model[Specialist model<br/>How likely is this setup to work?]
    Critics[Critics<br/>What can invalidate or damage it?]
    Router[Router<br/>Which hypothesis dominates?]
    Lifecycle[Lifecycle<br/>What happens after the decision?]

    Rules --> Model --> Router --> Lifecycle
    Critics --> Router
```

Rules do not estimate profitability. Models do not define what a strategy is.
Critics do not propose the opposite direction. The router does not manufacture
features or retrain specialists.

## Multi-timeframe interpretation

```mermaid
flowchart TB
    D1[1d<br/>macro location and broad regime]
    H4[4h<br/>trend maturity and structural map]
    H1[1h<br/>active setup and regime transition]
    M15[15m<br/>operational pattern]
    M5[5m<br/>confirmation and pullback state]
    M1[1m<br/>entry timing only]

    D1 --> H4 --> H1 --> M15 --> M5 --> M1
```

Higher timeframes provide context, not automatic vetoes. Their importance is
conditioned on the planned holding horizon. A short-lived setup may oppose 1d,
but it must then meet stricter timing, space, and invalidation requirements.

## Competing hypotheses example

```mermaid
flowchart TD
    S[Aegis proposes SHORT] --> C[Build one frozen snapshot]
    C --> T[Continuation SHORT<br/>0.48]
    C --> P[Pullback SHORT<br/>0.57]
    C --> B[Breakout LONG<br/>0.64]
    C --> V[Transition LONG<br/>0.67]
    T --> R[Router]
    P --> R
    B --> R
    V --> R
    R --> Q{Dominance margin sufficient?}
    Q -- No --> W[WAIT / SKIP]
    Q -- Yes --> A1[Select one hypothesis]
```

The numeric values are illustrative. Opposing specialists may be evaluated for
research, but an Aegis signal-conditioned deployment cannot flip direction
unless a separately approved directional-routing experiment exists.

## Position lifecycle

```mermaid
stateDiagram-v2
    [*] --> FLAT
    FLAT --> PENDING: candidate exists
    PENDING --> FLAT: timeout / cancel / ambiguity
    PENDING --> OPEN: dominant hypothesis confirmed
    OPEN --> INVALIDATING: current thesis loses evidence
    INVALIDATING --> OPEN: thesis recovers
    INVALIDATING --> EXITING: invalidation confirmed
    OPEN --> EXITING: risk or planned exit
    EXITING --> COOLDOWN
    COOLDOWN --> FLAT

    note right of EXITING
      Never LONG directly to SHORT
      Never SHORT directly to LONG
    end note
```

Entry arbitration and position management remain separate experiments. The
first implementation evaluates only `FLAT -> PENDING -> ENTER/SKIP`.

