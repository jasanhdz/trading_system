# Aegis Long Edge Research Summary

## Scope

This document summarizes the v0.4 long-edge research branch and explains why it is frozen as a research signal rather than promoted.

## What Was Tried

- BC prudent labeling variants.
- Long-only edge model v0.3.0.
- Regime gating.
- Risk guard calibration.
- Dynamic sizing.
- Meta-filtering.
- Score floor filtering.
- OOS validation across 100+ windows.
- Failure analysis of the OOS tail.
- Edge deterioration guards.

## What Worked

- A long edge existed in the median.
- Regime gating materially improved behavior versus naive LONG entry.
- Dynamic sizing reduced tail risk compared with fixed sizing.
- Risk guard settings reduced drawdown and trading frequency.
- The architecture is operationally safer than earlier branches.

## What Did Not Work

- No variant passed OOS as champion.
- Tail behavior remained weak: p25_pf stayed fragile and worst_balance stayed below the desired floor.
- Meta-filtering improved average trade quality but worsened or failed to repair the tail.
- Score floor filtering did not restore robust worst-balance behavior.
- Edge deterioration guards reduced re-entry after losses, but did not fix the tail.
- Fee stress continued to degrade performance materially.

## Final Status

- `aegis_long_edge_dynamic_v042` is a `RESEARCH_SIGNAL`.
- It is not champion.
- It is not live.
- It remains a benchmark for future signal families and horizon research.

