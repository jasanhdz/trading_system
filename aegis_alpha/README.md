# Aegis Alpha

Aegis Alpha is the new ML signal platform that replaces the Phantom V30 experiment line.

Design priorities:

- Protect capital first.
- Learn a prudent supervised baseline before PPO refinement.
- Evaluate by survivor utility, regimes, fees, direction dominance, and SignalQ.
- Keep a stable FastAPI inference contract for the TypeScript trading bot.

Initial production behavior is defensive: if no champion model exists, inference returns `IDLE`.
