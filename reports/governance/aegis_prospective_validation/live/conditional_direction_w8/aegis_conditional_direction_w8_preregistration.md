# Aegis W8 Conditional Direction - Preregistration

- Config SHA-256: `f537438b4664666b118a2dd5ee8363d9fa47fff0ee59ff8da2b7f44047918770`
- W1-W7 holdouts: prohibited and unopened.
- Unit: independent paired opportunity episode.
- W7 Opportunity model: frozen `H60:LOGISTIC_L2:P_GTE_0.7`; no W8 retuning.
- Targets: symmetric LONG/SHORT 30 bps barriers, 14 bps costs, adverse-first same-bar resolution, and explicit SKIP.
- TRAIN: 2025-08-09 to 2026-01-01.
- VALIDATION: 2026-01-01 to 2026-05-01.
- FINAL_HOLDOUT_W8: future independent evidence, `SEALED_NOT_OPENED`.
- Primary metric: net expectancy without leverage.
