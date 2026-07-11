# GEN2-ECON1 — Economic Backtest Specification

**Status:** FROZEN — econ1-spec-v1.0. Motor nuevo, aislado; ningún backtest previo reutilizado.

## Simulación (caso base, sin gestión)
SHORT, entrada = open de la barra siguiente a la señal (causal), salida = close a +12 barras (H12). Sin trailing/SL/TP/martingala/reinversión. Notional fijo $100/trade. PnL = (entry−exit)/entry × notional − costos.

## Costos pre-registrados (por lado, sobre notional)
| escenario | fee | slippage | funding/h |
|---|---|---|---|
| A optimista | 4bps | 1bp | 0.5bp |
| B base | 5bps | 2bps | 1bp |
| C pesimista | 5bps | 5bps | 2bps |
Aprobación exige pasar en B y no colapsar en C (PF_C ≥ 1.0).

## Baselines (idéntico presupuesto de trades, misma población y ventanas, mismo límite de correlación)
random+TRRM, momentum-rule (short si momentum_12<0 y close<ema_24; score=−momentum_12), vol-rule (score=atr_proxy_24), TRRM-solo (score=−P(tail)), reglas+TRRM, EQM-solo, EQM+TRRM. Phase O no reconstruible honestamente sobre D3 (modelos Gen1 inválidos) — documentado, se omite.

## Criterios de aprobación (pre-registrados, simultáneos, escenario B)
1) expectancy neta > 0 agregada; 2) > 0 en ≥3/4 folds; 3) límite inferior del CI bootstrap por bloques semanales ≥ −$0.02/trade; 4) PF ≥ 1.5 objetivo / ≥1.3 mínimo duro; 5) ningún fold con net < −30% del gross ganado; 6) max drawdown ≤ 25× expectancy·√N pre-registrado como ≤ $150 por $100-notional stream; 7) > random; 8) > ≥2 baselines de reglas; 9) share por símbolo ≤ 40% del net; 10) share por mes ≤ 40%; 11) ningún trade > 30% del net; 12) PF_C ≥ 1.0; 13) trades ≥ 300; 14) TRRM incremental (quitar veto empeora net/trade); 15) EQM incremental (EQM+TRRM > reglas+TRRM). Si solo reglas+TRRM pasa → RULES_PLUS_TRRM_CHAMPION.

## Incertidumbre
Bootstrap por bloques semanales (1,000 réplicas) para expectancy/net/PF/Δ-vs-baselines. Sharpe/Sortino solo diagnósticos.

## Decisión (auditor)
GEN2_ECONOMIC_EDGE_READY / RULES_PLUS_TRRM_CHAMPION / GEN2_ECONOMIC_EDGE_PROMISING / GEN2_ECONOMIC_EDGE_REJECTED / BACKTEST_INTEGRITY_ERROR / LEAKAGE_RISK_TOO_HIGH. Solo READY o RULES_CHAMPION permiten freeze; después freeze total por hashes + collector forward sin enforcement (enforcement_action=NONE, manual, sin PM2/cron). Live: emitir LIVE_NOT_ELIGIBLE o LIVE_CANDIDATE_PENDING_FORWARD; nunca activar.
