# GEN2-EQM1 — Entry Quality Model Specification

**Status:** FROZEN (owner ordered implementation 2026-07-11) — eqm1-spec-v1.0
**Input:** d3-v1-build_a (sha 86be5a15…), TRRM V2 congelado (rv2/20260711T171832Z, sha 69c03e12…)

## Hipótesis
- **H1:** existe un modelo causal cuyo ranking de EV separa calidad económica: decil superior de score con `net_quality_after_costs` medio > 0 en TODOS los folds, sobre la población retenida por el veto.
- **H2 (estabilidad):** worst-fold Spearman(score, net_quality) ≥ 0.60 × mean-fold.
- **H3 (incrementalidad):** TRRM+EQM > EQM-solo y > TRRM-solo en net_quality medio del top-k en ≥3/4 folds.
- **H0:** ningún ranking supera selección aleatoria → GEN2_EQM_REJECTED.

## Definición de oportunidad (congelada)
Una oportunidad = fila H12 del dense (H6/H24 = diagnóstico). Límite de correlación: ≤1 entrada por ventana de 30 min a nivel portafolio, resuelta por score (evaluado con y sin límite).

## Veto TRRM (congelado, no reentrenar/recalibrar)
Score = P(tail) calibrada isotónica del bundle RV2. Veto per-fold past-only: umbral = percentil 70 de los scores del train del fold (budget 30%, protocolo E2.1 validado). Población principal EQM = retenida.

## Targets (dos, nunca mezclados)
- Regresión: `future_eval.net_quality_after_costs` (ya neto de fees+slippage bajo semántica V4 congelada, SHORT, H12).
- Clasificación: `label.clean_entry_v4`.
- Score compuesto: EV = P(clean)·E[quality]; componentes evaluados por separado; si la mezcla no mejora, gana el más simple.

## Candidatos (configs pequeñas pre-registradas, sin grid search)
RF, ExtraTrees, HGB, GB (reg y clf), referencia lineal (inelegible). Ensemble pequeño (media RF+HGB+ET) solo como challenger con abstención por desacuerdo (std del score > p80 ⇒ abstener); se adopta solo si mejora material (≥ +10% net_quality top-k o mejor estabilidad).

## Folds y selección
Mismos 4 folds expansivos y embargos de RV2 (mismo módulo). Desarrollo < 2026-04-27. Selección pre-lockbox. Budgets: top 5/10/15/20/30%. Prioridad: worst-fold, luego media, luego calibración, luego simplicidad.

## Decisión (auditor)
GEN2_EQM_READY (H1∧H2∧H3), GEN2_EQM_PROMISING (H1 y una falla), GEN2_EQM_REJECTED (H0), LEAKAGE_RISK_TOO_HIGH, DATASET_CONTRACT_ERROR. READY ≠ rentabilidad: eso lo decide ECON1.
