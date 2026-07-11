# GEN2-RV2 — TRRM V2 + QMAE V2 Specification

**Status:** FROZEN (owner ordered immediate implementation 2026-07-11)
**Spec version:** rv2-spec-v1.0
**Input:** dataset `d3-v1-build_a` (D3_CANONICAL_READY), target frozen `target.tail_risk_roe_030`
**Lockbox:** GEN2_LOCKBOX_MANIFEST.json — development < 2026-04-27; semi-blind 2026-04-27→2026-07-11 (1 registered query per frozen candidate); forward untouched.

## 1. Hipótesis

- **H1 (señal):** al menos un modelo de árboles entrenado desde cero sobre D3 canónico alcanza PR-AUC ≥ 2× prevalencia en TODOS los folds de desarrollo.
- **H2 (estabilidad):** el mejor candidato cumple worst-fold PR-AUC ≥ 0.80 × mean-fold PR-AUC (consistencia sobre suerte).
- **H3 (riesgo cuantílico):** QMAE V2 con conformal split logra cobertura empírica q90 ∈ [0.87, 0.93] en todos los folds de validación, y pinball loss ≤ 0.9× el baseline incondicional.
- **H0 (nula):** ningún candidato supera consistentemente el baseline de prevalencia → GEN2_TRRM_REJECTED.

## 2. Variables

- **Controladas (congeladas):** dataset d3-v1-build_a, target tail_risk_roe_030, features (111 + 3 one-hot de horizon, diseño GLOBAL_CAUSAL_PLUS_HORIZON validado en Gen1), folds, embargo 120 min, seeds, entorno venv_rocm62.
- **Independiente:** familia/configuración del modelo.
- **Dependientes:** PR-AUC, ROC-AUC, capture@30% budget, Brier, ECE por fold; para QMAE: pinball q50/q90, cobertura conformal, estabilidad temporal de cobertura.

## 3. Diseño experimental

- **Población de desarrollo:** filas dense con id.timestamp < 2026-04-27 (≈ periodo 2024-07 → 2026-04).
- **Folds:** 4 expansivos: train hasta el 50/60/70/80% temporal del desarrollo, validación el siguiente 10%, embargo 120 min en la frontera. Idéntico protocolo para todos los candidatos; prohibido alterar folds durante la comparación.
- **Selección pre-registrada (en este orden):** (1) worst-fold PR-AUC, (2) mean PR-AUC, (3) menor std entre folds, (4) menor tiempo de entrenamiento. La regla favorece estabilidad sobre picos, por instrucción del owner.
- **Candidatos TRRM y justificación:**
  - RandomForest — incumbente Gen1; control de continuidad.
  - HistGradientBoosting — boosting con binning: rápido (crítico para el ciclo de retraining futuro), missing nativo, capacidad de monotonic constraints.
  - GradientBoosting — boosting clásico con subsample (stochastic GB): sesgo distinto al de HGB, históricamente estable en tabular pequeño-mediano.
  - ExtraTrees — splits aleatorizados: máxima reducción de varianza; candidato natural cuando el criterio es estabilidad.
  - LogisticRegression — piso lineal de referencia: si un árbol no supera al lineal, la complejidad no está justificada (no elegible como ganador de fase, solo referencia).
  - No se incluyen XGBoost/LightGBM/CatBoost: no están en el entorno congelado y está prohibido instalar dependencias.
- **Calibración:** para el ganador, sigmoid (Platt) e isotónica ajustadas SOLO en el tramo de calibración de cada fold; se selecciona por ECE medio; se reporta Brier antes/después.
- **QMAE V2:** HistGradientBoosting con pérdida quantile (q50 y q90) sobre `future_eval.future_mae_roe_proxy` + **conformalización split** (ajuste del q90 con residuos del tramo de calibración) → cobertura garantizada bajo intercambiabilidad. Baseline a superar: cuantil incondicional del train. Cobertura reportada global, por fold, por símbolo y por horizon.
- **Candidato congelado:** el ganador se re-entrena sobre TODO el desarrollo (calibración en el último 10% con embargo) y se congela por hash con manifest.
- **Consulta semi-ciega (única, registrada):** el candidato congelado se evalúa UNA vez sobre 2026-04-27→fin; la apertura se registra en GEN2_LOCKBOX_MANIFEST (query_log, contador). Uso: verificación de no-contradicción (no selección). Contradicción = PR-AUC semi-blind < 0.5 × mean dev → bloquea READY.

## 4. Criterios de decisión (emite el auditor)

- **GEN2_TRRM_READY:** H1 ∧ H2 ∧ H3 ∧ semi-blind no contradice ∧ artifacts íntegros.
- **GEN2_TRRM_PROMISING:** H1 ∧ (falla exactamente una de H2/H3) — investigable en RV2.1 sin bloquear el roadmap.
- **GEN2_TRRM_PARTIAL:** TRRM cumple (H1∧H2) pero QMAE falla H3 de forma no reparable con recalibración, o viceversa.
- **GEN2_TRRM_REJECTED:** H0 no rechazada, o semi-blind contradice, o integridad de artifacts rota.
- **Condición de paro de la investigación:** REJECTED, o cualquier evidencia de fuga (features no causales, folds contaminados) → detener y reportar, no iterar configuraciones.

## 5. Reproducibilidad y seguridad

Seeds fijos; manifests con hashes de dataset de entrada, configs, código y entorno; artifacts bajo `/home/jasan/Develop/aegis_gen2/rv2/<stamp>/`; ninguna escritura fuera de aegis_gen2; ningún artefacto Gen1 reutilizado (pesos desde cero); prohibido: live, shadow, promotion, EQM, tocar forward.
