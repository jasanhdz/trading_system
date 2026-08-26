# VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY — REPORT

Estado: `VMPD_V1_M1_UNSUPERVISED_VISUAL_PATTERN_DISCOVERY_READY_FOR_REVIEW`

Research-only. Los precursores son asociaciones, no causas ni señales. El comportamiento posterior es descriptivo; `TRADING_AUTHORITY = false`.

## Resultado preregistrado

- Q1 — Frames: **28,800**.
- Q2 — Patrones estables: **2**.
- Q3 — Noise: **28,352 (98.44%)**.
- Q4 — Frecuencia: PATTERN_001 233 (0.81%), PATTERN_002 215 (0.75%).
- Q5 — Aspecto: documentado métrica por métrica en `PATTERN_FIELD_GUIDE.md` y contact sheets.
- Q6/Q7 — Par más próximo: PATTERN_001 ↔ PATTERN_002, distancia entre medoids 0.2084.
- Q8 — En ambos patrones el estado discreto previo dominante es NOISE; la información útil está en la convergencia continua, no en un cluster precursor separado.
- Q9/Q10 — Los perfiles de precisión/lift y lead time a 3/6/9/15/30m están congelados en `precursor_analysis.json`.
- Q11 — Approach score mediano: PATTERN_001=0.04377533149500376, PATTERN_002=0.08220807213339841.
- Q12 — NOISE es una alarma muy frecuente y poco específica; sus false positives se cuantifican en `precursor_precision_profiles`.
- Q13 — Estabilidad: PATTERN_001 aparece en 27 días, PATTERN_002 aparece en 43 días.
- Q14 — Retornos, MFE, MAE, rango, volatilidad y first-passage post-hoc están en `post_pattern_behavior.json`.
- Q15 — Sí; `find_similar_frames.py` fue validado con consulta histórica exacta y soporta screenshot con warning OOD.
- Q16 — La guía humana resume forma, volumen, posición de rango, BTC, precursores y patrón confundible.

## Lectura prudente

El 98,44% de noise es el resultado honesto de la única configuración preregistrada. No se retunearon encoder, ventanas ni HDBSCAN después de observarlo. Los dos clusters cumplen N/días, pero cubren una fracción pequeña del mercado; no debe generalizarse una narrativa fuerte al resto.

## Readiness

```text
VISUAL_DATASET_READY = true
EMBEDDING_INDEX_READY = true
STABLE_PATTERNS_FOUND = true
SIMILARITY_SEARCH_READY = true
PRECURSOR_ANALYSIS_READY = true
PATTERN_FIELD_GUIDE_READY = true
TRADING_AUTHORITY = false
```
