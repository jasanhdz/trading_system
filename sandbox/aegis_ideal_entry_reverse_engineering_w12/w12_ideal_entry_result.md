# W12 Ideal Entry Reverse Engineering Result

## Veredicto

**C - NO_IDEAL_ENTRY_PREDICTIVE_EDGE**

Leakage audit: **PASS**. External holdouts accessed: **false**.

## Preguntas Finales

### 1. ¿Las entradas ideales presentan características detectables antes de ocurrir?

El modelo tuvo PR AUC 0.169 frente a prevalencia 0.166; veredicto NO_IDEAL_ENTRY_PREDICTIVE_EDGE.

### 2. ¿Qué features las distinguen?

Las mayores diferencias discovery, sin autoridad de selección posterior, fueron: [{'feature': 'cross_sectional_dispersion_15m_bps', 'standardized_median_difference': -0.018224152927013997, 'rank_biserial': -0.016306534760999303, 'decile_monotonicity': -0.5334217542773644, 'mutual_information': 0.05415945597581562}, {'feature': 'eth_return_15m_bps', 'standardized_median_difference': -0.008579898324767262, 'rank_biserial': -0.0009796330007384446, 'decile_monotonicity': -0.05324287870462868, 'mutual_information': 0.05325406171657421}, {'feature': 'alt_basket_return_15m_bps', 'standardized_median_difference': -0.010159381513841225, 'rank_biserial': 0.00034156802382701024, 'decile_monotonicity': 0.005288903672340914, 'mutual_information': 0.052564760964645574}, {'feature': 'btc_return_15m_bps', 'standardized_median_difference': -0.005100108591985202, 'rank_biserial': 0.0029741745662281716, 'decile_monotonicity': 0.10075358572796149, 'mutual_information': 0.05225302014129074}, {'feature': 'interaction_momentum_x_flow', 'standardized_median_difference': -0.00655817601055609, 'rank_biserial': -0.011823144537866037, 'decile_monotonicity': -0.6234500110468283, 'mutual_information': 0.01366626324202258}, {'feature': 'relative_volume_20m', 'standardized_median_difference': -0.008854482424423745, 'rank_biserial': -0.014657467930969004, 'decile_monotonicity': -0.9111913352249579, 'mutual_information': 0.013535606997238103}, {'feature': 'realized_vol_60m_bps', 'standardized_median_difference': -0.02611925407970804, 'rank_biserial': -0.02088137641001575, 'decile_monotonicity': -0.2743542295861701, 'mutual_information': 0.013046363966599639}, {'feature': 'atr_15m_bps', 'standardized_median_difference': -0.02489892043930666, 'rank_biserial': -0.02191399961256324, 'decile_monotonicity': -0.28648511951000777, 'mutual_information': 0.012867100175785628}].

### 3. ¿Existen secuencias temporales recurrentes antes de ellas?

Se incluyeron perfiles T-60/T-30/T-15/T-5/T-1; su evidencia está en feature_analysis y no se añadió ninguna secuencia post hoc.

### 4. ¿LONG y SHORT tienen ADN diferente?

Resultados por lado predicho: [{'dimension': 'predicted_side', 'value': 'LONG', 'trades': 997, 'gross_mean_bps': 0.07460769066881187, 'net14_mean_bps': -13.925392309331189, 'ideal_precision': 0.14543630892678033}, {'dimension': 'predicted_side', 'value': 'SHORT', 'trades': 1650, 'gross_mean_bps': -0.636547307596976, 'net14_mean_bps': -14.636547307596976, 'ideal_precision': 0.16606060606060605}].

### 5. ¿Qué teacher produce labels más consistentes?

Prevalencia prospectiva media por teacher: {'A': 0.20071116968692268, 'B': 0.28911999600173893, 'C': 0.2630974833897326, 'D': 0.0031705537627440728, 'E': 0.28911999600173893}; consistencia completa en label_analysis.csv.

### 6. ¿Consensus labels funcionan mejor?

Majority fue el label primario preregistrado; strict y weighted se conservaron como diagnóstico y no reemplazaron el resultado.

### 7. ¿Es mejor clasificación o quality score?

La formulación seleccionada fue OPPORTUNITY_THEN_SIDE (OPPORTUNITY_THEN_SIDE_60M).

### 8. ¿Qué horizonte funciona mejor?

El mejor horizonte validado y congelado fue 60 minutos.

### 9. ¿Cuántas oportunidades A+ existen?

Existían 91784 best-entry zones prospectivas entre todos los teachers/sides/horizontes.

### 10. ¿Qué porcentaje del mercado es SKIP?

El sistema permaneció SKIP en 97.719% del universo del candidato.

### 11. ¿Cuál es precision@top1/2/5/10%?

Precision top 1/2/5/10%: {'1': 0.14537107880642694, '2': 0.15829240649792217, '5': 0.16941953751769703, '10': 0.1709025100322606}.

### 12. ¿Cuál es gross bps?

Gross medio del top 2% congelado: -0.369 bps.

### 13. ¿Cuál es net bps con 14 bps?

Neto a 14 bps: -14.369 bps.

### 14. ¿Cuál es net bps con 20 bps?

Neto a 20 bps: -20.369 bps.

### 15. ¿Sobrevive fuera de muestra?

Prospective fue abierto una vez tras selección en validation; produjo 2647 señales.

### 16. ¿Sobrevive walk-forward?

No se ejecutó un walk-forward rolling. La evidencia disponible es un único recorrido discovery→validation→prospective con meses prospectivos reportados por separado; por tanto, este diagnóstico no demuestra supervivencia walk-forward.

### 17. ¿Depende de un símbolo?

Se operaron 10 símbolos; máxima concentración y detalle: [{'dimension': 'symbol', 'value': 'AVAXUSDT', 'trades': 339, 'gross_mean_bps': -0.428158627766992, 'net14_mean_bps': -14.428158627766992, 'ideal_precision': 0.17699115044247787}, {'dimension': 'symbol', 'value': 'SOLUSDT', 'trades': 326, 'gross_mean_bps': -1.4643663405026388, 'net14_mean_bps': -15.46436634050264, 'ideal_precision': 0.13803680981595093}, {'dimension': 'symbol', 'value': 'BNBUSDT', 'trades': 322, 'gross_mean_bps': 1.1878487426947888, 'net14_mean_bps': -12.812151257305212, 'ideal_precision': 0.18012422360248448}, {'dimension': 'symbol', 'value': 'ADAUSDT', 'trades': 295, 'gross_mean_bps': -0.4111470388366824, 'net14_mean_bps': -14.411147038836683, 'ideal_precision': 0.13898305084745763}, {'dimension': 'symbol', 'value': 'LTCUSDT', 'trades': 269, 'gross_mean_bps': 2.3058982757660385, 'net14_mean_bps': -11.69410172423396, 'ideal_precision': 0.1895910780669145}].

### 18. ¿Depende de un periodo?

Resultados mensuales: [{'dimension': 'month', 'value': '2022-12', 'trades': 878, 'gross_mean_bps': -0.2461693994038539, 'net14_mean_bps': -14.246169399403852, 'ideal_precision': 0.1469248291571754}, {'dimension': 'month', 'value': '2023-01', 'trades': 739, 'gross_mean_bps': 1.085812356820343, 'net14_mean_bps': -12.914187643179657, 'ideal_precision': 0.17320703653585928}, {'dimension': 'month', 'value': '2023-02', 'trades': 578, 'gross_mean_bps': -1.8713591299898242, 'net14_mean_bps': -15.871359129989822, 'ideal_precision': 0.15051903114186851}, {'dimension': 'month', 'value': '2023-03', 'trades': 452, 'gross_mean_bps': -1.063168610216231, 'net14_mean_bps': -15.063168610216232, 'ideal_precision': 0.16592920353982302}].

### 19. ¿Supera baselines?

Baselines completos: [{'name': 'ALWAYS_SKIP', 'kind': 'BASELINE', 'trades': 0, 'gross_mean_bps': nan, 'net14_mean_bps': nan, 'net20_mean_bps': nan}, {'name': 'ALWAYS_LONG', 'kind': 'BASELINE', 'trades': 116040, 'gross_mean_bps': -0.4160594144683939, 'net14_mean_bps': -14.416059414468394, 'net20_mean_bps': -20.416059414468393}, {'name': 'ALWAYS_SHORT', 'kind': 'BASELINE', 'trades': 116040, 'gross_mean_bps': 0.05930433358278497, 'net14_mean_bps': -13.940695666417215, 'net20_mean_bps': -19.940695666417216}, {'name': '15M_MOMENTUM', 'kind': 'BASELINE', 'trades': 116040, 'gross_mean_bps': -0.6998662123630119, 'net14_mean_bps': -14.699866212363013, 'net20_mean_bps': -20.699866212363013}, {'name': '15M_MEAN_REVERSION', 'kind': 'BASELINE', 'trades': 116040, 'gross_mean_bps': 0.34311113147740285, 'net14_mean_bps': -13.656888868522596, 'net20_mean_bps': -19.656888868522596}].

### 20. ¿Supera negative controls?

Controles negativos: [{'name': 'RANDOM_ENTRIES_MEAN', 'kind': 'NEGATIVE_CONTROL_DISTRIBUTION', 'trades': 100, 'gross_mean_bps': -0.1450566459275617, 'net14_mean_bps': -14.145056645927562, 'net20_mean_bps': -20.14505664592756}, {'name': 'TIME_SHIFT_24H', 'kind': 'NEGATIVE_CONTROL', 'trades': 2631, 'gross_mean_bps': -0.14487058651715562, 'net14_mean_bps': -14.144870586517156, 'net20_mean_bps': -20.144870586517154}, {'name': 'LABEL_SHUFFLE_MODEL', 'kind': 'NEGATIVE_CONTROL', 'trades': 1662, 'gross_mean_bps': -0.577438630380903, 'net14_mean_bps': -14.577438630380904, 'net20_mean_bps': -20.577438630380904}, {'name': 'RANDOM_FEATURE_MODEL', 'kind': 'NEGATIVE_CONTROL', 'trades': 2297, 'gross_mean_bps': -0.33892119483170363, 'net14_mean_bps': -14.338921194831704, 'net20_mean_bps': -20.338921194831702}]; gate falló.

### 21. ¿Existe realmente un subconjunto A+ económicamente rentable?

El subconjunto top 2% no fue rentable después de 14 bps.

### 22. ¿Existe suficiente evidencia para justificar un W12.1?

W12.1 no está justificado por los gates preregistrados.

### 23. ¿Existe suficiente evidencia para considerar Shadow?

Shadow no está justificado: incluso grado A solo autorizaría una segunda fase gobernada, nunca integración automática.

### 24. ¿O debemos cerrar esta línea?

Decisión final: NO_IDEAL_ENTRY_PREDICTIVE_EDGE; la línea se cierra con esta evidencia.

## Inferencia

Bootstrap UTC-day de 10000 draws: CI 95% [-15.296, -13.424] bps net14; P(mean>0)=0.000.

## Limitaciones

Los labels usan OHLC 1m y resolución adverse-first, pero no BBO, queue position, funding ni fills observados. MFE es movimiento disponible; la economía primaria usa una política fija +30/-20/neither-horizon. Los datos 2022-2023 no prueban vigencia actual.

## Autoridad

Este resultado no autoriza E4, TypeScript, Shadow, producción, órdenes ni despliegue.
