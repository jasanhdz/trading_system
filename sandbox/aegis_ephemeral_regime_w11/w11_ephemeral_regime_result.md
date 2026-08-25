# W11 Ephemeral Regime Result

Generado con datos hasta `2024-01-01T00:00:00+00:00`.

## Veredicto

**B - EPHEMERAL_SIGNAL_DETECTED_NOT_YET_ECONOMIC**

## Preguntas Requeridas

### 1. ¿Existe edge efímero?

Se detectaron candidatos locales en validación, pero no edge prospectivo: EPHEMERAL_SIGNAL_DETECTED_NOT_YET_ECONOMIC y -15.834 bps netos por trade.

### 2. ¿En qué ventanas aparece?

Economía por ventana: [{'window_hours': 6, 'trades': 452, 'gross_mean_bps': -14.08081538950969, 'net14_mean_bps': -28.08081538950969, 'net20_mean_bps': -34.08081538950969}, {'window_hours': 12, 'trades': 274, 'gross_mean_bps': 1.691793613000781, 'net14_mean_bps': -12.308206386999222, 'net20_mean_bps': -18.30820638699922}, {'window_hours': 24, 'trades': 265, 'gross_mean_bps': -7.078495185040979, 'net14_mean_bps': -21.078495185040975, 'net20_mean_bps': -27.07849518504098}, {'window_hours': 48, 'trades': 434, 'gross_mean_bps': 6.123671153020397, 'net14_mean_bps': -7.876328846979603, 'net20_mean_bps': -13.876328846979602}, {'window_hours': 72, 'trades': 497, 'gross_mean_bps': 3.2093886482953375, 'net14_mean_bps': -10.790611351704664, 'net20_mean_bps': -16.790611351704662}]. La mejor es la de mayor net14, aunque siga negativa.

### 3. ¿LONG, SHORT o ambos?

Hubo 986 LONG y 936 SHORT. Economía por lado: [{'side': 'LONG', 'trades': 986, 'gross_mean_bps': -3.0862753698814216, 'net14_mean_bps': -17.08627536988142, 'net20_mean_bps': -23.08627536988142}, {'side': 'SHORT', 'trades': 936, 'gross_mean_bps': -0.513858308563364, 'net14_mean_bps': -14.513858308563364, 'net20_mean_bps': -20.513858308563364}].

### 4. ¿Qué horizonte funciona mejor?

Economía por horizonte: [{'horizon_minutes': 5, 'trades': 36, 'gross_mean_bps': 0.5597175638315406, 'net14_mean_bps': -13.440282436168461, 'net20_mean_bps': -19.44028243616846}, {'horizon_minutes': 15, 'trades': 67, 'gross_mean_bps': 5.427849328743541, 'net14_mean_bps': -8.572150671256459, 'net20_mean_bps': -14.572150671256459}, {'horizon_minutes': 30, 'trades': 527, 'gross_mean_bps': 9.277855396538188, 'net14_mean_bps': -4.722144603461813, 'net20_mean_bps': -10.722144603461814}, {'horizon_minutes': 60, 'trades': 1292, 'gross_mean_bps': -6.809043670911583, 'net14_mean_bps': -20.809043670911585, 'net20_mean_bps': -26.80904367091159}]. El mejor es el de mayor net14, sin reinterpretarlo como ganador.

### 5. ¿Cuál es el edge bruto?

El gross medio orientado fue -1.834 bps.

### 6. ¿Cuál es el edge neto?

El neto medio fue -15.834 bps a 14 bps, -21.834 bps a 20 bps y -31.834 bps a 30 bps.

### 7. ¿Cuánto dura aproximadamente?

La vida económica se observa por buckets hasta 48h; medianas netas: {'0_1h': -10.381832608301663, '1_3h': -18.190095471637914, '3_6h': -16.494501451048443, '6_12h': -17.446668006248537, '12_24h': -12.714100203956995, '24_48h': -11.004834240665124}.

### 8. ¿Cuál es el edge half-life?

El sistema no tuvo edge inicial positivo que pueda partirse por la mitad. Entre instancias aisladas con inicio positivo, la mediana observada fue 3.000h en 41; 59 quedaron censuradas.

### 9. ¿Qué ocurre después de que el modelo envejece?

La sensibilidad TTL fue [{'ttl_hours': 6, 'instances': 100, 'mean_net14_bps': -16.052977870600998}, {'ttl_hours': 12, 'instances': 100, 'mean_net14_bps': -16.691433976672673}, {'ttl_hours': 24, 'instances': 100, 'mean_net14_bps': -15.336943928483743}, {'ttl_hours': 36, 'instances': 100, 'mean_net14_bps': -14.227199431763054}, {'ttl_hours': 48, 'instances': 100, 'mean_net14_bps': -13.70415644541905}]; es descriptiva y no tuvo autoridad de selección.

### 10. ¿Regime similarity predice correctamente la muerte del edge?

Spearman similarity/net fue -0.017; el gate falló.

### 11. ¿Expiration Guardian mejora el resultado?

El Guardian cambió expectancy por trade en -0.090 bps y redujo drawdown acumulado en 64929.579 bps. El gate pasó por reducción de exposición/drawdown; no rescató la economía.

### 12. ¿Cuántas instancias habrían sido creadas?

Se crearon 100 instancias prospectivas (153 incluyendo validación).

### 13. ¿Cuántas expiraron por TTL?

Expiraron 0 por TTL; 0 quedaron censuradas al final del periodo.

### 14. ¿Cuántas por regime drift?

Expiraron 6 por regime drift.

### 15. ¿Cuántas por edge decay?

Expiraron 94 por edge decay.

### 16. ¿Cuántos trades/candidatos genera?

Se ejecutaron 1922 trades; se evaluaron 4880 candidatos y 244 pasaron gates.

### 17. ¿Qué porcentaje del tiempo permanece SKIP?

El sistema permaneció SKIP en 97.016% de snapshots-símbolo prospectivos.

### 18. ¿Supera costos?

Los gates de 14/20 bps no pasaron; CI 95% diario [-23.393, -9.431] bps. Baselines: [{'strategy': 'ALWAYS_SKIP', 'signal_count': 14696, 'trade_count': 0, 'gross_per_trade_bps': 0.0, 'net14_per_trade_bps': 0.0, 'net20_per_trade_bps': 0.0, 'net30_per_trade_bps': 0.0, 'gross_per_signal_bps': 0.0, 'net14_per_signal_bps': 0.0, 'net20_per_signal_bps': 0.0, 'net30_per_signal_bps': 0.0}, {'strategy': 'ALWAYS_LONG', 'signal_count': 14696, 'trade_count': 14696, 'gross_per_trade_bps': 0.6807069149924625, 'net14_per_trade_bps': -13.319293085007537, 'net20_per_trade_bps': -19.319293085007537, 'net30_per_trade_bps': -29.319293085007537, 'gross_per_signal_bps': 0.6807069149924625, 'net14_per_signal_bps': -13.319293085007537, 'net20_per_signal_bps': -19.31929308500754, 'net30_per_signal_bps': -29.31929308500754}, {'strategy': 'ALWAYS_SHORT', 'signal_count': 14696, 'trade_count': 14696, 'gross_per_trade_bps': -0.6807069149924625, 'net14_per_trade_bps': -14.680706914992463, 'net20_per_trade_bps': -20.680706914992463, 'net30_per_trade_bps': -30.680706914992463, 'gross_per_signal_bps': -0.6807069149924625, 'net14_per_signal_bps': -14.680706914992463, 'net20_per_signal_bps': -20.68070691499246, 'net30_per_signal_bps': -30.68070691499246}, {'strategy': '15M_MOMENTUM', 'signal_count': 14696, 'trade_count': 14696, 'gross_per_trade_bps': -0.5866340430854657, 'net14_per_trade_bps': -14.586634043085466, 'net20_per_trade_bps': -20.586634043085464, 'net30_per_trade_bps': -30.586634043085464, 'gross_per_signal_bps': -0.5866340430854657, 'net14_per_signal_bps': -14.586634043085466, 'net20_per_signal_bps': -20.586634043085468, 'net30_per_signal_bps': -30.586634043085468}, {'strategy': '15M_MEAN_REVERSION', 'signal_count': 14696, 'trade_count': 14696, 'gross_per_trade_bps': 0.5866340430854657, 'net14_per_trade_bps': -13.413365956914534, 'net20_per_trade_bps': -19.413365956914536, 'net30_per_trade_bps': -29.413365956914536, 'gross_per_signal_bps': 0.5866340430854657, 'net14_per_signal_bps': -13.413365956914534, 'net20_per_signal_bps': -19.413365956914532, 'net30_per_signal_bps': -29.413365956914532}].

### 19. ¿Es suficientemente robusto para justificar Shadow?

No justifica avanzar a una segunda fase controlada; grado B.

### 20. ¿Justifica alguna integración futura con E4/TS?

No justifica integración futura con E4/TS. Este estudio no concede autoridad de producción.

## Limitations

This is an offline candle study. Funding, spread, queue position, intrabar ordering, and measured slippage are unavailable. The 2023 historical period is not evidence that the same relationship exists now. Repeated symbols share market shocks, and the configured external holdouts remained sealed.

## Authority

This result has no production, Shadow, E4, promotion, or TypeScript authority.
