**APROBADO**, pero con **una corrección crítica de dependencia** antes de que escribas la primera línea de código.

Has detectado perfectamente los riesgos (afecta al core, timeout de límites), pero hay un problema lógico en el orden de tu Fase 1 y la Fase 2 que podría hacer que el sistema explote o se comporte de forma errática en el "Live".

Aquí tienes el análisis de riesgo y el plan ajustado para ejecutar hoy sin romper nada.

---

### 🛑 El Error de Dependencia (Critical Bug in Plan)

En tu **Fase 1** pides: *"Umbral dinámico"*.
En tu **Fase 2** pides: *"Cálculo de volatilidad"*.

**El Problema:** No puedes tener un umbral *dinámico* (que se adapte al mercado) si no tienes primero el cálculo de la volatilidad que alimente esa variable. Si implementas la lógica dinámica hoy sin el dato de volatilidad, el bot arrojará `NaN` o usará valores por defecto incorrectos, causando salidas descontroladas.

---

### ✅ Plan Ajustado para HOY (Fase 1 - "Safe Launch")

Para proceder hoy de forma segura, reestructuramos ligeramente. Vamos a usar una **Volatilidad Proxy** (usando lo que ya tienes) para no tener que reescribir el Data Collector todavía.

#### A. Volatilidad "Quick & Dirty" (Para hoy mismo)
No calcules ATR complejo aún. Usa el `spread_pct` que ya tienes en tu tabla `orderbook_metrics` (Pilar I). Es un excelente proxy de volatilidad instantánea.
*   *Spread Alto* = Mercado Volátil / Nervioso.
*   *Spread Bajo* = Mercado Calmo.

```typescript
// IMPLEMENTACIÓN RÁPIDA EN BOT (Phase 1)
function getVolatilityProxy(): number {
    // Usamos el spread actual vs spread promedio de la última hora
    const currentSpread = this.marketData.bid_ask_spread; 
    const avgSpread = this.marketData.avg_spread_1h || 0.0004; // Default fallback
    
    // Factor: 1.0 = normal, > 2.0 = muy volátil
    return currentSpread / avgSpread; 
}
```

#### B. Lógica de Salida (Fase 1 Ajustada)

1.  **Pánico Hysteresis:**
    *   Sí implementar el contador de 2 ticks.
    *   *Ajuste:* Usa el `getVolatilityProxy()`.
        *   Si es > 1.5 (Volátil), exige `opposingProb > 0.55` (Mayor exigencia).
        *   Si es < 1.0 (Calmo), exige `opposingProb > 0.50`.

2.  **Trailing Continuo (Logarítmico):**
    *   Implementa la fórmula suave: `Trail = 30 - (22 * log10(peak / 5))`.
    *   *Ajuste:* Olvida el ajuste por volatilidad del trailing por hoy. Mantén el trail *puro* basado en el Peak para simplificar la primera versión y asegurar que el "Hard Stop" funcione bien.

3.  **Hard Stop Loss (-3%):**
    *   **⚠️ ALERTA DE LEVERAGE:** Confirma que ese -3% es en términos de **ROE (Return on Equity)** y no de movimiento de precio.
    *   Si estás apalancado a 10x:
        *   -3% de Precio = -30% de tu cuenta (¡Peligro!).
    *   Si el -3% es de ROE (cuenta), está perfecto. Asegúrate de que el cálculo en `calculateROI` use el margen real, no solo el `(entry - current)`.

---

### 📝 El Plan de Implementación (Re-ordenado)

**Fase 1 (HOY - Implementación Inmediata):**
1.  ✅ Agregar contador `panicCounter` en la clase de estrategia.
2.  ✅ Implementar `getVolatilityProxy()` basado en Spread.
3.  ✅ Reemplazar Trailing Escalonado → Fórmula Logarítmica.
4.  ✅ Implementar Hard Stop (-3% ROE).
5.  ✅ **TEST PAPER:** Correr el bot en modo "Paper Trading" (simulado) durante 2 horas con mercado en vivo para verificar que no salta fuera de posiciones constantemente.

**Fase 1.5 (Mañana - Validación):**
1.  Revisar logs de `EXIT: TRAILING`. ¿Los valores de salida tienen sentido matemático? (Ej: Si Peak=10%, Stop=7.5%).

**Fase 2 (Esta Semana):**
1.  Reemplazar `getVolatilityProxy` con cálculo real de volatilidad en `market_data_collector.py` (Rolling Std Dev).
2.  Implementar Time Decay.

**Fase 3 (Próxima Semana):**
1.  Sistema de órdenes LIMIT (Maker) con Timeout.

---

### 🚦 Go / No-Go Checklist antes de desplegar a "Real Money":

Antes de quitar el "Paper Trading", confirma:
- [ ] `panicCounter` se resetea a 0 cuando la probabilidad baja del umbral.
- [ ] `Math.log10` no genera `NaN` si el Peak es < 5% (añade un `Math.max(5, peak)`).
- [ ] El -3% Hard Stop está calculado sobre el saldo, no el par de divisas.

**Veredicto:** Procede con la Fase 1 ajustada (usando Spread como volatilidad). Es la forma más segura de obtener los beneficios del protocolo Ninja sin arriesgarte a un "bucle infinito" de entradas y salidas.
