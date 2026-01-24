# 📉 Phantom V9: Experimento de Trailing ROE

**Objetivo:** Probar si una salida dinámica (Trailing Stop basado en ROE) supera a la salida fija por tiempo.
**Configuración:**
*   Activación: 10% ROE.
*   Callback: 30% del Peak ROE.
*   Modo: Secuencial (Capital Lock).

## 📊 Resultados Comparativos

| Métrica | Salida Fija (4 Horas) | Trailing ROE (Dinámico) | Diferencia |
| :--- | :--- | :--- | :--- |
| **Balance Final** | **$225.60** | $169.49 | -25% |
| **Retorno** | **1,028%** | 747% | Inferior |
| **Trades** | 113 | 117 | Similar |

## 🧠 Análisis del Resultado

El Trailing ROE **perdió dinero** comparado con la estrategia simple de "Hold 4 Hours".
¿Por qué?

1.  **Volatilidad de ETH:** Ethereum es muy ruidoso. Un movimiento que retrocede un 30% de la ganancia (ej. de +20% a +14%) es común antes de seguir cayendo y dar un +40%.
2.  **Salidas Prematuras:** El Trailing Stop nos sacó de operaciones ganadoras demasiado pronto, perdiéndonos la parte más jugosa del movimiento (la "cola" de la distribución).
3.  **La "Segunda Ola":** Muchas señales de Phantom tienen un rebote inicial (drawdown) y luego colapsan. El Trailing a veces se activa en el primer impulso y nos saca antes del colapso real.

## 🏆 Recomendación

**Mantener la Salida por Tiempo (4 Horas).**
A veces, la simplicidad supera a la complejidad. El modelo aprendió a predecir un horizonte de 4 horas, y respetarlo es lo más rentable.
Si se desea usar Trailing, se sugiere un **Callback mucho más amplio (50-60%)** para darle aire al precio.
