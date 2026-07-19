# EXIT_POLICY_PROVENANCE_P1

## Alcance y corte temporal

Esta es una auditoría histórica de solo lectura. El corte de no contaminación es el commit D1A `fcd046cb12fbbd278aee4a0443d37737158fe818`, fechado `2026-07-19T15:38:28Z`. Ningún resultado de D1A se usó para seleccionar parámetros.

## Veredicto

`PHASE_O_POLICY_PARTIALLY_RECOVERABLE`

`E4_REQUIRES_OWNER_RISK_DEFINITION`

Phase O no contenía una política de salida autónoma. Su overlay Python congelaba entradas y modelos; las posiciones heredaban la política compuesta `AEGIS_TURBO` del bot TypeScript. Las reglas y fórmulas centrales son recuperables, pero no puede reproducirse toda la política sobre las trayectorias D1A sin decisiones nuevas.

## Respuestas obligatorias

1. **Stop inicial:** sí. Era un bracket de cierre total colocado en el exchange y recreado si faltaba.
2. **Unidad del stop:** ROE apalancado, `-0.40`. Para SHORT, `stop = entry * (1 + 0.40/leverage)`.
3. **Take profit fijo:** sí, `+0.50` ROE, también como bracket de cierre total.
4. **Activación de trailing:** sí.
5. **Condición:** `peak_roe >= 0.15`.
6. **Callback:** sí, como fallback cuando ATR no estaba disponible. Además existía una capa independiente de giveback en ExitEye/profit protection.
7. **Cálculo:** callback legacy `trigger_roe = peak_roe * (1 - 0.08)`; con ATR válido, el stop SHORT era `peak_price + 1.5 * ATR(14)`.
8. **Unidad del callback:** el `0.08` legacy era fracción del peak ROE, no 8 puntos de precio ni 8 puntos ROE. ExitEye usaba giveback absoluto de ROE y ATR usaba distancia de precio.
9. **Monotonicidad:** el stop solo podía mejorar. Para SHORT, el nuevo stop debía ser menor que el anterior; no podía ampliar riesgo.
10. **Break-even:** sí. Activación en `0.08` ROE; stop SHORT en `entry * (1 - 0.003)`.
11. **Cierres parciales:** no se encontró una regla Phase O. Los cierres y movimientos inspeccionados usaban la cantidad completa.
12. **Timeout:** existía `8h`, pero solo cerraba si `current_roe > 0.02`; no era una duración máxima dura.
13. **Conflictos intrabar:** el live dependía del orden real de eventos y órdenes del exchange. No se halló una regla OHLC determinista para conflictos dentro de la misma vela.
14. **SHORT:** sí. ROE, brackets, break-even, trailing y stop improvement tenían ramas SHORT explícitas.
15. **Estado operativo:** ejecutado. El ledger local registra una entrada SHORT real con brackets y los parámetros, y el stream de eventos registra break-even, activación de trailing y cierres ExitEye del stack AEGIS_TURBO.
16. **Versiones:** la ruta base de guardian nació antes de Phase O; Phase O SHORT live se vinculó en `22937cbd` (1 junio), amplió límites en `71742af` (2 junio) y recibió el fix de metadata `7f47abd` (7 junio). Los parámetros de salida no cambiaron entre las dos últimas revisiones.
17. **Parámetros predominantes:** stop `-0.40`, TP `0.50`, BE `0.08`, offset `0.003`, activación trailing `0.15`, callback legacy `0.08`, ATR `14x5m * 1.5`, duración condicional `8h`, profit giveback `0.05` y lock mínimo `0.01`.
18. **Parámetros incompatibles:** sí. Python conservó temporalmente `-15/25/15/8`, pero eran campos muertos. TypeScript también tenía defaults distintos, sobreescritos por YAML. El leverage real podía ser `8/10/15/20/25` según perfil/decisión.
19. **Último conjunto congelado anterior a D1A:** blob `fd651587d0931ad22ab8035f0f2f675bd136274e` en `7f47abd`, SHA-256 `4767cd424af055edb2c5b797a6ab47fa86391fc5b08e3f28a03c06730d5ef7b5`.
20. **Reconstrucción exacta:** no sin decisiones nuevas. Faltan bindings por entrada para leverage, ATR previo, señales/votos ExitEye, orden de eventos intrabar y una trayectoria de ocho horas.

## Evidencia temporal

Toda la autoridad principal es previa a D1A: commits TypeScript de mayo/junio, commit Python `4aeb603a` del 17 de julio a las 01:58 UTC, la auditoría arquitectónica `1841dbf9` del 17 de julio a las 17:28 UTC y logs del 1 de junio. La auditoría arquitectónica se clasifica `HISTORICALLY_DOCUMENTED`; corrobora, pero no sustituye, el código y los eventos operativos.

## Política recuperada

La política era una cadena compuesta:

1. Brackets iniciales completos: SL y TP por ROE apalancado.
2. ExitEye, evaluado antes del timeout y guardian, podía proteger beneficio o cerrar en ganancia según señales contemporáneas.
3. Timeout condicional tras ocho horas y solo en beneficio superior a 2% ROE.
4. ProfitGuardian: trailing activado por peak ROE, ATR cuando estaba disponible, callback de peak ROE como fallback, y break-even.
5. Todo movimiento de stop era monotónico y preservaba bracket cuando el exchange lo permitía.

## Unidad de riesgo

No se encontró una definición histórica nombrada `R`. Puede derivarse una equivalencia matemática, no una decisión científica:

`1R_price_fraction = abs(stop_roe) / leverage`

Con stop `-0.40`, a `20x` la distancia es `2%` de precio; a `10x`, `4%`. Por tanto, `-40% ROE` nunca significa `-40%` de movimiento del precio. Tampoco es automáticamente una fracción fija de equity, porque sizing y leverage variaban.

## Compatibilidad D1A

D1A contiene 1,292 entradas SHORT, entry price, timestamps y 12 velas OHLC de cinco minutos. Eso permite estudiar excursiones, pero no reproducir exactamente la política Phase O completa:

- no guarda leverage Phase O por entrada;
- no incluye las 14 velas pre-entrada requeridas por ATR dentro del artefacto de trayectoria;
- no guarda las acciones, votos y contadores futuros de ExitEye;
- OHLC no revela qué orden ejecutó primero dentro de una vela;
- termina a 60 minutos, siete horas antes del timeout histórico.

Una futura preregistración deberá definir esas decisiones de forma explícita. Esta auditoría no las adopta.

## Integridad y seguridad

No se ejecutó entrenamiento, replay, ECON, trailing simulation, D1A ni E3. No se consultó Binance, PM2 ni el lockbox. No se modificó código científico, TypeScript operativo, E3, D1A o configuraciones históricas.
