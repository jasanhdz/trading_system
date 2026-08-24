# AEGIS Range V2 - R2 TRAIN Implementation Defect

**Fecha:** 2026-08-24
**Estado:** `AEGIS_RANGE_R2_TRAIN_IMPLEMENTATION_DEFECT`
**Stop state:** `AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_R1_DEFECT`

## 1. Alcance

La auditoria previa obligatoria del backtester encontro un defecto reproducible
en `RangeEngineV1`. No se abrio TRAIN, no se leyo ninguna particion de mercado y
no se ejecuto ningun candidato. CALIBRATION, VALIDATION y HOLDOUT permanecieron
sellados.

No se modifico R0, R1, `RegimeEngineV2`, produccion ni el repositorio hijo. Este
documento registra el defecto y detiene R2 TRAIN sin aplicar un workaround.

## 2. Autoridad e integridad verificadas

```text
r1_implementation_manifest_sha256:
5b93160123aa7a8059e92ce256df135e7bdfe71ae055bc195ce6854a1c16ac81

r1_files_verified: 40/40

r2_source_gap_amendment_sha256:
3caab32ab68671100b89e8bf06b28c26dc472f6e260f23c2fe3c38aaa17a69a2

r2_source_gap_manifest_sha256:
2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52

gap_resolved_derived_manifest_sha256:
ccda624d7a683daad18888846cae53b8240f3a90bbc8838d6930ce301b290973

gap_resolved_logical_dataset_sha256:
8bcfb6ee88ece002e903774ba20e509536239c1f566617640fcd305b976fd2b4
```

## 3. Defecto confirmado

R0 exige que un episodio termine por el primer evento disponible cuando su
amplitud sale de `[min_range_amplitude_pct, 0.08]`:

- `AEGIS_RANGE_R0_PREREGISTRATION_REPORT.md`, seccion 6.2, puntos 4 y 6.2.
- El orden por vela expira observaciones, recalcula medianas, inserta los pivots
  disponibles y construye las parejas antes de publicar el snapshot causal.

En `RangeEngineV1.process` el chequeo de perdida de estructura ocurre antes de
insertar los pivots que quedan disponibles en el cierre actual:

```text
engine.py:138-143  chequea _active_pair() y puede terminar STRUCTURE_LOST
engine.py:145-147  inserta pivots disponibles y actualiza touches
engine.py:161-164  observa active_pair=None, pero solo retorna NOT_OPERABLE
```

Un pivot nuevo puede cambiar la mediana de un cluster activo y sacar la amplitud
del intervalo permitido. En ese caso el engine no termina el episodio en el
primer timestamp donde el evento esta disponible. Lo conserva activo, sin par
activo, hasta una vela posterior.

## 4. Reproduccion sintetica

La reproduccion usa solamente objetos sinteticos y el codigo R1 congelado:

```text
candidate.cluster_tolerance_atr: 0.20
candidate.min_range_amplitude_pct: 0.0125
support_before: 98.75
resistance: 100.00
amplitude_before: 0.012578616352201259

new_available_low_pivot: 98.90
support_after_median_update: 98.85
amplitude_after: 0.011566507417651554

RangeEngineV1 output:
episode_event: None
status: NOT_OPERABLE
episode_still_active: true
active_pair: None
```

La amplitud posterior es menor que `0.0125`, pero no se emite
`STRUCTURE_LOST` ni se resetea el episodio en esa vela.

## 5. Impacto cientifico

El desfase no es solamente descriptivo. En la vela siguiente, el engine evalua
primero breakout y expiracion sobre el snapshot previo antes de comprobar de
nuevo la estructura. Por ello puede cambiar:

- el timestamp y motivo de fin del episodio;
- el label `false_range`;
- el lineage de episodios y la construccion posterior del detector;
- conteos y metricas economicas por episodio.

Corregirlo en el runner R2 duplicaria o reordenaria logica estrategica y violaria
la obligacion de ejecutar `RangeEngineV1` sin cambios. Ignorarlo produciria
resultados TRAIN que no cumplen el contrato R0.

## 6. Disposicion

```text
TRAIN_ACCESS: false
CALIBRATION_ACCESS: false
VALIDATION_ACCESS: false
HOLDOUT_ACCESS: false
candidates_executed: 0/384
economic_metrics_computed: false
candidate_selected: false
production_modified: false
regime_engine_v2_modified: false
child_repository_modified: false
```

R2 TRAIN queda detenido en:

```text
AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_R1_DEFECT
```

Para reanudar se requiere una autorizacion explicita para corregir R1, agregar
una prueba de regresion que cubra la perdida de amplitud causada por un pivot
disponible en el cierre actual, regenerar el manifest R1 y repetir las
verificaciones de causalidad y determinismo antes de abrir TRAIN.
