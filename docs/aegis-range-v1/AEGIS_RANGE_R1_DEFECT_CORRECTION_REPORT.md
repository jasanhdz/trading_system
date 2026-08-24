# Aegis Range Strategy V1 - R1 Defect Correction Report

**Fecha:** 2026-08-24
**Estado:** `AEGIS_RANGE_R1_DEFECT_CORRECTION_READY_FOR_REVIEW`
**Autorizacion:** `AEGIS_RANGE_R1_DEFECT_CORRECTION_V1`

## 1. Resultado

Se corrigio exclusivamente `R1_ACTIVE_PAIR_POST_MUTATION_VALIDITY`. La
correccion elimina el retraso de una vela al invalidar un episodio cuya mediana
y amplitud cambian por un pivot que queda disponible en el cierre actual.

No se cambio ningun threshold, formula, grid, costo, pivot, touch, ranking,

## 2. Defecto y causa

R0 seccion 6.2 exige terminar el episodio por el primer evento disponible cuando
un cluster pierde minimos estructurales o la amplitud sale de
`[min_range_amplitude_pct, 0.08]`. El orden causal R0 secciones 4.5 y 5 exige
aplicar en el cierre los pivots que alcanzan `available_at`.

El codigo anterior comprobaba el active pair despues de expirar estado viejo,
pero no volvia a comprobarlo despues de insertar pivots, actualizar touches y
construir pairs. Un pivot nuevo podia cambiar la mediana, invalidar la amplitud y
dejar el episodio activo hasta la vela siguiente.

## 3. Correccion minima

`RangeDetectorV1.active_pair_invalid_reason` clasifica exclusivamente las dos
condiciones R0 ya existentes:

```text
STRUCTURE_LOST
  cluster ausente o con menos de dos pivots/touches activos

AMPLITUDE_OUT_OF_RANGE
  clusters estructuralmente validos, pero amplitude fuera de
  [min_range_amplitude_pct, 0.08]
```

`RangeEngineV1.process` conserva el chequeo posterior a expiracion y agrega una
validacion final despues de insertar pivots, contar/rearmar touches, construir
pairs y resolver `PAIR_REPLACED`. Una invalidacion final termina y resetea el
episodio en el mismo `decision_at`, retorna `NOT_OPERABLE` y omite snapshot,
`range_id`, rejection, signal y `PENDING_ENTRY`.

`PAIR_REPLACED` conserva precedencia cuando el winner tiene IDs distintos. No se
permite same-close rebirth.

## 4. Archivos R1 modificados

```text
sandbox/aegis_range_strategy_v1/src/aegis_range_v1/detector.py
old: cf61ccd7cd7a4e53e3bc13de557a041294578fd8d185e76a1a41f837f60b9fa1
new: 1187612c1be077c6bde06715c28a60fef52b89d514bd86ef6572fda16205faf1

sandbox/aegis_range_strategy_v1/src/aegis_range_v1/engine.py
old: a817a83d3847a91b7a40a76bfc0816a402be217662bebbd9c4cd715303a09f8d
new: 632f49e315d1542ca2d77095a768c2ca7505cf50a32f344a295d42448cd50006

sandbox/aegis_range_strategy_v1/tests/test_engine_master.py
old: c0934f228ccd3d82f104efc488fe87039b46ee4d1505fe16b570460a1950cd95
new: 2435d3d62114f39dc3f7eafb57f8ce8cc335c54fd5db0f42344580b10f01f570
```

## 5. Evidencia de regresion

Caso principal documentado:

```text
REGRESSION_R1_ACTIVE_PAIR_AMPLITUDE_INVALIDATED_BY_NEW_PIVOT: PASS
```

La suite sintetica cubre:

- pivot nuevo reduce amplitud por debajo del minimo;
- pivot nuevo aumenta amplitud por encima de `0.08`;
- expiracion deja menos de dos pivots;
- expiracion deja menos de dos touches;
- cambio de mediana valido conserva episodio e ID;
- pair distinto conserva `PAIR_REPLACED`;
- invalidacion no renace, no publica `range_id`, no evalua rejection/signal y no
  conserva pending entry;
- el episodio viejo no existe en T+1;
- terminacion en T estable ante futuro arbitrario;
- outputs repetidos deterministas.

El caso principal falla contra el codigo anterior porque devuelve
`episode_event=None`, `NOT_OPERABLE` y conserva el episodio activo.

## 6. Verificaciones

```text
python_tests: 79/79 PASS
typescript_frozen_tests: 17/17 PASS
typescript_python_parity: PASS
atr_binary64_parity: PASS
no_lookahead_master: PASS
post_mutation_no_lookahead: PASS
determinism_master: PASS
post_mutation_determinism: PASS
candidate_grid: 384 PASS
python_compileall: PASS
```

Golden thesis inalterado:

```text
dbc38ff08459e9673c96cfa72d675c4c6a817871a999604f13177e8ba2a745cb
```

## 7. Lineage R1

```text
previous_r1_manifest_sha256:
5b93160123aa7a8059e92ce256df135e7bdfe71ae055bc195ce6854a1c16ac81

new_r1_manifest_sha256:
reported_by_final_read_only_verification

defect_fixed:
R1_ACTIVE_PAIR_POST_MUTATION_VALIDITY
```

El manifest corregido supersede el manifest R1 anterior sin borrar la evidencia
historica. Su hash propio se excluye del payload para evitar autorreferencia y se
reporta despues de serializarlo.

## 8. Frontera experimental

```text
TRAIN_ACCESS: false
CALIBRATION_ACCESS: false
VALIDATION_ACCESS: false
HOLDOUT_ACCESS: false
TRAIN opened: false
candidates executed: 0/384
R2 strategy executed: false
economic metrics computed: false
production modified: false
E4 modified: false
RegimeEngineV2 modified: false
child repository modified: false
```

La revalidacion posterior se limita a checksums, archivos fuente y derivados de
datos. No ejecuta `RangeEngineV1` sobre mercado.
