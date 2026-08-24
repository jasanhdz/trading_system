# Aegis Range Strategy V1 - R2 Data Readiness Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY`

**Rama:** `work/entry-quality-evidence-20260726`

**HEAD base:** `b6907f30f8488d75658ccf5c3842e1a8d5691dbe`

## 1. Resultado

R2 Data Readiness verifico todos los archivos mensuales preregistrados y construyo
un dataset derivado no economico. La fase queda bloqueada porque los archivos
oficiales `markPriceKlines` omiten tres velas contractuales de funding por simbolo.
No se imputo mark price, no se uso una fuente sustituta y no se ejecuto estrategia.

```text
final_status: AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_SOURCE_INTEGRITY
ohlcv_archives: 341/341 VERIFIED
fundingRate_archives: 341/341 VERIFIED
markPriceKlines_archives: 341/341 VERIFIED
funding_events_mapped: 31075
funding_events_missing_mark_price: 33
```

El estado `VERIFIED` de un archivo significa que su path existe, el SHA-256 actual
coincide con ambos hashes del manifest, el byte size coincide, el CRC ZIP es valido
y contiene exactamente el CSV declarado. No implica continuidad interna del CSV.

## 2. Integridad y lineage

Se verificaron los dos manifests congelados de R0 antes de abrir los ZIP:

| Fuente | SHA-256 preregistrado | Cobertura |
|---|---|---:|
| M1A OHLCV futures 1m | `ce638d06f20f298f5be74c6b1c7648ad3df5abb1006b3861088302bd5da9f095` | 341/341 |
| M1B funding y mark price | `1cc559055937f3d2432f0559a6badda6865495fdfd26f52f3f02c0943836f92b` | 682/682 |

Tambien se reverificaron los 40 archivos congelados por
`r1_implementation_manifest.json`. Su SHA-256 sigue siendo
`5b93160123aa7a8059e92ce256df135e7bdfe71ae055bc195ce6854a1c16ac81`;
no existe drift R0/R1.

## 3. Dataset derivado

`RangeDataAdapter.aggregate_1m_to_5m` fue la unica implementacion usada para
agregar OHLCV. El constructor no importa ni invoca `RangeEngineV1`.

```text
source_interval: [2024-01-01T00:00:00Z, 2026-08-01T00:00:00Z)
symbols: 11
ohlcv_5m_rows: 2987424
ohlcv_segments: 11 total, 1 per symbol
ohlcv_integrity_gap_blocks: 0
funding_events_total_in_scope: 31108
funding_events_mapped: 31075
derived_artifacts: 682 deterministic csv.gz files
derived_artifact_size: 71 MiB
logical_sha256: 00ff950ae0c2605a941172a52b83a460f89b195093a2f9c444fd4050a27e24c1
```

El manifest local reproducible esta en
`sandbox/aegis_range_strategy_v1/artifacts/r2_data_readiness/derived_dataset_manifest.json`.
Su SHA-256 de archivo es
`55605c09e3f3de0d3f4d8b335beeac0eab4b0728a0f267512a6429ee8e2186b0`.
El directorio esta ignorado por Git y no constituye una particion abierta para
seleccion, evaluacion o analisis economico.

## 4. Funding y mark price

El campo oficial `calc_time` presenta jitter de procesamiento de milisegundos. El
derivado conserva `source_calc_time` y obtiene `funding_at` truncando solamente la
fraccion sub-minuto. Se exige `second=0`, se rechazan minutos duplicados y se fija:

```text
mark_open_time = funding_at - 60s
available_at = funding_at
mark_price = close de la vela exacta mark-price 1m
```

El evento en el limite inicial se excluye porque ningun trade iniciado dentro del
intervalo puede intersectarlo bajo `(entry_fill_at, exit_fill_at]`.

Los 11 simbolos presentan los mismos huecos internos en mark price:

| Periodo faltante | Minutos por simbolo | Intersecta funding |
|---|---:|---:|
| `2024-08-12T10:02Z..10:03Z` | 2 | 0 |
| `2026-06-29T00:00Z..23:59Z` | 1440 | 3 |

Por simbolo hay 2.825 eventos mapeados, 3 eventos sin mark price y 1.442 minutos
mark-price ausentes. Los tres eventos afectados son `2026-06-29T08:00Z`,
`2026-06-29T16:00Z` y `2026-06-30T00:00Z`, cuya vela requerida abre un minuto
antes. El total es 33 eventos faltantes.

No se permite completar este hueco con la vela que abre en funding, funding cero,
interpolacion, una API live ni un archivo diario no incluido en el manifest R0.

## 5. Particiones selladas

Los guards nuevos fallan cerrados. Sus defaults verificados son:

```text
TRAIN_ACCESS=false
CALIBRATION_ACCESS=false
VALIDATION_ACCESS=false
HOLDOUT_ACCESS=false
```

Cada acceso requiere el opt-in exacto de su particion; una particion desconocida
tambien falla cerrada. La construccion readiness solo asigna lineage temporal y no
ejecuta candidatos ni inspecciona resultados por particion.

## 6. Parity TS/Python

El bridge corre fuera del repositorio TypeScript y solo importa read-only el motor
congelado. Verifica los tres hashes aprobados antes de cada evaluacion. Cada caso
usa exactamente 160 velas 5m sinteticas, omite `market` y compara el objeto TS
completo contra golden, incluyendo strings e integers exactos y numeros publicados
a seis decimales. El ATR Wilder raw cruza JSON con igualdad binary64, sin epsilon.

```text
frozen_typescript_head: bb034431e0ce05c8e0f978453c46dcff6efb981c
synthetic_parity_cases: 3/3 PASS
golden_fixture_sha256: ad6983a6da415d9b0cea6565616a24ecc38fe51b435ad6df82bd56f0c1374f92
typescript_frozen_tests: 17/17 PASS
python_sandbox_tests: 67/67 PASS
```

El checkout hijo permanece limpio y no fue modificado.

## 7. Limites de fase

No se ejecuto backtest, TRAIN, CALIBRATION, VALIDATION, HOLDOUT, grid, seleccion de
candidato, PnL, metrica economica, E4, shadow, live ni codigo de produccion. No se
crearon runners prohibidos y no se modifico ningun archivo R0/R1 congelado.

## 8. Resolucion requerida

R2 Data Readiness solo puede volver a ejecutarse si ocurre una de estas condiciones:

1. Binance publica archivos mensuales corregidos y una nueva version autorizada del manifest los fija por SHA-256.
2. Una enmienda de autoridad permite explicitamente archivos diarios oficiales para gap-fill y define su manifest, checksums y precedence.

Hasta entonces no se emite `AEGIS_RANGE_R2_DATA_READINESS_READY_FOR_REVIEW` y toda
fase posterior permanece bloqueada.
