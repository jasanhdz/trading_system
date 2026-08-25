# Aegis Range Strategy V1 - Lineage Hash Reconciliation Report

**Fecha:** 2026-08-24
**Estado:** `AEGIS_RANGE_LINEAGE_HASH_RECONCILIATION_PASS`
**Commit reconciliado:** `5ecfa7009f6cfe3f700cdc1607aacb15ce245614`

## 1. Resultado

Los bytes del working tree previos a este reporte coinciden exactamente con los
bytes committed en `5ecfa70` para los ocho artefactos requeridos. No existe una
referencia incorrecta dentro de los manifests o reportes committed y no se
modifico codigo, tests, datos ni manifests durante la reconciliacion.

Los hashes distintos publicados en un resumen externo fueron valores stale de
serializaciones intermedias anteriores al commit:

```text
stale R1 manifest summary value:
69321f6b0e9cfe7a593db23c63f6657c09b83428d8ea756754427de809941c41

committed R1 manifest FILE_SHA256:
2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c

stale revalidation manifest summary value:
a6e38ca57137fdc87bbcba8097bf4d4d0af139c5bca0eb11ac52a3c7e08c08c7

committed revalidation manifest FILE_SHA256:
9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1
```

Los valores stale no aparecen en los artefactos committed. Su payload
intermedio no fue versionado y por tanto no se presenta como evidencia
reproducible. La fuente de verdad es el SHA-256 de los bytes extraidos
literalmente mediante `git show 5ecfa70:<path>`.

## 2. Clases de hash

```text
FILE_SHA256
  SHA-256 de los bytes exactos de un archivo.

R1_MANIFEST_FILE_SHA256
  FILE_SHA256 de r1_implementation_manifest.json.

GIT_BLOB_ID
  Identificador del objeto Git. No es FILE_SHA256 y no lo sustituye.

LOGICAL_SHA256
  Hash de un payload canonico cuando el esquema lo define.

DATASET_LOGICAL_SHA256
  LOGICAL_SHA256 del dataset derivado; no es el hash fisico de su manifest.
```

Se conserva la politica anti-autorreferencia: un manifest no contiene su propio
`FILE_SHA256`; ese valor se calcula read-only despues de escribirlo.

## 3. Reconciliacion de archivos

| Artifact | FILE_SHA256 committed | FILE_SHA256 working tree | Logical SHA aplicable | Status | Valores reportados previamente | Clasificacion |
|---|---|---|---|---|---|---|
| `r1_implementation_manifest.json` | `2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c` | `2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c` | N/A | MATCH | `69321f6b...`, `2a55d0a5...` | `69321f6b...` stale pre-final summary; committed file authoritative |
| `AEGIS_RANGE_R1_DEFECT_CORRECTION_REPORT.md` | `7bd0a6c00def02922e192c9edaffa6a930abdb87209e0137a05b4eec21a57c1d` | `7bd0a6c00def02922e192c9edaffa6a930abdb87209e0137a05b4eec21a57c1d` | N/A | MATCH | manifest binding `7bd0a6c0...` | FILE_SHA256 consistent |
| `r2_post_r1_fix_revalidation_manifest.json` | `9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1` | `9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1` | dataset `587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8` | MATCH | `a6e38ca5...`, `9fa69860...` | `a6e38ca5...` stale pre-final summary; FILE and dataset logical hashes separated |
| `AEGIS_RANGE_R2_POST_R1_FIX_REVALIDATION_REPORT.md` | `ea546f2bda86e8d3b218bf618da0e3b18b8bbdfa059c032bc49e90f4057a3220` | `ea546f2bda86e8d3b218bf618da0e3b18b8bbdfa059c032bc49e90f4057a3220` | dataset `587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8` | MATCH | R1 `2a55d0a5...`, revalidation `9fa69860...` | Internal references correct |
| `r2_data_readiness_manifest.json` | `442f842999e9a2d9e881932d5d1d21297e8f2a358da5ffca7e111487c21111d6` | `442f842999e9a2d9e881932d5d1d21297e8f2a358da5ffca7e111487c21111d6` | dataset `00ff950ae0c2605a941172a52b83a460f89b195093a2f9c444fd4050a27e24c1` | MATCH | historical manifest/file and logical hashes | Historical pre-gap artifact preserved |
| `r2_source_gap_manifest.json` | `2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52` | `2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52` | N/A | MATCH | `2f0c09d0...` | FILE_SHA256 consistent |
| `AEGIS_RANGE_R2_SOURCE_GAP_AMENDMENT.md` | `3caab32ab68671100b89e8bf06b28c26dc472f6e260f23c2fe3c38aaa17a69a2` | `3caab32ab68671100b89e8bf06b28c26dc472f6e260f23c2fe3c38aaa17a69a2` | N/A | MATCH | `3caab32a...` | FILE_SHA256 consistent |
| `AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_REPORT.md` | `b5e8c2aba26229c2bba7156d4ff2bab4c0525e45e2720cfeccec79760f3c78ba` | `b5e8c2aba26229c2bba7156d4ff2bab4c0525e45e2720cfeccec79760f3c78ba` | historical dataset `8bcfb6ee88ece002e903774ba20e509536239c1f566617640fcd305b976fd2b4` | MATCH | historical gap-resolved logical hash | Historical lineage preserved; superseded by explicit post-fix binding |

## 4. Git blob IDs

Estos IDs se registran solo como identidad Git y no se usaron como SHA-256 de
archivo:

| Artifact | GIT_BLOB_ID at `5ecfa70` |
|---|---|
| `r1_implementation_manifest.json` | `f6c535b6468548485961ffef8eaa1be1bcce0cba` |
| `AEGIS_RANGE_R1_DEFECT_CORRECTION_REPORT.md` | `fc44c80c9f33beb766cbeb50c3d6ffa5de9c48a9` |
| `r2_post_r1_fix_revalidation_manifest.json` | `ee937dac2f1494f9541657bd1a9c33381b1b8502` |
| `AEGIS_RANGE_R2_POST_R1_FIX_REVALIDATION_REPORT.md` | `35cabd187f5cedd877683d1c760414e611229b83` |
| `r2_data_readiness_manifest.json` | `27a8260fa91555231aa7ebbb8370959f2f126009` |
| `r2_source_gap_manifest.json` | `33d7535c2efa2962fefcc9203414a38dfa505bc8` |
| `AEGIS_RANGE_R2_SOURCE_GAP_AMENDMENT.md` | `e6589e42ecc79798155bc880127ede2bbfba6943` |
| `AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_REPORT.md` | `0477cfdb3aff225a60c6ca70594e4dafd99430da` |

## 5. R1 y estrategia

```text
R1 manifest files: 42/42 MATCH
R1_MANIFEST_FILE_SHA256:
2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c

detector.py:
1187612c1be077c6bde06715c28a60fef52b89d514bd86ef6572fda16205faf1

engine.py:
632f49e315d1542ca2d77095a768c2ca7505cf50a32f344a295d42448cd50006

test_engine_master.py:
2435d3d62114f39dc3f7eafb57f8ce8cc335c54fd5db0f42344580b10f01f570
```

La unica correccion de estrategia sigue siendo la validacion final del active
pair despues de las mutaciones causales de levels. No existe drift adicional de
estrategia.

## 6. Fuentes y datos

```text
OHLCV monthly: 341/341
fundingRate monthly: 341/341
markPrice monthly: 341/341
DAILY gap archives: 22/22
MONTHLY/DAILY overlap: 15818 compared, 15818 exact, 0 mismatch
funding events: 31108 total, 31108 mapped, 0 missing
remaining non-funding mark gaps: 22
replacement policy: MONTHLY_PRIMARY_DAILY_GAP_FILL_V1
economic data content: UNCHANGED
```

Los builds post-fix A/B son byte-identicos. Sus 341 artefactos funding/mark
coinciden en path, rows, source hashes, artifact hashes y bytes con el build
economico anterior. El cambio del dataset logical hash historico
`8bcfb6ee...` al post-fix `587476db...` corresponde solo al binding del manifest
R1 corregido:

```text
LINEAGE_ONLY_HASH_CHANGE
```

## 7. Tests pre-TRAIN

```text
Python full sandbox: 79/79 PASS
TypeScript frozen: 17/17 PASS
TS/Python parity: PASS
ATR binary64 parity: PASS
no-lookahead master: PASS
post-mutation no-lookahead: PASS
determinism master: PASS
post-mutation determinism: PASS
R1 defect regression: PASS
candidate grid: 384 PASS
thesis golden: PASS
```

## 8. Gate y frontera

Antes de crear este reporte, el working tree coincidia con `5ecfa70` salvo el
gitlink preexistente fuera de alcance. Este reporte es el unico cambio documental
de reconciliacion autorizado.

```text
TRAIN opened: false
candidates executed: 0/384
CALIBRATION opened: false
VALIDATION opened: false
HOLDOUT opened: false
economic metrics observed: false
strategy code modified by reconciliation: false
child repository modified: false
```

El gate PHASE A termina:

```text
AEGIS_RANGE_LINEAGE_HASH_RECONCILIATION_PASS
```
