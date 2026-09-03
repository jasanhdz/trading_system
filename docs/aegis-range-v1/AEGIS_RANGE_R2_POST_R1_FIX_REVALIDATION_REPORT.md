# Aegis Range Strategy V1 - R2 Post-R1-Fix Revalidation Report

**Fecha:** 2026-08-24
**Estado:** `AEGIS_RANGE_R2_POST_R1_FIX_REVALIDATION_READY_FOR_REVIEW`
**Clasificacion:** `LINEAGE_ONLY_HASH_CHANGE`

## 1. Alcance

Esta revalidacion responde exclusivamente al cambio autorizado de R1
`R1_ACTIVE_PAIR_POST_MUTATION_VALIDITY`. Verifica otra vez fuentes, contenido de
datos y reproducibilidad de los derivados sin ejecutar estrategia ni abrir una
particion.

Los manifests R2 historicos permanecen intactos como evidencia. El nuevo
artefacto de revalidacion los supersede para el lineage posterior a la
correccion, sin fingir que sus hashes R1 originales cambiaron.

## 2. Cambio de lineage

```text
old R1 manifest SHA-256:
5b93160123aa7a8059e92ce256df135e7bdfe71ae055bc195ce6854a1c16ac81

new R1 manifest SHA-256:
2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c

R1 files verified: 42/42
R1 mismatches: 0
```

Artefactos historicos preservados:

```text
r2_data_readiness_manifest.json:
442f842999e9a2d9e881932d5d1d21297e8f2a358da5ffca7e111487c21111d6

r2_source_gap_manifest.json:
2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52

old gap-resolved manifest:
ccda624d7a683daad18888846cae53b8240f3a90bbc8838d6930ce301b290973

old gap-resolved logical hash:
8bcfb6ee88ece002e903774ba20e509536239c1f566617640fcd305b976fd2b4
```

## 3. Integridad de fuentes

```text
OHLCV monthly archives:       341/341 VERIFIED
fundingRate monthly archives: 341/341 VERIFIED
markPrice monthly archives:   341/341 VERIFIED
DAILY gap archives:             22/22 VERIFIED

MONTHLY/DAILY overlap compared: 15818
MONTHLY/DAILY exact matches:    15818
MONTHLY/DAILY mismatches:           0

replacement_policy: MONTHLY_PRIMARY_DAILY_GAP_FILL_V1
source_changed: false
source_gap_policy_changed: false
```

El source-gap manifest regenerado es byte-identico al historico:

```text
2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52
```

## 4. Contenido de datos

```text
OHLCV artifacts: 341
OHLCV rows: 2987424
funding/mark artifacts: 341
funding rows: 31108

funding events total: 31108
funding events mapped: 31108
funding events missing mark price: 0
remaining non-funding mark gaps: 22

data_changed: false
economic_rows_changed: false
funding_mapping_changed: false
```

El manifest base OHLCV/funding historico no se reescribio:

```text
manifest SHA-256:
55605c09e3f3de0d3f4d8b335beeac0eab4b0728a0f267512a6429ee8e2186b0

logical SHA-256:
00ff950ae0c2605a941172a52b83a460f89b195093a2f9c444fd4050a27e24c1
```

## 5. Builds independientes

Se generaron dos builds independientes con el nuevo manifest R1:

```text
build A manifest SHA-256:
2bd850cbb88123f870dd3f26a5865a6749aa30025cc512997ac44f44ac17a1cc

build B manifest SHA-256:
2bd850cbb88123f870dd3f26a5865a6749aa30025cc512997ac44f44ac17a1cc

new logical SHA-256:
587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8
```

Para los 341 artefactos funding/mark se verifico en build A, build B y el build
historico:

```text
same artifact paths: true
same artifact rows: true
same artifact SHA-256: true
same artifact bytes: true
same source SHA-256 lineage: true
same symbol metrics: true
```

El cambio de logical hash de `8bcfb6ee...` a `587476db...` procede del binding
del manifest R1 corregido. No existe drift de fuentes, filas o contenido
economico. Por eso se clasifica `LINEAGE_ONLY_HASH_CHANGE`.

## 6. Evidencia y manifest

```text
daily audit SHA-256:
fc0c4c464d7c90b9023b5417657e5681ebfac15643419f05b8af38d460781760

r2_post_r1_fix_revalidation_manifest.json SHA-256:
9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1
```

Los builds y el audit viven bajo:

```text
sandbox/aegis_range_strategy_v1/artifacts/r2_post_r1_fix_revalidation/
```

## 7. Frontera experimental

```text
partition access flags enabled: false
RangeEngineV1 executed on market data: false
R2 strategy executed: false
TRAIN opened: false
candidates executed: 0/384
CALIBRATION opened: false
VALIDATION opened: false
HOLDOUT opened: false
economic metrics computed: false
candidate selection executed: false
production modified: false
E4 modified: false
RegimeEngineV2 modified: false
child repository modified: false
```

Esta revalidacion no reautoriza TRAIN. La ejecucion R2 permanece detenida hasta
una autorizacion posterior explicita.
