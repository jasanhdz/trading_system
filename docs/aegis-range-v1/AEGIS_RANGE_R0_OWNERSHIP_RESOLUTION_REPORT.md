# Aegis Range Strategy V1 - R0 Ownership Resolution Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R0_OWNERSHIP_RESOLVED_READY_FOR_REVIEW`

**Alcance:** ownership y migracion documental exclusivamente.

## 1. Repositorios auditados

### Parent

```text
path: /home/jasan/Develop/trading_system
branch: work/entry-quality-evidence-20260726
HEAD before migration: cfbde9bd47e9f29dfda35e9e9f05d57ca148f556
```

### Child

```text
path: /home/jasan/Develop/trading_system/binance-futures-bot-ts
branch: work/entry-quality-evidence-20260726
HEAD before migration: 970b26c0c49b8ba7d0da7f90c898ddf30e96995a
ownership redirect publication commit: 0a41b100ae620c7c478ecf4e05987e08f08964dc
```

## 2. Estado Git antes de migrar

Parent:

```text
 M binance-futures-bot-ts
```

La modificacion era exclusivamente el gitlink registrado en `814a302...` frente
al HEAD observado `970b26c...` del repo independiente.

Child:

```text
(clean)
```

## 3. Condiciones de seguridad previas

La migracion continuo porque se verifico:

```text
train_executed: false
calibration_executed: false
validation_opened: false
holdout_opened: false
pre_validation_spec_frozen: false
r1_executed: false
r2_executed: false
```

No existen `RangeDetectorV1`, `RangeLevelsV1`, `RangeSafetyV1`,
`RangeSignalV1` o `RangeBreakoutV1` en ninguno de los repositorios. No existe
backtester Range. Durante esta tarea no se leyeron resultados, features, senales,
episodios ni outcomes de TRAIN/CALIBRATION/VALIDATION/HOLDOUT.

## 4. Paths originales y canonicos

| Artefacto | Path original child | Path canonico parent |
|---|---|---|
| Plan | `docs/AEGIS_RANGE_STRATEGY_V1_PLAN.md` | `docs/aegis-range-v1/AEGIS_RANGE_STRATEGY_V1_PLAN.md` |
| R0 report | `docs/aegis-range-v1/AEGIS_RANGE_R0_PREREGISTRATION_REPORT.md` | `docs/aegis-range-v1/AEGIS_RANGE_R0_PREREGISTRATION_REPORT.md` |
| Source manifest | `docs/aegis-range-v1/r0_source_manifest.json` | `docs/aegis-range-v1/r0_source_manifest.json` |
| Split manifest | `docs/aegis-range-v1/r0_split_manifest.json` | `docs/aegis-range-v1/r0_split_manifest.json` |
| Production diff | `docs/aegis-range-v1/r0_production_diff.json` | `docs/aegis-range-v1/r0_production_diff.json` |
| Artifact manifest | `docs/aegis-range-v1/r0_artifact_manifest.json` | `docs/aegis-range-v1/r0_artifact_manifest.json` |

Los paths de la tabla son relativos a sus respectivos repositorios. El root
canonico absoluto es:

```text
/home/jasan/Develop/trading_system/docs/aegis-range-v1/
```

## 5. Hashes before/after

| Artefacto | SHA-256 before | SHA-256 after | Resultado |
|---|---|---|---|
| Plan | `2b6976fa9a90949084640991b5643da407a55cf97f3a8d435de8ba1cf6fc7288` | `2b6976fa9a90949084640991b5643da407a55cf97f3a8d435de8ba1cf6fc7288` | byte-identical |
| R0 report | `6cf7faef4590f80f1c9a224957a3064fa6999f00661e440e9b278521d06e6342` | `6cf7faef4590f80f1c9a224957a3064fa6999f00661e440e9b278521d06e6342` | byte-identical |
| Source manifest | `39cd5b8371ef4d193fcce22e6d6392ceaff8b802b52b4589d0c27cfa583a7704` | `39cd5b8371ef4d193fcce22e6d6392ceaff8b802b52b4589d0c27cfa583a7704` | byte-identical |
| Split manifest | `a67766d8ab446c657260550d37c55589d94cc11afba064ae3f043db803868c03` | `a67766d8ab446c657260550d37c55589d94cc11afba064ae3f043db803868c03` | byte-identical |
| Production diff | `cb8d5ddd3e4b84eff229f096387e199731c5bed32278415beadd33572c5b9c55` | `cb8d5ddd3e4b84eff229f096387e199731c5bed32278415beadd33572c5b9c55` | byte-identical |
| Artifact manifest | `424eb5bbbf320f24ddc6d32dd88f8b16a20fb7c963227967eaf9c457d09fd8cd` | `424eb5bbbf320f24ddc6d32dd88f8b16a20fb7c963227967eaf9c457d09fd8cd` | byte-identical |

Los artifacts historicos no se editaron. Sus referencias originales permanecen
como lineage historico y la nueva capa de ownership las interpreta sin
reescribirlas.

## 6. Nuevos documentos de ownership

- `AEGIS_RANGE_R0_OWNERSHIP_AMENDMENT.md`
  - SHA-256: `f1d2b61d90019e91a8dfaccf704828b220744c0a1a13de38cc0f08dfe238a603`
- `r0_ownership_manifest.json`
  - SHA-256: `d8bee965c3e0094779e20f894556298421bb508019719de80bf2ebf4ee3764a2`
- `README.md` canonico
  - SHA-256: `d6fd0ae1f3f6bb74e1dc56d2ae69cfc98ae54c305d7126c755f7ce6e8fadfba8`
- README de redireccion child
  - SHA-256: `333e48c21b339a7e27dfb60c0615a8071767dcf38fbad5aeae8833da44e50c98`

La enmienda tiene precedencia exclusivamente sobre ownership y ubicacion futura
de fases. El ownership manifest registra paths, HEADs, hashes y estado
experimental.

## 7. Archivos trasladados y deprecados

- Se trasladaron al padre el plan y los cinco artefactos R0, byte-identical.
- Se eliminaron sus working copies actuales del hijo; siguen preservadas en los
  commits historicos `bb03443` y `970b26c`.
- El hijo conserva solo `docs/aegis-range-v1/README.md`, sin autoridad cientifica.
- No existen dos copias actuales editables/canonicas.
- No se creo ningun path ni archivo R1/R2.

## 8. Ownership final

```text
CANONICAL SCIENTIFIC REPOSITORY:
/home/jasan/Develop/trading_system

CANONICAL R1/R2 IMPLEMENTATION, SI SE AUTORIZA DESPUES:
/home/jasan/Develop/trading_system/sandbox/aegis_range_strategy_v1/

RUNTIME INTEGRATION REPOSITORY, R4+:
/home/jasan/Develop/trading_system/binance-futures-bot-ts
```

El path R1/R2 es una reserva documental y no fue creado. Un port runtime futuro
debera demostrar parity contra el artefacto cientifico congelado.

## 9. Contrato cientifico sin cambios

La evidencia objetiva es la igualdad de los seis hashes before/after. No cambio:

- timeframe, causalidad ni pivots;
- clustering, touches, episodes o signals;
- TP, SL, breakout, max hold o fills;
- reentry, costos o funding;
- grids, search space, splits o purge;
- hipotesis, Holm-Bonferroni, bootstrap o gate;
- politica E4, datasets o manifests upstream.

La enmienda no genera un nuevo preregistro cientifico: corrige ownership antes de
cualquier ejecucion experimental.

## 10. Sellado experimental

- TRAIN no fue ejecutado ni leido.
- CALIBRATION no fue ejecutado y permanece sellado.
- VALIDATION no fue abierta y permanece sellada.
- HOLDOUT no fue abierto y permanece sellado.
- `PRE_VALIDATION_SPEC_FROZEN=false`.
- R1/R2 no fueron ejecutados ni autorizados por esta migracion.

## 11. Produccion intacta

| Archivo child | SHA-256 final |
|---|---|
| `regime_config.live.yaml` | `7b841c07bc8488827201a443a8682f676e08fc7863df903baa2527d200360858` |
| `src/domain/services/AegisRegimeGuard.ts` | `86211d956b290931bd92475886645ae9cd666ca7898b9147b56f3a3150efd755` |
| `src/app/services/TradingService.ts` | `db63d7996c5fc36c206b75f462c2e875602ffc6f8fc74a09fb26e3c1a8af012c` |
| `src/domain/services/regime-v2/RegimeEngineV2.ts` | `3726e28badfdba5acc81d87ccd3202fc43310a04d4b3cff2597f38acb2913134` |
| `src/domain/services/regime-v2/RegimeEngineV2.types.ts` | `3b3972153f7c977d50ec864a5d8a4c4b3d8d2e73822453eaed1a25391211d10c` |
| `src/domain/services/regime-v2/RegimeEngineV2.test.ts` | `80aa2619efdcb74fa8722f79ce62a01a4028bb213856f3a5b7fc0ea32e091cf1` |

Los hashes coinciden con baseline R0. No hubo cambios de PM2, exchange, orders,
live, E4, FeatureBridge, guards, execution o produccion.

## 12. Gitlink

```text
parent recorded gitlink: 814a302885e1d07bfd27404ebb5e69a30acebcc5
observed child HEAD:      970b26c0c49b8ba7d0da7f90c898ddf30e96995a
.gitmodules: absent
status: GITLINK_MISMATCH_RECORDED_NOT_RESOLVED
```

No se agrego submodule, no se actualizo el gitlink y no se reescribio historia.

## 13. Blocker R2 preservado

```text
fundingRate:      237/341 materializados
markPriceKlines:  237/341 materializados
policy: BLOCK_R2_UNTIL_DOWNLOADED_AND_VERIFIED
```

No se descargo ningun archivo.

## 14. Verificacion Git final

`git diff --check` completo sin output en ambos repositorios.

Parent `git status --short` observado:

```text
 M binance-futures-bot-ts
?? docs/aegis-range-v1/AEGIS_RANGE_R0_OWNERSHIP_AMENDMENT.md
?? docs/aegis-range-v1/AEGIS_RANGE_R0_OWNERSHIP_RESOLUTION_REPORT.md
?? docs/aegis-range-v1/AEGIS_RANGE_R0_PREREGISTRATION_REPORT.md
?? docs/aegis-range-v1/AEGIS_RANGE_STRATEGY_V1_PLAN.md
?? docs/aegis-range-v1/README.md
?? docs/aegis-range-v1/r0_artifact_manifest.json
?? docs/aegis-range-v1/r0_ownership_manifest.json
?? docs/aegis-range-v1/r0_production_diff.json
?? docs/aegis-range-v1/r0_source_manifest.json
?? docs/aegis-range-v1/r0_split_manifest.json
```

Child `git status --short` observado:

```text
 D docs/AEGIS_RANGE_STRATEGY_V1_PLAN.md
 D docs/aegis-range-v1/AEGIS_RANGE_R0_PREREGISTRATION_REPORT.md
 D docs/aegis-range-v1/r0_artifact_manifest.json
 D docs/aegis-range-v1/r0_production_diff.json
 D docs/aegis-range-v1/r0_source_manifest.json
 D docs/aegis-range-v1/r0_split_manifest.json
?? docs/aegis-range-v1/README.md
```

La resolucion se preparo y verifico sin commit. Su publicacion se hizo despues,
unicamente tras una instruccion explicita separada del usuario.

## 15. Resultado

La autoridad cientifica canonica queda resuelta en `trading_system` sin alterar
el contrato R0 ni abrir fases posteriores. El entregable vuelve a revision
externa; R1 continua sin autorizacion.

`AEGIS_RANGE_R0_OWNERSHIP_RESOLVED_READY_FOR_REVIEW`
