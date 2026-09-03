# Aegis Range Strategy V1 - R2 Source Gap Resolution Report

**Fecha:** 2026-08-24

**Estado:** `AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_READY_FOR_REVIEW`

**Rama:** `work/entry-quality-evidence-20260726`

**HEAD base:** `d6bed0913ac0d3b6285f6c93b43742b13f882259`

## 1. Resultado

Binance Vision publica los 22 ZIP DAILY oficiales investigados. Los DAILY de
`2026-06-29` recuperan los 1.440 minutos por simbolo y resuelven los 33 mark
prices contractuales de funding. Los DAILY de `2024-08-12` reproducen el gap de
dos minutos del MONTHLY, por lo que permanecen 22 minutos no-funding ausentes.

```text
replacement_policy: MONTHLY_PRIMARY_DAILY_GAP_FILL_V1
monthly_missing_minutes: 15862
daily_recovered_minutes: 15840
remaining_missing_minutes: 22
funding_events_total: 31108
funding_events_mapped: 31108
funding_events_missing_mark_price: 0
```

No hubo imputacion, interpolacion, reemplazo de una fila MONTHLY, API live, spot,
otra exchange ni cambio de funding.

## 2. Auditoria DAILY por archivo

`Symbol` queda ligado por URL, path, filename y member CSV oficiales. `Timeframe`
queda ligado a `1m`; todas las filas pertenecen a la fecha UTC indicada y estan
estrictamente ordenadas.

| Symbol | Date | Exists | Minutes | SHA-256 | CRC | CRC ok | Coverage start | Coverage end | Duplicates | Invalid rows |
|---|---|---:|---:|---|---|---:|---|---|---:|---:|
| BTCUSDT | 2024-08-12 | yes | 1438 | `139f800b758d844744b51a2d2fa1e0391447091faba2cccedb17028acb34d8cc` | `8142ff63` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| BTCUSDT | 2026-06-29 | yes | 1440 | `200e2ef30a86e9f91c011dae30d864f73c8480fddecc73ea227958596f6cd11c` | `2dc55886` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| ETHUSDT | 2024-08-12 | yes | 1438 | `11a3640b1e90491a5491c8abe6182e4b4cb6f6ce5dde3c7ba34a0e5ccbda6552` | `28327db6` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| ETHUSDT | 2026-06-29 | yes | 1440 | `cc07f9a3f5580a443512096c2de20d3d59a52be76f07668c87a6418bf762f07b` | `41bdd601` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| SOLUSDT | 2024-08-12 | yes | 1438 | `6dd1cb8302a8b4f001f20faa13db417010ec41e073dd720c4ed19529e70a09fe` | `e14a0962` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| SOLUSDT | 2026-06-29 | yes | 1440 | `98911fdf3ed80b200270f730004739f10034c8f1902227452cc87ad521df229d` | `2173a3fc` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| BNBUSDT | 2024-08-12 | yes | 1438 | `b7d799ea65a85800391962505dc5fbf835bbb3898eebb9491ea7766d55985ec0` | `a32de957` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| BNBUSDT | 2026-06-29 | yes | 1440 | `87521aba2193e665d9ce233c27169189d9e9e5c68d533d44a0be09652142ff96` | `35b5e01e` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| XRPUSDT | 2024-08-12 | yes | 1438 | `0e8749d7570ce6675a8b85fb83f5dac8c274d2236dc1d57b9654e7e94ab60959` | `f64eef00` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| XRPUSDT | 2026-06-29 | yes | 1440 | `d9322aaa740a8d48676eb97ee707e5e2f23d412849cf3f5aef9f1d60bbcbd835` | `864911e9` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| DOGEUSDT | 2024-08-12 | yes | 1438 | `ca17a4f41390d14a3798a0ded7b3f880dce0f174b87bb3d60883097a115734f7` | `d417cf78` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| DOGEUSDT | 2026-06-29 | yes | 1440 | `0f76c65be9b4d84ebca646acfa8bafae837801d1739a786d36c66a5969ca7024` | `33857287` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| ADAUSDT | 2024-08-12 | yes | 1438 | `eb15563626fcfc090b9ffc9fe599a47053cbfea7c67ed2842c12b86ccd413a24` | `7e16e611` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| ADAUSDT | 2026-06-29 | yes | 1440 | `997ad3b1b8a91cc243c64fa334425926d902b0ce06a57ad84df3c387f5f1e959` | `0523b8f5` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| AVAXUSDT | 2024-08-12 | yes | 1438 | `190d75ba2c6453b8bea86c95251e5be3eb76e94a46339d082d4388bde923549a` | `40e098fa` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| AVAXUSDT | 2026-06-29 | yes | 1440 | `eb7b11701d67eb5e303eab76e589850f1fe15cb3eb6b3ce2b77c585cbfbaaf73` | `019fb48a` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| LINKUSDT | 2024-08-12 | yes | 1438 | `aa30b860343a4e650605197c3b8f7ae958aed813e62b11798593eb34461e2b00` | `df242f03` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| LINKUSDT | 2026-06-29 | yes | 1440 | `281bb661ee8d53946a60f8514a64328ca6cdec45c97138baac238873d2b46774` | `b4b6cabd` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| SUIUSDT | 2024-08-12 | yes | 1438 | `fa7a7c1a05f5627f25b63d9c84df59df0b617a4fd0abf6a1306ec48ef37285bd` | `3c5424f2` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| SUIUSDT | 2026-06-29 | yes | 1440 | `456d942e4330e8138bafaf503034f5fc4d597962244a111f42df36b0238dbb62` | `7aa17b67` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |
| LTCUSDT | 2024-08-12 | yes | 1438 | `e5cb3ae7a9108aa2cae402f809d67163b228ffe5b2d0b7edc942fd2b35ff34c8` | `ecbab156` | true | `2024-08-12T00:00:00.000Z` | `2024-08-12T23:59:00.000Z` | 0 | 0 |
| LTCUSDT | 2026-06-29 | yes | 1440 | `0b485ead1804ed3d81aaeb0bac3c3e912b146771550deb3ed069bb8f94b36b16` | `7aac85a8` | true | `2026-06-29T00:00:00.000Z` | `2026-06-29T23:59:00.000Z` | 0 | 0 |

## 3. Overlap y conflictos

| Date | Overlap compared | Exact matches | Mismatches | Monthly missing | DAILY recovered | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| 2024-08-12 | 15818 | 15818 | 0 | 22 | 0 | 22 |
| 2026-06-29 | 0 | 0 | 0 | 15840 | 15840 | 0 |

Para 2024 se comparo todo el overlap del dia, incluyendo mas de 60 minutos antes
y despues del gap por simbolo. Para 2026 no existe overlap porque MONTHLY omite el
dia completo. Los DAILY tienen cobertura minuto a minuto completa y contienen
explicitamente `07:59Z`, `15:59Z` y `23:59Z` para los 11 simbolos.

No se registro `MONTHLY_DAILY_CONFLICT`. Los valores se compararon como strings
fuente exactas, sin normalizacion economica.

## 4. Lineage y precedencia

- Amendment SHA-256: `3caab32ab68671100b89e8bf06b28c26dc472f6e260f23c2fe3c38aaa17a69a2`.
- Source gap manifest SHA-256: `2f0c09d0489bdda45fe02787b3f626365bb1ec88bec67927a3ebb57b9e982c52`.
- MONTHLY manifest SHA-256: `1cc559055937f3d2432f0559a6badda6865495fdfd26f52f3f02c0943836f92b`.
- DAILY audit SHA-256: `fc0c4c464d7c90b9023b5417657e5681ebfac15643419f05b8af38d460781760`.

La precedencia es `MONTHLY_PRIMARY_DAILY_GAP_FILL_V1`: MONTHLY siempre gana si
existe; DAILY solo inserta un timestamp ausente. Una diferencia en overlap bloquea
la resolucion y un minuto ausente en ambas fuentes no se rellena.

## 5. Derivado reconstruido

Solo se reconstruyeron los 341 artefactos funding/mark. Los 2.987.424 rows OHLCV
y su manifest previo se heredaron por hash; no se recalculo OHLCV.

```text
derived_manifest_sha256: ccda624d7a683daad18888846cae53b8240f3a90bbc8838d6930ce301b290973
derived_logical_sha256: 8bcfb6ee88ece002e903774ba20e509536239c1f566617640fcd305b976fd2b4
funding_mark_artifacts: 341
funding_events_total: 31108
funding_events_mapped: 31108
funding_events_missing_mark_price: 0
remaining_mark_price_missing_minutes: 22
```

Los 22 minutos restantes son exclusivamente `2024-08-12T10:02Z` y `10:03Z`
para cada simbolo y no intersectan ningun evento funding.

## 6. Reproducibilidad

Se ejecutaron dos builds independientes en `build_a` y `build_b`:

```text
manifest byte equality: PASS
logical SHA-256 equality: PASS
341 artifact metadata comparisons: PASS
341 build_a artifact hashes: PASS
341 build_b artifact hashes: PASS
artifact mismatches: 0
```

Ambos manifests tienen SHA-256
`ccda624d7a683daad18888846cae53b8240f3a90bbc8838d6930ce301b290973`.

## 7. Drift, tests y parity

```text
R1 frozen files verified: 40/40
R1 manifest SHA-256: 5b93160123aa7a8059e92ce256df135e7bdfe71ae055bc195ce6854a1c16ac81
Python tests: 71/71 PASS
focused TS/Python and ATR tests: 10/10 PASS
TypeScript frozen tests: 17/17 PASS
TS/Python full decision parity: PASS
ATR binary64 parity without epsilon: PASS
child repository clean: true
```

Archivos nuevos de tooling, sin cambios R1:

- `source_gap.py`: `2274c519e0577f6e625b138f3a5d95d92268b5e4bc4e7f6f12595ffe7e4bd0e7`.
- `run_source_gap_resolution.py`: `0340502769634926dfe9e177b5ffa43442dd89dd02584c7074fffda316bdc0be`.
- `test_source_gap.py`: `232e8f30f0a1a279079ce6785b0fcb3436110c3f005c95e5da792318cc1365e5`.

## 8. Limites de fase

```text
economic_metrics_computed: false
RangeEngine_executed: false
TRAIN_ACCESS: false
CALIBRATION_ACCESS: false
VALIDATION_ACCESS: false
HOLDOUT_ACCESS: false
partitions_opened: false
production_modified: false
E4_modified: false
child_modified: false
backtest_executed: false
```

No se calculo PnL, win rate, expectancy, PF, DD, Sharpe, CVaR ni performance de
candidatos. No se ejecuto grid, seleccion, TRAIN, CALIBRATION, VALIDATION,
HOLDOUT, SHADOW o LIVE.

## 9. Git

```text
git diff --check: PASS
git status --short:
 M binance-futures-bot-ts
?? docs/aegis-range-v1/AEGIS_RANGE_R2_SOURCE_GAP_AMENDMENT.md
?? docs/aegis-range-v1/AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_REPORT.md
?? docs/aegis-range-v1/r2_source_gap_manifest.json
?? sandbox/aegis_range_strategy_v1/scripts/run_source_gap_resolution.py
?? sandbox/aegis_range_strategy_v1/src/aegis_range_v1/source_gap.py
?? sandbox/aegis_range_strategy_v1/tests/test_source_gap.py
```

El gitlink es preexistente y permanece fuera del alcance. No se hizo commit.

`AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_READY_FOR_REVIEW`
