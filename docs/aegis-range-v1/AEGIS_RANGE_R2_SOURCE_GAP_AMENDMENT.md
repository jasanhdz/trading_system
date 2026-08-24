# Aegis Range Strategy V1 - R2 Source Gap Amendment

**Fecha efectiva:** 2026-08-24

**Estado:** `AEGIS_RANGE_R2_SOURCE_GAP_AMENDMENT`

**Politica:** `MONTHLY_PRIMARY_DAILY_GAP_FILL_V1`

## 1. Motivo y alcance

Los 341 ZIP mensuales oficiales `markPriceKlines/1m` siguen siendo la fuente
primaria. Todos coinciden con sus checksums y no estan corruptos, pero sus CSV
contienen ausencias internas uniformes en los 11 simbolos:

- `2024-08-12T10:02Z` y `2024-08-12T10:03Z`;
- `2026-06-29T00:00Z` a `2026-06-29T23:59Z`.

Esta enmienda autoriza exclusivamente Binance Vision USD-M Futures DAILY
`markPriceKlines/1m` para buscar y completar minutos ausentes de esos dos dias.
No autoriza otras fechas, APIs live, spot, otra exchange, interpolacion,
imputacion, cambios de funding, cambios R1 ni ejecucion de estrategia.

## 2. Resultado DAILY

Los 22 ZIP DAILY oficiales existen y coinciden con sus sidecars `.CHECKSUM`.
Todos tienen CRC valido, un unico member CSV esperado, timestamps ordenados,
cero duplicados y cero filas OHLC invalidas.

- Los DAILY de `2024-08-12` contienen 1.438 minutos y omiten exactamente los
  mismos minutos `10:02Z` y `10:03Z`. No recuperan esas 22 filas totales.
- Los DAILY de `2026-06-29` contienen los 1.440 minutos completos para cada
  simbolo. Recuperan 15.840 filas totales, incluidos `07:59Z`, `15:59Z` y
  `23:59Z`, necesarios para los 33 eventos funding bloqueados.
- Se compararon exactamente 15.818 filas de overlap MONTHLY/DAILY:
  15.818 matches y 0 mismatches en `open_time`, `open`, `high`, `low`, `close`.

Los 22 minutos que siguen ausentes no intersectan funding. El contrato permite
documentar gaps restantes; el blocker contractual se resuelve solo si los 33
eventos funding alcanzan mapping exacto sin imputacion.

## 3. Precedencia congelada

`MONTHLY_PRIMARY_DAILY_GAP_FILL_V1` aplica por simbolo y minuto `T`:

1. Si MONTHLY contiene `T`, se usa MONTHLY. DAILY nunca lo reemplaza.
2. Si MONTHLY no contiene `T` y DAILY oficial contiene exactamente `T`, se usa DAILY.
3. Si ambos contienen `T` y difieren en cualquier OHLC, se registra `MONTHLY_DAILY_CONFLICT`; MONTHLY conserva autoridad y la resolucion queda bloqueada.
4. Si ambos omiten `T`, no se imputa ni interpola. El minuto sigue ausente.
5. DAILY no cambia fundingRate ni el contrato `mark_open_time=funding_at-60s`.

## 4. Archivos DAILY congelados

| Symbol | Date | Official path | SHA-256 |
|---|---|---|---|
| BTCUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2024-08-12.zip` | `139f800b758d844744b51a2d2fa1e0391447091faba2cccedb17028acb34d8cc` |
| BTCUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-06-29.zip` | `200e2ef30a86e9f91c011dae30d864f73c8480fddecc73ea227958596f6cd11c` |
| ETHUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/ETHUSDT/1m/ETHUSDT-1m-2024-08-12.zip` | `11a3640b1e90491a5491c8abe6182e4b4cb6f6ce5dde3c7ba34a0e5ccbda6552` |
| ETHUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/ETHUSDT/1m/ETHUSDT-1m-2026-06-29.zip` | `cc07f9a3f5580a443512096c2de20d3d59a52be76f07668c87a6418bf762f07b` |
| SOLUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/SOLUSDT/1m/SOLUSDT-1m-2024-08-12.zip` | `6dd1cb8302a8b4f001f20faa13db417010ec41e073dd720c4ed19529e70a09fe` |
| SOLUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/SOLUSDT/1m/SOLUSDT-1m-2026-06-29.zip` | `98911fdf3ed80b200270f730004739f10034c8f1902227452cc87ad521df229d` |
| BNBUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/BNBUSDT/1m/BNBUSDT-1m-2024-08-12.zip` | `b7d799ea65a85800391962505dc5fbf835bbb3898eebb9491ea7766d55985ec0` |
| BNBUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/BNBUSDT/1m/BNBUSDT-1m-2026-06-29.zip` | `87521aba2193e665d9ce233c27169189d9e9e5c68d533d44a0be09652142ff96` |
| XRPUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/XRPUSDT/1m/XRPUSDT-1m-2024-08-12.zip` | `0e8749d7570ce6675a8b85fb83f5dac8c274d2236dc1d57b9654e7e94ab60959` |
| XRPUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/XRPUSDT/1m/XRPUSDT-1m-2026-06-29.zip` | `d9322aaa740a8d48676eb97ee707e5e2f23d412849cf3f5aef9f1d60bbcbd835` |
| DOGEUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/DOGEUSDT/1m/DOGEUSDT-1m-2024-08-12.zip` | `ca17a4f41390d14a3798a0ded7b3f880dce0f174b87bb3d60883097a115734f7` |
| DOGEUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/DOGEUSDT/1m/DOGEUSDT-1m-2026-06-29.zip` | `0f76c65be9b4d84ebca646acfa8bafae837801d1739a786d36c66a5969ca7024` |
| ADAUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/ADAUSDT/1m/ADAUSDT-1m-2024-08-12.zip` | `eb15563626fcfc090b9ffc9fe599a47053cbfea7c67ed2842c12b86ccd413a24` |
| ADAUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/ADAUSDT/1m/ADAUSDT-1m-2026-06-29.zip` | `997ad3b1b8a91cc243c64fa334425926d902b0ce06a57ad84df3c387f5f1e959` |
| AVAXUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/AVAXUSDT/1m/AVAXUSDT-1m-2024-08-12.zip` | `190d75ba2c6453b8bea86c95251e5be3eb76e94a46339d082d4388bde923549a` |
| AVAXUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/AVAXUSDT/1m/AVAXUSDT-1m-2026-06-29.zip` | `eb7b11701d67eb5e303eab76e589850f1fe15cb3eb6b3ce2b77c585cbfbaaf73` |
| LINKUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/LINKUSDT/1m/LINKUSDT-1m-2024-08-12.zip` | `aa30b860343a4e650605197c3b8f7ae958aed813e62b11798593eb34461e2b00` |
| LINKUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/LINKUSDT/1m/LINKUSDT-1m-2026-06-29.zip` | `281bb661ee8d53946a60f8514a64328ca6cdec45c97138baac238873d2b46774` |
| SUIUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/SUIUSDT/1m/SUIUSDT-1m-2024-08-12.zip` | `fa7a7c1a05f5627f25b63d9c84df59df0b617a4fd0abf6a1306ec48ef37285bd` |
| SUIUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/SUIUSDT/1m/SUIUSDT-1m-2026-06-29.zip` | `456d942e4330e8138bafaf503034f5fc4d597962244a111f42df36b0238dbb62` |
| LTCUSDT | 2024-08-12 | `data/futures/um/daily/markPriceKlines/LTCUSDT/1m/LTCUSDT-1m-2024-08-12.zip` | `e5cb3ae7a9108aa2cae402f809d67163b228ffe5b2d0b7edc942fd2b35ff34c8` |
| LTCUSDT | 2026-06-29 | `data/futures/um/daily/markPriceKlines/LTCUSDT/1m/LTCUSDT-1m-2026-06-29.zip` | `0b485ead1804ed3d81aaeb0bac3c3e912b146771550deb3ed069bb8f94b36b16` |

Byte sizes, CRC, members, rows y coverage exacta se congelan en
`docs/aegis-range-v1/r2_source_gap_manifest.json`.

## 5. Limite experimental

Esta enmienda no ejecuta `RangeEngineV1`, backtest, TRAIN, CALIBRATION,
VALIDATION, HOLDOUT, grid, seleccion, PnL ni metricas economicas. Los cuatro
guards de particion permanecen cerrados y no se modifica R0, R1, E4, produccion
ni el repositorio TypeScript.
