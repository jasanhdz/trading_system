# Aegis News/Sentiment Risk Collector v0.1

## Purpose

The collector produces an external event risk snapshot for Aegis in SHADOW mode. It is designed for context only: it does not open trades, close trades, block entries, change leverage, or change the live Event Risk Overlay mode.

The output is meant to become a future Decision Brain feature.

## Outputs

Latest snapshot:

```text
aegis_alpha/data/processed/event_risk/latest_event_sentiment_risk.json
```

Daily audit log:

```text
aegis_alpha/logs/event_risk/event_sentiment_risk_YYYYMMDD.jsonl
```

The core fields are:

```json
{
  "mode": "SHADOW",
  "suggested_mode": "CAUTION",
  "risk_score": 0.68,
  "confidence": 0.6,
  "top_events": [],
  "sources": [],
  "summary": "...",
  "status": "ok",
  "execute": false,
  "production_allowed": false,
  "does_not_change_trading": true,
  "does_not_change_event_risk_mode": true
}
```

## Modes

- `NORMAL`: no meaningful external risk was detected.
- `CAUTION`: external conditions are mixed, uncertain, or moderately risky.
- `RISK_OFF`: external event risk is elevated enough that future systems should treat new entries conservatively.
- `MANUAL_ONLY`: extreme or unreliable conditions where human review should be preferred.

These are suggestions only. The real trading overlay remains controlled by the TS Event Risk Overlay and Telegram/YAML configuration.

## Sources

Version 0.1 uses structured public inputs when available:

- Manual fallback file.
- Crypto Fear & Greed Index.
- Public RSS headlines from crypto news sources.

No API keys are used. If HTTP/RSS fails, the collector still writes a snapshot with `status="source_unavailable"` or `status="partial"`.

## Manual Override

Manual fallback file:

```text
aegis_alpha/config/manual_event_risk.json
```

Example:

```json
{
  "enabled": true,
  "mode": "NORMAL",
  "expires_at": null,
  "reason": "manual default"
}
```

If the file is enabled and not expired, the collector includes it. A higher-risk manual mode acts as a conservative floor for `suggested_mode`; it does not lower risk detected from external sources.

## Classifier

Version 0.1 intentionally does not use an LLM. It uses keyword/rule scoring for categories such as:

- Geopolitics: `war`, `attack`, `ceasefire`, `sanctions`.
- Macro: `CPI`, `PPI`, `Fed`, `rates`, `inflation`.
- Crypto market structure: `ETF outflows`, `ETF inflows`, `liquidation`.
- Exchange/security events: `hack`, `exploit`, `depeg`, `exchange outage`.
- Political risk: `Trump`, `tariff`, `SEC`.

Positive/risk-on keywords reduce the score only for the matching event, for example `ETF inflows`, `rate cut`, `lower inflation`, and `ceasefire confirmed`.

## Run

Run once:

```bash
cd /home/jasan/Develop/trading_system
python3 aegis_alpha/tools/collect_event_sentiment_risk.py
```

Manual-only smoke run without HTTP:

```bash
python3 aegis_alpha/tools/collect_event_sentiment_risk.py --no-http
```

Review latest:

```bash
cat aegis_alpha/data/processed/event_risk/latest_event_sentiment_risk.json
```

Review daily log:

```bash
tail -n 20 aegis_alpha/logs/event_risk/event_sentiment_risk_$(date +%Y%m%d).jsonl
```

## Limitations

- Public sources can fail or change format.
- Headlines can be duplicated, stale, incomplete, or misleading.
- Keyword scoring is deliberately simple and may overreact to noisy headlines.
- This collector does not decide direction and does not make trading decisions.
