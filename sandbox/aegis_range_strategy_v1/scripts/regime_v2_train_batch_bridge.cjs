'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { once } = require('node:events');
const { RegimeEngineV2 } = require(path.join(
  process.cwd(),
  'src/domain/services/regime-v2/RegimeEngineV2',
));

async function writeAll(value) {
  if (!process.stdout.write(value)) {
    await once(process.stdout, 'drain');
  }
}

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
if (
  Object.prototype.hasOwnProperty.call(input, 'market') ||
  input.timeframe !== '5m' ||
  typeof input.symbol !== 'string' ||
  !Array.isArray(input.candles)
) {
  throw new Error('AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY');
}

let previousTimestamp;
for (const candle of input.candles) {
  const values = [candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume];
  if (
    values.some((value) => typeof value !== 'number' || !Number.isFinite(value)) ||
    candle.timestamp % 300000 !== 0 ||
    (previousTimestamp !== undefined && candle.timestamp !== previousTimestamp + 300000) ||
    Math.min(candle.open, candle.high, candle.low, candle.close) <= 0 ||
    candle.volume < 0 ||
    candle.high < Math.max(candle.open, candle.close, candle.low) ||
    candle.low > Math.min(candle.open, candle.close, candle.high)
  ) {
    throw new Error('AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY');
  }
  previousTimestamp = candle.timestamp;
}

async function main() {
  const output = [];
  for (let index = 159; index < input.candles.length; index += 1) {
    const decision = RegimeEngineV2.evaluate({
      symbol: input.symbol,
      timeframe: '5m',
      candles: input.candles.slice(index - 159, index + 1),
    });
    const indicators = decision.indicators;
    output.push(JSON.stringify({
      timestamp: input.candles[index].timestamp,
      technicalRegime: decision.technicalRegime,
      transitionRisk: decision.transition.risk,
      adx: indicators.adx,
      atrPercentile: indicators.atrPercentile,
      bollingerWidthPercentile: indicators.bollingerWidthPercentile,
      volumeRatio: indicators.volumeRatio,
      rangeBreakout: indicators.rangeBreakout,
      failedBreakoutCount: indicators.failedBreakoutCount,
      structure: indicators.structure,
      chopRisk: decision.scores.chopRisk,
    }));
    if (output.length === 1024) {
      await writeAll(`${output.join('\n')}\n`);
      output.length = 0;
    }
  }
  if (output.length) {
    await writeAll(`${output.join('\n')}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
