'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { RegimeEngineV2 } = require(path.join(
  process.cwd(),
  'src/domain/services/regime-v2/RegimeEngineV2',
));

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
if (
  Object.prototype.hasOwnProperty.call(input, 'market') ||
  input.timeframe !== '5m' ||
  !Array.isArray(input.candles) ||
  input.candles.length !== 160
) {
  throw new Error('AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_PARITY');
}
process.stdout.write(JSON.stringify(RegimeEngineV2.evaluate(input)));
