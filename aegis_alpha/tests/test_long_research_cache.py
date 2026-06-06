#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from aegis_alpha.turbo.long_research_cache import LongResearchCache, assert_research_cache_path, db_mtime_key


def main():
    tmp = Path(tempfile.mkdtemp())
    db = tmp / 'candles.db'
    db.write_text('a')
    cache = LongResearchCache(max_items=2)
    key = cache.ohlcv_key('ETHUSDT', 60, db)
    assert cache.get('ohlcv', *key) is None
    assert cache.misses == 1
    cache.set('ohlcv', *key, {'rows': 1})
    assert cache.get('ohlcv', *key)['rows'] == 1
    assert cache.hits == 1
    cache.set('labels', *cache.labels_key('ETHUSDT', 'long_roe12_before_minus8', 6, 60, db), {'hit': np.array([1])})
    cache.set('folds', *cache.folds_key(1000, 4, 100, 50), [(np.array([0]), np.array([1]))])
    assert cache.evictions == 1
    summary = cache.summary()
    assert summary['items'] == 2
    json.dumps(summary)
    cache.clear()
    assert cache.summary()['items'] == 0

    old = db_mtime_key(db)
    db.write_text('b')
    os.utime(db, (db.stat().st_atime + 2, db.stat().st_mtime + 2))
    new = db_mtime_key(db)
    assert old != new
    assert cache.labels_key('ETHUSDT', 'target_a', 6, 60, db) != cache.labels_key('ETHUSDT', 'target_b', 6, 60, db)
    assert cache.folds_key(1000, 4, 100, 50) != cache.folds_key(1001, 4, 100, 50)
    try:
        assert_research_cache_path(ROOT / 'aegis_alpha/models/turbo/ETHUSDT/active/x.joblib')
        raise AssertionError('expected active path rejection')
    except ValueError:
        pass
    print('PASS test_long_research_cache')


if __name__ == '__main__':
    main()
