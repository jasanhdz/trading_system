import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.storage.database_manager import db_manager
import pandas as pd

def check():
    with db_manager.get_session() as session:
        from data.storage.models import OHLCVData
        symbols = session.query(OHLCVData.symbol, OHLCVData.timeframe).distinct().all()
        print("Available Data:")
        for s, tf in symbols:
            count = session.query(OHLCVData).filter_by(symbol=s, timeframe=tf).count()
            print(f"  {s} [{tf}]: {count} rows")

if __name__ == "__main__":
    check()
