import sys
from pathlib import Path
from sqlalchemy import text
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.storage.database_manager import db_manager

def fast_clean():
    print("Fast cleaning duplicates using rowid...")
    sql = """
    DELETE FROM ohlcv_data 
    WHERE rowid NOT IN (
        SELECT MIN(rowid) 
        FROM ohlcv_data 
        GROUP BY symbol, timeframe, timestamp
    )
    """
    
    with db_manager.engine.connect() as conn:
        result = conn.execute(text(sql))
        conn.commit()
        print(f"Deleted {result.rowcount} duplicate rows.")

if __name__ == "__main__":
    fast_clean()
