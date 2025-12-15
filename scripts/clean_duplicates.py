import sys
from pathlib import Path
from sqlalchemy import text
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.storage.database_manager import db_manager

def clean():
    print("Cleaning duplicates...")
    # SQL to delete duplicates keeping the one with smallest ID
    sql = """
    DELETE FROM ohlcv_data 
    WHERE id NOT IN (
        SELECT MIN(id) 
        FROM ohlcv_data 
        GROUP BY symbol, timeframe, timestamp
    )
    """
    
    with db_manager.engine.connect() as conn:
        result = conn.execute(text(sql))
        conn.commit()
        print(f"Deleted {result.rowcount} duplicate rows.")

if __name__ == "__main__":
    clean()
