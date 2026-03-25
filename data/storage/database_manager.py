# =============================================================================
# 5. DATA/STORAGE/DATABASE_MANAGER.PY - Gestión de Base de Datos
# =============================================================================
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from contextlib import contextmanager
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from config.settings import settings
from data.storage.models import Base, OHLCVData, MarketData, TradingSignals

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def _create_engine(self):
        # Crear engine con configuraciones optimizadas
        if "sqlite" in self.database_url:
            # SQLite optimizations
            engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=False  # Set True for SQL debugging
            )
        else:
            # PostgreSQL optimizations
            engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
        return engine
    
    def create_tables(self):
        # Crear todas las tablas
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        # Eliminar todas las tablas (¡CUIDADO!)
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        # Context manager para sesiones de DB
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def insert_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        # Insertar datos OHLCV en batch
        with self.get_session() as session:
            records = []
            for _, row in df.iterrows():
                record = OHLCVData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row['timestamp'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    buy_volume=float(row['buy_volume']) if 'buy_volume' in row and pd.notna(row['buy_volume']) else None
                )
                records.append(record)
            
            # Insertar en batch
            session.bulk_save_objects(records)
            logger.info(f"Inserted {len(records)} OHLCV records for {symbol} {timeframe}")
            return len(records)
    
    def get_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        # Bypass SQLAlchemy for massive performance + avoid mysterious table locks
        import sqlite3
        conn = sqlite3.connect(self.database_url.replace("sqlite:///", ""))
        
        limit_clause = f"ORDER BY timestamp DESC LIMIT {limit}" if limit else "ORDER BY timestamp ASC"
        query = f"SELECT timestamp, open, high, low, close, volume, buy_volume FROM ohlcv_data WHERE symbol='{symbol}' AND timeframe='{timeframe}' {limit_clause}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        if limit:
            df = df.sort_index(ascending=True)
            
        return df
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        # Obtener el timestamp más reciente para un símbolo/timeframe
        with self.get_session() as session:
            result = session.query(OHLCVData.timestamp).filter(
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe
            ).order_by(OHLCVData.timestamp.desc()).first()
            
            return result[0] if result else None
    
    def data_exists(self, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        # Verificar si ya existen datos para un timestamp específico
        with self.get_session() as session:
            count = session.query(OHLCVData).filter(
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe,
                OHLCVData.timestamp == timestamp
            ).count()
            
            return count > 0
    
    def get_data_gaps(self, symbol: str, timeframe: str) -> List[tuple]:
        # Identificar gaps en los datos
        df = self.get_ohlcv_data(symbol, timeframe)
        if df.empty:
            return []
        
        # Calcular intervalos esperados según timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        
        if timeframe not in timeframe_minutes:
            return []
        
        interval = timedelta(minutes=timeframe_minutes[timeframe])
        gaps = []
        
        df_sorted = df.sort_index()
        timestamps = df_sorted.index
        
        for i in range(1, len(timestamps)):
            expected_time = timestamps[i-1] + interval
            actual_time = timestamps[i]
            
            if actual_time > expected_time:
                gaps.append((timestamps[i-1], timestamps[i]))
        
        return gaps
    
    def optimize_database(self):
        # Optimizar base de datos (SQLite específico)
        if "sqlite" in self.database_url:
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM;"))
                conn.execute(text("ANALYZE;"))
                logger.info("SQLite database optimized")

# Instancia global
db_manager = DatabaseManager()