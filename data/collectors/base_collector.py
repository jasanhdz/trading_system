# =============================================================================
# 6. DATA/COLLECTORS/BASE_COLLECTOR.PY - Clase abstracta
# =============================================================================
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseDataCollector(ABC):
    # Clase abstracta para recolectores de datos
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = logger.getChild(exchange_name)
    
    @abstractmethod
    def connect(self) -> bool:
        # Conectar al exchange
        pass
    
    @abstractmethod
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: datetime = None, 
        limit: int = 1000
    ) -> pd.DataFrame:
        # Obtener datos OHLCV
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        # Obtener símbolos disponibles
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict:
        # Obtener datos adicionales del mercado
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        # Validar datos básicos
        if df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Verificar que high >= low, close >= 0, etc.
        if (df['high'] < df['low']).any():
            self.logger.warning("Found invalid OHLC data: high < low")
            return False
        
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            self.logger.warning("Found negative prices")
            return False
        
        return True
    
    def format_ohlcv_data(self, raw_data: List) -> pd.DataFrame:
        # Formatear datos crudos a DataFrame estándar
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convertir a float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Eliminar duplicados
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        
        # Ordenar por timestamp
        df.sort_values('timestamp', inplace=True)
        
        return df

