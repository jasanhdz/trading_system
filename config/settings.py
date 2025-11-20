# =============================================================================
# 3. CONFIG/SETTINGS.PY - Configuraciones Centralizadas
# =============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    # Directorios base
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Database Configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{BASE_DIR}/data/xrp_trading.db"
    )
    
    # Exchange API Keys
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Trading Configuration - CORREGIDO: Consistencia en símbolo
    SYMBOL = os.getenv("SYMBOL", "XRP/USDT")  # Formato CCXT
    BINANCE_SYMBOL = os.getenv("SYMBOL", "XRPUSDT")  # Formato Binance nativo
    LEVERAGE = int(os.getenv("LEVERAGE", "10"))  # Reducido de 50 a 10 (más conservador)
    
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    DEFAULT_LIMIT = 1000
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting - MEJORADO
    REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.1"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = 30
    
    @classmethod
    def ensure_directories(cls):
        """Crear directorios necesarios"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.BASE_DIR / "data").mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validar configuración crítica"""
        issues = []
        
        if not cls.BINANCE_API_KEY and not cls.BINANCE_SECRET_KEY:
            # OK para datos públicos
            pass
        elif cls.BINANCE_API_KEY and not cls.BINANCE_SECRET_KEY:
            issues.append("BINANCE_SECRET_KEY missing")
        elif cls.BINANCE_SECRET_KEY and not cls.BINANCE_API_KEY:
            issues.append("BINANCE_API_KEY missing")

        if cls.LEVERAGE > 20:
            issues.append(f"High leverage detected: {cls.LEVERAGE}x (consider reducing)")
            
        return issues

settings = Settings()
