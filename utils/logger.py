# =============================================================================
# 8. UTILS/LOGGER.PY - Sistema de Logging
# =============================================================================
import logging
import logging.handlers
from pathlib import Path
from config.settings import settings

def setup_logger(name: str = None) -> logging.Logger:
    # Configurar logger con formato y handlers apropiados
    
    # Crear directorio de logs si no existe
    settings.ensure_directories()
    
    # Configurar logger principal
    logger = logging.getLogger(name or 'xrp_trading')
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler con rotación
    log_file = settings.LOGS_DIR / "trading_system.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error File Handler
    error_log_file = settings.LOGS_DIR / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

# Logger principal
main_logger = setup_logger()