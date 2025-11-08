# =============================================================================
# 9. UTILS/EXCEPTIONS.PY - Excepciones personalizadas
# =============================================================================

class TradingSystemError(Exception):
    # Base exception for trading system
    pass

class DataCollectionError(TradingSystemError):
    # # Error during data collection
    pass

class DatabaseError(TradingSystemError):
    # # Database related error
    pass

class ValidationError(TradingSystemError):
    # # Data validation error
    pass

class ConfigurationError(TradingSystemError):
    # # Configuration error
    pass

class StrategyError(TradingSystemError):
    # # Strategy execution error
    pass
