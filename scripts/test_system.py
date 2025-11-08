import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

def check_environment():
    """Verificar entorno y dependencias"""
    print("🔍 Checking environment...")
    
    try:
        import pandas as pd
        import numpy as np
        import ccxt
        import sqlalchemy
        from sqlalchemy import __version__ as sqlalchemy_version
        print(f"✅ pandas: {pd.__version__}")
        print(f"✅ numpy: {np.__version__}")
        print(f"✅ ccxt: {ccxt.__version__}")
        print(f"✅ sqlalchemy: {sqlalchemy_version}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_configuration():
    """Verificar configuración del sistema"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from config.settings import settings
        
        # Validar configuración
        issues = settings.validate_config()
        
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        print(f"✅ Symbol: {settings.SYMBOL}")
        print(f"✅ Leverage: {settings.LEVERAGE}x")
        print(f"✅ API Keys: {'Configured' if settings.BINANCE_API_KEY else 'Not configured (using public data)'}")
        
        if issues:
            print("\n⚠️ Configuration warnings:")
            for issue in issues:
                print(f"  - {issue}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_database_operations():
    """Verificar operaciones de base de datos"""
    print("\n🗄️ Checking database operations...")
    
    try:
        from data.storage.database_manager import db_manager
        from data.storage.models import Base
        
        # Crear tablas
        db_manager.create_tables()
        print("✅ Database tables created")
        
        # Test session
        with db_manager.get_session() as session:
            print("✅ Database session works")
        
        # Test data insertion/retrieval
        import pandas as pd
        test_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [0.5],
            'high': [0.6],
            'low': [0.4],
            'close': [0.55],
            'volume': [1000.0]
        })
        
        records = db_manager.insert_ohlcv_data(test_data, 'TEST/USDT', '1m')
        print(f"✅ Test data insertion: {records} records")
        
        # Retrieve test data
        retrieved_data = db_manager.get_ohlcv_data('TEST/USDT', '1m')
        print(f"✅ Test data retrieval: {len(retrieved_data)} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_binance_connection():
    """Verificar conexión a Binance"""
    print("\n📡 Checking Binance connection...")
    
    try:
        from data.collectors.binance_collector import BinanceDataCollector
        
        collector = BinanceDataCollector()
        
        if not collector.connect():
            print("❌ Failed to connect to Binance")
            return False
        
        print("✅ Binance connection successful")
        
        # Test market data
        market_data = collector.get_market_data('XRP/USDT')
        if market_data:
            price = market_data.get('mark_price', 'N/A')
            print(f"✅ XRP/USDT price: ${price}")
        
        # Test small data collection
        print("🔄 Testing small data collection...")
        df = collector.get_ohlcv('XRP/USDT', '1h', limit=5)
        
        if not df.empty:
            print(f"✅ Collected {len(df)} test candles")
            price_range = f"${df['low'].min():.4f} - ${df['high'].max():.4f}"
            print(f"✅ Price range: {price_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ Binance connection error: {e}")
        return False

def check_data_pipeline():
    """Verificar pipeline completo de datos"""
    print("\n🔄 Testing full data pipeline...")
    
    try:
        from data.collectors.binance_collector import BinanceDataCollector
        from data.storage.database_manager import db_manager
        
        collector = BinanceDataCollector()
        
        # Collect small dataset
        df = collector.get_ohlcv('XRP/USDT', '15m', limit=10)
        
        if df.empty:
            print("❌ No data collected")
            return False
        
        # Insert into database  
        records = db_manager.insert_ohlcv_data(df, 'XRP/USDT', '15m')
        print(f"✅ Inserted {records} records")
        
        # Retrieve from database
        retrieved = db_manager.get_ohlcv_data('XRP/USDT', '15m', limit=5)
        print(f"✅ Retrieved {len(retrieved)} records")
        
        # Check data integrity
        if len(retrieved) > 0:
            latest_price = retrieved.iloc[-1]['close']
            print(f"✅ Latest XRP price: ${latest_price:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        return False

def performance_benchmark():
    """Benchmark básico de performance"""
    print("\n⚡ Running performance benchmark...")
    
    try:
        from data.collectors.binance_collector import BinanceDataCollector
        
        collector = BinanceDataCollector()
        
        # Benchmark data collection speed
        start_time = time.time()
        df = collector.get_ohlcv('XRP/USDT', '5m', limit=100)
        collection_time = time.time() - start_time
        
        if not df.empty:
            records_per_second = len(df) / collection_time
            print(f"✅ Collection speed: {records_per_second:.1f} records/second")
            print(f"✅ Collected {len(df)} records in {collection_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return False

def security_check():
    """Verificaciones de seguridad básicas"""
    print("\n🔒 Security check...")
    
    try:
        from config.settings import settings
        
        # Check if .env file exists and is not in git
        env_file = Path(".env")
        if env_file.exists():
            print("✅ .env file found")
        else:
            print("⚠️ .env file not found")
        
        gitignore = Path(".gitignore")
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                if '.env' in f.read():
                    print("✅ .env is in .gitignore")
                else:
                    print("⚠️ .env should be added to .gitignore")
        else:
            print("⚠️ Create .gitignore file and add .env")
        
        # Check API key format (básico)
        if settings.BINANCE_API_KEY:
            if len(settings.BINANCE_API_KEY) > 50:
                print("✅ API key format looks valid")
            else:
                print("⚠️ API key format may be invalid")
        
        return True
        
    except Exception as e:
        print(f"❌ Security check error: {e}")
        return False

def main():
    """Ejecutar todas las verificaciones"""
    print("🚀 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Environment Check", check_environment),
        ("Configuration Check", check_configuration),
        ("Database Operations", check_database_operations),
        ("Binance Connection", check_binance_connection),
        ("Data Pipeline", check_data_pipeline),
        ("Performance Benchmark", performance_benchmark),
        ("Security Check", security_check),
    ]
    
    passed = 0
    total = len(checks)
    start_time = time.time()
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} - PASSED")
            else:
                print(f"❌ {check_name} - FAILED")
        except Exception as e:
            print(f"💥 {check_name} - ERROR: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"🏁 FINAL RESULTS: {passed}/{total} checks passed")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("🎉 SYSTEM IS FULLY OPERATIONAL!")
        print("\n📋 Ready for next steps:")
        print("1. Collect historical data: python scripts/collect_historical_data.py")
        print("2. Start data analysis and feature engineering")
        print("3. Begin pattern detection development")
    else:
        print("🔧 Some components need attention before proceeding")
        print("📝 Address the failed checks above")
    
    return passed == total

if __name__ == "__main__":
    main()