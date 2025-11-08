#!/usr/bin/env python3
"""
Script mejorado para recolectar datos históricos con opciones de limpieza automática
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import click

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.collectors.binance_collector import BinanceDataCollector
from data.storage.database_manager import db_manager
from config.settings import settings
from utils.logger import setup_logger

@click.command()
@click.option('--symbol', default='XRP/USDT', help='Trading symbol')
@click.option('--timeframes', default='5m,15m,1h,4h', help='Comma-separated timeframes')
@click.option('--days', default=90, help='Number of days to collect')
@click.option('--force', is_flag=True, help='Force re-download existing data')
@click.option('--clean', is_flag=True, help='Clean existing data for symbol/timeframe before collecting')
@click.option('--clean-all', is_flag=True, help='Clean ALL data in database before collecting (DANGEROUS)')
def collect_data(symbol, timeframes, days, force, clean, clean_all):
    """Recolectar datos históricos con opciones de limpieza avanzadas"""
    logger = setup_logger("data_collector")
    
    timeframes_list = [tf.strip() for tf in timeframes.split(',')]
    
    # Usar datetime con timezone UTC explícito
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Starting data collection for {symbol}")
    logger.info(f"Timeframes: {timeframes_list}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')} UTC")
    logger.info(f"Total days: {days}")
    
    # Mostrar opciones seleccionadas
    if clean_all:
        logger.warning("CLEAN ALL MODE: Will delete ALL existing data in database!")
    elif clean:
        logger.info(f"CLEAN MODE: Will delete existing data for {symbol} in specified timeframes")
    elif force:
        logger.info("FORCE MODE: Will re-download existing data")
    else:
        logger.info("INCREMENTAL MODE: Will only collect missing data")
    
    # Confirmación para clean-all
    if clean_all:
        response = input("\nThis will DELETE ALL DATA in the database. Are you sure? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Operation cancelled by user")
            return
    
    # Inicializar collector
    collector = BinanceDataCollector()
    
    if not collector.connect():
        logger.error("Failed to connect to Binance")
        sys.exit(1)
    
    # Limpieza de datos si se solicita
    if clean_all:
        logger.warning("Cleaning ALL data from database...")
        db_manager.drop_tables()
        db_manager.create_tables()
        logger.info("Database completely cleaned and recreated")
    elif clean:
        logger.info(f"Cleaning existing data for {symbol} in timeframes: {timeframes_list}")
        clean_existing_data(symbol, timeframes_list)
    
    total_records = 0
    successful_timeframes = []
    failed_timeframes = []
    
    for timeframe in timeframes_list:
        logger.info(f"Collecting {timeframe} data...")
        
        try:
            # Verificar datos existentes solo si no estamos en modo clean
            actual_start_date = start_date
            
            if not force and not clean and not clean_all:
                latest_timestamp = db_manager.get_latest_timestamp(symbol, timeframe)
                if latest_timestamp:
                    # Convertir a UTC si no tiene timezone
                    if latest_timestamp.tzinfo is None:
                        latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
                    
                    logger.info(f"Found existing data up to {latest_timestamp}")
                    
                    # Solo continuar si hay gap significativo
                    time_gap = end_date - latest_timestamp
                    if time_gap < timedelta(hours=1):
                        logger.info(f"Data for {timeframe} is already up to date (gap: {time_gap})")
                        continue
                    else:
                        # Continuar desde donde se quedó
                        actual_start_date = latest_timestamp + timedelta(minutes=get_timeframe_minutes(timeframe))
                        logger.info(f"Continuing from {actual_start_date}")
            
            # Recolectar datos
            df = collector.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=actual_start_date,
                end_date=end_date
            )
            
            if not df.empty:
                # Insertar en base de datos
                records_inserted = db_manager.insert_ohlcv_data(df, symbol, timeframe)
                total_records += records_inserted
                successful_timeframes.append(f"{timeframe}: {records_inserted:,} records")
                
                # Mostrar estadísticas de los datos
                price_range = f"${df['low'].min():.4f} - ${df['high'].max():.4f}"
                latest_price = df.iloc[-1]['close']
                logger.info(f"{timeframe}: {records_inserted:,} records inserted")
                logger.info(f"  Price range: {price_range}")
                logger.info(f"  Latest price: ${latest_price:.4f}")
                logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
            else:
                logger.warning(f"{timeframe}: No data collected")
                failed_timeframes.append(timeframe)
                
        except Exception as e:
            logger.error(f"Error collecting {timeframe} data: {e}")
            failed_timeframes.append(f"{timeframe} (error: {str(e)[:50]})")
            continue
    
    # Reporte final
    logger.info("=" * 60)
    logger.info(f"DATA COLLECTION COMPLETED!")
    logger.info(f"Total records collected: {total_records:,}")
    
    if successful_timeframes:
        logger.info("Successful collections:")
        for success in successful_timeframes:
            logger.info(f"  ✓ {success}")
    
    if failed_timeframes:
        logger.warning("Failed collections:")
        for failure in failed_timeframes:
            logger.warning(f"  ✗ {failure}")
    
    # Verificación final de datos
    logger.info("Final database status:")
    for tf in timeframes_list:
        df = db_manager.get_ohlcv_data(symbol, tf, limit=1)
        if not df.empty:
            total_records_tf = len(db_manager.get_ohlcv_data(symbol, tf))
            latest_price = df.iloc[-1]['close']
            logger.info(f"  {tf}: {total_records_tf:,} total records, latest price: ${latest_price:.4f}")
        else:
            logger.info(f"  {tf}: No data")
    
    # Optimizar base de datos
    if total_records > 0:
        logger.info("Optimizing database...")
        db_manager.optimize_database()
        logger.info("Database optimization completed")
    
    logger.info("Setup complete!")

def clean_existing_data(symbol: str, timeframes: list):
    """Limpiar datos existentes para símbolo y timeframes específicos"""
    logger = setup_logger("data_cleaner")
    
    with db_manager.get_session() as session:
        from data.storage.models import OHLCVData
        
        total_deleted = 0
        for timeframe in timeframes:
            # Contar registros antes de eliminar
            count = session.query(OHLCVData).filter(
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe
            ).count()
            
            if count > 0:
                # Eliminar registros
                session.query(OHLCVData).filter(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timeframe == timeframe
                ).delete()
                
                logger.info(f"Deleted {count:,} existing records for {symbol} {timeframe}")
                total_deleted += count
            else:
                logger.info(f"No existing data found for {symbol} {timeframe}")
        
        logger.info(f"Total records deleted: {total_deleted:,}")

def get_timeframe_minutes(timeframe: str) -> int:
    """Convertir timeframe string a minutos"""
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440
    }
    return timeframe_minutes.get(timeframe, 15)

@click.command()
def show_data_status():
    """Mostrar estado actual de los datos en la base de datos"""
    logger = setup_logger("data_status")
    
    print("\n" + "="*60)
    print("DATABASE DATA STATUS")
    print("="*60)
    
    symbols = ['XRP/USDT']  # Expandir si tienes más símbolos
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    total_records = 0
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 40)
        
        for tf in timeframes:
            df = db_manager.get_ohlcv_data(symbol, tf)
            if not df.empty:
                count = len(df)
                total_records += count
                date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
                latest_price = df.iloc[-1]['close']
                print(f"  {tf:>4}: {count:>6,} records | {date_range} | ${latest_price:.4f}")
            else:
                print(f"  {tf:>4}: {'No data':>6}")
    
    print("\n" + "-"*60)
    print(f"Total records in database: {total_records:,}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Si se llama con 'status', mostrar estado de datos
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        show_data_status()
    else:
        collect_data()