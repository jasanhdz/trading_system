# =============================================================================
# DATA/COLLECTORS/BINANCE_COLLECTOR.PY - Versión Corregida
# =============================================================================
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import time
import logging

from data.collectors.base_collector import BaseDataCollector
from config.settings import settings
from utils.exceptions import DataCollectionError

logger = logging.getLogger(__name__)

class BinanceDataCollector(BaseDataCollector):
    def __init__(self):
        super().__init__("binance")
        self.exchange = None
        self.rate_limiter = time.time()
    
    def connect(self) -> bool:
        """Conectar a Binance"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': False,  # True for testnet
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Para futuros
                }
            })
            
            # Test connection
            markets = self.exchange.load_markets()
            self.logger.info(f"Connected to Binance. Found {len(markets)} markets")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    def _rate_limit(self):
        """Implementar rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.rate_limiter
        if time_diff < settings.REQUEST_DELAY:
            time.sleep(settings.REQUEST_DELAY - time_diff)
        self.rate_limiter = time.time()

    def _candidate_symbols(self, symbol: str) -> List[str]:
        """Devuelve variantes posibles para el símbolo solicitado."""
        candidates = [symbol]
        if ":" not in symbol and "/" in symbol:
            base, quote = symbol.split("/", 1)
            quote = quote.upper()
            colon_variant = f"{base}/{quote}:{quote}"
            if colon_variant not in candidates:
                candidates.append(colon_variant)
        return candidates

    def _resolve_market_symbol(self, symbol: str) -> str:
        """Encuentra el símbolo que CCXT reconoce para el mercado solicitado."""
        if not self.exchange:
            if not self.connect():
                raise DataCollectionError("Cannot connect to Binance")

        try:
            if not getattr(self.exchange, "markets", None):
                self.exchange.load_markets()
        except Exception as exc:
            raise DataCollectionError(f"Failed to load markets: {exc}") from exc

        candidates = self._candidate_symbols(symbol)
        for candidate in candidates:
            if candidate in self.exchange.symbols:
                if candidate != symbol:
                    self.logger.debug(
                        "Resolved market symbol variant",
                        extra={"requested": symbol, "resolved": candidate},
                    )
                return candidate
        raise DataCollectionError(
            f"Symbol {symbol} not available on Binance futures. Tried variants: {candidates}"
        )
    
    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: datetime = None, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtener datos OHLCV de Binance"""
        if not self.exchange:
            if not self.connect():
                raise DataCollectionError("Cannot connect to Binance")
        
        try:
            market_symbol = self._resolve_market_symbol(symbol)
            self._rate_limit()
            
            # Convertir datetime a timestamp en ms
            since_ms = None
            if since:
                # Asegurar que since tenga timezone UTC
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
                since_ms = int(since.timestamp() * 1000)
            
            # Obtener datos
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=market_symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=min(limit, 1000)  # Binance limit
            )
            
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Formatear datos
            df = self.format_ohlcv_data(ohlcv)
            
            if not self.validate_data(df):
                raise DataCollectionError(f"Invalid data received for {symbol}")
            
            self.logger.info(
                f"Collected {len(df)} candles for {symbol} {timeframe}",
                extra={"market_symbol": market_symbol},
            )
            return df
            
        except ccxt.BaseError as e:
            self.logger.error(f"CCXT error collecting {symbol} {timeframe}: {e}")
            raise DataCollectionError(f"CCXT error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error collecting {symbol} {timeframe}: {e}")
            raise DataCollectionError(f"Unexpected error: {e}")
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Obtener datos históricos completos con manejo correcto de timezones"""
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        # Asegurar que las fechas tengan timezone UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        all_data = []
        current_date = start_date
        
        # Calcular tamaño de chunk según timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        
        if timeframe not in timeframe_minutes:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # 1000 candles por request
        chunk_size = timedelta(minutes=timeframe_minutes[timeframe] * 1000)
        
        self.logger.info(f"Collecting historical data for {symbol} {timeframe} from {start_date} to {end_date}")
        
        while current_date < end_date:
            chunk_end = min(current_date + chunk_size, end_date)
            
            try:
                df_chunk = self.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=1000
                )
                
                if not df_chunk.empty:
                    # CORRECCIÓN: Convertir timestamps a timezone-aware para comparación
                    # Asegurar que df_chunk['timestamp'] sea timezone-aware
                    if df_chunk['timestamp'].dt.tz is None:
                        df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_localize('UTC')
                    
                    # Convertir current_date y chunk_end a pandas Timestamp con timezone
                    current_date_pd = pd.Timestamp(current_date).tz_convert('UTC')
                    chunk_end_pd = pd.Timestamp(chunk_end).tz_convert('UTC')
                    
                    # Filtrar datos dentro del rango
                    df_chunk = df_chunk[
                        (df_chunk['timestamp'] >= current_date_pd) & 
                        (df_chunk['timestamp'] < chunk_end_pd)
                    ]
                    
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                        
                        # Actualizar current_date al último timestamp + 1 periodo
                        last_timestamp = df_chunk['timestamp'].max()
                        # Convertir back to datetime for next iteration
                        current_date = last_timestamp.to_pydatetime() + timedelta(minutes=timeframe_minutes[timeframe])
                    else:
                        current_date = chunk_end
                else:
                    current_date = chunk_end
                
                # Progress logging
                if (end_date - start_date).total_seconds() > 0:
                    progress = ((current_date - start_date) / (end_date - start_date)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Current date: {current_date}")
                
            except Exception as e:
                self.logger.error(f"Error collecting chunk from {current_date}: {e}")
                current_date += chunk_size  # Skip problematic chunk
                continue
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.drop_duplicates(subset=['timestamp'], inplace=True)
            final_df.sort_values('timestamp', inplace=True)
            final_df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Collected total {len(final_df)} candles for {symbol} {timeframe}")
            return final_df
        else:
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Obtener símbolos disponibles en futuros"""
        if not self.exchange:
            if not self.connect():
                return []
        
        try:
            markets = self.exchange.load_markets()
            future_symbols = [
                symbol for symbol, market in markets.items()
                if market.get('type') == 'future' and market.get('active')
            ]
            return future_symbols
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Obtener datos adicionales del mercado"""
        if not self.exchange:
            if not self.connect():
                return {}
        
        try:
            self._rate_limit()
            
            # Ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Funding rate (si está disponible)
            funding_rate = None
            try:
                funding_info = self.exchange.fetch_funding_rate(symbol)
                funding_rate = funding_info.get('fundingRate')
            except:
                pass
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'mark_price': ticker.get('last'),
                'index_price': ticker.get('index'),
                'funding_rate': funding_rate,
                'open_interest': ticker.get('info', {}).get('openInterest'),
                'volume_24h': ticker.get('baseVolume'),
                'price_change_24h': ticker.get('change')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
