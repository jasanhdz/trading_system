"""Main bot runner with scheduled execution."""
import math
import asyncio
import time
from typing import Any, Dict

from .strategy_runner import StrategyRunner
from .guards.sync_state import sync_state_guard
from .guards.ensure_brackets import brackets_guard
from .guards.intelligent_take_profit import intelligent_take_profit
from .guards.profit_guard import enforce_profit_guard


class Bot:
    """Main bot with guard system."""
    
    def __init__(
        self,
        runner: StrategyRunner,
        symbol: str,
        exchange: Any,
        state: Any,
        logger: Any,
        interval_sec: int = 5,
    ):
        """Initialize bot."""
        self.runner = runner
        self.symbol = symbol
        self.exchange = exchange
        self.state = state
        self.logger = logger
        self.interval_sec = interval_sec
        self.running = False
        self.seq = 0
    
    async def tick(self) -> None:
        """Execute one bot tick."""
        if self.running:
            return
        
        self.running = True
        
        try:
            # Heartbeat time drift check (every 12 ticks)
            if self.seq % 12 == 1:
                try:
                    server_time = await self.exchange.get_server_time()
                    drift_ms = server_time - int(time.time() * 1000)
                    self.logger.debug("time_drift", {"driftMs": drift_ms})
                except Exception as e:
                    self.logger.warn("time_drift_fail", {"err": str(e)})
            
            # Execute guards
            # await sync_state_guard(self.symbol, self.exchange, self.state, self.logger)
            # await brackets_guard(self.symbol, self.exchange, self.state, self.logger)
            # await enforce_profit_guard(self.symbol, self.exchange, self.state, self.logger)
            # await intelligent_take_profit(self.symbol, self.exchange, self.state, self.logger)

            # Execute strategy
            await self.runner.tick(self.symbol)
            
            self.seq += 1
            
        except Exception as e:
            self.logger.error("tick_error", {"err": str(e)})
        finally:
            self.running = False
    
    async def run(self) -> None:
        """
        Ejecuta el bot a ritmo fijo (fixed-rate):
        - Programa cada tick cada `interval_sec` exactos, compensando la duración del tick.
        - Se ancla a la pared de tiempo (múltiplos exactos de los 5s: :00, :05, :10, …).
        """
        loop = asyncio.get_running_loop()

        # (Opcional) primer tick inmediato para arrancar como antes:
        await self.tick()

        # Alinear al siguiente múltiplo de interval_sec del reloj de pared
        now_wall = time.time()
        next_wall = math.ceil(now_wall / self.interval_sec) * self.interval_sec
        await asyncio.sleep(max(0.0, next_wall - now_wall))

        # Usamos reloj monotónico para evitar saltos si cambia el reloj del SO
        next_target = loop.time()

        while True:
            t_start = loop.time()
            await self.tick()

            # Programar el siguiente instante objetivo a ritmo fijo
            next_target += self.interval_sec

            # Si el tick se tardó “demasiado”, no acumules retraso infinito: reancla
            # (por ejemplo, si nos pasamos más de 2 intervalos, reprograma a +intervalo desde ahora)
            now = loop.time()
            if next_target < now - self.interval_sec:
                next_target = now + self.interval_sec

            sleep_for = max(0.0, next_target - now)
            await asyncio.sleep(sleep_for)


def start_bot(
    runner: StrategyRunner,
    symbol: str,
    exchange: Any,
    state: Any,
    logger: Any,
    interval_sec: int = 5,
) -> Bot:
    """Create and start bot."""
    bot = Bot(
        runner=runner,
        symbol=symbol,
        exchange=exchange,
        state=state,
        logger=logger,
        interval_sec=interval_sec,
    )
    return bot
