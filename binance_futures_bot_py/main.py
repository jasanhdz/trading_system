"""Main entry point for the trading bot."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.infra.binance.exchange import BinanceExchange
from src.infra.fs.state_store import FsStateStore
from src.infra.fs.logger import FsLogger
from src.infra.config import CONFIG
from src.app.strategy_runner import StrategyRunner
from src.app.bot import start_bot
from src.strategies.ml_probability import MLProbabilityStrategy

STATE_ROOT = Path(__file__).parent / "state"


async def run_symbol(symbol: str, exchange: BinanceExchange, base_logger: FsLogger) -> None:
    """Launch a bot instance for a single symbol."""
    symbol_logger = base_logger.bind(symbol=symbol)
    state_dir = STATE_ROOT / symbol.lower()
    state = FsStateStore(str(state_dir))
    symbol_settings = CONFIG.get_symbol_config(symbol)

    artifacts = {
        "model": symbol_settings.model_path,
        "scaler": symbol_settings.scaler_path,
        "meta": symbol_settings.meta_path,
    }
    missing = {name: str(path) for name, path in artifacts.items() if not path.exists()}
    if missing:
        symbol_logger.warn(
            "ml_artifacts_missing",
            {"missing": missing, "timeframe": symbol_settings.timeframe},
        )
        return

    strategy = MLProbabilityStrategy(history_bars=CONFIG.ML_HISTORY_BARS)

    runner = StrategyRunner(
        exchange=exchange,
        logger=symbol_logger,
        state=state,
        strategy=strategy,
        config=CONFIG,
        symbol_config=symbol_settings,
    )

    bot = start_bot(
        runner=runner,
        symbol=symbol,
        exchange=exchange,
        state=state,
        logger=symbol_logger,
        interval_sec=CONFIG.BOT_INTERVAL_SEC,
    )

    symbol_logger.info(
        "bot_started",
        {
            "symbol": symbol,
            "strategy": strategy.name,
            "leverage": symbol_settings.leverage,
            "timeframe": symbol_settings.timeframe,
            "testnet": CONFIG.IS_TESTNET,
        },
    )

    await bot.run()


async def main() -> None:
    """Main bot entry point."""
    base_logger = FsLogger()
    exchange = BinanceExchange(base_logger)

    symbols = CONFIG.SYMBOLS or [CONFIG.SYMBOL]
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    tasks: List[asyncio.Task[None]] = []

    try:
        await exchange.initialize()
        base_logger.info(
            "bot_manager_start",
            {"symbols": symbols, "testnet": CONFIG.IS_TESTNET},
        )

        for idx, symbol in enumerate(symbols):
            task = asyncio.create_task(
                run_symbol(symbol, exchange, base_logger),
                name=f"bot::{symbol}",
            )
            tasks.append(task)

            if idx < len(symbols) - 1 and CONFIG.BOT_STAGGER_MS > 0:
                await asyncio.sleep(CONFIG.BOT_STAGGER_MS / 1000.0)

        if tasks:
            await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        base_logger.info("bot_stopped", {"reason": "user_interrupt"})
    except Exception as exc:
        base_logger.error("bot_error", {"error": str(exc)})
        raise
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await exchange.close()
        base_logger.info("bot_manager_stop", {"reason": "shutdown"})


if __name__ == "__main__":
    asyncio.run(main())
