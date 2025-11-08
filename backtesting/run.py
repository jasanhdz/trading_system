import argparse
from datetime import datetime, timedelta, timezone

from backtesting.generic_engine import GenericBacktester, EngineParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, help="Ruta con puntos a la clase, e.g. pkg.mod.Class")
    p.add_argument("--symbol", default="XRP/USDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--warmup", type=int, default=300)
    args = p.parse_args()

    # UTC-aware → UTC-naive para el motor (evita warnings de deprecación)
    end_aware = datetime.now(timezone.utc)
    start_aware = end_aware - timedelta(days=args.days)
    end = end_aware.replace(tzinfo=None)
    start = start_aware.replace(tzinfo=None)

    params = EngineParams(
        symbol=args.symbol,
        timeframe=args.timeframe,
        leverage=args.leverage,
        days=args.days,
        warmup_bars=args.warmup,
        start=start,
        end=end,
    )
    bt = GenericBacktester(strategy_dotted=args.strategy, params=params)
    df_signals = bt.run()

    print(f"\nSeñales generadas: {len(df_signals)}")
    if not df_signals.empty:
        print(df_signals.tail(50).to_string())


if __name__ == "__main__":
    main()
