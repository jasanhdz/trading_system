"""Binance Exchange implementation."""
import asyncio
import os
from typing import Any, Dict, List, Literal, Optional, Tuple
from binance import AsyncClient, BinanceSocketManager  # noqa: F401 (mantén si luego usarás websockets)
from binance.exceptions import BinanceAPIException

from ...core.ports.exchange import Exchange
from ...core.ports.logger import Logger
from ...core.types import Candle, OrderResult, PositionInfo, Side, SymbolFilters
from ..config import CONFIG
from ..rate_limit import (
    is_rate_limited,
    ms_until_reset,
    note_rate_limit_from_error,
)

def is_trueish(value: Any) -> bool:
    return value in (True, "true", "TRUE", 1, "1")

class BinanceExchange(Exchange):
    """Binance Futures exchange implementation."""
    def __init__(self, logger: Logger):
        self.log = logger
        self.client: Optional[AsyncClient] = None
        self.hedge_cache: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._request_lock = asyncio.Lock()
        self._next_request_at_ms = 0.0
        self._min_req_gap_ms = max(20, int(os.getenv("BINANCE_REQ_GAP_MS", "60")))
        self._candles_cache: Dict[tuple[str, str, int], Dict[str, Any]] = {}
        self._candle_cache_ttl_ms = int(os.getenv("BINANCE_CANDLE_CACHE_TTL_MS", "4000"))

    async def initialize(self):
        if self._initialized:
            return
        self.client = await AsyncClient.create(
            CONFIG.API_KEY,
            CONFIG.API_SECRET,
            testnet=CONFIG.IS_TESTNET
        )
        try:
            await self.client.futures_ping()
            self.log.info("binance_connected", {
                "net": "TESTNET" if CONFIG.IS_TESTNET else "PROD",
                "http": CONFIG.HTTP_FUTURES,
                "ws": CONFIG.WS_FUTURES,
            })
            self._initialized = True
        except Exception as e:
            self.log.error("binance_connect_error", {"err": str(e)})
            raise

    async def close(self):
        if self.client:
            await self.client.close_connection()
            self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def is_hedge_mode(self) -> bool:
        await self._ensure_initialized()
        if (self.hedge_cache
            and self.hedge_cache.get("at")
            and asyncio.get_event_loop().time() - self.hedge_cache["at"] < 60):
            return self.hedge_cache["value"]
        try:
            pm = await self._call(self.client.futures_get_position_mode)
            val = bool(pm.get("dualSidePosition", False))
            self.hedge_cache = {"value": val, "at": asyncio.get_event_loop().time()}
            return val
        except Exception:
            self.hedge_cache = {"value": False, "at": asyncio.get_event_loop().time()}
            return False

    @staticmethod
    def pos_side_mismatch(e: Exception) -> bool:
        msg = str(e).lower()
        return "positionside" in msg or "position side" in msg

    async def get_server_time(self) -> int:
        await self._ensure_initialized()
        result = await self._call(self.client.futures_time)
        return int(result.get("serverTime", 0))

    async def _enqueue(self, coro_factory):
        await self._ensure_initialized()
        async with self._request_lock:
            if is_rate_limited():
                wait_ms = ms_until_reset()
                if wait_ms > 0:
                    self.log.warn("binance_rate_limit_wait", {"waitMs": wait_ms})
                    await asyncio.sleep(wait_ms / 1000)

            now_ms = asyncio.get_event_loop().time() * 1000
            wait_ms = max(0.0, self._next_request_at_ms - now_ms)
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000)

            try:
                result = await coro_factory()
                return result
            except BinanceAPIException as err:
                until = note_rate_limit_from_error(err)
                if until is not None:
                    self._next_request_at_ms = max(self._next_request_at_ms, float(until))
                    self.log.warn("binance_rate_limited", {"until": until})
                raise
            finally:
                self._next_request_at_ms = max(
                    self._next_request_at_ms,
                    asyncio.get_event_loop().time() * 1000 + self._min_req_gap_ms,
                )

    async def _call(self, func, *args, **kwargs):
        return await self._enqueue(lambda: func(*args, **kwargs))

    async def get_candles(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        key: Tuple[str, str, int] = (symbol, interval, limit)
        now_ms = asyncio.get_event_loop().time() * 1000
        cached = self._candles_cache.get(key)
        if cached and now_ms - cached["ts"] < self._candle_cache_ttl_ms:
            candles = cached["data"]
            if len(candles) >= limit:
                return candles[-limit:]

        raw = await self._call(self.client.futures_klines, symbol=symbol, interval=interval, limit=limit)
        candles = [
            Candle(
                open_time=c[0],
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
                close_time=c[6],
            )
            for c in raw
        ]
        self._candles_cache[key] = {"ts": now_ms, "data": candles}
        return candles[-limit:]

    async def get_mark_price(self, symbol: str) -> float:
        """Query only what we need; support SDK returning dict or list."""
        res = await self._call(self.client.futures_mark_price, symbol=symbol)
        if isinstance(res, dict):
            return float(res["markPrice"])
        # Fallback (older clients might ignore symbol param)
        for m in res:
            if m["symbol"] == symbol:
                return float(m["markPrice"])
        raise ValueError(f"Mark price not found for {symbol}")

    async def get_usdt_balance(self) -> float:
        balances = await self._call(self.client.futures_account_balance)
        for b in balances:
            if b["asset"] == "USDT":
                return float(b.get("availableBalance", "0"))
        return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._call(self.client.futures_change_leverage, symbol=symbol, leverage=leverage)

    async def get_symbol_filters(self, symbol: str, leverage: int) -> SymbolFilters:
        recv_window = int(os.getenv("BINANCE_RECV_WINDOW", "20000"))
        info = await self._call(self.client.futures_leverage_bracket, symbol=symbol, recvWindow=recv_window)

        cap_tier = None
        for bracket_info in info:
            if bracket_info["symbol"] == symbol:
                for bracket in bracket_info.get("brackets", []):
                    if leverage <= int(bracket["initialLeverage"]):
                        cap_tier = bracket
                        break
                break

        ex_info = await self._call(self.client.futures_exchange_info)
        symbol_info = next((s for s in ex_info["symbols"] if s["symbol"] == symbol), None)
        if not symbol_info:
            raise ValueError(f"Symbol {symbol} not found in exchange info")

        price_filter = None
        lot_filter = None
        min_notional = 5.0

        for f in symbol_info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                price_filter = f
            elif f["filterType"] in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                # Si existe MARKET_LOT_SIZE, úsalo; si no, LOT_SIZE
                if not lot_filter or f["filterType"] == "MARKET_LOT_SIZE":
                    lot_filter = f
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", "5"))

        tick_size_str = price_filter["tickSize"] if price_filter else "0.0001"
        step_size_str = lot_filter["stepSize"] if lot_filter else "0.1"

        price_precision = len(tick_size_str.split(".")[-1]) if "." in tick_size_str else 0
        qty_precision = len(step_size_str.split(".")[-1]) if "." in step_size_str else 0

        return SymbolFilters(
            tick_size=float(tick_size_str),
            step_size=float(step_size_str),
            price_precision=price_precision,
            qty_precision=qty_precision,
            min_notional=min_notional,
            notional_cap=float(cap_tier["notionalCap"]) if cap_tier else None,
        )

    async def has_open_position(self, symbol: str, side: Literal["LONG", "SHORT", "ANY"]) -> bool:
        account = await self._call(self.client.futures_account)
        positions = account.get("positions", [])
        for p in positions:
            if p["symbol"] != symbol:
                continue
            amt = float(p["positionAmt"])
            if side == "ANY":
                return abs(amt) > 0
            if p["positionSide"] == "BOTH":
                return (side == "LONG" and amt > 0) or (side == "SHORT" and amt < 0)
            return p["positionSide"] == side and abs(amt) > 0
        return False

    async def read_active_position(self, symbol: str, side_hint: Side) -> Optional[PositionInfo]:
        account = await self._call(self.client.futures_account)
        positions = account.get("positions", [])
        for p in positions:
            if p["symbol"] != symbol:
                continue
            amt = float(p["positionAmt"])
            if p["positionSide"] == "BOTH":
                if not ((side_hint == Side.LONG and amt > 0) or (side_hint == Side.SHORT and amt < 0)):
                    continue
            elif p["positionSide"] != side_hint.value or abs(amt) == 0:
                continue
            fallback_lev = CONFIG.get_symbol_config(symbol).leverage
            return PositionInfo(
                side_mode=p["positionSide"],
                qty_abs=abs(amt),
                entry_price=float(p["entryPrice"]),
                leverage=float(p.get("leverage", fallback_lev)),
            )
        return None

    async def market_open(self, symbol: str, side: Side, quantity: float) -> OrderResult:
        hedge = await self.is_hedge_mode()
        params = {
            "symbol": symbol,
            "type": "MARKET",
            "quantity": str(quantity),
            "newOrderRespType": "RESULT",
            "side": "BUY" if side == Side.LONG else "SELL",
        }
        if hedge:
            params["positionSide"] = side.value

        t0 = asyncio.get_event_loop().time()
        try:
            res = await self._call(self.client.futures_create_order, **params)
            self.log.debug("api_market_open", {"ms": int((asyncio.get_event_loop().time() - t0) * 1000),
                                               "symbol": symbol, "side": side.value, "qty": quantity})
            return OrderResult(
                order_id=str(res["orderId"]),
                avg_price=float(res.get("avgPrice", 0)),
                executed_qty=float(res.get("executedQty", quantity)),
                status=res.get("status", ""),
                side=side,
                type="MARKET",
            )
        except BinanceAPIException as e:
            if self.pos_side_mismatch(e):
                params.pop("positionSide", None)
                res = await self._call(self.client.futures_create_order, **params)
                self.hedge_cache = None
                self.log.warn("api_market_open_fallback", {"symbol": symbol, "side": side.value, "qty": quantity})
                return OrderResult(
                    order_id=str(res["orderId"]),
                    avg_price=float(res.get("avgPrice", 0)),
                    executed_qty=float(res.get("executedQty", quantity)),
                    status=res.get("status", ""),
                    side=side,
                    type="MARKET",
                )
            raise

    async def place_stop_close(self, symbol: str, side: Side, stop_price: float) -> None:
        hedge = await self.is_hedge_mode()
        params = {
            "symbol": symbol,
            "type": "STOP_MARKET",
            "side": "SELL" if side == Side.LONG else "BUY",
            "stopPrice": str(stop_price),
            "closePosition": True,
            "workingType": "MARK_PRICE",
        }
        if hedge:
            params["positionSide"] = side.value

        t0 = asyncio.get_event_loop().time()
        try:
            await self._call(self.client.futures_create_order, **params)
            self.log.debug("api_stop_upsert", {"ms": int((asyncio.get_event_loop().time() - t0) * 1000),
                                               "symbol": symbol, "side": side.value, "stopPrice": stop_price})
        except BinanceAPIException as e:
            if self.pos_side_mismatch(e):
                params.pop("positionSide", None)
                await self._call(self.client.futures_create_order, **params)
                self.hedge_cache = None
                self.log.warn("api_stop_upsert_fallback", {"symbol": symbol, "side": side.value, "stopPrice": stop_price})
                return
            raise

    async def place_tp_close(self, symbol: str, side: Side, trigger_price: float) -> None:
        hedge = await self.is_hedge_mode()
        params = {
            "symbol": symbol,
            "type": "TAKE_PROFIT_MARKET",
            "side": "SELL" if side == Side.LONG else "BUY",
            "stopPrice": str(trigger_price),
            "closePosition": True,
            "workingType": "MARK_PRICE",
        }
        if hedge:
            params["positionSide"] = side.value

        t0 = asyncio.get_event_loop().time()
        try:
            await self._call(self.client.futures_create_order, **params)
            self.log.debug("api_tp_upsert", {"ms": int((asyncio.get_event_loop().time() - t0) * 1000),
                                             "symbol": symbol, "side": side.value, "tp": trigger_price})
        except BinanceAPIException as e:
            if self.pos_side_mismatch(e):
                params.pop("positionSide", None)
                await self._call(self.client.futures_create_order, **params)
                self.hedge_cache = None
                self.log.warn("api_tp_upsert_fallback", {"symbol": symbol, "side": side.value, "tp": trigger_price})
                return
            raise

    async def close_side_market_safe(
        self, symbol: str, side: Side, qty_abs: float, side_mode: Literal["BOTH", "LONG", "SHORT"],
    ) -> None:
        params = {
            "symbol": symbol,
            "type": "MARKET",
            "quantity": str(qty_abs),
            "newOrderRespType": "RESULT",
            "side": "SELL" if side == Side.LONG else "BUY",
        }
        try:
            if side_mode == "BOTH":
                await self._call(self.client.futures_create_order, **params)
            else:
                params["positionSide"] = side.value
                params["reduceOnly"] = True
                await self._call(self.client.futures_create_order, **params)
        except BinanceAPIException as e:
            if self.pos_side_mismatch(e):
                params.pop("positionSide", None)
                params.pop("reduceOnly", None)
                await self._call(self.client.futures_create_order, **params)
                self.hedge_cache = None
                return
            msg = str(e).lower()
            if "reduceonly" in msg or "reduce only" in msg:
                params.pop("reduceOnly", None)
                await self._call(self.client.futures_create_order, **params)
                return
            raise

    async def open_stop_for_side(self, symbol: str, side: Side) -> Optional[Dict[str, Any]]:
        orders = await self.list_close_orders_for_side(symbol, side)
        stops = [o for o in orders if o["type"] in ("STOP_MARKET", "STOP")]
        if not stops:
            return None
        pick = max(stops, key=lambda x: x["stopPrice"]) if side == Side.LONG else min(stops, key=lambda x: x["stopPrice"])
        return {"stopPrice": pick["stopPrice"], "orderId": pick["orderId"]}

    async def cancel_order_by_id(self, symbol: str, order_id: str) -> None:
        await self._call(self.client.futures_cancel_order, symbol=symbol, orderId=int(order_id))

    async def cancel_orders_by_ids(self, symbol: str, order_ids: List[str]) -> None:
        for oid in order_ids:
            try:
                await self.cancel_order_by_id(symbol, oid)
            except Exception as e:
                self.log.warn("cancel_order_fail", {"id": oid, "err": str(e)})

    async def cancel_close_orders_for_side(self, symbol: str, side: Side) -> None:
        open_orders = await self._call(self.client.futures_get_open_orders, symbol=symbol)
        for o in open_orders:
            if (o["type"] in ("STOP_MARKET", "TAKE_PROFIT_MARKET")
                and is_trueish(o.get("closePosition"))
                and (not o.get("positionSide") or o["positionSide"] == side.value)):
                await self._call(self.client.futures_cancel_order, symbol=symbol, orderId=o["orderId"])

    async def read_liquidation_price(self, symbol: str, side: Side) -> Optional[float]:
        risks = await self._call(self.client.futures_position_risk)
        for r in risks:
            if r["symbol"] != symbol:
                continue
            amt = float(r["positionAmt"])
            if r["positionSide"] == "BOTH":
                if not ((side == Side.LONG and amt > 0) or (side == Side.SHORT and amt < 0)):
                    continue
            elif r["positionSide"] == side.value and abs(amt) == 0:
                continue
            liq = float(r.get("liquidationPrice", 0))
            return liq if liq > 0 else None
        return None

    def _order_side_for_position(self, side: Side) -> str:
        return "SELL" if side == Side.LONG else "BUY"

    async def list_close_orders_for_side(self, symbol: str, side: Side) -> List[Dict[str, Any]]:
        open_orders = await self._call(self.client.futures_get_open_orders, symbol=symbol)
        want_side = self._order_side_for_position(side)

        self.log.debug("raw_open_orders", {
            "count": len(open_orders),
            "sample": [
                {
                    "id": o["orderId"],
                    "type": o["type"],
                    "side": o["side"],
                    "positionSide": o.get("positionSide"),
                    "closePosition": o.get("closePosition"),
                    "reduceOnly": o.get("reduceOnly"),
                    "stopPrice": o.get("stopPrice"),
                    "workingType": o.get("workingType"),
                }
                for o in open_orders[:5]
            ],
        })

        result = []
        for o in open_orders:
            order_type = o["type"]
            if order_type not in ("STOP_MARKET", "STOP", "TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                continue
            if o["side"] != want_side:
                continue
            pos_side = o.get("positionSide")
            if pos_side and pos_side not in (side.value, "BOTH"):
                continue
            stop_price = float(o.get("stopPrice", 0) or 0)
            result.append({"orderId": str(o["orderId"]), "type": order_type, "stopPrice": stop_price})
        return result

    async def open_tp_for_side(self, symbol: str, side: Side) -> Optional[Dict[str, Any]]:
        orders = await self.list_close_orders_for_side(symbol, side)
        tps = [o for o in orders if o["type"] in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")]
        if not tps:
            return None
        # Elegimos cualquiera válido; podrías optimizar a “más cercano al precio objetivo”
        pick = tps[0]
        return {"stopPrice": pick["stopPrice"], "orderId": pick["orderId"]}
    
    async def read_liquidation_price(self, symbol: str, side: Side) -> Optional[float]:
        """
        Lee el precio de liquidación de la posición. Compatibilidad con SDKs que
        NO tienen futures_position_risk: usa futures_position_information.
        """
        risks = None
        # 1) Intento con .futures_position_risk (si existe en tu SDK)
        try:
            if hasattr(self.client, "futures_position_risk"):
                # algunos SDKs aceptan symbol=..., otros ignoran y devuelven todo
                risks = await self._call(self.client.futures_position_risk)
        except AttributeError:
            risks = None
        except Exception:
            # Si falla por otro motivo, pasamos al fallback
            risks = None

        # 2) Fallback robusto: .futures_position_information (presente en AsyncClient moderno)
        if not risks:
            risks = await self._call(self.client.futures_position_information, symbol=symbol)

        # Normaliza lista
        if isinstance(risks, dict):
            risks = [risks]
        if not isinstance(risks, list):
            return None

        # Selección por símbolo y side
        pick = None
        for r in risks:
            try:
                if r.get("symbol") != symbol:
                    continue
                amt = float(r.get("positionAmt", 0) or 0)
                if abs(amt) == 0:
                    continue

                pos_side = r.get("positionSide", "BOTH")
                if pos_side == "BOTH":
                    # oneway: usa signo del amount
                    if (side == Side.LONG and amt <= 0) or (side == Side.SHORT and amt >= 0):
                        continue
                else:
                    # hedge: debe coincidir con el lado
                    if pos_side != side.value:
                        continue

                pick = r
                break
            except Exception:
                continue

        if not pick:
            return None

        try:
            liq = float(pick.get("liquidationPrice", 0) or 0)
            return liq if liq > 0 else None
        except Exception:
            return None
