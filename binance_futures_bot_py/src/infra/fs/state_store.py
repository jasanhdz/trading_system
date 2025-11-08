"""File system state store implementation."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
import aiofiles.os

from ...core.ports.state_store import StateStore
from ...core.types import BotMode, BotState, Side, Trade


class FsStateStore(StateStore):
    """File system state persistence."""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "bot_state.json"
        self.trades_file = self.state_dir / "trades.json"
        
        # Create directory synchronously
        self.state_dir.mkdir(exist_ok=True)
    
    async def get(self) -> BotState:
        """Alias usado por tu app: devuelve IDLE si no hay estado."""
        st = await self.load_state()
        if st is None:
            st = BotState(mode=BotMode.IDLE)
        return st

    async def set(self, state: BotState) -> None:
        """Alias usado por tu app."""
        await self.save_state(state)
    
    async def load_state(self) -> Optional[BotState]:
        """Load bot state from storage."""
        if not await aiofiles.os.path.exists(self.state_file):
            return None
        
        try:
            async with aiofiles.open(self.state_file, 'r') as f:
                data = await f.read()
                state_dict = json.loads(data)
                
                # Convert string enums back to enum types
                return BotState(
                    mode=BotMode(state_dict['mode']),
                    last_side=Side(state_dict['last_side']) if state_dict.get('last_side') else None,
                    last_entry_price=state_dict.get('last_entry_price'),
                    last_leverage=state_dict.get('last_leverage'),
                    last_entry_at=state_dict.get('last_entry_at'),
                    peak_roe=state_dict.get('peak_roe'),
                    last_tp_at=state_dict.get('last_tp_at'),
                    last_exit_reason=state_dict.get('last_exit_reason'),
                    last_exit_at=state_dict.get('last_exit_at'),
                    brackets_armed_at=state_dict.get('brackets_armed_at'),
                    pos_side_mode=state_dict.get('pos_side_mode'),
                    last_entry_qty=state_dict.get('last_entry_qty'),
                    pyramid_units=state_dict.get('pyramid_units'),
                    last_pyramid_price=state_dict.get('last_pyramid_price'),
                    last_trail_stop=state_dict.get('last_trail_stop'),
                    brackets_attached=state_dict.get('brackets_attached'),
                    last_intelli_tp_at=state_dict.get('last_intelli_tp_at'),
                    intelli_tp_state=state_dict.get('intelli_tp_state'),
                )
                
        except Exception as e:
            # Log error but don't crash - return None
            print(f"Error loading state: {e}")
            return None
    
    async def save_state(self, state: BotState) -> None:
        """Save bot state to storage."""
        # Create backup first
        if await aiofiles.os.path.exists(self.state_file):
            backup_file = self.state_file.with_suffix('.json.bak')
            async with aiofiles.open(self.state_file, 'rb') as src:
                async with aiofiles.open(backup_file, 'wb') as dst:
                    await dst.write(await src.read())
        
        # Convert to dict, handling enums
        state_dict = {
            'mode': state.mode.value,
            'last_side': state.last_side.value if state.last_side else None,
            'last_entry_price': state.last_entry_price,
            'last_leverage': state.last_leverage,
            'last_entry_at': state.last_entry_at,
            'peak_roe': state.peak_roe,
            'last_tp_at': state.last_tp_at,
            'last_exit_reason': state.last_exit_reason,
            'last_exit_at': state.last_exit_at,
            'brackets_armed_at': state.brackets_armed_at,
            'pos_side_mode': state.pos_side_mode,
            'last_entry_qty': state.last_entry_qty,
            'pyramid_units': state.pyramid_units,
            'last_pyramid_price': state.last_pyramid_price,
            'last_trail_stop': state.last_trail_stop,
            'brackets_attached': state.brackets_attached,
            'last_intelli_tp_at': state.last_intelli_tp_at,
            'intelli_tp_state': state.intelli_tp_state,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        # Write atomically
        temp_file = self.state_file.with_suffix('.json.tmp')
        async with aiofiles.open(temp_file, 'w') as f:
            await f.write(json.dumps(state_dict, indent=2))
        
        # Replace original file
        await aiofiles.os.rename(temp_file, self.state_file)
    
    async def append_trade(self, trade: Trade) -> None:
        """Append trade to history."""
        trades = await self.get_trades()
        
        # Convert trade to dict
        trade_dict = {
            'side': trade.side.value,
            'entry_idx': trade.entry_idx,
            'entry_ts': trade.entry_ts,
            'entry_px': trade.entry_px,
            'exit_idx': trade.exit_idx,
            'exit_ts': trade.exit_ts,
            'exit_px': trade.exit_px,
            'exit': trade.exit,
            'bars_held': trade.bars_held,
            'pnl_pct': trade.pnl_pct,
            'mfe_pct': trade.mfe_pct,
            'mae_pct': trade.mae_pct,
            'reason': trade.reason,
            'adx': trade.adx,
            'v_ratio': trade.v_ratio,
            'bb_upper': trade.bb_upper,
            'bb_lower': trade.bb_lower,
            'dist_top_pct': trade.dist_top_pct,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        trades.append(trade_dict)
        
        # Save all trades
        async with aiofiles.open(self.trades_file, 'w') as f:
            await f.write(json.dumps(trades, indent=2))
    
    async def get_trades(self) -> List[Trade]:
        """Get all trades."""
        if not await aiofiles.os.path.exists(self.trades_file):
            return []
        
        try:
            async with aiofiles.open(self.trades_file, 'r') as f:
                data = await f.read()
                trade_dicts = json.loads(data)
                
                trades = []
                for td in trade_dicts:
                    # Skip the saved_at field when creating Trade
                    td.pop('saved_at', None)
                    
                    trades.append(Trade(
                        side=Side(td['side']),
                        entry_idx=td['entry_idx'],
                        entry_ts=td['entry_ts'],
                        entry_px=td['entry_px'],
                        exit_idx=td['exit_idx'],
                        exit_ts=td['exit_ts'],
                        exit_px=td['exit_px'],
                        exit=td['exit'],
                        bars_held=td['bars_held'],
                        pnl_pct=td['pnl_pct'],
                        mfe_pct=td['mfe_pct'],
                        mae_pct=td['mae_pct'],
                        reason=td.get('reason'),
                        adx=td.get('adx'),
                        v_ratio=td.get('v_ratio'),
                        bb_upper=td.get('bb_upper'),
                        bb_lower=td.get('bb_lower'),
                        dist_top_pct=td.get('dist_top_pct')
                    ))
                
                return trades
                
        except Exception as e:
            print(f"Error loading trades: {e}")
            return []
    
    async def clear_trades(self) -> None:
        """Clear all trades (useful for testing)."""
        if await aiofiles.os.path.exists(self.trades_file):
            async with aiofiles.open(self.trades_file, 'w') as f:
                await f.write("[]")
    
    async def get_statistics(self) -> dict:
        """Calculate trading statistics."""
        trades = await self.get_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_pnl': 0.0
            }
        
        # Calculate statistics
        winning_trades = [t for t in trades if t.pnl_pct > 0]
        losing_trades = [t for t in trades if t.pnl_pct < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = sum(t.pnl_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(abs(t.pnl_pct) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        gross_profit = sum(t.pnl_pct for t in winning_trades)
        gross_loss = sum(abs(t.pnl_pct) for t in losing_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate cumulative returns for drawdown
        cumulative_returns = []
        cumsum = 0
        for trade in trades:
            cumsum += trade.pnl_pct
            cumulative_returns.append(cumsum)
        
        # Max drawdown
        peak = 0
        max_dd = 0
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            dd = (peak - ret) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd * 100,
            'total_pnl': sum(t.pnl_pct for t in trades),
            'avg_trade': sum(t.pnl_pct for t in trades) / len(trades),
            'avg_bars_held': sum(t.bars_held for t in trades) / len(trades)
        }
