"""Position sizing and risk management calculations."""

from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from ..types import SymbolFilters


def floor_to_step(value: float, step: float, precision: int) -> float:
    """Floor value to step size."""
    if step <= 0:
        return 0.0
    
    steps = int(value / step)
    result = steps * step
    return round(result, precision)


def ceil_to_step(value: float, step: float, precision: int) -> float:
    """Ceil value to step size."""
    if step <= 0:
        return 0.0
    
    steps = int(value / step)
    if value > steps * step:
        steps += 1
    
    result = steps * step
    return round(result, precision)


def size_by_budget(
    usdt_balance: float,
    reserve: float,
    capital_pct: float,
    price: float,
    leverage: int,
    fee_pct: float,
    filters: SymbolFilters,
) -> Dict[str, Any]:
    """
    Calculate position size based on budget constraints.
    
    Args:
        usdt_balance: Available USDT balance
        reserve: Minimum wallet reserve in USDT
        capital_pct: Percentage of capital to use (0.0 to 1.0)
        price: Current price
        leverage: Leverage to use
        fee_pct: Fee buffer percentage
        filters: Symbol trading filters
        
    Returns:
        Dictionary with sizing information
    """
    # Available capital after reserve
    available = usdt_balance - reserve
    if available <= 0:
        return {
            "qty": 0,
            "reason": "insufficient_balance",
            "balance": usdt_balance,
            "reserve": reserve,
        }
    
    # Apply capital usage percentage
    capital_pct = max(0.0, min(1.0, capital_pct))
    target_capital = available * capital_pct

    # Account for fees
    capital_after_fees = target_capital * (1 - fee_pct)
    
    # Calculate position value with leverage
    position_value = capital_after_fees * leverage
    
    # Convert to quantity
    qty = position_value / price
    
    # Apply step size filter
    qty = floor_to_step(qty, filters.step_size, filters.qty_precision)
    
    # Check minimum notional
    notional = qty * price
    if notional < filters.min_notional:
        return {
            "qty": 0,
            "reason": "min_notional_not_met",
            "notional": notional,
            "minNotional": filters.min_notional,
        }
    
    # Check notional cap if exists
    if filters.notional_cap and notional > filters.notional_cap:
        # Reduce to cap
        qty = filters.notional_cap / price
        qty = floor_to_step(qty, filters.step_size, filters.qty_precision)
    
    notional = qty * price
    if notional == 0:
        return {
            "qty": 0,
            "reason": "qty_zero_after_filters",
        }

    effective_capital = notional / max(1, leverage)
    capital_pct_used = effective_capital / available if available > 0 else 0.0

    return {
        "qty": qty,
        "notional": notional,
        "capital": effective_capital,
        "capital_requested": target_capital,
        "leverage": leverage,
        "capital_pct_requested": capital_pct,
        "capital_pct_used": capital_pct_used,
    }


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    leverage: int = 1,
    filters: Optional[SymbolFilters] = None,
) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        balance: Account balance in USDT
        risk_percent: Risk percentage per trade (e.g., 1.0 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        leverage: Leverage to use
        filters: Symbol trading filters
        
    Returns:
        Position size in base currency
    """
    if balance <= 0 or risk_percent <= 0:
        return 0.0
    
    # Calculate stop distance as percentage
    stop_distance = abs(entry_price - stop_price) / entry_price
    
    if stop_distance == 0:
        return 0.0
    
    # Risk amount in USDT
    risk_amount = balance * (risk_percent / 100)
    
    # Position value (notional)
    position_value = risk_amount / stop_distance
    
    # Apply leverage
    position_value_with_leverage = position_value / leverage
    
    # Convert to quantity
    quantity = position_value_with_leverage / entry_price
    
    # Apply filters if provided
    if filters:
        quantity = floor_to_step(quantity, filters.step_size, filters.qty_precision)
        
        # Check minimum notional
        notional = quantity * entry_price
        if notional < filters.min_notional:
            # Try to meet minimum notional
            min_qty = filters.min_notional / entry_price
            min_qty = ceil_to_step(min_qty, filters.step_size, filters.qty_precision)
            
            # Check if min quantity exceeds risk
            min_notional = min_qty * entry_price
            min_risk = (min_notional / leverage) * stop_distance
            
            if min_risk <= risk_amount:
                quantity = min_qty
            else:
                # Cannot meet minimum notional within risk
                quantity = 0.0
    
    return quantity


def validate_quantity(
    quantity: float,
    price: float,
    filters: SymbolFilters,
) -> bool:
    """
    Validate quantity against exchange filters.
    
    Args:
        quantity: Quantity to validate
        price: Price for notional calculation
        filters: Symbol trading filters
        
    Returns:
        True if quantity is valid
    """
    if quantity <= 0:
        return False
    
    # Check step size alignment
    steps = quantity / filters.step_size
    if abs(steps - round(steps)) > 1e-9:
        return False
    
    # Check minimum notional
    notional = quantity * price
    if notional < filters.min_notional:
        return False
    
    # Check notional cap
    if filters.notional_cap and notional > filters.notional_cap:
        return False
    
    # Check min/max quantity
    if filters.min_qty and quantity < filters.min_qty:
        return False
    
    if filters.max_qty and quantity > filters.max_qty:
        return False
    
    return True


def apply_quantity_filters(qty: float, price: float, filters: SymbolFilters) -> float:
    """
    Ajusta la cantidad de contratos/tokens según step_size y min_notional.
    - Redondea al múltiplo de step_size.
    - Verifica que cumpla con min_notional.
    """
    step = filters.step_size
    min_notional = filters.min_notional

    # Redondear al múltiplo más cercano del step_size
    qty_adj = (qty // step) * step

    # Asegurar que no caiga en 0
    if qty_adj <= 0:
        qty_adj = step

    # Verificar min_notional
    notional = qty_adj * price
    if notional < min_notional:
        # sube al mínimo permitido
        qty_adj = (min_notional // price) * step

    return round(qty_adj, filters.qty_precision)

def kelly_criterion(win_rate: float, reward_risk: float) -> float:
    """
    Calcula el factor de Kelly (fracción de capital a arriesgar).
    
    win_rate: probabilidad de ganar (0.0 - 1.0)
    reward_risk: ratio de recompensa/riesgo (ej: 2.0 significa reward 2x risk)
    """
    q = 1 - win_rate
    f = (win_rate * (reward_risk + 1) - 1) / reward_risk
    return max(0.0, f)  # nunca negativo
