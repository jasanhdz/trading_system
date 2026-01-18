#!/usr/bin/env python3
"""
Risk Engine (Bloodbath V3 - Phase 4)
Implements Dynamic Leverage Sizing using the Kelly Criterion.
Translates Model Confidence (Probability) into Capital Allocation.
"""
import numpy as np
import pandas as pd

class RiskEngine:
    def __init__(self, max_leverage=20.0, kelly_fraction=0.25, risk_free_rate=0.0):
        """
        Args:
            max_leverage: Maximum allowed leverage (e.g., 20x).
            kelly_fraction: Safety factor (e.g., 0.25 = Quarter Kelly).
                            Full Kelly is too risky for crypto.
            risk_free_rate: Theoretical risk-free rate (usually 0 for crypto trading context).
        """
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.risk_free_rate = risk_free_rate
        
    def calculate_leverage(self, probability, payoff_ratio=3.0):
        """
        Calculates optimal leverage using Kelly Criterion.
        
        Kelly Formula: f* = p - q/b
        Where:
            f* = Fraction of bankroll to wager
            p  = Probability of winning (Model Confidence)
            q  = Probability of losing (1 - p)
            b  = Payoff ratio (Win Amount / Loss Amount)
            
        Args:
            probability (float): Model's predicted probability of a crash (0.0 to 1.0).
            payoff_ratio (float): Expected Risk/Reward ratio of the trade. 
                                  For "Crash Catching", this is usually high (e.g., 1:3 or 1:5).
                                  Defaulting to 3.0 (conservative estimate for a good crash).
                                  
        Returns:
            float: Recommended Leverage (0.0 to max_leverage).
        """
        if probability <= 0.5:
            return 0.0
            
        # Kelly Calculation
        q = 1.0 - probability
        b = payoff_ratio
        
        f_star = probability - (q / b)
        
        # If f_star is negative (edge is not enough), bet 0
        if f_star <= 0:
            return 0.0
            
        # Apply Safety Fraction (Quarter Kelly is standard for pro traders)
        safe_f = f_star * self.kelly_fraction
        
        # Convert Bankroll Fraction to Leverage
        # In crypto futures, Leverage = Position Size / Collateral
        # If Kelly says "bet 20% of bankroll" on a trade with 1% margin requirement, that's 20x leverage?
        # No. Kelly gives the % of *equity* to risk.
        # If we define "Risk" as the Stop Loss distance.
        # Let's simplify for the Thesis:
        # We treat "Leverage" as a direct multiplier of exposure.
        # If Kelly says "Invest 100% of bankroll", that's 1x Leverage.
        # If Kelly says "Invest 200% of bankroll", that's 2x Leverage.
        
        # Wait, Kelly usually assumes binary bets where you lose the whole wager.
        # In trading, we don't lose the whole account on one trade (hopefully).
        # We use Stop Loss.
        # Position Size = (Risk % * Equity) / (Stop Loss %)
        # Leverage = Position Size / Equity = Risk % / Stop Loss %
        
        # Let's assume a fixed Stop Loss for the "Bloodbath" strategy.
        # In a crash, volatility is high. Let's say Stop Loss is 2%.
        STOP_LOSS_PCT = 0.02
        
        # Kelly f* is the % of Capital to RISK.
        # So Position Size = (safe_f * Equity) / STOP_LOSS_PCT
        # Leverage = Position Size / Equity = safe_f / STOP_LOSS_PCT
        
        leverage = safe_f / STOP_LOSS_PCT
        
        # Cap at Max Leverage
        leverage = min(leverage, self.max_leverage)
        
        # Round to 1 decimal
        return round(leverage, 1)

    def simulate_curve(self):
        """
        Generates a simulation of Leverage vs Probability for visualization.
        """
        probs = np.linspace(0, 1.0, 100)
        levs = [self.calculate_leverage(p) for p in probs]
        return pd.DataFrame({'Probability': probs, 'Leverage': levs})

if __name__ == "__main__":
    # Test Case
    engine = RiskEngine(max_leverage=20.0, kelly_fraction=0.25) # Quarter Kelly
    
    print("--- Risk Engine Test (Quarter Kelly) ---")
    print(f"Max Leverage: {engine.max_leverage}x")
    print(f"Assumed Stop Loss: 2%")
    print(f"Assumed Payoff (R:R): 1:3\n")
    
    test_probs = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
    for p in test_probs:
        lev = engine.calculate_leverage(p)
        print(f"Model Confidence: {p*100:>3.0f}% -> Recommended Leverage: {lev:>4.1f}x")
