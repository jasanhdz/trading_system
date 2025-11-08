# =============================================================================
# analysis/statistics/visualizations.py - VERSIÓN CORREGIDA
# =============================================================================
"""
Sistema de visualizaciones para EDA
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional

# Import correcto de la clase EDA
from analysis.statistics.exploratory import ExploratoryDataAnalysis

class TradingVisualizations:
    """Clase para generar visualizaciones de trading"""
    
    def __init__(self, eda_results: ExploratoryDataAnalysis):
        self.eda = eda_results
        
    def plot_price_overview(self, timeframe: str = '1h') -> go.Figure:
        """Gráfico de precios con volumen"""
        if timeframe not in self.eda.data:
            raise ValueError(f"Timeframe {timeframe} not available in data")
            
        df = self.eda.data[timeframe]
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=['Price', 'Volume'],
            vertical_spacing=0.1
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['volume'], 
                name='Volume', 
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{self.eda.symbol} - {timeframe.upper()} Price & Volume",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_returns_distribution(self, timeframe: str = '1h') -> go.Figure:
        """Análisis completo de distribución de returns"""
        if timeframe not in self.eda.data:
            raise ValueError(f"Timeframe {timeframe} not available in data")
            
        df = self.eda.data[timeframe]
        returns = df['returns'].dropna()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Returns Distribution', 
                'Cumulative Returns', 
                'Rolling Volatility (20-period)', 
                'Returns vs Volume'
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Histogram of returns
        fig.add_trace(
            go.Histogram(
                x=returns, 
                nbinsx=50, 
                name='Returns',
                histnorm='probability density',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        mu, std = returns.mean(), returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = ((1 / (std * np.sqrt(2 * np.pi))) * 
                  np.exp(-0.5 * ((x_norm - mu) / std) ** 2))
        
        fig.add_trace(
            go.Scatter(
                x=x_norm, 
                y=y_norm, 
                name='Normal Dist', 
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index, 
                y=cum_returns, 
                name='Cumulative Returns',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # 3. Rolling volatility
        vol = returns.rolling(20).std()
        fig.add_trace(
            go.Scatter(
                x=vol.index, 
                y=vol, 
                name='20-period Vol',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # 4. Returns vs Volume scatter
        # Resample volume to match returns frequency if needed
        volume_aligned = df['volume'].reindex(returns.index, method='ffill')
        
        fig.add_trace(
            go.Scatter(
                x=volume_aligned,
                y=returns,
                mode='markers',
                name='Returns vs Volume',
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=returns,
                    colorscale='RdYlBu',
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Returns Analysis - {timeframe.upper()} - {self.eda.symbol}",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Returns", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Volume", row=2, col=2)
        fig.update_yaxes(title_text="Returns", row=2, col=2)
        
        return fig
    
    def plot_correlation_matrix(self) -> Optional[go.Figure]:
        """Heatmap de correlaciones entre timeframes"""
        if 'correlations' not in self.eda.results:
            print("No correlation analysis found. Run eda.correlation_analysis() first.")
            return None
            
        corr_data = self.eda.results['correlations']['returns_correlation_matrix']
        corr_matrix = pd.DataFrame(corr_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Returns Correlation Matrix Across Timeframes',
            height=500,
            width=600,
            xaxis_title="Timeframes",
            yaxis_title="Timeframes"
        )
        
        return fig
    
    def plot_volatility_regimes(self, timeframe: str = '1h') -> go.Figure:
        """Visualización de regímenes de volatilidad"""
        if timeframe not in self.eda.data:
            raise ValueError(f"Timeframe {timeframe} not available in data")
            
        df = self.eda.data[timeframe]
        returns = df['returns'].dropna()
        vol = returns.rolling(20).std()
        
        # Definir regímenes de volatilidad
        vol_median = vol.median()
        high_vol = vol > vol_median
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=['Price with Volatility Regimes', 'Volatility'],
            vertical_spacing=0.1
        )
        
        # Price chart with volatility coloring
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add background coloring for high volatility periods
        for i, (idx, is_high_vol) in enumerate(high_vol.items()):
            if is_high_vol and i < len(high_vol) - 1:
                next_idx = high_vol.index[i + 1] if i + 1 < len(high_vol) else idx
                fig.add_vrect(
                    x0=idx, x1=next_idx,
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0,
                    row=1, col=1
                )
        
        # Volatility plot
        colors = ['red' if v > vol_median else 'green' for v in vol]
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol,
                mode='lines',
                name='Volatility',
                line=dict(color='orange'),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Add median line
        fig.add_hline(
            y=vol_median,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {vol_median:.6f}",
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Volatility Regimes Analysis - {timeframe.upper()}',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_temporal_patterns(self, timeframe: str = '1h') -> go.Figure:
        """Visualización de patrones temporales"""
        if 'temporal' not in self.eda.results or timeframe not in self.eda.results['temporal']:
            print("No temporal analysis found. Run eda.temporal_patterns() first.")
            return None
            
        temporal_data = self.eda.results['temporal'][timeframe]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Mean Returns by Hour', 
                'Volatility by Hour',
                'Mean Returns by Day of Week',
                'Volume by Hour'
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Returns by hour
        hourly_returns = temporal_data['hourly_patterns']['mean_returns_by_hour']
        hours = list(hourly_returns.keys())
        returns_values = list(hourly_returns.values())
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=returns_values,
                name='Hourly Returns',
                marker_color=['green' if r > 0 else 'red' for r in returns_values]
            ),
            row=1, col=1
        )
        
        # 2. Volatility by hour
        hourly_vol = temporal_data['hourly_patterns']['volatility_by_hour']
        vol_values = list(hourly_vol.values())
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=vol_values,
                name='Hourly Volatility',
                marker_color='orange'
            ),
            row=1, col=2
        )
        
        # 3. Returns by day of week
        daily_returns = temporal_data['daily_patterns']['mean_returns_by_day']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_values = [daily_returns.get(i, 0) for i in range(7)]
        
        fig.add_trace(
            go.Bar(
                x=days,
                y=daily_values,
                name='Daily Returns',
                marker_color=['green' if r > 0 else 'red' for r in daily_values]
            ),
            row=2, col=1
        )
        
        # 4. Volume by hour
        hourly_volume = temporal_data['hourly_patterns']['volume_by_hour']
        volume_values = list(hourly_volume.values())
        
        fig.add_trace(
            go.Bar(
                x=hours,
                y=volume_values,
                name='Hourly Volume',
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Temporal Patterns Analysis - {timeframe.upper()}',
            height=800,
            showlegend=False
        )
        
        return fig

# =============================================================================
# Función de utilidad para crear visualizaciones rápidamente
# =============================================================================

def create_quick_visualizations(eda: ExploratoryDataAnalysis, timeframe: str = '1h'):
    """Crear visualizaciones rápidas para análisis"""
    viz = TradingVisualizations(eda)
    
    print(f"Creating visualizations for {timeframe}...")
    
    # Price overview
    fig1 = viz.plot_price_overview(timeframe)
    fig1.show()
    
    # Returns distribution
    fig2 = viz.plot_returns_distribution(timeframe)
    fig2.show()
    
    # Volatility regimes
    fig3 = viz.plot_volatility_regimes(timeframe)
    fig3.show()
    
    print("Visualizations created successfully!")