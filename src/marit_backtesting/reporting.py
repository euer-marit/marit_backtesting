"""
Reporting module for VectorBacktester.

Provides interactive Plotly-based visualizations for strategy analysis.
Supports both inline display (Jupyter) and HTML file export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    from marit_backtesting.backtester import VectorBacktester


# Color palette - Modern, professional look
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'success': '#28A745',      # Green
    'danger': '#DC3545',       # Red
    'warning': '#FFC107',      # Yellow
    'info': '#17A2B8',         # Cyan
    'dark': '#343A40',         # Dark gray
    'light': '#F8F9FA',        # Light gray
    'gradient': ['#667eea', '#764ba2'],  # Purple gradient
}

# Asset color palette for multi-asset charts
ASSET_COLORS = px.colors.qualitative.Set2 if PLOTLY_AVAILABLE else []


def _check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly kaleido"
        )


def _show_or_save(fig: "go.Figure", output_path: str | None = None, show: bool = True) -> "go.Figure":
    """Display figure and/or save to file."""
    if output_path:
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")
    if show:
        fig.show()
    return fig


def plot_equity_curve(
    bt: "VectorBacktester",
    benchmark_equity: pd.Series | None = None,
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot interactive equity curve.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    benchmark_equity : pd.Series, optional
        Benchmark equity curve to compare against.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    fig = go.Figure()
    
    # Strategy equity
    fig.add_trace(go.Scatter(
        x=bt.equity_curve.index,
        y=bt.equity_curve.values,
        name="Strategy",
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate="<b>Strategy</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>",
    ))
    
    # Benchmark if provided
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            name="Benchmark",
            line=dict(color=COLORS['secondary'], width=2, dash='dot'),
            hovertemplate="<b>Benchmark</b><br>Date: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text="Equity Curve", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_drawdown(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot underwater (drawdown) chart.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    # Calculate drawdown
    rolling_max = bt.equity_curve.cummax()
    drawdown = (bt.equity_curve - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        name="Drawdown",
        line=dict(color=COLORS['danger'], width=1),
        fillcolor='rgba(220, 53, 69, 0.3)',
        hovertemplate="<b>Drawdown</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    
    fig.update_layout(
        title=dict(text="Underwater (Drawdown) Chart", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Drawdown [%]",
        template="plotly_white",
        height=400,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_trailing_returns(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot trailing returns (3M, 6M, 12M).
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    equity = bt.equity_curve
    
    # Calculate trailing returns
    trailing_3m = equity.pct_change(periods=3 * 21).dropna() * 100
    trailing_6m = equity.pct_change(periods=6 * 21).dropna() * 100
    trailing_12m = equity.pct_change(periods=12 * 21).dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trailing_3m.index, y=trailing_3m.values,
        name="3 Month",
        line=dict(color=COLORS['info'], width=1.5),
    ))
    
    fig.add_trace(go.Scatter(
        x=trailing_6m.index, y=trailing_6m.values,
        name="6 Month",
        line=dict(color=COLORS['warning'], width=1.5),
    ))
    
    fig.add_trace(go.Scatter(
        x=trailing_12m.index, y=trailing_12m.values,
        name="12 Month",
        line=dict(color=COLORS['primary'], width=2),
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    
    fig.update_layout(
        title=dict(text="Trailing Returns", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Return [%]",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_weights(
    bt: "VectorBacktester",
    freq: str = "ME",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot portfolio weights as aggregated stacked bar chart.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with weights.
    freq : str, optional
        Aggregation frequency: 'ME' (monthly), 'QE' (quarterly), 'YE' (yearly).
        Default is 'ME' (monthly).
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    weights = bt.weights.dropna()
    
    # Aggregate to reduce noise
    weights_agg = weights.resample(freq).mean()
    
    freq_labels = {'ME': 'Monthly', 'QE': 'Quarterly', 'YE': 'Yearly'}
    freq_label = freq_labels.get(freq, freq)
    
    fig = go.Figure()
    
    for i, col in enumerate(weights_agg.columns):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        fig.add_trace(go.Bar(
            x=weights_agg.index,
            y=weights_agg[col].values,
            name=col,
            marker_color=color,
            hovertemplate=f"<b>{col}</b><br>Period: %{{x}}<br>Avg Weight: %{{y:.1%}}<extra></extra>",
        ))
    
    fig.update_layout(
        barmode='stack',
        title=dict(text=f"{freq_label} Average Portfolio Weights", font=dict(size=20)),
        xaxis_title="Period",
        yaxis_title="Weight",
        yaxis=dict(tickformat='.0%'),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_asset_returns(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot cumulative returns of individual assets.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with weights and returns.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    # Calculate weighted asset returns
    weighted_returns = bt.weights * bt.returns
    asset_equity = (1 + weighted_returns).cumprod()
    
    fig = go.Figure()
    
    for i, col in enumerate(asset_equity.columns):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        fig.add_trace(go.Scatter(
            x=asset_equity.index,
            y=asset_equity[col].values,
            name=col,
            line=dict(width=2, color=color),
        ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=dict(text="Asset Contribution (Weighted)", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_monthly_heatmap(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot monthly returns heatmap.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    # Calculate monthly returns
    monthly = bt.equity_curve.resample('ME').last().pct_change().dropna() * 100
    monthly_df = monthly.to_frame(name='return')
    monthly_df['Year'] = monthly_df.index.year
    monthly_df['Month'] = monthly_df.index.month
    
    pivot = monthly_df.pivot(index='Year', columns='Month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return [%]"),
    ))
    
    fig.update_layout(
        title=dict(text="Monthly Returns Heatmap", font=dict(size=20)),
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
        height=max(400, len(pivot) * 30 + 100),
    )
    
    return _show_or_save(fig, output_path, show)


def plot_quarterly_heatmap(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot quarterly returns heatmap.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    # Calculate quarterly returns
    quarterly = bt.equity_curve.resample('QE').last().pct_change().dropna() * 100
    quarterly_df = quarterly.to_frame(name='return')
    quarterly_df['Year'] = quarterly_df.index.year
    quarterly_df['Quarter'] = quarterly_df.index.quarter
    
    pivot = quarterly_df.pivot(index='Year', columns='Quarter', values='return')
    pivot.columns = ['Q1', 'Q2', 'Q3', 'Q4'][:len(pivot.columns)]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Year: %{y}<br>Quarter: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return [%]"),
    ))
    
    fig.update_layout(
        title=dict(text="Quarterly Returns Heatmap", font=dict(size=20)),
        xaxis_title="Quarter",
        yaxis_title="Year",
        template="plotly_white",
        height=max(350, len(pivot) * 35 + 100),
    )
    
    return _show_or_save(fig, output_path, show)


def plot_yearly_returns(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot annual returns as bar chart.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    # Calculate yearly returns
    yearly = bt.equity_curve.resample('YE').last().pct_change().dropna() * 100
    
    colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in yearly.values]
    
    fig = go.Figure(data=go.Bar(
        x=yearly.index.year,
        y=yearly.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in yearly.values],
        textposition='outside',
        hovertemplate="Year: %{x}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    
    # Add average line
    avg_return = yearly.mean()
    fig.add_hline(
        y=avg_return, 
        line_dash="dash", 
        line_color=COLORS['info'],
        annotation_text=f"Avg: {avg_return:.1f}%",
        annotation_position="right",
    )
    
    fig.update_layout(
        title=dict(text="Annual Returns", font=dict(size=20)),
        xaxis_title="Year",
        yaxis_title="Return [%]",
        template="plotly_white",
        height=400,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_rolling_sharpe(
    bt: "VectorBacktester",
    window: int = 126,
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot rolling Sharpe ratio over time.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    window : int, optional
        Rolling window in trading days (default 126 = ~6 months).
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.net_returns is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    returns = bt.net_returns
    trading_days = bt.trading_days
    
    # Calculate rolling Sharpe
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean * np.sqrt(trading_days)) / rolling_std
    rolling_sharpe = rolling_sharpe.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        name=f"Rolling Sharpe ({window}d)",
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate="Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS['success'], 
                  annotation_text="Sharpe = 1.0", annotation_position="right")
    fig.add_hline(y=2, line_dash="dot", line_color=COLORS['info'],
                  annotation_text="Sharpe = 2.0", annotation_position="right")
    
    # Add fill color based on positive/negative
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.clip(lower=0).values,
        fill='tozeroy',
        fillcolor='rgba(40, 167, 69, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.clip(upper=0).values,
        fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    fig.update_layout(
        title=dict(text=f"Rolling Sharpe Ratio ({window}-day)", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        height=400,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_rolling_volatility(
    bt: "VectorBacktester",
    window: int = 21,
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot rolling volatility (annualized) over time.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    window : int, optional
        Rolling window in trading days (default 21 = ~1 month).
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.net_returns is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    returns = bt.net_returns
    trading_days = bt.trading_days
    
    # Calculate rolling volatility (annualized)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(trading_days) * 100
    rolling_vol = rolling_vol.dropna()
    
    # Calculate average volatility
    avg_vol = rolling_vol.mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        name=f"Rolling Vol ({window}d)",
        fill='tozeroy',
        line=dict(color=COLORS['warning'], width=2),
        fillcolor='rgba(255, 193, 7, 0.3)',
        hovertemplate="Date: %{x}<br>Vol: %{y:.1f}%<extra></extra>",
    ))
    
    fig.add_hline(
        y=avg_vol, 
        line_dash="dash", 
        line_color=COLORS['danger'],
        annotation_text=f"Avg: {avg_vol:.1f}%",
        annotation_position="right",
    )
    
    fig.update_layout(
        title=dict(text=f"Rolling Annualized Volatility ({window}-day)", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Volatility [%]",
        template="plotly_white",
        height=400,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_returns_distribution(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot returns distribution histogram with VaR markers.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    if bt.net_returns is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    returns = bt.net_returns * 100  # Convert to percentage
    
    # Calculate statistics
    var_5 = returns.quantile(0.05)
    var_1 = returns.quantile(0.01)
    cvar_5 = returns[returns <= var_5].mean()
    mean_ret = returns.mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=50,
        name="Daily Returns",
        marker_color=COLORS['primary'],
        opacity=0.7,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    
    # Add VaR lines
    fig.add_vline(x=var_5, line_dash="dash", line_color=COLORS['warning'], line_width=2,
                  annotation_text=f"VaR 5%: {var_5:.2f}%", annotation_position="top")
    fig.add_vline(x=var_1, line_dash="dash", line_color=COLORS['danger'], line_width=2,
                  annotation_text=f"VaR 1%: {var_1:.2f}%", annotation_position="bottom")
    fig.add_vline(x=mean_ret, line_dash="solid", line_color=COLORS['success'], line_width=2,
                  annotation_text=f"Mean: {mean_ret:.3f}%", annotation_position="top right")
    
    # Add CVaR annotation
    fig.add_annotation(
        x=cvar_5, y=0,
        text=f"CVaR 5%: {cvar_5:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['danger'],
        font=dict(color=COLORS['danger']),
    )
    
    fig.update_layout(
        title=dict(text="Daily Returns Distribution", font=dict(size=20)),
        xaxis_title="Daily Return [%]",
        yaxis_title="Frequency",
        template="plotly_white",
        height=450,
        showlegend=False,
    )
    
    return _show_or_save(fig, output_path, show)


def plot_risk_contribution(
    bt: "VectorBacktester",
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot risk contribution by asset (based on variance contribution).
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with weights and returns.
    output_path : str, optional
        Path to save HTML file.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    # Calculate average weights
    avg_weights = bt.weights.mean()
    
    # Calculate variance contribution (simplified)
    asset_vol = bt.returns.std() * np.sqrt(bt.trading_days)
    weighted_vol = avg_weights * asset_vol
    total_weighted_vol = weighted_vol.sum()
    risk_contrib = (weighted_vol / total_weighted_vol * 100).sort_values(ascending=True)
    
    fig = go.Figure()
    
    colors = [ASSET_COLORS[i % len(ASSET_COLORS)] for i in range(len(risk_contrib))]
    
    fig.add_trace(go.Bar(
        x=risk_contrib.values,
        y=risk_contrib.index,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in risk_contrib.values],
        textposition='outside',
        hovertemplate="Asset: %{y}<br>Risk Contrib: %{x:.1f}%<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text="Risk Contribution by Asset", font=dict(size=20)),
        xaxis_title="Risk Contribution [%]",
        yaxis_title="Asset",
        template="plotly_white",
        height=max(300, len(risk_contrib) * 40 + 100),
    )
    
    return _show_or_save(fig, output_path, show)


def generate_report(
    bt: "VectorBacktester",
    output_path: str = "backtest_report.html",
    show: bool = True,
) -> "go.Figure":
    """
    Generate comprehensive HTML report with all visualizations.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with calculated equity.
    output_path : str
        Path to save HTML report.
    show : bool
        Whether to display the chart inline.
    
    Returns
    -------
    plotly.graph_objects.Figure
        Combined dashboard figure.
    """
    _check_plotly()
    
    if bt.equity_curve is None:
        raise RuntimeError("Must call calculate_equity() first.")
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Equity Curve", "Drawdown",
            "Trailing Returns", "Portfolio Weights",
            "Monthly Returns", "Quarterly Returns",
            "Annual Returns", "Performance Metrics"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    
    equity = bt.equity_curve
    
    # 1. Equity Curve
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity", line=dict(color=COLORS['primary'], width=2),
        showlegend=False,
    ), row=1, col=1)
    
    # 2. Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill='tozeroy', name="Drawdown",
        line=dict(color=COLORS['danger'], width=1),
        fillcolor='rgba(220, 53, 69, 0.3)',
        showlegend=False,
    ), row=1, col=2)
    
    # 3. Trailing Returns
    trailing_6m = equity.pct_change(periods=6 * 21).dropna() * 100
    trailing_12m = equity.pct_change(periods=12 * 21).dropna() * 100
    fig.add_trace(go.Scatter(
        x=trailing_6m.index, y=trailing_6m.values,
        name="6M", line=dict(color=COLORS['warning'], width=1.5),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=trailing_12m.index, y=trailing_12m.values,
        name="12M", line=dict(color=COLORS['primary'], width=2),
    ), row=2, col=1)
    
    # 4. Weights (line chart for dashboard)
    weights = bt.weights.dropna()
    for i, col in enumerate(weights.columns[:5]):  # Limit to 5 for readability
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        fig.add_trace(go.Scatter(
            x=weights.index, y=weights[col].values,
            name=col, line=dict(color=color, width=1.5),
        ), row=2, col=2)
    
    # 5. Monthly Heatmap
    monthly = equity.resample('ME').last().pct_change().dropna() * 100
    monthly_df = monthly.to_frame(name='return')
    monthly_df['Year'] = monthly_df.index.year
    monthly_df['Month'] = monthly_df.index.month
    pivot_m = monthly_df.pivot(index='Year', columns='Month', values='return')
    
    fig.add_trace(go.Heatmap(
        z=pivot_m.values, x=list(range(1, 13)), y=pivot_m.index,
        colorscale='RdYlGn', zmid=0, showscale=False,
    ), row=3, col=1)
    
    # 6. Quarterly Heatmap
    quarterly = equity.resample('QE').last().pct_change().dropna() * 100
    quarterly_df = quarterly.to_frame(name='return')
    quarterly_df['Year'] = quarterly_df.index.year
    quarterly_df['Quarter'] = quarterly_df.index.quarter
    pivot_q = quarterly_df.pivot(index='Year', columns='Quarter', values='return')
    
    fig.add_trace(go.Heatmap(
        z=pivot_q.values, x=['Q1', 'Q2', 'Q3', 'Q4'][:len(pivot_q.columns)], y=pivot_q.index,
        colorscale='RdYlGn', zmid=0, showscale=False,
    ), row=3, col=2)
    
    # 7. Annual Returns
    yearly = equity.resample('YE').last().pct_change().dropna() * 100
    colors_bar = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in yearly.values]
    fig.add_trace(go.Bar(
        x=yearly.index.year, y=yearly.values,
        marker_color=colors_bar, showlegend=False,
    ), row=4, col=1)
    
    # 8. Metrics Table
    metrics = bt.get_performance_metrics()
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color=COLORS['primary'], 
                    font=dict(color='white', size=11), align='left'),
        cells=dict(
            values=[
                list(metrics.index)[:10],  # Limit rows
                [str(v) for v in metrics.values[:10]],
            ],
            fill_color=COLORS['light'],
            align='left',
            font=dict(size=10),
        ),
    ), row=4, col=2)
    
    fig.update_layout(
        title=dict(text="VectorBacktester Report", font=dict(size=24)),
        template="plotly_white",
        height=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.02, xanchor="center", x=0.5),
    )
    
    return _show_or_save(fig, output_path, show)
