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
    output_path: str | None = None,
    show: bool = True,
) -> "go.Figure":
    """
    Plot portfolio weights over time as stacked area chart.
    
    Parameters
    ----------
    bt : VectorBacktester
        Backtester instance with weights.
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
    
    fig = go.Figure()
    
    for i, col in enumerate(weights.columns):
        color = ASSET_COLORS[i % len(ASSET_COLORS)]
        fig.add_trace(go.Scatter(
            x=weights.index,
            y=weights[col].values,
            name=col,
            mode='lines',
            stackgroup='one',
            line=dict(width=0.5, color=color),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Weight: %{{y:.2%}}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text="Portfolio Weights Over Time", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis=dict(tickformat='.0%'),
        template="plotly_white",
        hovermode="x unified",
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
