"""
Marit Backtesting Library
=========================

A production-grade vectorized portfolio backtesting library.

Example usage:
    >>> from marit_backtesting import VectorBacktester
    >>> bt = VectorBacktester(returns_df, weights_df)
    >>> bt.calculate_equity(transaction_costs_bps=10)
    >>> metrics = bt.get_performance_metrics()
    >>> bt.report()  # Generate interactive HTML report
"""

from marit_backtesting.backtester import VectorBacktester
from marit_backtesting.reporting import (
    plot_equity_curve,
    plot_drawdown,
    plot_trailing_returns,
    plot_weights,
    plot_asset_returns,
    plot_monthly_heatmap,
    plot_quarterly_heatmap,
    plot_yearly_returns,
    generate_report,
)

__version__ = "0.2.0"
__all__ = [
    "VectorBacktester",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_trailing_returns",
    "plot_weights",
    "plot_asset_returns",
    "plot_monthly_heatmap",
    "plot_quarterly_heatmap",
    "plot_yearly_returns",
    "generate_report",
]
