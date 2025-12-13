"""
Marit Backtesting Library
=========================

A production-grade vectorized portfolio backtesting library.

Example usage:
    >>> from marit_backtesting import VectorBacktester
    >>> bt = VectorBacktester(returns_df, weights_df)
    >>> bt.calculate_equity(transaction_costs_bps=10)
    >>> metrics = bt.get_performance_metrics()
"""

from marit_backtesting.backtester import VectorBacktester

__version__ = "0.1.0"
__all__ = ["VectorBacktester"]
