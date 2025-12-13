"""
VectorBacktester - Production-grade vectorized portfolio backtesting.

This module provides the VectorBacktester class for backtesting portfolio
strategies using vectorized operations for optimal performance.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marit_backtesting.utils import (
    calculate_annualized_return,
    calculate_calmar_ratio,
    calculate_drawdown_series,
    calculate_max_drawdown_duration,
    calculate_sharpe_ratio,
    calculate_skewness,
    calculate_sortino_ratio,
    safe_divide,
)


class VectorBacktester:
    """
    A vectorized portfolio backtester for efficient strategy analysis.
    
    This class takes daily asset returns and portfolio weights, performs
    data alignment, calculates equity curves with transaction costs, and
    computes comprehensive performance metrics.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily asset returns with datetime index and asset tickers as columns.
        Values should be decimal returns (e.g., 0.01 for 1%).
    weights_df : pd.DataFrame
        Target portfolio weights with datetime index and asset tickers as columns.
        Weights should sum to approximately 1.0 per row for long-only portfolios.
    trading_days : int, optional
        Number of trading days per year for annualization (default 252).
    
    Attributes
    ----------
    returns : pd.DataFrame
        Aligned and cleaned returns DataFrame.
    weights : pd.DataFrame
        Aligned and cleaned weights DataFrame.
    equity_curve : pd.Series or None
        Equity curve after running calculate_equity().
    daily_returns : pd.Series or None
        Daily portfolio returns after running calculate_equity().
    
    Examples
    --------
    >>> import pandas as pd
    >>> returns_df = pd.DataFrame({'SPY': [0.01, -0.005, 0.02]})
    >>> weights_df = pd.DataFrame({'SPY': [1.0, 1.0, 1.0]})
    >>> bt = VectorBacktester(returns_df, weights_df)
    >>> bt.calculate_equity(transaction_costs_bps=10)
    >>> print(bt.get_performance_metrics())
    """
    
    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        trading_days: int = 252,
    ) -> None:
        """Initialize the backtester with data alignment and cleaning."""
        self.trading_days = trading_days
        self._validate_inputs(returns_df, weights_df)
        
        # Align DataFrames by datetime index (inner join)
        common_dates = returns_df.index.intersection(weights_df.index)
        common_cols = returns_df.columns.intersection(weights_df.columns)
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between returns and weights DataFrames.")
        if len(common_cols) == 0:
            raise ValueError("No overlapping columns between returns and weights DataFrames.")
        
        # Align and sort
        self.returns = returns_df.loc[common_dates, common_cols].sort_index()
        self.weights = weights_df.loc[common_dates, common_cols].sort_index()
        
        # Handle missing values
        self.weights = self.weights.ffill()  # Forward-fill weights
        self.weights = self.weights.fillna(0.0)  # Fill any remaining NaNs
        self.returns = self.returns.fillna(0.0)  # Fill missing returns with 0
        
        # Results (calculated later)
        self.equity_curve: pd.Series | None = None
        self.daily_returns: pd.Series | None = None
        self.net_returns: pd.Series | None = None
        self.turnover: pd.Series | None = None
        self.transaction_costs_applied: float = 0.0
        
        # Benchmark data (set via compare_benchmark)
        self._benchmark_returns: pd.Series | None = None
        self._benchmark_equity: pd.Series | None = None
    
    def _validate_inputs(self, returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> None:
        """Validate input DataFrames."""
        if not isinstance(returns_df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame.")
        if not isinstance(weights_df, pd.DataFrame):
            raise TypeError("weights_df must be a pandas DataFrame.")
        
        if not isinstance(returns_df.index, pd.DatetimeIndex):
            try:
                returns_df.index = pd.to_datetime(returns_df.index)
            except Exception as e:
                raise ValueError(f"returns_df index must be convertible to datetime: {e}")
        
        if not isinstance(weights_df.index, pd.DatetimeIndex):
            try:
                weights_df.index = pd.to_datetime(weights_df.index)
            except Exception as e:
                raise ValueError(f"weights_df index must be convertible to datetime: {e}")
    
    def calculate_equity(
        self,
        transaction_costs_bps: float = 2.0,
        initial_capital: float = 1.0,
        borrowing_fee_rate: float = 0.0,
        margin_spread: float = 0.0,
        fed_fund_rate_df: pd.DataFrame | None = None,
        default_cash_rate: float = 0.05,
    ) -> pd.Series:
        """
        Calculate the portfolio equity curve with comprehensive fee structure.
        
        Parameters
        ----------
        transaction_costs_bps : float, optional
            Transaction costs in basis points per turnover (default 2.0).
            For example, 10 bps = 0.10% per unit of turnover.
        initial_capital : float, optional
            Starting portfolio value (default 1.0).
        borrowing_fee_rate : float, optional
            Annual borrowing fee rate for short positions (default 0.0).
            For example, 0.012 = 1.2% annual borrowing cost.
        margin_spread : float, optional
            Spread above the risk-free rate for margin/cash usage (default 0.0).
            For example, 1.5 means margin rate = fed_fund_rate + 1.5%.
        fed_fund_rate_df : pd.DataFrame, optional
            DataFrame with 'FEDFUNDS' column containing the Fed Funds rate (in %).
            Index should be datetime. If None, uses default_cash_rate.
        default_cash_rate : float, optional
            Default annual cash/margin rate when fed_fund_rate_df is not provided
            (default 0.05 = 5%).
        
        Returns
        -------
        pd.Series
            The cumulative equity curve indexed by date.
        
        Notes
        -----
        Fee structure:
        
        1. **Transaction costs**: turnover * transaction_costs_bps / 10000
        
        2. **Borrowing fees** (for short positions):
           daily_borrowing_fee = |sum(negative_weights)| * borrowing_fee_rate / 252
        
        3. **Margin/Cash fees** (for leveraged positions):
           leverage_amount = |sum(abs(weights))| - 1  (excess over 100% allocation)
           daily_margin_fee = leverage_amount * (fed_funds_rate + margin_spread) / 252 / 100
        
        Examples
        --------
        >>> # Simple backtest with just transaction costs
        >>> bt.calculate_equity(transaction_costs_bps=10)
        
        >>> # Full fee structure with margin costs
        >>> fed_funds = pd.read_csv('FEDFUNDS.csv', index_col='date', parse_dates=True)
        >>> bt.calculate_equity(
        ...     transaction_costs_bps=2,
        ...     borrowing_fee_rate=0.012,
        ...     margin_spread=1.5,
        ...     fed_fund_rate_df=fed_funds
        ... )
        """
        self.transaction_costs_applied = transaction_costs_bps
        self._borrowing_fee_rate = borrowing_fee_rate
        self._margin_spread = margin_spread
        
        # Calculate gross daily portfolio returns
        # Element-wise multiplication and sum across assets
        gross_returns = (self.weights * self.returns).sum(axis=1)
        
        # ===== 1. Transaction Costs =====
        # Calculate turnover (sum of absolute weight changes)
        weight_changes = self.weights.diff().abs()
        self.turnover = weight_changes.sum(axis=1)
        self.turnover.iloc[0] = self.weights.iloc[0].abs().sum()  # Initial allocation
        
        # Transaction costs based on turnover
        transaction_costs = self.turnover * (transaction_costs_bps / 10000)
        
        # ===== 2. Borrowing Fees (for short positions) =====
        # Sum of absolute values of negative weights (short exposure)
        short_exposure = self.weights.where(self.weights < 0, 0).abs().sum(axis=1)
        borrowing_fees = short_exposure * (borrowing_fee_rate / 252)
        
        # ===== 3. Margin/Cash Fees (for leveraged positions) =====
        # Calculate leverage amount: how much we exceed 100% allocation
        total_exposure = self.weights.abs().sum(axis=1)
        leverage_amount = (total_exposure - 1).clip(lower=0)  # Only charge when leveraged
        
        # Get the applicable rate
        if fed_fund_rate_df is not None and 'FEDFUNDS' in fed_fund_rate_df.columns:
            # Align fed funds rate with weights index
            aligned_rate = self.weights.join(
                fed_fund_rate_df[['FEDFUNDS']], 
                how='left'
            ).ffill().bfill()['FEDFUNDS']
            
            # Fed funds is in %, margin_spread is in percentage points
            # Total rate = (FEDFUNDS + margin_spread) / 100 for annual rate
            cash_rate = (aligned_rate + margin_spread) / 100
        else:
            # Use default cash rate
            cash_rate = default_cash_rate + (margin_spread / 100)
        
        margin_fees = leverage_amount * cash_rate / 252
        
        # ===== Total Fees =====
        total_fees = transaction_costs + borrowing_fees + margin_fees
        
        # Store fee components for analysis
        self._fee_breakdown = pd.DataFrame({
            'transaction_costs': transaction_costs,
            'borrowing_fees': borrowing_fees,
            'margin_fees': margin_fees,
            'total_fees': total_fees,
        })
        
        # Calculate net returns
        self.net_returns = gross_returns - total_fees
        self.daily_returns = self.net_returns.copy()
        
        # Calculate equity curve using cumulative product
        # geo_ret = 1 + net_returns
        # equity = initial_capital * cumprod(geo_ret)
        self.equity_curve = initial_capital * (1 + self.net_returns).cumprod()
        
        return self.equity_curve
    
    def get_fee_breakdown(self) -> pd.DataFrame:
        """
        Get the breakdown of fees applied during the backtest.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: transaction_costs, borrowing_fees, 
            margin_fees, and total_fees. Each row represents daily fees.
        
        Raises
        ------
        RuntimeError
            If calculate_equity() has not been called first.
        """
        if not hasattr(self, '_fee_breakdown') or self._fee_breakdown is None:
            raise RuntimeError("Must call calculate_equity() before get_fee_breakdown().")
        return self._fee_breakdown
    
    def get_total_fees(self) -> dict:
        """
        Get the total fees paid over the backtest period.
        
        Returns
        -------
        dict
            Dictionary with total fees by category and overall total.
        """
        if not hasattr(self, '_fee_breakdown') or self._fee_breakdown is None:
            raise RuntimeError("Must call calculate_equity() before get_total_fees().")
        
        breakdown = self._fee_breakdown
        return {
            'transaction_costs': breakdown['transaction_costs'].sum(),
            'borrowing_fees': breakdown['borrowing_fees'].sum(),
            'margin_fees': breakdown['margin_fees'].sum(),
            'total_fees': breakdown['total_fees'].sum(),
            'avg_daily_fee_bps': breakdown['total_fees'].mean() * 10000,
        }
    
    def get_performance_metrics(
        self,
        risk_free_rate: float = 0.0,
    ) -> pd.Series:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        risk_free_rate : float, optional
            Annual risk-free rate for Sharpe/Sortino calculations (default 0.0).
        
        Returns
        -------
        pd.Series
            A Series containing all performance metrics.
        
        Raises
        ------
        RuntimeError
            If calculate_equity() has not been called first.
        
        Notes
        -----
        Metrics included:
        
        **Period/Values:**
        - Start Period, End Period
        - Start Value, End Value
        
        **Returns:**
        - Total Return [%]
        - Annual Return [%] (CAGR)
        
        **Risk/Drawdown:**
        - Max Drawdown [%]
        - Average Daily Drawdown [%]
        - Max Drawdown Duration (days)
        - Standard Deviation [%] (annualized)
        
        **Ratios:**
        - Sharpe Ratio
        - Sortino Ratio
        - Calmar Ratio
        
        **Distribution:**
        - Skew
        - Lower Tail (5th percentile)
        - Upper Tail (95th percentile)
        """
        if self.equity_curve is None or self.net_returns is None:
            raise RuntimeError("Must call calculate_equity() before get_performance_metrics().")
        
        equity = self.equity_curve
        returns = self.net_returns
        n_days = len(returns)
        
        # Period/Values
        start_period = equity.index[0].strftime("%Y-%m-%d")
        end_period = equity.index[-1].strftime("%Y-%m-%d")
        start_value = equity.iloc[0] / (1 + returns.iloc[0])  # Value before first return
        end_value = equity.iloc[-1]
        
        # Returns
        total_return = (end_value / start_value) - 1
        annual_return = calculate_annualized_return(total_return, n_days, self.trading_days)
        
        # Risk/Drawdown
        drawdown = calculate_drawdown_series(equity)
        max_drawdown = abs(drawdown.min())
        avg_daily_drawdown = abs(drawdown.mean())
        max_dd_duration = calculate_max_drawdown_duration(drawdown)
        
        # Annualized volatility
        daily_std = returns.std()
        annual_std = daily_std * np.sqrt(self.trading_days)
        
        # Ratios
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, self.trading_days)
        sortino = calculate_sortino_ratio(returns, risk_free_rate, self.trading_days)
        calmar = calculate_calmar_ratio(annual_return, max_drawdown)
        
        # Distribution stats
        skew = calculate_skewness(returns)
        lower_tail = returns.quantile(0.05)
        upper_tail = returns.quantile(0.95)
        
        metrics = {
            # Period/Values
            "Start Period": start_period,
            "End Period": end_period,
            "Start Value": round(start_value, 4),
            "End Value": round(end_value, 4),
            
            # Returns
            "Total Return [%]": round(total_return * 100, 2),
            "Annual Return [%]": round(annual_return * 100, 2),
            
            # Risk/Drawdown
            "Max Drawdown [%]": round(max_drawdown * 100, 2),
            "Avg Daily Drawdown [%]": round(avg_daily_drawdown * 100, 2),
            "Max Drawdown Duration (days)": max_dd_duration,
            "Std Deviation [%] (Annual)": round(annual_std * 100, 2),
            
            # Ratios
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Calmar Ratio": round(calmar, 3),
            
            # Distribution
            "Skew": round(skew, 3),
            "Lower Tail (5%)": round(lower_tail * 100, 4),
            "Upper Tail (95%)": round(upper_tail * 100, 4),
        }
        
        return pd.Series(metrics)
    
    def compare_benchmark(
        self,
        benchmark: str | pd.Series,
        generate_tearsheet: bool = False,
        tearsheet_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Compare portfolio performance against a benchmark.
        
        Parameters
        ----------
        benchmark : str or pd.Series
            Either a ticker symbol (e.g., 'SPY') to fetch via yfinance,
            or a pd.Series of daily returns with a datetime index.
        generate_tearsheet : bool, optional
            Whether to generate an HTML tearsheet using quantstats (default False).
            Requires quantstats to be installed.
        tearsheet_path : str, optional
            Path to save the HTML tearsheet. If None, saves to 'tearsheet.html'.
        
        Returns
        -------
        pd.DataFrame
            Comparison of strategy vs benchmark metrics.
        
        Raises
        ------
        RuntimeError
            If calculate_equity() has not been called first.
        ImportError
            If generate_tearsheet=True but quantstats is not installed.
        """
        if self.net_returns is None:
            raise RuntimeError("Must call calculate_equity() before compare_benchmark().")
        
        # Get benchmark returns
        if isinstance(benchmark, str):
            self._benchmark_returns = self._fetch_benchmark_returns(benchmark)
        else:
            if not isinstance(benchmark, pd.Series):
                raise TypeError("benchmark must be a string ticker or pd.Series of returns.")
            self._benchmark_returns = benchmark
        
        # Normalize dates to date-only (remove time component) for alignment
        # This handles TradingView data (which has time like 21:30) vs yfinance (midnight)
        strategy_returns_normalized = self.net_returns.copy()
        strategy_returns_normalized.index = pd.to_datetime(strategy_returns_normalized.index).normalize()
        
        benchmark_returns_normalized = self._benchmark_returns.copy()
        benchmark_returns_normalized.index = pd.to_datetime(benchmark_returns_normalized.index).normalize()
        
        # Remove duplicate dates (keep first occurrence)
        strategy_returns_normalized = strategy_returns_normalized[~strategy_returns_normalized.index.duplicated(keep='first')]
        benchmark_returns_normalized = benchmark_returns_normalized[~benchmark_returns_normalized.index.duplicated(keep='first')]
        
        # Align benchmark with strategy
        common_dates = strategy_returns_normalized.index.intersection(benchmark_returns_normalized.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between strategy and benchmark.")
        
        strategy_returns = strategy_returns_normalized.loc[common_dates]
        benchmark_returns = benchmark_returns_normalized.loc[common_dates]
        
        # Calculate benchmark equity curve
        self._benchmark_equity = (1 + benchmark_returns).cumprod()
        
        # Generate comparison metrics
        strategy_metrics = self._calculate_metrics_for_series(strategy_returns)
        benchmark_metrics = self._calculate_metrics_for_series(benchmark_returns)
        
        comparison = pd.DataFrame({
            "Strategy": strategy_metrics,
            "Benchmark": benchmark_metrics,
        })
        
        # Generate tearsheet if requested
        if generate_tearsheet:
            self._generate_quantstats_tearsheet(
                strategy_returns,
                benchmark_returns,
                tearsheet_path or "tearsheet.html",
            )
        
        return comparison
    
    def _fetch_benchmark_returns(self, ticker: str) -> pd.Series:
        """Fetch benchmark returns from yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required to fetch benchmark data. Install with: pip install yfinance")
        
        start_date = self.returns.index.min()
        end_date = self.returns.index.max()
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"Could not fetch data for ticker: {ticker}")
        
        # Handle different yfinance column formats
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker download format
            prices = data["Close"][ticker]
        elif "Adj Close" in data.columns:
            # Old format with Adj Close
            prices = data["Adj Close"]
        else:
            # New auto-adjusted format
            prices = data["Close"]
        
        returns = prices.pct_change().dropna()
        returns.name = ticker
        
        return returns
    
    def _calculate_metrics_for_series(self, returns: pd.Series) -> pd.Series:
        """Calculate key metrics for a returns series."""
        n_days = len(returns)
        equity = (1 + returns).cumprod()
        
        total_return = equity.iloc[-1] - 1
        annual_return = calculate_annualized_return(total_return, n_days, self.trading_days)
        drawdown = calculate_drawdown_series(equity)
        max_drawdown = abs(drawdown.min())
        annual_std = returns.std() * np.sqrt(self.trading_days)
        sharpe = calculate_sharpe_ratio(returns, 0.0, self.trading_days)
        sortino = calculate_sortino_ratio(returns, 0.0, self.trading_days)
        calmar = calculate_calmar_ratio(annual_return, max_drawdown)
        
        return pd.Series({
            "Total Return [%]": round(total_return * 100, 2),
            "Annual Return [%]": round(annual_return * 100, 2),
            "Max Drawdown [%]": round(max_drawdown * 100, 2),
            "Std Deviation [%]": round(annual_std * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Calmar Ratio": round(calmar, 3),
        })
    
    def _generate_quantstats_tearsheet(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        output_path: str,
    ) -> None:
        """Generate HTML tearsheet using quantstats."""
        try:
            import quantstats as qs
        except ImportError:
            raise ImportError(
                "quantstats is required for tearsheet generation. "
                "Install with: pip install quantstats"
            )
        
        qs.reports.html(
            strategy_returns,
            benchmark=benchmark_returns,
            output=output_path,
            title="VectorBacktester Strategy Report",
        )
        print(f"Tearsheet saved to: {output_path}")
    
    def plot_results(
        self,
        include_benchmark: bool = True,
        log_scale: bool = False,
        figsize: tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Plot strategy performance visualization.
        
        Parameters
        ----------
        include_benchmark : bool, optional
            Whether to include benchmark in plots (default True).
            Requires compare_benchmark() to have been called first.
        log_scale : bool, optional
            Whether to use logarithmic scale for equity curves (default False).
        figsize : tuple[int, int], optional
            Figure size as (width, height) in inches (default (14, 10)).
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        
        Raises
        ------
        RuntimeError
            If calculate_equity() has not been called first.
        """
        if self.equity_curve is None:
            raise RuntimeError("Must call calculate_equity() before plot_results().")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # --- Equity Curve Plot ---
        ax1 = axes[0]
        ax1.plot(self.equity_curve.index, self.equity_curve.values, 
                 label="Strategy", linewidth=1.5, color="#2E86AB")
        
        if include_benchmark and self._benchmark_equity is not None:
            # Align benchmark to same start
            benchmark_aligned = self._benchmark_equity.reindex(self.equity_curve.index)
            if not benchmark_aligned.isna().all():
                ax1.plot(benchmark_aligned.index, benchmark_aligned.values,
                        label="Benchmark", linewidth=1.5, color="#A23B72", alpha=0.8)
        
        ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_ylabel("Portfolio Value (log scale)", fontsize=12)
        
        # --- Drawdown (Underwater) Plot ---
        ax2 = axes[1]
        drawdown = calculate_drawdown_series(self.equity_curve) * 100  # Convert to percentage
        
        ax2.fill_between(drawdown.index, 0, drawdown.values, 
                         color="#E74C3C", alpha=0.7, label="Strategy Drawdown")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        
        if include_benchmark and self._benchmark_equity is not None:
            benchmark_dd = calculate_drawdown_series(self._benchmark_equity) * 100
            benchmark_dd = benchmark_dd.reindex(drawdown.index)
            if not benchmark_dd.isna().all():
                ax2.plot(benchmark_dd.index, benchmark_dd.values,
                        color="#A23B72", alpha=0.6, linewidth=1, label="Benchmark Drawdown")
        
        ax2.set_title("Underwater (Drawdown) Chart", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown [%]", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def summary(self) -> str:
        """
        Return a formatted summary string of the backtest.
        
        Returns
        -------
        str
            Formatted summary of backtest configuration and results.
        """
        lines = [
            "=" * 60,
            "VectorBacktester Summary",
            "=" * 60,
            f"Date Range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}",
            f"Assets: {list(self.returns.columns)}",
            f"Trading Days: {len(self.returns)}",
            f"Transaction Costs: {self.transaction_costs_applied} bps",
        ]
        
        if self.equity_curve is not None:
            metrics = self.get_performance_metrics()
            lines.extend([
                "-" * 60,
                "Performance Metrics:",
                "-" * 60,
            ])
            for key, value in metrics.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("\n[Call calculate_equity() to compute metrics]")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of the backtester."""
        return (
            f"VectorBacktester("
            f"assets={len(self.returns.columns)}, "
            f"dates={len(self.returns)}, "
            f"equity_calculated={self.equity_curve is not None})"
        )
