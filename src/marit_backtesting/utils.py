"""Utility functions for backtesting calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def safe_divide(
    numerator: float | np.ndarray,
    denominator: float | np.ndarray,
    default: float = 0.0,
) -> float | np.ndarray:
    """
    Safely divide, returning default when denominator is zero.
    
    Parameters
    ----------
    numerator : float or array-like
        The numerator value(s).
    denominator : float or array-like
        The denominator value(s).
    default : float, optional
        Value to return when denominator is zero (default 0.0).
    
    Returns
    -------
    float or array-like
        Result of division, or default when denominator is zero.
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    
    result = np.where(denominator != 0, numerator / denominator, default)
    return result


def calculate_annualized_return(total_return: float, n_days: int, trading_days: int = 252) -> float:
    """
    Calculate annualized return (CAGR) from total return.
    
    Parameters
    ----------
    total_return : float
        Total return as a decimal (e.g., 0.10 for 10%).
    n_days : int
        Number of trading days in the period.
    trading_days : int, optional
        Trading days per year (default 252).
    
    Returns
    -------
    float
        Annualized return as a decimal.
    """
    if n_days <= 0:
        return 0.0
    years = n_days / trading_days
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    risk_free_rate : float, optional
        Annual risk-free rate (default 0.0).
    trading_days : int, optional
        Trading days per year (default 252).
    
    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    
    return safe_divide(mean_excess * np.sqrt(trading_days), std_excess, 0.0)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation based).
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    risk_free_rate : float, optional
        Annual risk-free rate (default 0.0).
    trading_days : int, optional
        Trading days per year (default 252).
    
    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    
    mean_excess = excess_returns.mean()
    
    # Downside deviation: std of negative returns only
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    return safe_divide(mean_excess * np.sqrt(trading_days), downside_std, 0.0)


def calculate_calmar_ratio(
    annual_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Parameters
    ----------
    annual_return : float
        Annualized return as a decimal.
    max_drawdown : float
        Maximum drawdown as a positive decimal (e.g., 0.20 for 20%).
    
    Returns
    -------
    float
        Calmar ratio.
    """
    return safe_divide(annual_return, max_drawdown, 0.0)


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (starting from 1.0 typically).
    
    Returns
    -------
    pd.Series
        Drawdown series (negative values representing decline from peak).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown


def calculate_max_drawdown_duration(drawdown: pd.Series) -> int:
    """
    Calculate maximum drawdown duration in days.
    
    Parameters
    ----------
    drawdown : pd.Series
        Drawdown series from calculate_drawdown_series.
    
    Returns
    -------
    int
        Maximum number of consecutive days in drawdown.
    """
    if len(drawdown) == 0:
        return 0
    
    # Find periods where we're in drawdown (< 0)
    in_drawdown = (drawdown < 0).astype(int)
    
    # Group consecutive drawdown periods
    drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
    drawdown_groups = drawdown_groups[in_drawdown == 1]
    
    if len(drawdown_groups) == 0:
        return 0
    
    # Find the longest drawdown duration
    durations = drawdown_groups.groupby(drawdown_groups).size()
    return int(durations.max()) if len(durations) > 0 else 0


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calculate skewness of returns distribution.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    
    Returns
    -------
    float
        Skewness of the distribution.
    """
    if len(returns) < 3:
        return 0.0
    return float(stats.skew(returns.dropna()))
