#!/usr/bin/env python3
"""
Test script for VectorBacktester using yfinance data.

This script demonstrates the VectorBacktester using real market data
fetched via yfinance, simulating a similar strategy to the BETA model.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from marit_backtesting import VectorBacktester


def fetch_data(tickers: list[str], start: str = "2015-01-01", end: str = "2024-01-01") -> pd.DataFrame:
    """Fetch adjusted close prices for multiple tickers."""
    print(f"    Fetching data for: {tickers}")
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    
    return prices


def generate_momentum_weights(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Generate dynamic portfolio weights based on momentum.
    Similar to the BETA strategy's rolling optimization approach.
    """
    returns = prices.pct_change()
    
    # Calculate momentum signal (trailing returns)
    momentum = returns.rolling(window=lookback).sum()
    
    # Convert to weights (positive momentum gets higher weight)
    positive_momentum = momentum.clip(lower=0)
    weight_sum = positive_momentum.sum(axis=1)
    
    # Normalize to sum to 1
    weights = positive_momentum.div(weight_sum, axis=0).fillna(1/len(prices.columns))
    
    # Ensure weights sum to 1
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(1/len(prices.columns))
    
    return weights


def main():
    print("=" * 70)
    print("VectorBacktester Test - Momentum Strategy with Real Data")
    print("=" * 70)
    
    # Define assets (similar to BETA strategy)
    tickers = ["GLD", "QQQ", "SPY", "BRK-B", "EEM"]
    
    # Fetch price data
    print("\n[1] Fetching historical price data via yfinance...")
    prices = fetch_data(tickers, start="2015-01-01", end="2024-01-01")
    prices = prices.ffill().dropna()
    print(f"    Data shape: {prices.shape}")
    print(f"    Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Calculate returns (next-day open returns simulation)
    print("\n[2] Calculating daily returns...")
    returns = prices.pct_change().shift(-1).dropna()  # Forward-looking returns
    print(f"    Returns shape: {returns.shape}")
    
    # Generate momentum-based weights
    print("\n[3] Generating momentum-based portfolio weights...")
    weights = generate_momentum_weights(prices, lookback=20)
    # Align with returns
    weights = weights.loc[returns.index]
    print(f"    Weights shape: {weights.shape}")
    
    # Display weight statistics
    print("\n[4] Weight Statistics:")
    print("-" * 50)
    print(weights.describe().round(3))
    
    # Create VectorBacktester
    print("\n[5] Creating VectorBacktester...")
    bt = VectorBacktester(returns, weights)
    print(f"    {bt}")
    
    # Calculate equity with 2 bps transaction costs (matching your default)
    print("\n[6] Calculating equity curve (2 bps transaction costs)...")
    equity = bt.calculate_equity(transaction_costs_bps=2)
    print(f"    Starting value: {equity.iloc[0]:.4f}")
    print(f"    Ending value: {equity.iloc[-1]:.4f}")
    print(f"    Total turnover: {bt.turnover.sum():.2f}")
    
    # Get performance metrics
    print("\n[7] Performance Metrics:")
    print("-" * 60)
    metrics = bt.get_performance_metrics()
    for key, value in metrics.items():
        print(f"    {key:.<40} {value}")
    
    # Compare to benchmark
    print("\n[8] Comparing against SPY benchmark...")
    try:
        comparison = bt.compare_benchmark("SPY", generate_tearsheet=False)
        print("\n    Strategy vs Benchmark Comparison:")
        print("-" * 60)
        print(comparison.to_string())
    except Exception as e:
        print(f"    Error comparing benchmark: {e}")
    
    # Plot results
    print("\n[9] Generating visualization...")
    fig = bt.plot_results(include_benchmark=True, log_scale=False)
    output_path = os.path.join(os.path.dirname(__file__), "momentum_strategy_backtest.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Saved plot to: {output_path}")
    
    # Print full summary
    print("\n" + bt.summary())
    
    # Show plot
    plt.show()
    
    return bt, metrics


if __name__ == "__main__":
    bt, metrics = main()
