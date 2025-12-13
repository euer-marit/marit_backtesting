#!/usr/bin/env python3
"""
Example usage of the VectorBacktester library.

This script demonstrates:
1. Creating synthetic returns and weights data
2. Running a backtest with transaction costs
3. Displaying performance metrics
4. Comparing against a benchmark (SPY)
5. Generating visualizations
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from marit_backtesting import VectorBacktester


def generate_synthetic_data(
    n_days: int = 504,  # ~2 years
    n_assets: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic returns and weights for demonstration."""
    np.random.seed(seed)
    
    # Generate date range (business days)
    dates = pd.date_range(start="2022-01-03", periods=n_days, freq="B")
    
    # Asset tickers
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"][:n_assets]
    
    # Generate synthetic daily returns
    # Each asset has different mean and volatility
    means = np.array([0.0005, 0.0004, 0.0006, 0.0003, 0.0007])[:n_assets]
    vols = np.array([0.02, 0.025, 0.018, 0.03, 0.028])[:n_assets]
    
    returns_data = np.random.randn(n_days, n_assets) * vols + means
    returns_df = pd.DataFrame(returns_data, index=dates, columns=tickers)
    
    # Generate dynamic weights (simple momentum-based rebalancing)
    # Start with equal weights, then adjust based on trailing returns
    weights_data = np.ones((n_days, n_assets)) / n_assets
    
    lookback = 20
    for i in range(lookback, n_days):
        # Simple momentum: higher weight to assets with positive trailing returns
        trailing = returns_df.iloc[i-lookback:i].sum()
        adjusted = np.exp(trailing * 10)  # Scale for effect
        weights_data[i] = adjusted / adjusted.sum()
    
    weights_df = pd.DataFrame(weights_data, index=dates, columns=tickers)
    
    return returns_df, weights_df


def main():
    print("=" * 70)
    print("VectorBacktester Example")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic returns and weights data...")
    returns_df, weights_df = generate_synthetic_data()
    print(f"    Returns shape: {returns_df.shape}")
    print(f"    Weights shape: {weights_df.shape}")
    print(f"    Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"    Assets: {list(returns_df.columns)}")
    
    # Create backtester
    print("\n[2] Creating VectorBacktester instance...")
    bt = VectorBacktester(returns_df, weights_df)
    print(f"    {bt}")
    
    # Calculate equity with transaction costs
    print("\n[3] Calculating equity curve (10 bps transaction costs)...")
    equity = bt.calculate_equity(transaction_costs_bps=10)
    print(f"    Starting value: {equity.iloc[0]:.4f}")
    print(f"    Ending value: {equity.iloc[-1]:.4f}")
    print(f"    Total turnover: {bt.turnover.sum():.2f}")
    
    # Get performance metrics
    print("\n[4] Performance Metrics:")
    print("-" * 50)
    metrics = bt.get_performance_metrics()
    for key, value in metrics.items():
        print(f"    {key:.<35} {value}")
    
    # Compare to benchmark (uncomment if you have internet access)
    print("\n[5] Comparing against SPY benchmark...")
    try:
        comparison = bt.compare_benchmark("SPY", generate_tearsheet=False)
        print("\n    Strategy vs Benchmark Comparison:")
        print("-" * 50)
        print(comparison.to_string())
    except Exception as e:
        print(f"    Skipping benchmark comparison: {e}")
    
    # Plot results
    print("\n[6] Generating visualization...")
    fig = bt.plot_results(include_benchmark=True, log_scale=False)
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("    Saved plot to 'backtest_results.png'")
    
    # Show summary
    print("\n" + bt.summary())
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
