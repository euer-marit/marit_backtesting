#!/usr/bin/env python3
"""
Test script for BETA Strategy with VectorBacktester.

This script:
1. Initializes the Marit_model from the BETA_Strategy
2. Gets the base model weights and returns
3. Runs a backtest using VectorBacktester
4. Compares results with SPY benchmark
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BETA_Strategy_For_Test"))

import pandas as pd
import matplotlib.pyplot as plt

# Import VectorBacktester
from marit_backtesting import VectorBacktester

# Import BETA Strategy components
from marit_beta_model import Marit_model


def main():
    print("=" * 70)
    print("BETA Strategy Backtest with VectorBacktester")
    print("=" * 70)
    
    # Load Fed Funds Rate data
    print("\n[1] Loading Fed Funds Rate data...")
    fed_fund_path = os.path.join(os.path.dirname(__file__), "..", "BETA_Strategy_For_Test", "FEDFUNDS.csv")
    fed_fund_rate = pd.read_csv(fed_fund_path)
    fed_fund_rate['date'] = pd.to_datetime(fed_fund_rate['date'])
    fed_fund_rate.set_index('date', inplace=True)
    fed_fund_rate.index = pd.to_datetime(fed_fund_rate.index).normalize() + pd.Timedelta(hours=21, minutes=30)
    print(f"    Fed Funds data: {len(fed_fund_rate)} rows")
    
    # Initialize Marit model and get base model weights/returns
    print("\n[2] Initializing Marit model (fetching data from TradingView)...")
    print("    This may take a few minutes...")
    marit_model = Marit_model()
    base_weights, base_returns = marit_model.base_model_initiation()
    
    print(f"\n    Weights shape: {base_weights.shape}")
    print(f"    Returns shape: {base_returns.shape}")
    print(f"    Assets: {list(base_weights.columns)}")
    print(f"    Date range: {base_weights.index[0].date()} to {base_weights.index[-1].date()}")
    
    # Display weight statistics
    print("\n[3] Weight Statistics:")
    print("-" * 50)
    print(base_weights.describe().round(3))
    
    # # Align returns with weights
    # base_weights = base_weights.dropna()
    # base_returns = base_returns.loc[base_weights.index]

    # Create VectorBacktester
    print("\n[4] Creating VectorBacktester...")
    bt = VectorBacktester(base_returns, base_weights * 2)
    print(f"    {bt}")
    
    # Calculate equity with full fee structure
    print("\n[5] Calculating equity curve with full fee structure...")
    print("    - Transaction costs: 2 bps")
    print("    - Borrowing fee: 1.2% annual")
    print("    - Margin spread: 1.5% above Fed Funds")
    equity = bt.calculate_equity(
        transaction_costs_bps=2,
        borrowing_fee_rate=0.012,  # 1.2% annual for shorts
        margin_spread=1.5,         # 1.5% above Fed Funds rate
        fed_fund_rate_df=fed_fund_rate,
    )
    print(f"    Starting value: {equity.iloc[0]:.4f}")
    print(f"    Ending value: {equity.iloc[-1]:.4f}")
    print(f"    Total turnover: {bt.turnover.sum():.2f}")
    
    # Show fee breakdown
    print("\n[5.5] Fee Breakdown:")
    print("-" * 50)
    total_fees = bt.get_total_fees()
    for key, value in total_fees.items():
        print(f"    {key:.<35} {value:.4f}")
    
    # Get performance metrics
    print("\n[6] Performance Metrics:")
    print("-" * 60)
    metrics = bt.get_performance_metrics()
    for key, value in metrics.items():
        print(f"    {key:.<40} {value}")
    
    # Compare to benchmark
    print("\n[7] Comparing against SPY benchmark...")
    try:
        comparison = bt.compare_benchmark("SPY", generate_tearsheet=False)
        print("\n    Strategy vs Benchmark Comparison:")
        print("-" * 60)
        print(comparison.to_string())
    except Exception as e:
        print(f"    Error comparing benchmark: {e}")
    
    # Plot results
    print("\n[8] Generating visualization...")
    fig = bt.plot_results(include_benchmark=True, log_scale=False)
    output_path = os.path.join(os.path.dirname(__file__), "beta_strategy_backtest.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Saved plot to: {output_path}")
    
    # Print summary
    print("\n" + bt.summary())
    
    # Optional: Generate quantstats tearsheet
    try:
        import quantstats
        print("\n[9] Generating QuantStats HTML tearsheet...")
        tearsheet_path = os.path.join(os.path.dirname(__file__), "beta_strategy_tearsheet.html")
        bt.compare_benchmark("SPY", generate_tearsheet=True, tearsheet_path=tearsheet_path)
    except ImportError:
        print("\n[9] Skipping HTML tearsheet (quantstats not installed)")
    except Exception as e:
        print(f"\n[9] Error generating tearsheet: {e}")
    
    plt.show()
    
    return bt, metrics


if __name__ == "__main__":
    bt, metrics = main()
