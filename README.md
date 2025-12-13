# Marit Backtesting

A production-grade vectorized portfolio backtesting library for Python.

## Features

- **Vectorized Operations**: Fast backtesting using NumPy/Pandas vectorization
- **Comprehensive Fee Structure**: Transaction costs, borrowing fees (shorts), and margin costs
- **15+ Performance Metrics**: Sharpe, Sortino, Calmar, drawdowns, skew, tail risk
- **Benchmark Comparison**: Compare against any ticker or custom benchmark series
- **QuantStats Integration**: Optional HTML tearsheet generation
- **Visualization**: Equity curves and underwater (drawdown) charts

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/marit-backtesting.git
cd marit-backtesting

# Install with pip (editable mode)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from marit_backtesting import VectorBacktester

# Load your data (daily returns and target weights)
returns_df = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
weights_df = pd.read_csv("weights.csv", index_col=0, parse_dates=True)

# Create backtester
bt = VectorBacktester(returns_df, weights_df)

# Run backtest with full fee structure
bt.calculate_equity(
    transaction_costs_bps=2,       # 2 bps per turnover
    borrowing_fee_rate=0.012,      # 1.2% annual for shorts
    margin_spread=1.5,             # 1.5% above Fed Funds rate
    fed_fund_rate_df=fed_funds,    # Optional: Fed Funds rate DataFrame
)

# Get performance metrics
metrics = bt.get_performance_metrics()
print(metrics)

# Compare against SPY and generate HTML tearsheet
comparison = bt.compare_benchmark("SPY", generate_tearsheet=True)
print(comparison)

# Plot results
bt.plot_results()
```

## Fee Structure

| Fee Type | Description | Parameter |
|----------|-------------|-----------|
| Transaction | Cost per unit of turnover | `transaction_costs_bps` |
| Borrowing | Annual cost for short positions | `borrowing_fee_rate` |
| Margin | Spread above Fed Funds for leverage | `margin_spread` + `fed_fund_rate_df` |

## Performance Metrics

| Category | Metrics |
|----------|---------|
| Period | Start/End dates, Start/End values |
| Returns | Total Return, CAGR |
| Risk | Max Drawdown, Avg Drawdown, Max DD Duration, Annualized Std |
| Ratios | Sharpe, Sortino, Calmar |
| Distribution | Skew, 5th/95th percentile tails |

## Examples

```bash
# Run the synthetic data demo
python examples/demo.py

# Run with real market data
python examples/test_real_data.py
```

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- matplotlib
- yfinance (for benchmark fetching)
- quantstats (optional, for HTML tearsheets)

## License

MIT License - see [LICENSE](LICENSE) for details.
