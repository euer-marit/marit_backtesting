# Marit Backtesting

A production-grade vectorized portfolio backtesting library for Python.

## Features

- **Vectorized Operations**: Fast backtesting using NumPy/Pandas vectorization
- **Comprehensive Fee Structure**: Transaction costs, borrowing fees (shorts), and margin costs
- **15+ Performance Metrics**: Sharpe, Sortino, Calmar, drawdowns, skew, tail risk
- **Interactive Visualizations**: Plotly-based charts with hover tooltips and HTML export
- **Professional Quant Analytics**: Rolling Sharpe, volatility regime detection, VaR/CVaR
- **Benchmark Comparison**: Compare against any ticker or custom benchmark series
- **QuantStats Integration**: Optional HTML tearsheet generation

## Installation

```bash
# Install directly via pip
pip install git+https://github.com/euer-marit/marit_backtesting.git

# Or clone and install locally
git clone https://github.com/euer-marit/marit_backtesting.git
cd marit-backtesting
pip install -e .
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

# Generate comprehensive HTML report
bt.report("backtest_report.html")

# Compare against SPY benchmark
comparison = bt.compare_benchmark("SPY")
```

## Visualizations

### All-in-One Report
```python
bt.report("full_report.html")  # 10-panel dashboard
```

### Individual Charts
```python
from marit_backtesting import (
    # Core charts
    plot_equity_curve,
    plot_drawdown,
    plot_trailing_returns,
    
    # Portfolio analysis
    plot_weights,           # Monthly aggregated stacked bar
    plot_asset_returns,
    
    # Returns analysis
    plot_monthly_heatmap,   # With % annotations
    plot_quarterly_heatmap, # With % annotations
    plot_yearly_returns,    # With % annotations
    
    # Professional quant analytics
    plot_rolling_sharpe,       # Rolling Sharpe with reference lines
    plot_rolling_volatility,   # Regime detection
    plot_returns_distribution, # Histogram with VaR/CVaR
    plot_risk_contribution,    # Risk attribution by asset
)

# Example usage
plot_rolling_sharpe(bt, window=126, output_path="sharpe.html")
plot_returns_distribution(bt)  # Shows VaR 1%, VaR 5%, CVaR
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
- numpy, pandas, scipy, matplotlib
- yfinance (for benchmark fetching)
- plotly, kaleido (for interactive visualizations)
- quantstats (optional, for HTML tearsheets)

## License

MIT License - see [LICENSE](LICENSE) for details.
