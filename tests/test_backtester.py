"""Unit tests for VectorBacktester."""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from marit_backtesting import VectorBacktester


@pytest.fixture
def sample_data():
    """Create sample returns and weights DataFrames."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    tickers = ["A", "B", "C"]
    
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 3) * 0.02,
        index=dates,
        columns=tickers,
    )
    weights = pd.DataFrame(
        np.ones((100, 3)) / 3,
        index=dates,
        columns=tickers,
    )
    
    return returns, weights


class TestVectorBacktesterInit:
    """Tests for VectorBacktester initialization."""
    
    def test_basic_init(self, sample_data):
        """Test basic initialization."""
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        
        assert len(bt.returns) == 100
        assert len(bt.weights) == 100
        assert list(bt.returns.columns) == ["A", "B", "C"]
    
    def test_data_alignment(self):
        """Test that returns and weights are properly aligned."""
        dates_r = pd.date_range(start="2023-01-01", periods=50, freq="B")
        dates_w = pd.date_range(start="2023-01-20", periods=50, freq="B")
        
        returns = pd.DataFrame({"A": np.random.randn(50)}, index=dates_r)
        weights = pd.DataFrame({"A": np.ones(50)}, index=dates_w)
        
        bt = VectorBacktester(returns, weights)
        
        # Should only have overlapping dates
        assert len(bt.returns) < 50
        assert bt.returns.index.equals(bt.weights.index)
    
    def test_column_alignment(self):
        """Test that columns are properly aligned."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="B")
        
        returns = pd.DataFrame({"A": np.zeros(50), "B": np.zeros(50)}, index=dates)
        weights = pd.DataFrame({"B": np.ones(50), "C": np.ones(50)}, index=dates)
        
        bt = VectorBacktester(returns, weights)
        
        # Should only have common columns
        assert list(bt.returns.columns) == ["B"]
        assert list(bt.weights.columns) == ["B"]
    
    def test_missing_value_handling(self):
        """Test that missing values are handled correctly."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
        
        returns = pd.DataFrame({"A": [0.01, np.nan, 0.02, 0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01]}, index=dates)
        weights = pd.DataFrame({"A": [1.0, np.nan, np.nan, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0]}, index=dates)
        
        bt = VectorBacktester(returns, weights)
        
        # Returns should be filled with 0
        assert bt.returns.isna().sum().sum() == 0
        # Weights should be forward-filled
        assert bt.weights.isna().sum().sum() == 0
    
    def test_no_overlap_raises_error(self):
        """Test that no date overlap raises ValueError."""
        dates_r = pd.date_range(start="2022-01-01", periods=10, freq="B")
        dates_w = pd.date_range(start="2023-01-01", periods=10, freq="B")
        
        returns = pd.DataFrame({"A": np.zeros(10)}, index=dates_r)
        weights = pd.DataFrame({"A": np.ones(10)}, index=dates_w)
        
        with pytest.raises(ValueError, match="No overlapping dates"):
            VectorBacktester(returns, weights)


class TestCalculateEquity:
    """Tests for calculate_equity method."""
    
    def test_basic_equity_calculation(self, sample_data):
        """Test basic equity curve calculation."""
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        
        equity = bt.calculate_equity()
        
        assert len(equity) == 100
        assert equity.iloc[0] != 0
        assert bt.net_returns is not None
    
    def test_transaction_costs(self, sample_data):
        """Test that transaction costs reduce returns."""
        returns, weights = sample_data
        
        bt_no_cost = VectorBacktester(returns.copy(), weights.copy())
        bt_with_cost = VectorBacktester(returns.copy(), weights.copy())
        
        equity_no_cost = bt_no_cost.calculate_equity(transaction_costs_bps=0)
        equity_with_cost = bt_with_cost.calculate_equity(transaction_costs_bps=100)
        
        # With costs should end lower
        assert equity_with_cost.iloc[-1] < equity_no_cost.iloc[-1]
    
    def test_turnover_tracking(self):
        """Test that turnover is correctly calculated."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="B")
        
        returns = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01, 0.01]}, index=dates)
        weights = pd.DataFrame({"A": [0.5, 0.5, 1.0, 1.0, 0.5]}, index=dates)
        
        bt = VectorBacktester(returns, weights)
        bt.calculate_equity()
        
        # Day 0: initial = 0.5, Day 2: 0.5 change, Day 4: 0.5 change
        assert bt.turnover.iloc[0] == 0.5
        assert bt.turnover.iloc[2] == 0.5
        assert bt.turnover.iloc[4] == 0.5


class TestPerformanceMetrics:
    """Tests for get_performance_metrics method."""
    
    def test_metrics_without_equity_raises_error(self, sample_data):
        """Test that calling metrics before equity raises error."""
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        
        with pytest.raises(RuntimeError, match="Must call calculate_equity"):
            bt.get_performance_metrics()
    
    def test_metrics_returns_series(self, sample_data):
        """Test that metrics returns a Series with expected keys."""
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        bt.calculate_equity()
        
        metrics = bt.get_performance_metrics()
        
        assert isinstance(metrics, pd.Series)
        assert "Total Return [%]" in metrics.index
        assert "Sharpe Ratio" in metrics.index
        assert "Max Drawdown [%]" in metrics.index
    
    def test_positive_returns_metrics(self):
        """Test metrics for consistently positive returns."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
        
        returns = pd.DataFrame({"A": np.ones(252) * 0.001}, index=dates)
        weights = pd.DataFrame({"A": np.ones(252)}, index=dates)
        
        bt = VectorBacktester(returns, weights)
        bt.calculate_equity()
        metrics = bt.get_performance_metrics()
        
        assert metrics["Total Return [%]"] > 0
        assert metrics["Sharpe Ratio"] > 0
        assert metrics["Max Drawdown [%]"] == 0  # No drawdowns


class TestPlotResults:
    """Tests for plot_results method."""
    
    def test_plot_without_equity_raises_error(self, sample_data):
        """Test that plotting before equity raises error."""
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        
        with pytest.raises(RuntimeError, match="Must call calculate_equity"):
            bt.plot_results()
    
    def test_plot_returns_figure(self, sample_data):
        """Test that plot returns a figure object."""
        import matplotlib.pyplot as plt
        
        returns, weights = sample_data
        bt = VectorBacktester(returns, weights)
        bt.calculate_equity()
        
        fig = bt.plot_results()
        
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
