import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import plotly.graph_objects as go

class MCSimulation:
    """
    A Python class for running Monte Carlo simulation on portfolio price data.
    """
    
    def __init__(self, portfolio_data, weights=None, num_simulation=1000, num_trading_days=252):
        """
        Initialize the Monte Carlo simulation attributes.
        """
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")
        
        # Pivot the dataset to have tickers as columns
        pivot_data = portfolio_data.pivot(columns='symbol', values='close')
        
        # Calculate daily returns
        daily_returns = pivot_data.pct_change()
        
        # If weights are not provided, assume equal distribution
        if weights is None:
            weights = [1.0 / len(daily_returns.columns)] * len(daily_returns.columns)
        elif sum(weights) != 1:
            raise ValueError("Sum of portfolio weights must equal one.")
        
        self.daily_returns = daily_returns.dropna()
        self.weights = np.array(weights)
        self.nSim = num_simulation
        self.nTrading = num_trading_days
        self.simulated_return = None
        self.confidence_interval = None

    def calc_cumulative_return(self, clear_previous=True):
        """
        Calculates the cumulative return using Monte Carlo simulation with GBM.
        """
        # Clear previous simulations to free up memory
        if clear_previous:
            self.simulated_return = None
            self.confidence_interval = None
            gc.collect()  # Run the garbage collector

        number_of_tickers = len(self.daily_returns.columns)
        mean_return = self.daily_returns.mean().values
        std_return = self.daily_returns.std().values

        # Get the last prices
        last_prices = self.daily_returns.iloc[-1].values

        # Initialize an array to hold the simulated prices
        simulated_prices = np.zeros((self.nTrading, self.nSim, number_of_tickers))
        simulated_prices[0] = last_prices

        # Simulate the stock prices using GBM
        for t in range(1, self.nTrading):
            for s in range(number_of_tickers):
                simulated_prices[t, :, s] = simulated_prices[t-1, :, s] * (1 + np.random.normal(mean_return[s], std_return[s], self.nSim))

        # Convert simulated prices to returns
        simulated_returns = simulated_prices[1:] / simulated_prices[:-1] - 1

        # Calculate the cumulative returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)

        # Average cumulative returns across tickers (assuming equal weights for simplicity)
        avg_cumulative_returns = np.mean(cumulative_returns, axis=2)

        self.simulated_return = pd.DataFrame(avg_cumulative_returns)
        self.confidence_interval = self.simulated_return.iloc[-1].quantile([0.025, 0.975])

        return self.simulated_return

    
    def plot_simulation(self):
        """
        Plot the mean, median, and 95% confidence interval of the simulated stock trajectories using Plotly.
        """
        if self.simulated_return is None:
            self.calc_cumulative_return()

        # Calculate mean, median, and 95% confidence intervals
        mean_return = self.simulated_return.mean(axis=1)
        median_return = self.simulated_return.median(axis=1)
        lower_bound = self.simulated_return.quantile(0.025, axis=1)
        upper_bound = self.simulated_return.quantile(0.975, axis=1)

        # Plotting with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_return.index, y=mean_return, mode='lines', name='Mean', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=median_return.index, y=median_return, mode='lines', name='Median', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=lower_bound.index, y=lower_bound, fill='tonexty', mode='none', name='Lower 95% CI'))
        fig.add_trace(go.Scatter(x=upper_bound.index, y=upper_bound, fill='tonexty', mode='none', name='Upper 95% CI'))

        fig.update_layout(title=f"{self.nSim} Simulations of Cumulative Portfolio Return Trajectories Over {self.nTrading} Trading Days", showlegend=True)
        
        return fig

    def plot_distribution(self):
        """
        Plot the distribution of cumulative returns using Plotly.
        """
        if self.simulated_return is None:
            self.calc_cumulative_return()

        # Plotting with Plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.simulated_return.iloc[-1], name='Returns', opacity=0.75, nbinsx=10))
        fig.add_shape(dict(type='line', x0=self.confidence_interval.iloc[0], x1=self.confidence_interval.iloc[0], y0=0, y1=1, yref='paper', line=dict(color='red')))
        fig.add_shape(dict(type='line', x0=self.confidence_interval.iloc[1], x1=self.confidence_interval.iloc[1], y0=0, y1=1, yref='paper', line=dict(color='red')))

        fig.update_layout(title=f"Distribution of Final Cumulative Returns Across All {self.nSim} Simulations", showlegend=True)
        
        return fig

    def summarize_cumulative_return(self):
        """
        Summarize the final cumulative return statistics.
        """
        if self.simulated_return is None:
            self.calc_cumulative_return()
        
        stats = self.simulated_return.iloc[-1].describe()
        ci_series = self.confidence_interval
        ci_series.index = ["95% CI Lower", "95% CI Upper"]
        return pd.concat([stats,ci_series])
