import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
np.random.seed(42)  # For reproducibility
T = 252  # Trading days in a year
mu = 0.1  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
S0 = 100  # Initial stock price
initial_capital = 10_000  # Initial capital in dollars
initial_investment_fraction = 0.5  # Use 5% of cash for initial asset purchase
sell_fraction = 0.2  # Fraction of assets to sell on decline
threshold = 0.005  # 0.5% threshold for returns

# 1. Simulate stock prices using GBM
def simulate_gbm(S0, mu, sigma, T):
    """Generate GBM stock prices."""
    dt = 1 / T  # Time increment
    prices = [S0]
    for _ in range(T):
        Z = np.random.normal()
        dS = mu * prices[-1] * dt + sigma * prices[-1] * np.sqrt(dt) * Z
        prices.append(prices[-1] + dS)
    return np.array(prices)

# 2. Calculate portfolio returns
def portfolio_return(portfolio_value):
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    return np.concatenate([[0], returns])  # Include 0 for the first day

# 3. Generate buy/sell signals based on portfolio returns
def generate_signals(portfolio_value, holdings, threshold, sell_fraction, buy_fraction):
    returns = portfolio_return(portfolio_value)
    signals = np.zeros(len(portfolio_value))  # 0: Hold, 1: Buy, -1: Sell

    for i in range(1, len(returns)):
        if returns[i] <= -threshold and holdings[i - 1] > 0:  # Sell condition
            signals[i] = -1  # Sell 20% of holdings
        elif returns[i] >= threshold and holdings[i - 1] > 0:  # Buy condition
            signals[i] = 1  # Buy a fraction of cash reserve

    return signals

# 4. Backtest with optimized buy fraction
def backtest(prices, signals, initial_capital, buy_fraction, sell_fraction):
    """Run the backtest and return portfolio value over time."""
    capital = initial_capital * (1 - initial_investment_fraction)  # Remaining cash after initial buy
    holdings = (initial_capital * initial_investment_fraction) / prices[0]  # Initial shares bought
    portfolio_value = []  # Store portfolio value over time

    for i, price in enumerate(prices):
        if signals[i] == -1:  # Sell signal
            sell_amount = sell_fraction * holdings
            capital += sell_amount * price
            holdings -= sell_amount

        elif signals[i] == 1 and capital > 0:  # Buy signal
            buy_amount = buy_fraction * capital / price
            holdings += buy_amount
            capital -= buy_amount * price
        
        v = capital + holdings * price
        print(f'-- portfolio: {v:,.2f}')
        portfolio_value.append(capital + holdings * price)

    return np.array(portfolio_value)

# 5. Objective function for optimization
def objective_function(buy_fraction, prices, initial_capital, sell_fraction, threshold):
    signals = generate_signals(
        np.full_like(prices, initial_capital), np.zeros(len(prices)), threshold, sell_fraction, buy_fraction
    )
    portfolio_value = backtest(prices, signals, initial_capital, buy_fraction, sell_fraction)
    return -portfolio_value[-1]  # Negative for maximization

# 6. Optimize buy fraction using scipy.optimize
def optimize_buy_fraction(prices, initial_capital, sell_fraction, threshold):
    result = minimize(
        objective_function,
        x0=0.1,  # Initial guess for buy fraction
        bounds=[(0.0, 1.0)],  # Fraction between 0 and 1
        args=(prices, initial_capital, sell_fraction, threshold)
    )
    return result.x[0]

if __name__ == "__main__":
    prices = simulate_gbm(S0, mu, sigma, T)

    optimal_buy_fraction = optimize_buy_fraction(prices, initial_capital, sell_fraction, threshold)
    print(f"Optimal Buy Fraction: {optimal_buy_fraction:.4f}")

    signals = generate_signals(
        np.full_like(prices, initial_capital), np.zeros(len(prices)), threshold, sell_fraction, optimal_buy_fraction
    )

    portfolio_value = backtest(prices, signals, initial_capital, optimal_buy_fraction, sell_fraction)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(prices, color='tab:blue', label='Stock Price')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Stock Price ($)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(portfolio_value, color='tab:green', label='Portfolio Value')
    ax2.set_ylabel('Portfolio Value ($)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()  # Prevent overlapping labels
    plt.title('Stock Price and Portfolio Value Over Time')
    plt.show()

