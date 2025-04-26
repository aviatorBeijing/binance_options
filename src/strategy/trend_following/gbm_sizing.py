import numpy as np
import pandas as pd
import datetime

from scipy.optimize import minimize

from butil.portfolio_stats import max_drawdowns,sharpe

# Constants
np.random.seed(42)  # For reproducibility
T = 252  # Trading days in a year
mu = 0.1  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
S0 = 100  # Initial stock price
initial_capital = 10_000  # Initial capital in dollars
initial_investment_fraction = 0.95  # 50% of capital used initially
sell_fraction = 0.2  # Fraction of holdings to sell on decline
threshold = 0.005  # 0.5% threshold for returns
bootstrap_runs = 10  # Number of simulations for bootstrapping

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

def backtest(prices, initial_capital, buy_fraction, sell_fraction, threshold):
    """Run the backtest with integrated signal generation."""
    # Initial setup: 50% capital invested in the asset
    capital = initial_capital * (1 - initial_investment_fraction)  # Remaining cash
    holdings = (initial_capital * initial_investment_fraction) / prices[0]  # Initial shares bought

    portfolio_value = []  # Store portfolio value over time

    #print('-- settings:', f'buy = {buy_fraction}, sell = {sell_fraction}') 
    for i in range(len(prices)):
        # Calculate current portfolio value (cash + asset holdings)
        current_value = capital + holdings * prices[i]
        portfolio_value.append(current_value)  # Append scalar value

        # Skip signal generation on the first day
        if i == 0:
            continue
        
        # Calculate portfolio return from the previous day
        portfolio_return = (portfolio_value[i] - portfolio_value[i - 1]) / portfolio_value[i - 1]
        #portfolio_return = (prices[i] - prices[i-1])/prices[i-1]

        # Trade logic based on portfolio return
        if portfolio_return <= -threshold and holdings > 0:  # Sell signal
            sell_amount = sell_fraction * holdings
            capital += sell_amount * prices[i]
            holdings -= sell_amount
            #print('   -- sell', f'{(portfolio_return*100):.1f}%, $ {portfolio_value[-1]:.2f}')

        elif portfolio_return >= threshold and capital > 0:  # Buy signal
            buy_amount = buy_fraction * capital / prices[i]
            holdings += buy_amount
            capital -= buy_amount * prices[i]
            #print('-- buy', f'{(portfolio_return*100):.1f}%, $ {portfolio_value[-1]:.2f}')
    #print() 
    return np.array(portfolio_value)  # Return as NumPy array

dts = pd.date_range( end=datetime.datetime.now(),periods=T+1 )
def objective_function(params, prices, initial_capital, threshold):
    """Objective function to maximize final portfolio value."""
    global dts
    buy_fraction = params[0]
    sell_fraction = params[1]
    portfolio_value = backtest(prices, initial_capital, buy_fraction, sell_fraction, threshold)

    pdf = pd.DataFrame()
    pdf['pval'] = portfolio_value
    pdf.index = dts
    rt = pdf.pval.pct_change()
    s = sharpe( rt ) # Optimizing on Sharpe
    if s<0:
        s = 0
    
    return -s
    
    #return -portfolio_value[-1]  # Negative for maximization

def optimize_buy_fraction(prices, initial_capital, sell_fraction, threshold):
    """Find the optimal buy fraction to maximize final portfolio value."""
    result = minimize(
        objective_function,
        x0=[0.05, 0.05],  # Initial guess for buy fraction
        bounds=[(0.0, 1.0,), (0.0,1.0) ],  # Fraction between 0 and 1
        args=(prices, initial_capital, threshold)
    )
    return result.x[0], result.x[1], -result.fun

def bootstrap_optimization(runs, initial_capital, sell_fraction, threshold):
    """Run multiple optimizations to get a robust optimal buy fraction."""
    optimal_params = []

    for _ in range(runs):
        prices = simulate_gbm(S0, mu, sigma, T)  # Simulate new price series
        param0, param1, val = optimize_buy_fraction(prices, initial_capital, sell_fraction, threshold)
        optimal_params.append( [param0,param1,val] )

    return optimal_params

if __name__ == "__main__":
    rst = bootstrap_optimization(
        bootstrap_runs, initial_capital, sell_fraction, threshold
    )
    
    df = pd.DataFrame.from_records( rst )
    df.columns = ['buy_fraction', 'sell_fraction', 'final_portfolio']
    df.sort_values('final_portfolio', ascending=False,inplace=True)
    
    print('--  buy fraction:', f'   mean = {df.buy_fraction.mean():.2f}, std = {df.buy_fraction.std():.2f}')
    print('-- sell fraction:', f'   mean = {df.sell_fraction.mean():.2f}, std = {df.sell_fraction.std():.2f}')

    from tabulate import tabulate
    print( tabulate(df, headers="keys") )

