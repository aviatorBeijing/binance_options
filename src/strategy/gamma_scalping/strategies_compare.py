import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Parameters
S0 = 100           # Initial price
K = 100            # Strike price
mu = 0.05          # Drift
sigma = 0.2        # Volatility
r = 0.01           # Risk-free rate
T = 1              # Time to maturity (1 year)
n = 252            # Number of steps (daily steps)
n_simulations = 10 # Number of simulation paths
dt = T / n         # Time increment

# Trading parameters
transaction_fee = 0.01    # Fee per transaction
price_threshold = 0.005   # 0.5% price move to trigger rebalancing

# Simulate GBM paths
np.random.seed(42) # Set seed for reproducibility
dW = np.random.normal(0, np.sqrt(dt), (n_simulations, n))
price_paths = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dW, axis=1))
price_paths = np.hstack([np.full((n_simulations, 1), S0), price_paths])

# Black-Scholes option pricing formula for call
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Delta of the call option
def call_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Initialize P&L tracking arrays
PnL_dynamic_delta = np.zeros(n_simulations)
PnL_static_delta = np.zeros(n_simulations)
PnL_no_hedging = np.zeros(n_simulations)
PnL_gamma_scalping = np.zeros(n_simulations)

# Perform simulations
for i in range(n_simulations):
    # Initial option and delta
    delta_initial = call_delta(S0, K, T, r, sigma)
    
    # For static delta hedging, set initial hedge
    hedge_static = delta_initial
    cash_static = -hedge_static * S0
    
    # For dynamic delta hedging, start with initial delta
    hedge_dynamic = delta_initial
    cash_dynamic = -hedge_dynamic * S0
    
    # For gamma scalping, start with initial delta
    hedge_gamma = delta_initial
    cash_gamma = -hedge_gamma * S0
    
    # No hedging: simply hold the option
    initial_option_value = black_scholes_call(S0, K, T, r, sigma)
    
    for t in range(1, n + 1):
        S = price_paths[i, t]
        T_remaining = T - t * dt
        option_value = black_scholes_call(S, K, T_remaining, r, sigma)
        new_delta = call_delta(S, K, T_remaining, r, sigma)
        
        # No hedging P&L
        PnL_no_hedging[i] = option_value - initial_option_value
        
        # Static delta hedging P&L
        PnL_static_delta[i] = option_value + hedge_static * S + cash_static
        
        # Dynamic delta hedging P&L
        PnL_dynamic_delta[i] = option_value + hedge_dynamic * S + cash_dynamic
        delta_adjustment_dynamic = new_delta - hedge_dynamic
        hedge_dynamic = new_delta
        cash_dynamic -= delta_adjustment_dynamic * S
        
        # Gamma scalping P&L
        PnL_gamma_scalping[i] = option_value + hedge_gamma * S + cash_gamma
        if abs(S - price_paths[i, t-1]) / price_paths[i, t-1] > price_threshold:
            delta_adjustment_gamma = new_delta - hedge_gamma
            hedge_gamma = new_delta
            cash_gamma -= delta_adjustment_gamma * S

# Plot the P&L distributions using sns.kdeplot
plt.figure(figsize=(12, 8))

sns.kdeplot(PnL_dynamic_delta, label='Dynamic Delta Hedging', fill=True)
sns.kdeplot(PnL_static_delta, label='Static Delta Hedging', fill=True)
sns.kdeplot(PnL_no_hedging, label='No Hedging', fill=True)
#sns.kdeplot(PnL_gamma_scalping, label='Gamma Scalping', fill=True)

plt.title('P&L Distribution Comparison Across Strategies')
plt.xlabel('P&L')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.grid(True)
import os;fn=os.getenv('USER_HOME','')+'/tmp/options_strategies_compare.png'
plt.savefig(fn)
print('--saved:',fn)
