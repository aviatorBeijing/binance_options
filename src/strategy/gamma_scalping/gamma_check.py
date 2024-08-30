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

# Gamma of the call option
def call_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Initialize arrays to track gamma values for each simulation
gamma_values = np.zeros((n_simulations, n+1))

# Perform simulations
for i in range(n_simulations):
    # Initial option and delta
    delta_initial = call_delta(S0, K, T, r, sigma)
    
    # For gamma scalping, start with initial delta
    hedge_gamma = delta_initial
    cash_gamma = -hedge_gamma * S0
    
    # Track gamma
    gamma_values[i, 0] = call_gamma(S0, K, T, r, sigma)
    
    for t in range(1, n + 1):
        S = price_paths[i, t]
        T_remaining = T - t * dt
        option_value = black_scholes_call(S, K, T_remaining, r, sigma)
        new_delta = call_delta(S, K, T_remaining, r, sigma)
        current_gamma = call_gamma(S, K, T_remaining, r, sigma)
        
        # Store gamma value
        gamma_values[i, t] = current_gamma
        
        # Gamma scalping adjustments
        if abs(S - price_paths[i, t-1]) / price_paths[i, t-1] > price_threshold:
            delta_adjustment_gamma = new_delta - hedge_gamma
            hedge_gamma = new_delta
            cash_gamma -= delta_adjustment_gamma * S

# Plot the gamma values for one of the simulations
plt.figure(figsize=(12, 6))
for i in range(n_simulations):
    plt.plot(gamma_values[i], label=f'Simulation {i+1}', alpha=0.7)

plt.title('Gamma Values Over Time for Gamma Scalping Strategy')
plt.xlabel('Time Step')
plt.ylabel('Gamma')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.show()

