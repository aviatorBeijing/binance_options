import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Option parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
T = 1.0        # Time to maturity in years
dt = 1/252     # Time step (daily)
N = int(T/dt)  # Number of time steps
n_sim = 1000   # Number of simulations

def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def simulate_gbm_paths(S0, r, sigma, T, dt, n_sim):
    N = int(T/dt)
    t = np.linspace(0, T, N)
    S_paths = np.zeros((n_sim, N))
    S_paths[:, 0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(n_sim)
        S_paths[:, i] = S_paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths, t

def dynamic_delta_hedging(S_paths, K, r, sigma, T, dt):
    n_sim, N = S_paths.shape
    portfolio_values = np.zeros((n_sim, N))
    for i in range(n_sim):
        cash = 0
        shares = 0
        for t in range(N):
            S = S_paths[i, t]
            tau = T - t*dt
            delta = black_scholes_delta(S, K, tau, r, sigma) if tau > 0 else 0
            if t == 0:
                shares = delta
                cash = black_scholes_call_price(S, K, tau, r, sigma) - shares * S
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                cash -= delta_shares * S
                shares = shares_new
            portfolio_values[i, t] = shares * S + cash
        # At maturity, settle the option
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff
    pnl = portfolio_values[:, -1]
    return pnl

def static_delta_hedging(S_paths, K, r, sigma, T, dt):
    n_sim, N = S_paths.shape
    portfolio_values = np.zeros((n_sim, N))
    for i in range(n_sim):
        S_initial = S_paths[i, 0]
        delta = black_scholes_delta(S_initial, K, T, r, sigma)
        shares = delta
        option_price = black_scholes_call_price(S_initial, K, T, r, sigma)
        cash = option_price - shares * S_initial * np.exp(r * T)
        portfolio_values[i, :] = shares * S_paths[i, :] + cash * np.exp(r * (T - np.linspace(0, T, N)))
        # At maturity, settle the option
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff
    pnl = portfolio_values[:, -1]
    return pnl

def no_hedging(S_paths, K):
    n_sim, N = S_paths.shape
    pnl = np.maximum(S_paths[:, -1] - K, 0) - black_scholes_call_price(S0, K, T, r, sigma)
    return pnl

# Simulate asset price paths
S_paths, t = simulate_gbm_paths(S0, r, sigma, T, dt, n_sim)

# Dynamic delta hedging
pnl_dynamic = dynamic_delta_hedging(S_paths, K, r, sigma, T, dt)

# Static delta hedging
pnl_static = static_delta_hedging(S_paths, K, r, sigma, T, dt)

# No hedging
pnl_no_hedging = no_hedging(S_paths, K)

# Plot results
plt.figure(figsize=(12,6))
sns.kdeplot(pnl_dynamic, label='Dynamic Delta Hedging', shade=True)
sns.kdeplot(pnl_static, label='Static Delta Hedging', shade=True)
sns.kdeplot(pnl_no_hedging, label='No Hedging', shade=True)
plt.title('PnL Distribution Comparison')
plt.xlabel('Profit and Loss')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Summary statistics
print('Dynamic Hedging PnL:')
print('Mean:', np.mean(pnl_dynamic))
print('Std Dev:', np.std(pnl_dynamic))
print('Median:', np.median(pnl_dynamic))
print('5th Percentile:', np.percentile(pnl_dynamic, 5))
print('95th Percentile:', np.percentile(pnl_dynamic, 95))
print('\nStatic Hedging PnL:')
print('Mean:', np.mean(pnl_static))
print('Std Dev:', np.std(pnl_static))
print('Median:', np.median(pnl_static))
print('5th Percentile:', np.percentile(pnl_static, 5))
print('95th Percentile:', np.percentile(pnl_static, 95))
print('\nNo Hedging PnL:')
print('Mean:', np.mean(pnl_no_hedging))
print('Std Dev:', np.std(pnl_no_hedging))
print('Median:', np.median(pnl_no_hedging))
print('5th Percentile:', np.percentile(pnl_no_hedging, 5))
print('95th Percentile:', np.percentile(pnl_no_hedging, 95))

