import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

# Option parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
T = 1.0        # Time to maturity in years
dt = 1/252     # Time step (daily)
N = int(T/dt)  # Number of time steps
n_sim = 10   # Number of simulations

# Black-Scholes functions
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def black_scholes_gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# Simulate GBM paths
def simulate_gbm_paths(S0, r, sigma, T, dt, n_sim):
    N = int(T/dt)
    S_paths = np.zeros((n_sim, N))
    S_paths[:, 0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(n_sim)
        S_paths[:, i] = S_paths[:, i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths

# Gamma scalping strategy
def gamma_scalping(S_paths, K, r, sigma, T, dt):
    n_sim, N = S_paths.shape
    portfolio_values = np.zeros((n_sim, N))
    
    for i in tqdm( range(n_sim)): 
        cash = 0
        shares = 0
        for t in range(N):
            S = S_paths[i, t]
            tau = T - t*dt
            
            delta = black_scholes_delta(S, K, tau, r, sigma) if tau > 0 else 0
            gamma = black_scholes_gamma(S, K, tau, r, sigma) if tau > 0 else 0
            
            if t == 0:
                shares = delta
                cash = black_scholes_call_price(S, K, tau, r, sigma) - shares * S
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                cash -= delta_shares * S
                shares = shares_new

                # Gamma scalping: profit from mean-reverting price movements
                if t > 0 and np.abs(S_paths[i, t] - S_paths[i, t-1]) > 0:
                    gamma_scalping_pnl = 0.5 * gamma * (S_paths[i, t] - S_paths[i, t-1])**2
                    cash += gamma_scalping_pnl
            
            portfolio_values[i, t] = shares * S + cash
        
        # At maturity, settle the option
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff
    
    pnl = portfolio_values[:, -1]
    return pnl

# Simulate asset price paths
S_paths = simulate_gbm_paths(S0, r, sigma, T, dt, n_sim)

# Gamma scalping PnL
pnl_gamma_scalping = gamma_scalping(S_paths, K, r, sigma, T, dt)

# Plot PnL distribution for gamma scalping
plt.figure(figsize=(12,6))
sns.kdeplot(pnl_gamma_scalping, label='Gamma Scalping', shade=True)
plt.title('PnL Distribution for Gamma Scalping')
plt.xlabel('Profit and Loss')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Summary statistics
print('Gamma Scalping PnL:')
print('Mean:', np.mean(pnl_gamma_scalping))
print('Std Dev:', np.std(pnl_gamma_scalping))
print('Median:', np.median(pnl_gamma_scalping))
print('5th Percentile:', np.percentile(pnl_gamma_scalping, 5))
print('95th Percentile:', np.percentile(pnl_gamma_scalping, 95))
