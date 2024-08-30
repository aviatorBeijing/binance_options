import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Parameters
S0 = 100     # Initial stock price
mu = 0.05    # Drift
sigma = 0.2  # Volatility
T = 1.0      # Time to maturity
dt = 0.01    # Time step
N_paths = 10 # Number of paths

K = 100      # Strike price
r = 0.01     # Risk-free rate

def simulate_gbm_paths(S0, mu, sigma, T, dt, N_paths):
    """Simulate GBM paths."""
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    S = np.zeros((N_paths, N+1))
    S[:, 0] = S0
    
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), N_paths)
        S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return t, S

def black_scholes_delta(S, K, T, r, sigma):
    """Calculate delta of a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def black_scholes_gamma(S, K, T, r, sigma):
    """Calculate gamma of a European call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def gamma_hedging(S, K, T, r, sigma, initial_position, N_paths, dt):
    """Gamma hedging."""
    N = S.shape[1]
    delta = np.zeros(N_paths)
    gamma = np.zeros(N_paths)
    PnL = np.zeros((N_paths, N))
    
    for i in range(N_paths):
        hedge_position = initial_position
        for t_index in range(1, N):
            current_time = t[t_index]
            current_price = S[i, t_index]
            remaining_time = T - current_time
            
            delta[i] = black_scholes_delta(current_price, K, remaining_time, r, sigma)
            gamma[i] = black_scholes_gamma(current_price, K, remaining_time, r, sigma)
            
            # Adjust hedge position
            hedge_position = delta[i] * S0 - gamma[i] * (S0 ** 2)
            option_value = max(current_price - K, 0)  # Simplified call payoff
            PnL[i, t_index] = option_value - hedge_position
        
    return PnL

def delta_gamma_hedging(S, K, T, r, sigma, initial_position, N_paths, dt):
    """Delta-Gamma hedging."""
    N = S.shape[1]
    delta = np.zeros(N_paths)
    gamma = np.zeros(N_paths)
    PnL = np.zeros((N_paths, N))
    
    for i in range(N_paths):
        hedge_position = initial_position
        for t_index in range(1, N):
            current_time = t[t_index]
            current_price = S[i, t_index]
            remaining_time = T - current_time
            
            delta[i] = black_scholes_delta(current_price, K, remaining_time, r, sigma)
            gamma[i] = black_scholes_gamma(current_price, K, remaining_time, r, sigma)
            
            # Adjust hedge position
            hedge_position = delta[i] * S0 - 0.5 * gamma[i] * (S0 ** 2)
            option_value = max(current_price - K, 0)  # Simplified call payoff
            PnL[i, t_index] = option_value - hedge_position
        
    return PnL

# Simulate GBM paths
t, S = simulate_gbm_paths(S0, mu, sigma, T, dt, N_paths)

# Calculate P&L for both strategies
PnL_gamma = gamma_hedging(S, K, T, r, sigma, 0, N_paths, dt)
PnL_delta_gamma = delta_gamma_hedging(S, K, T, r, sigma, 0, N_paths, dt)

# Final P&L values
final_PnL_gamma = PnL_gamma[:, -2]
final_PnL_delta_gamma = PnL_delta_gamma[:, -2]

# Plot P&L and distributions
plt.figure(figsize=(18, 12))

# P&L for Gamma Hedging
plt.subplot(2, 2, 1)
for i in range(N_paths):
    plt.plot(t[1:], PnL_gamma[i, 1:], lw=1, label=f'Path {i+1}')
plt.title('P&L of Gamma Hedging')
plt.xlabel('Time')
plt.ylabel('P&L')
plt.grid(True)
plt.legend()

# P&L for Delta-Gamma Hedging
plt.subplot(2, 2, 2)
for i in range(N_paths):
    plt.plot(t[1:], PnL_delta_gamma[i, 1:], lw=1, label=f'Path {i+1}')
plt.title('P&L of Delta-Gamma Hedging')
plt.xlabel('Time')
plt.ylabel('P&L')
plt.grid(True)
plt.legend()

# Distribution of Final P&L for Gamma Hedging
plt.subplot(2, 2, 3)
sns.kdeplot(final_PnL_gamma, fill=True, color='blue', label='Gamma Hedging')
plt.title('Distribution of Final P&L (Gamma Hedging)')
plt.xlabel('Final P&L')
plt.ylabel('Density')
plt.grid(True)

# Distribution of Final P&L for Delta-Gamma Hedging
plt.subplot(2, 2, 4)
sns.kdeplot(final_PnL_delta_gamma, fill=True, color='red', label='Delta-Gamma Hedging')
plt.title('Distribution of Final P&L (Delta-Gamma Hedging)')
plt.xlabel('Final P&L')
plt.ylabel('Density')
plt.grid(True)

plt.tight_layout()
plt.savefig('foobar.png')

