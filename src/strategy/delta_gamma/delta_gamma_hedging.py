import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define Parameters
S0 = 100      # Initial stock price
K = 100       # Strike price (ATM)
T = 1         # Time to maturity in years
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility
N = 252       # Number of trading days
M = 10        # Number of simulation paths
dt = T / N    # Time step
trading_fee = 0.001  # 0.1% trading fee
margin_cost = 0.01   # 1% margin cost

# Simulation of GBM Paths
def simulate_gbm_paths(S0, r, sigma, T, N, M):
    dt = T / N
    S = np.zeros((N + 1, M))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal(0, 1, M)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

# Calculate Black-Scholes option price and Greeks
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    price = S * N_d1 - K * np.exp(-r * T) * N_d2
    delta = N_d1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) # Vega is not used in this example, but it's useful for other scenarios
    return price, delta, gamma

# Hedging Strategy Simulation
def simulate_hedging(S, K, T, r, sigma, trading_fee, margin_cost):
    N, M = S.shape
    pnl = np.zeros(M)
    fees = np.zeros(N)
    margin_costs = np.zeros(N)
    delta_hist = np.zeros((N, M))
    gamma_hist = np.zeros((N, M))
    
    # Initial option and spot prices
    option_prices, deltas, gammas = black_scholes(S[0], K, T, r, sigma)
    
    # Initial positions
    option_position = np.ones(M)  # Long 1 ATM option initially
    spot_position = -deltas  # Initial hedge position in the spot
    
    for t in range(N):
        if t > 0:
            # Time to maturity and recalculation
            time_to_maturity = (T - t * dt)
            option_prices, deltas, gammas = black_scholes(S[t], K, time_to_maturity, r, sigma)
        
        delta_hist[t] = deltas
        gamma_hist[t] = gammas
        
        # Delta and Gamma adjustments
        target_delta = deltas
        target_gamma = gammas
        
        # Calculate necessary hedge adjustments
        delta_adjustment = target_delta - spot_position
        gamma_adjustment = target_gamma - (option_position * gammas)
        
        # Adjust spot positions
        if np.abs(delta_adjustment).mean() > 0.005:
            fees[t] = trading_fee * np.abs(delta_adjustment).mean()
            margin_costs[t] = margin_cost * np.abs(delta_adjustment).mean()
            spot_position += delta_adjustment
        
        # Adjust option positions
        if np.abs(gamma_adjustment).mean() > 0.005:
            fees[t] += trading_fee * np.abs(gamma_adjustment).mean()
            margin_costs[t] += margin_cost * np.abs(gamma_adjustment).mean()
            option_position += gamma_adjustment / gammas
        
        # Calculate P&L
        pnl += option_prices - spot_position * S[t] - option_position * option_prices
    
    return pnl, fees, margin_costs, delta_hist, gamma_hist

# Plotting Results
def plot_results(pnl, fees, margin_costs, delta_hist, gamma_hist):
    plt.figure(figsize=(14, 10))
    
    # Final P&L
    plt.subplot(2, 2, 1)
    plt.hist(pnl, bins=10, edgecolor='black')
    plt.title('Final P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    
    # Fee and Margin Costs
    plt.subplot(2, 2, 2)
    plt.plot(fees, label='Trading Fees')
    plt.plot(margin_costs, label='Margin Costs')
    plt.title('Fees and Margin Costs Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cost')
    plt.legend()
    
    # Delta and Gamma
    plt.subplot(2, 2, 3)
    plt.plot(delta_hist, label='Delta')
    plt.plot(gamma_hist, label='Gamma')
    plt.title('Delta and Gamma Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main Simulation
S = simulate_gbm_paths(S0, r, sigma, T, N, M)
pnl, fees, margin_costs, delta_hist, gamma_hist = simulate_hedging(S, K, T, r, sigma, trading_fee, margin_cost)
plot_results(pnl, fees, margin_costs, delta_hist, gamma_hist)

