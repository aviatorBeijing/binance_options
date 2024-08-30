import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 56_000  # Initial asset price
K = 60_000   # Strike price (ATM option)
r = 0.05  # Risk-free rate
T = 1.0   # Time to maturity (in years)
sigma = 0.5  # Volatility of the underlying asset
n_simulations = 100  # Number of simulations

# Simulate terminal asset prices using GBM
np.random.seed(47)
Z = np.random.standard_normal(n_simulations)
Vt = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# Calculate the payoff for the straddle strategy
call_payoff = np.maximum(Vt - K, 0)
put_payoff = np.maximum(K - Vt, 0)
straddle_payoff = call_payoff + put_payoff

# Total cost of the straddle (premium paid for both options)
# Assuming Black-Scholes pricing, we'll calculate the premium for call and put
# Black-Scholes formula for call and put options
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    from scipy.stats import norm
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

call_premium = black_scholes_price(S0, K, T, r, sigma, option_type="call")
put_premium = black_scholes_price(S0, K, T, r, sigma, option_type="put")
total_premium_paid = call_premium + put_premium

# P&L for the straddle
straddle_pnl = straddle_payoff - total_premium_paid

# Plotting the relation between asset price and the straddle P&L
plt.figure(figsize=(10, 6))
plt.scatter(Vt, straddle_pnl, color="blue", alpha=0.5, s=10, label="Simulated P&L")
plt.axhline(0, color="red", linestyle="--", label="Break-even line")
plt.title("Straddle Strategy P&L vs. Asset Price at Maturity")
plt.xlabel("Asset Price at Maturity (S_T)")
plt.ylabel("Straddle P&L")
plt.legend()
plt.grid(True)
import os;fn = os.getenv('USER_HOME','') + '/tmp/gbm_straddle.png'
plt.savefig( fn )
print('-- saved:', fn)


