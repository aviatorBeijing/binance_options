from butil.options_calculator import deltafunc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
# Parameters
S0 = 100           # Initial price
K = 100            # Strike price
mu = 0.05          # Drift
sigma = 0.2        # Volatility
r = 0.05           # Risk-free rate
T = 1              # Time to maturity (1 year)
n = 252            # Number of steps (daily steps)
n_simulations = 10 # Number of simulation paths
dt = T / n         # Time increment
leverage_ratio = 1  # Leverage ratio

# Trading parameters
transaction_fee = 0.005   # Fee per transaction
price_threshold = 0.005   # 0.5% price move to trigger rebalancing
margin_cost_rate = 0.02   # 2% annual margin cost

# Liquidation threshold (assuming margin requirement of 50%)
liquidation_threshold = 0.5

# Simulate GBM paths
np.random.seed(47) # Set seed for reproducibility
dW = np.random.normal(0, np.sqrt(dt), (n_simulations, n))
price_paths = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dW, axis=1))
price_paths = np.hstack([np.full((n_simulations, 1), S0), price_paths])

# Black-Scholes option pricing formula for call
def black_scholes_call(S, K, T, r, sigma):
    from butil.options_calculator import callprice
    return callprice(S,K,T,sigma, r)

# Delta of the call option
def call_delta(S, K, T, r, sigma):
    from butil.options_calculator import deltafunc
    return deltafunc(S,K,T,sigma,r)

# Initialize tracking arrays
PnL = np.zeros((n_simulations, n + 1))
fee_costs = np.zeros((n_simulations, n + 1))
margin_costs = np.zeros((n_simulations, n + 1))
liquidated = np.zeros(n_simulations, dtype=bool)

# Gamma scalping strategy with cost tracking and liquidation check
for i in tqdm( range(n_simulations) ):
    # Initial option and delta
    delta = call_delta(S0, K, T, r, sigma)
    hedge_position = delta * leverage_ratio
    cash_position = -delta * S0 * leverage_ratio
    margin_cost = -cash_position * (margin_cost_rate / 252)
    
    premium = black_scholes_call(S0, K, T, r, sigma)
    PnL[i, 0] = premium - transaction_fee - margin_cost
    fee_costs[i, 0] = transaction_fee
    margin_costs[i, 0] = margin_cost
    
    for t in range(1, n + 1):
        if liquidated[i]:
            break
        
        # Update option value and delta
        S = price_paths[i, t]
        tau = T - t * dt
        Vt = black_scholes_call(S, K, tau, r, sigma)
        new_delta = call_delta(S, K, tau, r, sigma)
        price_shift = abs(S - price_paths[i, t-1]) / price_paths[i, t-1]

        if price_shift>price_threshold:
            delta_adjustment = new_delta - hedge_position / leverage_ratio
            cash_position -= delta_adjustment * S * leverage_ratio
            hedge_position = new_delta * leverage_ratio
            
            transaction_cost = abs(delta_adjustment) * transaction_fee * leverage_ratio
            margin_cost = cash_position * (margin_cost_rate / 252)
            
            fee_costs[i, t] = transaction_cost
            margin_costs[i, t] = margin_cost
            PnL[i, t] = PnL[i, t-1] + Vt + delta_adjustment * S * leverage_ratio - transaction_cost - margin_cost
        else:
            # Accumulate margin cost
            margin_cost = -cash_position * (margin_cost_rate / 252)
            fee_costs[i, t] = 0
            margin_costs[i, t] = margin_cost
            PnL[i, t] = PnL[i, t-1] - abs(margin_cost)

        # Check for liquidation scenario
        #if PnL[i, t] / (delta * S * leverage_ratio) < liquidation_threshold:
        #    liquidated[i] = True
        #    print(f"Simulation {i+1} liquidated at step {t} with P&L {PnL[i, t]:.4f}")
        #    break

# Plot the results
plt.figure(figsize=(14, 10))

# Subplot 1: P&L values
plt.subplot(2, 2, 1)
for i in range(n_simulations):
    plt.plot(PnL[i], label=f'Path {i+1}' if not liquidated[i] else f'Path {i+1} (Liquidated)')
plt.title('P&L of Gamma Scalping Strategy')
plt.xlabel('Time Steps')
plt.ylabel('P&L')
plt.legend(loc='upper left')
plt.grid(True)

# Subplot 2: P&L distribution
plt.subplot(2, 2, 2)
final_PnL = PnL[:, -1]
#plt.hist(final_PnL, bins=15, density=True, alpha=0.6, color='blue')
sns.kdeplot(final_PnL, label='P&L Density Distribution', fill=True)
plt.title('Final P&L Density Distribution')
plt.xlabel('Final P&L')
plt.ylabel('Density')
plt.grid(True)

# Subplot 3: Fee costs
plt.subplot(2, 2, 3)
for i in range(n_simulations):
    plt.plot(fee_costs[i], label=f'Path {i+1}')
plt.title('Fee Costs in Gamma Scalping Strategy')
plt.xlabel('Time Steps')
plt.ylabel('Fee Cost')
plt.legend(loc='upper right')
plt.grid(True)

# Subplot 4: Margin costs
plt.subplot(2, 2, 4)
for i in range(n_simulations):
    plt.plot(margin_costs[i], label=f'Path {i+1}')
plt.title('Margin Costs in Gamma Scalping Strategy')
plt.xlabel('Time Steps')
plt.ylabel('Margin Cost')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
import os;fn=os.getenv('USER_HOME','')+'/tmp/gamma_scalping-long_call.png'
plt.savefig( fn )
print('-- saved:', fn)
