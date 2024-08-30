import numpy as np,os
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

from butil.options_calculator import callprice
from butil.options_calculator import deltafunc
from butil.options_calculator import gamma as calc_gamma

# Option parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
r = 0.05       # Risk-free interest rate
sigma = 0.5    # Volatility
T = 1.0        # Time to maturity in years
dt = 1/252     # Time step (daily)
N = int(T/dt)  # Number of time steps
n_sim = 100   # Number of simulations

# Simulate GBM paths
np.random.seed(47) # Set seed for reproducibility
def simulate_gbm_paths(S0, r, sigma, T, dt, n_sim):
    N = int(T/dt)
    S_paths = np.zeros((n_sim, N))
    S_paths[:, 0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(n_sim)
        mu = r - 0.5 * sigma**2 # Drifting
        S_paths[:, i] = S_paths[:, i-1] * np.exp(mu * dt + sigma * np.sqrt(dt) * z)
    return S_paths

# Gamma scalping strategy
def gamma_scalping(S_paths, K, r, sigma, T, dt):
    n_sim, N = S_paths.shape
    portfolio_values = np.zeros((n_sim, N))
    
    for i in tqdm( range(n_sim) ):
        cash = 0
        shares = 0
        price_at_rebalance = 0.
        for t in range(N): # price path
            S = S_paths[i, t]
            tau = T - t*dt
            
            delta = deltafunc(S, K, tau, sigma, r) if tau > 0 else 0
            gamma = calc_gamma(S, K, tau, sigma, r) if tau > 0 else 0
            
            if t == 0:
                shares = delta
                premium = callprice(S, K, tau, sigma, r) 
                cash = premium - shares * S     # sell options & buy shares
                price_at_rebalance = S # init & record the 1st rebalance by buying shares
                #print('\t sell 1 call gain: $', premium )
                #print(f'buy {shares:.2f} stock cost: $', shares * S, ', net capital: $', cash )
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                trade_volume = delta_shares * S
                
                # Gamma scalping: profit from mean-reverting price movements
                if abs(trade_volume)>0 and t > 0 and np.abs(S - price_at_rebalance) > 0:
                    gamma_scalping_pnl = 0.5 * gamma * (S - price_at_rebalance)**2
                    rtn = gamma_scalping_pnl/(abs(trade_volume))*10_000
                    if rtn>50: # Trade only if the potential gain is great enough.
                        cash += gamma_scalping_pnl
                        cash -= trade_volume
                        shares = shares_new
                        price_at_rebalance = S

                        """
                        if delta_shares>0:
                            print(f'buy {delta_shares:.6f} stock')
                        elif delta_shares<0:
                            print(f'\tsell {-delta_shares:.6f} stock')
                        """
                        #print(f'\t\t\tscalping pnl: $ {gamma_scalping_pnl:.3f}, {rtn:.0f} bps (prices: {p0} --> {p1}')
                        
                    else:
                        pass

            portfolio_values[i, t] = shares * S + cash
        
        # At maturity, settle the option
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff   # Because the sell position on options
    
    pnl = portfolio_values[:, -1]
    return pnl

S_paths = simulate_gbm_paths(S0, r, sigma, T, dt, n_sim)
pnl_gamma_scalping = gamma_scalping(S_paths, K, r, sigma, T, dt)

# Summary statistics
print('Gamma Scalping PnL:')
print('Mean:', np.mean(pnl_gamma_scalping))
print('Std Dev:', np.std(pnl_gamma_scalping))
print('Median:', np.median(pnl_gamma_scalping))
print('5th Percentile:', np.percentile(pnl_gamma_scalping, 5))
print('95th Percentile:', np.percentile(pnl_gamma_scalping, 95))


plt.figure(figsize=(12,6))
sns.kdeplot(pnl_gamma_scalping, label='Gamma Scalping', fill=True)
plt.title('PnL Distribution for Gamma Scalping')
plt.xlabel('Profit and Loss')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

fn = os.getenv('USER_HOME','') + '/tmp/gamma_scalping.png'
plt.savefig(fn)
print('-- saved:', fn)
