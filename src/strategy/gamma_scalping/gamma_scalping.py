import numpy as np,os
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
n_sim = 100   # Number of simulations

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
    
    for i in tqdm( range(n_sim) ):
        cash = 0
        shares = 0
        p0 = -1
        price_at_rebalance = 0.
        for t in range(N): # price path
            S = S_paths[i, t]
            tau = T - t*dt
            
            delta = black_scholes_delta(S, K, tau, r, sigma) if tau > 0 else 0
            gamma = black_scholes_gamma(S, K, tau, r, sigma) if tau > 0 else 0
            
            if t == 0:
                p0 = S
                shares = delta
                premium = black_scholes_call_price(S, K, tau, r, sigma) 
                cash = premium - shares * S
                price_at_rebalance = p0 # init
                #print('\t sell 1 call gain: $', premium )
                #print(f'buy {shares:.2f} stock cost: $', shares * S, ', net capital: $', cash )
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                trade_volume = delta_shares * S
                
                # Gamma scalping: profit from mean-reverting price movements
                p1 = S_paths[i, t]
                if t > 0 and np.abs(p1-price_at_rebalance) > 0:
                    gamma_scalping_pnl = 0.5 * gamma * (p1- price_at_rebalance)**2
                    cash += gamma_scalping_pnl
                    rtn = gamma_scalping_pnl/(abs(trade_volume))*10_000
                    if rtn>50: # Trade only if the potential gain is great enough.
                        cash -= trade_volume
                        shares = shares_new
                        price_at_rebalance = p1

                        """
                        if delta_shares>0:
                            print(f'buy {delta_shares:.6f} stock')
                        elif delta_shares<0:
                            print(f'\tsell {-delta_shares:.6f} stock')
                        
                        print(f'\t\t\tscalping pnl: $ {gamma_scalping_pnl:.3f}, {rtn:.0f} bps (prices: {p0} --> {p1}')
                        """
                    else:
                        pass
                p0 = p1 # Update price

            portfolio_values[i, t] = shares * S + cash
        
        # At maturity, settle the option
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff
    
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
