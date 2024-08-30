import numpy as np,os
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

from butil.options_calculator import callprice
from butil.options_calculator import deltafunc
from butil.options_calculator import gamma as calc_gamma

plt.style.use('fivethirtyeight')

# Option parameters
S0 = 60000       # Initial stock price
K = 60000        # Strike price
r = 0.05       # Risk-free interest rate
sigma = 0.5    # Volatility
T = 10.0        # Time to maturity in years
dt = 1/252     # Time step (daily)
N = int(T/dt)  # Number of time steps
fee_rate = .5/100   # Spot trading fee rate
n_sim = 10   # Number of simulations

# Simulate GBM paths
#np.random.seed(47) # Set seed for reproducibility
def simulate_gbm_paths(S0, r, sigma, T, dt, n_sim):
    N = int(T/dt)
    S_paths = np.zeros((n_sim, N))
    S_paths[:, 0] = S0
    for i in range(1, N):
        z = np.random.standard_normal(n_sim)
        mu = r - 0.5 * sigma**2 # Drifting
        S_paths[:, i] = S_paths[:, i-1] * np.exp(mu * dt + sigma * np.sqrt(dt) * z)
    return S_paths

def simulate_jump_paths(S0, sigma, T, dt, n_sim):
    # Parameters
    N = int(T/dt)        # Number of time steps
    mu = 0.1        # Drift coefficient

    # Jump parameters
    lambda_jump = 0.1  # Jump intensity (average number of jumps per year)
    mu_jump = 0.05     # Mean of jump size
    sigma_jump = 0.1   # Std deviation of jump size

    # Simulate GBM with jumps for multiple paths
    all_paths = np.zeros((n_sim, N))

    for i in range(n_sim):
        S = np.zeros(N)
        S[0] = S0
        
        for t in range(1, N):
            # Standard GBM term
            Wt = np.random.normal(0, np.sqrt(dt))
            dS = mu * S[t-1] * dt + sigma * S[t-1] * Wt
            
            # Jump component
            if np.random.poisson(lambda_jump * dt) > 0:
                Yt = np.random.normal(mu_jump, sigma_jump)
                S[t] = S[t-1] * (1 + dS / S[t-1]) * np.exp(Yt)
            else:
                S[t] = S[t-1] * (1 + dS / S[t-1])
        
        all_paths[i] = S

    return all_paths

# Gamma scalping strategy
def gamma_scalping(S_paths, K, r, sigma, T, dt):
    n_sim, N = S_paths.shape
    portfolio_values = np.zeros((n_sim, N))
    cum_fees = np.zeros(n_sim)
    cum_vols = np.zeros(n_sim) # $ amount
    cum_amt = np.zeros(n_sim)  # count 
    
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
                cum_fees[i] += abs(shares*S*fee_rate)
                cum_vols[i] += abs(shares*S)
                cum_amt[i] += shares
                #print('\t sell 1 call gain: $', premium )
                #print(f'buy {shares:.2f} stock cost: $', shares * S, ', net capital: $', cash )
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                trade_volume = delta_shares * S
                
                # Gamma scalping: profit from mean-reverting price movements
                if abs(trade_volume)>0 and t > 0 and np.abs(S - price_at_rebalance) > 0:
                    fee = abs(trade_volume) * fee_rate
                    gamma_scalping_pnl = 0.5 * gamma * (S - price_at_rebalance)**2 - fee
                    rtn = gamma_scalping_pnl/(abs(trade_volume))*10_000
                    if rtn>50: # Trade only if the potential gain is great enough.
                        cash += gamma_scalping_pnl
                        cash -= trade_volume
                        shares = shares_new
                        price_at_rebalance = S
                        cum_fees[i] += fee
                        cum_vols[i] += abs(trade_volume)
                        cum_amt[i] += delta_shares

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
    return pnl, cum_fees, cum_vols, cum_amt

#S_paths = simulate_gbm_paths(S0, r, sigma, T, dt, n_sim)
S_paths = simulate_jump_paths(S0, sigma, T, dt, n_sim)
pnl_gamma_scalping, cum_fees, cum_vols, cum_amt = gamma_scalping(S_paths, K, r, sigma, T, dt)

# Summary statistics
print('Gamma Scalping PnL:')
print('\tMean:', np.mean(pnl_gamma_scalping))
print('\tStd Dev:', np.std(pnl_gamma_scalping))
print('\tMedian:', np.median(pnl_gamma_scalping))
print('\t5th Percentile:', np.percentile(pnl_gamma_scalping, 5))
print('\t95th Percentile:', np.percentile(pnl_gamma_scalping, 95))


plt.figure(figsize=(24,16))

nrow = 2
ncol = 3

plt.subplot(nrow, ncol, 1)
sns.kdeplot(pnl_gamma_scalping, label='Net Gain', fill=True)
plt.title(f'Net PnL Distribution for Scalping ({n_sim} sims)')
plt.legend()

plt.subplot(nrow, ncol, 2)
sns.kdeplot(cum_fees, label='Fee', fill=True)
sns.kdeplot(pnl_gamma_scalping, label='Net Gain', fill=True)
plt.title(f'Fee Distribution (rate={(fee_rate*100):.2f}%)')
plt.legend()

plt.subplot(nrow, ncol, 3)
sns.kdeplot(cum_vols, label='Volume', fill=True)
plt.title('Trading Volume ($) Distribution')
plt.legend()

plt.subplot(nrow, ncol, 4)
sns.kdeplot(cum_amt, label='Volume', fill=True)
plt.title('Trading Amount (#) Distribution')
plt.legend()

plt.subplot(nrow, ncol, 5)
for i in range(5): plt.plot( S_paths[i,:] )
plt.title(f'Path Examples ({T} yrs)')
plt.ylabel('Price ($)')
plt.legend()

plt.grid(True)

fn = os.getenv('USER_HOME','') + '/tmp/gamma_scalping.png'
plt.savefig(fn)
print('-- saved:', fn)
