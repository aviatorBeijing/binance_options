import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gamma_core import simulate_gbm_paths, gamma_scalping

# Option parameters
S0 = 100       
K = 100        
r = 0.05       
sigma = 0.2    
T = 1.0        
dt = 1/252     
n_sim = 1000   

# Simulate asset price paths using Cython-optimized function
S_paths = simulate_gbm_paths(S0, r, sigma, T, dt, n_sim)

# Gamma scalping PnL using Cython-optimized function
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

