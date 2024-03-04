import math
import numpy as np
import scipy as sp
import numpy.random as npr  
import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy.random as npr
import seaborn
seaborn.set_style("ticks")

np.set_printoptions(suppress=True)

# Call option pprice
def callprice(S,K,T,sigma,r):
    d1=(sp.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * sp.sqrt(T))
    d2=(sp.log(S/K) + (r - 0.5 * sigma**2)*T) / (sigma * sp.sqrt(T))
    return S*scs.norm.cdf(d1) - math.exp(-r *T) * K * scs.norm.cdf(d2)

# Delta
def deltafunc(S, K, T, sigma, r):
    d1=(sp.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * sp.sqrt(T))
    return scs.norm.cdf(d1) 

# Gamma
def gamma(S, K, T, sigma, r):
    d1=(sp.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * sp.sqrt(T))
    return scs.norm.pdf(d1) / (S * sigma * sp.sqrt(T))

S0 = 100
k  = 100 
K2 = 100 
T1 = 1   # time to maturity
T2 = 1.1 # time to maturity
sigma = 0.2 # vola
n = 10000 # number of simulations
m = 252  # number of realisations of stock
r = 0.05 # interest rate
dt = T1/m

s = np.zeros([n,m+1])
w = npr.standard_normal([n,m])
s[:,0] = S0

#GBM
for i in range(1,m+1):
    s[:,i] = s[:,i-1]*((1 + r*dt) + sigma / np.sqrt(252) * w[:,i-1])

#Greeks
bscall  = np.zeros([n,m+1])
deltabs = np.zeros([n,m+1])
gamma1  = np.zeros([n,m+1])
gamma2  = np.zeros([n,m+1])
delta2 = np.zeros([n,m+1])
ttm     = np.arange(1, m+1, 1)/m
mu = r

bscall[:,0]     = callprice(S0, k, T1, sigma, mu)
deltabs[:,0]    = deltafunc(S0, k, T1, sigma, mu)
delta2[:,0] = deltafunc(s[:,0], k, T2, sigma, mu)
gamma1[:,0] = gamma(s[:,0], k, T1, sigma, mu)
gamma2[:,0] = gamma(s[:,0], k, T2, sigma, mu)
for i in range(1,m+1):
    bscall[:,i]  = callprice(s[:,i], k, T1-ttm[i-1], sigma, mu)
    deltabs[:,i] = deltafunc(s[:,i], k, T1-ttm[i-1], sigma, mu)
    delta2[:, i] = deltafunc(s[:,i], k, T2-ttm[i-1], sigma, mu)
    gamma1[:, i] = gamma(s[:,i], k, T1-ttm[i-1], sigma, mu)
    gamma2[:, i] = gamma(s[:,i], k, T2-ttm[i-1], sigma, mu)
    
#rebalance
h2 = gamma1/gamma2
strategy = deltabs - delta2*h2

#Simulation
st               = s[:,0]
amount           = callprice(s[:,0], k, T1, sigma, mu)
delta            = deltabs[:,0] - delta2[:,0]*gamma1[:,0]/gamma2[:,0]
gammacoef        = gamma1[:,0]/gamma2[:,0]
Pnl              = amount - delta*st - gamma1[:,0]/gamma2[:,0]*callprice(s[:,0], k, T2, sigma, mu)
interest         = np.exp(r * dt)
for i in range(1, m):
    Pnl          = interest * Pnl
    newdelta     = deltabs[:,i] - delta2[:,i]*gamma1[:,i]/gamma2[:,i]
    newgammacoef = gamma1[:,i]/gamma2[:,i]
    Pnl          = Pnl - ((newdelta-delta)*s[:,i] + (newgammacoef-gammacoef)*callprice(s[:,i], k, T2-ttm[i-1], sigma, r))
    delta        = newdelta
    gammacoef    = newgammacoef

Pnl              = Pnl * interest
PnL_final        = Pnl + strategy[:,-2] * s[:,-1] + gamma1[:,-2]/gamma2[:,-2]*callprice(s[:,-1], k, T2-ttm[-1], sigma, r)/interest - np.max(s[:,-1]-k,0)
PnL_final2       = [x/callprice(s[0,0], k, T1, sigma, mu) for x in PnL_final]

print('-- PnL:', PnL_final)

#plt.hist(PnL_final2)
#plt.show()