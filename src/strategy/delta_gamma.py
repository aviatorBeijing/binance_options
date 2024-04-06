import math
import numpy as np
import scipy as sp
import numpy.random as npr  
import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy.random as npr
#import seaborn
#seaborn.set_style("ticks")

from butil.options_calculator import callprice,fair_call_vol,putprice,fair_put_vol,deltafunc,gamma

if __name__ == '__main__':
    S0 = 100
    k  = 100 
    K2 = 100 
    T1 = 1   # time to maturity
    T2 = 1.1 # time to maturity
    sigma = 0.2 # vola
    n = 10_000 # number of simulations
    m = 252  # number of realisations of stock
    r = 0.05 # interest rate
    dt = T1/m

    s = np.zeros([n,m+1])
    w = npr.standard_normal([n,m])
    s[:,0] = S0

    #GBM
    for i in range(1,m+1):
        s[:,i] = s[:,i-1]*((1 + r*dt) + sigma / np.sqrt(252) * w[:,i-1])
    #plt.plot( s[0,:] )
    #plt.plot( s[1,:] )
    #plt.show()

    #Greeks
    bscall  = np.zeros([n,m+1])
    deltabs = np.zeros([n,m+1])
    gamma1  = np.zeros([n,m+1])
    gamma2  = np.zeros([n,m+1])
    delta2 = np.zeros([n,m+1])
    premium2 = np.zeros([n,m+1])
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
        premium2[:,i] = callprice(s[:,i], k, T2-ttm[i-1], sigma, r)
        gamma1[:, i] = gamma(s[:,i], k, T1-ttm[i-1], sigma, mu)
        gamma2[:, i] = gamma(s[:,i], k, T2-ttm[i-1], sigma, mu)
        
    #rebalance
    h2 = gamma1/gamma2
    balanced = deltabs - delta2*h2

    #Simulation
    st               = s[:,0]
    premium          = callprice(s[:,0], k, T1, sigma, mu)
    p2               = premium2[:,0]
    delta            = balanced[:,0]
    gammacoef        = h2[:,0]
    Pnl              = premium - delta*st - h2[:,0]*p2
    interest         = np.exp(r * dt)
    for i in range(1, m): # day by day
        newgammacoef = gamma1[:,i]/gamma2[:,i]
        newdelta     = deltabs[:,i] - delta2[:,i]*newgammacoef
        Pnl          = Pnl*interest - ((newdelta-delta)*s[:,i] + (newgammacoef-gammacoef)*premium2[:,i])
        
        delta        = newdelta
        gammacoef    = newgammacoef

    Call_final       = np.max(s[:,-1]-k,0)
    PnL_final        = Pnl*interest + balanced[:,-2] * s[:,-1] + h2[:,-2]*premium2[:,-1]/interest + Call_final
    PnL_final2       = PnL_final/premium[0]

    print('-- PnL:', PnL_final)

    plt.hist(PnL_final2)
    plt.show()
