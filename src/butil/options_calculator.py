import numpy as np
import scipy.stats as scs

np.set_printoptions(suppress=True)
"""
Black-Schole-Merton Theory (BSM)
European-style Options
"""

# Call option pprice
def callprice(S,K,T,sigma,r)->float:
    d1=(np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2=(np.log(S/K) + (r - 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return S*scs.norm.cdf(d1) - np.exp(-r *T) * K * scs.norm.cdf(d2)

# Parity: C+PV(S) = P+S
def putprice(S,K,T,sigma,r)->float:
    C = callprice( S,K,T,sigma, r)
    return C + K*np.exp(-r*T) -S

# Delta
def deltafunc(S, K, T, sigma, r)->float:
    d1=(np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return scs.norm.cdf(d1) 

# Gamma (call), d_put + d_call = 1
def gamma(S, K, T, sigma, r)->float:
    d1=(np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return scs.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def fair_call_vol(c, S,K,T,r=0):
    epsilon = 999.
    best = -70
    for vol in np.arange(0.1,200,0.1):
        x = callprice(S,K,T,vol/100,r)
        if abs(c-x)<epsilon:
            epsilon  = abs(c-x)
            best = vol
    return best 
def fair_put_vol(p, S,K,T,r=0):
    epsilon = 999.
    best = -70
    for vol in np.arange(0.1,200,0.1):
        x = putprice(S,K,T,vol/100,r)
        #print(x,p,S,K,T,vol)
        if abs(p-x)<epsilon:
            epsilon  = abs(p-x)
            best = vol
    return best 

def invert_(c,K,T,sigma,r, is_call=True):
    """
    @brief Find spot price that matchs the given call price.
    @param c: the call price
    """
    # Coarse search
    forward_ = callprice if is_call else putprice
    spots = np.arange(K//2,K*2, 100)
    calls = list(map(lambda s: forward_(s,K,T,sigma,r), spots))
    calls_diff = np.array( calls) - c
    srange = spots[ np.where(abs(calls_diff)<150) ]
    n = srange.shape[0]
    i = srange[n//2-1]
    j = srange[n//2+1]

    # Fine search
    spots = np.arange(i,j,.5) # FIXME for BTC/USDT only, for DOGE, this might need a smaller steps.
    calls = list(map(lambda s: forward_(s,K,T,sigma,r), spots))
    calls_diff = np.array( calls) - c
    srange = spots[ np.where(abs(calls_diff)<2) ]
    n = srange.shape[0]
    spot = srange[n//2] if n%2==1 else np.mean( srange[n//2-1:n//2])
    return spot

def invert_callprice(c,K,T,sigma,r):
    return invert_(c,K,T,sigma,r,is_call=True)  

def invert_putprice(p,K,T,sigma,r):
    return invert_(p,K,T,sigma,r,is_call=False)       

#test
if __name__ == '__main__':
    c=1_000
    K = 57_000
    T = 3/365
    sigma = 0.7
    r = 0.25

    xS = invert_callprice(3600.,K,T,sigma,r)
    print( callprice(xS, K,T,sigma, r), ' == ', 3600, ', calculated spot:', xS)

    xS = invert_putprice(1600.,K,T,sigma,r)
    print( putprice(xS, K, T, sigma, r ), ' == ', 1600, ', calculated spot:', xS )
    