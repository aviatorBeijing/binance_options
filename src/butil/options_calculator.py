import numpy as np
import scipy.stats as scs
import datetime 

np.set_printoptions(suppress=True)
"""
Black-Schole-Merton Theory (BSM)
European-style Options
"""


def get_maturity(contract):
    fds = contract.split('-')
    ts = datetime.datetime.strptime('20'+fds[1], '%Y%m%d') + datetime.timedelta(hours=8) # Settle at 08:00 (UTC) of the date.
    tnow = datetime.datetime.utcnow()
    dt = (ts-tnow).total_seconds()/3600./24
    return dt

def extract_specs( contract):
    sym,expiry,strike,typ = contract.split('-')
    maturity = get_maturity( contract )
    strike = float(strike)
    ctype = 'call' if typ=='C' else 'put'
    spot = f'{sym.upper()}/USDT'
    return spot, maturity, strike, ctype

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
    spots = np.arange(K/2,K*2, 100)
    calls = list(map(lambda s: forward_(s,K,T,sigma,r), spots))
    calls_diff = np.array( calls) - c
    
    if is_call:
        i = spots[ np.where(calls_diff<0) ][-1]
        j = spots[ np.where(calls_diff>0) ][0]
    else:
        i = spots[ np.where(calls_diff>0) ][-1]
        j = spots[ np.where(calls_diff<0) ][0]
    
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
    
    #BTC-240410-68250-C  1385  0.538611  68642.0 (1731.25)  69236.5 (2077.5)  70283.0 (2770.0)
    #BTC-240408-66250-P  OTM        230    245  7.3,5.0    0.405   -0.214  0.000152  -156.798    0.4      0.41   nan% nan 306.25        nan% nan 367.5        nan% nan 490.0
    contract = 'BTC-240408-66250-P'
    _,T,K,ctype = extract_specs( contract )
    p = 245
    sigma = .41
    print(f"{contract}\tK = {K}, T = {T}")
    print( invert_putprice(p,K,T/365,sigma,0.) )
 