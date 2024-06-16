import os,datetime
import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

"""
Ref:
https://quantpy.com.au/risk-management/value-at-risk-var-and-conditional-var-cvar/
"""
def getData(returns):
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(Time)
    return returns, std

def historicalVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the percentile of the distribution at the given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")
def historicalCVaR(returns, alpha=5):
    """
    Read in a pandas dataframe of returns / a pandas series of returns
    Output the CVaR for dataframe / series
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    # A passed user-defined-function will be passed a Series for evaluation.
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")

def var_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    # because the distribution is symmetric
    if distribution == 'normal':
        VaR = norm.ppf(1-alpha/100)*portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        VaR = np.sqrt((nu-2)/nu) * t.ppf(1-alpha/100, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR
def cvar_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        xanu = t.ppf(alpha/100, nu)
        CVaR = -1/(alpha/100) * (1-nu)**(-1) * (nu-2+xanu**2) * t.pdf(xanu, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR


if __name__ == '__main__':
    fn =  os.getenv('USER_HOME','')+'/tmp/'
    doge = fn + 'doge-usdt_1d.csv'
    btc = fn + 'btc-usdt_1d.csv'
    sol = fn + 'sol-usdt_1d.csv'
    
    doge  =  pd.read_csv( doge )
    btc  = pd.read_csv(btc)
    sol = pd.read_csv(sol)
    doge.set_index('timestamp',inplace=True)
    btc.set_index('timestamp',inplace=True)
    sol.set_index('timestamp',inplace=True)

    rtns = pd.concat([ doge.close,btc.close,sol.close],axis=1,ignore_index=False)
    rtns = rtns.dropna().pct_change().dropna()
    
    returns, meanReturns, covMatrix = getData( rtns )

    weights = [0.3,0.3,0.4] #np.random.random(len(returns.columns))
    print('weights:',weights)
    weights /= np.sum(weights)
    returns['portfolio'] = returns.dot(weights)

    pt =  99
    Time = 30 # Days
    InitialInvestment = 10_000
    
    pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)
    hVaR = -historicalVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    hCVaR = -historicalCVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    
    print()
    print(f'Time horizon                :      {Time} days')
    print( 'Investment                  :      $', InitialInvestment)
    print( 'Expected Portfolio Return   :      $', round(InitialInvestment*pRet,2))
    
    print()
    print('-- modeless:')
    print(f'  VaR {pt}th CI             :      $', round(InitialInvestment*hVaR,2))
    print(f'  CVaR {pt}th CI            :      $', round(InitialInvestment*hCVaR,2))

    normVaR = var_parametric(pRet, pStd,alpha=100-pt)
    normCVaR = cvar_parametric(pRet, pStd,alpha=100-pt)
    tVaR = var_parametric(pRet, pStd, distribution='t-distribution',alpha=100-pt)
    tCVaR = cvar_parametric(pRet, pStd, distribution='t-distribution',alpha=100-pt)
    print('-- normal model:')
    print(f"  Normal VaR {pt}th CI       :      $", round(InitialInvestment*normVaR,2))
    print(f"  Normal CVaR {pt}th CI      :      $", round(InitialInvestment*normCVaR,2))
    print('-- t-dist model (fat-tail):')
    print(f"  t-dist VaR {pt}th CI       :      $", round(InitialInvestment*tVaR,2))
    print(f"  t-dist CVaR {pt}th CI      :      $", round(InitialInvestment*tCVaR,2))