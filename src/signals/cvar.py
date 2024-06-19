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

import click
@click.command()
@click.option('--cryptos')
@click.option('--weights', default='')
def main(cryptos, weights):
    fn =  os.getenv('USER_HOME','')+'/tmp/'
    
    dfs = []
    cols = cryptos.split(',') # ['btc','doge','sol','gld']
    for s in cols:
        cn = fn + f'{s}-usdt_1d.csv'
        df  =  pd.read_csv( cn )
        df.timestamp = df.timestamp.apply(pd.Timestamp)
        df.set_index('timestamp',inplace=True,drop=True)
        c = df[['close']]
        c = c.resample('1d').agg('last')
        dfs += [c.close]

    rtns = pd.concat(dfs,axis=1,ignore_index=False)
    rtns = rtns.dropna().pct_change().dropna()
    rtns.columns = cols
    
    returns, meanReturns, covMatrix = getData( rtns )
    print(returns )
    if weights:
        weights = weights.split(',')
        weights = list(map(lambda s: float(s), weights ) )
        assert len(weights)==len(cols), f'cryptos and weights doesn\'t match on size. {len(cols)}, {len(weights)}'
        weights = np.array(weights)
        weights /= np.sum(weights)
    else:
        weights = [0.3,0.3,0.4] #np.random.random(len(returns.columns))
        weights = np.ones( rtns.shape[1]) * (1/rtns.shape[1] )
        weights /= np.sum(weights)

    print('weights:',weights)
    returns['portfolio'] = returns.dot(weights)

    pt =  99
    Time = 30 # Days
    InitialInvestment = 10_000
    
    pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)
    hVaR = -historicalVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    hCVaR = -historicalCVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    
    print()
    print(f'Time horizon                :      {Time} days')
    print( 'Investment                  :      $', f'{InitialInvestment:,.2f}')
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

if __name__ == '__main__':
    main()
