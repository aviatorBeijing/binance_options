import os,datetime
import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

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

    df = pd.concat([ doge.close,btc.close,sol.close],axis=1,ignore_index=False)
    df = df.dropna().pct_change().dropna()
    
    returns, meanReturns, covMatrix = getData( df )

    weights = [0.3,0.3,0.4] #np.random.random(len(returns.columns))
    print('weights:',weights)
    weights /= np.sum(weights)
    returns['portfolio'] = returns.dot(weights)

    pt =  99
    Time = 100
    hVaR = -historicalVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    hCVaR = -historicalCVaR(returns['portfolio'], alpha=100-pt)*np.sqrt(Time)
    pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)
    InitialInvestment = 10000
    print('Investment:       $', InitialInvestment)
    print('Expected Portfolio Return:      $', round(InitialInvestment*pRet,2))
    print(f'VaR {pt}th CI    :      $', round(InitialInvestment*hVaR,2))
    print(f'CVaR {pt}th CI  :      $', round(InitialInvestment*hCVaR,2))
