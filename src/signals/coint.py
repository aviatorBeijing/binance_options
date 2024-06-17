import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def _data(sym, offline=False):
    sym = sym.lower()
    fn = os.getenv("USER_HOME","") + f'/tmp/{sym}-usdt_1d.csv'
    if not offline or not os.path.exists(fn):
        from butil.butils import binance_kline
        df = binance_kline(f'{sym.upper()}/USDT', span='1d', grps=5)
        print(df.tail(3))
    else:
        df = pd.read_csv( fn,index_col=0 )
    
    df.timestamp = df.timestamp.apply(pd.Timestamp)
    df.set_index('timestamp',inplace=True,drop=True)
    return df

def main():
    syms = ['doge','sol']
    dfs = []
    for c in syms:
        df = _data(c,offline=True)
        dfs+= [df.close]
    df = pd.concat(dfs,axis=1,ignore_index=False).dropna()
    df.columns = syms
    df = df.pct_change().dropna()
    print(df)

    beta = np.polyfit(df.doge, df.sol, 1)[0]
    df['fitted'] = df.doge + beta * df.sol 
    fig,ax1 = plt.subplots(1,1, figsize=(12,6))
    ax11 = ax1.twinx()
    df = df.tail(200)
    df['fitted'].plot(ax=ax1,color='red')
    df['doge'].plot(ax=ax11,alpha=0.5)
    df['sol'].plot(ax=ax11,alpha=0.5)
    ax1.legend()
    ax11.legend()
    plt.savefig(os.getenv('USER_HOME','')+'/tmp/coint.png')

if __name__ ==  '__main__':
    main()