import os,datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

doge = os.getenv("USER_HOME")  + '/tmp/doge-usdt_5m.csv'
btc = os.getenv("USER_HOME")  + '/tmp/btc-usdt_5m.csv'

def get_ohlcv(fn):
    df = pd.read_csv(fn,index_col=0)
    df['index'] = df.timestamp.apply(pd.Timestamp)
    df.set_index('index',inplace=True,drop=True)
    df['rtn'] = (df.close-df.open)/df.open
    return df

btc  = get_ohlcv(btc)
doge = get_ohlcv(doge)

dual = pd.concat( [doge.rtn, btc.rtn],axis=1, ignore_index=False)
dual.columns = ['doge','btc']
print(dual)

dual = dual.shift(3).dropna()
print(dual.doge.corr(dual.btc))
dual.plot();plt.show()