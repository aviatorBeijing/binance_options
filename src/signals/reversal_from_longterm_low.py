import os,datetime,click
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

"""
- Purelly technical valleys, not related to any * motivations *.
- For motivation driven valleys, ref: Net Unrealized Profit/loss (NUPL) on-chain.
"""

def find_valleys(ts, closes,volume):
    rtn = closes.pct_change()
    cumrtn = (1+rtn).cumprod()
    cum_max = cumrtn.cummax()
    volume = talib.EMA(volume, timeperiod=5)
    volrank = volume.rolling(30*9).rank(pct=True)
    dd = (cumrtn - cum_max)/cum_max
    
    # remove nan
    dd = dd[1:]
    ts = ts[1:]
    volume = volume[1:]
    volrank = volrank[1:]
    
    # find min
    i_mm = np.argmin(dd)
    if i_mm+1 != dd.shape[0]-1:
        i_mm += 1
    print( i_mm, ts[i_mm], volrank[i_mm], f'{(dd[i_mm]*100):.1f}%')

    df = pd.DataFrame()
    df['ts'] = ts;df.ts = df.ts.apply(pd.Timestamp)
    df['dd'] = dd
    df['closes'] = closes
    df['volrank'] = volrank
    df.set_index('ts',inplace=True)
    #df = df[ -365: ]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax11 = ax1.twinx()

    df['dd'].plot(ax=ax1,color='red')
    df['volrank'].plot(ax=ax11,alpha=0.5)
    df['closes'].plot(ax=ax2)
    fn = os.getenv("USER_HOME",'')+'/tmp/reversal.pdf'
    plt.savefig(fn)
    print('-- saved:',fn)


def main():
    fn = os.getenv("USER_HOME","") + '/tmp/doge-usdt_1d.csv'
    df = pd.read_csv( fn,index_col=0 )

    ts = df.timestamp
    closes = df.close
    volume = df.volume 
    valleys = find_valleys( ts, closes, volume )

if __name__ == '__main__':
    main()