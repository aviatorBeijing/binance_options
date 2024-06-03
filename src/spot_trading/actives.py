import os,click,talib
import pandas as pd
from tabulate import tabulate

from butil.butils import binance_kline

@click.command()
@click.option('--sym')
@click.option('--wd', default=7)
@click.option('--dt', default='1d')
def main(sym,wd,dt):
    df = binance_kline(f'{sym.upper()}/USDT', span=dt, grps=5)
    volume = df.volume
    df.volume = talib.EMA(volume, timeperiod=5)
    df['volrank'] = volume.dropna().rolling(wd).rank(pct=True)
    df['dir'] = df.close > df.open
    
    df['dir'] = df['dir'].apply(lambda d: '+' if d else '-')
    df.volrank = df.volrank.apply(lambda v: f"{(v*100):.1f}%")
    print(df.tail(10) )
    print(f'-- activity based on dt={dt}, window={wd} (in unit of "dt")')

if __name__ == '__main__':
    main()
