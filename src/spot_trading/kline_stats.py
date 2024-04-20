import os,datetime,click 
import pandas as pd
from butil.butils import binance_kline

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--span', default='5m')
def main(ric,span):
    df = binance_kline( symbol=ric, span=span )
    row = df.iloc[-1]
    ts,open,high,low,close,volume = row.timestamp,row.open,row.high,row.low,row.close,row.volume 
    ts = pd.Timestamp(ts).to_pydatetime()
    print( ts, close )
if __name__ == '__main__':
    main()