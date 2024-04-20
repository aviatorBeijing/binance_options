import os,datetime,click 
import pandas as pd
from butil.butils import binance_kline

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--span', default='5m')
def main(ric,span):
    df = binance_kline( symbol=ric, span=span )
    print(df)
if __name__ == '__main__':
    main()