from email.policy import default
import os,datetime,click 
import pandas as pd 

from butil.butils import get_binance_next_funding_rate,get_binance_spot
from perp_trading.marketdata import adhoc_ticker as get_perp_ticker

@click.command()
@click.option('--ric', default='BTC/USDT')
def main(ric):
    ric = ric.upper()
    apr, r, ts = get_binance_next_funding_rate(ric)
    print( apr, r, ts)

    bid,ask = get_perp_ticker(ric)
    print( bid, ask )

    bid,ask = get_binance_spot(ric)
    print( bid, ask)

if __name__ == '__main__':
    main()