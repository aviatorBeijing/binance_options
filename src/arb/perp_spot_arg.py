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
    print( 'funding:',apr, r, ts)

    pbid,pask = get_perp_ticker(ric)
    print( 'perp:', pbid, pask )

    sbid,sask = get_binance_spot(ric)
    print('spot:', sbid, sask)

    x = (sask-pbid)/(pbid+sask)*2*10_000 
    y = (pask-sbid)/(sbid+pask)*2*10_000 
    print(f"buy spot/sell perp: {x:.1f} bps")
    print(f"buy perp/sell spot: {y:.1f} bps")

if __name__ == '__main__':
    main()
