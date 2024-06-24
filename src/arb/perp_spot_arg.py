from email.policy import default
import os,datetime,click 
import pandas as pd 

from butil.butils import get_binance_next_funding_rate,get_binance_spot
from perp_trading.marketdata import adhoc_ticker as get_perp_ticker

@click.command()
@click.option('--ric', default='BTC/USDT')
@click.option('--sym', default='')
def main(ric,sym):
    if sym:
        ric = sym.upper() + '/USDT'
    else:
        ric = ric.upper()
    apr, r, ts = get_binance_next_funding_rate(ric)
    print( 'funding:', f'{(apr*100):.1f}% (annual)', f'{(r*10000):.2f} bps (next)', ts)

    pbid,pask = get_perp_ticker(ric)
    print( 'perp:', pbid, pask )

    sbid,sask = get_binance_spot(ric)
    print('spot:', sbid, sask)

    x = (-sask+pbid)/(pbid+sask)*2*10_000 
    y = (-pask+sbid)/(sbid+pask)*2*10_000 
    
    tradable = lambda v: "(yes)" if v>0 else "(ng)"

    print("\n   -- Profit margin --")
    print(f"buy spot/sell perp: {x:.1f} bps {tradable(x)}")
    print(f"buy perp/sell spot: {y:.1f} bps {tradable(y)}")

if __name__ == '__main__':
    main()
