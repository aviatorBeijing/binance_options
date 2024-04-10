import click,datetime
from tabulate import tabulate
import pandas as pd
import numpy as np

from bbroker.settings import BianceSpot
from butil.butils import binance_spot

def portfolio_check(ric):
    mkt = BianceSpot(ric.replace('-','/'))
    openDf = mkt.check_open_orders()
    
    tds = mkt.check_trades_today()
    tds = tds[tds.symbol.contains(ric.split('-')[0])]
    tds['sign'] = tds.side.apply(lambda s: 1 if s=='BUY' else -1)
    tds['qty'] = tds.sign * tds.qty.astype(float)
    tds['agg'] = tds.qty.cumsum()
    tds['$agg'] = -(tds.qty*tds.price.astype(float)).cumsum()
    tds['neutral'] = ''
    tds.loc[tds['agg']==0,'neutral'] = 'ok'
    print( tds )
    
    pceMap = {}
    syms = list(set(tds.commissionAsset.values))
    for s in syms:
        if s =='USDT':
            pceMap[s] = 1.
        else:
            feeric = f'{s}/USDT'
            bid,ask = binance_spot( feeric )
            pceMap[s] = bid
    tds['commAssetPrice'] = tds.commissionAsset.apply(lambda s: pceMap[s])
    fee = (tds.commission.astype(float)*tds.commAssetPrice).sum()
    print(f'-- fee: ${fee}')

@click.command()
@click.option('--ric',default="DOGE-USDT")
def main(ric):
    portfolio_check(ric)

if __name__ == '__main__':
    main()
