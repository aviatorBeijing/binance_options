import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from bbroker.settings import BianceSpot
from butil.butils import binance_spot

def portfolio_check(ric):
    mkt = BianceSpot(ric.replace('-','/'))
    
    tds = mkt.check_trades(hours=48)
    tds = tds[tds.symbol==ric.replace('-','')]
    tds['sign'] = tds.side.apply(lambda s: 1 if s=='BUY' else -1)
    tds['qty'] = tds.sign * tds.qty.astype(float)
    tds['agg'] = tds.qty.cumsum()
    tds['$agg'] = -(tds.qty*tds.price.astype(float)).cumsum()
    tds['neutral'] = ''
    tds.loc[tds['agg']==0,'neutral'] = 'ok'
    print('-- [trades]')
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

    pce,_ = binance_spot( ric.replace('-','/') )
    port_value = tds.iloc[-1]['agg'] * pce  + tds.iloc[-1]['$agg'] - fee 
    print(f'-- gain (after liquidating): $ {port_value:,.4f}')
    
    # orders
    openDf = mkt.check_open_orders()

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
def main(ric,start_ts):    
    portfolio_check(ric)

if __name__ == '__main__':
    main()
