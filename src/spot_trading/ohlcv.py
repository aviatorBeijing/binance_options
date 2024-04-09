import click,datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Binance

from bbroker.settings import BianceSpot
from butil.butils import binance_spot

WINDOW_IN_SECONDS = 5
stacks_len=10*12 # Working with WINDOW_IN_SECONDS,  defines the length of history
rows = []

async def ohlcv(data):
    global stacks_len
    global rows 
    for k,v in data.items():
        vv = v;v['ric']=k
        if 'DOGE' in k:
            rows+= [ vv ]
        if len(rows) > stacks_len:
            rows = rows[ -stacks_len:]

    df = pd.DataFrame.from_records( rows )
    print(tabulate(df,headers="keys"))

    volumes = df.volume.values
    if len(volumes)>3:
        vrk = scipy.stats.percentileofscore( volumes, volumes[-1] )
        print( len(volumes), vrk )

def market_check(ric):
    mkt = BianceSpot(ric.replace('-','/'))
    openDf = mkt.check_open_orders()
    
    tds = mkt.check_trades_today()
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
@click.option('--ric',default="BTC-USDT")
def main(ric):
    
    market_check(ric)

    f = FeedHandler()
    f.add_feed(Binance(symbols=[ric],channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))
    f.run()


if __name__ == '__main__':
    main()
