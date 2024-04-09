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

@click.command()
@click.option('--ric',default="BTC-USDT")
def main(ric):
    
    mkt = BianceSpot(ric.replace('-','/'))
    openDf = mkt.check_open_orders()

    f = FeedHandler()
    #f.add_feed(Coinbase(symbols=['BTC-USD', 'ETH-USD', 'BCH-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=10)}))
    f.add_feed(Binance(symbols=ric,channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))

    f.run()


if __name__ == '__main__':
    main()
