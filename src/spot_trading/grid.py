import click,datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Binance

from spot_trading.portfolio import PriceGrid,price_range

WINDOW_IN_SECONDS = 5
stacks_len=10*12 # Working with WINDOW_IN_SECONDS,  defines the length of history
rows = []

pgrid = None 
async def ohlcv(data):
    global stacks_len
    global rows 
    global pgrid 
    for k,v in data.items():
        vv = v;v['ric']=k
        rows+= [ vv ]
        if len(rows) > stacks_len:
            rows = rows[ -stacks_len:]

    df = pd.DataFrame.from_records( rows )
    print(tabulate(df,headers="keys"))
    print(pgrid)
    closep = df.iloc[-1].close.astype(float)
    ddf = pgrid.distance( closep )
    print(ddf)

    volumes = df.volume.values
    if len(volumes)>3:
        vrk = scipy.stats.percentileofscore( volumes, volumes[-1] )
        print( len(volumes), vrk )

@click.command()
@click.option('--ric',default="DOGE-USDT")
def main(ric):
    global pgrid 
    if not pgrid: # Init
        pgrid = price_range(ric,span='5m')

    f = FeedHandler()
    f.add_feed(Binance(symbols=[ric],channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))
    f.run()


if __name__ == '__main__':
    main()
