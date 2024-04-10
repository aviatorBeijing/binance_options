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

    if pgrid.bound_breached(closep):
        print('  -- update on new hi/lo')
        pgrid.update()

    print(pgrid)
    closep = float(df.iloc[-1].close)
    ddf = pgrid.distance( closep )
    print(ddf)

    volumes = df.volume.values
    if len(volumes)>3:
        vrk = scipy.stats.percentileofscore( volumes, volumes[-1] )
        print( len(volumes), vrk )

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
def main(ric,start_ts):
    global pgrid 
    if not pgrid: # Init
        pgrid = price_range(ric,span='5m',start_ts=start_ts)

    f = FeedHandler()
    f.add_feed(Binance(symbols=[ric],channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))
    f.run()


if __name__ == '__main__':
    main()
