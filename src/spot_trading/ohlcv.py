import click
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Binance

WINDOW_IN_SECONDS = 5

doge_volume_stacks = []
doge_volume_stacks_len=10*12 # Working with WINDOW_IN_SECONDS,  defines the length of history
rows = []
async def ohlcv(data):
    global doge_volume_stacks
    global doge_volume_stacks_len
    global rows 
    for k,v in data.items():
        vv = v;v['ric']=k
        rows+= [ vv ]
        if len(rows) > doge_volume_stacks_len:
            rows = rows[ -doge_volume_stacks_len:]
        if 'DOGE' in k:
            doge_volume_stacks += [ v['volume'] ]
            l = doge_volume_stacks_len
            if len(doge_volume_stacks)> l :
                doge_volume_stacks = doge_volume_stacks[-l:] # pop front

    df = pd.DataFrame.from_records( rows )
    print(tabulate(df,headers="keys"))
    if len(doge_volume_stacks)>3:
        vrk = scipy.stats.percentileofscore( doge_volume_stacks, doge_volume_stacks[-1] )
        print( len(doge_volume_stacks), vrk )

@click.command()
@click.option('--rics',default="BTC-USDT")
def main(rics):
    f = FeedHandler()
    #f.add_feed(Coinbase(symbols=['BTC-USD', 'ETH-USD', 'BCH-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=10)}))
    f.add_feed(Binance(symbols=rics.split(','),channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))

    f.run()


if __name__ == '__main__':
    main()
