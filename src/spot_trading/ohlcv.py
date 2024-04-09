import click,datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Binance

from bbroker.settings import spot_ex

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

def orders_status(ex,ric)->pd.DataFrame:
    #tnow = datetime.datetime.utcnow().timestamp()*1000;tnow=int(tnow)
    ods = ex.fetchOpenOrders(ric)
    ods = list(map(lambda e: e['info'],ods))
    df = pd.DataFrame.from_records(ods)
    if df.empty: 
        print('*** No outstanding orders.')
        return pd.DataFrame()
    print(df.columns)
    #df['dt'] = (tnow - df.updateTime.apply(int))/1000
    #df = df[['status','orderId','symbol','side','price','avgPrice','quantity','executedQty','updateTime','source','priceScale','quantityScale']]
    df = df['symbol,type,side,status,orderId,price,origQuoteOrderQty,executedQty,cummulativeQuoteQty,updateTime'.split(',')]
    df['datetime'] = df.updateTime.apply(int).apply(lambda v: datetime.datetime.fromtimestamp(v/1000))
    df = df.sort_values('updateTime', ascending=False)
    print('--[ orders ]\n',tabulate(df,headers="keys"))
    return df

@click.command()
@click.option('--rics',default="BTC-USDT")
def main(rics):
    orders_status( spot_ex, 'DOGE/USDT' )
    f = FeedHandler()
    #f.add_feed(Coinbase(symbols=['BTC-USD', 'ETH-USD', 'BCH-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=10)}))
    f.add_feed(Binance(symbols=rics.split(','),channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))

    f.run()


if __name__ == '__main__':
    main()
