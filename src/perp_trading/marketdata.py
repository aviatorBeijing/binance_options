
import datetime,os
import pandas as pd 
from tabulate import tabulate
from bbroker.settings import perp_ex

def adhoc_ticker(symbol='BTC/USDT')->tuple:
    symbol = symbol.replace('-','/')
    qts = perp_ex.fapiPublicGetTickerBookTicker({'symbol': symbol.replace('/','').replace('-','').upper()})
    bid,ask = qts['bidPrice'],qts['askPrice'] # bidQty,askQty
    bid = float(bid)
    ask = float(ask)

    spread = (ask-bid)/(ask+bid)*2
    assert spread< 5./10_000, f'spread is too wide: {spread} (bid:{bid},ask:{ask})'

    return float(bid),float(ask)

def get_perp_klines(ric, span)->pd.DataFrame:
    ric = ric.replace('/','').replace('-','').upper()
    h = perp_ex.fapiPublicGetKlines({'symbol':ric, 'interval':span})
    """
        https://github.com/ccxt/ccxt/issues/18555
       #         1591478520000,  # open time
        #         "0.02501300",  # open
        #         "0.02501800",  # high
        #         "0.02500000",  # low
        #         "0.02500000",  # close
        #         "22.19000000",  # volume
        #         1591478579999,  # close time
        #         "0.55490906",  # quote asset volume
        #         40,            # number of trades
        #         "10.92900000",  # taker buy base asset volume
        #         "0.27336462",  # taker buy quote asset volume
        #         "0"            # ignore
    """
    cols = ['starttime','open','high','low','close','volume','endtime',
            'quote_volume','trades','taker_base_volume','taker_quote_volume','foobar']
    df = pd.DataFrame.from_records(h, columns=cols)

    print(tabulate(df.tail(5),headers="keys"))
    rk_last = df.volume.rolling(100).rank(pct=True).iloc[-2:].values
    print(f'-- latest volumes: {(rk_last[0]*100):.1f}%, {(rk_last[1]*100):.1f}%')

    # cache
    fn = os.getenv('USER_HOME','/Users/junma')
    fn = f'{fn}/tmp/perp_{ric.lower()}_{span}.csv'
    df.to_csv( fn, index=0)
    print('-- saved: ',fn)
    return df

def price_crossing( price, ohlcv, ric, span='15m'):
    """
    @brief Given a price, check the crossing stats in recent window of times
    """
    #df = get_perp_klines(ric,span)
    df = ohlcv.copy()
    df['end'] = df.endtime.apply(int).apply(lambda e: int(e/1000)).apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
    print(df)

import click
@click.command()
@click.option('--ric')
@click.option('--span')
def main(ric,span):
    get_perp_klines(ric,span)

if __name__ == '__main__':
    main()
