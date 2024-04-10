import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from bbroker.settings import BianceSpot
from butil.butils import binance_kline,binance_spot

def portfolio_check(ric):
    mkt = BianceSpot(ric.replace('-','/'))
    openDf = mkt.check_open_orders()
    
    tds = mkt.check_trades(hours=48)
    tds = tds[tds.symbol==ric.replace('-','')]
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

    pce,_ = binance_spot( ric.replace('-','/') )
    port_value = tds.iloc[-1]['agg'] * pce  + tds.iloc[-1]['$agg'] - fee 
    print(f'-- gain (after liquidating): $ {port_value:,.4f}')

class PriceGrid:
    def __init__(self,span, lbound,hbound,median_v,from_ts,end_ts,ric) -> None:
        self.span = span
        self.ric = ric
        self.lb = lbound
        self.hb = hbound
        self.md = median_v
        self.t0 = from_ts
        self.t1 = end_ts
        self.updated_utc = int(datetime.datetime.utcnow().timestamp())
    def __str__(self) -> str:
        return f"Grid ({self.span}): \n\t[{self.lb}, {self.hb}] \n\t50%: {self.md} \n\tsamples from {self.t0} to {self.t1} \n\tlast_update_utc: {self.updated_utc}\n\tage: {self.age()} secs"
    def __repr__(self) -> str:
        return self.__str__()
    def distance(self, d)->pd.DataFrame:
        df = pd.DataFrame.from_dict([{'hb':self.hb, 'md':self.md,'lb':self.lb}] ).transpose()
        df.columns = ['price'];df.price = df.price.apply(float)
        df['distance'] = d-df.price
        df['bps'] = (d-df.price)/df.price*1_0000
        df['bps'] = df['bps'].apply(lambda v: f"{v:.0f}")
        return df
    def bound_breached(self,d)->bool:
        return d>self.hb or d<self.lb
    def age(self)->int: # seconds
        d = int(datetime.datetime.utcnow().timestamp()) - self.updated_utc
        return d
    def update(self):
        ohlcv = binance_kline(symbol=self.ric.replace('-','/'),span=self.span,grps=1)
        assert len(self.t0) == len('2024-04-09T20:35:00.000Z'), f"Wrong format {self.t0}"
        ohlcv = ohlcv[ohlcv.timestamp>self.t0]
        if os.getenv('BINANCE_DEGUG'):
            print(f'-- [{ohlcv.shape[0]}]', ohlcv.iloc[0].timestamp, '~', ohlcv.iloc[-1].timestamp)
        
        self.lb = np.min(ohlcv.low) #np.percentile(ohlcv.low,0.1)
        self.md = np.percentile(ohlcv.close, 50)
        self.hb = np.max(ohlcv.high) #np.percentile(ohlcv.high,99.9)
        self.t0 = ohlcv.iloc[0].timestamp
        self.t1 = ohlcv.iloc[-1].timestamp
        self.updated_utc = int(datetime.datetime.utcnow().timestamp())
        
def price_range(ric, span='5m', start_ts=None) -> PriceGrid:
    ohlcv = binance_kline(symbol=ric.replace('-','/'),span=span,grps=1)
    if start_ts:
        print('-- selecting by input timestamp')
        assert len(start_ts) == len('2024-04-09T20:35:00.000Z'), f"Wrong format {start_ts}"
        #t = pd.Timestamp(start_ts)
        ohlcv = ohlcv[ohlcv.timestamp>start_ts]
    else:
        print('-- selecting last rows')
        ohlcv = ohlcv.tail( int(8*60/5) )
    print(f'-- [{ohlcv.shape[0]}]', ohlcv.iloc[0].timestamp, '~', ohlcv.iloc[-1].timestamp)
    
    lbound = np.min(ohlcv.low) #np.percentile(ohlcv.low,0.1)
    md = np.percentile(ohlcv.close, 50)
    hbound = np.max(ohlcv.high) #np.percentile(ohlcv.high,99.9)
    return PriceGrid( span, lbound,hbound,md, ohlcv.iloc[0].timestamp, ohlcv.iloc[-1].timestamp, ric )

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
def main(ric,start_ts):
    #print( price_range(ric, span="5m", start_ts=start_ts) )
    
    print('\n-- trades:')
    portfolio_check(ric)

if __name__ == '__main__':
    main()
