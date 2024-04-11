import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from butil.butils import binance_kline

class PriceGrid_:
    def __init__(self,span, lbound,hbound,median_v,
                            from_ts,end_ts,
                            ric, 
                            ohlcv=pd.DataFrame()) -> None:
        self.span = span
        self.ric = ric
        self.lb = lbound
        self.hb = hbound
        self.md = median_v
        self.t0 = from_ts
        self.t1 = end_ts

        self.raw = ohlcv

        self.updated_utc = int(datetime.datetime.utcnow().timestamp())
        
        self.grid_ = np.array([])

    def __str__(self) -> str:
        return f"Grid ({self.span}): \n\t[{self.lb}, {self.hb}] \n\t50%: {self.md} \n\tsamples from {self.t0} to {self.t1} \n\tlast_update_utc: {self.updated_utc}\n\tage: {self.age()} secs"
    def __repr__(self) -> str:
        return self.__str__()
    def generate_grid(self):
        raise Exception("See derived classes.")
    @property 
    def grid(self):
        if not len(self.grid_)>0:
            self.grid_ = self.generate_grid()
        return self.grid_

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
    
    def plot(self):
        if self.raw.empty:
            print('-- no ohlcv df provided, ignore plotting')
            return
        ohlcv = self.raw.copy()
        import matplotlib.pyplot as plt
        fig, (ax1,ax2,) = plt.subplots(2,1,figsize=(8,7))
         
        ohlcv['index'] = ohlcv.timestamp.apply(pd.Timestamp)
        ohlcv.set_index('index', inplace=True)
        ohlcv['hbound'] = self.hb
        ohlcv['lbound'] = self.lb
        ohlcv['md'] = self.md
        for i in range(0, len(self.grid)):
            ohlcv[f'grid{i}'] = self.grid[i]
        
        ohlcv[['close','hbound','lbound','md']].plot(ax=ax1,linewidth=2, style='-')
        for i in range(0, len(self.grid)):
            ohlcv[f'grid{i}'].plot(ax=ax1,linewidth=1, style='-.')

        user_home = os.getenv('USER_HOME','')
        gridfig = f'{user_home}/tmp/{self.ric.lower().replace("/","-")}_{self.span}.png'
        plt.savefig(gridfig)
        print('  -- saved grid figure:', gridfig)
        
class UniformGrid(PriceGrid_):
    def __init__(self, gap_bps, span, lbound, hbound, median_v, from_ts, end_ts, ric, ohlcv=pd.DataFrame()) -> None:
        super().__init__(span, lbound, hbound, median_v, from_ts, end_ts, ric, ohlcv)
        self.gap_bps = gap_bps
        self.gap_dollar = 0.
    @property 
    def gap(self):
        if self.gap_dollar>0:
            return self.gap_dollar
        else: 
            self.generate_grid()
            assert self.gap_dollar>0, f"Gap can't be less than 0. Found {self.gap_dollar}"
            return self.gap_dollar
    def generate_grid(self):
        self.gap_dollar = stp  = self.lb * self.gap_bps/10_000.
        print(f'  -- uniform grid, gap={self.gap_bps} bps, ${stp}')
        grid = np.arange(self.lb + stp,self.hb - stp, stp)
        print('*'*10, f'Grid ({len(grid)})','*'*10)
        for g in grid: print('\t', f"{g:.4f}")

        return grid

def price_range(ric, span='5m', start_ts=None, is_test=False) -> PriceGrid_:
    user_home = os.getenv('USER_HOME','')
    if is_test:
        fn =user_home+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
        ohlcv = pd.read_csv( fn ).tail(200)
    else:
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

    return UniformGrid( 200, span, lbound,hbound,md, ohlcv.iloc[0].timestamp, ohlcv.iloc[-1].timestamp, ric, ohlcv )

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

    closep = float(df.iloc[-1].close)
    if pgrid.bound_breached(closep):
        print('  -- update on new hi/lo')
        pgrid.update()

    print(pgrid)
    ddf = pgrid.distance( closep )
    print('-- current:', closep)
    print(ddf)

    volumes = df.volume.values
    if len(volumes)>3:
        vrk = scipy.stats.percentileofscore( volumes, volumes[-1] )
        print( f'-- volume stack: {len(volumes)}, latest volume rank: {vrk:.1f}%' )

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
@click.option('--test', is_flag=True, default=False)
def main(ric,start_ts,test):
    global pgrid 
    if not pgrid: # Init
        pgrid = price_range(ric,span='5m',start_ts=start_ts, is_test=test)
        pgrid.plot()

    if not test:
        from cryptofeed import FeedHandler
        from cryptofeed.backends.aggregate import OHLCV
        from cryptofeed.defines import TRADES
        from cryptofeed.exchanges import Binance

        f = FeedHandler()
        f.add_feed(Binance(symbols=[ric],channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))
        f.run()
    else:
        print('-- [testing mode] ignore realtime wss connectivity')


if __name__ == '__main__':
    main()