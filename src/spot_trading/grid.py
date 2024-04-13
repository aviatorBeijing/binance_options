import click,datetime,os
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy

from butil.butils import binance_kline

GRID_DEBUG=os.getenv('GRID_DEBUG', None)

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
        self.last_updated_price = float(ohlcv.iloc[-1].close) if not ohlcv.empty else -1.
        
        self.grid_ = np.array([])

        self.trendy()

    def __str__(self) -> str:
        info = f"Grid ({self.span}): \n\tN={len(self.grid)}\n\t[{self.lb}, {self.hb}] \n\t50%: {self.md} \n\tsamples from {self.t0} to {self.t1} \n\tlast_update_utc: {self.updated_utc}\n\tage: {self.age()} secs"
        if self.grid:
            gdf = pd.DataFrame.from_records( self.grid )
            gdf.columns = ['action','price']
            gdf['bps']=(gdf.price-self.current_price)/self.current_price*10_000
            gdf.bps = gdf.bps.apply(int)
            gdf['inc'] = gdf.bps.diff()
            info += '\n' + str(gdf)
        return info
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
    def price_moved(self, d)->bool:
        sf = abs(d-self.last_updated_price)
        return sf > self.gap * 1
    def age(self)->int: # seconds
        d = int(datetime.datetime.utcnow().timestamp()) - self.updated_utc
        return d
    def update(self):
        self.raw = ohlcv = binance_kline(symbol=self.ric.replace('-','/'),span=self.span,grps=1)
        assert len(self.t0) == len('2024-04-09T20:35:00.000Z'), f"Wrong format {self.t0}"
        ohlcv = ohlcv[ohlcv.timestamp>self.t0]
        if os.getenv('BINANCE_DEGUG'):
            print(f'-- [{ohlcv.shape[0]}]', ohlcv.iloc[0].timestamp, '~', ohlcv.iloc[-1].timestamp)
        
        self.trendy()

        self.lb = np.min(ohlcv.low) #np.percentile(ohlcv.low,0.1)
        self.md = np.percentile(ohlcv.close, 50)
        self.hb = np.max(ohlcv.high) #np.percentile(ohlcv.high,99.9)
        self.t0 = ohlcv.iloc[0].timestamp
        self.t1 = ohlcv.iloc[-1].timestamp
        self.updated_utc = int(datetime.datetime.utcnow().timestamp())
        self.last_updated_price = float(ohlcv.iloc[-1].close)
        self.grid_ = self.generate_grid()

    def trendy(self):
        if not self.raw.empty:
            # Trend
            ohlcv = self.raw.copy()
            ttv = ohlcv.volume.sum()
            ohlcv['wrtn_bps'] = (ohlcv['close'] - ohlcv['open'])/ohlcv['open']*10_000
            ohlcv.wrtn_bps = ohlcv.wrtn_bps * ohlcv.volume / ttv
            s = ohlcv.wrtn_bps.describe()
            #print( s.loc['25%'], s.loc['50%'], s.loc['75%'])
            print( '-- trend:', np.array([ s.loc['25%'], s.loc['50%'], s.loc['75%']]), ' (25%,50%,75%)')
            del ohlcv
            
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
            ohlcv[f'grid{i}'] = self.grid[i][1]
        
        ohlcv[['close','hbound','lbound','md']].plot(ax=ax1,linewidth=2, style='-')
        for i in range(0, len(self.grid)):
            ohlcv[f'grid{i}'].plot(ax=ax1,linewidth=1, style='-.')

        ohlcv[['high','low']].plot(ax=ax2,linewidth=1, style='-')

        user_home = os.getenv('USER_HOME','')
        gridfig = f'{user_home}/tmp/{self.ric.lower().replace("/","-")}_{self.span}.png'
        plt.savefig(gridfig)
        print('  -- saved grid figure:', gridfig)
        
class HFTUniformGrid(PriceGrid_):
    """
    @brief "HFT" indicates the grid is designed with HFT in mind.

    @param current_price (float)  : current price
    @param gap_bps (int): pre-determined gap between grids
    """
    def __init__(self, current_price, gap_bps, 
                span, lbound, hbound, median_v, from_ts, end_ts, ric, ohlcv=pd.DataFrame()) -> None:
        super().__init__(span, lbound, hbound, median_v, from_ts, end_ts, ric, ohlcv)
        self.gap_bps = gap_bps
        self.gap_dollar = 0.
        self.current_price = current_price
        assert current_price>0, f"Negative or zero prices input: current_price={current_price}"
    def __str__(self) -> str:
        others = super().__str__()
        return f'[HFT UNIFORM GRID]\ncurrent price: {self.current_price:.5f}, gap= {self.gap_bps}bps,gap= ${self.gap:.5f}\n{others}'
    @property 
    def gap(self):
        if self.gap_dollar>0:
            return self.gap_dollar
        else: 
            self.grid_ = self.generate_grid()
            assert self.gap_dollar>0, f"Gap can't be less than 0. Found {self.gap_dollar}"
            return self.gap_dollar

    def generate_grid(self) ->list:
        self.gap_dollar = stp  = self.lb * self.gap_bps/10_000.
        #print(f'  -- uniform grid, gap={self.gap_bps} bps, ${stp:.4f}')
        #print(f'     last updated price: {self.last_updated_price:.5f}')
        grid = np.arange(self.lb + stp,self.hb - stp, stp)
        grid = list(reversed(grid))
        aug_grid = []

        if self.ric.upper().startswith('DOGE'): fa = 1e4
        if self.ric.upper().startswith('BTC'): fa = 1e2
        fa_ = lambda v: int(v*fa)/fa

        for g in grid: 
            action = 'sell' if g>self.current_price else 'buy' if g<self.current_price else 'sit'
            aug_grid += [ (action, fa_(g)) ]

        if GRID_DEBUG:
            print(' *'*10, f'grids={len(grid)}',' *'*10)
            for g in aug_grid: 
                action = g[0]
                if action != 'sit':
                    print('\t', ' '*10, f"{action}  {g[1]}")
        return aug_grid

def low_freq_price_range(ric, span='5m', start_ts=None, is_test=False) -> PriceGrid_:
    """
    @brief This function should be used in low-frequency (>1~5min) mode,
            because https respests are used here. In high-frequency
            mode, ip  might be banned by any exchanges.
    """
    user_home = os.getenv('USER_HOME','')
    if is_test:
        fn =user_home+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
        ohlcv = pd.read_csv( fn ).tail(200)
        ohlcv['rtn'] = (ohlcv.close - ohlcv.open)/ohlcv.open
        ohlcv['rtn_prev'] = ohlcv['rtn'].shift(1) 
        ohlcv.loc[(ohlcv.rtn_prev>-5/1000.) & (ohlcv.rtn_prev<5/1000.), 'rtn'] = 0
        ohlcv = ohlcv[ohlcv.timestamp>start_ts]
        
        print(ohlcv.columns)
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
    
    if not is_test:
        print('-- [live mode grid]')
        lbound = np.min(ohlcv.low) #np.percentile(ohlcv.low,0.1)
        md = np.percentile(ohlcv.close, 50)
        hbound = np.max(ohlcv.high) #np.percentile(ohlcv.high,99.9)
    else:
        print('-- [test mode grid]')
        n = 150 # mimic the impact of trade now but using history to generate grid
        lbound = np.min(ohlcv[:n].low) 
        md = np.percentile(ohlcv[:n].close, 50)
        hbound = np.max(ohlcv[:n].high)

    return span,lbound,hbound,md, ohlcv.iloc[0].timestamp, ohlcv.iloc[-1].timestamp, ric, ohlcv 

WINDOW_IN_SECONDS = 5
stacks_len=10*12 # Working with WINDOW_IN_SECONDS,  defines the length of history
rows = []

pgrid = None #global
async def ohlcv(data):
    global stacks_len
    global rows 
    global pgrid 
    for k,v in data.items():
        vv = v;v['ric']=k
        rows+= [ vv ]
        if len(rows) > stacks_len:
            rows = rows[ -stacks_len:]

    # sub-minutes trading history (transient)
    df = pd.DataFrame.from_records( rows )
    print(tabulate(df.tail(5),headers="keys"))
    closep = float(df.iloc[-1].close)

    # transient trading volume info
    volumes = df.volume.values
    large_volume_move = False
    if len(volumes)>3:
        vrk = scipy.stats.percentileofscore( volumes, volumes[-1] )
        if vrk>99.:
            large_volume_move = True
        print( f'-- volume stack: {len(volumes)}, latest volume rank: {vrk:.1f}%' )

    # grid
    bound_breached = pgrid.bound_breached(closep)
    price_moved = pgrid.price_moved(closep)
    if bound_breached or price_moved or large_volume_move:
        print(f"""  
        *** update on ***
           {"new hi/lo" if bound_breached else "price moved" if price_moved else "large trading volumes" if large_volume_move else "somehow?"}
        """)
        pgrid.update()

    # grid info
    print(pgrid)
    ddf = pgrid.distance( closep )
    print('-- current: $', closep)
    print(ddf)

@click.command()
@click.option('--ric',default="DOGE-USDT")
@click.option('--start_ts', default='2024-04-10T07:10:00.000Z', help='for selecting the start of timeframe, usually from visual detection')
@click.option('--test', is_flag=True, default=False)
@click.option('--uniform_grid_gap', default=200., help="bps for uniform grid")
@click.option('--span',default='5m')
def main(ric,start_ts,test, uniform_grid_gap,span):
    ric = ric.upper()
    global pgrid 
    if not pgrid: # Init
        prange = low_freq_price_range(ric,span=span,start_ts=start_ts, is_test=test)
        current_price = prange[-1].iloc[-1].close #test
        pgrid = HFTUniformGrid( current_price, uniform_grid_gap, *prange)
        pgrid.plot()
        print(pgrid)

    if not test:
        from cryptofeed import FeedHandler
        from cryptofeed.backends.aggregate import OHLCV
        from cryptofeed.defines import TRADES
        from cryptofeed.exchanges import Binance

        f = FeedHandler()
        f.add_feed(Binance(symbols=[ric.replace('/','-')],channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=WINDOW_IN_SECONDS)}))
        f.run()
    else:
        print('-- [testing mode] ignore realtime wss connectivity')


if __name__ == '__main__':
    main()
