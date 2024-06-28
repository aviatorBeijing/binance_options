import os,click
import pandas as pd
import numpy as np
import talib
from tabulate import tabulate

from butil.portfolio_stats import max_drawdowns,sharpe,sortino

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

"""
- Purelly technical valleys, not related to any * motivations *.
- For motivation driven valleys, ref: Net Unrealized price_delta/loss (NUPL) on-chain.
"""

#------------- Settings ------------
init_cap = 1_000_000
ff = 8/10000 # fee rate

#strategies: 1) Sell at fixed days in the future; or 2) obey TP/SL rules.
trading_horizon = -1 #7*4 # days in case of "sell at fixed days in the future"
cash_utility_factor = -1 #0.3 # Each buy can use up to 25%
tp = profit_margin = 25/100. # Useless if trading_horizon>0
sl = 15/100.

#data
def select_data(df): return df#[ -365*3: ] # Recent 3 years (since) are considered a "normal" market.

#flags
order_by_order = trading_horizon<0 # sell when reaching profit margin
hold_fix_days = not order_by_order   # sell on a fixed trading_horizon
#------------------------------------

if order_by_order: print('-- [order_by_order]')
elif hold_fix_days: print('-- [hold_fix_days]')
else: print('*** unknown')

# utils
def _cl(s): return str(s).replace('00:00:00+00:00','')
def _cagr(rtn,n):
    cagr = np.log(1+rtn) / (n/365)
    cagr = (np.exp( cagr ) -1 )*100
    return cagr

import enum 
class ActionT(enum.Enum):
    BUY = 'buy'
    SELL = 'sell'
    TP  = 'tp'
    SL  = 'sl'
class TradeAction:
    def __init__(self, sym: str,act:ActionT, price:float, sz:float, sz_f: float, ts:str) -> None:
        assert sz_f<=1., 'assume no leverage'
        self.act = act 
        self.price = price
        self.sz=sz
        self.ts = ts 
        self.sz_f = sz_f

        sym = sym.upper()
        self.ric = f'{sym}/USDT' if not 'USDT' in sym else sym
    def to_df(self):
        df = pd.DataFrame.from_dict({
            'ric': [self.ric],
            'action': [ self.act.value ],
            'price': [self.price],
            'sz': [self.sz],
            'sz_f': [self.sz_f],
            'ts': [ str(self.ts) ],
        })
        return df
    def is_buy(self):
        return self.act == ActionT.BUY

    def __str__(self) -> str:
        s = f' {self.ts}: {self.ric} {self.act}, ${self.price}, {self.sz}, {self.sz_f:.3f}'
        return s

def climb_fall(sym, ts, closes,volume,up_inc=1,down_inc=1, rsi=pd.DataFrame(),file_ts:str=''):
    months = 9 # volume ranking window
    trade_horizon = 30*2    # how long to wait to sell after buy

    up_inc /= 100
    down_inc /= 100

    wd = 30*months
    rtn = closes.pct_change()
    cumrtn = (1+rtn).cumprod()
    cum_max = cumrtn.cummax()
    volume = talib.EMA(volume, timeperiod=3)
    volrank = volume.rolling(wd).rank(pct=True)
    dd = (cumrtn - cum_max)/cum_max

    #dd = dd.rolling(wd).rank(pct=True)
    #dd = dd-1
    
    # remove nan
    dd = dd[1:]
    ts = ts[1:]
    volume = volume[1:]
    volrank = volrank[1:]
    closes = closes[1:]
    if not rsi.empty:
        rsi = rsi[1:]
    
    # find min
    i_mm = np.argmin(dd)
    if i_mm+1 != dd.shape[0]-1:
        i_mm += 1
    print( '-- max drawdown:', ts[i_mm] )
    print( f'  -- volume rank: {(volrank[i_mm]*100):.1f}%')
    print( f'  -- max drawdown: {(dd[i_mm]*100):.1f}%', '\n')

    df = pd.DataFrame()
    df['ts'] = ts;df.ts = df.ts.apply(pd.Timestamp)
    df['dd'] = dd
    df['closes'] = closes
    df['volrank'] = volrank
    if not rsi.empty: df['rsi'] = rsi
    df['rtn'] = df['closes'].pct_change()
    df['1sigma'] = df.closes.rolling(120).std() # TODO: use ARIMA to predict the std (i.e., vol) of return.
    df = df.drop_duplicates(subset=['ts'])
    df.set_index('ts',inplace=True)
    df = df[wd:]
    df = select_data(df)

    is_not_volatile = df.rtn <= .25

    #------------------------- Signals ---------------------
    buy_signals = up_xing = (df.rsi<20 if 'rsi' in df else False ) | \
                              (   is_not_volatile \
                                & df.rtn> up_inc
                              )
    sell_signals = down_xing = (df.rsi>80 if 'rsi' in df else False ) | \
                              (   is_not_volatile \
                                & df.rtn < -down_inc
                              )

    df.loc[ up_xing, 'sig'] = df.closes

    #------------------------- Signals ---------------------

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,12))

    df['bought'] = df.sig.shift(trade_horizon) # after one month (trading strategy)
    xdf = df[df.bought>0][['bought','closes']].copy()
    xdf['price_delta'] = df.closes-df.bought
    xdf['return'] = xdf.price_delta/xdf.bought
    aggrtn =  (((xdf['return']+1).prod() -1 )*100)
    
    annual = _cagr( aggrtn/100, df.shape[0])
    ds = df.shape[0]

    latest = df[df.sig>0].tail(1)
    lastsell = xdf.tail(1)

    bh = (df.closes.iloc[-1] - df.closes.iloc[0])/df.closes.iloc[0]*100
    bh_annual = _cagr( bh/100, df.shape[0])
    cap = init_capital = 10_000
   
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()

    rx = df['closes'].pct_change()
    (df['dd']*100).plot(ax=ax1,color='red')
    df['closes'].plot(ax=ax2,linewidth=1.5)
    (((1+rx).cumprod()-1)*100).plot(ax=ax22,linewidth=1.5,linestyle='none')
    ax1.set_ylabel('drawdown%', color='red')

    df['sig'].plot(ax=ax2,marker="^",linestyle="none",color="red", alpha=0.6)
    
    ax1.set_title(f'equity drawdown v.s. volume ranking (data: {file_ts})')
    ax2.set_title('price & buying signals')
    ax2.set_ylabel('price ($)')
    ax22.set_ylabel('return (%)')
    ax22.grid(False)
    fn = os.getenv("USER_HOME",'')+f'/tmp/climb_fall_{sym}.pdf' # Trading signals
    plt.savefig(fn)
    print('-- saved:',fn)

    """if not latest.empty:
        print(f'-- trades:\n  -- {latest.index[0]}, buy at ${latest.closes.iloc[0]}, sell after {trade_horizon} days')
    if not lastsell.empty:
        print(f'  -- {lastsell.index[0]}, sell at ${lastsell.closes.iloc[0]}, gain {(lastsell["return"].iloc[0]*100):.2f}% (cost ${lastsell.bought.iloc[0]:.2f})', '\n')
    """

    return {
        'crypto': sym.upper(),
        'start': df.index[0],
        'end': df.index[-1],
        'wins': xdf[xdf["price_delta"]>0].shape[0],
        'losses': xdf[xdf["price_delta"]<0].shape[0],
        'single_max_gain_pct': xdf["return"].max()*100,
        'single_max_loss_pct': xdf["return"].min()*100,
        'cagr_pct': annual,
        'days': ds,
        'last_buy': f'{latest.index[0]},{latest.closes.iloc[0]}' if not latest.empty else "",
        'price': df.iloc[-1].closes,
    }

def _file_ts(fn):
    import datetime
    s = os.stat(fn)
    t = int( s.st_mtime )
    d = datetime.datetime.fromtimestamp(int(s.st_mtime) )
    return str(d)

def _main(sym, up_inc, down_inc, offline=False):
    print('-'*14)
    print(f'|    {sym.upper()}    |')
    print('-'*14)

    sym = sym.lower()
    fn = os.getenv("USER_HOME","") + f'/tmp/{sym}-usdt_1d.csv'
    file_ts = _file_ts( fn )
    if not offline or not os.path.exists(fn):
        from butil.butils import binance_kline
        df = binance_kline(f'{sym.upper()}/USDT', span='1d', grps=5)
        print(df.tail(3))
    else:
        df = pd.read_csv( fn,index_col=0 )
    ts = df.timestamp

    #rsi
    df['rsi'] = talib.RSI(df['close'],timeperiod=14)
    
    df = df.dropna()
    rsi = df.rsi
    
    closes = df.close
    volume = df.volume 
    rec = climb_fall(sym, ts, closes, volume, up_inc, down_inc, rsi, file_ts if offline else 'realtime' )
    return rec

@click.command()
@click.option("--syms", default='')
@click.option("--up_inc", default=1, help="percent of price move")
@click.option("--down_inc", default=1, help="percent of price move")
@click.option("--offline", is_flag=True,default=False )
def main(syms,up_inc,down_inc,offline):
    global cash_utility_factor, trading_horizon, init_cap
 
    if syms:
        recs = []
        if len(syms.split(","))>1:
            import multiprocessing,functools
            with multiprocessing.Pool(5) as pool:
                recs = pool.map( functools.partial(_main, up_inc=up_inc,down_inc=down_inc,offline=offline), syms.split(","))
                pool.close();pool.join()
        else:
            for sym in syms.split(','):
                rec = _main(sym, up_inc,down_inc, offline )
                recs += [rec]
        df = pd.DataFrame.from_records( recs )
        df.sort_values('last_buy', ascending=False, inplace=True)
        print(df)

       
    
if __name__ == '__main__':
    main()
