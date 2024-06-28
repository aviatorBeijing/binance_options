from argparse import Action
import os,click
import pandas as pd
import numpy as np
import talib
from tabulate import tabulate

from butil.portfolio_stats import max_drawdowns,sharpe,sortino,calc_cagr

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

"""
- Purelly technical valleys, not related to any * motivations *.
- For motivation driven valleys, ref: Net Unrealized price_delta/loss (NUPL) on-chain.
"""

#------------- Settings ------------
init_cap = 1_000_000
ff = 8/10000 # fee rate

from signals.meta import ActionT,TradeAction,SignalEmitter

def climb_fall(sym, ts, closes,volume,up_inc=1,down_inc=1, rsi=pd.DataFrame(),file_ts:str=''):
    months = 9 # volume ranking window

    up_inc /= 100
    down_inc /= 100

    wd = 30*months
    rtn = closes.pct_change()
    cumrtn = (1+rtn).cumprod()
    cum_max = cumrtn.cummax()
    volume = talib.EMA(volume, timeperiod=3)
    volrank = volume.rolling(wd).rank(pct=True)
    dd = (cumrtn - cum_max)/cum_max

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
    df = df.drop_duplicates(subset=['ts'])
    df.set_index('ts',inplace=True)
    df = df[wd:]

    is_not_volatile = (df.rtn <= .25) & (df.rtn >= -.25)

    df['rolling_rtn_pct'] = df.rtn.rolling(150).apply(lambda arr: np.percentile(arr, 68)).fillna(up_inc)

    #------------------------- Signals ---------------------
    buy_signals = up_xing = (df.rtn> df.rolling_rtn_pct) & is_not_volatile & df.volrank>0.68
    sell_signals = down_xing = (df.rtn < -df.rolling_rtn_pct) & is_not_volatile & df.volrank>0.68
    #------------------------- Signals ---------------------

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,12))
    
    df.loc[buy_signals, 'bought'] = df.closes
    df.loc[sell_signals, 'sold'] = df.closes
    df.loc[buy_signals, 'bought_n'] = 1
    df.loc[sell_signals, 'sold_n'] = 1

    fee_f = ff
    actions = []
    init_capital = 1_000_000
    cash = init_capital; agg_cash = []
    asset = 0; agg_asset = []
    fee = 0; agg_fee = []
    sz_f = 0.1
    for i, row in df.iterrows():
        is_buy = row.bought >0
        is_sell = row.sold >0
        pce = row.closes
        sz = cash * sz_f / pce
        if is_buy and (cash - sz*pce)>0:
            f = fee_f * sz * pce
            fee += f
            cash -= (sz*pce + f)
            asset += sz
            actions += [ TradeAction(SignalEmitter.CLIMB_AND_FALL, sym, ActionT.BUY, pce, sz, sz_f, i)]
        elif is_sell and asset>0:
            sz = asset * sz_f 
            f = fee_f * sz * pce
            cash += (sz * pce - f)
            fee += f
            asset -= sz
            actions += [ TradeAction(SignalEmitter.CLIMB_AND_FALL, sym, ActionT.SELL, pce, sz, sz_f, i)]
            
        agg_asset += [ asset ]
        agg_cash += [cash]
        agg_fee  += [fee]
    df['agg_pos'] = agg_asset
    df['agg_cash'] = agg_cash
    df['agg_fee']  = agg_fee
    gain = cash + asset*df.closes.iloc[-1] - init_capital
    df = df.fillna(0)

    df['portfolio'] = df.agg_cash + df.agg_pos*df.closes - df.agg_fee
    df['port_rtn'] = df.portfolio.pct_change().fillna(0)
    
    annual = calc_cagr(df.port_rtn)*100
    ds = df.shape[0]

    bh_annual = calc_cagr(df.rtn)*100

    print(f'-- {sym.upper():6s} | ttl% = {(gain/init_capital*100):8,.1f}% | cagr = {annual:8,.1f}% (v.s. {bh_annual:8,.1f}%) {"good" if annual>bh_annual else "    "} | gain = ${gain:20,.2f}, cash = ${cash:,.2f}, asset = {asset:,.1f}, asset value = ${(asset*df.closes.iloc[-1]):,.2f}')

    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    ax33 = ax3.twinx()

    rx = df['closes'].pct_change()
    #(df['dd']*100).plot(ax=ax11,color='blue')
    (((df.rtn+1).cumprod()-1)*100).plot(ax=ax11,color='blue')
    (((df.port_rtn+1).cumprod()-1)*100).plot(ax=ax1,color='red')
    df['agg_pos'].plot(ax=ax2,color='red')
    df['agg_cash'].plot(ax=ax22,color='blue')
    (((1+rx).cumprod()-1)*100).plot(ax=ax22,linewidth=1.5,linestyle='none')
    (df['rolling_rtn_pct']*100).plot(ax=ax3,color='red')
    (df['closes']).plot(ax=ax33,color='blue')
    

    sot = sortino(df.port_rtn)
    bh_sot = sortino(df.rtn)
    ax1.set_ylabel('return%', color='red')
    ax11.set_ylabel('asset return%', color='blue')
    ax1.set_title(f'Portfolio return v.s. asset price (data: {file_ts})\n Sortino: {sot:.2f} v.s. {bh_sot:.2f}')
    ax2.set_title('Asset & Cash')
    ax2.set_ylabel('#Asset',color='red')
    ax22.set_ylabel('$Cash',color='blue')
    ax22.grid(False)
    ax3.set_title('rolling_rtn_pct & price')
    ax3.set_ylabel('%',color='red')
    ax33.set_ylabel('$price',color='blue')
    fn = os.getenv("USER_HOME",'')+f'/tmp/climb_fall_{sym}.pdf' # Trading signals
    plt.savefig(fn)
    print('-- saved:',fn)

    last_action = actions[-1]
    return {
        'symbol': sym.upper(),
        'end': df.index[-1],
        'yrs': round(ds/365,1),
        'single_max_gain_pct': df["port_rtn"].max()*100,
        'single_max_loss_pct': df["port_rtn"].min()*100,
        'cagr%': annual,
        'bh_cagr%': bh_annual,
        'sortino': sot,
        'bn_sortino': bh_sot,
        'last_action': f'{last_action.act.value},{last_action.ts},{last_action.price}' if len(actions)>0 else "",
        'price_now': df.iloc[-1].closes,
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
@click.option("--up_inc", default=1., help="percent of price move")
@click.option("--down_inc", default=1., help="percent of price move")
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
        df.sort_values('last_action', ascending=False, inplace=True)
        print(df)

       
    
if __name__ == '__main__':
    main()
