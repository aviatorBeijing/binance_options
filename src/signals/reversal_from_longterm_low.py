import os,datetime,click
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

# settings
init_cap = 1_000_000;cash=init_cap;pos=0;fees=0.
cash_utility_factor = 0.1 # Each buy can use up to 50%
profit_margin = 100/100. # Useless if trading_horizon>0
ff = 8/10000 # fee rate
# --
trading_horizon = 30*2 # days
order_by_order = trading_horizon<0 # sell when reaching profit margin
hold_fix_days = not order_by_order   # sell on a fixed trading_horizon

if order_by_order: print('-- [order_by_order]')
elif hold_fix_days: print('-- [hold_fix_days]')
else: print('*** unknown')

def pseudo_trade(sym, df, ax=None):
    df = df.copy()

    if trading_horizon>0:
        df['bought'] = df.sig.shift(trading_horizon)

    # trading
    buys = [];nbuys=0;nsells=0
    max_cost = 0;max_cash = 0
    portfolio = [];assets=[]
    cash = init_cap;pos=0;fees=0
    for i, row in df.iterrows():
        #print(i, row.dd, row.closes, row.volrank, row.sig )
        if row.sig>0:
            pce = row.sig;ts = i
            sz = cash * cash_utility_factor / pce
            if sz*pce > init_cap/100: # enough cash FIXME
                buys += [ (ts, pce, sz, )]
                cash -= sz*pce*(1+ff)
                pos += sz
                fees += sz*pce*ff
                nbuys += 1
                if sz*pce > max_cost: max_cost = sz*pce
            else:
                print(f'* insufficient fund: {sz*pce} < {init_cap/100}, {sz}, {pce}')
        else:
            pce = row.closes;ts=i 
            if order_by_order:
                if buys:
                    _buys = list(map(lambda e: e[1], buys))
                    _ix = np.argmin( _buys )
                    #ts0, last_buy, last_buy_sz = buys[-1]
                    ts0, last_buy, last_buy_sz = buys[_ix]
                    if (pce-last_buy)/last_buy > profit_margin: # 10% profit
                        #_ = buys.pop()
                        buys = buys[:_ix] + buys[_ix+1:] if (_ix+1)<len(buys) else []
                        cash += pce*last_buy_sz*(1-ff)
                        pos -= last_buy_sz
                        fees += last_buy_sz * pce * ff
                        nsells += 1
            elif hold_fix_days:
                if row.bought>0:
                    sz = 0; ix = -1
                    for i,b in enumerate( buys):
                        sz = b[2]
                        ix = i
                    if sz>0:
                        cash += pce*sz
                        pos -= sz
                        fees += pce*sz *ff
                        nsells += 1
                        buys = buys[:ix] + (buys[ix+1:] if (ix+1)<len(buys) else [])
            else:
                raise Exception('Strategy is not specified.')
        if cash > max_cash: max_cash = cash
        portfolio += [ cash + pos * row.closes - fees ]
        assets += [ pos ]
    pdf = pd.DataFrame.from_dict({'v': portfolio, 'assets': assets})
    pdf.index = list( map(lambda e: pd.Timestamp(e), df.index ))
    pdf  = pdf.resample('1d').agg('sum')
    
    pdf = pdf[pdf.v>0]
    
    r1 = pdf['v'].pct_change()
    rr = df['closes'].pct_change()
    rinc = r1.sum()/rr.sum()
    
    max_dd = max_drawdowns( r1 )
    max_lev = 1./(-max_dd)
    #r1 *= max_lev 

    if not ax:
        fig, ax1 = plt.subplots(1,1,figsize=(24,8))
    else:
        ax1 = ax
    ax11 = ax1.twinx()
    ((r1*100).cumsum()).plot(ax=ax1)
    pdf['assets'].plot(ax=ax11,color='gray',alpha=0.5)
    #df['closes'].pct_change().cumsum().plot(ax=ax11,color='gray',alpha=0.5)
    ax1.set_ylabel('Return%',color='blue')
    ax11.set_ylabel('Position (#)',color='gray')
    ax1.set_title( f'{sym}\nsell in {trading_horizon} days after each buy, max lev: {max_lev:.1f}, r1/rr: {(rinc):.1f} (w/ lev:{(rinc*max_lev):.1f})' )
    
    if not ax:
        fn = os.getenv("USER_HOME","")+f"/tmp/port_{sym}.png"
        plt.savefig( fn )
        print('-- saved:', fn)

    val = cash+pos*float(df.iloc[-1].closes)-fees
    rtn = (val/init_cap-1)*100 #total return
    profits = val - init_cap
    cagr = np.log(1+rtn/100) / (df.shape[0]/365)
    cagr = (np.exp( cagr ) -1 )*100

    max_dd_ref = max_drawdowns( rr )
    sortino1 = sortino(r1)
    sortino_ref = sortino(rr)
    sharpe1 = sharpe(r1)
    sharpe_ref = sharpe(rr)

    print('-- pseudo trading (strategy specific!):')
    print(f'  -- max drawdown: {(max_dd*100):.1f}%, {(max_dd_ref*100):.1f}%, sortino: {sortino1:.2f}, {sortino_ref:.2f}, sharpe: {sharpe1:.2f}, {sharpe_ref:.2f}')
    print(f'  -- nbuys: {nbuys}, nsells: {nsells}, max cost: ${max_cost:,.2f}, max cash: ${max_cash:,.2f}')
    print(f'  -- liquidation value: ${(val):,.2f}, rtn: {rtn:,.1f}%, cagr: {cagr:,.1f}%')
    print(f'    -- cash: ${cash:,.2f}')
    print(f'    -- pos: {pos}, ${(pos*float(df.iloc[-1].closes)):,.2f}')
    print(f'    -- net profits: ${profits:,.2f}')
    print(f'    -- fees: ${fees:,.2f}, {(fees/profits*100):,.2f}%')
    print()
    
    return sym, int(cagr*10)/10, \
        int(max_dd*100)/100, int(max_dd_ref*100)/100, \
        nbuys,nsells, \
        (int(max_dd_ref/max_dd*100)/100), \
        (int(rinc*100)/100), \
        (int(sharpe1/sharpe_ref*100)/100), (int(sortino1/sortino_ref*100)/100)

def find_reversals(sym, ts, closes,volume,volt=50):
    months = 9 # volume ranking window
    trade_horizon = 30*2    # how long to wait to sell after buy

    wd = 30*months
    rtn = closes.pct_change()
    cumrtn = (1+rtn).cumprod()
    cum_max = cumrtn.cummax()
    volume = talib.EMA(volume, timeperiod=5)
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
    df.set_index('ts',inplace=True)
    df = df[wd:]
    df = df[ -365*2: ]

    #------------------------- Signals ---------------------
    #rank_xing = (df.volrank.shift(3) <= volt/100) & (df.volrank.shift(2) <= volt/100) & (df.volrank.shift(1) >= volt/100) & (df.volrank>=volt/100)
    rank_xing = (df.volrank.shift(2) <= volt/100) & (df.volrank.shift(1) <= volt/100) & (df.volrank>=volt/100)
    # (df.dd<-0.5) & 

    df.loc[ rank_xing, 'sig'] = df.closes
    df.loc[ rank_xing, 'r1'] = df.volrank

    # remove high frequency xing
    fwd = 14
    if fwd>0:
        df['sig_count'] = df.sig.fillna(0).rolling(fwd).apply(lambda arr: np.count_nonzero( np.array(arr) ) )
        df.loc[ df.sig_count>1, 'sig'] = np.nan;df.loc[ df.sig_count>1, 'r1'] = np.nan
    #------------------------- Signals ---------------------

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(24,18))

    pseudo_metrics = pseudo_trade(sym, df, ax3) # a different trading strategy!

    df['bought'] = df.sig.shift(trade_horizon) # after one month (trading strategy)
    xdf = df[df.bought>0][['bought','closes']].copy()
    xdf['price_delta'] = df.closes-df.bought
    xdf['return'] = xdf.price_delta/xdf.bought
    aggrtn =  (((xdf['return']+1).prod() -1 )*100)
    
    annual = np.log(aggrtn/100+1)/(df.shape[0]/365)
    annual = ( np.exp(annual)- 1)*100
    ds = df.shape[0]

    latest = df[df.sig>0].tail(1)
    lastsell = xdf.tail(1)

    bh = (df.closes.iloc[-1] - df.closes.iloc[0])/df.closes.iloc[0]*100
    bh_annual = np.log(bh/100+1)/(df.shape[0]/365)
    bh_annual = ( np.exp(bh_annual)- 1)*100

    cap = init_capital = 10_000
    """if not latest.empty:
        print(f'-- trades:\n  -- {latest.index[0]}, buy at ${latest.closes.iloc[0]}, sell after {trade_horizon} days')
    if not lastsell.empty:
        print(f'  -- {lastsell.index[0]}, sell at ${lastsell.closes.iloc[0]}, gain {(lastsell["return"].iloc[0]*100):.2f}% (cost ${lastsell.bought.iloc[0]:.2f})', '\n')
    """
    
    print(f'-- gain: ${(xdf["return"].sum()*cap ):,.2f} (initial capital: ${cap:,.2f}, fixed investment mode)')
    print(f'-- ttl return: {aggrtn:.1f}% ({ds} days, {(ds/365):.1f} yrs, reinvest mode)')
    print(f'  -- buy&hold: {bh:.1f}%, {bh_annual:.1f}%')
    print(f'  -- cagr: {annual:.1f}%')
    print(f'  -- single max gain: {(xdf["return"].max()*100):.1f}%' )
    print(f'  -- single max loss: {(xdf["return"].min()*100):.1f}%' )
    print(f'  -- {xdf.shape[0]}, wins: {xdf[xdf["price_delta"]>0].shape[0]}, losses: {xdf[xdf["price_delta"]<0].shape[0]}')

    
    ax11 = ax1.twinx()

    df['dd'].plot(ax=ax1,color='red')
    df['volrank'].plot(ax=ax11,alpha=0.5)
    df['closes'].plot(ax=ax2,linewidth=1.5)
    df['sig'].plot(ax=ax2,marker="^",linestyle="none",color="red", alpha=0.6)
    df['r1'].plot(ax=ax11,marker="^",linestyle="none",color="red", alpha=0.6)
    
    ax1.set_title(f'equity drawdown v.s. volume ranking')
    ax2.set_title('price & buying signals')
    ax2.set_ylabel('price ($)')
    fn = os.getenv("USER_HOME",'')+f'/tmp/reversal_{sym}.pdf'
    plt.savefig(fn)
    print('-- saved:',fn)

    if not latest.empty:
        print(f'-- trades:\n  -- {latest.index[0]}, buy at ${latest.closes.iloc[0]}, sell after {trade_horizon} days')
    if not lastsell.empty:
        print(f'  -- {lastsell.index[0]}, sell at ${lastsell.closes.iloc[0]}, gain {(lastsell["return"].iloc[0]*100):.2f}% (cost ${lastsell.bought.iloc[0]:.2f})', '\n')

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
        #'last_sell': f'{lastsell.index[0]},{lastsell.closes.iloc[0]},{lastsell.bought.iloc[0]}' if not lastsell.empty else "",
        'pseudo_trade': pseudo_metrics,
    }

def _main(sym, volt):
    print('-'*14)
    print(f'|    {sym.upper()}    |')
    print('-'*14)

    sym = sym.lower()
    fn = os.getenv("USER_HOME","") + f'/tmp/{sym}-usdt_1d.csv'
    if not os.path.exists(fn):
        print(f'*** {fn} doesn\'t exist.')
        from butil.butils import binance_kline
        df = binance_kline(f'{sym.upper()}/USDT', span='1d', grps=5)
    else:
        df = pd.read_csv( fn,index_col=0 )
    ts = df.timestamp
    closes = df.close
    volume = df.volume 
    rec = find_reversals(sym, ts, closes, volume,volt )
    return rec

@click.command()
@click.option("--sym", default='doge')
@click.option("--syms", default='')
@click.option("--volt", default=50, help="percentage of volume considered to be significant")
def main(sym,syms,volt):
    global cash_utility_factor, trading_horizon
    if syms:
        recs = []
        if len(syms.split(","))>1:
            import multiprocessing,functools
            with multiprocessing.Pool(5) as pool:
                recs = pool.map( functools.partial(_main, volt=volt), syms.split(","))
                pool.close();pool.join()
        else:
            for sym in syms.split(','):
                rec = _main(sym, volt )
                recs += [rec]
        df = pd.DataFrame.from_records( recs )
        df.sort_values('last_buy', ascending=False, inplace=True)
        print(df)

        pseudo_df = pd.DataFrame.from_records(
            list(df.pseudo_trade),
            columns=['crypto', 'cagr','max_dd','max_dd_ref','#buys','#sells',
            'dd_inc','tt_rtn_inc','sharpe_inc','sortino_inc'])
        pseudo_df['lev'] = 1/(-pseudo_df['max_dd']);pseudo_df.lev = pseudo_df.lev.apply(lambda v: int(v*10)/10)
        pseudo_df['tt_cagr (lev.)'] = pseudo_df['cagr'] * pseudo_df['lev']
        pseudo_df = pseudo_df.sort_values(['max_dd'], ascending=False)
        print()
        print( tabulate(pseudo_df,headers='keys') )
    else:
       rec = _main(sym,volt)
    
    print('-- settings:')
    print(f'  trading horizon: {trading_horizon} days after a buy')
    print(f'  cash_utility_factor: {(cash_utility_factor*100):.1f}%')
    

if __name__ == '__main__':
    main()