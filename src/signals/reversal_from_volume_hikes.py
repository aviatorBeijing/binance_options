from email import header
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
cash_utility_factor = 0.25 # Each buy can use up to 25%
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
    def __init__(self, sym: str,act:ActionT, price:float, sz:float, ts:str) -> None:
        self.act = act 
        self.price = price;self.sz=sz
        self.ts = ts 

        sym = sym.upper()
        self.ric = f'{sym}/USDT' if not 'USDT' in sym else sym
    def to_df(self):
        df = pd.DataFrame.from_dict({
            'ric': [self.ric],
            'action': [ self.act.value ],
            'price': [self.price],
            'sz': [self.sz],
            'ts': [ str(self.ts) ],
        })
        return df
    def is_buy(self):
        return self.act == ActionT.BUY

    def __str__(self) -> str:
        s = f' {self.ts}: {self.ric} {self.act}, ${self.price}, {self.sz}'
        return s
    
def pseudo_trade(sym, df, ax=None):
    print('-- pseudo trading (strategy specific!):')
    df = df.copy()
    print('-- latest data:', df.index[-1], f"${df.iloc[-1].closes}")

    if trading_horizon>0:
        df['bought'] = df.sig.shift(trading_horizon)

    cash=init_cap
    pos=0
    fees=0.

    # trading
    buys = [];nbuys=0;nsells=0;stoplosses=0
    max_cost = 0;max_cash = 0
    portfolio = [];assets=[]
    cash = init_cap;pos=0;fees=0
    cash_min = init_cap
    trade_actions = []
    wins = 0; losses = 0
    for i, row in df.iterrows():
        #print(i, row.dd, row.closes, row.volrank, row.sig )
        is_breaking_down = row['1sigma_dw_sig_flag']
        is_breaking_up   = row['1sigma_up_sig_flag']
        if row.sig>0 and not is_breaking_down:
            pce = row.sig;ts = i
            #sz = cash * cash_utility_factor / pce # Use the fixed factor
            sz = cash * row.volrank / pce # Using the volume rank as the factor

            if sz*pce > init_cap/100: # enough cash FIXME
                buys += [ (ts, pce, sz, )]
                cash -= sz*pce*(1+ff)
                pos += sz
                fees += sz*pce*ff
                nbuys += 1
                if cash < cash_min: cash_min = cash # record the cash usage
                if sz*pce > max_cost: max_cost = sz*pce
                print(f'  [buy] {sym},', _cl(str(i)), f'${pce}, sz: {sz}, cap%: { (row.volrank*100):.1f}%')

                trade_actions+=[ TradeAction(sym, ActionT.BUY, pce, sz, ts) ]
            else:
                print(f'* insufficient fund: {sz*pce} < {init_cap/100}, {sz}, {pce}')
        else:
            pce = row.closes;ts=i 
            if order_by_order:
                if buys:
                    # tp
                    _buys = list(map(lambda e: e[1], buys))
                    _ix = np.argmin( _buys )
                    ts0, last_buy, last_buy_sz = buys[_ix]
                    if (pce-last_buy)/last_buy > (profit_margin): # met the profit traget
                        buys = buys[:_ix] + (buys[_ix+1:] if (_ix+1)<len(buys) else [])
                        print( '    [tp]:', _cl(str(i)), f'${pce}', ', the buy:', _cl(ts0), f'${last_buy}', ',holding:', (i-ts0).total_seconds()/3600/24, 'days') #, (pce-last_buy)/last_buy,'>', profit_margin)
                        cash += pce*last_buy_sz*(1-ff)
                        pos -= last_buy_sz
                        fees += last_buy_sz * pce * ff
                        nsells += 1
                        trade_actions+=[ TradeAction(sym, ActionT.TP, pce, last_buy_sz, str(i)) ]
                        wins += 1
                # sl
                if buys:
                        _buys = list(map(lambda e: e[1], buys))
                        _ix = np.argmax( _buys )
                        ts0, last_buy, last_buy_sz = buys[_ix]
                        if (pce-last_buy)/last_buy < -sl or is_breaking_down: # sl
                            buys = buys[:_ix] + (buys[_ix+1:] if (_ix+1)<len(buys) else [])
                            print( '      [sl]:', _cl(str(i)), f'${pce}', ', the buy:', _cl(ts0), f'${last_buy}', len(buys), _ix ) #, (pce-last_buy)/last_buy,'<', -sl)
                            cash += pce*last_buy_sz*(1-ff)
                            pos -= last_buy_sz
                            fees += last_buy_sz * pce * ff
                            nsells += 1
                            stoplosses += 1
                            trade_actions+=[ TradeAction(sym, ActionT.SL, pce, last_buy_sz, str(i)) ]

                            if (pce-last_buy)<0: losses += 1
                            else: wins += 1

            elif hold_fix_days:
                if row.bought>0:
                    sz = 0; ix = -1
                    for j,b in enumerate( buys):
                        sz = b[2]
                        ix = j
                    if sz>0:
                        cash += pce*sz
                        pos -= sz
                        fees += pce*sz *ff
                        nsells += 1
                        buys = buys[:ix] + (buys[ix+1:] if (ix+1)<len(buys) else [])
                        trade_actions+=[ TradeAction(sym, ActionT.SELL, pce, sz, str(i)) ]
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
    
    if not ax:
        fn = os.getenv("USER_HOME","")+f"/tmp/port_{sym}.png"
        plt.savefig( fn )
        print('-- saved:', fn)

    val = cash+pos*float(df.iloc[-1].closes)-fees
    rtn = (val/init_cap-1)*100 #total return
    
    profits = val - init_cap
    cagr = _cagr( rtn/100, df.shape[0])

    ax1.set_title( f'{sym.upper()}, cagr={cagr:,.1f}%, sell in {trading_horizon} days after each buy, max lev: {max_lev:.1f}, r1/rr: {(rinc):.1f} (w/ lev:{(rinc*max_lev):.1f})' )
    
    max_dd_ref = max_drawdowns( rr )
    sortino1 = sortino(r1)
    sortino_ref = sortino(rr)
    sharpe1 = sharpe(r1)
    sharpe_ref = sharpe(rr)

    print(f'\n  -- max drawdown: {(max_dd*100):.1f}%, {(max_dd_ref*100):.1f}%, sortino: {sortino1:.2f}, {sortino_ref:.2f}, sharpe: {sharpe1:.2f}, {sharpe_ref:.2f}')
    print(f'  -- nbuys: {nbuys}, nsells: {nsells}, max cost: ${max_cost:,.2f}, max cash: ${max_cash:,.2f}')
    print(f'  -- liquidation value: ${(val):,.2f}, rtn: {rtn:,.1f}%, cagr: {cagr:,.1f}%')
    print(f'    -- cash: ${cash:,.2f}')
    print(f'    -- pos: {pos}, ${(pos*float(df.iloc[-1].closes)):,.2f}')
    print(f'    -- net profits: ${profits:,.2f}')
    print(f'    -- fees: ${fees:,.2f}, {(fees/profits*100):,.2f}%')
    print()
    
    years = int(df.shape[0]/365.*100)/100
    return sym, years, \
        int(cagr*10)/10, \
        rtn, \
        sortino1, cash_min, \
        int(max_dd*100)/100, int(max_dd_ref*100)/100, \
        nbuys,nsells,stoplosses, \
        (int(max_dd_ref/max_dd*100)/100), \
        (int(rinc*100)/100), \
        (int(sharpe1/sharpe_ref*100)/100), (int(sortino1/sortino_ref*100)/100), \
        r1, rr, \
            trade_actions

def add_sigma_signals(df):
    df = df.copy()
    df['1sigma_up'] = df['closes'].shift(1)+df['1sigma']
    df['1sigma_dw'] = df['closes'].shift(1)-df['1sigma']
    
    #print(((df['closes']-df['1sigma_dw']).rolling(90).rank(pct=True)).min() )
    
    df['sigma_level'] = df['1sigma'].rolling(120).rank(pct=True)

    df['1sigma_up_sig_flag'] = ( 
                                    (df['closes'] > df['1sigma_up']) 
                                    | ((df['1sigma_up']-df['closes']).rolling(120).rank(pct=True)<0.02) 
                                ) \
                                & (
                                    ((df.closes - df['1sigma_up'])/df['1sigma']).apply(abs)<10./100
                                )
    df['1sigma_dw_sig_flag'] = ( 
                                    (df['closes'] < df['1sigma_dw']) 
                                    | ((df['closes']-df['1sigma_dw']).rolling(120).rank(pct=True)<0.02) # relative
                                ) \
                                & (
                                    ((df.closes - df['1sigma_dw'])/df['1sigma']).apply(abs)<10./100  # absolute
                                )
    
    df.loc[df['1sigma_up_sig_flag']==True, '1sigma_up_sig'] = df.closes
    df.loc[df['1sigma_dw_sig_flag']==True, '1sigma_dw_sig'] = df.closes
    return df 

def find_reversals(sym, ts, closes,volume,volt=50,rsi=pd.DataFrame()):
    months = 9 # volume ranking window
    trade_horizon = 30*2    # how long to wait to sell after buy

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

    #------------------------- Signals ---------------------
    buy_signals = rank_xing = (df.rsi<20 if 'rsi' in df else False ) | \
                              (   (df.rtn <= .25) \
                                & (df.volrank.shift(2) <= volt/100) \
                                & (df.volrank.shift(1) <= volt/100) \
                                & (df.volrank>=volt/100)
                              )
    # (df.dd<-0.5) & 

    df.loc[ rank_xing, 'sig'] = df.closes
    df.loc[ rank_xing, 'r1'] = df.volrank

    # Add sigma signals
    df = add_sigma_signals( df )

    # remove high frequency xing
    fwd = 14
    if fwd>0:
        df['sig_count'] = df.sig.fillna(0).rolling(fwd).apply(lambda arr: np.count_nonzero( np.array(arr) ) )
        df.loc[ df.sig_count>1, 'sig'] = np.nan;df.loc[ df.sig_count>1, 'r1'] = np.nan
    #------------------------- Signals ---------------------

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(18,12))

    pseudo_metrics = pseudo_trade(sym, df, ax3) # a different trading strategy!

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
    """if not latest.empty:
        print(f'-- trades:\n  -- {latest.index[0]}, buy at ${latest.closes.iloc[0]}, sell after {trade_horizon} days')
    if not lastsell.empty:
        print(f'  -- {lastsell.index[0]}, sell at ${lastsell.closes.iloc[0]}, gain {(lastsell["return"].iloc[0]*100):.2f}% (cost ${lastsell.bought.iloc[0]:.2f})', '\n')
    
    print(f'-- FIXED DATE SELL TEST ({trade_horizon} days)')
    print(f'-- gain: ${(xdf["return"].sum()*cap ):,.2f} (initial capital: ${cap:,.2f}, fixed investment mode)')
    print(f'-- ttl return: {aggrtn:.1f}% ({ds} days, {(ds/365):.1f} yrs, reinvest mode)')
    print(f'  -- buy&hold: {bh:.1f}%, {bh_annual:.1f}%')
    print(f'  -- cagr: {annual:.1f}%')
    print(f'  -- single max gain: {(xdf["return"].max()*100):.1f}%' )
    print(f'  -- single max loss: {(xdf["return"].min()*100):.1f}%' )
    print(f'  -- {xdf.shape[0]}, wins: {xdf[xdf["price_delta"]>0].shape[0]}, losses: {xdf[xdf["price_delta"]<0].shape[0]}')
    """
    
    ax11 = ax1.twinx()

    df['dd'].plot(ax=ax1,color='red')
    df['volrank'].plot(ax=ax11,alpha=0.5)
    df['closes'].plot(ax=ax2,linewidth=1.5)

    df['1sigma_up'].plot(ax=ax2,linewidth=1,color='gray',alpha=0.3)
    df['1sigma_dw'].plot(ax=ax2,linewidth=1,color='gray',alpha=0.3)

    df['1sigma_up_sig'].plot(ax=ax2,marker="x",linestyle="none",color="red", alpha=0.6)
    df['1sigma_dw_sig'].plot(ax=ax2,marker="x",linestyle="none",color="black", alpha=0.6)
    
    df['sig'].plot(ax=ax2,marker="^",linestyle="none",color="red", alpha=0.6)
    df['r1'].plot(ax=ax11,marker="^",linestyle="none",color="red", alpha=0.6)
    
    ax1.set_title(f'equity drawdown v.s. volume ranking')
    ax2.set_title('price & buying signals')
    ax2.set_ylabel('price ($)')
    fn = os.getenv("USER_HOME",'')+f'/tmp/reversal_{sym}.pdf' # Trading signals
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
        #'last_sell': f'{lastsell.index[0]},{lastsell.closes.iloc[0]},{lastsell.bought.iloc[0]}' if not lastsell.empty else "",
        'pseudo_trade': pseudo_metrics,
    }

def _main(sym, volt,offline=False):
    print('-'*14)
    print(f'|    {sym.upper()}    |')
    print('-'*14)

    sym = sym.lower()
    fn = os.getenv("USER_HOME","") + f'/tmp/{sym}-usdt_1d.csv'
    if not offline or not os.path.exists(fn):
        from butil.butils import binance_kline
        df = binance_kline(f'{sym.upper()}/USDT', span='1d', grps=5)
        print(df.tail(3))
    else:
        df = pd.read_csv( fn,index_col=0 )
    ts = df.timestamp
    df['rsi'] = talib.RSI(df['close'],timeperiod=14)
    df = df.dropna()
    rsi = df.rsi
    closes = df.close
    volume = df.volume 
    rec = find_reversals(sym, ts, closes, volume,volt,rsi )
    return rec

@click.command()
@click.option("--sym", default='doge')
@click.option("--syms", default='')
@click.option("--volt", default=50, help="percentage of volume considered to be significant")
@click.option("--offline", is_flag=True, default=False)
@click.option("--do_mpt", is_flag=True,default=False, help='turn on the MPT simulation')
def main(sym,syms,volt,offline,do_mpt):
    global cash_utility_factor, trading_horizon, init_cap
    def _settings():
        print('-- settings:')
        if trading_horizon>0:
            print(f'  trading horizon: {trading_horizon} days after a buy')
        else:
            print(f'  tp/sl: {(profit_margin*100):.1f}%/-{(sl*100):.1f}% (for every buy order)')
        print(f'  cash_utility_factor: {(cash_utility_factor*100):.1f}%')

    if syms:
        recs = []
        if len(syms.split(","))>1:
            import multiprocessing,functools
            with multiprocessing.Pool(5) as pool:
                recs = pool.map( functools.partial(_main, volt=volt,offline=offline), syms.split(","))
                pool.close();pool.join()
        else:
            for sym in syms.split(','):
                rec = _main(sym, volt, offline )
                recs += [rec]
        df = pd.DataFrame.from_records( recs )
        df.sort_values('last_buy', ascending=False, inplace=True)
        #print(df)

        pseudo_df = pd.DataFrame.from_records(
            list(df.pseudo_trade),
            columns=['crypto', 'years', 'cagr','tt_rtn','sortino','cash_util','max_dd','max_dd_ref','#buys','#sells','sl/sell',
            'dd/ref','tt_rtn/ref','sharpe/ref','sortino/ref',
            'r1','rr',
            'trade_actions'
            ])
        pseudo_df['sl/sell'] = ( pseudo_df['sl/sell']/pseudo_df['#sells']*100 ).apply(lambda v: f"{v:.0f}%")
        pseudo_df['lev'] = 1/(-pseudo_df['max_dd']);pseudo_df.lev = pseudo_df.lev.apply(lambda v: int(v*10)/10)
        pseudo_df['cagr (lev.)'] = pseudo_df['cagr'] * pseudo_df['lev']
        pseudo_df['tt_rtn/ref (lev.)'] = pseudo_df['tt_rtn/ref'] * pseudo_df['lev']
        pseudo_df = pseudo_df.sort_values(['cagr'], ascending=False)
        pseudo_df.reset_index(drop=True,inplace=True)
        
        pseudo_df['cash_util']=1 - pseudo_df['cash_util']/init_cap # FIXME TODO
        for col in ['sortino']:
            pseudo_df[col] = pseudo_df[col].apply(lambda v: f"{v:.1f}")
        pseudo_df['max_dd'] *= 100
        pseudo_df['cash_util'] *= 100
        for col in ['max_dd','cagr','tt_rtn','cagr (lev.)','cash_util']: 
            pseudo_df[col] = pseudo_df[col].apply(lambda v: f"{(v):,.1f}%")

        pseudo_df['tt_rtn'] = pseudo_df['tt_rtn'] + pseudo_df['tt_rtn/ref (lev.)'].apply(lambda v: f" (lev./ref: {abs(v):.1f})")
        pseudo_df.drop(['max_dd_ref','sharpe/ref','tt_rtn/ref (lev.)'],axis=1,inplace=True)

        r1 = list(pseudo_df.r1.values)
        r1 = pd.concat(r1,axis=1).dropna()
        r1.columns = pseudo_df.crypto.values

        rr = list(pseudo_df.rr.values)
        rr = pd.concat(rr,axis=1).dropna()
        rr.columns = pseudo_df.crypto.values

        import datetime
        def _tdiff(t):
            try:
                d = pd.Timestamp(t).to_pydatetime()# 2020-03-12T00:00:00.000Z
            except Exception as e:
                print( f'*** failed to parse: {t}')
                raise e
            t0 = pd.Timestamp.now(tz='UTC')
            x = t0-d
            x = x.total_seconds()/3600/24
            return x
        
        def _last_actions(tta):
            if tta:
                tta = pd.concat(tta).sort_values('ts',ascending=False).reset_index(drop=True)
                tta['dt'] = tta.ts.apply(_tdiff)
                print('*'*30, 'Latest trades', '*'*30)
                print( tabulate(tta,headers="keys") )
            else:print('*** empty trades')
        last_trade_actions = list(
                map(lambda trs: trs[-1].to_df(), pseudo_df['trade_actions'] )
            )
        _last_actions(last_trade_actions)
        last_buy_actions = map(lambda el: list(filter(lambda e: e.is_buy(), el)), pseudo_df['trade_actions'])
        last_buy_actions = list( last_buy_actions )
        last_buy_actions = list(
                map(lambda trs: trs[-1].to_df(), last_buy_actions )
            )
        _last_actions(last_buy_actions)
        
        pseudo_df.drop(['r1','rr','trade_actions'],axis=1,inplace=True)
        print()
        print('#'*30, 'Metrics', '#'*30)
        print( tabulate(pseudo_df,headers='keys') )
        print('\n[cash_util: effectiveness of cash usage, larger is goal; 0 indicates not used at all.]\n')
        _settings()

        #-- MPT
        if do_mpt:
            from signals.mpt import optimized_mpt
            fig,((ax1,ax2), (ax3,ax4))=plt.subplots(2,2,figsize=(24,24))
            def _mpt(r1, ax1, ax2):
                print()
                o = optimized_mpt(r1,10_000,5./100,do_plot=False)
                wts = np.array( list( # optimized weights
                    map(lambda c: o['allocation_pct'][c], r1.columns)
                ))/100
                levs = list(
                    map(lambda arr: -1./max_drawdowns(arr), [r1[col] for col in r1 ])
                )
                #levs = np.array(list(pseudo_df.lev.values))
                rp = r1.dot(wts)
                try:
                    print(f'-- Optimized portfolio Sharpe: {sharpe(rp):.2f}, Sortino: {sortino(rp):.2f}, Max DD: {max_drawdowns(rp)*100:.1f}%')
                except Exception as e:
                    pass
                
                for col in r1:
                    r1[col].cumsum().plot(ax=ax1,linewidth=1)
                rp.cumsum().plot(ax=ax1,linewidth=5,color='blue',alpha=0.6) 
                ax1.set_ylabel('returns',color='blue')
                
                for i, col in enumerate(r1):
                    (r1[col]*levs[i]).cumsum().plot(ax=ax2,linewidth=1)
                rpl = r1.dot(levs * wts)
                rpl.cumsum().plot(ax=ax2,linewidth=5,color='blue',alpha=0.6) 
                ax2.set_ylabel('returns (lev.)',color='blue')

                ax1.legend(r1.columns)
                ax2.legend(r1.columns)
                ax1.set_title('Returns by wts')
                ax2.set_title('Returns by wts & leverages')
            _mpt(r1, ax1, ax2)
            _mpt(rr, ax3, ax4)
            fn = os.getenv("USER_HOME","")+'/tmp/reversal_hickes_mpt_returns.png'
            plt.savefig(fn)
            print('-- saved:', fn)
        #-- MPT

    else:
       rec = _main(sym,volt,offline)
    
if __name__ == '__main__':
    main()
