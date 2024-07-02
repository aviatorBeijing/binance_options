import os,click
import pandas as pd
import numpy as np
import talib
from tabulate import tabulate

from butil.portfolio_stats import calc_cagr, max_drawdowns,sharpe,sortino
from signals.meta import ActionT,TradeAction,VolumeHikesEmitter
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

def pseudo_trade(sym, df, volt=68, new_stuct=False, ax=None):
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
    emitter = VolumeHikesEmitter(init_cap, volt)
    for i, row in df.iterrows():
        #print(i, row.dd, row.closes, row.volrank, row.sig )
        is_breaking_down = row['1sigma_dw_sig_flag']
        is_breaking_up   = row['1sigma_up_sig_flag']

        ####### Buys #######
        if (row.sig>0 and not is_breaking_down):
            pce = row.sig;ts = i
            sz_f = 0
            if cash_utility_factor>0:
                sz_f = cash_utility_factor # Use the fixed factor
            else:
                sz_f = row.volrank # Using the volume rank as the factor
            sz = cash * sz_f / pce 

            if sz*pce > init_cap/100: # enough cash FIXME
                buys += [ (ts, pce, sz, )]
                cash -= sz*pce*(1+ff)
                pos += sz
                fees += sz*pce*ff
                nbuys += 1
                if cash < cash_min: cash_min = cash # record the cash usage
                if sz*pce > max_cost: max_cost = sz*pce
                #print(f'  [buy] {sym},', _cl(str(i)), f'${pce}, sz: {sz}, cap%: { (row.volrank*100):.1f}%')

                trade_actions+=[ TradeAction(emitter, sym, ActionT.BUY, pce, sz, sz_f, ts) ]
            else:
                #print(f'* insufficient fund: {sz*pce} < {init_cap/100}, {sz}, {pce}')
                pass
        else:
            pce = row.closes;ts=i 

            ######## Sells ########
            if order_by_order:
                if buys:
                    # tp
                    _buys = list(map(lambda e: e[1], buys))
                    _ix = np.argmin( _buys )
                    ts0, last_buy, last_buy_sz = buys[_ix]
                    if (pce-last_buy)/last_buy > (profit_margin): # met the profit traget
                        buys = buys[:_ix] + (buys[_ix+1:] if (_ix+1)<len(buys) else [])
                        #print( '    [tp]:', _cl(str(i)), f'${pce}', ', the buy:', _cl(ts0), f'${last_buy}', ',holding:', (i-ts0).total_seconds()/3600/24, 'days') #, (pce-last_buy)/last_buy,'>', profit_margin)
                        cash += pce*last_buy_sz*(1-ff)
                        pos -= last_buy_sz
                        fees += last_buy_sz * pce * ff
                        nsells += 1
                        trade_actions+=[ TradeAction(emitter, sym, ActionT.TP, pce, last_buy_sz,1., str(i)) ]
                        wins += 1
                # sl
                if buys: # Stop-loss: Only SL the buy trade that potentially lost the MOST.
                        _buys = list(map(lambda e: e[1], buys))
                        _ix = np.argmax( _buys )
                        ts0, last_buy, last_buy_sz = buys[_ix]
                        if (pce-last_buy)/last_buy < -sl or is_breaking_down: # sl
                            buys = buys[:_ix] + (buys[_ix+1:] if (_ix+1)<len(buys) else [])
                            #print( '      [sl]:', _cl(str(i)), f'${pce}', ', the buy:', _cl(ts0), f'${last_buy}', len(buys), _ix ) #, (pce-last_buy)/last_buy,'<', -sl)
                            cash += pce*last_buy_sz*(1-ff)
                            pos -= last_buy_sz
                            fees += last_buy_sz * pce * ff
                            nsells += 1
                            stoplosses += 1
                            trade_actions+=[ TradeAction(emitter, sym, ActionT.SL, pce, last_buy_sz,1., str(i)) ]

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
                        trade_actions+=[ TradeAction(emitter, sym, ActionT.SELL, pce, sz,1., str(i)) ]
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
    rinc = (pdf['v'].iloc[-1] - pdf['v'].iloc[0])/pdf['v'].iloc[0] /(
        (df['closes'].iloc[-1] - df['closes'].iloc[0])/df['closes'].iloc[0]
    )
    
    max_dd = max_drawdowns( r1 )
    max_lev = 1./(-max_dd)
    #r1 *= max_lev 

    if not ax:
        fig, ax1 = plt.subplots(1,1,figsize=(24,8))
    else:
        ax1 = ax
    
    ax11 = ax1.twinx()
    (((1+r1).cumprod()-1)*100).plot( ax=ax1 )
    pdf['assets'].plot(ax=ax11,color='gray',alpha=0.5)
    ax1.set_ylabel('(Pseudo Trade) Return%',color='blue')
    ax11.set_ylabel('Position (#)',color='gray')
    ax11.grid(False)
    
    if not ax:
        fn = os.getenv("USER_HOME","")+f"/tmp/port_{sym}.png"
        plt.savefig( fn )
        print('-- saved:', fn)

    val = cash+pos*float(df.iloc[-1].closes)-fees
    rtn = (val/init_cap-1)*100 #total return
    
    profits = val - init_cap
    annual = cagr = calc_cagr( r1 )*100
    bh_annual = calc_cagr(df.rtn)*100
    
    ax1.set_title( f'{sym.upper()}, cagr={cagr:,.1f}%, sell in {trading_horizon} days after each buy, max lev: {max_lev:.1f}, r1/rr: {(rinc):.1f} (w/ lev:{(rinc*max_lev):.1f})' )
    
    max_dd_ref = max_drawdowns( rr )
    sot = sortino1 = sortino(r1)
    bh_sot = sortino_ref = sortino(rr)
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

    fn = os.getenv("USER_HOME","")+"/data/strategies/volume_hikes"
    if not os.path.exists(fn): os.makedirs( fn )
    sym = sym.lower()
    if sym.lower() in ['btc','sol','doge','eth','xrp','pendle']:
        xsym = f'{sym.lower()}usdt' if not 'usdt' in sym else sym
        fn += f'/portfolio_rtn_matrics_{xsym}.csv'
    else:
        fn += f'/portfolio_rtn_matrics_{sym}.csv'
    print(df.closes.shape, len(portfolio), len(assets))
    rtns_df = pd.DataFrame.from_dict({
        'Date': pdf.index,
        'rtn': ((1+r1).cumprod()-1)*100,
        'rtn_dollar': portfolio,
        'position': assets,
        'ref_rtn':(((1+rr).cumprod()-1)*100).values,
        'close': list(df.closes)
    }).fillna(0).reset_index(drop=True)
    rtns_df.to_csv(fn, index=0)
    print('-- saved:', fn)

    yrs = round(df.shape[0]/365., 1)
    
    if new_stuct: # New datastucture
        from signals.meta import construct_lastest_signal
        rec = construct_lastest_signal(
            sym.upper(),
            df.index[-1],
            yrs,   
            r1.max()*100,
            r1.min()*100,
            annual,
            bh_annual,
            sot,
            bh_sot,
            max_dd,
            max_dd_ref,
            trade_actions[-1],
            df.iloc[-1].closes
        )
        return rec 
    
    return sym, yrs, \
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

def find_reversals(sym, ts, closes,volume,volt=50,rsi=pd.DataFrame(),file_ts:str='', new_struct=False):
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

    is_not_volatile = df.rtn <= .25
    is_jump = (df.volrank.shift(2) <= volt/100) \
                & (df.volrank.shift(1) <= volt/100) \
                & (df.volrank>=volt/100) 

    #------------------------- Signals ---------------------
    buy_signals = rank_xing = (df.rsi<20 if 'rsi' in df else False ) | \
                              (   is_not_volatile \
                                & is_jump
                              )

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

    pseudo_metrics = rec = pseudo_trade(sym, df, volt=volt, new_stuct=new_struct, ax=ax3) # a different trading strategy!
    if new_struct:
        return rec 

    df['bought'] = df.sig.shift(trade_horizon) # after one month (trading strategy)
    xdf = df[df.bought>0][['bought','closes']].copy()
    xdf['price_delta'] = df.closes-df.bought
    xdf['return'] = xdf.price_delta/xdf.bought
    
    annual = calc_cagr( xdf['return'])*100
    ds = df.shape[0]

    latest = df[df.sig>0].tail(1)
    
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()

    rx = df['closes'].pct_change()
    (df['dd']*100).plot(ax=ax1,color='red')
    df['volrank'].plot(ax=ax11,alpha=0.5)
    df['closes'].plot(ax=ax2,linewidth=1.5)
    (((1+rx).cumprod()-1)*100).plot(ax=ax22,linewidth=1.5,linestyle='none')
    ax1.set_ylabel('drawdown%', color='red')
    ax11.set_ylabel('volrank', color='blue')

    df['1sigma_up'].plot(ax=ax2,linewidth=1,color='gray',alpha=0.3)
    df['1sigma_dw'].plot(ax=ax2,linewidth=1,color='gray',alpha=0.3)

    df['1sigma_up_sig'].plot(ax=ax2,marker="x",linestyle="none",color="red", alpha=0.6)
    df['1sigma_dw_sig'].plot(ax=ax2,marker="x",linestyle="none",color="black", alpha=0.6)
    
    df['sig'].plot(ax=ax2,marker="^",linestyle="none",color="red", alpha=0.6)
    df['r1'].plot(ax=ax11,marker="^",linestyle="none",color="red", alpha=0.6)
    
    ax1.set_title(f'equity drawdown v.s. volume ranking (data: {file_ts})')
    ax2.set_title('price & buying signals')
    ax2.set_ylabel('price ($)')
    ax22.set_ylabel('return (%)')
    ax22.grid(False)
    fn = os.getenv("USER_HOME",'')+f'/tmp/reversal_{sym}.pdf' # Trading signals
    plt.savefig(fn)
    print('-- saved:',fn)

    return {
        'crypto': sym.upper(),
        'start': df.index[0],
        'end': df.index[-1],
        #'wins': xdf[xdf["price_delta"]>0].shape[0],
        #'losses': xdf[xdf["price_delta"]<0].shape[0],
        'single_max_gain_pct': xdf["return"].max()*100,
        'single_max_loss_pct': xdf["return"].min()*100,
        'cagr_pct': annual,
        'days': ds,
        'last_buy': f'{latest.index[0]},{latest.closes.iloc[0]}' if not latest.empty else "",
        'price': df.iloc[-1].closes,
        #'last_sell': f'{lastsell.index[0]},{lastsell.closes.iloc[0]},{lastsell.bought.iloc[0]}' if not lastsell.empty else "",
        'pseudo_trade': pseudo_metrics,
    }

def _file_ts(fn):
    import datetime
    s = os.stat(fn)
    t = int( s.st_mtime )
    d = datetime.datetime.fromtimestamp(int(s.st_mtime) )
    return str(d)

def _main(sym, volt,offline=False, new_struct=False):
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
    rec = find_reversals(sym, ts, closes, volume, 
                volt=volt,rsi=rsi, file_ts=file_ts if offline else 'realtime',
                new_struct=new_struct )
    return rec

@click.command()
@click.option("--sym", default='doge')
@click.option("--syms", default='')
@click.option("--volt", default=50, help="percentage of volume considered to be significant")
@click.option("--offline", is_flag=True, default=False)
@click.option("--do_mpt", is_flag=True,default=False, help='turn on the MPT simulation')
@click.option("--new_struct", is_flag=True, default=False, help='using new data structure as in the climb_and_fall algo.')
def main(sym,syms,volt,offline,do_mpt, new_struct):
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
                recs = pool.map( functools.partial(_main, volt=volt,offline=offline,new_struct=new_struct), syms.split(","))
                pool.close();pool.join()
        else:
            for sym in syms.split(','):
                rec = _main(sym, volt=volt, offline=offline, new_struct=new_struct )
                recs += [rec]
        df = pd.DataFrame.from_records( recs )
        if new_struct:
            df['x'] = df.last_action.apply(lambda s: s.split(',')[1])
            df.sort_values('x', ascending=False, inplace=True)
            df.drop('x', inplace=True, axis=1)
            print(df)
            return 

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
        med_cagr = pseudo_df.cagr.median()
        avg_cagr = pseudo_df.cagr.mean()

        print(f'    -- CAGR median: {(med_cagr):,.1f}%')
        print(f'    -- CAGR mean:   {(avg_cagr):,.1f}%')

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
        
        trade_actions = pseudo_df['trade_actions']

        def _last_actions(tta):
            if tta:
                tta = pd.concat(tta).sort_values('ts',ascending=False).reset_index(drop=True)
                tta['dt'] = tta.ts.apply(_tdiff)
                print('*'*30, 'Latest trades', '*'*30)
                print( tabulate(tta,headers="keys") )
                return tta 
            else:
                print('*** empty trades')
                return pd.DataFrame()
        # Buys & Sells
        last_trade_actions = list(
                map(lambda trs: trs[-1].to_df(), trade_actions )
            )
        last_trades_all =  _last_actions(last_trade_actions)

        # Buys Only
        last_buy_actions = map(lambda el: list(filter(lambda e: e.is_buy(), el)), trade_actions)
        last_buy_actions = list( last_buy_actions )
        last_buy_actions = list(
                map(lambda trs: trs[-1].to_df(), last_buy_actions )
            )
        last_trades_buyonly = _last_actions(last_buy_actions)
        
        pseudo_df.drop(['r1','rr','trade_actions'],axis=1,inplace=True)
        print()
        print(' '*60,'#'*30, 'Metrics', '#'*30)
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

                from signals.cvar import historicalVaR, historicalCVaR, var_parametric, cvar_parametric, portfolioPerformance,getData
                pt=99
                tm=30 # days
                initCap = 10_000
                
                hvar = -historicalVaR(rp, alpha=100-pt)*np.sqrt(tm)
                hcvar = -historicalCVaR(rp, alpha=100-pt)*np.sqrt(tm)

                pRet, pStd = portfolioPerformance(wts, r1.mean(), r1.cov(), tm)
                mdl_var = var_parametric(  pRet, pStd, distribution='t-distribution', alpha=100-pt)
                mdl_cvar = cvar_parametric(pRet, pStd, distribution='t-distribution', alpha=100-pt)
                print(f'-- VaR, CVaR (CI {(pt):.0f}%, in future {tm} days, ${initCap:,.2f} initial cash.)')
                print(f'   historical            :   $ {(hvar*initCap):,.2f}, $ {(hcvar*initCap):,.2f}')
                print(f'   model (Student-t)     :   $ {(mdl_var*initCap):,.2f}, $ {(mdl_cvar*initCap):,.2f}')

                try:
                    print(f'-- Optimized portfolio Sharpe: {sharpe(rp):.2f}, Sortino: {sortino(rp):.2f}, Max DD: {max_drawdowns(rp)*100:.1f}%')
                except Exception as e:
                    pass
                
                for col in r1:
                    ((1+r1[col]).cumprod()-1).plot(ax=ax1,linewidth=1)
                ((1+rp).cumprod()).plot(ax=ax1,linewidth=5,color='blue',alpha=0.6) 
                ax1.set_ylabel('returns',color='blue')
                
                for i, col in enumerate(r1):
                    y  = r1[col]*levs[i]
                    ((1+y).cumprod()-1).plot(ax=ax2,linewidth=1)
                rpl = r1.dot(levs * wts)
                ((1+rpl).cumprod()-1).plot(ax=ax2,linewidth=5,color='blue',alpha=0.6) 
                ax2.set_ylabel('returns (lev.)',color='blue')

                ax1.legend(r1.columns)
                ax2.legend(r1.columns)
                ax1.set_title('Returns by wts')
                ax2.set_title('Returns by wts & leverages')
            print('\n[strategy]')
            _mpt(r1, ax1, ax2)
            print('\n[buy&hold]')
            _mpt(rr, ax3, ax4)
            fn = os.getenv("USER_HOME","")+'/tmp/reversal_hickes_mpt_returns.png'
            plt.savefig(fn)
            print('-- saved:', fn)
        #-- MPT

    else:
       rec = _main(sym,volt,offline)
    
if __name__ == '__main__':
    main()
