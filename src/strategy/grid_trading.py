from time import tzname
import pandas as pd 
import click,os,datetime
import numpy as np 
from tabulate import tabulate

from butil.portfolio_stats import sharpe,sortino,max_drawdowns,annual_returns
from butil.butils import binance_kline
from butil.bmath import gen_random_sets

def _t(b):
        tnow = datetime.datetime.utcnow()
        b = datetime.datetime.strftime( b.to_pydatetime(), '%Y-%m-%d %H:%M:%S' )
        x = tnow-datetime.datetime.strptime( b, '%Y-%m-%d %H:%M:%S' )
        return f'{x.days}d {int(x.seconds/3600)}h {int(x.seconds%3600/60)}m ago'

"""
Toy algorithm (Don't use for real trading!!!)
"""
def _print(s):
    if os.getenv('BINANCE_DEBUG'):
        print(s)

class Metrics:
    def __init__(self, nd,ar,st,sp,dd) -> None:
        self.annual_return = ar
        self.sortino = st
        self.sharpe = sp
        self.max_drawdown = dd
        self.days = nd

    def __str__(self) -> str:
        return f"sortino = {self.sortino:.2f}, sharpe = {self.sharpe:.2f}, max_dd = {self.max_drawdown:.1f}% annual_rtn = {self.annual_return:.1f}% ({self.days} days)"

def griding(df):
    df =  df.copy()           
    # significance
    df['insignificant'] = (df.close-df.open).apply(abs).rolling(60).rank(pct=True)
    df['insignificant'] = df['insignificant']<0.99
    df['insignificant'] = df['insignificant'].shift(1)

    df['prev_up'] = (df['close'] > df['open']).shift(1)
    df['prev_down'] = (df['close'] < df['open']).shift(1)
    
    # filter-out signals
    df = df.dropna()

    # trading signals
    df.loc[(df.prev_up ), 'buy'] =  df.buyp
    df.loc[(df.prev_down), 'sell'] = df.sellp
    df.loc[df.insignificant, 'sell'] = 0
    df.loc[df.insignificant, 'buy'] = 0
    return df

def needles( df ):
    df = df.copy()
    df['volume_nontrivial'] = df.volume.rolling(14).rank(pct=True) > 0.3
    df['oc/hl'] = (df['open']-df['close']).apply(abs)/(df['high']-df['low']) < 0.5 # needle found

    cn = 2 # consecutive buys (or sells)
    df['hc'] = (df['high'] - df['close'])/df['high'] < 5/10000 # high,close are nearyby
    df['ho'] = (df['high'] -  df['open'])/df['high'] < 5/10000   # high,open are nearyby
    df['needle_below'] = 0 
    df.loc[(df.hc | df.ho) & df['oc/hl'], 'needle_below' ] = 1
    df.needle_below = (df.needle_below.rolling(cn).sum()>0).shift(1)
    
    df['lc'] = (df['close'] - df['low'])/df['low'] < 5/10000 # low,close are nearyby
    df['lo'] = (df['open']  - df['low'])/df['low'] < 5/10000   # low,open are nearyby
    df['needle_up'] = 0 
    df.loc[(df.lc | df.lo) & df['oc/hl'], 'needle_up' ] = 1
    df.needle_up = (df.needle_up.rolling(cn).sum()>0).shift(1)

    df = df.dropna()
    df.loc[df.needle_below & df.volume_nontrivial, 'buy'] = df.buyp
    df.loc[df.needle_up & df.volume_nontrivial, 'sell'] = df.sellp

    return df 

def paper_trading(df, max_pos,stop_loss, take_profit, short_allowed=False, do_plot=False):
    df = df.copy()
    df['avg'] = (df.open+df.close+df.high+df.low)/4
    
    # trading price
    df['buyp'] = df.open
    df['sellp'] = df.open
    df['buy'] = 0.;df['sell'] = 0.
    
    #df = griding(df)
    df = needles(df)

    fee = 1e-3

    df['cost'] = 0.;df['asset_n'] = 0.;df['neg_asset_n'] = 0.;df['short_avg_entry']=df['close'];df['profit'] = 0.;df['fee_i'] = 0.
    portfolio = [];short_portfolio=[];profits = [];vol=0.0
    buys = 0;sells=0;tp=0;sl=0;max_cost=0.0;max_down = 0.
    action = []
    wins = 0;losses=0
    for i,row in df.iterrows():
        profit = 0.
        mdd = 0.
        if portfolio:
            avg_cost = np.mean( portfolio)
            mdd = ( row.close - avg_cost )/avg_cost
        if mdd < max_down: 
            max_down = mdd 
        # stop-loss & stop-win
        stoploss = (stop_loss !=0 and mdd < stop_loss)
        takeprofit = (take_profit !=0 and mdd > take_profit)
        if stoploss or takeprofit:
            action += [{'action': 'sl' if stoploss else 'tp','price': row.close, 'ts': i}]
            if stoploss: sl += 1
            elif takeprofit: tp+= 1
            pn = len(portfolio)
            df.loc[i,'fee_i'] = pn*row.close*fee
            vol += pn * row.close
            sells += pn
            profit = np.sum( row.close - np.array(portfolio))
            wins += len(list(filter(lambda v: v<row.close, portfolio)))
            losses += len(list(filter(lambda v: v>row.close, portfolio)))
            _print(f'\t\t\t {"sl" if stoploss else "tp"} {pn} positions, profit = {profit:.6f}, {row.close}, {portfolio}')
            portfolio = []
        else: # not stop loss
            if row.buy > 0:
                pce= row.buy
                if short_portfolio: # [canceling naked shorts]
                    short = short_portfolio.pop()
                    profit = short - pce # sell high buy low, ideally
                    if profit>0: wins+=1
                    else: losses+=1
                    vol += pce 
                    buys+=1
                    action += [{'action': 'long_buy','price': pce, 'ts': i}]
                    df.loc[i,'fee_i'] = pce*fee
                    _print(f"\tclosing short ({short}) {pce} @ {i}")
                elif len(portfolio) < max_pos-1: # [buy]
                    portfolio+= [ pce ]
                    vol += pce
                    buys+=1
                    action += [{'action': 'buy ','price': pce, 'ts': i}]
                    df.loc[i,'fee_i'] = pce*fee
                    _print(f"\t buying {pce} @ {i}")
            elif row.sell > 0:
                sold = row.sell
                if portfolio: # [sell one]
                    pos = portfolio.pop()
                    profit = sold-pos
                    if profit>0: wins+=1
                    else: losses+=1
                    vol += sold
                    sells+=1
                    action += [{'action': 'sell','price': pce, 'ts': i}]
                    df.loc[i,'fee_i'] = pce*fee
                    _print(f"\t\t selling ({pos}) {sold} @ {i}, profit {profit:.6f}")
                elif short_allowed and len(short_portfolio)< max_pos-1: # [naked shorting]
                    #print('shorting...')
                    short_portfolio += [pce]
                    vol += pce
                    sells += 1
                    action += [{'action': 'short_sell ','price': pce, 'ts': i}]
                    df.loc[i,'fee_i'] = pce*fee
                    df.loc[i,'short_avg_entry'] = np.mean( short_portfolio )

        profits += [(i, profit)]
        leverage = 2. # assumed no leverage used
        fund_occupied = np.sum(portfolio)*(1+fee) + ( np.sum(short_portfolio)/leverage + np.sum(short_portfolio)*fee )
        if max_cost < fund_occupied: max_cost = fund_occupied
        df.loc[i,'cost'] = np.sum( portfolio ) + np.sum( short_portfolio )/leverage
        df.loc[i,'asset_n'] = len(portfolio)
        df.loc[i,'neg_asset_n'] = len(short_portfolio)
        df.loc[i,'profit'] = profit

        #print( '###', len(portfolio), len(short_portfolio))
    if (wins+losses) == 0:
        print('-- no trades')
    else:
        print(f'-- wins: {wins}, losses: {losses}, win rate: {(wins/(wins+losses)*100):.1f}%')
    res = 0.
    if portfolio:
        res += len(portfolio)*df.iloc[-1].close - np.sum(portfolio)
    if short_portfolio:
        res += np.sum(short_portfolio) - len(short_portfolio)*df.iloc[-1].close
    
    fee = vol*fee + ( len(portfolio) + len(short_portfolio) )*df.iloc[-1].close*fee # liquidation fee included
    ttl = np.sum(list( map(lambda e: e[1],profits)) )
    net_ttl =  ttl + res - fee

    df['cash'] = max_cost - df['cost'] - df['fee_i'].cumsum() + df['profit'].cumsum()
    df['portfolio'] = df['cash'] + df['asset_n']*df['close'] + ( df['short_avg_entry'] - df['close'] ) * df['neg_asset_n']
    
    df['portfolio'] = df['portfolio'].pct_change().fillna(0)
    df['ref'] = df['close'].pct_change().fillna(0)

    rsharpe = sharpe( df.ref)
    psharpe = sharpe( df.portfolio)
    rsortino = sortino( df.ref)
    psortino = sortino( df.portfolio)
    rmax_drawdowns = max_drawdowns( df.ref)*100
    pmax_drawdowns = max_drawdowns( df.portfolio)*100
    rrtn = annual_returns(df.ref)*100
    prtn = annual_returns(df.portfolio)*100
    pmetrics = Metrics(df.portfolio.resample('1D').agg('sum').shape[0],
                    prtn,psortino,psharpe,pmax_drawdowns)
    rmetrics = Metrics(df.ref.resample('1D').agg('sum').shape[0],
                    rrtn,rsortino,rsharpe,rmax_drawdowns)
    print( 'p:' ,pmetrics, '\nr:', rmetrics)

    """print( '\t', f'{rrtn:.1f}%', f'{prtn:.1f}%', 
    '\t', f'{rsortino:.1f}', f'{psortino:.1f}', 
    '\t', f'{rsharpe:.1f}', f'{psharpe:.1f}', 
    '\t', f'{rmax_drawdowns:.1f}%', f'{pmax_drawdowns:.1f}%')
    """

    df['portfolio'] = df['portfolio'].cumsum()
    df['ref'] = df['ref'].cumsum()
    
    if do_plot:
        import matplotlib.pyplot as plt 
        df[['portfolio','ref']].plot();plt.show()
    
    return df.iloc[-1].close,buys,sells,\
            res,fee,ttl,net_ttl,\
            max_cost,max_down, action[-1] if action else None, sl, tp, \
                pmetrics, rmetrics

def _dt(t1,t2):
    import datetime 
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)
    return (t2-t1).days+1

@click.command()
@click.option('--ric', default="BTC/USDT")
@click.option('--span', default='1h')
@click.option('--test', default=False, is_flag=True)
@click.option('--max_pos', default=50)
@click.option('--nominal', default=1.0, help="scale up or down the trading size for small valued assets")
@click.option('--stop_loss',default=-0.2, help="if average holding cost is too high, sl. (set to 0 to disable)")
@click.option('--take_profit',default=0.05, help="if profit is good, tp. (set to 0 to disable)")
@click.option('--random_sets', default=0)
@click.option('--plot', is_flag=True, default=False)
@click.option('--spans', default='30m,1h,4h,1d')
@click.option('--short_allowed', is_flag=True, default=False)
def main(ric,span,test,max_pos,nominal,stop_loss,take_profit,random_sets,plot,spans,short_allowed):
    ric = ric.upper()
    print('\n--', ric)
    print('-- spans:', spans.split(','))
    print('-- max pos:',max_pos, ' (i.e., sizing)')
    print('-- short ', f'{"enabled" if short_allowed else "disabled"}')
    print('-- stop loss:', f'{(stop_loss*100):.0f}%' if stop_loss!=0 else 'disabled')
    print('-- take profit:', f'{(take_profit*100):.0f}%' if take_profit!=0 else 'disabled')
    recs = []

    def _paper_trading(adf, short_allowed=False, do_plot=False):
        close,buys,sells,res,fee,ttl,net_ttl,max_cost,max_down,action,sl,tp,pmet, rmet = paper_trading(adf,max_pos,stop_loss,take_profit,short_allowed=short_allowed,do_plot=do_plot)
        t1,t2 = adf.iloc[0].timestamp,adf.iloc[-1].timestamp
        is_last_candle = t2 == df.iloc[-1].timestamp
        if action:
            is_now = t2 == action['ts']
            act = f"({action['ts']}, {_t(action['ts'])}) " + action['action']+" "+f"{action['price']:.6f} {'*' if is_last_candle else ''} {'@' if is_now else ''}"
        else:
            act = "no trades"
        return (span,t1,t2,close,buys,sells,res,fee,ttl,net_ttl,max_cost,max_down,_dt(t1,t2),act,sl,tp,
                    pmet.annual_return)
    
    #for span in ['30m','1h','4h','1d']:
    for span in spans.split(','):
        fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
        if not test:
            df = binance_kline(ric, span=span)
            df.to_csv(fn)
            print('-- saved:', fn)
        else:
            #print('-- reading:',fn)
            df = pd.read_csv( fn )
        
        df['timestamp'] = df['timestamp'].apply(pd.Timestamp)
        df.set_index('timestamp', inplace=True)
        df['timestamp'] = df.index
        # subsets
        print( span, df.shape[0] )
        for rs in gen_random_sets(0,df.shape[0],df.shape[0]//5, random_sets) if random_sets>0 else []:
            istart = rs[0]; iend=rs[-1]
            idf = df.copy()[istart:iend]
            rec = _paper_trading( idf,short_allowed=short_allowed)
            recs += [ rec ]
        rec =  _paper_trading(df,short_allowed=short_allowed, do_plot=plot)
        recs += [ rec ]
        
    df = pd.DataFrame.from_records( recs )
    df.columns='span,t1,t2,close,buys,sells,asset,fee,cash_gain,net_ttl,max_cost,max_down,days,last_action,sl,tp,cagr%'.split(',')
    df['daily_net_ttl'] = df.net_ttl/df.days
    df['fee%'] = df.fee/(df.fee+df.net_ttl)*100
    df['net_ttl%'] = df.net_ttl/df.max_cost*100 / (df.days/365)
    for col in 'asset,fee,cash_gain,net_ttl,max_cost,daily_net_ttl'.split(','):
        df[col] = df[col].apply(lambda v: v*nominal)
    for col in 'asset,fee,cash_gain,net_ttl,max_cost,daily_net_ttl'.split(','):
        df[col] = df[col].apply(lambda s: f"{s:,.0f}")
    for col in 'net_ttl%,fee%,cagr%'.split(','):
        df[col] = df[col].apply(lambda s: f"{s:.1f}%")
    for col in 'max_down'.split(','):
        df[col] = df[col].apply(lambda s: f"{(s*100):.0f}% {(np.floor(-1/s) if s!=0 else 0):.0f}")
    
    df['age (last candle)'] = df['t2'].apply(lambda t: _t(t) )

    print(tabulate(df['span,t2,close,buys,sells,fee,sl,tp,asset,cash_gain,net_ttl,max_cost,max_down,cagr%,days,daily_net_ttl'.split(',')].sort_values('t2',ascending=True), headers="keys"))
    print(tabulate(df['t2,close,net_ttl,max_cost,max_down,days,span,age (last candle),last_action'.split(',')].sort_values('t2',ascending=True), headers="keys"))
if __name__ == '__main__':
    main()
