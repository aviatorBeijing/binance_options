from time import tzname
import pandas as pd 
import click,os,datetime
import numpy as np 
from tabulate import tabulate

from butil.portfolio_stats import sharpe,sortino,max_drawdowns,annual_returns
from butil.butils import binance_kline
from butil.bmath import gen_random_sets

# Toy strategy

class Metrics:
    def __init__(self, nd,ar,st,sp,dd) -> None:
        self.annual_return = ar
        self.sortino = st
        self.sharpe = sp
        self.max_drawdown = dd
        self.days = nd

    def __str__(self) -> str:
        return f"sortino = {self.sortino:.2f}, max_dd = {self.max_drawdown:.1f}% annual_rtn = {self.annual_return:.1f}% ({self.days} days)"

def  _max_down( portfolio, close ):
    if portfolio:
        avg_cost = np.mean( portfolio)
        return ( close - avg_cost )/avg_cost
    return 0

def paper_trading(df, max_pos,stop_loss, do_plot=False):
    df = df.copy()
    df['avg'] = (df.open+df.close+df.high+df.low)/4
    
    # trading price
    df['buy'] = df.avg
    df['sell'] = df.avg
    
    # significance
    df['insignificant'] = ((df.close-df.open).apply(abs).rolling(21).rank(pct=True)<0.95).shift(1)
    #df['insignificant'] =( ( (df.close-df.open)/(df.high-df.low)).apply(abs) < 0.6 ).shift(1)

    df['prev_up'] = (df['close'] > df['open']).shift(1)
    df['prev_down'] = (df['close'] < df['open']).shift(1)
    
    # Pseudo-trading
    df = df.dropna()
    df.loc[df.insignificant, 'sell'] = 0
    df.loc[df.insignificant, 'buy'] = 0
    df.loc[df.prev_up, 'sell'] = 0 # buy
    df.loc[df.prev_down, 'buy'] = 0 # sell
    #print(df)

    fee = 1e-3

    df['cost'] = 0.;df['asset_n'] = 0.;df['profit'] = 0.;df['fee_i'] = 0.
    pos = 0;portfolio = []; profits = [];vol=0.0
    buys = 0;sells=0;sl=0;max_cost=0.0;max_down = 0.0
    action = []
    for i,row in df.iterrows():
        profit = 0.
        mdd = _max_down( portfolio, row.close )
        if mdd < max_down: max_down = mdd 
        
        # stop-loss & stop-win
        if (stop_loss !=0 and mdd < stop_loss):# or mdd > 150/100:
            action += [{'action': 'sell_all','price': row.close, 'ts': i}]
            sold = portfolio.pop()
            sl += 1
            df.loc[i,'fee_i'] = len(portfolio)*row.close*fee
            while sold:
                pos -=1
                profit = sold - row.close
                vol += row.close
                sells+=1
                if portfolio: sold = portfolio.pop()
                else: break
        else: # not stop 
            if row.buy > 0 and pos < max_pos:
                pce= row.buy
                portfolio+= [ pce ]
                vol += pce
                pos +=1
                buys+=1
                action += [{'action': 'buy ','price': pce, 'ts': i}]
                df.loc[i,'fee_i'] = pce*fee
            elif row.sell > 0 and portfolio:
                pce = row.sell
                sold = portfolio.pop()
                pos -=1
                profit = sold - pce
                vol += pce
                sells+=1
                action += [{'action': 'sell','price': pce, 'ts': i}]
                df.loc[i,'fee_i'] = pce*fee
        profits += [(i, profit)]
        fund_occupied = np.sum(portfolio)*(1+fee)
        if max_cost < fund_occupied: max_cost = fund_occupied
        df.loc[i,'cost'] = np.sum( portfolio )
        df.loc[i,'asset_n'] = pos
        df.loc[i,'profit'] = profit
    
    res = 0.
    if portfolio:
        res = len(portfolio)*df.iloc[-1].close - np.sum(portfolio) 

    fee = vol*fee + len(portfolio)*df.iloc[-1].close*fee # liquidation fee included
    ttl = np.sum(list( map(lambda e: e[1],profits)) )
    net_ttl =  ttl + res - fee

    df['cash'] = max_cost - df['cost'] - df['fee_i'].cumsum() + df['profit'].cumsum()
    df['portfolio'] = df['cash'] + df['asset_n']*df['close']
    
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
            max_cost,max_down, action[-1], sl, \
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
@click.option('--stop_loss',default=-0.2, help="if average holding cost is too high, liquidate")
@click.option('--random_sets', is_flag=True, default=False)
@click.option('--plot', is_flag=True, default=False)
@click.option('--spans', default='30m,1h,4h,1d')
def main(ric,span,test,max_pos,nominal,stop_loss,random_sets,plot,spans):
    ric = ric.upper()
    print('\n--', ric)
    print('-- max pos:',max_pos, ' (i.e., sizing)')
    print('-- stop loss:', f'{(stop_loss*100):.0f}%' if stop_loss!=0 else 'disabled')
    recs = []

    def _paper_trading(adf, recs, do_plot=False):
        close,buys,sells,res,fee,ttl,net_ttl,max_cost,max_down,action,sl, pmet, rmet = paper_trading(adf,max_pos,stop_loss,do_plot=do_plot)
        t1,t2 = adf.iloc[0].timestamp,adf.iloc[-1].timestamp
        is_last_candle = t2 == df.iloc[-1].timestamp
        is_now = t2 == action['ts']
        act = f"({action['ts']}) " + action['action']+" "+f"{action['price']:.6f} {'*' if is_last_candle else ''} {'@' if is_now else ''}"
        recs += [(span,t1,t2,close,buys,sells,res,fee,ttl,net_ttl,max_cost,max_down,_dt(t1,t2),act,sl,
                    pmet.annual_return)]
    
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
        print( span )
        for rs in gen_random_sets(0,df.shape[0],df.shape[0]//5, 5) if random_sets else []:
            istart = rs[0]; iend=rs[-1]
            idf = df.copy()[istart:iend]
            _paper_trading( idf,recs)
        _paper_trading(df,recs, plot)
        
    df = pd.DataFrame.from_records( recs )
    df.columns='span,t1,t2,close,buys,sells,asset,fee,cash_gain,net_ttl,max_cost,max_down,days,last_action,stop_loss,cagr%'.split(',')
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
        df[col] = df[col].apply(lambda s: f"{(s*100):.0f}% {(-1/s):.1f}")
    
    tnow = datetime.datetime.utcnow()
    def _t(b):
        b = datetime.datetime.strftime( b.to_pydatetime(), '%Y-%m-%d %H:%M:%S' )
        x = tnow-datetime.datetime.strptime( b, '%Y-%m-%d %H:%M:%S' )
        return f'{x.days}d {int(x.seconds/3600)}h {int(x.seconds%3600/60)}m ago'
    df['age (last candle)'] = df['t2'].apply(lambda t: _t(t) )

    print(tabulate(df['span,t1,t2,close,buys,sells,fee,stop_loss,asset,cash_gain,net_ttl,max_cost,max_down,cagr%,days,daily_net_ttl'.split(',')], headers="keys"))
    print(tabulate(df['t1,t2,close,net_ttl,max_cost,max_down,days,span,age (last candle),last_action'.split(',')], headers="keys"))
if __name__ == '__main__':
    main()