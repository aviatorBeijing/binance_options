import multiprocessing
from time import tzname
import pandas as pd 
import click,os,datetime
import numpy as np 
from tabulate import tabulate

from butil.butils import binance_kline

def _t(b):
        tnow = datetime.datetime.utcnow()
        b = datetime.datetime.strftime( b.to_pydatetime(), '%Y-%m-%d %H:%M:%S' )
        x = tnow-datetime.datetime.strptime( b, '%Y-%m-%d %H:%M:%S' )
        return f'{x.days}d {int(x.seconds/3600)}h {int(x.seconds%3600/60)}m ago'

def _dt(t1,t2):
    import datetime 
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)
    return (t2-t1).days+1

@click.command()
@click.option('--ric', default="BTC/USDT")
@click.option('--dollar', is_flag=True, default=False, help="if presented, convert the volume to $volume, default volume is in BTC counts")
@click.option('--test', default=False, is_flag=True)
def main(ric, dollar, test):
    ric = ric.upper()
    print('\n--', ric)
    
    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_1h.csv'
    if not test:
        df = binance_kline(ric, span='1h')
        df.to_csv(fn)
        print('-- saved:', fn)
    else:
        df = pd.read_csv( fn )
        
    df['timestamp'] = df['timestamp'].apply(pd.Timestamp)
    df.set_index('timestamp', inplace=True)
    df['timestamp'] = df.index
        
    print( df.shape[0] )

    df['utc_hour'] = df.timestamp.apply(str).apply(lambda s: s.split(' ')[1])
    
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].apply(float)
    
    colume = 'volume'
    if dollar:
        colume = '$volume'
        df['$volume'] = df.volume * (df.open+df.close+df.high+df.low)/4

    hours = sorted(list(set(df['utc_hour'].values)))
    recs = []
    for hour in hours:
        rec = df[df.utc_hour == hour][colume].describe()
        recs += [rec]
    vdf = pd.DataFrame.from_records( recs )
    vdf.columns = ['num','mean','std','min','25%','50%','75%','max']
    vdf['utc_hours'] = hours

    current = df.iloc[-1]
    vdf['current'] = ''
    
    rk = df[colume].rolling(df.shape[0]).rank(pct=True).iloc[-1]*100 # the rank of percentile of the last row volume
    latest_ts = df.timestamp.iloc[-1]

    vdf.loc[vdf.utc_hours == current.utc_hour, 'current'] = f'{rk:.1f}% ({latest_ts})'
    for col in ['mean','std','min','25%','50%','75%','max']:
        vdf[col] = vdf[col].apply(lambda v: f"{v:,.0f}")
    print( vdf )
    print('')
    print( '-- sort by 50%\n',vdf.sort_values('75%', ascending=False).head(5))
    print( '-- sort by 75%\n',vdf.sort_values('75%', ascending=False).head(5))

if __name__ == '__main__':
    main()
