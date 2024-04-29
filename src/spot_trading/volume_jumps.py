import pandas as pd 
import datetime,click,os,talib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def ref_seconds( span):
    span = span.lower()
    if span.endswith('h'): return int(span.split('h')[0])*3600
    elif span.endswith('d'): return int(span.split('d')[0])*24*3600
    elif span.endswith('m'): return int(span.split('m')[0])*60

user_home = os.getenv('USER_HOME','')
def cached_ohlcv(ric,span):
    rs = ref_seconds(span)
    fn =user_home+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    print('-- ',fn)
    df = ohlcv = pd.read_csv( fn )
    for col in 'open,high,low,close,volume'.split(','):
        df[col] = df[col].apply(float)
    df.timestamp = df.timestamp.apply(pd.Timestamp)
    df['index'] = df.timestamp
    tnow = datetime.datetime.utcnow()
    tnow = datetime.datetime.strftime(tnow,'%Y-%m-%dT%H:%M:%SZ')
    tnow = pd.Timestamp(tnow)
    dt = tnow - df.timestamp.iloc[-1]
    print('  -- tnow:',tnow, f', last bar: {(dt.seconds/rs*100):.1f}% (of {span})' )
    df.set_index('index', inplace=True, drop=True)

    return df
@click.command()
@click.option('--ric',default='DOGE/USDT')
@click.option('--span',default="1h")
def main(ric,span):
    df = cached_ohlcv(ric,span)
    df['rtn'] = df.close.pct_change()
    DAYS=365
    df['atr'] = talib.ATR(df.high,df.low,df.close,timeperiod=14)
    df['atr_rank'] = df.atr.rolling(DAYS).rank(pct=True)
    df['vol_rank'] = df.volume.rolling(DAYS).rank(pct=True)
    
    vdf = df[['high','low','close','rtn','volume','atr_rank','vol_rank']].dropna().tail(200)
    print(vdf)

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(18,14))
    ax11 = ax1.twinx() 
    ax22 = ax2.twinx() 

    #vdf.vol_rank.plot(ax=ax1,color='blue')
    ax1.step(vdf.index, vdf.vol_rank.values)
    #vdf[['high','low']].plot(ax=ax11,color='gold')
    ax11.plot(vdf.index, vdf.high.values,color='gold' )
    ax11.plot(vdf.index, vdf.low.values,color='gold' )

    ax2.step(vdf.index, vdf.vol_rank.values)
    ax22.plot(vdf.index, vdf.rtn.values,color='red')

    for tic in ax1.get_xticklabels(): tic.set_rotation(35)
    for tic in ax2.get_xticklabels(): tic.set_rotation(35)

    gf = f'{user_home}/tmp/volume_jumps_{ric.lower().replace("/","-")}_{span}.png'
    plt.savefig(gf)
    print('-- saved:',gf)

if __name__ == '__main__':
    main()