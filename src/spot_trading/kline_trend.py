import os,datetime,click 
import pandas as pd
import talib 
from statsmodels.tsa.seasonal import STL

from butil.butils import binance_kline
from spot_trading.market_data import cached as kcached

def _plot(ric,df,period):
    df = df.copy()
    df.timestamp =  df.timestamp.apply(pd.Timestamp)
    df.set_index('timestamp',inplace=True)
    import matplotlib.pyplot as plt 
    plt.style.use('fivethirtyeight')

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(24,12), sharex=True)
    ax22=ax2.twinx()
    df[['trend','close']].plot(ax=ax1)
    df.season.plot(ax=ax2,linewidth=2,color='red')
    df.resid.plot(ax=ax22,linewidth=1,color="green",alpha=0.7)
    ax1.set_title(f"period={period}")
    ax1.set_ylabel('Decomposed & Price')
    ax2.set_ylabel('Seasonal',color='red')
    ax22.set_ylabel('Residual',color='green')

    fn = os.getenv("USER_HOME",'') + f"/tmp/kline_trend_{ric.replace('/','-').lower()}.png"
    plt.savefig(fn)
    print('-- saved:', fn)

def _main(ric, df,period=7):
    print('-- [SEASONAL] period:', period)
    df = df.copy()
    mdl = STL(df.close,period=period,robust=False).fit()
    df['trend'] = mdl.trend 
    df['season'] = mdl.seasonal
    df['resid'] = mdl.resid 

    _plot(ric,df,period)
    return df 

@click.command()
@click.option('--ric', default='BTC/USDT')
@click.option('--span', default='5m')
@click.option('--period', default=7, help="period of STL")
@click.option('--use_cached', default=False, is_flag=True)
def main(ric,span,period,use_cached):
    print('-'*12)
    print(f'| {ric} |')
    print('-'*12)
    ric = ric.upper()
    if not use_cached:
        df = binance_kline( symbol=ric, span=span )
    else:
        df = kcached(ric,span)
    df.columns = [s.lower() for s in df.columns]
    for col in ['open','high','low','close','volume']: df[col] = df[col].apply(float)
    
    df = _main(ric, df, period=period)

if __name__ == '__main__':
    main()