import os,datetime,click,pandas as pd
from tabulate import tabulate

from butil.butils import binance_kline, _file_ts
from butil.yahoo_api import AssetClass, get_asset_class,get_data

def cached(ric,span):
    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    if os.path.exists(fn):
        df = pd.read_csv(fn)
        return df 
    else:
        print(f'*** {fn} not found.')
        return pd.DataFrame()

def _main(ric,span):
    ric = ric.upper()
    fn =os.getenv('USER_HOME','')+f'/tmp/{ric.lower().replace("/","-")}_{span}.csv'
    df = binance_kline(ric, span=span, grps=5 if span=='1d'  else 50)

    print(tabulate(df.tail(5),headers="keys"))
    rk_last = df.volume.rolling(100).rank(pct=True).iloc[-5:].values
    print(f'-- latest volumes: {(rk_last[0]*100):.1f}%, {(rk_last[1]*100):.1f}%, {(rk_last[2]*100):.1f}%, {(rk_last[3]*100):.1f}%, {(rk_last[4]*100):.1f}%')

    df.to_csv(fn)
    print('-- saved:', fn)

def get_filebased_kline(sym, offline=True):
    sym = sym.lower()
    if any([ sym in s for s in ['btc','doge','sol','bnb']]):
        sym = sym.replace('/','-').replace('-usdt','').replace('-usd','')
        fn = os.getenv("USER_HOME","") + f'/tmp/{sym}-usdt_1d.csv'
    else:
        fn = os.getenv("USER_HOME","") + f'/tmp/{sym}_1d.csv'
    
    if os.path.exists(fn):
        file_ts = _file_ts( fn )
    if not offline or not os.path.exists(fn):
        if get_asset_class(sym) == AssetClass.CRYPTO:
            df = binance_kline(f'{sym.upper()}/USDT', span='1d', grps=20)
            ds = df.iloc[-1].timestamp
            file_ts = datetime.datetime.strptime(ds, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            df = get_data(f'{sym.upper()}', '1d', 365*10, realtime=not offline)
            fn = os.getenv("USER_HOME","") + f'/tmp/{sym.lower().replace("/","-")}_1d.csv'
            df.to_csv(fn,index=0)
            print('-- market data saved:', fn)
            df.columns = [s.lower() for s in df.columns ]
            df.timestamp = df.timestamp.apply(datetime.datetime.fromtimestamp).apply(pd.Timestamp)
            df.timestamp = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
            try:
                file_ts = datetime.datetime.fromtimestamp(int(df.iloc[-1].timestamp) )
            except Exception as e:
                file_ts = df.iloc[-1].timestamp
                if not isinstance( file_ts, datetime.datetime): raise e
                #datetime.datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")
    else:
        df = pd.read_csv( fn, index_col=0 )
    return df, file_ts

@click.command()
@click.option('--ric', default='DOGE/USDT')
@click.option('--rics',default='')
@click.option('--span', default='5m')
def main(ric,rics,span):
    if rics:
        rics = rics.split(',')
    else:
        assert len(ric)>0, 'Empty ric or rics'
        rics = [ric]

    rics = list(map(lambda ric: ric.upper(),rics ))
    
    if len(rics)>1:
        from multiprocessing import Pool
        from functools import partial
        with Pool(5) as pool:
            pool.map( partial(_main, span=span), rics )
            pool.close();pool.join()
    else:
        _main(rics[0], span )


if __name__ == '__main__':
    main()
