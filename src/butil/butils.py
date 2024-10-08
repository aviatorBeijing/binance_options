import os,datetime
import ccxt
import pandas as pd

DEBUG = os.getenv("expo", None)

DATADIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/options'
if not os.path.exists( DATADIR):
    try:
        os.makedirs( DATADIR )
    except Exception as e:
        print('*** Make sure set the "USER_HOME" directory for temporary data storage!')

FUNDINGDIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/funding'
if not os.path.exists( FUNDINGDIR): os.makedirs( FUNDINGDIR )
INDICESDIR=os.getenv('USER_HOME','/home/ubuntu')+'/data/binance/indices'
if not os.path.exists( INDICESDIR): os.makedirs( INDICESDIR )

def bjnow():
    t = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return t.astimezone()
def bjnow_str():
    return bjnow().isoformat()

ex_binance = ccxt.binance({'enableRateLimit': True})
ex_bmex = ccxt.bitmex({'enableRateLimit': True})
ex_bybit = ccxt.bybit({'enableRateLimit': True})
ex_okx = ccxt.okx({'enableRateLimit': True})

def get_bmex_next_funding_rate(spot_symbol)->tuple: 
    if 'BTC' in spot_symbol: spot_symbol.replace('BTC','XBT')
    if 'USDT' in spot_symbol: spot_symbol.replace('USDT','USD')
    symbol = spot_symbol.replace('/','').replace('-','') 
    resp = ex_bmex.fetchFundingRate(symbol)
    rt = float(resp['fundingRate'])
    ts = resp['fundingDatetime'] 
    annual = (1+rt)**(365*3)-1 # Every 8 hours
    return annual, rt, ts

def get_binance_next_funding_rate(spot_symbol, use_cached=False)->tuple:
    symbol = spot_symbol.replace('/','').replace('-','')    
    
    # Cache
    if not use_cached:
        cached_files =  os.listdir(FUNDINGDIR)
        fd = list(filter(lambda s: s.startswith(symbol), cached_files))
        if len(fd) == 1:
            fd = fd[0]
            with open(f"{FUNDINGDIR}/{fd}", 'r') as fh:
                t, annual,rt,ts = fh.read().split('\n')
                cachedt = datetime.datetime.fromisoformat( t )
                if (bjnow() - cachedt).seconds < 10: # less than 10 sec
                    return float(annual), float(rt), ts

    resp = ex_binance.fapiPublicGetPremiumIndex()
    r = list(filter(lambda e: symbol == e['symbol'],resp ) )
    assert len(r)==1, f"Funding rate of related perpetual not found: {spot_symbol}"
    '''
    {'estimatedSettlePrice': '396.73116476',
    'indexPrice': '396.03627621',
    'interestRate': '0.00000000',
    'lastFundingRate': '-0.00050799',
    'markPrice': '395.61000000',
    'nextFundingTime': '1709107200000',
    'symbol': 'BNBUSDT',
    'time': '1709090146000'}
    '''
    ts = r[0]['nextFundingTime'] # int in str format
    ts = datetime.datetime.fromtimestamp(int(ts)/1000).isoformat() # str
    rt = float(r[0]['lastFundingRate']) # Binance mis-interperated the "last*" as the "next*"
    annual = (1+rt)**(365*3)-1 # Every 8 hours

    # Cache
    caching = f"{FUNDINGDIR}/{symbol}"
    with open(caching, 'w') as fh:
        fh.write(f"{bjnow_str()}\n{annual}\n{rt}\n{ts}")

    return annual, rt, ts

def get_binance_index(contract)->tuple:
    spot_symbol = contract.split('-')[0]+'/USDT'
    symbol = spot_symbol.replace('/','').replace('-','')    
    
    # Cache
    cached_files =  os.listdir(INDICESDIR)
    fd = list(filter(lambda s: s.startswith(symbol), cached_files))
    if len(fd) == 1:
        fd = fd[0]
        try:
            with open(f"{INDICESDIR}/{fd}", 'r') as fh:
                print(f'-- reading: {INDICESDIR}/{fd}')
                t,v = fh.read().split('\n')
                cachedt = datetime.datetime.fromisoformat( t )
                if (bjnow() - cachedt).seconds < 10: # less than 10 sec
                    return float(v)
        except Exception as e:
            print('*** error:', str(e) )
            print('-- re-fetching index data for:', contract)

    resp = ex_binance.fapiPublicGetPremiumIndex()
    r = list(filter(lambda e: symbol == e['symbol'],resp ) )
    assert len(r)==1, f"Funding rate of related perpetual not found: {spot_symbol}"
    '''
    {'estimatedSettlePrice': '396.73116476',
    'indexPrice': '396.03627621',
    'interestRate': '0.00000000',
    'lastFundingRate': '-0.00050799',
    'markPrice': '395.61000000',
    'nextFundingTime': '1709107200000',
    'symbol': 'BNBUSDT',
    'time': '1709090146000'}
    '''
    v = r[0]['indexPrice']

    # Cache
    caching = f"{INDICESDIR}/{symbol}"
    with open(caching, 'w') as fh:
        fh.write(f"{bjnow_str()}\n{v}")

    return float(v)

def get_binance_spot( symbol='BTC/USDT'): #Alias
    from spot_trading.bs_spot_sql import read_latest_ticker
    try: # use database data first
        bid,ask,ts,timestamp = read_latest_ticker(symbol)
        return bid,ask 
    except Exception as e: # use https
        print( '***', str(e))
        return binance_spot( symbol )
def binance_spot(symbol='BTC/USDT')->tuple:
    symbol = symbol.replace('-','/')
    qts = ex_binance.fetch_ticker(symbol)
    bid,ask = qts['bid'],qts['ask']
    try:
        return float(bid),float(ask)
    except Exception as e:
        print('*** bid/ask data errors:')
        print('***', symbol, qts )
        raise e

def get_underlying(contract):
    fds = contract.split('-')
    return f"{fds[0]}/USDT"

def _file_ts(fn):
    s = os.stat(fn)
    t = int( s.st_mtime )
    d = datetime.datetime.fromtimestamp(int(s.st_mtime) )
    return str(d)

def binance_kline(symbol='BTC/USDT', span="1d", grps=10) -> pd.DataFrame:
    """
    @param grps (int): how many sets of data needed? Each set contains the "limit" by exchange limit, 1000.
    """
    span = span.lower()
    tnow = int(datetime.datetime.utcnow().timestamp()*1000)
    dfs = []
    tsecs = -1
    gap = 990
    if span.endswith('d'):
        tsecs = 24*3600 * int(span.split('d')[0])
    elif span.endswith('h'):
        tsecs = 3600 * int(span.split('h')[0])
    elif span.endswith('m'):
        tsecs = 60 * int(span.split('m')[0])
        gap = 100
    else:
        raise Exception( f"Unsupported span type: {span}")

    for i in range(1,grps+1):
        from_ts = tnow - tsecs*1000 *gap*i
        ohlcvs = ex_binance.fetch_ohlcv(symbol, span,since=from_ts,limit=1000)
        recs = []
        for ohlcv in ohlcvs:
            ts = ex_binance.iso8601(ohlcv[0])
            vals = ohlcv[1:]
            recs += [(ts,vals[0], vals[1],vals[2], vals[3],vals[4])]
        df = pd.DataFrame.from_records( recs, columns = ['timestamp','open','high','low','close','volume'] )
        dfs += [ df ]
        if DEBUG:
            print(df.shape, df.iloc[0].timestamp, df.iloc[-1].timestamp)
    df = pd.concat( dfs, axis=0)
    df = df.sort_values('timestamp', ascending=True).drop_duplicates().reset_index(drop=True)
    
    # Validate data
    xdf = df[['timestamp']].copy()
    xdf['ts'] = xdf.timestamp.apply(pd.Timestamp)
    xdf['dt'] = xdf.ts.diff() # The adjacent data should have SAME time differences, o.w., missing data is possible.
    td = set(list(xdf.dt))
    #assert len( set(list(xdf.dt))) ==2, f'Missing data? symbol={symbol}, span={span}, td: {td} \n {xdf.sort_values("dt", ascending=False)} \n{df}'
    #if len( set(list(xdf.dt))) > 2:
    #    print( f'*** Missing data? symbol={symbol}, span={span}, td: {td} \n {xdf.sort_values("dt", ascending=False)} \n{df}' )

    return df

# Options
def get_maturity( contract:str )->float:
    fds = contract.split('-')
    ts = datetime.datetime.strptime('20'+fds[1], '%Y%m%d')
    tnow = datetime.datetime.utcnow()
    dt = (ts-tnow).total_seconds()/3600./24
    return max(0,dt)

#tests
if __name__ == '__main__':
    df = binance_kline(span='1d')
    print(df)

    df = binance_kline(span='2h')
    print(df)

    df = binance_kline(span='30m')
    print(df)
