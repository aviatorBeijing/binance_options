import pandas as pd
import requests, os, datetime, json,sys
import numpy as np
import socks, socket
import enum
from sqlalchemy import create_engine, text

uname = os.getenv('PG_USERNAME', '')
psw = os.getenv('PG_PASSWORD', '')

kline_engine = engine = create_engine(f"postgresql://{uname}:{psw}@localhost:5432/klines_yahoo")

class AssetClass(enum.Enum):
    MERCH   = 'merch' # gld, crude, etc.
    FOREX   = 'forex'
    CRYPTO  = 'crypto'
    HK_STOCK = 'hk_stock'
    CN_STOCK = 'cn_other_stock'
    CN_ETF = 'bond_etf'
    CN_ASHARE = 'a_share'
    US_STOCK = 'us_stock'
    INDICES = 'indices'

    _UNKNOWN = 'unregistered'

def get_asset_class(ric):
    ric = ric.upper()
    if ric.endswith('.SS') or ric.endswith('.SZ'):
        if ric.startswith('51') or ric.startswith('15'):
            return AssetClass.CN_ETF
        elif ric.startswith('0') or ric.startswith('3') or ric.startswith('6'):
            return AssetClass.CN_ASHARE
        else:
            return AssetClass.CN_STOCK
    elif ric.endswith('.HK'):
        return AssetClass.HK_STOCK
    elif ric.endswith('-USD') or ric.endswith('/USDT') \
        or ric.lower() in 'BTC DOGE ETH BNB SOL XRP ADA AVAX LINK DOT TRX LTC FTM BCH MATIC ARB FIL OP ATOM PEPE LTC SHIB UNI FIL'.lower().split(' '):
        return AssetClass.CRYPTO
    elif ric.endswith('=X'):
        return AssetClass.FOREX
    elif ric.lower() in ['gld'] or ric.upper().endswith('=F'):
        return AssetClass.MERCH
    elif ric.lower() in ['msft','aapl','amzn','goog','nvda','lbrdk','cere','bkng','armk','cdns','nvo','j','tsla']:
        return AssetClass.US_STOCK
    elif ric.startswith('^'):
        return AssetClass.INDICES
    
    return AssetClass._UNKNOWN

tds = datetime.datetime.utcnow()+datetime.timedelta(hours=8)
tds = datetime.datetime.strftime(tds, '%Y_%m_%d')
sim_dir = f"{os.getenv('USER_HOME','/Users/junma')}/data/strategies/intraday/{tds}/"
states_dir = f"{sim_dir}/states"
if not os.path.exists(states_dir): os.makedirs(states_dir)
def is_running(ric):
    state_file = f"{states_dir}/{ric.lower()}.tunning.dat"
    with open( state_file, 'w') as fh:
            fh.write('RUNNING')
def set_complete(ric):
    state_file = f"{states_dir}/{ric.lower()}.tunning.dat"
    ts = datetime.datetime.strftime(bjnow(), '%Y-%m-%d %H:%M:%S')
    with open( state_file, 'w') as fh:
            fh.write(f'COMPLETED,{ts}')
            
class Price:
    def __init__(self,ric,p,ts) -> None:
        self.ric = ric
        self.price = p
        self.ts = ts 
    def __str__(self):
        return f"{self.ric}={self.price} @{self.ts}"
    def __expr__(self):
        return self.__str__()

def table_exists( tbname, _engine=None ):
    stmt = f'''
SELECT EXISTS 
(
	SELECT 1
	FROM information_schema.tables 
	WHERE table_name = '{tbname}'
);
    '''
    df = pd.read_sql(stmt, _engine if _engine else engine)
    tbname_exists = df.iloc[0].exists

    if tbname_exists: # Not only table exists, but also not empty
        stmt = f'SELECT FROM "{tbname}"'
        with _engine.connect() as conn:
            rst = conn.execute( text(stmt ) )
            rows = rst.fetchall()   
            if len(rows)>0:
                return True 
    return False

def func_timeit( funcp ):
    def func_wrapper( *args, **kwargs ):
        import time
        begin = time.time()
        rtn = funcp( *args, **kwargs )
        delta = time.time()-begin
        
        print('* Elapsed time | %s | %12.6f sec'%( funcp.__name__,
                                                    delta ))
        return rtn
    return func_wrapper

def fetch_yahoo( symbol, dtype, startts:int, endts:int )->pd.DataFrame:
    USE_FUTU = False 
    if USE_FUTU: # Need to start FutuBull customer service hub (FutuOpenD) first!
        startts = datetime.datetime.fromtimestamp( startts )
        endts = datetime.datetime.fromtimestamp( endts )
        startts = datetime.datetime.strftime( startts, '%Y-%m-%d')
        endts = datetime.datetime.strftime( endts, '%Y-%m-%d')
        kline = futu_kline(symbol, dtype, startts, endts )
        
        try:
            kline['Timestamp'] = kline['timestamp'] = kline['time_key'].apply(lambda e: (datetime.datetime.strptime(e, '%Y-%m-%d %H:%M:%S')).timestamp() )
            kline['Open'] = kline['open']
            kline['High'] = kline['high']
            kline['Low'] = kline['low']
            kline['Close'] = kline['close']
            kline['Volume'] = kline['volume']
        
            return kline 
        except Exception as e:
            print('***', e)
            return pd.DataFrame()

    else: # Yahooo
        if dtype in ['30m', '15m', '5m', '1h']:
            dt = endts-startts
            if dt > 24*3600*60: # Yahoo intraday data is limited for merely 60days.
                print('      -- modifing the startts to comply with Yahoo limitation.')
                startts = endts - 59*24*3600
        endpoint = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={dtype}&period1={startts}&period2={endts}'
        url = endpoint.format(symbol=symbol, dtype=dtype, startts=int(startts), endts=endts)
        print('--', url)

        tmpsocket = None
        if os.getenv("YAHOO_LOCAL", None):
            tmpsocket = socket.socket
            socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 50000)
            socket.socket = socks.socksocket
                
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        resp = requests.get(url, headers=headers)
        try:
            data = resp.json()['chart']['result'][0]
        except Exception as e:
            with open(f'yahoo_data_error_{symbol}_{dtype}_{startts}_{endts}.err', 'w') as fh:
                fh.writelines([ endpoint, '\n', str(e)])
            return pd.DataFrame()

        if os.getenv("YAHOO_LOCAL", None):
            if tmpsocket:
                socket.socket = tmpsocket
    if 'timestamp' in data:
        t = data['timestamp']
        data = data['indicators']['quote'][0]
        o = data['open']
        h = data['high']
        l = data['low' ]
        c = data['close']
        v = data['volume']
        df = pd.DataFrame.from_records( zip(t,o,h,l,c,v), columns=['Timestamp','Open','High','Low','Close','Volume'])
        df = df.dropna()
        return df
    else:
        return pd.DataFrame()


def get_tbname( symbol, dtype ):
    tbname = f"k_{symbol.lower().replace('.','_')}_{dtype}"
    tbname = tbname.replace('-','_')
    tbname = tbname.replace('^','')
    return tbname

def normalize_data( df )->pd.DataFrame:
    if 'close' in df:
        df['Close'] = df.close
        df['High'] = df.high
        df['Low'] = df.low
        df['Open'] = df.open
        df['Volume'] = df.volume
        df['Timestamp'] = df.timestamp
        if 'adjclose' in df:
            df['AdjClose'] = df.adjclose
    print('-- normalizing: ', df.shape)
    if not df.empty:
        if not 'AdjClose' in df:
            df['AdjClose'] = df.Close

        df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]
    return df 


def _update_data( symbol, dtype, startts=None, endts=None) ->pd.DataFrame:
    tbname = get_tbname( symbol, dtype )
    if '-' in tbname: tbname.replace('-','_')

    tnow = int( datetime.datetime.utcnow().timestamp()) + 3600*8
    try:
        if table_exists( tbname, kline_engine ):
            stmt = f'SELECT MAX("Timestamp") AS latest_ts FROM "{tbname}";'
            df = pd.read_sql(stmt, engine )
            t_latest_db = df.iloc[0].latest_ts
           
            
            stmt = f'SELECT MIN("Timestamp") AS latest_ts FROM "{tbname}";'
            df = pd.read_sql(stmt, engine )
            t_startin_db = df.iloc[0].latest_ts

            def _td( i ):
                return datetime.datetime.fromtimestamp(int(i))
            print(f'  -- in {tbname}')
            print(f'  -- ({symbol}) ts in db: [{t_startin_db}, {t_latest_db}], {_td(t_startin_db)}, {_td(t_latest_db)}')
            print(f'  -- ({symbol}) ts req`d: [{startts}, {endts}]')

            if t_latest_db is None:

                pd.DataFrame().to_csv(f'tmp/{symbol}_{tbname}.none.csv')
                print('*** saved:', f'tmp/{symbol}_{tbname}.none.csv' )
                return pd.DataFrame()
            dt = tnow-t_latest_db
            def _t (t): 
                return datetime.datetime.fromtimestamp(t)
            print(f'-- dt={dt}, {_t(t_latest_db)} ~ {_t(tnow)} ')
            
            if dt>0 or (startts and t_startin_db>startts):
                print(f'-- updating data starting: {t_latest_db}, {datetime.datetime.fromtimestamp(t_latest_db)}, dt={(tnow-t_latest_db)/3600/24} days')
                if startts and t_startin_db>startts:
                    df = fetch_yahoo(symbol, dtype, startts, tnow )   #fetch whole new dataset
                else:
                    df = fetch_yahoo(symbol, dtype, t_latest_db, tnow )
                df = normalize_data( df )
                #print( df.Timestamp.apply(lambda v: datetime.datetime.fromtimestamp(v) ) )
                return df
        return pd.DataFrame()
    except Exception as e:
        print('*** _update_data() errored out.')
        print('*** ', symbol, dtype )
        raise e

def augment_with_latest_price(symbol, odf, dtype):
    if is_market_close() or is_pre_market() or is_noon_break(): 
        print(f'  -- [{symbol}] market closed, will NOT augment with "latest" l1 price ')
        return odf

    t = datetime.datetime.fromtimestamp( datetime.datetime.utcnow().timestamp() + 3600*8 )
    t = t.strftime('%Y%m%d')

    tgap = 24*60*60 # second
    if dtype == '1d': pass 
    elif dtype=='1h': tgap = 60*60
    elif dtype=='30m': tgap = 30*60
    elif dtype=='15m': tgap = 15*60
    elif dtype=='5m': tgap=5*60
    elif dtype=='1wk': tgap*=7
    else:
        raise Exception(f"{dtype} is not supported.")

    ex, code = None, None
    if any( map(lambda s: s in symbol, ['.SS','.SZ','.HK'] ) ):
        ric = symbol.split('.')
        ex = ric[1].lower(); ex = 'sh' if ex == 'ss' else ex
        code = ric[0]
        tbname = f'ob_{t}_{ex}_{code}'
    else:
        tbname = f'ob_{t}_{symbol.lower()}'
        print('  -- non conventional table created: ', tbname )
    
    df = pd.DataFrame()

    if not df.empty:
        bid = df.iloc[0].bid
        ts =  df.iloc[0].timestamp
        if len(ts) != len('2023-02-08 13:39:10.914000'):
            raise Exception(f"Datetime format is incorrect: {ts}. Ex. 2023-02-08 13:39:10.914000 ")
        ts = datetime.datetime.strptime(ts,'%Y-%m-%d %H:%M:%S.%f').timestamp()

        last_ts = odf.iloc[-1].Timestamp # make sure the timestamp is up-to-dated: 
        if int(last_ts) < int(ts): #       the last timestamp is always the latest data.
            #print('--', int(ts)-int(last_ts), bid, odf.iloc[-1].High - bid )
            # Augment the df by one additional row
            ext = pd.DataFrame.from_records([[
                        ts, -1, -1, -1, 
                        bid, bid,
                        -1
                    ]], columns= odf.columns)
            
            doappend = True
            if odf.shape[0]>3:
                """t0,t1 = int(odf.iloc[0].Timestamp), int(odf.iloc[1].Timestamp)
                t2,t3 = int(odf.iloc[2].Timestamp), int(odf.iloc[3].Timestamp)
                if (t1-t0) != (t3-t2): raise Exception(f"{tbname} data timestamp is NOT consistent. {t1-t0} != {t3-t2}")

                dt = t1-t0 # nominal delta time, corresponding to "dtype", 1d, 1h, 30m, etc.
                """

                if (int(ts)-int(last_ts)) >0 and (int(ts)-int(last_ts))<tgap:
                    # do update instead of append
                    print('  -- do update (no concat)')
                    doappend = False # disable the concat below
                    
                    odf.loc[odf.index[-1], 'Timestamp'] = ts
                    odf.loc[odf.index[-1], 'Close'] = bid
                    odf.loc[odf.index[-1], 'AdjClose'] = bid
                    if odf.iloc[-1].High < bid: odf.loc[odf.index[-1], 'High'] = bid 
                    if odf.iloc[-1].Low > bid: odf.loc[odf.index[-1], 'Low'] = bid 

                    #print(odf)
            else:
                pass
            if doappend:
                odf = pd.concat( [odf, ext ])   # Augmenting !!!
            
            # printing
            """
            xodf = odf.copy();xodf.Timestamp = xodf.Timestamp.apply(lambda t: datetime.datetime.fromtimestamp(int(t)))
            print( f"    -- [{symbol}] first 5 rows:\n", xodf.head(5) )
            print( f"    -- [{symbol}] last 5 rows:\n", xodf.tail(5) )
            """

    return odf

def get_factor( dtype ):
    supported = {'5m': 5*60,
                 '15m': 15*60,
                 '30m': 30*60,
                 '1h': 60*60,
                 '1d': 60*60*24,
                 '1wk': 60*60*24*7,
                 '1mo': 60*60*24*30}
    if not dtype in supported: raise Exception('Not supported.')
    return supported[dtype]

def get_data(
                symbol,        # Ex.: 603442.SS
                dtype,         # Acceptable string: '1d', '1h', '30m', etc.
                days=10*365,       # Backdating days, counting from now
                realtime=False # From db, or fetch latest and combine with db
                ):
    df = pd.DataFrame() # Final data
    endts = bjnow().timestamp()
    endts += 24*3600 # Add 1-day ahead to acquire more recent data
    startts = endts - 24*3600 * days
    startts = int(startts); endts = int(endts)
    tbname = get_tbname( symbol, dtype )
    
    if '-' in tbname: 
        tbname = tbname.replace('-','_')
    print('-- data table name:', tbname)
    
    def _c():
        return pd.read_sql(f'SELECT * FROM "{tbname}" WHERE "Timestamp"<{endts} AND "Timestamp">{startts};', engine ) 

    if table_exists( tbname, kline_engine ) and _c().shape[0]>0: # Table existed => fetch + update & return
        df = _c() # Existing data
        
        print( f'-- [rt={"off" if not realtime else "on"}] existing data length, tbname={tbname}, data shape={df.shape} (equivalent {get_factor(dtype)*df.shape[0]/24/3600} days)')
        if df.shape[0] == 0:
            print('*'*50)
            print('   Empty data:', symbol)
            print('*'*50)
        if not df.empty: # Old data
            new_df = pd.DataFrame()
            if realtime:
                try:
                    # TODO
                    # efficiency issue here?? fetching the new data in certain date range only might be better.
                    new_df = _update_data( symbol, dtype, startts, endts )
                    print(f'-- new data shape: {new_df.shape}')
                except Exception as e:
                    print('*** get_date() error: ', symbol)
                    raise e
            if not new_df.empty: # New data
                if df.iloc[-1].Timestamp == new_df.iloc[-1].Timestamp: # Same ending ts, ignore
                    print(f'-- {symbol} ignore, db v.s. updates, same ending timestamp.')
                    pass
                elif df.iloc[-1].Timestamp < new_df.iloc[-1].Timestamp and df.iloc[0].Timestamp > new_df.iloc[0].Timestamp: # inclusive
                    df = new_df
                    df.reset_index(inplace=True)
                    df.sort_values('Timestamp', inplace=True, ascending=True)
                else:
                    print( f'-- new data length: {new_df.shape[0]}, last new:')
                    if df.iloc[-1].Timestamp % get_factor(dtype) != 0:
                        print('  -- excluding one incomplete record at: ', df.iloc[-1].Timestamp )
                        if df.iloc[-1].Timestamp > new_df.iloc[0].Timestamp \
                            and df.iloc[-2].Timestamp < new_df.iloc[0].Timestamp:
                            print('  -- deleting one record at: ', df.iloc[-1].Timestamp )
                            #engine.execute(f"DELETE FROM \"{tbname}\" WHERE \"Timestamp\"={df.iloc[-1].Timestamp}" )
                        df = df[:-1]

                    for col in ['level_0', 'index']:
                        if col in df: df.drop(columns=[col], inplace=True)
                    #print('***** db df:\n', df)
                    #print('***** new_df:\n', new_df )
                    df = pd.concat( [df, new_df], ignore_index=True) 
                    df.reset_index(inplace=True)
                    df.sort_values('Timestamp', inplace=True, ascending=True)

                for col in ['level_0', 'index']:
                    if col in df: df.drop(columns=[col], inplace=True)

                df = df.groupby('Timestamp').mean().reset_index() # average possible duplicates
                #print('**** merged:')
                #print(df)
                #sys.exit()
                try:
                    df.to_sql( tbname, engine, index=False, if_exists="replace" )
                except Exception as e:
                    import time, random
                    time.sleep( random.randint(0,20) )
                    df.to_sql( tbname, engine, index=False, if_exists="replace" ) # try again
                #engine.execute(f'DELETE FROM "{tbname}" WHERE "Timestamp"=1670205300 AND "Volume"=0;')
                with engine.connect() as conn:
                    conn.execute( text(f"ALTER TABLE \"{tbname}\" ADD PRIMARY KEY (\"Timestamp\");") )
                
            #return df
    else: # Otherwise, fetch & create the table.
        print('-- kline not found in db, fetching new data')
        def get_xd_data(endts, days=1):
            endts += 24*3600 # Add 1-day ahead to acquire more recent data
            startts = endts - 24*3600 * days
            startts = int(startts); endts = int(endts)
            print(f'    -- fetching yahoo for {symbol} {(endts-startts)/3600/24} days')
            return fetch_yahoo(symbol, dtype, startts, endts)

        def get_1d_data(endts):
            return get_xd_data(endts, days=1)

        endts = bjnow().timestamp()
        
        if False:
            dfs = []
            print( f'-- fetching {days} days')
            for i in range(days):
                df = get_1d_data( endts - 24*3600*i)
                if not df.empty:
                    dfs+= [df[:-1]]
            df = pd.concat( dfs[::-1], ignore_index=True )
        else:
            if not dtype in ['5m','15m','30m', '1h']:
                df = get_xd_data(endts, max(days, 365) )
            else:
                df = get_xd_data(endts, min(days, 59) ) # Limited by Yahoo API, intraday data is limited within 60 days.

        df = normalize_data(df)
        if not df.empty:
            df.to_sql( get_tbname(symbol, dtype), engine, index=False, if_exists="replace")
            with engine.connect() as conn:
                conn.execute( text( f"ALTER TABLE \"{tbname}\" ADD PRIMARY KEY (\"Timestamp\");") )
        else:
            print('*** empty yahoo fin data:', symbol )

    # Now, df is ready.
    if df.empty:
        #raise Exception(f'Empty data? {symbol}, {dtype}')
        return pd.DataFrame()

    df = augment_with_latest_price( symbol, df, dtype )

    return df

# Utils
def datestr(d:datetime.datetime) -> str:
    if not isinstance( d, datetime.datetime ): return d
    return d.strftime('%Y-%m-%d %H:%M:%S.%f')

def bjnow() -> datetime.datetime:
    return datetime.datetime.fromtimestamp( datetime.datetime.utcnow().timestamp() + 3600*8 )

def is_market_close() ->bool:
    beijing_now = bjnow()
    beijing_now_str = datestr( beijing_now )
    beijing_close = datetime.datetime.strptime( beijing_now_str.split()[0] + ' 15:00:00', '%Y-%m-%d %H:%M:%S')
    beijing_post_market :bool = beijing_now > beijing_close
    return beijing_post_market

def is_noon_break() ->bool:
    beijing_now = bjnow()
    beijing_now_str = datestr( beijing_now )
    beijing_break_starts = datetime.datetime.strptime( beijing_now_str.split()[0] + ' 11:30:00', '%Y-%m-%d %H:%M:%S')
    beijing_break_ends = datetime.datetime.strptime( beijing_now_str.split()[0] + ' 13:00:00', '%Y-%m-%d %H:%M:%S')
    beijing_noon :bool = ( beijing_now > beijing_break_starts and beijing_now < beijing_break_ends)
    return beijing_noon

def is_pre_market() ->bool:
    beijing_now = bjnow()
    beijing_now_str = datestr( beijing_now )
    beijing_open = datetime.datetime.strptime( beijing_now_str.split()[0] + ' 09:30:00', '%Y-%m-%d %H:%M:%S')
    beijing_pre_market :bool = beijing_now < beijing_open
    return beijing_pre_market

if __name__ == '__main__':
    df = get_data('000729.SZ','1d',days=365*10,realtime=False)
    print( df )
