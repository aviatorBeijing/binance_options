import os,tqdm
from datetime import datetime
import numpy  as np
import ccxt
import pandas as pd
import time
from scipy.stats import percentileofscore

exchange = ccxt.binance()

def get_markets():
    markets = exchange.load_markets()
    usdt_pairs = [market['symbol'] for market in markets.values() if market['quote'] == 'USDT']
    usdt_pairs = list( filter(lambda s: ':USDT' not in s, usdt_pairs) )
    syms = list(map(lambda s: s.replace('/USDT',''), usdt_pairs) )
    return syms

def fetch_ohlcv(symbol, timeframe, since, limit=1000):
    all_ohlcv = []
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1  # Move to the next batch of data
            all_ohlcv += ohlcv
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    return all_ohlcv

def get_5_years_data(symbol):
    timeframe = '1d'  # Daily data
    limit = 1000  # Max number of records per request
    since = exchange.parse8601('2018-10-15T00:00:00Z')  # Start date (5 years ago)
    ohlcv_data = fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert to human-readable date
    return df[['timestamp', 'close', 'volume']]  # Return only date, close, and volume

recs = []
symbols = get_markets() #;print(m)
#symbols = ['btc','eth','doge', 'ltc']
for sym in tqdm.tqdm( symbols):
    pair = f'{sym.upper()}/USDT'
    df = get_5_years_data(pair)
    last_timestamp = df.iloc[-1]['timestamp']
    try:
        assert last_timestamp.date() == datetime.utcnow().date(), f'{pair}: last date is not today!\n{df.tail()}'
    except AssertionError as e:
        #print('***', str(e))
        continue
    recs += [{
        'pair': pair,
        'pct_close': percentileofscore(df.close, df.close.iloc[-1] ),
        'pct_volume (recent max)': percentileofscore(df.volume, np.max(df.volume.iloc[-6:-1]) ), # recent 5 days
        'pct_volume_today (partial)': percentileofscore(df.volume, df.volume.iloc[-1]),
        }]
pdf = pd.DataFrame.from_records(recs)
pdf.sort_values('pct_close', ascending=True,inplace=True)
print( pdf )

