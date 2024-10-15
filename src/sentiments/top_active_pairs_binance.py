import os
import pandas as pd
import ccxt
import numpy  as np

def top_rics(n):
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    usdt_pairs = [market for market in markets.values() if market['quote'] == 'USDT']
    # Fetch 24h ticker data for each USDT pair
    tickers = exchange.fetch_tickers()

    volume_data = []
    for pair in usdt_pairs:
        symbol = pair['symbol']
        if symbol in tickers:
            volume = tickers[symbol]['quoteVolume']
            close = tickers[symbol]['close']
            volume_data.append((symbol, volume, close))

    sorted_volume_data = sorted(volume_data, key=lambda x: x[1], reverse=True)

    top_10_pairs = sorted_volume_data[:n]

    df = pd.DataFrame.from_records( top_10_pairs, columns=['ric','volume','close'] )
    return df

def main():
    df = top_rics( n=10 )
    fn = os.getenv('USER_HOME','')  + '/tmp/top_active_rics.csv'
    df.to_csv(fn, index=0)
    print(df)
    print('-- save:', fn)

if __name__  == '__main__':
    main()
