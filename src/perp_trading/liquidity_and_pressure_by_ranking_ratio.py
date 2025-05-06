import asyncio
import json
import pandas as pd
import numpy as np
import websockets
import ccxt
from collections import deque
from datetime import datetime
import pytz
import click

# --- Configurable ---
MARKET_TYPE = 'perp'  # 'spot' or 'perp'
INTERVAL = '1m'
ROLLING_WINDOW = 100  # 99 from REST + 1 from WebSocket

# --- Fetch input symbols from command line ---
@click.command()
@click.option('--syms', default='btcusdt,moveusdt,layerusdt', help='Comma-separated list of symbols')
@click.option('--market', default='perp', type=click.Choice(['spot', 'perp']), help="Market type: 'spot' or 'perp'")
def main(syms, market):
    # Parse the symbols from the command line
    BASE_SYMBOLS = syms.split(',')
    BASE_SYMBOLS = list(
            map(lambda s: s.lower() if 'usdt' in s.lower() else f'{s.lower()}usdt', 
                BASE_SYMBOLS
                ))

    # Adjust for market type
    if market == 'spot':
        SYMBOLS = BASE_SYMBOLS
        stream_format = lambda sym: f"{sym}@kline_{INTERVAL}"
        WS_BASE = "wss://stream.binance.com:9443/stream"
        EXCHANGE = ccxt.binance()
    elif market == 'perp':
        SYMBOLS = [s.upper() for s in BASE_SYMBOLS]
        stream_format = lambda sym: f"{sym.lower()}@kline_{INTERVAL}"
        WS_BASE = "wss://fstream.binance.com/stream"
        EXCHANGE = ccxt.binanceusdm()
    else:
        raise ValueError("MARKET_TYPE must be 'spot' or 'perp'")

    # Prepare WebSocket stream
    stream_names = [stream_format(symbol) for symbol in SYMBOLS]
    stream_url = f"{WS_BASE}?streams={'/'.join(stream_names)}"

    # --- Global state ---
    kline_data = {}
    last_rest_ts = {}

    # --- Fetch historical data ---
    def fetch_initial_klines():
        print("Fetching historical data...")
        for symbol in SYMBOLS:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=INTERVAL, limit=ROLLING_WINDOW - 1)
            deque_data = deque(maxlen=ROLLING_WINDOW)
            for o in ohlcv:
                ts, open_, high, low, close, volume = o
                ret = (close - open_) / open_
                deque_data.append({'return': ret, 'volume': volume, 'ts': ts})
            kline_data[symbol] = deque_data
            last_rest_ts[symbol] = ohlcv[-1][0]  # Last bar's timestamp (ms)
        print("Historical data loaded.")

    # --- Process new kline ---
    async def process_kline(symbol, kline):
        kline_ts = int(kline['t'])  # Candle open time in ms

        # Avoid duplicate processing
        if kline_ts == last_rest_ts.get(symbol):
            return

        close = float(kline['c'])
        open_ = float(kline['o'])
        volume = float(kline['v'])
        ret = (close - open_) / open_

        # Append new data
        kline_data[symbol].append({'return': ret, 'volume': volume, 'ts': kline_ts})

        # Extract returns and volumes for this symbol
        symbol_returns = pd.Series([item['return'] for item in kline_data[symbol]])
        symbol_volumes = pd.Series([item['volume'] for item in kline_data[symbol]])

        # Use absolute return for ranking
        abs_returns = symbol_returns.abs()

        # Rank latest return and volume among this symbol's history
        return_rank = abs_returns.rank(pct=True).iloc[-1]
        volume_rank = symbol_volumes.rank(pct=True).iloc[-1]
        ratio = return_rank / volume_rank if volume_rank != 0 else float('inf')

        # Timestamp with timezone
        tz = pytz.timezone("Asia/Shanghai")
        current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        print(f"\n--- RANK RATIOS @ {current_time} --- (for {symbol})")
        print(f"{symbol} | Return Rank: {return_rank*100:.1f}% ({ret*100:.2f}%) | Volume Rank: {volume_rank*100:.1f}% | Ratio: {ratio:.2f}")

    async def websocket_loop():
        fetch_initial_klines()
        async with websockets.connect(stream_url) as websocket:
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)

                payload = data['data']
                symbol = payload['s']
                if payload['k']['x']:  # Closed kline
                    await process_kline(symbol, payload['k'])

    asyncio.run(websocket_loop())

if __name__ == "__main__":
    main()

