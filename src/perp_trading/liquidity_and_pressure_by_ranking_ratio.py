import asyncio
import json
import pandas as pd
import numpy as np
import websockets
import ccxt
from collections import deque
from datetime import datetime
import pytz

# --- Configurable ---
MARKET_TYPE = 'perp'  # 'spot' or 'perp'
INTERVAL = '1m'
ROLLING_WINDOW = 100  # 99 from REST + 1 from WebSocket

# Base symbols (no caps)
BASE_SYMBOLS = ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt']

# Adjust for market type
if MARKET_TYPE == 'spot':
    SYMBOLS = BASE_SYMBOLS
    stream_format = lambda sym: f"{sym}@kline_{INTERVAL}"
    WS_BASE = "wss://stream.binance.com:9443/stream"
    EXCHANGE = ccxt.binance()
elif MARKET_TYPE == 'perp':
    SYMBOLS = [s.upper() for s in BASE_SYMBOLS]
    stream_format = lambda sym: f"{sym.lower()}@kline_{INTERVAL}"
    WS_BASE = "wss://fstream.binance.com/stream"
    EXCHANGE = ccxt.binanceusdm()
else:
    raise ValueError("MARKET_TYPE must be 'spot' or 'perp'")

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

    kline_data[symbol].append({'return': ret, 'volume': volume, 'ts': kline_ts})

    # Prepare rank data
    latest_returns = {s: d[-1]['return'] for s, d in kline_data.items()}
    latest_volumes = {s: d[-1]['volume'] for s, d in kline_data.items()}

    returns_series = pd.Series(latest_returns)
    volumes_series = pd.Series(latest_volumes)

    return_ranks = returns_series.rank()
    volume_ranks = volumes_series.rank()
    ratio = return_ranks / volume_ranks

    # Timestamp with timezone
    tz = pytz.timezone("Asia/Shanghai")  # Change as needed
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"\n--- RANK RATIOS @ {current_time} ---")
    for sym in SYMBOLS:
        print(f"{sym} | Return Rank: {return_ranks[sym]:.1f} | Volume Rank: {volume_ranks[sym]:.1f} | Ratio: {ratio[sym]:.3f}")

# --- WebSocket loop ---
async def main():
    fetch_initial_klines()
    async with websockets.connect(stream_url) as websocket:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)

            payload = data['data']
            symbol = payload['s']
            if payload['k']['x']:  # Closed kline
                await process_kline(symbol, payload['k'])

asyncio.run(main())

