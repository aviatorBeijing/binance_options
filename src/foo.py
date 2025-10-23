import threading
import websocket
import rel
import time
import json
import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# === Option Contract Classes ===
class OptionContract:
    def __init__(self, symbol):
        self.symbol = symbol
        self.bid = None
        self.ask = None
        self.iv = None
        self.last_trade = None
        self.delta = self.gamma = self.theta = self.vega = None
        self.iv_bid = self.iv_ask = None
        self.timestamp = None

    def update(self, data):
        self.last_trade = data['last_trade']
        self.bid = data['bid']
        self.ask = data['ask']
        self.iv = data['impvol']
        self.iv_bid = data['impvol_bid']
        self.iv_ask = data['impvol_ask']
        self.delta = data['delta']
        self.gamma = data['gamma']
        self.theta = data['theta']
        self.vega = data['vega']
        self.timestamp = data['ts_beijing']

    def mid_price(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    def __repr__(self):
        return f"<{self.symbol}: bid={self.bid}, ask={self.ask}, iv={self.iv}>"


class OptionContractManager:
    def __init__(self):
        self.contracts = {}

    def update_from_message(self, msg):
        df = pd.DataFrame.from_records([msg])
        if df.empty:
            return

        df['spread'] = df['ao'].astype(float) - df['bo'].astype(float)
        df['delta'] = df['d']
        df['gamma'] = df['g']
        df['theta'] = df['t']
        df['vega'] = df['v']
        df['impvol'] = df['vo']
        df['impvol_bid'] = df['b']
        df['impvol_ask'] = df['a']

        sym = df['s'].values[0]
        contract_data = {
            "last_trade": df.iloc[0].c,
            "bid": df.iloc[0].bo,
            "ask": df.iloc[0].ao,
            "delta": df.iloc[0].delta,
            "gamma": df.iloc[0].gamma,
            "theta": df.iloc[0].theta,
            "vega": df.iloc[0].vega,
            "impvol": df.iloc[0].impvol,
            "impvol_bid": df.iloc[0].impvol_bid,
            "impvol_ask": df.iloc[0].impvol_ask,
            "ts_beijing": int((datetime.datetime.utcnow() + datetime.timedelta(hours=8)).timestamp()),
        }

        if sym not in self.contracts:
            self.contracts[sym] = OptionContract(sym)
        self.contracts[sym].update(contract_data)


# === WebSocket App ===
class OptionDataApp:
    def __init__(self, symbols, channel='trade'):
        self.symbols = symbols
        self.channel = channel
        self.manager = OptionContractManager()
        self.thread = threading.Thread(target=self._start_ws_loop, daemon=True)
        self.thread.start()

    def _start_ws_loop(self):
        uri = self._build_ws_uri()
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            uri,
            on_open=self._on_open,
            on_message=self._wrap_message_handler(),
            on_error=self._on_error,
            on_close=self._on_close
        )
        ws.run_forever(dispatcher=rel, reconnect=5)
        rel.signal(2, rel.abort)
        rel.dispatch()

    def _wrap_message_handler(self):
        def handler(ws, message):
            try:
                self.manager.update_from_message(json.loads(message))
            except Exception as e:
                print("Error in message handler:", e)
        return handler

    def _on_open(self, ws):
        print("Opened WebSocket connection")

    def _on_error(self, ws, error):
        print("WebSocket error:", error)

    def _on_close(self, ws, code, msg):
        print("WebSocket closed")

    def _build_ws_uri(self):
        endpoint = 'wss://nbstream.binance.com/eoptions/ws/'
        streams = [f"{s}@{self.channel}" for s in self.symbols]
        return endpoint + "/".join(streams)

    def get_contract(self, symbol):
        return self.manager.contracts.get(symbol.upper())

    def get_all_contracts(self):
        return list(self.manager.contracts.values())

    def wait_until_ready(self):
        while not self.manager.contracts:
            print("Waiting for option data...")
            time.sleep(1)


# === Example Usage ===
if __name__ == "__main__":
    symbols = ["BTC-250829-120000-C", "BTC-250829-118000-P"]
    app = OptionDataApp(symbols)
    app.wait_until_ready()

    try:
        while True:
            contracts = app.get_all_contracts()
            for c in contracts:
                print(c)
            print("---")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Terminated by user")

