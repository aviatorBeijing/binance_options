import json
import time
import math
import os
import asyncio
from typing import List, Dict, Any
from collections import deque
from asyncio import get_event_loop

# --- EXTERNAL LIBRARIES (Assumed installed for live operation) ---
import websockets
import ccxt
import pandas as pd
import numpy as np
# ----------------------------------------------------------------

# --- GLOBAL FIREBASE/ENV VARIABLES (MANDATORY INSTRUCTION for FIREBASE mode) ---
_app_id = "quant-grid-btc-usdt"
__firebase_config = '{"apiKey": "AIzaSy...", "authDomain": "...", "projectId": "..."}'
__initial_auth_token = "some-jwt-token-if-available"
# -----------------------------------------------------------------------------

# --- 1. CONFIGURATION AND PARAMETERS ---
class GridConfig:
    """Stores all configurable parameters for the Live Grid Trading Framework."""
    def __init__(self):
        # --- CORE TRADING PARAMETERS (User input symbols integrated here) ---
        self.SYMBOLS = ["BTCUSDT", "ETHUSDT"] # List of symbols to track/trade (uppercase)
        self.MARKET_TYPE = 'perp'           # 'spot' or 'perp'
        self.INTERVAL = '1m'                # Kline interval for volatility calculation
        self.ROLLING_WINDOW = 240           # 4 hours of 1m klines for volatility lookback
        self.TRADING_SYMBOL = self.SYMBOLS[0] # The specific symbol the bot will trade
        
        self.LEVERAGE = 5.0
        self.MARGIN_MODE = "ISOLATED"
        self.GRID_COUNT = 20 # Number of buy/sell levels
        self.INITIAL_PRICE = 60000.0 # Starting price for grid center (will be updated by live price)

        # --- PERSISTENCE SETTINGS ---
        # Options: 'LOCAL_CACHE' (for testing/sim) or 'FIREBASE' (for production)
        self.PERSISTENCE_MODE = "LOCAL_CACHE"
        self.LOCAL_CACHE_DIR = "/tmp/binance_hft"
        self.LOCAL_CACHE_FILENAME = f"{self.TRADING_SYMBOL.lower()}_grid_state.json"
        
        # --- GRID GEOMETRY OPTIONS ---
        self.SPACING_METHOD = "GEOMETRIC"
        self.PRICE_RANGE_PERCENT = 0.02

        # --- RISK & SIZING ---
        self.BASE_QTY_USD = 100 
        self.HARD_STOP_LOSS_PRICE = self.INITIAL_PRICE * 0.95 

        # --- DYNAMIC VOLATILITY ADJUSTMENT ---
        self.USE_DYNAMIC_VOLATILITY = True
        self.VOLATILITY_SCALING_FACTOR = 1.5 

        # --- VOLATILITY-ADJUSTED SIZING ---
        self.USE_VOLATILITY_SIZING = True
        self.MIN_VOLATILITY_FOR_SIZING = 0.0001
        self.VOLATILITY_SIZE_FACTOR = 1000

        # --- INVENTORY CONTROL ---
        self.USE_INVENTORY_CONTROL = True
        self.INVENTORY_SKEW_FACTOR = 0.25
        self.MAX_INVENTORY_THRESHOLD = 500.0

        # --- VELOCITY FILTER (Momentum/Trend Risk) ---
        self.USE_VELOCITY_FILTER = True
        self.CRITICAL_VELOCITY_THRESHOLD = 0.005
        self.VELOCITY_CHECK_PERIOD = 3

        # --- FIREBASE & PERSISTENCE (Production Only) ---
        self.APP_ID = _app_id
        self.COLLECTION_NAME = "grid_states"

    def to_dict(self):
        # Convert deque to list for serialization
        d = self.__dict__.copy()
        d['SYMBOLS'] = self.SYMBOLS
        return d

# --- 2. FIREBASE UTILITIES (Conceptual Imports for Production Mode) ---
# Functions remain conceptual as the default mode is local cache.
def get_db_and_auth():
    """Simulates Firebase initialization and authentication (Production only)."""
    print("--- Initializing Firebase and Auth (Conceptual/Production Mode) ---")
    # [Actual Firebase initialization code would go here]
    return "mock_db_instance", "mock_user_id"

def get_firestore_path(app_id: str, user_id: str, collection: str, doc_id: str) -> str:
    """Generates the Firestore path for private user data (Production only)."""
    return f"/artifacts/{app_id}/users/{user_id}/{collection}/{doc_id}"

# --- 3. QUANTITATIVE STRATEGY LOGIC ---

def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculates the standard deviation of returns (sigma).
    Requires pd.Series from the kline history.
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std())

def generate_grid(config: GridConfig, current_price: float, volatility: float) -> List[float]:
    """
    Generates dynamic grid levels based on configuration and calculated volatility.
    
    """
    grid_levels = []
    P_mid = current_price
    N = config.GRID_COUNT // 2

    if config.USE_DYNAMIC_VOLATILITY and volatility > 0:
        # Dynamic Spacing: Delta P = k * sigma
        # Volatility is usually calculated on log returns, but for simplicity, 
        # using the return stddev here as the base step ratio.
        base_step_ratio = volatility * config.VOLATILITY_SCALING_FACTOR
        print(f"   [Dynamic] Base Step Ratio (sigma * k): {base_step_ratio:.5f}")
    else:
        base_step_ratio = config.PRICE_RANGE_PERCENT / N

    if config.SPACING_METHOD == "ARITHMETIC":
        base_step = base_step_ratio * P_mid
        for i in range(1, N + 1):
            grid_levels.append(P_mid - (i * base_step)) # Buy
            grid_levels.append(P_mid + (i * base_step)) # Sell
    else: # GEOMETRIC
        ratio = 1 + base_step_ratio
        for i in range(1, N + 1):
            grid_levels.append(P_mid / (ratio ** i)) # Buy
            grid_levels.append(P_mid * (ratio ** i)) # Sell

    grid_levels.sort()
    return grid_levels

def velocity_filter_check(config: GridConfig, price_history: List[float]) -> bool:
    """
    Implements the 'Velocity Filter' to detect high momentum (trend).
    """
    if not config.USE_VELOCITY_FILTER or len(price_history) < config.VELOCITY_CHECK_PERIOD:
        return False

    recent_prices = price_history[-config.VELOCITY_CHECK_PERIOD:]
    P_start = recent_prices[0]
    P_end = recent_prices[-1]

    RPC = abs((P_end - P_start) / P_start)

    if RPC > config.CRITICAL_VELOCITY_THRESHOLD:
        return True

    return False

# --- 4. THE GRID TRADING FRAMEWORK CLASS ---

class GridTradingFramework:
    def __init__(self, config: GridConfig):
        self.config = config
        
        # --- CCXT/WebSocket Setup ---
        if config.MARKET_TYPE == 'perp':
            self.exchange = ccxt.binanceusdm()
            ws_base = "wss://fstream.binance.com/stream"
        else:
            self.exchange = ccxt.binance()
            ws_base = "wss://stream.binance.com:9443/stream"
        
        # Prepare WebSocket stream URL for all configured symbols
        stream_format = lambda sym: f"{sym.lower()}@kline_{config.INTERVAL}"
        stream_names = [stream_format(symbol) for symbol in config.SYMBOLS]
        self.stream_url = f"{ws_base}?streams={'/'.join(stream_names)}"

        # --- Persistence Setup ---
        if self.config.PERSISTENCE_MODE == 'FIREBASE':
            self.db, self.user_id = get_db_and_auth()
            self.state_path = get_firestore_path(
                self.config.APP_ID, self.user_id, self.config.COLLECTION_NAME, self.config.TRADING_SYMBOL
            )
        else: # LOCAL_CACHE
            self.cache_filepath = os.path.join(self.config.LOCAL_CACHE_DIR, self.config.LOCAL_CACHE_FILENAME)
        
        # --- State Variables ---
        self.grid_levels: List[float] = []
        self.active_orders: Dict[str, Any] = {}
        self.price_history: List[float] = [] # For velocity filter
        self.is_active = True
        self.last_sync_time = 0
        self.inventory: float = 0.0
        self.current_volatility: float = 0.0
        self.klines_data: Dict[str, deque] = {} # {SYMBOL: deque({'close': price, 'return': ret})}

    # --- PERSISTENCE METHODS ---
    def load_state(self):
        """Loads persistent state based on the configured mode."""
        if self.config.PERSISTENCE_MODE == 'LOCAL_CACHE':
            self._load_state_local()
        elif self.config.PERSISTENCE_MODE == 'FIREBASE':
            self._load_state_firebase()

    def save_state(self):
        """Saves current state based on the configured mode."""
        if self.config.PERSISTENCE_MODE == 'LOCAL_CACHE':
            self._save_state_local()
        elif self.config.PERSISTENCE_MODE == 'FIREBASE':
            self._save_state_firebase()

    def _load_state_local(self):
        """Loads persistent state from the local JSON cache."""
        print(f"Attempting to load state from local file: {self.cache_filepath}...")
        try:
            with open(self.cache_filepath, 'r') as f:
                state = json.load(f)
                self.grid_levels = state.get('grid_levels', [])
                self.active_orders = state.get('active_orders', {})
                self.is_active = state.get('is_active', True)
                self.inventory = state.get('inventory', 0.0)
                self.current_volatility = state.get('volatility', 0.0)
            print("Local state loaded successfully.")
        except FileNotFoundError:
            print("No local cache file found. Initializing new grid.")
        except json.JSONDecodeError:
            print("Error decoding local cache file. Initializing new grid.")

    def _save_state_local(self):
        """Saves current state to the local JSON cache."""
        current_state = {
            "timestamp": time.time(),
            "config": self.config.to_dict(),
            "grid_levels": self.grid_levels,
            "active_orders": self.active_orders,
            "is_active": self.is_active,
            "inventory": self.inventory,
            "volatility": self.current_volatility
        }
        
        os.makedirs(self.config.LOCAL_CACHE_DIR, exist_ok=True)
        
        try:
            with open(self.cache_filepath, 'w') as f:
                json.dump(current_state, f, indent=4)
            print(f"State saved to local cache at {time.strftime('%H:%M:%S')} | Inventory: {self.inventory:.4f} BTC")
        except IOError as e:
            print(f"Error saving state to local cache: {e}")

    def _load_state_firebase(self):
        """Simulates loading persistent state from Firestore (Production only)."""
        # [Actual Firestore loading logic would go here]
        pass

    def _save_state_firebase(self):
        """Simulates saving the current state to Firestore (Production only)."""
        # [Actual Firestore saving logic would go here]
        print(f"State saved to Firestore at {time.strftime('%H:%M:%S')} | Inventory: {self.inventory:.4f} BTC")
    
    # --- CCXT DATA FETCHING ---
    async def fetch_initial_klines(self):
        """Fetch historical data for all symbols using CCXT (REST)."""
        print("Fetching historical data...")
        self.klines_data = {}
        limit = self.config.ROLLING_WINDOW - 1
        
        # CCXT uses synchronous calls, but we wrap it in a thread pool for async compatibility
        def sync_fetch(symbol):
            return self.exchange.fetch_ohlcv(symbol, timeframe=self.config.INTERVAL, limit=limit)

        for symbol in self.config.SYMBOLS:
            try:
                # --- MODIFIED LINE START ---
                # Get the current running event loop
                loop = get_event_loop()
                # Use loop.run_in_executor to run the blocking sync_fetch in a thread pool
                ohlcv = await loop.run_in_executor(None, sync_fetch, symbol)
                # --- MODIFIED LINE END ---

                # python>=3.9 Use asyncio.to_thread to run the synchronous CCXT call
                #ohlcv = await asyncio.to_thread(sync_fetch, symbol)
                
                deque_data = deque(maxlen=self.config.ROLLING_WINDOW)
                for o in ohlcv:
                    # [ts, open, high, low, close, volume]
                    ts, open_, _, _, close, _ = o
                    ret = (close - open_) / open_ if open_ else 0.0
                    deque_data.append({'close': close, 'return': ret, 'ts': ts})
                
                self.klines_data[symbol] = deque_data
                print(f"  Loaded {len(ohlcv)} klines for {symbol}.")
                
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
                
        # Set the initial price for the trading symbol
        if self.config.TRADING_SYMBOL in self.klines_data:
            last_close = self.klines_data[self.config.TRADING_SYMBOL][-1]['close']
            self.config.INITIAL_PRICE = last_close
            print(f"  Initial price set to: ${last_close:,.2f}")
        else:
            raise RuntimeError(f"Failed to load data for trading symbol {self.config.TRADING_SYMBOL}")


    # --- TRADING LOGIC IMPLEMENTATION ---

    def calculate_order_qty(self, side: str, price: float) -> float:
        """
        Calculates the quantity (in base currency) applying:
        1. Volatility-Adjusted Sizing
        2. Inventory Sizing Skew
        """
        P_mid = price
        
        # 1. Volatility-Adjusted Base Size (in USD)
        base_qty_usd = self.config.BASE_QTY_USD
        if self.config.USE_VOLATILITY_SIZING and self.current_volatility > self.config.MIN_VOLATILITY_FOR_SIZING:
            # Sizing is inversely proportional to volatility
            volatility_factor = self.config.VOLATILITY_SIZE_FACTOR / (self.current_volatility * P_mid)
            base_qty_usd = max(self.config.BASE_QTY_USD, volatility_factor)
        
        # 2. Inventory Sizing Skew (Limits size to reduce inventory risk)
        skew_factor = 1.0
        if self.config.USE_INVENTORY_CONTROL and abs(self.inventory) > 0.0:
            inventory_norm = min(abs(self.inventory) / (self.config.MAX_INVENTORY_THRESHOLD / P_mid), 1.0)
            
            if (self.inventory > 0 and side == 'BUY') or \
               (self.inventory < 0 and side == 'SELL'):
                # Aggressive side (adds to existing inventory) -> Decrease size
                skew_factor = 1.0 - (inventory_norm * self.config.INVENTORY_SKEW_FACTOR)
            elif (self.inventory < 0 and side == 'BUY') or \
                 (self.inventory > 0 and side == 'SELL'):
                # Defensive side (reduces existing inventory) -> Increase size
                skew_factor = 1.0 + (inventory_norm * self.config.INVENTORY_SKEW_FACTOR)
        
        final_qty_usd = base_qty_usd * skew_factor
        
        qty = final_qty_usd / P_mid * self.config.LEVERAGE
        
        min_qty = 0.0001 # Binance futures minimum lot size for BTC
        return max(qty, min_qty)

    async def place_order(self, side: str, price: float):
        """Simulates async order placement."""
        qty = self.calculate_order_qty(side, price)
        order_id = f"ORDER_{side}_{int(time.time() * 1000)}"
        
        print(f"  [EXECUTION] Placing {side} order ID {order_id} at ${price:,.2f} (Qty: {qty:.4f} BTC)")
        
        # In a real environment, you'd use self.exchange.create_order_limit(symbol, side, qty, price)
        
        self.active_orders[order_id] = {"side": side, "price": price, "qty": qty, "status": "NEW"}

    async def cancel_all_orders(self):
        """Simulates async cancellation of all active limit orders."""
        print(f"  [EXECUTION] Cancelling {len(self.active_orders)} active orders.")
        # In a real environment, you'd use self.exchange.cancel_all_orders(self.config.TRADING_SYMBOL)
        self.active_orders.clear()

    def check_order_execution(self, last_trade_price: float):
        """
        Simulates checking if any limit orders in the grid were filled by the new price.
        NOTE: In a real bot, fills are confirmed via the User Data Stream, not by price checks.
        """
        filled_ids = []
        for order_id, order in list(self.active_orders.items()):
            
            is_filled = (order['side'] == 'BUY' and last_trade_price <= order['price']) or \
                        (order['side'] == 'SELL' and last_trade_price >= order['price'])
            
            if is_filled:
                filled_ids.append(order_id)
                print(f"  [FILL] Order {order_id} ({order['side']}) filled at ~${order['price']:,.2f}")
                
                inventory_change = order['qty'] if order['side'] == 'BUY' else -order['qty']
                self.inventory += inventory_change
                print(f"    [INVENTORY UPDATE] Net Inventory: {self.inventory:.4f} BTC")
                
                del self.active_orders[order_id]
                return True, order['side'], order['price']
        
        return False, None, None

    async def rebalance_grid(self, filled_side: str, filled_price: float):
        """
        After a fill, place the corresponding opposite order one grid step away (TP)
        and perform hard risk checks.
        """
        P_mid = self.grid_levels[len(self.grid_levels) // 2] if self.grid_levels else filled_price
        
        # --- Hard SL Check ---
        if filled_side == 'BUY' and filled_price < self.config.HARD_STOP_LOSS_PRICE:
            print(f"!!! HARD STOP LOSS TRIGGERED at ${filled_price:,.2f} !!!")
            # In a real system, you would market-sell to close the position.
            await self.cancel_all_orders()
            self.is_active = False
            return
        
        # --- Simple Rebalancing (Place counter-order) ---
        if filled_side == 'BUY':
            new_side = 'SELL'
            # Find the nearest grid level above the filled price to set the take-profit
            new_price = min([p for p in self.grid_levels if p > filled_price], default=P_mid)
        else: # filled_side == 'SELL'
            new_side = 'BUY'
            # Find the nearest grid level below the filled price to set the take-profit
            new_price = max([p for p in self.grid_levels if p < filled_price], default=P_mid)

        await self.place_order(new_side, new_price)

    async def initialize_grid(self, current_price: float):
        """Initial grid setup and parameter refresh."""
        
        if self.config.TRADING_SYMBOL in self.klines_data and self.klines_data[self.config.TRADING_SYMBOL]:
            returns_series = pd.Series([d['return'] for d in self.klines_data[self.config.TRADING_SYMBOL]])
            self.current_volatility = calculate_volatility(returns_series)
        else:
            self.current_volatility = 0.0

        self.grid_levels = generate_grid(self.config, current_price, self.current_volatility)
        
        print("\n--- GRID RE-INITIALIZED ---")
        print(f"Center Price: ${current_price:,.2f} | Volatility (sigma): {self.current_volatility:.5f}")
        
        await self.cancel_all_orders()
        
        mid_index = len(self.grid_levels) // 2
        
        # Place limit BUY orders on all levels below the midpoint
        for price in self.grid_levels[:mid_index]:
            await self.place_order('BUY', price)

        # Place limit SELL orders on all levels above the midpoint
        for price in self.grid_levels[mid_index:]:
            await self.place_order('SELL', price)
        
        print(f"Active Buy/Sell Orders: {len(self.active_orders)}")
        self.save_state()

    async def process_kline_data(self, symbol: str, kline: Dict[str, Any]):
        """
        Updates kline history and processes grid logic for the trading symbol
        when a new kline closes.
        """
        
        is_closed = kline['x'] # True if kline is closed
        
        close = float(kline['c'])
        open_ = float(kline['o'])
        ret = (close - open_) / open_ if open_ else 0.0
        
        # Update rolling kline history for the symbol
        if symbol not in self.klines_data:
             # Should not happen if initial fetch worked, but safe guard
            self.klines_data[symbol] = deque(maxlen=self.config.ROLLING_WINDOW)

        # Append new kline data if it's closed and not a duplicate
        if is_closed and (not self.klines_data[symbol] or self.klines_data[symbol][-1]['ts'] != kline['t']):
            self.klines_data[symbol].append({'close': close, 'return': ret, 'ts': kline['t']})
            print(f"[{symbol}] New {self.config.INTERVAL} kline closed at ${close:,.2f}. Ret: {ret*100:.3f}%")
            
            # --- ONLY run grid logic on the TRADING_SYMBOL ---
            if symbol == self.config.TRADING_SYMBOL:
                await self.handle_market_event(close)

    async def handle_market_event(self, current_price: float):
        """
        Main logic loop triggered by a closed kline for the trading symbol.
        """
        if not self.is_active:
            print(f"Grid is paused. Current Inventory: {self.inventory:.4f} BTC")
            return

        self.price_history.append(current_price)
        if len(self.price_history) > self.config.VELOCITY_CHECK_PERIOD:
            self.price_history.pop(0)

        # 1. Hard SL Check 
        if current_price < self.config.HARD_STOP_LOSS_PRICE:
            print(f"!!! HARD STOP LOSS TRIGGERED BY MARKET PRICE at ${current_price:,.2f} !!!")
            await self.cancel_all_orders()
            self.is_active = False
            self.save_state()
            return
        
        # 2. Velocity Filter Check
        if self.config.USE_VELOCITY_FILTER and velocity_filter_check(self.config, self.price_history):
            print("!!! VELOCITY FILTER TRIGGERED. PAUSING GRID AND CANCELLING ALL LIMITS. !!!")
            self.is_active = False
            await self.cancel_all_orders()
            self.save_state()
            return
            
        # 3. Order Execution Check (Updates inventory and rebalances)
        is_filled, filled_side, filled_price = self.check_order_execution(current_price)
        if is_filled:
            await self.rebalance_grid(filled_side, filled_price)
            
        # 4. Dynamic Grid Re-initialization (every time a kline closes, we recalculate volatility and potentially shift/re-size)
        print(f"\n[SYNC] Recalculating grid geometry at ${current_price:,.2f}...")
        self.last_sync_time = time.time()
        await self.initialize_grid(current_price)


    async def websocket_loop(self):
        """Connects to Binance WebSocket and processes real-time kline updates."""
        try:
            async with websockets.connect(self.stream_url) as websocket:
                print(f"--- CONNECTED to WebSocket: {self.stream_url} ---")
                
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)

                    if 'data' in data and 'k' in data['data']:
                        payload = data['data']
                        symbol = payload['s']
                        kline = payload['k']
                        
                        # Process both open (updates price history) and closed klines (triggers logic)
                        await self.process_kline_data(symbol, kline)
        
        except Exception as e:
            print(f"WebSocket error: {e}")
            await asyncio.sleep(5) # Wait before attempting reconnection
            await self.websocket_loop() # Simple reconnection logic

# --- 5. MAIN EXECUTION BLOCK ---

async def start_bot():
    """Starts the asynchronous trading bot."""
    
    # Initialize the Framework
    config = GridConfig()
    config.PERSISTENCE_MODE = "LOCAL_CACHE" # Default for testing
    framework = GridTradingFramework(config)

    # 1. Load previous state
    framework.load_state()

    # 2. Fetch historical data (updates INITIAL_PRICE and populates klines_data)
    await framework.fetch_initial_klines()

    # 3. Initialize the grid geometry based on the live price and initial volatility
    await framework.initialize_grid(framework.config.INITIAL_PRICE)

    # 4. Start the real-time WebSocket loop
    await framework.websocket_loop()


if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")