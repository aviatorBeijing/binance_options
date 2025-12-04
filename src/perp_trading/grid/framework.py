import json
import time
import math
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from asyncio import get_event_loop

# --- EXTERNAL LIBRARIES (Assumed installed for live operation) ---
import websockets
import ccxt
import pandas as pd
import numpy as np
# ----------------------------------------------------------------

# --- GLOBAL FIREBASE/ENV VARIABLES (Used for conceptual Firebase Mode) ---
# NOTE: These are defined as protected globals (single underscore) to prevent
# Python's name-mangling when referenced inside the GridConfig class.
_APP_ID = "quant-grid-btc-usdt"
_FIREBASE_CONFIG = '{"apiKey": "AIzaSy...", "authDomain": "...", "projectId": "..."}'
_INITIAL_AUTH_TOKEN = "some-jwt-token-if-available"
# -----------------------------------------------------------------------------

# --- 1. CONFIGURATION AND PARAMETERS ---
class GridConfig:
    """Stores all configurable parameters for the Live Grid Trading Framework."""
    def __init__(self):
        # --- CORE TRADING PARAMETERS ---
        # List of symbols to track klines for (used for multi-asset volatility calculation)
        self.SYMBOLS = ["SYSUSDT", "BTCUSDT", "ETHUSDT"] 
        # NOTE: Changing TRADING_SYMBOL must be reflected in LOCAL_CACHE_FILENAME
        self.MARKET_TYPE = 'perp'           # 'spot' or 'perp' (using Binance USDM futures)
        self.INTERVAL = '1m'                # Kline interval for volatility calculation
        self.ROLLING_WINDOW = 240           # 4 hours of 1m klines for volatility lookback
        self.TRADING_SYMBOL = self.SYMBOLS[0] # The specific symbol the bot will trade (BTCUSDT)
        
        self.LEVERAGE = 5.0
        self.MARGIN_MODE = "ISOLATED"
        self.GRID_COUNT = 20 # Number of buy/sell levels (10 Buy, 10 Sell)
        self.INITIAL_PRICE = 60000.0 # Placeholder, updated by live data on startup
        
        # --- NEW: TRADING MODE ---
        # Set to True for paper trading (simulated fills only, no CCXT trading)
        self.PAPER_TRADE = True
        self.TRADING_FEE_RATE = 0.0004 # 0.04% taker fee

        # --- PERSISTENCE SETTINGS ---
        # Options: 'LOCAL_CACHE' (default for safety/sim) or 'FIREBASE'
        self.PERSISTENCE_MODE = "LOCAL_CACHE"
        self.LOCAL_CACHE_DIR = "/tmp/binance_hft"
        # Adjusted filename to dynamically use the TRADING_SYMBOL
        self.LOCAL_CACHE_FILENAME = f"{self.TRADING_SYMBOL.lower()}_grid_state.json"
        
        # --- GRID GEOMETRY OPTIONS ---
        self.SPACING_METHOD = "GEOMETRIC" # GEOMETRIC or ARITHMETIC
        self.PRICE_RANGE_PERCENT = 0.02   # Fallback range if volatility is zero

        # --- RISK & SIZING ---
        self.BASE_QTY_USD = 100 # Base USD value for each grid order
        self.HARD_STOP_LOSS_PRICE = self.INITIAL_PRICE * 0.95 

        # --- NEW CAPITAL MANAGEMENT PARAMETERS ---
        self.TOTAL_AVAILABLE_CAPITAL_USD = 10000.0 # Total account equity (assumed for risk calc)
        # Maximum fraction of capital allowed to be used (notional, including leverage)
        self.MAX_CAPITAL_ALLOCATION_PERCENT = 0.50 # 50% max allocation
        
        # Absolute maximum notional USD value for the open position (inventory * price * leverage)
        self.MAX_POSITION_USD = self.TOTAL_AVAILABLE_CAPITAL_USD * self.MAX_CAPITAL_ALLOCATION_PERCENT 
        # ----------------------------------------------------------------------------------

        # --- DYNAMIC VOLATILITY ADJUSTMENT ---
        self.USE_DYNAMIC_VOLATILITY = True
        self.VOLATILITY_SCALING_FACTOR = 1.5 # Multiplies sigma to set grid step size

        # --- VOLATILITY-ADJUSTED SIZING ---
        self.USE_VOLATILITY_SIZING = True
        self.MIN_VOLATILITY_FOR_SIZING = 0.0001
        # Factor used to normalize volatility for sizing. Tune this based on expected sigma magnitude.
        # Example: If sigma is 0.0001, a factor of 1000 makes the ratio 1.0.
        self.VOLATILITY_SIZE_FACTOR = 1000.0 

        # --- INVENTORY CONTROL ---
        self.USE_INVENTORY_CONTROL = True
        self.INVENTORY_SKEW_FACTOR = 0.25
        self.MAX_INVENTORY_THRESHOLD = 500.0 # Maximum allowed inventory in USD value

        # --- VELOCITY FILTER (Momentum/Trend Risk) ---
        self.USE_VELOCITY_FILTER = True
        self.CRITICAL_VELOCITY_THRESHOLD = 0.005 # Price change percentage over CHECK_PERIOD
        self.VELOCITY_CHECK_PERIOD = 3 # Number of klines to check velocity over

        # --- FIREBASE & PERSISTENCE (Production Only) ---
        self.APP_ID = _APP_ID
        self.COLLECTION_NAME = "grid_states"

    def to_dict(self):
        # Convert necessary attributes for serialization
        d = self.__dict__.copy()
        d['SYMBOLS'] = self.SYMBOLS
        return d

# --- 2. FIREBASE UTILITIES (Conceptual Imports for Production Mode) ---
def get_db_and_auth():
    """Simulates Firebase initialization and authentication (Production only)."""
    print("--- Initializing Firebase and Auth (Conceptual/Production Mode) ---")
    # In a real application, Firebase initialization using _FIREBASE_CONFIG 
    # and sign-in using _INITIAL_AUTH_TOKEN would occur here.
    return "mock_db_instance", "mock_user_id"

def get_firestore_path(app_id: str, user_id: str, collection: str, doc_id: str) -> str:
    """Generates the Firestore path for private user data (Production only)."""
    return f"/artifacts/{app_id}/users/{user_id}/{collection}/{doc_id}"

# --- 3. QUANTITATIVE STRATEGY LOGIC ---

def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculates the standard deviation of returns (sigma) for the rolling window.
    """
    if len(returns) < 2:
        return 0.0
    # Standard deviation of returns is a standard measure of local volatility
    return float(returns.std())

def generate_grid(config: GridConfig, current_price: float, volatility: float) -> List[float]:
    """
    Generates dynamic grid levels based on configuration and calculated volatility.
    
    """
    grid_levels = []
    P_mid = current_price
    N = config.GRID_COUNT // 2

    if config.USE_DYNAMIC_VOLATILITY and volatility > 0:
        # Step size is proportional to market volatility
        base_step_ratio = volatility * config.VOLATILITY_SCALING_FACTOR
        print(f"   [Dynamic] Base Step Ratio (sigma * k): {base_step_ratio:.5f}")
    else:
        # Fallback to static percentage spacing if dynamic volatility is off or zero
        base_step_ratio = config.PRICE_RANGE_PERCENT / N

    if config.SPACING_METHOD == "ARITHMETIC":
        base_step = base_step_ratio * P_mid
        for i in range(1, N + 1):
            grid_levels.append(P_mid - (i * base_step)) # Buy levels
            grid_levels.append(P_mid + (i * base_step)) # Sell levels
    else: # GEOMETRIC (compounding steps, default)
        ratio = 1 + base_step_ratio
        for i in range(1, N + 1):
            grid_levels.append(P_mid / (ratio ** i)) # Buy levels (decreasing ratio)
            grid_levels.append(P_mid * (ratio ** i)) # Sell levels (increasing ratio)

    grid_levels.sort()
    return grid_levels

def velocity_filter_check(config: GridConfig, price_history: List[float]) -> bool:
    """
    Implements the 'Velocity Filter' to detect high momentum (trending market) 
    and pause grid trading to avoid heavy inventory accumulation on one side.
    """
    if not config.USE_VELOCITY_FILTER or len(price_history) < config.VELOCITY_CHECK_PERIOD:
        return False

    recent_prices = price_history[-config.VELOCITY_CHECK_PERIOD:]
    P_start = recent_prices[0]
    P_end = recent_prices[-1]

    # Relative Price Change (RPC) over the period
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
        self.price_history: List[float] = [] 
        self.is_active = True
        self.last_sync_time = 0
        self.inventory: float = 0.0 # Net position (positive=LONG, negative=SHORT)
        self.cost_basis_usd: float = 0.0 # Total USD value of the current net position
        self.total_pnl_usd: float = 0.0
        self.total_fees_usd: float = 0.0
        self.filled_orders: List[Dict] = []
        self.current_volatility: float = 0.0
        # Stores historical klines for volatility calculation
        self.klines_data: Dict[str, deque] = {}

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
        """
        Loads persistent state from the local JSON cache and includes sanitization
        for corrupted/massive inventory and P&L data.
        """
        print(f"Attempting to load state from local file: {self.cache_filepath}...")
        try:
            with open(self.cache_filepath, 'r') as f:
                state = json.load(f)
                self.grid_levels = state.get('grid_levels', [])
                self.active_orders = state.get('active_orders', {})
                self.is_active = state.get('is_active', True)
                
                # --- START SANITIZATION LOGIC (Addressing the log anomalies) ---
                
                # Use a small threshold (e.g., 10000 units) to flag corrupted inventory 
                # This assumes your trading quantity is usually much less than 10k units.
                MAX_SANITY_QTY = 10000.0
                
                # 1. Sanitize Inventory
                loaded_inventory = state.get('inventory', 0.0)
                if abs(loaded_inventory) > MAX_SANITY_QTY:
                    print(f"!!! WARNING: Loaded inventory ({loaded_inventory:.4f}) is massive. RESETTING to 0.0")
                    self.inventory = 0.0
                    self.cost_basis_usd = 0.0
                else:
                    self.inventory = loaded_inventory
                    # 2. Sanitize Cost Basis (must be loaded along with inventory)
                    self.cost_basis_usd = state.get('cost_basis_usd', 0.0)

                # 3. Sanitize Filled Orders (check for non-numeric/corrupted entries that caused crash)
                loaded_filled_orders = state.get('filled_orders', [])
                sanitized_filled_orders = []
                for order in loaded_filled_orders:
                    try:
                        # Attempt to convert key numerical fields to float, discarding corrupted ones
                        order['price'] = float(order.get('price', 0.0))
                        order['qty'] = float(order.get('qty', 0.0))
                        order['fee'] = float(order.get('fee', 0.0))
                        sanitized_filled_orders.append(order)
                    except (ValueError, TypeError):
                        print("!!! WARNING: Skipping corrupted order history entry.")
                        continue
                self.filled_orders = sanitized_filled_orders
                
                # 4. Load P&L and fees after sanitizing filled orders
                self.total_pnl_usd = state.get('total_pnl_usd', 0.0)
                self.total_fees_usd = state.get('total_fees_usd', 0.0)
                
                # --- END SANITIZATION LOGIC ---
                
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
            # NEW: Save P&L state
            "inventory": self.inventory,
            "cost_basis_usd": self.cost_basis_usd,
            "total_pnl_usd": self.total_pnl_usd,
            "total_fees_usd": self.total_fees_usd,
            "filled_orders": self.filled_orders,

            "volatility": self.current_volatility
        }
        
        os.makedirs(self.config.LOCAL_CACHE_DIR, exist_ok=True)
        
        try:
            with open(self.cache_filepath, 'w') as f:
                json.dump(current_state, f, indent=4)
            print(f"State saved to local cache at {time.strftime('%H:%M:%S')} | Inventory: {self.inventory:.4f} {self.config.TRADING_SYMBOL[:3]}")
        except IOError as e:
            print(f"Error saving state to local cache: {e}")

    def _load_state_firebase(self):
        """Simulates loading persistent state from Firestore (Production only)."""
        # Actual Firestore retrieval logic using self.state_path would be here
        pass

    def _save_state_firebase(self):
        """Simulates saving the current state to Firestore (Production only)."""
        # Actual Firestore write logic using self.state_path would be here
        print(f"State saved to Firestore at {time.strftime('%H:%M:%S')} | Inventory: {self.inventory:.4f} {self.config.TRADING_SYMBOL[:3]}")
    
    # --- CCXT DATA FETCHING ---
    async def fetch_initial_klines(self):
        """Fetch historical data for all symbols using CCXT (REST)."""
        print("Fetching historical data...")
        self.klines_data = {}
        limit = self.config.ROLLING_WINDOW - 1
        
        # Use asyncio.to_thread to safely run synchronous CCXT calls
        def sync_fetch(symbol):
            # Fetch OHLCV data: [ts, open, high, low, close, volume]
            return self.exchange.fetch_ohlcv(symbol, timeframe=self.config.INTERVAL, limit=limit)

        for symbol in self.config.SYMBOLS:
            try:
                # --- MODIFIED LINE START ---
                # Get the current running event loop
                loop = get_event_loop()
                # Use loop.run_in_executor to run the blocking sync_fetch in a thread pool
                ohlcv = await loop.run_in_executor(None, sync_fetch, symbol)
                # --- MODIFIED LINE END ---
                
                deque_data = deque(maxlen=self.config.ROLLING_WINDOW)
                for o in ohlcv:
                    ts, open_, _, _, close, _ = o
                    ret = (close - open_) / open_ if open_ else 0.0
                    deque_data.append({'close': close, 'return': ret, 'ts': ts})
                
                self.klines_data[symbol] = deque_data
                print(f"  Loaded {len(ohlcv)} klines for {symbol}.")
                
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
                
        # Set the initial price for the trading symbol
        if self.config.TRADING_SYMBOL in self.klines_data and self.klines_data[self.config.TRADING_SYMBOL]:
            last_close = self.klines_data[self.config.TRADING_SYMBOL][-1]['close']
            self.config.INITIAL_PRICE = last_close
            # Update hard stop loss based on current price
            self.config.HARD_STOP_LOSS_PRICE = last_close * 0.95
            print(f"  Initial price set to: ${last_close:,.2f}")
        else:
            raise RuntimeError(f"Failed to load data for trading symbol {self.config.TRADING_SYMBOL}")


    # --- TRADING LOGIC IMPLEMENTATION (Simulated) ---

    def calculate_order_qty(self, side: str, price: float) -> float:
        """
        Calculates the quantity applying Volatility-Adjusted Sizing and Inventory Skew,
        and enforces the MAX_POSITION_USD capital limit.
        """
        P_mid = price
        
        # 1. Volatility-Adjusted Base Size (in USD)
        base_qty_usd = self.config.BASE_QTY_USD
        
        if self.config.USE_VOLATILITY_SIZING and self.current_volatility > self.config.MIN_VOLATILITY_FOR_SIZING:
            # --- FIX APPLIED HERE ---
            # The calculation was replacing base_qty_usd with the factor, leading to exponential numbers.
            # FIX: Use the volatility calculation as a multiplier (scaling factor) for the BASE_QTY_USD.
            
            # The scaling ratio: (Target Volatility / Current Volatility) 
            # We use the VOLATILITY_SIZE_FACTOR * Price to create a stable, unit-less target.
            
            # Example: If sigma is 0.0001 (low) and price is 100, the divisor is 0.01.
            # If Factor is 1000, ratio is 1000 / 0.01 = 100, which is too aggressive.
            
            # Let's adjust the formula to create a simple inverse scaling:
            # We want size to INCREASE as volatility DECREASES.
            
            # Calculate a stable target volatility reference.
            target_vol_ref = self.config.VOLATILITY_SIZE_FACTOR / P_mid
            
            # Scaling Factor: If current_volatility is low (e.g., 0.0001), the factor is high.
            # We normalize against a small target volatility reference.
            scaling_factor = target_vol_ref / self.current_volatility
            
            # Cap the scaling factor to prevent massive order sizes during extreme low volatility
            MAX_SCALING_FACTOR = 3.0 
            scaling_factor = min(scaling_factor, MAX_SCALING_FACTOR)
            
            # Apply the scaling factor to the base USD quantity
            base_qty_usd *= scaling_factor
            
            print(f"    [SIZE ADJUST] Volatility Scaling Factor: {scaling_factor:.2f}. New Base USD: ${base_qty_usd:.2f}")


        # 2. Inventory Sizing Skew (Adjusts size based on current net position)
        skew_factor = 1.0
        if self.config.USE_INVENTORY_CONTROL and abs(self.inventory) > 0.0:
            # Normalize inventory against max threshold (must convert MAX_INVENTORY_THRESHOLD from USD to QTY)
            max_inventory_qty = self.config.MAX_INVENTORY_THRESHOLD / P_mid 
            inventory_norm = min(abs(self.inventory) / max_inventory_qty, 1.0)
            
            if (self.inventory > 0 and side == 'BUY') or \
               (self.inventory < 0 and side == 'SELL'):
                # Aggressive side (adds to existing position) -> Decrease size
                skew_factor = 1.0 - (inventory_norm * self.config.INVENTORY_SKEW_FACTOR)
            elif (self.inventory < 0 and side == 'BUY') or \
                 (self.inventory > 0 and side == 'SELL'):
                # Defensive side (reduces existing position) -> Increase size
                skew_factor = 1.0 + (inventory_norm * self.config.INVENTORY_SKEW_FACTOR)
        
        final_qty_usd = base_qty_usd * skew_factor
        
        # Quantity calculation: USD Value * Leverage / Price (This is the desired quantity)
        desired_qty = final_qty_usd / P_mid * self.config.LEVERAGE
        
        # --- NEW: CAPITAL/MAX POSITION LIMIT CHECK ---
        
        # Calculate the current notional exposure (absolute value)
        current_notional = abs(self.inventory * P_mid * self.config.LEVERAGE)
        max_notional = self.config.MAX_POSITION_USD
        
        # 1. Determine if the new order is aggressive (adds to the current position's magnitude)
        is_aggressive_side = (self.inventory >= 0 and side == 'BUY') or \
                             (self.inventory <= 0 and side == 'SELL')

        qty = desired_qty # Start with the desired quantity

        if is_aggressive_side:
            # 2. Calculate the capacity remaining in USD notional
            # Note: We are using math.copysign for consistency with inventory sign, 
            # but since we use abs(inventory) for current_notional, this is simplified.
            capacity_remaining_notional = max(0.0, max_notional - current_notional)
            
            # 3. Convert remaining notional capacity to quantity
            # Max quantity we can place without exceeding the limit: Notional / (Price * Leverage)
            # Safe division check (in case Price or Leverage is zero, though unlikely)
            divisor = (P_mid * self.config.LEVERAGE)
            max_add_qty_allowed = capacity_remaining_notional / divisor if divisor else 0.0
            
            # 4. Cap the desired quantity
            qty = min(desired_qty, max_add_qty_allowed)
            
            if qty < desired_qty:
                # The logged message was missing the TRADING_SYMBOL
                print(f"    [CAPITAL LIMIT] Capping {side} order from {desired_qty:.4f} qty to {qty:.4f} qty due to MAX_POSITION_USD limit.")
        
        # Final check for minimum quantity
        min_qty = 0.0001
        # If the capped quantity is less than the min_qty, it means we cannot place a meaningful order 
        # without violating the capital limit, so we return 0.0 to signal skipping the order.
        if qty < min_qty:
            return 0.0
            
        return qty # Return the capped, non-zero quantity

    async def place_order(self, side: str, price: float):
        """Simulates async order placement. Uses CCXT if PAPER_TRADE is False."""
        qty = self.calculate_order_qty(side, price)
        # Prevent placing an order if quantity is effectively zero due to the cap
        if qty == 0.0:
            print(f"  [EXECUTION] Skipped {side} order at ${price:,.2f}. Quantity too low (capped to zero).")
            return

        order_id = f"ORDER_{side}_{int(time.time() * 1000)}"
        
        print(f"  [EXECUTION {'PAPER' if self.config.PAPER_TRADE else 'LIVE'}] Placing {side} order ID {order_id} at ${price:,.2f} (Qty: {qty:.4f} {self.config.TRADING_SYMBOL[:3]})")
        
        if not self.config.PAPER_TRADE:
            # Real Call: await self.exchange.create_order_limit(self.config.TRADING_SYMBOL, side, qty, price)
            pass
        
        self.active_orders[order_id] = {"side": side, "price": price, "qty": qty, "status": "NEW"}

    async def cancel_all_orders(self):
        """Simulates async cancellation of all active limit orders."""
        print(f"  [EXECUTION] Cancelling {len(self.active_orders)} active orders.")
        
        if not self.config.PAPER_TRADE:
            # Real Call: await self.exchange.cancel_all_orders(self.config.TRADING_SYMBOL)
            pass
            
        self.active_orders.clear()

    def _process_fill_pnl(self, filled_side: str, fill_price: float, fill_qty: float):
        """
        Calculates and updates realized P&L based on the fill.
        Uses Average Cost Basis tracking for the open position.
        """
        trade_usd_value = fill_qty * fill_price
        
        # Calculate fee
        fee_usd = trade_usd_value * self.config.TRADING_FEE_RATE
        self.total_fees_usd += fee_usd
        
        # 1. Opening/Adding to Position (e.g., Inventory 0 -> Buy, or Inventory > 0 -> Buy)
        if (self.inventory >= 0 and filled_side == 'BUY') or \
           (self.inventory <= 0 and filled_side == 'SELL'):
            
            # Update cost basis using weighted average (assuming the trade is fully opening/adding)
            if filled_side == 'BUY':
                # New cost basis = (Old cost + New USD value)
                self.cost_basis_usd += trade_usd_value
                self.inventory += fill_qty
            else: # SELL (adding to short position - cost basis decreases in magnitude)
                # The cost basis for a short is how much USD was received.
                self.cost_basis_usd -= trade_usd_value # Negative value means money received
                self.inventory -= fill_qty
            
            print(f"    [P&L] Adding to position. New Inv: {self.inventory:.4f}, Cost Basis: ${self.cost_basis_usd:,.2f}")

        # 2. Closing/Reducing Position (e.g., Inventory > 0 -> Sell, or Inventory < 0 -> Buy)
        else: 
            # This fill reduces the current net position magnitude. Realized P&L calculation.
            
            realized_pnl = 0.0
            
            # Long position being reduced by a SELL
            if self.inventory > 0 and filled_side == 'SELL':
                # Check for zero inventory before division (safety)
                if abs(self.inventory) < 0.000001:
                    avg_entry_price = 0.0
                else:
                    avg_entry_price = self.cost_basis_usd / self.inventory
                    
                # PnL = (Exit Price - Entry Price) * Quantity
                realized_pnl = (fill_price - avg_entry_price) * fill_qty
                
                # Adjust cost basis (remove the cost associated with the closed quantity)
                cost_of_closed_qty = avg_entry_price * fill_qty
                self.cost_basis_usd -= cost_of_closed_qty
                self.inventory -= fill_qty
                
            # Short position being reduced by a BUY
            elif self.inventory < 0 and filled_side == 'BUY':
                # Check for zero inventory before division (safety)
                if abs(self.inventory) < 0.000001:
                    avg_entry_price = 0.0
                else:
                    # Avg cost basis for short is negative (USD received) / negative qty
                    avg_entry_price = abs(self.cost_basis_usd / self.inventory)
                    
                # PnL = (Entry Price - Exit Price) * Quantity
                realized_pnl = (avg_entry_price - fill_price) * fill_qty

                # Adjust cost basis (reduce the liability/money received associated with the closed quantity)
                # Cost basis is negative (money received). The closed portion is: avg_entry_price * fill_qty
                cost_of_closed_qty = -(avg_entry_price * fill_qty) # Convert to USD received equivalent
                self.cost_basis_usd -= cost_of_closed_qty
                self.inventory += fill_qty
            
            # Net P&L (minus fees)
            realized_pnl -= fee_usd
            self.total_pnl_usd += realized_pnl
            
            print(f"    [P&L] **Realized PnL: ${realized_pnl:,.2f}** | Total PnL: ${self.total_pnl_usd:,.2f}")

        # Record the filled order history (optional, for detailed logging)
        self.filled_orders.append({
            "ts": time.time(), "side": filled_side, "price": fill_price, "qty": fill_qty, 
            "fee": fee_usd
        })

    def check_order_execution(self, last_trade_price: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Simulates checking for order fills by comparing the latest price to limit order levels.
        """
        for order_id, order in list(self.active_orders.items()):
            
            # Check for cross-fill (price moves to or beyond the order level)
            is_filled = (order['side'] == 'BUY' and last_trade_price <= order['price']) or \
                        (order['side'] == 'SELL' and last_trade_price >= order['price'])
            
            if is_filled:
                # Use the order's price and quantity for the fill
                fill_price = order['price']
                fill_qty = order['qty']
                
                print(f"  [FILL] Order {order_id} ({order['side']}) filled at ~${fill_price:,.2f}")
                
                # --- NEW: P&L and Inventory Calculation ---
                self._process_fill_pnl(order['side'], fill_price, fill_qty)
                
                del self.active_orders[order_id]
                return True, order['side'], fill_price
        
        return False, None, None

    async def rebalance_grid(self, filled_side: str, filled_price: float):
        """
        After a fill, perform risk checks and place the corresponding opposite order (Take Profit).
        """
        P_mid = self.config.INITIAL_PRICE
        
        # --- Hard SL Check ---
        # Note: inventory < 0 means net short position. We check if the price dumps (goes lower)
        if self.inventory < 0 and filled_price < self.config.HARD_STOP_LOSS_PRICE: 
            print(f"!!! HARD STOP LOSS TRIGGERED (Short position vs low price) at ${filled_price:,.2f} !!!")
            await self.cancel_all_orders()
            self.is_active = False
            return
        # Also check for long position vs price drop
        elif self.inventory > 0 and filled_price < self.config.HARD_STOP_LOSS_PRICE:
            print(f"!!! HARD STOP LOSS TRIGGERED (Long position vs low price) at ${filled_price:,.2f} !!!")
            await self.cancel_all_orders()
            self.is_active = False
            return
        
        # --- Simple Rebalancing (Place counter-order for Take Profit) ---
        if filled_side == 'BUY':
            new_side = 'SELL'
            # Find the nearest grid level above the filled price to set the take-profit
            new_price = min([p for p in self.grid_levels if p > filled_price], default=P_mid)
        else: # filled_side == 'SELL'
            new_side = 'BUY'
            # Find the nearest grid level below the filled price to set the take-profit
            new_price = max([p for p in self.grid_levels if p < filled_price], default=P_mid)

        # Place the new limit order
        await self.place_order(new_side, new_price)

    async def initialize_grid(self, current_price: float):
        """Initial grid setup and parameter refresh."""
        
        # Calculate volatility based on the trading symbol's returns history
        if self.config.TRADING_SYMBOL in self.klines_data and self.klines_data[self.config.TRADING_SYMBOL]:
            returns_series = pd.Series([d['return'] for d in self.klines_data[self.config.TRADING_SYMBOL]])
            self.current_volatility = calculate_volatility(returns_series)
        else:
            self.current_volatility = 0.0

        self.grid_levels = generate_grid(self.config, current_price, self.current_volatility)
        
        print("\n--- GRID RE-INITIALIZED ---")
        print(f"Center Price: ${current_price:,.2f} | Volatility (sigma): {self.current_volatility:.5f}")
        
        # --- NEW: P&L Report on Adjustment ---
        print(f"--- P&L REPORT ---")
        print(f"REALIZED P&L: ${self.total_pnl_usd:,.2f}")
        print(f"TOTAL FEES: ${self.total_fees_usd:,.2f}")
        
        # Check if we have an open position before calculating U-PnL
        if abs(self.inventory) > 0.000001:
            # Safe division check
            avg_price = self.cost_basis_usd / self.inventory if abs(self.inventory) > 0.000001 else 0.0
            
            # Unrealized P&L calculation
            # For LONG (inventory > 0): (Current Price - Avg Price) * Quantity
            # For SHORT (inventory < 0): (Avg Price - Current Price) * |Quantity|
            if self.inventory > 0:
                unrealized_pnl = (current_price - avg_price) * self.inventory
            else:
                unrealized_pnl = (abs(avg_price) - current_price) * abs(self.inventory)

            print(f"OPEN POSITION: {self.inventory:.4f} @ ${abs(avg_price):,.2f}")
            print(f"UNREALIZED P&L: ${unrealized_pnl:,.2f}")
        else:
             print(f"OPEN POSITION: 0.0000 @ $0.00")
             print(f"UNREALIZED P&L: $0.00")
             
        print("------------------")
        
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
        Updates kline history from WebSocket and processes grid logic on kline close.
        """
        is_closed = kline['x'] # True if kline is closed
        
        # Extract data
        close = float(kline['c'])
        open_ = float(kline['o'])
        ret = (close - open_) / open_ if open_ else 0.0
        
        # Ensure kline history is initialized for this symbol
        if symbol not in self.klines_data:
            self.klines_data[symbol] = deque(maxlen=self.config.ROLLING_WINDOW)

        # Append new kline data if it's closed and not a duplicate
        if is_closed and (not self.klines_data[symbol] or self.klines_data[symbol][-1]['ts'] != kline['t']):
            self.klines_data[symbol].append({'close': close, 'return': ret, 'ts': kline['t']})
            
            # --- ONLY run grid logic on the TRADING_SYMBOL when its kline closes ---
            if symbol == self.config.TRADING_SYMBOL:
                print(f"[{symbol}] New {self.config.INTERVAL} kline closed at ${close:,.2f}. Ret: {ret*100:.3f}%")
                await self.handle_market_event(close)

    async def handle_market_event(self, current_price: float):
        """
        Main logic loop triggered by a closed kline for the trading symbol.
        """
        if not self.is_active:
            # Re-entry condition: Inventory flat and trend risk (velocity) is clear
            if abs(self.inventory) < 0.0001 and not velocity_filter_check(self.config, self.price_history):
                print("\n[RESUME] Paused state cleared. Re-initializing grid and resuming.")
                self.is_active = True
                await self.initialize_grid(current_price)
            else:
                return

        # Track recent prices for velocity check
        self.price_history.append(current_price)
        if len(self.price_history) > self.config.VELOCITY_CHECK_PERIOD:
            self.price_history.pop(0)

        # 1. Hard SL Check (Check if price has dropped below the safety price)
        if current_price < self.config.HARD_STOP_LOSS_PRICE:
            print(f"!!! HARD STOP LOSS TRIGGERED BY MARKET PRICE at ${current_price:,.2f} !!!")
            await self.cancel_all_orders()
            self.is_active = False
            self.save_state()
            return
        
        # 2. Velocity Filter Check (Check for trending/momentum market)
        if self.config.USE_VELOCITY_FILTER and velocity_filter_check(self.config, self.price_history):
            print("!!! VELOCITY FILTER TRIGGERED. PAUSING GRID AND CANCELLING ALL LIMITS. !!!")
            self.is_active = False
            await self.cancel_all_orders()
            self.save_state()
            return
            
        # 3. Order Execution Check (Updates inventory, cost basis, P&L, and triggers rebalancing)
        is_filled, filled_side, filled_price = self.check_order_execution(current_price)
        if is_filled:
            await self.rebalance_grid(filled_side, filled_price)
            
        # 4. Dynamic Grid Re-initialization (Recalculate volatility and re-place orders)
        print(f"\n[SYNC] Recalculating grid geometry at ${current_price:,.2f}...")
        self.last_sync_time = time.time()
        await self.initialize_grid(current_price)


    async def websocket_loop(self):
        """Connects to Binance WebSocket and processes real-time kline updates."""
        while True:
            try:
                # Use exponential backoff for reconnection
                reconnect_delay = 1
                async with websockets.connect(self.stream_url) as websocket:
                    print(f"\n--- CONNECTED to WebSocket: {self.stream_url} ---")
                    reconnect_delay = 1 # Reset delay on successful connection
                    
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)

                        if 'data' in data and 'k' in data['data']:
                            payload = data['data']
                            symbol = payload['s']
                            kline = payload['k']
                            
                            await self.process_kline_data(symbol, kline)
            
            except websockets.exceptions.ConnectionClosedOK:
                print("WebSocket connection closed gracefully. Attempting to reconnect...")
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting in {reconnect_delay} seconds...")
            
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60) # Exponential backoff up to 60s


async def start_bot():
    """Starts the asynchronous trading bot lifecycle."""
    
    config = GridConfig()
    framework = GridTradingFramework(config)

    # 1. Load previous state
    framework.load_state()

    # 2. Fetch historical data (sets the current market price and initial volatility)
    await framework.fetch_initial_klines()

    # 3. Initialize the grid geometry
    await framework.initialize_grid(framework.config.INITIAL_PRICE)

    # 4. Start the real-time WebSocket loop
    await framework.websocket_loop()


if __name__ == "__main__":
    try:
        # Standard way to run asynchronous code
        if hasattr(asyncio, 'run'):
            asyncio.run(start_bot())
        else:
            # Fallback for older versions 
            loop = asyncio.get_event_loop()
            loop.run_until_complete(start_bot())
    except KeyboardInterrupt:
        print("\nBot stopped by user. Final state saved.")
        # Note: Proper state saving on interrupt would require a handler accessing the framework instance
    except Exception as e:
        print(f"\nAn unhandled error occurred: {e}")