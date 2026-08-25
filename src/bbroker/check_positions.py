import asyncio
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode
import aiohttp

# Retrieve environment variables
API_KEY = os.getenv("BINANCE_MAIN_OPTIONS_APIKEY", None)
SECRET_KEY = os.getenv("BINANCE_MAIN_OPTIONS_SECRET", None)

BASE_URL = "https://eapi.binance.com"


def generate_signature(params: dict, secret: str) -> str:
    """Generates HMAC-SHA256 signature for Binance API parameters."""
    query_string = urlencode(params)
    return hmac.new(
        secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


async def fetch_signed(
    session: aiohttp.ClientSession, endpoint: str, params: dict = None
) -> dict:
    """Executes an authenticated GET request against Binance Options API."""
    if params is None:
        params = {}

    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params["signature"] = generate_signature(params, SECRET_KEY)

    headers = {"X-MBX-APIKEY": API_KEY}
    url = f"{BASE_URL}{endpoint}"

    async with session.get(url, headers=headers, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def fetch_public(
    session: aiohttp.ClientSession, endpoint: str, params: dict = None
) -> dict:
    """Executes an unauthenticated GET request."""
    url = f"{BASE_URL}{endpoint}"
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def get_positions_and_pnl():
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "API Key and Secret must be provided via environment variables."
        )

    async with aiohttp.ClientSession() as session:
        # 1. Concurrently fetch account positions and top order book tickers
        positions_task = fetch_signed(session, "/eapi/v1/position")
        tickers_task = fetch_public(session, "/eapi/v1/ticker")

        positions_raw, tickers_raw = await asyncio.gather(
            positions_task, tickers_task
        )

        # Filter for non-zero position sizes
        active_positions = [
            p for p in positions_raw if float(p.get("quantity", 0)) != 0
        ]

        if not active_positions:
            print("No active Binance options positions found.")
            return

        # Map tickers by symbol for O(1) lookups
        ticker_map = {t["symbol"]: t for t in tickers_raw}

        total_unrealized_pnl = 0.0
        total_liquidation_pnl = 0.0

        print(
            f"{'Symbol':<25} {'Qty':<8} {'Entry Price':<12} {'Mark Price':<12} {'Best Bid':<10} {'Best Ask':<10} {'Unrealized PnL':<15} {'Liquidation PnL':<15}"
        )
        print("-" * 110)

        for pos in active_positions:
            symbol = pos["symbol"]
            qty = float(pos["quantity"])
            entry_price = float(pos["entryPrice"])
            mark_price = float(pos.get("markPrice", 0.0))

            ticker = ticker_map.get(symbol, {})
            best_bid = float(ticker.get("bidPrice", 0.0))
            best_ask = float(ticker.get("askPrice", 0.0))

            # 1. Standard Mark Price Unrealized PnL
            unrealized_pnl = (mark_price - entry_price) * qty
            total_unrealized_pnl += unrealized_pnl

            # 2. Virtual Liquidation PnL (Sell at Bid / Buy at Ask)
            if qty > 0:  # Long position: Sell at Best Bid
                liq_pnl = (best_bid - entry_price) * qty
            else:  # Short position: Buy back at Best Ask
                liq_pnl = (entry_price - best_ask) * abs(qty)

            total_liquidation_pnl += liq_pnl

            print(
                f"{symbol:<25} {qty:<8.2f} {entry_price:<12.2f} {mark_price:<12.2f} {best_bid:<10.2f} {best_ask:<10.2f} {unrealized_pnl:<15.2f} {liq_pnl:<15.2f}"
            )

        print("-" * 110)
        print(f"Total Mark-to-Market Unrealized PnL: ${total_unrealized_pnl:,.2f}")
        print(f"Total Net Liquidation PnL (Slippage/Spread Applied): ${total_liquidation_pnl:,.2f}")


if __name__ == "__main__":
    asyncio.run(get_positions_and_pnl())
