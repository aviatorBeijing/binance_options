from __future__ import annotations
import asyncio
import hashlib
import hmac
import os
import re
import time
from urllib.parse import urlencode
import aiohttp

API_KEY = os.getenv("BINANCE_MAIN_OPTIONS_APIKEY", None)
SECRET_KEY = os.getenv("BINANCE_MAIN_OPTIONS_SECRET", None)

OPTIONS_BASE_URL = "https://eapi.binance.com"
SPOT_BASE_URL = "https://api.binance.com"


def generate_signature(params: dict, secret: str) -> str:
    """Generates HMAC-SHA256 signature for Binance API parameters."""
    query_string = urlencode(params)
    return hmac.new(
        secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


async def fetch_options_signed(
    session: aiohttp.ClientSession, endpoint: str, params: dict = None
) -> dict:
    """Executes an authenticated GET request against Binance Options API."""
    if params is None:
        params = {}

    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params["signature"] = generate_signature(params, SECRET_KEY)

    headers = {"X-MBX-APIKEY": API_KEY}
    url = f"{OPTIONS_BASE_URL}{endpoint}"

    async with session.get(url, headers=headers, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def fetch_options_public(
    session: aiohttp.ClientSession, endpoint: str, params: dict = None
) -> dict:
    """Executes an unauthenticated GET request against Binance Options API."""
    url = f"{OPTIONS_BASE_URL}{endpoint}"
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def fetch_spot_price(session: aiohttp.ClientSession, symbol: str) -> float:
    """Fetches exact real-time spot price from Binance Spot API (/api/v3/ticker/price)."""
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    url = f"{SPOT_BASE_URL}/api/v3/ticker/price"
    params = {"symbol": symbol}

    async with session.get(url, headers=headers, params=params) as response:
        response.raise_for_status()
        data = await response.json()
        return float(data["price"])


def parse_option_symbol(symbol: str):
    """Extracts underlying asset, expiry, strike price, and option type ('C' or 'P')."""
    match = re.search(r"^([A-Z]+)-(\d+)-(\d+(?:\.\d+)?)-([CP])$", symbol)
    if not match:
        raise ValueError(f"Could not parse option symbol: {symbol}")
    underlying = match.group(1)
    expiry = match.group(2)
    strike = float(match.group(3))
    opt_type = match.group(4)
    return underlying, expiry, strike, opt_type


def compute_portfolio_expiry_spot_targets_cost_basis(
    positions: list, percentages: list, current_spot: float
) -> tuple[dict, float, float]:
    """Calculates required underlying spot prices at expiration for combined portfolio value gains
    based on initial **COST BASIS** (Entry Price), comparing targets against current spot price.
    Returns calculated spot targets, individual contract intrinsic prices at those targets, total cost, and total mark.
    """
    total_cost_basis = sum(p["entry_val"] for p in positions)
    total_current_mark = sum(p["current_val"] for p in positions)

    net_upside_qty = sum(
        p["qty"] if p["opt_type"] == "C" else 0 for p in positions
    )
    net_downside_qty = sum(
        p["qty"] if p["opt_type"] == "P" else 0 for p in positions
    )

    call_strikes = [p["strike"] for p in positions if p["opt_type"] == "C"]
    put_strikes = [p["strike"] for p in positions if p["opt_type"] == "P"]

    highest_call_strike = max(call_strikes) if call_strikes else max(p["strike"] for p in positions)
    lowest_put_strike = min(put_strikes) if put_strikes else min(p["strike"] for p in positions)

    targets = {}

    for pct in percentages:
        mult = 1.0 + (pct / 100.0)
        target_val = total_cost_basis * mult  # Applied to COST BASIS

        req_upside_spot = None
        req_downside_spot = None

        # 1. Upside Target Spot & Contract Prices at Expiration
        if net_upside_qty > 0:
            req_upside_spot = highest_call_strike + (target_val / net_upside_qty)
            upside_pct_change = ((req_upside_spot - current_spot) / current_spot) * 100

            # Individual contract prices at upside spot target
            upside_contract_prices = {}
            for p in positions:
                if p["opt_type"] == "C":
                    px = max(req_upside_spot - p["strike"], 0.0)
                else:
                    px = max(p["strike"] - req_upside_spot, 0.0)
                upside_contract_prices[p["symbol"]] = px

            upside_prices_str = ", ".join(
                [f"{sym.split('-')[-1]}: ${px:,.2f}" for sym, px in upside_contract_prices.items()]
            )
            upside_str = f">= ${req_upside_spot:,.2f} ({upside_pct_change:+.2f}%) [{upside_prices_str}]"
        else:
            upside_str = "N/A"

        # 2. Downside Target Spot & Contract Prices at Expiration
        if net_downside_qty > 0:
            req_downside_spot = lowest_put_strike - (target_val / net_downside_qty)
            if req_downside_spot > 0:
                downside_pct_change = ((req_downside_spot - current_spot) / current_spot) * 100

                # Individual contract prices at downside spot target
                downside_contract_prices = {}
                for p in positions:
                    if p["opt_type"] == "C":
                        px = max(req_downside_spot - p["strike"], 0.0)
                    else:
                        px = max(p["strike"] - req_downside_spot, 0.0)
                    downside_contract_prices[p["symbol"]] = px

                downside_prices_str = ", ".join(
                    [f"{sym.split('-')[-1]}: ${px:,.2f}" for sym, px in downside_contract_prices.items()]
                )
                downside_str = f"<= ${req_downside_spot:,.2f} ({downside_pct_change:+.2f}%) [{downside_prices_str}]"
            else:
                downside_str = "N/A"
        else:
            downside_str = "N/A"

        if upside_str != "N/A" and downside_str != "N/A":
            targets[pct] = f"{downside_str} OR {upside_str}"
        elif upside_str != "N/A":
            targets[pct] = upside_str
        elif downside_str != "N/A":
            targets[pct] = downside_str
        else:
            targets[pct] = "N/A"

    return targets, total_cost_basis, total_current_mark


async def get_positions_and_pnl_targets():
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "API Key and Secret must be provided via environment variables."
        )

    pnl_targets_pct = [5, 10, 20, 50, 100]

    async with aiohttp.ClientSession() as session:
        positions_task = fetch_options_signed(session, "/eapi/v1/position")
        tickers_task = fetch_options_public(session, "/eapi/v1/ticker")

        positions_raw, tickers_raw = await asyncio.gather(
            positions_task, tickers_task
        )

        active_positions = [
            p for p in positions_raw if float(p.get("quantity", 0)) != 0
        ]

        if not active_positions:
            print("No active Binance options positions found.")
            return

        underlying_base = parse_option_symbol(active_positions[0]["symbol"])[0]
        spot_symbol = f"{underlying_base}USDT"

        current_spot = await fetch_spot_price(session, spot_symbol)
        ticker_map = {t["symbol"]: t for t in tickers_raw}

        total_unrealized_pnl = 0.0
        total_liquidation_pnl = 0.0
        parsed_positions = []

        # --- Table 1: Current Individual Option Positions ---
        print("=========================================================================================")
        print(f"       CURRENT POSITIONS & PNL SUMMARY ({spot_symbol} Spot: ${current_spot:,.2f})         ")
        print("=========================================================================================")
        print(
            f"{'Symbol':<24} {'Qty':<7} {'Entry':<10} {'Mark':<10} {'Best Bid':<10} {'Best Ask':<10} {'Unrealized PnL':<15} {'Net Liq PnL':<15}"
        )
        print("-" * 105)

        for pos in active_positions:
            symbol = pos["symbol"]
            qty = float(pos["quantity"])
            entry_price = float(pos["entryPrice"])
            mark_price = float(pos.get("markPrice", 0.0))

            underlying, expiry, strike, opt_type = parse_option_symbol(symbol)

            ticker = ticker_map.get(symbol, {})
            best_bid = float(ticker.get("bidPrice", 0.0))
            best_ask = float(ticker.get("askPrice", 0.0))

            entry_val = entry_price * abs(qty)
            current_pos_val = mark_price * abs(qty)

            unrealized_pnl = (mark_price - entry_price) * qty
            total_unrealized_pnl += unrealized_pnl

            if qty > 0:
                liq_pnl = (best_bid - entry_price) * qty
            else:
                liq_pnl = (entry_price - best_ask) * abs(qty)
            total_liquidation_pnl += liq_pnl

            parsed_positions.append(
                {
                    "symbol": symbol,
                    "underlying": underlying,
                    "expiry": expiry,
                    "strike": strike,
                    "opt_type": opt_type,
                    "qty": qty,
                    "entry_val": entry_val,
                    "current_val": current_pos_val,
                }
            )

            print(
                f"{symbol:<24} {qty:<7.2f} {entry_price:<10.2f} {mark_price:<10.2f} {best_bid:<10.2f} {best_ask:<10.2f} {unrealized_pnl:<15.2f} {liq_pnl:<15.2f}"
            )

        print("-" * 105)
        print(f"Total Mark-to-Market Unrealized PnL: ${total_unrealized_pnl:,.2f}")
        print(f"Total Net Liquidation PnL: ${total_liquidation_pnl:,.2f}\n\n")

        # --- Table 2: COMBINED Portfolio Expiry Spot Price Targets + Individual Contract Prices ---
        portfolio_targets, total_cost_basis, total_mark_val = (
            compute_portfolio_expiry_spot_targets_cost_basis(
                parsed_positions, pnl_targets_pct, current_spot
            )
        )

        print("================================================================================================================================================================")
        print(f"       COMBINED PORTFOLIO EXPIRY TARGETS (Cost Basis: ${total_cost_basis:,.2f} | Current Mark: ${total_mark_val:,.2f})                                            ")
        print("================================================================================================================================================================")
        print(f"{'Target Gain over Cost (%)':<25} {'Target Portfolio Value ($)':<28} {'Required Expiry Spot & Contract Prices ($) [C/P]':<95}")
        print("-" * 160)

        for pct in pnl_targets_pct:
            target_val = total_cost_basis * (1.0 + (pct / 100.0))
            req_spot = portfolio_targets[pct]
            print(f"{f'+{pct}%':<25} {f'${target_val:,.2f}':<28} {req_spot:<95}")

        print("-" * 160)


if __name__ == "__main__":
    asyncio.run(get_positions_and_pnl_targets())
