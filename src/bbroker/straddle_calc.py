from __future__ import annotations
import time
import math
import datetime
import click
from scipy.stats import norm
from scipy.optimize import brentq
from tabulate import tabulate

from bbroker.settings import ex, ensure_markets
from bbroker.ext_order_mgr import order as order_cmd
from butil.bsql import fetch_bidask

BINANCE_OPTION_TRADING_FEE_RATE = 0.00024  # 0.024% of Spot Index Price

def calculate_straddle_fees(spot_index: float, call_ask: float, put_ask: float, qty: float) -> dict:
    """Calculates entry and estimated exit fees for a 2-leg straddle on Binance."""
    # Fee per leg is capped at 10% of option traded price
    fee_per_leg_unit = min(BINANCE_OPTION_TRADING_FEE_RATE * spot_index, 0.10 * call_ask) + \
                       min(BINANCE_OPTION_TRADING_FEE_RATE * spot_index, 0.10 * put_ask)
    
    entry_fee_total = fee_per_leg_unit * qty
    roundtrip_fee_total = entry_fee_total * 2.0  # Open + Close
    
    fee_per_unit_roundtrip = fee_per_leg_unit * 2.0
    return {
        'entry_fee_total': entry_fee_total,
        'roundtrip_fee_total': roundtrip_fee_total,
        'fee_per_unit_roundtrip': fee_per_unit_roundtrip
    }

def parse_contract_info(symbol: str) -> tuple[float, datetime.datetime]:
    """Parses strike price and expiration date from standard symbol e.g. BTC-260828-75000-C."""
    parts = symbol.split('-')
    expiry_str = parts[1]  # YYMMDD
    strike = float(parts[2])

    expiry_dt = datetime.datetime.strptime(expiry_str, "%y%m%d").replace(
        hour=8, minute=0, second=0, tzinfo=datetime.timezone.utc
    )
    return strike, expiry_dt


def bs_price(S: float, K: float, T: float, sigma: float, option_type: str = 'C') -> float:
    """Black-Scholes price assuming zero interest rate (r = 0)."""
    if T <= 1e-6:
        return max(0.0, S - K) if option_type == 'C' else max(0.0, K - S)
    if sigma <= 1e-4:
        return max(0.0, S - K) if option_type == 'C' else max(0.0, K - S)

    d1 = (math.log(S / K) + 0.5 * (sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'C':
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)


def combo_bs_value(S: float, k_call: float, k_put: float, T: float, sigma: float) -> float:
    """Calculates combined value of Call + Put under Black-Scholes."""
    return bs_price(S, k_call, T, sigma, 'C') + bs_price(S, k_put, T, sigma, 'P')


def solve_implied_vol(k_call: float, k_put: float, target_premium: float, T: float, S0: float) -> float:
    """Solves for market implied volatility given total combined premium."""
    if T <= 1e-6 or target_premium <= 0:
        return 0.0
    obj = lambda sig: combo_bs_value(S0, k_call, k_put, T, sig) - target_premium
    try:
        return brentq(obj, 1e-3, 10.0)
    except Exception:
        return 0.0

def calculate_breakevens(k_call: float, k_put: float, premium: float, T: float, sigma: float) -> dict:
    """Calculates expiry breakevens and interim Black-Scholes breakeven spot prices."""
    lower_expiry = k_put - premium
    upper_expiry = k_call + premium

    lower_interim, upper_interim = None, None
    mid_strike = (k_call + k_put) / 2.0
    status = "OK"

    if T > 1e-5 and sigma > 0:
        obj = lambda S: combo_bs_value(S, k_call, k_put, T, sigma) - premium
        bs_val_at_mid = combo_bs_value(mid_strike, k_call, k_put, T, sigma)

        # Regime 1: Model IV > Market IV -> Position is instantly in profit at current spot
        if bs_val_at_mid >= premium:
            status = "ALWAYS_PROFITABLE"
        # Regime 2: Model IV < Market IV -> Spot divergence required to hit BE
        else:
            status = "NEEDS_DIVERGENCE"
            try:
                lower_interim = brentq(obj, 1e-2, mid_strike)
            except Exception:
                lower_interim = None

            try:
                upper_interim = brentq(obj, mid_strike, mid_strike * 10.0)
            except Exception:
                upper_interim = None

    return {
        'lower_expiry': lower_expiry,
        'upper_expiry': upper_expiry,
        'lower_interim': lower_interim,
        'upper_interim': upper_interim,
        'status': status
    }

def _calculate_breakevens(k_call: float, k_put: float, premium: float, T: float, sigma: float) -> dict:
    """Calculates both expiry breakeven spot prices and interim Black-Scholes breakeven spot prices."""
    lower_expiry = k_put - premium
    upper_expiry = k_call + premium

    lower_interim, upper_interim = None, None
    mid_strike = (k_call + k_put) / 2.0

    if T > 1e-5 and sigma > 0:
        obj = lambda S: combo_bs_value(S, k_call, k_put, T, sigma) - premium

        # Verify if BS theoretical value exceeds premium at center
        if obj(mid_strike) >= 0:
            try:
                lower_interim = brentq(obj, 1e-2, mid_strike)
            except Exception:
                lower_interim = None

            try:
                upper_interim = brentq(obj, mid_strike, mid_strike * 10.0)
            except Exception:
                upper_interim = None

    return {
        'lower_expiry': lower_expiry,
        'upper_expiry': upper_expiry,
        'lower_interim': lower_interim,
        'upper_interim': upper_interim
    }


def get_option_market_data(symbol: str) -> dict:
    """Fetch order book / bidask data for a given option contract."""
    quote = fetch_bidask(symbol)
    bid = float(quote.get('bid', 0.0))
    ask = float(quote.get('ask', 0.0))
    iv = float(quote.get('iv', quote.get('markIV', 0.0)))

    if bid <= 0 or ask <= 0:
        ensure_markets(ex)
        orderbook = ex.fetch_order_book(symbol)
        bid = float(orderbook['bids'][0][0]) if orderbook.get('bids') else 0.0
        ask = float(orderbook['asks'][0][0]) if orderbook.get('asks') else 0.0

    spread_abs = ask - bid
    mid_price = (ask + bid) / 2.0 if (ask + bid) > 0 else 0.0
    spread_bps = (spread_abs / mid_price * 10000.0) if mid_price > 0 else 0.0

    return {
        'symbol': symbol,
        'bid': bid,
        'ask': ask,
        'spread_abs': spread_abs,
        'spread_bps': spread_bps,
        'mid': mid_price,
        'iv': iv
    }


def execute_straddle_leg(ctx, action: str, symbol: str, size: float, t_bps: float):
    """Executes a single leg of the straddle via the chase order manager logic."""
    click.secho(f"\n[EXECUTING LEG] {action.upper()} {size} {symbol} via CHASE mode...", fg='magenta', bold=True)
    ctx.invoke(
        order_cmd,
        action=action,
        contract=symbol,
        qty=size,
        order_type='chase',
        order_price=None,
        t_bps=t_bps,
        execute=True
    )


@click.command()
@click.option('--call', required=True, help="Call option symbol, e.g., BTC-260828-75000-C")
@click.option('--put', required=True, help="Put option symbol, e.g., BTC-260828-75000-P")
@click.option('--size', type=float, default=0.01, show_default=True, help="Order size per leg")
@click.option('--action', type=click.Choice(['buy', 'sell'], case_sensitive=False), default='buy', show_default=True, help="Straddle position side: buy (long) or sell (short)")
@click.option('--iv', type=float, default=None, help="Model IV for evaluation (e.g., 0.55 or 55 for 55%)")
@click.option('--t_bps', type=float, default=10.0, show_default=True, help="Max allowed price shift threshold in bps for chase orders")
@click.option('--execute', is_flag=True, default=False, help="Execute live orders using chase mode")
@click.pass_context
def main(ctx, call, put, size, action, iv, t_bps, execute):
    """Calculate costs and Black-Scholes breakeven levels for Straddle/Strangle strategies."""
    call = call.upper()
    put = put.upper()
    action = action.lower()

    click.secho(f"\n{'='*25} Straddle Cost & Breakeven Calculator {'='*25}", fg='cyan', bold=True)
    click.secho(f"Strategy Side : {action.upper()}", fg='yellow', bold=True)
    click.secho(f"Position Size : {size} BTC per leg\n", fg='yellow')

    call_data = get_option_market_data(call)
    put_data = get_option_market_data(put)

    table_data = [
        [
            "Call",
            call_data['symbol'],
            f"${call_data['bid']:.2f}",
            f"${call_data['ask']:.2f}",
            f"${call_data['spread_abs']:.2f}",
            f"{call_data['spread_bps']:.1f} bps"
        ],
        [
            "Put",
            put_data['symbol'],
            f"${put_data['bid']:.2f}",
            f"${put_data['ask']:.2f}",
            f"${put_data['spread_abs']:.2f}",
            f"{put_data['spread_bps']:.1f} bps"
        ]
    ]

    click.echo(tabulate(table_data, headers=["Leg", "Symbol", "Best Bid", "Best Ask", "Spread (Abs)", "Spread (bps)"], tablefmt="grid"))

    if action == 'buy':
        call_unit_price = call_data['ask']
        put_unit_price = put_data['ask']
        leg_desc = "Best Ask (Buy)"
    else:
        call_unit_price = call_data['bid']
        put_unit_price = put_data['bid']
        leg_desc = "Best Bid (Sell)"

    call_cost = call_unit_price * size
    put_cost = put_unit_price * size
    total_cost = call_cost + put_cost
    unit_straddle_price = call_unit_price + put_unit_price

    # Parse contract specifications
    k_call, expiry_dt = parse_contract_info(call)
    k_put, _ = parse_contract_info(put)

    now = datetime.datetime.now(datetime.timezone.utc)
    time_to_expiry_years = max(1e-6, (expiry_dt - now).total_seconds() / (365.25 * 86400.0))
    underlying_ref = (k_call + k_put) / 2.0

    # Solve market implied volatility
    mkt_implied_iv = solve_implied_vol(k_call, k_put, unit_straddle_price, time_to_expiry_years, underlying_ref)

    # Normalize user input --iv
    if iv is not None:
        model_sigma = iv / 100.0 if iv > 3.0 else iv
    else:
        model_sigma = mkt_implied_iv

    be = calculate_breakevens(k_call, k_put, unit_straddle_price, time_to_expiry_years, model_sigma)

    click.secho(f"\n{'-'*65}", fg='bright_black')
    click.secho(f"Execution Price Standard : {leg_desc}", fg='white')
    click.secho(f"Call Leg Cost ({size} qty) : ${call_cost:,.2f} (@ ${call_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Put Leg Cost  ({size} qty) : ${put_cost:,.2f} (@ ${put_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Combined Unit Premium   : ${unit_straddle_price:,.2f}", fg='green', bold=True)
    click.secho(f"Total Required Cost     : ${total_cost:,.2f}", fg='green', bold=True)
    click.secho(f"{'-'*65}", fg='bright_black')

    # Black-Scholes Breakeven Section
    # Black-Scholes Breakeven Section Output
    click.secho("Black-Scholes Breakeven Spot Levels:", fg='yellow', bold=True)
    click.secho(f"  • At-Expiry Lower Breakeven  : ${be['lower_expiry']:,.2f}", fg='white')
    click.secho(f"  • At-Expiry Upper Breakeven  : ${be['upper_expiry']:,.2f}", fg='white')
    click.secho(f"  • Market Implied Volatility  : {mkt_implied_iv*100:.1f}% (DTE: {time_to_expiry_years*365.25:.1f}d)", fg='bright_blue')
    click.secho(f"  • Selected Model Volatility  : {model_sigma*100:.1f}%", fg='bright_blue')

    if be['status'] == "ALWAYS_PROFITABLE":
        click.secho(f"  • Interim Breakeven          : Instantly Profitable at ALL Spot Levels (Vega profit covers premium)", fg='bright_green')
    elif be['lower_interim'] and be['upper_interim']:
        click.secho(f"  • Interim Lower Breakeven    : ${be['lower_interim']:,.2f}", fg='bright_cyan')
        click.secho(f"  • Interim Upper Breakeven    : ${be['upper_interim']:,.2f}", fg='bright_cyan')
    else:
        click.secho(f"  • Interim Breakeven          : N/A (Model IV {model_sigma*100:.1f}% < Market Implied IV {mkt_implied_iv*100:.1f}%)", fg='bright_red')

    spot_index = (k_call + k_put) / 2.0
    fees = calculate_straddle_fees(spot_index, call_unit_price, put_unit_price, size)

    all_in_unit_premium = unit_straddle_price + fees['fee_per_unit_roundtrip']
    fee_adj_lower_be = k_put - all_in_unit_premium
    fee_adj_upper_be = k_call + all_in_unit_premium

    click.secho(f"  • Est. Round-Trip Trading Fee : ${fees['roundtrip_fee_total']:.2f} (${fees['fee_per_unit_roundtrip']:.2f}/unit)", fg='magenta')
    click.secho(f"  • Fee-Adjusted Lower BE      : ${fee_adj_lower_be:,.2f}", fg='yellow')
    click.secho(f"  • Fee-Adjusted Upper BE      : ${fee_adj_upper_be:,.2f}", fg='yellow')

    click.secho(f"{'-'*65}\n", fg='bright_black')

    if not execute:
        click.secho("[INFO] Dry run completed. Add '--execute' flag to send live chase orders.", fg='yellow')
        return

    click.secho("Initiating Live Straddle Execution via Chase Orders...", fg='magenta', bold=True)
    execute_straddle_leg(ctx, action, call, size, t_bps)
    execute_straddle_leg(ctx, action, put, size, t_bps)
    click.secho("\n[COMPLETED] Both straddle legs sent for execution.", fg='green', bold=True)


if __name__ == '__main__':
    main()
