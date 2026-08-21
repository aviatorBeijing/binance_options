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
MAX_SAFE_SPREAD_BPS = 100.0                # Warning & safety threshold for spread


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
    if T <= 1e-6 or sigma <= 1e-4:
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


def calculate_target_pnl_spots(
    k_call: float, 
    k_put: float, 
    entry_premium: float, 
    S0: float, 
    T: float, 
    sigma: float, 
    pnl_targets: list[float]
) -> list[dict]:
    """Calculates required spot prices and percentage drifts for target portfolio PnLs."""
    results = []

    for target_pct in pnl_targets:
        target_value = entry_premium * (1.0 + target_pct)

        # 1. At Expiry Spot Levels
        upper_expiry_spot = k_call + target_value
        lower_expiry_spot = k_put - target_value

        upper_expiry_drift = (upper_expiry_spot - S0) / S0 * 100.0
        lower_expiry_drift = (lower_expiry_spot - S0) / S0 * 100.0

        # 2. Interim Spot Levels
        lower_interim_spot, upper_interim_spot = None, None
        lower_interim_drift, upper_interim_drift = None, None

        if T > 1e-5 and sigma > 0:
            obj = lambda S: combo_bs_value(S, k_call, k_put, T, sigma) - target_value
            try:
                lower_interim_spot = brentq(obj, 1e-2, S0)
                lower_interim_drift = (lower_interim_spot - S0) / S0 * 100.0
            except Exception:
                pass

            try:
                upper_interim_spot = brentq(obj, S0, S0 * 10.0)
                upper_interim_drift = (upper_interim_spot - S0) / S0 * 100.0
            except Exception:
                pass

        results.append({
            'target_pct': target_pct,
            'target_value': target_value,
            'upper_expiry_spot': upper_expiry_spot,
            'upper_expiry_drift': upper_expiry_drift,
            'lower_expiry_spot': lower_expiry_spot,
            'lower_expiry_drift': lower_expiry_drift,
            'upper_interim_spot': upper_interim_spot,
            'upper_interim_drift': upper_interim_drift,
            'lower_interim_spot': lower_interim_spot,
            'lower_interim_drift': lower_interim_drift,
        })

    return results


def get_option_market_data(symbol: str, max_retries: int = 5, retry_delay: float = 0.5) -> dict:
    """Fetch bid/ask data for an option contract with up to 5 retries on zero/unreasonable quotes."""
    bid, ask = 0.0, 0.0

    for attempt in range(1, max_retries + 1):
        quote = fetch_bidask(symbol)
        bid = float(quote.get('bid', 0.0))
        ask = float(quote.get('ask', 0.0))

        # Fallback to exchange orderbook fetch if bid/ask are trivial
        if bid <= 0.0 or ask <= 0.0:
            try:
                ensure_markets(ex)
                orderbook = ex.fetch_order_book(symbol)
                bid = float(orderbook['bids'][0][0]) if orderbook.get('bids') else 0.0
                ask = float(orderbook['asks'][0][0]) if orderbook.get('asks') else 0.0
            except Exception:
                pass

        # Valid non-zero quotes obtained
        if bid > 0.0 and ask > 0.0:
            break

        if attempt < max_retries:
            click.secho(
                f"[WARN] Invalid/Zero quote for {symbol} (Bid: ${bid:.2f}, Ask: ${ask:.2f}). "
                f"Retrying query ({attempt}/{max_retries})...", 
                fg='yellow'
            )
            time.sleep(retry_delay)

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
    }


def validate_market_safety(call_data: dict, put_data: dict, action: str) -> tuple[bool, list[str]]:
    """Evaluates market structure across both legs. Returns (is_safe, failure_reasons)."""
    reasons = []

    # 1. Zero/Invalid Price Check
    if action == 'buy':
        call_p, put_p = call_data['ask'], put_data['ask']
    else:
        call_p, put_p = call_data['bid'], put_data['bid']

    if call_p <= 0.0 or put_p <= 0.0 or call_data['bid'] <= 0.0 or put_data['bid'] <= 0.0:
        reasons.append(
            f"Zero/Invalid quote persistent after retries: "
            f"Call ({call_data['symbol']}) Bid=${call_data['bid']:.2f}/Ask=${call_data['ask']:.2f} | "
            f"Put ({put_data['symbol']}) Bid=${put_data['bid']:.2f}/Ask=${put_data['ask']:.2f}"
        )

    # 2. Spread Safety Check (> 100 bps)
    for leg, data in [("Call", call_data), ("Put", put_data)]:
        if data['spread_bps'] > MAX_SAFE_SPREAD_BPS:
            reasons.append(
                f"{leg} Leg ({data['symbol']}) spread is {data['spread_bps']:.1f} bps "
                f"(${data['spread_abs']:.2f}), exceeding threshold of {MAX_SAFE_SPREAD_BPS:.0f} bps."
            )

    is_safe = len(reasons) == 0
    return is_safe, reasons


def execute_straddle_leg(ctx, action: str, symbol: str, size: float, t_bps: float):
    """Executes a single leg of the straddle via chase mode."""
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
@click.option('--action', type=click.Choice(['buy', 'sell'], case_sensitive=False), default='buy', show_default=True)
@click.option('--iv', type=float, default=None, help="Model IV for evaluation (e.g., 0.55 for 55%)")
@click.option('--t_bps', type=float, default=10.0, show_default=True, help="Max price shift threshold in bps")
@click.option('--execute', is_flag=True, default=False, help="Execute live orders using chase mode")
@click.pass_context
def main(ctx, call, put, size, action, iv, t_bps, execute):
    """Calculate portfolio target PnL spot drift requirements for Straddle strategies."""
    call = call.upper()
    put = put.upper()
    action = action.lower()

    click.secho(f"\n{'='*25} Straddle Cost & Target PnL Drift Calculator {'='*25}", fg='cyan', bold=True)
    click.secho(f"Strategy Side : {action.upper()}", fg='yellow', bold=True)
    click.secho(f"Position Size : {size} BTC per leg\n", fg='yellow')

    call_data = get_option_market_data(call)
    put_data = get_option_market_data(put)

    table_data = [
        ["Call", call_data['symbol'], f"${call_data['bid']:.2f}", f"${call_data['ask']:.2f}", f"${call_data['spread_abs']:.2f}", f"{call_data['spread_bps']:.1f} bps"],
        ["Put", put_data['symbol'], f"${put_data['bid']:.2f}", f"${put_data['ask']:.2f}", f"${put_data['spread_abs']:.2f}", f"{put_data['spread_bps']:.1f} bps"]
    ]
    click.echo(tabulate(table_data, headers=["Leg", "Symbol", "Best Bid", "Best Ask", "Spread (Abs)", "Spread (bps)"], tablefmt="grid"))

    # Validate price sanity and spread limits across the whole structure
    is_safe, failure_reasons = validate_market_safety(call_data, put_data, action)

    if not is_safe:
        click.secho(f"\n[DANGER WARNING] Safety Checks Failed:", fg='red', bold=True)
        for reason in failure_reasons:
            click.secho(f"  ⚠️  {reason}", fg='red')

        if execute:
            click.secho(
                f"\n[GLOBAL SAFETY BLOCK] Execution completely halted for ALL legs of strategy {call} / {put}.", 
                fg='red', 
                bold=True
            )
            raise click.ClickException("Strategy execution blocked due to unsafe market conditions on one or more legs.")

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
    model_sigma = (iv / 100.0 if iv > 3.0 else iv) if iv is not None else mkt_implied_iv

    click.secho(f"\n{'-'*75}", fg='bright_black')
    click.secho(f"Execution Price Standard : {leg_desc}", fg='white')
    click.secho(f"Call Leg Cost ({size} qty) : ${call_cost:,.2f} (@ ${call_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Put Leg Cost  ({size} qty) : ${put_cost:,.2f} (@ ${put_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Combined Unit Premium   : ${unit_straddle_price:,.2f}", fg='green', bold=True)
    click.secho(f"Total Required Premium  : ${total_cost:,.2f}", fg='green', bold=True)
    click.secho(f"Spot Reference Anchor   : ${underlying_ref:,.2f} | IV: {mkt_implied_iv*100:.1f}%", fg='bright_blue')
    click.secho(f"{'-'*75}", fg='bright_black')

    # Calculate required spot price movements for target PnLs
    pnl_targets = [0.01, 0.05, 0.10, 0.50, 1.00]  # 1%, 5%, 10%, 50%, 100%
    drift_data = calculate_target_pnl_spots(
        k_call=k_call,
        k_put=k_put,
        entry_premium=unit_straddle_price,
        S0=underlying_ref,
        T=time_to_expiry_years,
        sigma=model_sigma,
        pnl_targets=pnl_targets
    )

    # Format output table
    pnl_table = []
    for d in drift_data:
        target_label = f"+{int(d['target_pct']*100)}%"
        
        up_exp = f"${d['upper_expiry_spot']:,.2f} (+{d['upper_expiry_drift']:.2f}%)"
        dn_exp = f"${d['lower_expiry_spot']:,.2f} ({d['lower_expiry_drift']:.2f}%)"

        up_int = f"${d['upper_interim_spot']:,.2f} (+{d['upper_interim_drift']:.2f}%)" if d['upper_interim_spot'] else "N/A"
        dn_int = f"${d['lower_interim_spot']:,.2f} ({d['lower_interim_drift']:.2f}%)" if d['lower_interim_spot'] else "N/A"

        pnl_table.append([target_label, f"${d['target_value']:,.2f}", up_exp, dn_exp, up_int, dn_int])

    click.secho("\nSpot Price Movement Required for Portfolio Return Targets:", fg='yellow', bold=True)
    headers = ["Target PnL", "Unit Value", "Expiry Up Drift", "Expiry Down Drift", "Interim Up Drift", "Interim Down Drift"]
    click.echo(tabulate(pnl_table, headers=headers, tablefmt="grid"))
    click.secho(f"{'-'*75}\n", fg='bright_black')

    if not execute:
        click.secho("[INFO] Dry run completed. Add '--execute' flag to send live chase orders.", fg='yellow')
        return

    click.secho("Initiating Live Straddle Execution via Chase Orders...", fg='magenta', bold=True)
    execute_straddle_leg(ctx, action, call, size, t_bps)
    execute_straddle_leg(ctx, action, put, size, t_bps)
    click.secho("\n[COMPLETED] Both straddle legs sent for execution.", fg='green', bold=True)


if __name__ == '__main__':
    main()
