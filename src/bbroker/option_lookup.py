import click
import datetime
import pandas as pd
import requests
from tabulate import tabulate

from strategy.price_disparity import extract_specs
from butil.bsql import fetch_bidask
from bbroker.settings import ex, ensure_markets

from sentiments.atms import refresh_contracts


def get_strike_from_symbol(symbol: str) -> float:
    """Extract strike price float from Binance standard option symbol format."""
    try:
        parts = symbol.split('-')
        return float(parts[2])
    except Exception:
        return 0.0


def fetch_mark_iv_map() -> dict:
    """Fetch mark price and implied volatility directly from Binance Options REST API."""
    try:
        url = "https://eapi.binance.com/eapi/v1/mark"
        res = requests.get(url, timeout=5).json()
        iv_map = {}
        if isinstance(res, list):
            for item in res:
                sym = item.get('symbol')
                mark_iv = float(item.get('markIV', 0.0))
                if sym:
                    iv_map[sym] = mark_iv
        return iv_map
    except Exception:
        return {}


def get_option_quote(symbol: str, iv_map: dict = None) -> dict:
    """Fetch bid/ask quotes and implied volatility with REST fallback."""
    quote = fetch_bidask(symbol)
    bid = float(quote.get('bid', 0.0))
    ask = float(quote.get('ask', 0.0))
    raw_iv = float(quote.get('iv', quote.get('markIV', 0.0)))

    # Fallback to REST IV map if raw_iv is missing/invalid
    if raw_iv <= 0 and iv_map and symbol in iv_map:
        raw_iv = iv_map[symbol]

    # Fallback to direct orderbook fetch if bid/ask prices are zero
    if bid <= 0 or ask <= 0:
        ensure_markets(ex)
        try:
            orderbook = ex.fetch_order_book(symbol)
            bid = float(orderbook['bids'][0][0]) if orderbook.get('bids') else 0.0
            ask = float(orderbook['asks'][0][0]) if orderbook.get('asks') else 0.0
        except Exception:
            pass

    # Binance eapi returns markIV as a decimal multiplier (e.g., 0.45 = 45%, 1.45 = 145%)
    iv_pct = raw_iv * 100.0 if raw_iv > 0 else 0.0

    return {'symbol': symbol, 'bid': bid, 'ask': ask, 'iv': iv_pct}


@click.command()
@click.option('--date', required=True, help="Expiry date string, e.g. '260828'")
@click.option('--target_price', type=float, required=True, help="Target strike/underlying price, e.g. 75000")
@click.option('--crypto', default='BTC', show_default=True, help="Crypto underlying asset symbol, e.g. BTC, ETH")
@click.option('--n_strikes', type=int, default=2, show_default=True, help="Number of strike levels above/below target price")
@click.option('--update', is_flag=True, default=False, help="Force refresh of Binance contract metadata cache")
def main(date: str, target_price: float, crypto: str, n_strikes: int, update: bool):
    """Finds Binance options contracts around target_price for a given date and generates combination cost matrix."""
    crypto = crypto.upper().strip()
    date = str(date).strip()

    click.secho(f"\n{'='*25} Options Combination Cost Matrix {'='*25}", fg='cyan', bold=True)
    click.secho(f"Underlying: {crypto} | Expiry Date: {date} | Target Price: ${target_price:,.2f}\n", fg='yellow')

    contracts_df = refresh_contracts(crypto, update=update)

    if contracts_df.empty:
        click.secho(f"[ERROR] Failed to fetch contract information for underlying '{crypto}'.", fg='red', bold=True)
        return

    matching_df = contracts_df[contracts_df['symbol'].str.contains(f"-{date}-")]

    if matching_df.empty:
        available_expiries = sorted(list(set(contracts_df['expiry'].dropna().astype(str).values)))
        click.secho(f"[ERROR] No options found matching expiry date '{date}' for {crypto}.", fg='red', bold=True)
        if available_expiries:
            click.secho(f"Available active expiries on Binance: {', '.join(available_expiries)}", fg='bright_black')
        return

    all_strikes = sorted(matching_df['strikePrice'].unique())
    
    closest_single_strike = min(all_strikes, key=lambda k: abs(k - target_price))
    closest_strikes = sorted(all_strikes, key=lambda k: abs(k - target_price))[:(n_strikes * 2 + 1)]
    selected_strikes = sorted(closest_strikes)

    strike_map = {}
    for k in selected_strikes:
        strike_df = matching_df[matching_df['strikePrice'] == k]
        c_syms = strike_df[strike_df['symbol'].str.endswith('-C')]['symbol'].values
        p_syms = strike_df[strike_df['symbol'].str.endswith('-P')]['symbol'].values
        
        strike_map[k] = {
            'C': c_syms[0] if len(c_syms) > 0 else None,
            'P': p_syms[0] if len(p_syms) > 0 else None
        }

    # Fetch batch IV mapping directly from Binance eapi endpoint
    iv_map = fetch_mark_iv_map()

    quotes = {}
    for k in selected_strikes:
        c_sym = strike_map[k]['C']
        p_sym = strike_map[k]['P']
        if c_sym:
            quotes[c_sym] = get_option_quote(c_sym, iv_map=iv_map)
        if p_sym:
            quotes[p_sym] = get_option_quote(p_sym, iv_map=iv_map)

    combo_rows = []
    for k_call in selected_strikes:
        c_sym = strike_map[k_call]['C']
        if not c_sym:
            continue
        c_quote = quotes.get(c_sym, {})
        c_ask = c_quote.get('ask', 0.0)
        c_bid = c_quote.get('bid', 0.0)
        c_iv = c_quote.get('iv', 0.0)

        for k_put in selected_strikes:
            p_sym = strike_map[k_put]['P']
            if not p_sym:
                continue
            p_quote = quotes.get(p_sym, {})
            p_ask = p_quote.get('ask', 0.0)
            p_bid = p_quote.get('bid', 0.0)
            p_iv = p_quote.get('iv', 0.0)

            buy_cost = c_ask + p_ask
            sell_credit = c_bid + p_bid
            combo_type = "Straddle" if k_call == k_put else "Strangle"

            is_target_row = (k_call == closest_single_strike and k_put == closest_single_strike)

            c_strike_str = f"${k_call:,.0f}"
            p_strike_str = f"${k_put:,.0f}"
            combo_str = combo_type
            c_ask_str = f"${c_ask:,.2f}"
            p_ask_str = f"${p_ask:,.2f}"
            buy_cost_str = f"${buy_cost:,.2f}"
            sell_credit_str = f"${sell_credit:,.2f}"
            c_iv_str = f"{c_iv:.1f}%" if c_iv > 0 else "N/A"
            p_iv_str = f"{p_iv:.1f}%" if p_iv > 0 else "N/A"
            c_sym_str = str(c_sym)
            p_sym_str = str(p_sym)

            if is_target_row:
                c_strike_str = click.style(c_strike_str, fg='bright_green', bold=True)
                p_strike_str = click.style(p_strike_str, fg='bright_green', bold=True)
                combo_str = click.style(combo_str, fg='bright_green', bold=True)
                c_ask_str = click.style(c_ask_str, fg='bright_green', bold=True)
                p_ask_str = click.style(p_ask_str, fg='bright_green', bold=True)
                buy_cost_str = click.style(buy_cost_str, fg='bright_green', bold=True)
                sell_credit_str = click.style(sell_credit_str, fg='bright_green', bold=True)
                c_iv_str = click.style(c_iv_str, fg='bright_green', bold=True)
                p_iv_str = click.style(p_iv_str, fg='bright_green', bold=True)
                c_sym_str = click.style(c_sym_str, fg='bright_green', bold=True)
                p_sym_str = click.style(p_sym_str, fg='bright_green', bold=True)

            combo_rows.append({
                "Call Strike": c_strike_str,
                "Put Strike": p_strike_str,
                "Type": combo_str,
                "Call Ask": c_ask_str,
                "Put Ask": p_ask_str,
                "Buy Cost": buy_cost_str,
                "Sell Credit": sell_credit_str,
                "Call IV": c_iv_str,
                "Put IV": p_iv_str,
                "Call Symbol": c_sym_str,
                "Put Symbol": p_sym_str
            })

    df = pd.DataFrame(combo_rows)
    click.echo(tabulate(df, headers="keys", tablefmt="grid"))


if __name__ == '__main__':
    main()
