import time
import click
from tabulate import tabulate
from bbroker.settings import ex, ensure_markets
from bbroker.ext_order_mgr import order as order_cmd
from butil.bsql import fetch_bidask

def get_option_market_data(symbol: str) -> dict:
    """Fetch order book / bidask data for a given option contract."""
    quote = fetch_bidask(symbol)
    bid = float(quote.get('bid', 0.0))
    ask = float(quote.get('ask', 0.0))
    
    # Fallback to CCXT synchronously if local quote is not populated
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
        'mid': mid_price
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
@click.option('--put', required=True, help="Put option symbol, e.g., BTC-260828-76000-P")
@click.option('--size', type=float, default=0.01, show_default=True, help="Order size per leg")
@click.option('--action', type=click.Choice(['buy', 'sell'], case_sensitive=False), default='buy', show_default=True, help="Straddle position side: buy (long) or sell (short)")
@click.option('--t_bps', type=float, default=10.0, show_default=True, help="Max allowed price shift threshold in bps for chase orders")
@click.option('--execute', is_flag=True, default=False, help="Execute live orders using chase mode")
@click.pass_context
def main(ctx, call, put, size, action, t_bps, execute):
    """Calculate costs and execution parameters for building a Straddle strategy."""
    call = call.upper()
    put = put.upper()
    action = action.lower()

    click.secho(f"\n{'='*25} Straddle Cost Calculator {'='*25}", fg='cyan', bold=True)
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

    # Determine execution pricing logic
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

    click.secho(f"\n{'-'*65}", fg='bright_black')
    click.secho(f"Execution Price Standard : {leg_desc}", fg='white')
    click.secho(f"Call Leg Cost ({size} qty) : ${call_cost:,.2f} (@ ${call_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Put Leg Cost  ({size} qty) : ${put_cost:,.2f} (@ ${put_unit_price:,.2f}/unit)", fg='cyan')
    click.secho(f"Combined Unit Premium   : ${unit_straddle_price:,.2f}", fg='green', bold=True)
    click.secho(f"Total Required Cost     : ${total_cost:,.2f}", fg='green', bold=True)
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
