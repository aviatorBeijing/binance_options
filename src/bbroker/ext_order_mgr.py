import time
import click
import numpy as np
from tabulate import tabulate
from bbroker.settings import ex, ensure_markets
from bbroker.check_status import orders_status, position_status
from butil.bsql import fetch_bidask

def validate_sell_quantity(symbol: str, qty: float):
    df = position_status()
    if df.empty:
        raise click.ClickException(f"No existing positions found.")
    
    df = df[df.symbol == symbol]
    if df.empty:
        raise click.ClickException(f"No existing position for symbol: {symbol}")

    odf = orders_status()
    if not odf.empty:
        odf = odf[(odf.symbol == symbol) & (odf.side == 'SELL')]
        if not odf.empty:
            existing_position = abs(float(df.iloc[0].quantity))
            existing_sell_qty = np.sum(odf.quantity.astype(float).apply(abs))
            if existing_position < (existing_sell_qty + qty):
                raise click.ClickException(
                    f"Position limit exceeded: position={existing_position}, "
                    f"existing_sells={existing_sell_qty}, requested={qty}"
                )

@click.group()
def cli():
    """Order Management Tool for Binance Options"""
    pass

@cli.command()
@click.option('--action', type=click.Choice(['buy', 'sell'], case_sensitive=False), required=True)
@click.option('--contract', required=True, help="Contract symbol, e.g., BTC-240329-70000-C")
@click.option('--qty', type=float, required=True, help="Order quantity")
@click.option('--limit', 'order_type', flag_value='limit', default=True, help="Limit order")
@click.option('--market', 'order_type', flag_value='market', help="Market order")
@click.option('--chase', 'order_type', flag_value='chase', help="Chase order mode")
@click.option('--order_price', type=float, default=None, help="Required if --limit is set")
@click.option('--t_bps', type=float, default=10.0, help="Price shift threshold in bps for --chase mode")
@click.option('--execute', is_flag=True, default=False, help="Execute order on exchange")
def order(action, contract, qty, order_type, order_price, t_bps, execute):
    """Place a buy or sell order."""
    action = action.lower()
    contract = contract.upper()

    if order_type == 'limit' and (order_price is None or order_price <= 0):
        raise click.BadParameter("Parameter '--order_price' (> 0) is required when using --limit mode.")

    click.secho(f"\n[ORDER REQ] {action.upper()} {qty} {contract} | Type: {order_type.upper()}", fg='cyan', bold=True)

    if action == 'sell':
        validate_sell_quantity(contract, qty)

    if not execute:
        click.secho("-- Dry run mode. Pass '--execute' to place actual order.", fg='yellow')
        return

    ensure_markets(ex)

    if order_type == 'limit':
        click.secho(f"--> Sending LIMIT {action.upper()} at ${order_price:.2f}", fg='green')
        res = ex.create_order(contract, 'limit', action, qty, order_price)
        click.secho(f"Order Placed ID: {res.get('id')}", fg='green', bold=True)

    elif order_type == 'market':
        click.secho(f"--> Sending MARKET {action.upper()}", fg='green')
        res = ex.create_order(contract, 'market', action, qty)
        click.secho(f"Market Order Executed ID: {res.get('id')}", fg='green', bold=True)

    elif order_type == 'chase':
        click.secho(f"--> Starting CHASE mode for {action.upper()} (Max shift tolerance: {t_bps} bps)", fg='magenta')
        
        # Initial quote anchor
        init_quote = fetch_bidask(contract)
        target_side = 'ask' if action == 'buy' else 'bid'
        base_price = float(init_quote[target_side])

        if base_price <= 0:
            raise click.ClickException(f"Invalid initial quote price: {base_price}")

        click.secho(f"Initial {target_side.upper()} price: ${base_price:.2f}", fg='blue')
        
        # Place initial order at best quote
        current_order = ex.create_order(contract, 'limit', action, qty, base_price)
        order_id = current_order['id']
        click.secho(f"Placed initial chase order ID: {order_id} at ${base_price:.2f}", fg='green')

        while True:
            time.sleep(2)
            quote = fetch_bidask(contract)
            curr_best = float(quote[target_side])

            # Check market drift in bps relative to starting quote
            price_shift_bps = abs(curr_best - base_price) / base_price * 10000.0

            if price_shift_bps > t_bps:
                click.secho(
                    f"\n[WARNING] Market shifted {price_shift_bps:.1f} bps (Threshold: {t_bps} bps). "
                    f"Initial: ${base_price:.2f} -> Current: ${curr_best:.2f}. Halting chase process.",
                    fg='red', bold=True
                )
                click.secho(f"Leaving active order {order_id} open on book.", fg='yellow')
                break

            # Check open orders to confirm status
            open_orders = ex.eapiPrivateGetOpenOrders()
            active_ids = [str(o['orderId']) for o in open_orders] if open_orders else []
            
            if str(order_id) not in active_ids:
                click.secho(f"\n[SUCCESS] Chase order {order_id} filled or closed!", fg='green', bold=True)
                break

            # Re-price order if best bid/ask moved within tolerance
            click.secho(f"Updating chase price to current best {target_side.upper()}: ${curr_best:.2f}...", fg='cyan')
            try:
                ex.cancel_order(order_id, contract)
                time.sleep(0.5)
                new_order = ex.create_order(contract, 'limit', action, qty, curr_best)
                order_id = new_order['id']
                click.secho(f"Replaced order ID: {order_id} at ${curr_best:.2f}", fg='green')
            except Exception as e:
                click.secho(f"Failed to replace order during chase: {e}", fg='red')
                break

@cli.command()
@click.option('--contract', required=True, help="Contract symbol")
@click.option('--order_id', required=True, help="Order ID to cancel")
def cancel(contract, order_id):
    """Cancel an open order."""
    contract = contract.upper()
    click.secho(f"Attempting cancellation for order {order_id} on {contract}...", fg='cyan')
    
    df = orders_status()
    if df.empty or df[(df.symbol == contract) & (df.orderId == order_id)].empty:
        click.secho(f"No active order found matching ID: {order_id}", fg='yellow')
        return

    ex.cancel_order(order_id, contract)
    click.secho(f"Order {order_id} cancelled successfully.", fg='green', bold=True)
    time.sleep(1)
    orders_status()

@cli.command()
def status():
    """Show current open orders and positions."""
    click.secho("=" * 30 + " Open Orders " + "=" * 30, fg='blue', bold=True)
    orders_status()
    click.secho("\n" + "=" * 30 + " Positions " + "=" * 30, fg='blue', bold=True)
    position_status()

if __name__ == '__main__':
    cli()
