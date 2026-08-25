from __future__ import annotations
import time
import click
import numpy as np
from tabulate import tabulate
from bbroker.settings import ex, ensure_markets
from bbroker.check_status import orders_status, position_status
from butil.bsql import fetch_bidask


def extract_passive_price(quote_dict: dict, action: str) -> float:
    """Safely extracts Best Bid for BUY (Maker) and Best Ask for SELL (Maker)."""
    if not isinstance(quote_dict, dict):
        return 0.0
    
    # Passive order placement:
    # BUY  -> Join Best Bid
    # SELL -> Join Best Ask
    if action == 'buy':
        keys = ['best_bid', 'bid', 'bestBid']
    else:
        keys = ['best_ask', 'ask', 'bestAsk']

    for k in keys:
        if k in quote_dict and quote_dict[k] is not None:
            val = quote_dict[k]
            if isinstance(val, list) and len(val) > 0:
                return float(val[0][0])
            return float(val)

    # Fallback checking nested bids/asks lists if dictionary uses raw depth structure
    if action == 'buy' and 'bids' in quote_dict and quote_dict['bids']:
        return float(quote_dict['bids'][0][0])
    elif action == 'sell' and 'asks' in quote_dict and quote_dict['asks']:
        return float(quote_dict['asks'][0][0])

    return 0.0


def validate_sell_quantity(symbol: str, qty: float):
    df = position_status()
    if df.empty:
        raise click.ClickException("No existing positions found.")
    
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
@click.option('--contract', required=True, help="Contract symbol, e.g., BTC-260828-75000-C")
@click.option('--qty', type=float, required=True, help="Order quantity")
@click.option('--limit', 'order_type', flag_value='limit', default=True, help="Limit order")
@click.option('--market', 'order_type', flag_value='market', help="Market order")
@click.option('--chase', 'order_type', flag_value='chase', help="Chase order mode")
@click.option('--order_price', type=float, default=None, help="Required if --limit is set")
@click.option('--t_bps', type=float, default=10.0, help="Price shift threshold in bps for --chase mode")
@click.option('--execute', is_flag=True, default=False, help="Execute order on exchange")
def order(action, contract, qty, order_type, order_price, t_bps, execute):
    """Place a buy or sell order with passive Chase mode."""
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
        click.secho(f"--> Starting PASSIVE CHASE mode for {action.upper()} (Max shift tolerance: {t_bps} bps)", fg='magenta')
        
        # 1. Anchor Initial Passive Price (BUY -> Best Bid, SELL -> Best Ask)
        init_quote = fetch_bidask(contract)
        base_price = extract_passive_price(init_quote, action)

        if base_price <= 0:
            # Fallback to direct exchange orderbook (BUY -> bids[0], SELL -> asks[0])
            ob = ex.fetch_order_book(contract)
            if action == 'buy':
                base_price = float(ob['bids'][0][0]) if ob.get('bids') else 0.0
            else:
                base_price = float(ob['asks'][0][0]) if ob.get('asks') else 0.0

        if base_price <= 0:
            raise click.ClickException(f"Invalid initial passive price quote: ${base_price:.2f}")

        click.secho(f"Initial Passive Chase Target ({action.upper()}): ${base_price:.2f}", fg='blue')
        
        # 2. Place Initial Order on Inside Market
        current_order = ex.create_order(contract, 'limit', action, qty, base_price)
        order_id = str(current_order['id'])
        current_posted_price = base_price
        click.secho(f"Placed initial passive limit order ID: {order_id} at ${base_price:.2f}", fg='green')

        # 3. Active Passive Chase Loop
        while True:
            time.sleep(1.5)

            # Check order status directly from Binance
            try:
                order_info = ex.eapiPrivateGetOrder({'symbol': contract, 'orderId': order_id})
                order_status = order_info.get('status', '')

                if order_status == 'FILLED':
                    click.secho(f"\n[SUCCESS] Chase order {order_id} FULLY FILLED!", fg='green', bold=True)
                    break
                elif order_status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    click.secho(f"\n[TERMINATED] Order {order_id} ended with status: {order_status}", fg='yellow')
                    break
            except Exception:
                open_orders = ex.eapiPrivateGetOpenOrders({'symbol': contract})
                active_ids = [str(o['orderId']) for o in open_orders] if open_orders else []
                if str(order_id) not in active_ids:
                    click.secho(f"\n[SUCCESS] Order {order_id} no longer active on book.", fg='green', bold=True)
                    break

            # Poll current live passive quote
            quote = fetch_bidask(contract)
            curr_passive = extract_passive_price(quote, action)

            if curr_passive <= 0:
                try:
                    ob = ex.fetch_order_book(contract)
                    curr_passive = float(ob['bids'][0][0]) if action == 'buy' else float(ob['asks'][0][0])
                except Exception:
                    continue

            if curr_passive <= 0:
                continue

            # Determine if market moved in a direction that warrants re-pricing
            # BUY:  Best Bid increased (curr_passive > current_posted_price)
            # SELL: Best Ask decreased (curr_passive < current_posted_price)
            should_reprice = (
                (action == 'buy' and curr_passive > current_posted_price) or
                (action == 'sell' and curr_passive < current_posted_price)
            )

            if should_reprice:
                price_shift_bps = abs(curr_passive - base_price) / base_price * 10000.0

                if price_shift_bps > t_bps:
                    click.secho(
                        f"\n[WARNING] Market shifted {price_shift_bps:.1f} bps (Threshold: {t_bps} bps). "
                        f"Anchor: ${base_price:.2f} -> Current: ${curr_passive:.2f}. Halting chase.",
                        fg='red', bold=True
                    )
                    click.secho(f"Leaving active limit order {order_id} open on orderbook at ${current_posted_price:.2f}.", fg='yellow')
                    break

                click.secho(f"Passive quote improved! Repricing {action.upper()} to ${curr_passive:.2f}...", fg='cyan')
                try:
                    ex.cancel_order(order_id, contract)
                    time.sleep(0.1)
                    new_order = ex.create_order(contract, 'limit', action, qty, curr_passive)
                    order_id = str(new_order['id'])
                    current_posted_price = curr_passive
                    click.secho(f"Replaced order ID: {order_id} at ${curr_passive:.2f}", fg='green')
                except Exception as e:
                    click.secho(f"Failed to cancel/replace order during chase: {e}", fg='red')
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


def get_avg_entry_price(symbol: str) -> float:
    """Fetches user trade history to compute exact fill-weighted entry price."""
    try:
        trades = ex.fetch_my_trades(symbol)
    except Exception:
        try:
            trades = ex.eapiPrivateGetUserTrades({'symbol': symbol})
        except Exception:
            return 0.0

    if not trades:
        return 0.0

    total_qty = 0.0
    total_cost = 0.0

    for t in trades:
        price = float(t.get('price', 0.0))
        qty = float(t.get('amount', t.get('qty', 0.0)))
        side = str(t.get('side', '')).lower()
        if side in ['buy', '']:
            total_qty += qty
            total_cost += price * qty

    return total_cost / total_qty if total_qty > 0 else 0.0


@cli.command()
def status():
    """Show current open orders, positions with mark values, and calculated PnL."""
    click.secho("=" * 35 + " Open Orders " + "=" * 35, fg='blue', bold=True)
    orders_status()

    click.secho("\n" + "=" * 35 + " Positions & PnL " + "=" * 35, fg='blue', bold=True)
    df = position_status()

    if df is None or df.empty:
        click.secho("No open positions found.", fg='yellow')
        return

    df_active = df[df['quantity'].astype(float) != 0].copy()

    if df_active.empty:
        click.secho("No active non-zero positions found.", fg='yellow')
        return

    formatted_rows = []
    total_unrealized_pnl = 0.0
    total_mark_value = 0.0

    for _, row in df_active.iterrows():
        symbol = row['symbol']
        qty = float(row['quantity'])
        mark_val = float(row['markValue'])
        mark_price = mark_val / qty if qty != 0 else 0.0
        
        pos_cost = row.get('positionCost')
        if pos_cost is not None and not np.isnan(float(pos_cost)) and float(pos_cost) > 0:
            entry_cost_total = float(pos_cost)
            entry_price = entry_cost_total / qty
        else:
            entry_price = get_avg_entry_price(symbol)
            entry_cost_total = entry_price * qty

        pnl = mark_val - entry_cost_total if entry_price > 0 else 0.0
        total_unrealized_pnl += pnl
        total_mark_value += mark_val

        pnl_str = f"${pnl:^+10.2f}" if entry_price > 0 else "N/A"

        formatted_rows.append([
            symbol,
            row['side'],
            f"{qty:+.4f}",
            f"${entry_price:,.2f}" if entry_price > 0 else "N/A",
            f"${mark_price:,.2f}",
            f"${mark_val:,.2f}",
            pnl_str
        ])

    headers = ["Symbol", "Side", "Quantity", "Avg Entry Price", "Mark Price", "Mark Value", "Unrealized PnL"]
    click.echo(tabulate(formatted_rows, headers=headers, tablefmt="grid"))

    pnl_color = 'green' if total_unrealized_pnl >= 0 else 'red'
    click.secho(f"\nTotal Position Mark Value    : ${total_mark_value:,.2f}", fg='cyan', bold=True)
    click.secho(f">>> Total Portfolio Unrealized PnL: ${total_unrealized_pnl:+,.2f} <<<", fg=pnl_color, bold=True)

from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_chase_close(contract: str, action: str, qty: float, execute: bool, t_bps: float = 50.0, entry_price: float = 0.0):
    """Internal helper to chase-close an open position via passive quote updates."""
    click.secho(f"[{contract}] [PASSIVE CHASE CLOSE] Closing {qty} via {action.upper()}...", fg='magenta', bold=True)
    
    if not execute:
        click.secho(f"[{contract}] -- Dry run mode: Tracking live orderbook quotes.", fg='yellow')

    # 1. Fetch initial passive price anchor
    init_quote = fetch_bidask(contract)
    base_price = extract_passive_price(init_quote, action)

    if base_price <= 0:
        ensure_markets(ex)
        ob = ex.fetch_order_book(contract)
        if action == 'buy':
            base_price = float(ob['bids'][0][0]) if ob.get('bids') else 0.0
        else:
            base_price = float(ob['asks'][0][0]) if ob.get('asks') else 0.0

    if base_price <= 0:
        click.secho(f"[{contract}] [-] Could not resolve valid live passive quote. Skipping.", fg='red')
        return

    # Calculate estimated trade PnL against entry price
    # Long position (action == 'sell'): PnL = (base_price - entry_price) * qty
    # Short position (action == 'buy'):  PnL = (entry_price - base_price) * qty
    if entry_price > 0:
        pnl_val = (base_price - entry_price) * qty if action == 'sell' else (entry_price - base_price) * qty
        pnl_str = f" | PnL: ${pnl_val:+,.2f}"
    else:
        pnl_str = " | PnL: N/A"

    # 2. Initial order anchor
    current_posted_price = base_price
    if execute:
        ensure_markets(ex)
        current_order = ex.create_order(contract, 'limit', action, qty, base_price)
        order_id = str(current_order['id'])
        click.secho(f"[{contract}] --> [LIVE] Placed initial close order ID: {order_id} at ${base_price:.2f}{pnl_str}", fg='green')
    else:
        order_id = "DRY_RUN_SIM_001"
        click.secho(f"[{contract}] --> [DRY-RUN] Initial target price: ${base_price:.2f}{pnl_str}", fg='cyan')

    # 3. Passive Chase Loop
    while True:
        time.sleep(1.5)

        if execute:
            try:
                order_info = ex.eapiPrivateGetOrder({'symbol': contract, 'orderId': order_id})
                order_status = order_info.get('status', '')

                if order_status == 'FILLED':
                    click.secho(f"[{contract}] [SUCCESS] Position fully closed! Order ID: {order_id}", fg='green', bold=True)
                    break
                elif order_status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    click.secho(f"[{contract}] [TERMINATED] Order ended with status: {order_status}", fg='yellow')
                    break
            except Exception:
                open_orders = ex.eapiPrivateGetOpenOrders({'symbol': contract})
                active_ids = [str(o['orderId']) for o in open_orders] if open_orders else []
                if str(order_id) not in active_ids:
                    click.secho(f"[{contract}] [SUCCESS] Position closed!", fg='green', bold=True)
                    break

        quote = fetch_bidask(contract)
        curr_passive = extract_passive_price(quote, action)

        if curr_passive <= 0:
            try:
                ob = ex.fetch_order_book(contract)
                curr_passive = float(ob['bids'][0][0]) if action == 'buy' else float(ob['asks'][0][0])
            except Exception:
                continue

        if curr_passive <= 0:
            continue

        price_shift_bps = abs(curr_passive - base_price) / base_price * 10000.0
        if price_shift_bps > t_bps:
            click.secho(
                f"[{contract}] [WARNING] Price shifted {price_shift_bps:.1f} bps (Max: {t_bps} bps). Stopping chase.",
                fg='red', bold=True
            )
            break

        should_reprice = (
            (action == 'buy' and curr_passive > current_posted_price) or
            (action == 'sell' and curr_passive < current_posted_price)
        )

        if should_reprice:
            # Update dynamic PnL on target price shift
            if entry_price > 0:
                curr_pnl = (curr_passive - entry_price) * qty if action == 'sell' else (entry_price - curr_passive) * qty
                curr_pnl_str = f" | PnL: ${curr_pnl:+,.2f}"
            else:
                curr_pnl_str = " | PnL: N/A"

            if execute:
                click.secho(f"[{contract}] --> [LIVE] Market quote shifted! Updating to ${curr_passive:.2f}{curr_pnl_str}...", fg='cyan')
                try:
                    ex.cancel_order(order_id, contract)
                    time.sleep(0.1)
                    new_order = ex.create_order(contract, 'limit', action, qty, curr_passive)
                    order_id = str(new_order['id'])
                    current_posted_price = curr_passive
                    click.secho(f"[{contract}] --> [LIVE] Replaced order ID: {order_id} at ${curr_passive:.2f}", fg='green')
                except Exception as e:
                    click.secho(f"[{contract}] [-] Failed to replace close order: {e}", fg='red')
                    break
            else:
                click.secho(
                    f"[{contract}] --> [DRY-RUN] Target price shift! (${current_posted_price:.2f} -> ${curr_passive:.2f}){curr_pnl_str}", 
                    fg='cyan'
                )
                current_posted_price = curr_passive


@cli.command()
@click.option('--contracts', required=True, help="Comma-separated contract symbols, e.g. BTC-260828-76000-C,BTC-260828-76000-P")
@click.option('--t_bps', type=float, default=50.0, help="Max price shift tolerance in bps before stopping chase")
@click.option('--execute', is_flag=True, default=False, help="Execute order on exchange")
def close(contracts, t_bps, execute):
    """Find open positions for specified contracts and execute PARALLEL chase orders to close them."""
    target_contracts = [c.strip().upper() for c in contracts.split(',') if c.strip()]
    if not target_contracts:
        raise click.BadParameter("No valid contracts specified.")

    df = position_status()
    if df is None or df.empty:
        click.secho("No open positions found on exchange.", fg='yellow')
        return

    df_active = df[df['quantity'].astype(float) != 0].copy()
    if df_active.empty:
        click.secho("No active non-zero positions found.", fg='yellow')
        return

    tasks = []
    for contract in target_contracts:
        pos_row = df_active[df_active['symbol'] == contract]
        if pos_row.empty:
            click.secho(f"No active open position found for: {contract}", fg='yellow')
            continue

        row = pos_row.iloc[0]
        qty = abs(float(row['quantity']))
        side = str(row.get('side', '')).upper()
        close_action = 'sell' if side in ['LONG', 'BUY'] or float(row['quantity']) > 0 else 'buy'

        # Resolve entry price for PnL calculation
        pos_cost = row.get('positionCost')
        if pos_cost is not None and not np.isnan(float(pos_cost)) and float(pos_cost) > 0:
            entry_price = float(pos_cost) / qty
        else:
            entry_price = get_avg_entry_price(contract)

        tasks.append((contract, close_action, qty, entry_price))

    if not tasks:
        click.secho("No matching active positions to close.", fg='yellow')
        return

    click.secho(f"\n[PARALLEL EXECUTION] Launching {len(tasks)} parallel close task(s)...", fg='cyan', bold=True)

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(
                execute_chase_close, 
                contract=c, 
                action=a, 
                qty=q, 
                execute=execute, 
                t_bps=t_bps,
                entry_price=ep
            ): c for c, a, q, ep in tasks
        }

        for future in as_completed(futures):
            contract = futures[future]
            try:
                future.result()
            except Exception as exc:
                click.secho(f"[{contract}] Generated an exception during parallel close: {exc}", fg='red', bold=True)

    click.secho("\n[COMPLETED] All parallel close tasks finished.", fg='green', bold=True)

if __name__ == '__main__':
    cli()
