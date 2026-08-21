import click
import datetime
import time
import numpy as np
from bbroker.settings import ex
from bbroker.check_status import orders_status, position_status

def mgr(symbol, action, qty, pce, timing='limit'):
    qty = float(qty)
    pce = float(pce)
    assert action in ['buy', 'sell'], f"Not supported action: {action}"
    assert timing in ['limit', 'market'], f"Not supported timing: {timing}"
    
    orderinfo = ex.create_order(symbol, timing, action, qty, pce)
    return orderinfo

def validate_buy(symbol, qty, pce):
    from bbroker.check_balances import balances
    df = balances()
    print(df)

def validate_sell(symbol, qty, pce):
    print('-- existing positions:')
    df = position_status()
    if df.empty:
        raise Exception(f"You don't have existing positions for sale.")
        
    df = df[df.symbol == symbol]
    if df.empty:
        raise Exception(f"You don't have existing {symbol} for sale.")
    assert df.shape[0] == 1, f"Why more than 1 rows exists for: {symbol}"

    print('-- existing orders')
    odf = orders_status()
    if not odf.empty:
        odf = odf[(odf.symbol == symbol) & (odf.side == 'SELL')]
        if not odf.empty:
            print(odf)
            existing_position = abs(float(df.iloc[0].quantity))
            existing_sell_qty = np.sum(odf.quantity.astype(float).apply(abs))
            oids = [f"{oid} {symbol}" for oid in odf.orderId.values]
            
            if existing_position < (existing_sell_qty + qty):
                raise Exception(f"""
                ***
                {",".join(oids)}
                existing sell order qty: {existing_sell_qty};
                position for sell {existing_position};
                requesting sell qty {qty} is too much.
                """)

    cost = float(df.iloc[0].positionCost)
    amount = float(df.iloc[0].quantity)
    avg_cost = cost / amount if amount != 0 else 0

    assert pce > avg_cost, '\n*** Sell low, take the loss? ***'
    potential_gain = (pce - avg_cost) * qty 
    rt = (potential_gain / (avg_cost * qty) * 100) if (avg_cost * qty) != 0 else 0
    print(f'-- potential gain (if filled): ${potential_gain:.2f}, {rt:.2f}%')

def buy_(symbol, qty, pce):
    return mgr(symbol, 'buy', qty, pce, timing='limit')

def sell_(symbol, qty, pce):
    validate_sell(symbol, qty, pce)
    return mgr(symbol, 'sell', qty, pce, timing='limit')

def cancel_(symbol, oid):
    df = orders_status()
    if df.empty:
        print(f'*** No existing order found for: {symbol}')
        return 
    df = df[df.symbol == symbol]
    if df.empty:
        print(f'*** No existing order found for: {symbol}')
        return 
    df = df[df.orderId == oid]
    if df.empty:
        print(f'*** No existing order id: {oid}')
        return 
    
    print('-- to be cancelled:\n', df)
    ex.cancel_order(oid, symbol)
    
    print('-- checking order status')
    time.sleep(2)
    orders_status()

@click.command()
@click.option('--action', default="", help="buy or sell")
@click.option('--contract', default="", help="Option contract symbol e.g., BTC-240329-70000-C")
@click.option('--price', default=0.0, type=float)
@click.option('--qty', default=0.0, type=float)
@click.option('--cancel_order_id', default='')
@click.option('--execute', is_flag=True, default=False, help="Send to exchange? O.w., only checking info.")
def main(action, contract, price, qty, cancel_order_id, execute):
    action = action.lower()
    contract = contract.upper()

    if cancel_order_id:
        if not contract:
            raise click.BadParameter("Contract symbol required for cancellation.")
        print('-- [cancelling]')
        cancel_(contract, cancel_order_id)
        print('-- checking status...')
        time.sleep(5)
        orders_status()
        return 

    assert contract, "Contract symbol must be provided."
    assert action in ['buy', 'sell'], "action must be 'buy' or 'sell'."
    assert price > 0, 'Price must be > 0'
    assert qty > 0, 'Quantity (qty) must be > 0'

    if execute:
        print('-- [executing] --')
        if action == 'buy':
            buy_(contract, qty, price)
        elif action == 'sell':
            sell_(contract, qty, price)
        
        print('-- checking status...')
        time.sleep(5)
        orders_status()
    else:
        print('-- [checking] --')
        print(f'-- {action} {qty} {contract} at price ${price}')
        if action == 'sell':
            validate_sell(contract, qty, price)
        print('\n-- use "--execute" to send order.')

if __name__ == '__main__':
    main()
