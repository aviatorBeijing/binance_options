import click,datetime,time
from bbroker.settings import ex
from bbroker.check_status import orders_status,position_status

def mgr(symbol,action,qty,pce, timing='limit'):
    qty = float(qty);pce=float(pce)
    assert action in ['buy','sell'], f"Not supported action: {action}"
    assert timing in ['limit','market'], f"Not supported timing: {timing}"
    #if symbol.endswith('-C'):
        #print( ex.market(symbol ))
        #ex.markets[symbol]['precision']['amount'] = 2
        #ex.markets[symbol]['precision']['price'] = 1
    ex.create_order(symbol,timing,action,qty,pce)


def validate_buy(symbol,qty,pce):
    from bbroker.check_balances import balances
    df = balances()
    print(df)

def validate_sell(symbol,qty,pce):
    print('-- existing positions:')
    df = position_status()
    df = df[df.symbol==symbol] # Binance doesn't allow naked sell for non-marketmaker users.
    if df.empty:
        raise Exception(f"You don't have existing {symbol} for sale.")
    assert df.shape[0] == 1, f"Why more than 1 rows exists for: {symbol}"

    print('-- existing orders')
    odf = orders_status()
    if not odf.empty:
        odf = odf[(odf.symbol==symbol)&(odf.side=='SELL')]
        assert odf.shape[0] == 1, f'Error\n{odf}'
        existing_position = abs( float(df.iloc[0].quantity) )
        existing_sell_qty = abs( float(odf.iloc[0].quantity) )
        if existing_position < (existing_sell_qty + qty):
            raise Exception(f"\n***\n\t{symbol}\n\texisting sell order qty: {existing_sell_qty};\n\tposition for sell {existing_position};\n\trequesting sell qty {qty} is too much.")

    
    # existing position
    cost = float(df.iloc[0].positionCost)
    amount = float(df.iloc[0].quantity)
    avg_cost = cost/amount 

    assert pce > avg_cost, '\n*** Sell low, take the loss? ***'
    potential_gain = (pce-avg_cost) * qty 
    rt = potential_gain/(avg_cost*qty)*100
    print(f'-- potential gain (if filled): ${potential_gain}, {rt:.2f}%')

def buy_(symbol,qty,pce):
    validate_buy(symbol,qty,pce)
    mgr(symbol,'buy', qty,pce,timing='limit')

def sell_(symbol,qty,pce):
    validate_sell(symbol,qty,pce)
    mgr(symbol,'sell', qty,pce,timing='limit')

def cancel_(symbol, oid):
    df = orders_status()
    df = df[df.symbol==symbol]
    if df.empty:
        print(f'*** No existing order found for: {symbol}')
        return 
    df = df[df.orderId == oid]
    if df.empty:
        print(f'*** No existing order id: {oid}')
        return 
    
    print('-- to be cancelled:\n',df)
    ex.cancel_order(oid,symbol)
    
    print('-- checking order status')
    time.sleep(2)
    orders_status()

@click.command()
@click.option('--action',default="", help="buy or sell")
@click.option('--contract')
@click.option('--price', default=0.0)
@click.option('--qty',default=0.0)
@click.option('--cancel_order_id',default='')
@click.option('--execute', is_flag=True, default=False, help="Send to exchange? O.w., only checking info.")
def main( action,contract, price, qty, cancel_order_id, execute ):
    action = action.lower()
    contract = contract.upper()
    assert len(contract) == len('BTC-240329-70000-C'), 'Wrong contract.'
    assert contract.split('-')[0] == 'BTC', 'Only support BTC contracts.'
    
    # Cancel order
    if cancel_order_id:
        print('-- [cancelling]')
        cancel_(contract, cancel_order_id)
        
        print('-- checking status...')
        time.sleep(5)
        orders_status()
        return 

    # Making orders
    assert action == 'buy' or action == 'sell', "buy|sell, must be provided."
    assert price>0, 'Price must be >0'
    assert qty>0, 'Quantity (qty) must be >0'

    if execute:
        print('-- [executing] --')
        if action == 'buy':
            buy_( contract, qty, price )
        elif action == 'sell':
            sell_(contract, qty, price)
        
        print('-- checking status...')
        time.sleep(5)
        orders_status()
    else:
        print('-- [checking] --')
        print(f'-- {action} {qty} {contract} at price ${price}')
        
        if action=='sell':
            validate_sell(contract,qty,price)
        elif action == 'buy':
            validate_buy(contract,qty,price)
        else:
            raise Exception("Error:", action)

        print('\n-- use "--execute" to send order.')

if __name__ == '__main__':
    main()

    
