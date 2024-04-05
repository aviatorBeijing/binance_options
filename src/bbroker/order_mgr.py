import click,datetime 
from bbroker.settings import ex
from bbroker.check_status import orders_status,position_status

def mgr(symbol,action,qty,pce, timing='limit'):
    qty = float(qty);pce=float(pce)
    assert action in ['buy','sell'], f"Not supported action: {action}"
    assert timing in ['limit','market'], f"Not supported timing: {timing}"
    if symbol.endswith('-C'):
        #print( ex.market(symbol ))
        #ex.markets[symbol]['precision']['amount'] = 2
        #ex.markets[symbol]['precision']['price'] = 1
        ex.create_order(symbol,timing,action,qty,pce)
    elif symbol.endswith('-P'):
        ex.create_order(symbol,'put',action,qty,pce)

def buy_(symbol,qty,pce):
    mgr(symbol,'buy', qty,pce,timing='limit')

def sell_(symbol,qty,pce):
    print('-- existing positions:')
    df = position_status()
    df = df[df.symbol==symbol] # Binance doesn't allow naked sell for non-marketmaker users.
    if df.empty:
        raise Exception(f"You don't have existing {symbol} for sale.")
    assert df.shape[0] == 1, f"Why more than 1 rows exists for: {symbol}"
    
    assert pce*qty > float(df.iloc[0].positionCost), '\n*** Sell low, take the loss? ***'
    #mgr(symbol,'sell', qty,pce,timing='limit')

@click.command()
@click.option('--action',default="", help="buy or sell")
@click.option('--contract')
@click.option('--price', default=0.0)
@click.option('--qty',default=0.0)
@click.option('--execute', is_flag=True, default=False, help="Send to exchange? O.w., only checking info.")
def main( action,contract, price, qty, execute ):
    action = action.lower()
    contract = contract.upper()

    assert action == 'buy' or action == 'sell', "buy|sell, must be provided."
    assert len(contract) == len('BTC-240329-70000-C'), 'Wrong contract.'
    assert contract.split('-')[0] == 'BTC', 'Only support BTC contracts.'
    assert price>0, 'Price must be >0'
    assert qty>0, 'Quantity (qty) must be >0'

    if execute:
        print('-- [executing] --')
        if action == 'buy':
            buy_( contract, qty, price )
        elif action == 'sell':
            sell_(contract, qty, price)
        
        print('-- checking status --')
        import time
        time.sleep(5)
        orders_status()
    else:
        print('-- [checking] --')
        print(f'-- {action} {qty} {contract} at price ${price}')
        print(f'-- {"cost" if action=="buy" else "gain"}: ${(price * qty):.2f}')
        print('-- use "--execute" to send order.')

if __name__ == '__main__':
    main()

    
