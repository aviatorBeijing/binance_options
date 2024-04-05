import click,datetime 
from bbroker.settings import ex
from bbroker.check_status import orders_status

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

def buy_call(symbol,qty,pce):
    mgr(symbol,'buy', qty,pce,timing='limit')

@click.command()
@click.option('--contract')
@click.option('--price', default=0.0)
@click.option('--qty',default=0.0)
@click.option('--execute', is_flag=True, default=False, help="Send to exchange? O.w., only checking info.")
def main( contract, price, qty, execute ):
    assert len(contract) == len('BTC-240329-70000-C'), 'Wrong contract.'
    assert contract.split('-')[0] == 'BTC', 'Only support BTC contracts.'
    assert price>0, 'Price must be >0'
    assert qty>0, 'Quantity (qty) must be >0'

    if execute:
        print('-- [executing] --')
        if contract[-1] == 'C':
            buy_call( contract, qty, price )
        
        import time
        time.sleep(5)
        orders_status()
    else:
        print('-- [checking] --')
        print(f'-- cost: ${(price * qty):.2f}')

if __name__ == '__main__':
    main()

    
