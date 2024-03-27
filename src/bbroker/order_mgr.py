from optparse import check_builtin
from bbroker.settings import ex

def mgr(symbol,action,qty,pce, timing='limit'):
    qty = float(qty);pce=float(pce)
    assert action in ['buy','sell'], f"Not supported action: {action}"
    assert timing in ['limit','market'], f"Not supported timing: {timing}"
    if symbol.endswith('-C'):
        ex.create_order(symbol,timing,action,qty,pce)
    elif symbol.endswith('-P'):
        ex.create_order(symbol,'put',action,qty,pce)

def buy_call(symbol,qty,pce):
    mgr(symbol,'buy', qty,pce,timing='limit')

# Test  
if __name__ == '__main__':
    from bbroker.check_status import orders_status

    symbol = 'BTC-240329-24000-C'
    buy_call(symbol, 0.01, 10.)

    import time
    while True:
        orders_status()
        time.sleep(5)