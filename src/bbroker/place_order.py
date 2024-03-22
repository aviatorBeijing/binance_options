import datetime,os,click,time
import pandas as pd 
import numpy  as np
import functools
from multiprocessing import Process 
import ccxt 
from ws_bcontract import _main as ws_connector, sync_fetch_ticker

apikey = os.getenv('BINANCE_SUB01_APIKEY', None)
secret = os.getenv('BINANCE_SUB01_SECRET', None)
ex = ccxt.binance({
    'apiKey': apikey,
    'secret': secret,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'option',
    }
})
import pprint;pprint.pprint(ex.fetch_balance())
"""
'defaultType': 'spot',  # 'spot', 'future', 'margin', 'delivery', 'option'
"""

LHISTORY=30 # past 30 seconds
history = [] # bid history

class BOrder:
    def __init__(self, contract,action,qty,pce) -> None:
        self.contract = contract
        self.action = action 
        self.qty =  qty 
        self.pce = pce 

        assert any([action.lower() == e for e in [
            'sit',
            'sell-limit','buy-limit',
            'sell-mkt','buy-mkt']]), f"unsuppported: {action}"
        assert qty>0, f'must be positive, but found {qty}'
        assert pce>0, f'must be positive, but found {pce}'
    def __repr__(self) -> str:
        return f"{self.action} {self.qty} {self.contract} at price {self.pce}"
    def __str__(self) -> str:
        return f"[{self.action} {self.qty} {self.contract} at price {self.pce}]"

def hadd( new_data:tuple):
    global history
    history += [new_data]
    tnow = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    tnow = int(tnow.timestamp())
    history = filter(lambda e: (tnow-e[0])<LHISTORY, history)
    history = list(history)

    bp = list(map(lambda e:e[1], history))
    ap = list(map(lambda e:e[2], history))
    print(f'  -- bid: {min(bp)} ~ {max(bp)}, {new_data[1]} \t ask: {min(ap)} ~ {max(ap)}, {new_data[2]}')

def on_new_market_price( md, border=None ):
    ts = md['ts_beijing']
    bid,ask =  md['bid'], md['ask']
    bv,av = md['bidv'], md['askv']

    iv_bid, iv_ask = md['impvol_bid'],md['impvol_ask']
    last = md['last_trade']
    #print(  bid,ask,'\t', iv_bid,iv_ask, '\t', bv,av,'\t',  datetime.datetime.fromtimestamp(int(ts))  )

    hadd( (int(ts), float(bid), float(ask), float(bv), float(av), float(iv_bid), float(iv_ask) ) )

def _main(contract, border):
    print(f'-- will show price range in {LHISTORY} secs')
    try:
        while True:
            try:
                sync_fetch_ticker(contract, functools.partial( on_new_market_price, border=border) )
            except AssertionError as ae:
                print('*** data outdated, wait.', str(ae))
            time.sleep(1)
    except KeyboardInterrupt as ke:
        print('-- shutting down')
        time.sleep(2)

@click.command()
@click.option('--contract')
@click.option('--action', default='SIT', help="SIT | BUY | SELL")
@click.option('--qty', default=0.)
@click.option('--pce', default=0.)
def main(contract,action,qty,pce):
    bo = BOrder(contract,action,qty,pce)
    conn = Process( target=ws_connector, args=(f"{contract}", "ticker",) )
    calc = Process( target=_main, args=(contract,bo,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join() 

if __name__ == '__main__':
    main()