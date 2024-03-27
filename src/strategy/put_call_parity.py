import time
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask
from butil.butils import get_maturity,get_binance_spot
from ws_bcontract import _main as ws_connector

def _multiprocess_main(contracts):
    contracts = contracts.split(',')
    cts = []
    for c in contracts:
        head = c[:-1]
        if head not in cts:
            cts += [ head ]
    spot_symbol = cts[0].split('-')[0] +  '/USDT'

    while True:
        bid,ask = get_binance_spot( spot_symbol )
        S = (bid+ask)*0.5
        recs = []
        for t in cts:
            call = t + 'C'
            put  = t + 'P' 
            T = get_maturity( call )/365
            K = float( t.split('-')[2] )
            rf = 0
            try:
                r = fetch_bidask(call)
                call_ask = C = float(r['ask']);call_bid = float(r['bid'])
                r = fetch_bidask(put)
                put_ask = P = float(r['ask']);put_bid = float(r['bid'])
                PV = K * np.exp(rf*T)
                # C + PV = P + S
                # replicating
                print('-- replicating call:', f'{call_ask:.2f}', f'{(put_ask + S - PV):.2f}' )
                print('-- replicating  put:', f'{put_ask:.2f}', f'{(call_ask + PV -S):.2f}' )   
            except AssertionError as e:
                print(f'*** waiting for data: {call}, {put}, {str(e)}')
                time.sleep(5)
                continue
        time.sleep(5)

@click.command()
@click.option('--contracts')
def main(contracts):
    conn = Process( target=ws_connector, args=(contracts, "ticker",) )
    calc = Process( target=_multiprocess_main, args=(contracts,) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ ==  '__main__':
    main()