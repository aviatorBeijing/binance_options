import time
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from ws_bcontract import _main as ws_connector

def _multiprocess_main(contracts):
    contracts = contracts.split(',')
    while True:
        for c in contracts:
            rec = fetch_bidask(c)
            print( rec['gamma'], rec['last_trade'], rec['delta'], rec['theta'], rec['impvol'] )
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