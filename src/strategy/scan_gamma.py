import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import ccxt
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from ws_bcontract import _main as ws_connector

def _multiprocess_main(contracts):
    for c in contracts.split(','):
        rec = fetch_bidask(c)
        print( rec )

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