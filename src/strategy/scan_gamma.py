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
        recs = []
        for c in contracts:
            try:
                rec = fetch_bidask(c)
                recs += [rec]
            except AssertionError as e:
                print('*** waiting for data')
                time.sleep(5)
                continue
        df = pd.DataFrame.from_records(recs)
        df = df[['contract','gamma','ask','last_trade','delta','theta','impvol']]
        df.gamma = df.gamma.apply(float)
        df.sort_values('gamma', ascending=False,inplace=True)
        df = df.head(10)
        print( '-- first 10:\n', tabulate(df, headers="keys") )
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