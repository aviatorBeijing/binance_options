import pandas as pd
import click,time,os
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
        fn = os.getenv("USER_HOME","")+'/tmp/binance_greeks.csv'
        df.to_csv(fn,index=0)
        print('--  written  greeks in:', fn)
        n = 50;df = df.head(n)
        print( f'-- first {n}:\n', tabulate(df, headers="keys") )
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
