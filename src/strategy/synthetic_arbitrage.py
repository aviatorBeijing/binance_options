import datetime,os
import pandas as pd
import requests
import click,time
import numpy as np
import pandas as pd
from tabulate import tabulate
from multiprocessing import Process

from butil.butils import binance_spot
from strategy.price_disparity import extract_specs
from sentiments.atms import fetch_oi,fetch_contracts
from ws_bcontract import _main as ws_connector
from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG,
                get_binance_next_funding_rate,
                get_maturity )



def _multiprocess_main(contracts:list):
    time.sleep(5)
    cs = []
    for c in contracts:
        cs += [c[:-1]]
    cs = list(set(cs))
    recs = []
    for c in cs:
        spot_symbol, T,K, ctype = extract_specs( c+'C' )
        _,S = binance_spot(spot_symbol)
        call = fetch_bidask( c+'C' )
        C = call['ask']
        put = fetch_bidask( c+'P' )
        P = put['ask']
        recs += [ {'call': C, 'put': P, 'spot': S, 'strike': K}]
    df = pd.DataFrame.from_records( recs )
    print( tabulate(df,headers='keys'))


@click.command()
@click.option('--contracts')
def main(contracts):
    conn = Process( target=ws_connector, args=(contracts, "ticker",) )
    calc = Process( target=_multiprocess_main, args=(contracts.split(','),) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()