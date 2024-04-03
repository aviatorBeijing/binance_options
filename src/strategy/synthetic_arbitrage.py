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
    spot = c[0].split('-')[0] + '/USDT'
    bid,S = binance_spot(spot)

    for c in contracts:
        d = fetch_bidask( c )
        print(d)

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