import os,datetime,json
import pandas as pd
import click,time
from tabulate import tabulate
import numpy  as np
from multiprocessing import Process

from butil.bsql import fetch_bidask 
from butil.butils import ( DATADIR,DEBUG, get_binance_spot,
                get_binance_next_funding_rate,
                get_maturity, get_underlying )
from brisk.bfee import calc_fee
from ws_bcontract import _main as ws_connector
from strategy.delta_gamma import callprice,putprice
from butil.options_calculator import extract_specs, callprice, putprice, invert_callprice, invert_putprice


def _main(contracts:list, reference_spots:list):
    """
    @brief 
    """
    underlying = get_underlying( contracts[0])
    spot_now, _ = get_binance_spot(underlying)
    recs = [] 
    
    for contract in contracts:
        _, T, K, ctype = extract_specs(contract)
        cdata = fetch_bidask(contract.upper())
        ask = float(cdata['ask'])
        sigma = float(cdata['impvol_ask'])

        func_ = None
        if ctype == 'call':
            func_ = callprice
        elif ctype == 'put':
            func_ = putprice
        for S in reference_spots:
            op = func_(S,K,T/365,sigma,0.)
            recs += [ {'contract': contract, 'spot': S, 'option': op, 'option_ask':ask} ]
    df = pd.DataFrame.from_records( recs )
    df.sort_values('spot', ascending=False, inplace=True)
    print( tabulate(df[df.contract.str.contains('-C')], headers="keys"))
    print( tabulate(df[df.contract.str.contains('-P')], headers="keys"))

def _multiprocess_main(contracts:list,projected_spot_prices:list):
    print('-- waiting data...')
    time.sleep(2)
    while True:
        try:
            _main( contracts, projected_spot_prices)
            time.sleep(5)
        except KeyboardInterrupt as e:
            print("-- Exiting --")
            break

@click.command()
@click.option('--contracts', help="contract name")
@click.option('--projected_spot_prices')
def main(contracts,projected_spot_prices):
    projected_spot_prices = list(map(lambda s: float(s), projected_spot_prices.split(',')))
    
    conn = Process( target=ws_connector, args=(f"{contracts}", "ticker",) )
    calc = Process( target=_multiprocess_main, args=(contracts.split(','), projected_spot_prices) )
    conn.start()
    calc.start()
    
    conn.join()
    calc.join()

if __name__ == '__main__':
    main()
